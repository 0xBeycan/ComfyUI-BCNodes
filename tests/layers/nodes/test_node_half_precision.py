"""The image and mask nodes on a half-precision clip (Load Video's float16 default), through their node
methods: each takes an 8-bit clip as float32 (k / 255, what a float32 load holds) and as its float16
copy, and the float16 run must

  - give its outputs in float16 (bfloat16 stays bfloat16 too: Repeat Mask Batch, Fill Holes);
  - give, for 8-bit-level or 0/1 outputs, exactly the float32 run's values once requantized
    (libs/image.requantized): Fill Holes, Grow, Blockify, Repeat, Image Scale By Aspect Ratio, Draw
    Mask On Image with an opaque colour on a 0/1 mask, BiRefNet's colour channels;
  - give, for continuous outputs, the float32 run's values within float16's own rounding (2^-11 of
    the value, a twentieth of an 8-bit level at most): Draw Mask On Image at half alpha, BiRefNet's
    matte, its composite, Depth Anything, PostFx, Skin Texture.
Every level passes through Grow and Image Scale By Aspect Ratio unchanged, float32 and float16 alike:
float16(1 / 255) read as `255 * x.float()` without the requantize truncates to 0.
"""

import pytest
import torch

HALF_STEP = 2.0 ** -11  # float16's spacing just below 1.0: a rounding moves a value in 0..1 by at most half of it


def levels(*shape, seed=0):
    """An 8-bit clip as float32 k / 255."""
    return torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed)).float() / 255


@pytest.fixture
def requantized(bcnodes):
    return bcnodes["libs.image"].requantized


def same_levels(half, full, requantized):
    return half.dtype == torch.float16 and torch.equal(requantized(half), full)


def close(half, full):
    return half.dtype == torch.float16 and half.shape == full.shape and float((half.float() - full).abs().max()) <= HALF_STEP


def binary_mask(b, h, w, seed=1):
    m = torch.zeros(b, h, w)
    m[:, 4:h - 4, 3:w - 3] = 1.0
    m[:, h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 0.0  # a hole
    return m


def test_mask_nodes_keep_half_and_its_levels(bcnodes, requantized):
    mask = bcnodes["mask"]
    soft = levels(3, 24, 20, seed=2)
    hard = binary_mask(3, 24, 20)
    for m in (soft, hard):
        assert same_levels(mask.MaskFillHoles().fill_region(m.half())[0], mask.MaskFillHoles().fill_region(m)[0], requantized)
        for invert in (False, True):
            for grow, blur in ((2, 3), (-1, 0), (0, 2)):
                args = (invert, grow, blur)
                assert same_levels(mask.MaskGrow().mask_grow(*args, m.half())[0], mask.MaskGrow().mask_grow(*args, m)[0], requantized)
        assert same_levels(mask.BlockifyMask().process(m.half(), 8)[0], mask.BlockifyMask().process(m, 8)[0], requantized)
        assert same_levels(mask.RepeatMaskBatch().repeat(m.half(), 3)[0], mask.RepeatMaskBatch().repeat(m, 3)[0], requantized)
    for node, call in ((mask.MaskFillHoles(), lambda n, x: n.fill_region(x)), (mask.RepeatMaskBatch(), lambda n, x: n.repeat(x, 2))):
        assert call(node, hard.bfloat16())[0].dtype == torch.bfloat16


def test_every_level_passes_grow_and_scale_unchanged(bcnodes, requantized):
    every = (torch.arange(256, dtype=torch.float32) / 255).view(1, 16, 16)
    for clip in (every, every.half()):
        out = bcnodes["mask"].MaskGrow().mask_grow(False, 0, 0, clip)[0]
        assert torch.equal(requantized(out), every)
        rgb = every[..., None].expand(1, 16, 16, 3).contiguous().to(clip.dtype)
        image, m = bcnodes["image_scale"].ImageScaleByAspectRatio().scale(
            "original", 1, 1, "fill", "nearest", "None", "None", 1024, "#000000", image=rgb, mask=clip)[:2]
        assert image.dtype == m.dtype == clip.dtype
        assert torch.equal(requantized(image), every[..., None].expand(1, 16, 16, 3)) and torch.equal(requantized(m), every)


def test_draw_mask_on_image(bcnodes, requantized):
    draw = bcnodes["mask"].DrawMaskOnImage()
    image, hard = levels(2, 16, 12, 3, seed=3), binary_mask(2, 16, 12)
    assert same_levels(draw.apply(image.half(), hard.half(), "0, 0, 0")[0], draw.apply(image, hard, "0, 0, 0")[0], requantized)
    soft = levels(2, 8, 6, seed=4)  # scaled to the image
    for img, m in ((image.half(), soft.half()), (image.half(), soft)):
        assert close(draw.apply(img, m, "255, 128, 0, 128")[0], draw.apply(image, soft, "255, 128, 0, 128")[0])
    rgba = levels(2, 16, 12, 4, seed=5)
    assert close(draw.apply(rgba.half(), soft.half(), "#ff000080")[0], draw.apply(rgba, soft, "#ff000080")[0])
    assert draw.apply(image, soft.half(), "0, 0, 0")[0].dtype == torch.float32  # the output follows the image


def test_image_scale_by_aspect_ratio(bcnodes, requantized):
    node = bcnodes["image_scale"].ImageScaleByAspectRatio()
    image, m = levels(2, 30, 40, 3, seed=6), levels(2, 30, 40, seed=7)
    for fit in ("letterbox", "crop", "fill"):
        args = ("16:9", 1, 1, fit, "lanczos", "16", "longest", 48, "#204060")
        half = node.scale(*args, image=image.half(), mask=m.half())
        full = node.scale(*args, image=image, mask=m)
        assert same_levels(half[0], full[0], requantized) and same_levels(half[1], full[1], requantized)
        assert half[2:] == full[2:]
    zeros = node.scale("1:1", 1, 1, "crop", "bicubic", "8", "longest", 32, "#000000", image=image.half())[1]
    assert zeros.dtype == torch.float16 and zeros.shape == (2, 32, 32) and not zeros.any()


class _Net:
    """BiRefNet stand-in: (1, 3, res, res) -> (1, 1, res, res) logits."""

    def __call__(self, x):
        return x.mean(1, keepdim=True) * 4.0 - 2.0


def test_birefnet(bcnodes, monkeypatch, requantized):
    monkeypatch.setattr(bcnodes["models.birefnet.inference"], "load", lambda model: (_Net(), torch.device("cpu"), torch.float32))
    node = bcnodes["birefnet"].BiRefNetRemoveBackground()
    image = levels(2, 24, 20, 3, seed=8)
    for widgets in (dict(), dict(sensitivity=0.8, mask_blur=2, mask_offset=1, refine_foreground=True),
                    dict(background="Color", background_color="#33669980", invert_output=True)):
        half = node.remove_background(image.half(), "BiRefNet-general", **widgets)
        full = node.remove_background(image, "BiRefNet-general", **widgets)
        for h_out, f_out in zip(half, full):
            assert close(h_out, f_out)
        if not widgets:  # the colour channels are the frames' own levels
            assert torch.equal(requantized(half[0][..., :3]), full[0][..., :3])


def test_depth_anything(bcnodes, monkeypatch):
    class Gradient(torch.nn.Module):
        def forward(self, x):  # inverse depth from the input itself, so a wrong read of the frame shows
            return x.mean(1) * torch.linspace(1.0, 2.0, x.shape[-1])

    monkeypatch.setattr(bcnodes["models.depth_anything_v2.inference"], "load", lambda: (Gradient(), torch.device("cpu")))
    node = bcnodes["depth_anything"].DepthAnythingV2()
    image = levels(2, 28, 42, 3, seed=9)
    assert close(node.estimate_depth(image.half(), 14)[0], node.estimate_depth(image, 14)[0])


def test_postfx_apply_and_signature_sheet(bcnodes):
    pytest.importorskip("postfx")
    node = bcnodes["postfx"]
    image, m = levels(2, 24, 32, 3, seed=10), levels(2, 24, 32, seed=11)
    args = ("signature/01_portra_400", "neutral", 1.0, 7, "increment")
    for mask in (None, m):
        half = node.PostFxApply().apply(image.half(), *args, mask=None if mask is None else mask.half())[0]
        assert close(half, node.PostFxApply().apply(image, *args, mask=mask)[0])
    sheet = node.PostFxSignatureSheet().build(image.half(), "signature", "neutral", 1.0, 5)[0]
    assert close(sheet, node.PostFxSignatureSheet().build(image, "signature", "neutral", 1.0, 5)[0])


def test_skin_texture_with_a_mask(bcnodes):
    node = bcnodes["skin_texture"].SkinTexture()
    image, m = levels(2, 32, 24, 3, seed=12), binary_mask(2, 32, 24)
    widgets = dict(sam3_model="unused", texture=0.7, detail=0.6, pore_scale=1.0, feather=2, seed=5, threshold=0.5)
    half = node.run(image.half(), mask=m.half(), exclude_mask=levels(1, 16, 12, seed=13).half(), **widgets)
    full = node.run(image, mask=m, exclude_mask=levels(1, 16, 12, seed=13), **widgets)
    assert close(half[0], full[0]) and close(half[1], full[1])

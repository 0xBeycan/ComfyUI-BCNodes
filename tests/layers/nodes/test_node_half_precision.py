"""The image and mask nodes on half-precision input, through their node methods. A half input is read
with a plain .float() (the exact value of each half number), worked on in float32 and stored in the
input's dtype, so against the float32 run of the same values (`.float()` of the input) the half run

  - gives its outputs in float16 (bfloat16 stays bfloat16: Repeat Mask Batch, Fill Holes);
  - is the float32 run rounded to float16, bit for bit, wherever no 8-bit step runs: Draw Mask On Image,
    Repeat, Blockify, BiRefNet, Depth Anything, PostFx, Skin Texture, Image Resize (but its lanczos);
  - where an 8-bit step runs (Fill Holes, MaskGrow, Image Scale By Aspect Ratio, lanczos), gives an 8-bit level
    k / 255 back as k (libs/image.tensor_to_u8's half margin), so an 8-bit clip given as float16 gives
    exactly what its float32 load (k / 255) gives; float16(1 / 255) read as `255 * x` without the margin
    truncates to 0.
"""

import pytest
import torch

def levels(*shape, seed=0):
    """An 8-bit clip as float32 k / 255."""
    return torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed)).float() / 255


def continuous(*shape, seed=0):
    """Values that are not 8-bit levels, as float16."""
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed)).half()


def on_levels(half, full):
    """A half 8-bit-level output equals the float32 one level for level."""
    return half.dtype == torch.float16 and torch.equal(half.float().mul(255).round().div(255), full)


def close(half, full):
    """A half output is the float32 one rounded to float16."""
    return half.dtype == torch.float16 and torch.equal(half, full.to(torch.float16))


def binary_mask(b, h, w):
    m = torch.zeros(b, h, w)
    m[:, 4:h - 4, 3:w - 3] = 1.0
    m[:, h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 0.0  # a hole
    return m


def test_mask_nodes_on_an_8_bit_clip(bcnodes):
    mask = bcnodes["mask"]
    for m in (levels(3, 24, 20, seed=2), binary_mask(3, 24, 20)):
        assert on_levels(mask.MaskFillHoles().fill_region(m.half())[0], mask.MaskFillHoles().fill_region(m)[0])
        for args in ((False, 2, 3), (True, -1, 0), (True, 0, 2)):
            assert on_levels(mask.MaskGrow().mask_grow(*args, m.half())[0], mask.MaskGrow().mask_grow(*args, m)[0])
    for node, call in ((mask.MaskFillHoles(), lambda n, x: n.fill_region(x)), (mask.RepeatMaskBatch(), lambda n, x: n.repeat(x, 2))):
        assert call(node, binary_mask(2, 8, 8).bfloat16())[0].dtype == torch.bfloat16


def test_mask_nodes_without_an_8_bit_step_are_the_float32_run(bcnodes):
    mask = bcnodes["mask"]
    m = continuous(3, 24, 20, seed=3)
    assert close(mask.BlockifyMask().process(m, 8)[0], mask.BlockifyMask().process(m.float(), 8)[0])
    assert close(mask.RepeatMaskBatch().repeat(m, 3)[0], mask.RepeatMaskBatch().repeat(m.float(), 3)[0])


def test_every_level_passes_grow_and_scale_unchanged(bcnodes):
    every = (torch.arange(256, dtype=torch.float32) / 255).view(1, 16, 16)
    plain = (255 * every.half().float()).to(torch.uint8)  # the cast without the margin loses levels
    assert int((plain.float() / 255 != every).sum()) > 0
    for clip in (every, every.half(), every.bfloat16()):
        out = bcnodes["mask"].MaskGrow().mask_grow(False, 0, 0, clip)[0]
        assert out.dtype == (clip.dtype if clip.dtype != torch.float32 else torch.float32)
        assert torch.equal(out.float().mul(255).round().div(255), every)
        rgb = every[..., None].expand(1, 16, 16, 3).contiguous().to(clip.dtype)
        image, m = bcnodes["image_scale"].ImageScaleByAspectRatio().scale(
            "original", 1, 1, "fill", "nearest", "None", "None", 1024, "#000000", image=rgb, mask=clip)[:2]
        assert image.dtype == m.dtype == clip.dtype
        assert torch.equal(image.float().mul(255).round().div(255), every[..., None].expand(1, 16, 16, 3))
        assert torch.equal(m.float().mul(255).round().div(255), every)


def test_draw_mask_on_image(bcnodes):
    draw = bcnodes["mask"].DrawMaskOnImage()
    image, soft = continuous(2, 16, 12, 3, seed=4), continuous(2, 8, 6, seed=5)  # the mask scaled to the image
    for img, m in ((image, soft), (image, soft.float())):
        assert close(draw.apply(img, m, "255, 128, 0, 128")[0], draw.apply(image.float(), soft.float(), "255, 128, 0, 128")[0])
    rgba = continuous(2, 16, 12, 4, seed=6)
    assert close(draw.apply(rgba, soft, "#ff000080")[0], draw.apply(rgba.float(), soft.float(), "#ff000080")[0])
    assert draw.apply(image.float(), soft, "0, 0, 0")[0].dtype == torch.float32  # the output follows the image


def test_image_scale_by_aspect_ratio(bcnodes):
    node = bcnodes["image_scale"].ImageScaleByAspectRatio()
    image, m = levels(2, 30, 40, 3, seed=7), levels(2, 30, 40, seed=8)
    for fit in ("letterbox", "crop", "fill"):
        args = ("16:9", 1, 1, fit, "lanczos", "16", "longest", 48, "#204060")
        half, full = node.scale(*args, image=image.half(), mask=m.half()), node.scale(*args, image=image, mask=m)
        assert on_levels(half[0], full[0]) and on_levels(half[1], full[1]) and half[2:] == full[2:]
    zeros = node.scale("1:1", 1, 1, "crop", "bicubic", "8", "longest", 32, "#000000", image=image.half())[1]
    assert zeros.dtype == torch.float16 and zeros.shape == (2, 32, 32) and not zeros.any()


def test_image_resize(bcnodes):
    node = bcnodes["image_scale"].ImageResize()
    image, m = continuous(2, 30, 40, 3, seed=15), continuous(2, 30, 40, seed=16)
    for method, keep in (("bilinear", "pad"), ("bicubic", "crop"), ("area", "pillarbox_blur"), ("nearest-exact", "pad_edge")):
        half = node.resize(image, 64, 48, method, keep, "10, 20, 30", "center", 0, mask=m)
        full = node.resize(image.float(), 64, 48, method, keep, "10, 20, 30", "center", 0, mask=m.float())
        assert close(half[0], full[0]) and half[1:3] == full[1:3] and close(half[3], full[3]), (method, keep)
    lv = levels(2, 30, 40, 3, seed=17)  # lanczos runs on 8 bits: an 8-bit clip as float16 gives its float32 run
    assert on_levels(node.resize(lv.half(), 64, 48, "lanczos", "resize", "0, 0, 0", "center", 0)[0],
                     node.resize(lv, 64, 48, "lanczos", "resize", "0, 0, 0", "center", 0)[0])


class _Net:
    """BiRefNet stand-in: (1, 3, res, res) -> (1, 1, res, res) logits."""

    def __call__(self, x):
        return x.mean(1, keepdim=True) * 4.0 - 2.0


def test_birefnet(bcnodes, monkeypatch):
    monkeypatch.setattr(bcnodes["models.birefnet.inference"], "load", lambda model: (_Net(), torch.device("cpu"), torch.float32))
    node = bcnodes["birefnet"].BiRefNetRemoveBackground()
    image = continuous(2, 24, 20, 3, seed=9)
    for widgets in (dict(), dict(sensitivity=0.8, mask_blur=2, mask_offset=1, refine_foreground=True),
                    dict(background="Color", background_color="#33669980", invert_output=True)):
        half = node.remove_background(image, "BiRefNet-general", **widgets)
        full = node.remove_background(image.float(), "BiRefNet-general", **widgets)
        assert all(close(h_out, f_out) for h_out, f_out in zip(half, full))


def test_depth_anything(bcnodes, monkeypatch):
    class Gradient(torch.nn.Module):
        def forward(self, x):  # inverse depth from the input itself, so a wrong read of the frame shows
            return x.mean(1) * torch.linspace(1.0, 2.0, x.shape[-1])

    monkeypatch.setattr(bcnodes["models.depth_anything_v2.inference"], "load", lambda: (Gradient(), torch.device("cpu")))
    node = bcnodes["depth_anything"].DepthAnythingV2()
    image = continuous(2, 28, 42, 3, seed=10)
    assert close(node.estimate_depth(image, 14)[0], node.estimate_depth(image.float(), 14)[0])


def test_postfx_apply_and_signature_sheet(bcnodes):
    pytest.importorskip("postfx")
    node = bcnodes["postfx"]
    image, m = continuous(2, 24, 32, 3, seed=11), continuous(2, 24, 32, seed=12)
    args = ("signature/01_portra_400", "neutral", 1.0, 7, "increment")
    for mask in (None, m):
        half = node.PostFxApply().apply(image, *args, mask=mask)[0]
        assert close(half, node.PostFxApply().apply(image.float(), *args, mask=None if mask is None else mask.float())[0])
    sheet = node.PostFxSignatureSheet().build(image, "signature", "neutral", 1.0, 5)[0]
    assert close(sheet, node.PostFxSignatureSheet().build(image.float(), "signature", "neutral", 1.0, 5)[0])


def test_skin_texture_with_a_mask(bcnodes):
    node = bcnodes["skin_texture"].SkinTexture()
    image, m, ex = continuous(2, 32, 24, 3, seed=13), binary_mask(2, 32, 24).half(), continuous(1, 16, 12, seed=14)
    widgets = dict(sam3_model="unused", texture=0.7, detail=0.6, pore_scale=1.0, feather=2, seed=5, threshold=0.5)
    half = node.run(image, mask=m, exclude_mask=ex, **widgets)
    full = node.run(image.float(), mask=m.float(), exclude_mask=ex.float(), **widgets)
    assert close(half[0], full[0]) and close(half[1], full[1])

"""The image and mask nodes on float16 input, through their node methods. A half frame is read through
libs/image.float_frame (float16 8-bit data as its levels k / 255, anything else as its exact values),
worked on in float32 and stored in the input's dtype. So each node's float16 run gives, bit for bit, its
float32 run rounded to float16, for both kinds of float16 input:

  - an 8-bit clip given as float16 (Load Video's default) against the float32 run of the clip (k / 255);
  - continuous float16 values (SeedVR2 PostProcess's) against the float32 run of the same values.

Repeat Mask Batch and Fill Holes keep bfloat16 too. Every level passes Grow and Image Scale By Aspect Ratio
unchanged: float16(1 / 255) read as `255 * x` without float_frame truncates to 0.
"""

import pytest
import torch


def levels(*shape, seed=0):
    """An 8-bit clip as float32 k / 255."""
    return torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed)).float() / 255


def continuous(*shape, seed=0):
    """Values that are not 8-bit levels, as float16."""
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed)).half()


def kinds(*shapes, seed=0):
    """[(float16 inputs, float32 inputs)]: an 8-bit clip and its float16 copy; continuous float16 values and
    the same values as float32."""
    lv = [levels(*s, seed=seed + j) for j, s in enumerate(shapes)]
    ct = [continuous(*s, seed=seed + 10 + j) for j, s in enumerate(shapes)]
    return [([x.half() for x in lv], lv), (ct, [x.float() for x in ct])]


def same(half, full):
    """A float16 output is the float32 one rounded to float16."""
    return half.dtype == torch.float16 and torch.equal(half, full.to(torch.float16))


def outputs_same(half, full):
    return all(same(h, f) for h, f in zip(half, full) if isinstance(h, torch.Tensor)) and \
        [h for h in half if not isinstance(h, torch.Tensor)] == [f for f in full if not isinstance(f, torch.Tensor)]


def binary_mask(b, h, w):
    m = torch.zeros(b, h, w)
    m[:, 4:h - 4, 3:w - 3] = 1.0
    m[:, h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2] = 0.0  # a hole
    return m


def test_mask_nodes(bcnodes):
    mask = bcnodes["mask"]
    for (m16,), (m32,) in kinds((3, 24, 20), seed=1) + [([binary_mask(3, 24, 20).half()], [binary_mask(3, 24, 20)])]:
        assert same(mask.MaskFillHoles().fill_region(m16)[0], mask.MaskFillHoles().fill_region(m32)[0])
        for args in ((False, 2, 3), (True, -1, 0), (True, 0, 2)):
            assert same(mask.MaskGrow().mask_grow(*args, m16)[0], mask.MaskGrow().mask_grow(*args, m32)[0]), args
        assert same(mask.BlockifyMask().process(m16, 8)[0], mask.BlockifyMask().process(m32, 8)[0])
        assert same(mask.RepeatMaskBatch().repeat(m16, 3)[0], mask.RepeatMaskBatch().repeat(m32, 3)[0])
    for node, call in ((mask.MaskFillHoles(), lambda n, x: n.fill_region(x)), (mask.RepeatMaskBatch(), lambda n, x: n.repeat(x, 2))):
        assert call(node, binary_mask(2, 8, 8).bfloat16())[0].dtype == torch.bfloat16


def test_every_level_passes_grow_and_scale_unchanged(bcnodes):
    every = (torch.arange(256, dtype=torch.float32) / 255).view(1, 16, 16)
    plain = (255 * every.half().float()).to(torch.uint8)  # the cast without float_frame loses levels
    assert int((plain.float() / 255 != every).sum()) > 0
    for clip in (every, every.half(), every.bfloat16()):
        out = bcnodes["mask"].MaskGrow().mask_grow(False, 0, 0, clip)[0]
        assert torch.equal(out.float().mul(255).round().div(255), every)
        rgb = every[..., None].expand(1, 16, 16, 3).contiguous().to(clip.dtype)
        image, m = bcnodes["image_scale"].ImageScaleByAspectRatio().scale(
            "original", 1, 1, "fill", "nearest", "None", "None", 1024, "#000000", image=rgb, mask=clip)[:2]
        assert image.dtype == m.dtype == clip.dtype
        assert torch.equal(image.float().mul(255).round().div(255), every[..., None].expand(1, 16, 16, 3))
        assert torch.equal(m.float().mul(255).round().div(255), every)


def test_draw_mask_on_image(bcnodes):
    draw = bcnodes["mask"].DrawMaskOnImage()
    for (img, rgba, m), (img32, rgba32, m32) in kinds((2, 16, 12, 3), (2, 16, 12, 4), (2, 8, 6), seed=2):
        assert same(draw.apply(img, m, "255, 128, 0, 128")[0], draw.apply(img32, m32, "255, 128, 0, 128")[0])
        assert same(draw.apply(img, m32, "0, 0, 0")[0], draw.apply(img32, m32, "0, 0, 0")[0])
        assert same(draw.apply(rgba, m, "#ff000080")[0], draw.apply(rgba32, m32, "#ff000080")[0])
        assert draw.apply(img32, m, "0, 0, 0")[0].dtype == torch.float32  # the output follows the image


def test_image_scale_by_aspect_ratio(bcnodes):
    node = bcnodes["image_scale"].ImageScaleByAspectRatio()
    for (img, m), (img32, m32) in kinds((2, 30, 40, 3), (2, 30, 40), seed=3):
        for fit in ("letterbox", "crop", "fill"):
            args = ("16:9", 1, 1, fit, "lanczos", "16", "longest", 48, "#204060")
            assert outputs_same(node.scale(*args, image=img, mask=m), node.scale(*args, image=img32, mask=m32))
        zeros = node.scale("1:1", 1, 1, "crop", "bicubic", "8", "longest", 32, "#000000", image=img)[1]
        assert zeros.dtype == torch.float16 and zeros.shape == (2, 32, 32) and not zeros.any()


def test_image_resize(bcnodes):
    node = bcnodes["image_scale"].ImageResize()
    for (img, m), (img32, m32) in kinds((2, 30, 40, 3), (2, 30, 40), seed=4):
        for method, keep in (("bilinear", "pad"), ("bicubic", "crop"), ("area", "pillarbox_blur"), ("nearest-exact", "pad_edge"),
                             ("lanczos", "resize")):
            args = (64, 48, method, keep, "10, 20, 30", "center", 0)
            assert outputs_same(node.resize(img, *args, mask=m), node.resize(img32, *args, mask=m32)), (method, keep)


class _Net:
    """BiRefNet stand-in: (1, 3, res, res) -> (1, 1, res, res) logits."""

    def __call__(self, x):
        return x.mean(1, keepdim=True) * 4.0 - 2.0


def test_birefnet(bcnodes, monkeypatch):
    monkeypatch.setattr(bcnodes["models.birefnet.inference"], "load", lambda model: (_Net(), torch.device("cpu"), torch.float32))
    node = bcnodes["birefnet"].BiRefNetRemoveBackground()
    for (img,), (img32,) in kinds((2, 24, 20, 3), seed=5):
        for widgets in (dict(), dict(sensitivity=0.8, mask_blur=2, mask_offset=1, refine_foreground=True),
                        dict(background="Color", background_color="#33669980", invert_output=True, refine_foreground=True)):
            assert outputs_same(node.remove_background(img, "BiRefNet-general", **widgets),
                                node.remove_background(img32, "BiRefNet-general", **widgets)), widgets


def test_depth_anything(bcnodes, monkeypatch):
    class Gradient(torch.nn.Module):
        def forward(self, x):  # inverse depth from the input itself, so a wrong read of the frame shows
            return x.mean(1) * torch.linspace(1.0, 2.0, x.shape[-1])

    monkeypatch.setattr(bcnodes["models.depth_anything_v2.inference"], "load", lambda: (Gradient(), torch.device("cpu")))
    node = bcnodes["depth_anything"].DepthAnythingV2()
    for (img,), (img32,) in kinds((2, 28, 42, 3), seed=6):
        assert same(node.estimate_depth(img, 14)[0], node.estimate_depth(img32, 14)[0])


def test_postfx_apply_and_signature_sheet(bcnodes):
    pytest.importorskip("postfx")
    node = bcnodes["postfx"]
    args = ("signature/01_portra_400", "neutral", 1.0, 7, "increment")
    for (img, m), (img32, m32) in kinds((2, 24, 32, 3), (2, 24, 32), seed=7):
        assert same(node.PostFxApply().apply(img, *args)[0], node.PostFxApply().apply(img32, *args)[0])
        assert same(node.PostFxApply().apply(img, *args, mask=m)[0], node.PostFxApply().apply(img32, *args, mask=m32)[0])
        assert same(node.PostFxSignatureSheet().build(img, "signature", "neutral", 1.0, 5)[0],
                    node.PostFxSignatureSheet().build(img32, "signature", "neutral", 1.0, 5)[0])


def test_skin_texture_with_a_mask(bcnodes):
    node = bcnodes["skin_texture"].SkinTexture()
    widgets = dict(sam3_model="unused", texture=0.7, detail=0.6, pore_scale=1.0, feather=2, seed=5, threshold=0.5)
    for (img, ex), (img32, ex32) in kinds((2, 32, 24, 3), (1, 16, 12), seed=8):
        m = binary_mask(2, 32, 24)
        assert outputs_same(node.run(img, mask=m.half(), exclude_mask=ex, **widgets),
                            node.run(img32, mask=m, exclude_mask=ex32, **widgets))

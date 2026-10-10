"""models/qwen_lm/images.py: the image batch at the official Qwen processor's size and resampling.

  - smart_resize: the build spec's sizes, transformers' own smart_resize (patch 16 x merge 2 = factor 32,
    min_pixels 65536, max_pixels 16777216) on a grid of sizes, the aspect-ratio limit (more than 200
    refused, as the official processor refuses it) and an image without pixels refused, saying what to do;
  - prepare_images: float32 (N, h, w, 3) out; None for None or an empty batch; RGBA's alpha dropped;
    values clamped to [0, 1]; a frame at its own size kept as it is; on 8-bit data within about one
    level of PIL's bicubic resize (the official 8-bit resize: width pass, then height pass, each clamped)
    where one 2-D pass clamped at the end is far off; a half-precision batch read as CLAUDE.md's rule
    says (float16 8-bit data as its float32 source, other half frames as their exact values).
"""

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def images(bcnodes):
    return bcnodes["models.qwen_lm.images"]


@pytest.mark.parametrize("hw, expected", [
    ((128, 128), (256, 256)),
    ((300, 200), (320, 224)),
    ((200, 300), (224, 320)),
    ((512, 512), (512, 512)),
    ((1080, 1920), (1088, 1920)),
    ((1920, 1080), (1920, 1088)),
    ((4096, 4096), (4096, 4096)),
    ((5000, 5000), (4096, 4096)),
    ((1, 200), (32, 3648)),
])
def test_smart_resize_sizes(images, hw, expected):
    assert images.smart_resize(*hw) == expected


def test_smart_resize_is_the_official_one(images):
    official = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl").smart_resize
    sizes = [1, 2, 15, 16, 17, 31, 32, 33, 47, 48, 63, 64, 100, 255, 256, 257, 333, 480, 511, 720, 1000, 1080, 1440,
             2047, 2160, 3000, 4095, 4097, 6000, 8192]
    pairs = [(h, w) for h in sizes for w in sizes if max(h, w) / min(h, w) <= 200]
    assert len(pairs) > 800
    for h, w in pairs:
        assert images.smart_resize(h, w) == official(h, w, factor=32, min_pixels=65536, max_pixels=16777216), (h, w)


@pytest.mark.parametrize("hw", [(1, 201), (201, 1), (10, 2001)])
def test_aspect_ratio_over_200_is_refused(images, hw):
    with pytest.raises(ValueError, match="more than 200 times wider than tall .* crop it"):
        images.smart_resize(*hw)
    with pytest.raises(ValueError, match="more than 200 times"):
        images.prepare_images(torch.zeros((1, *hw, 3)))


@pytest.mark.parametrize("hw", [(0, 64), (64, 0), (0, 0)])
def test_an_image_without_pixels_is_refused(images, hw):
    h, w = hw
    action = "has no pixels: connect an image with at least one pixel per side, or disconnect image"
    with pytest.raises(ValueError, match=f"an image of {w}x{h} {action}"):
        images.smart_resize(h, w)
    with pytest.raises(ValueError, match=f"an image of {w}x{h} {action}"):
        images.prepare_images(torch.zeros((1, h, w, 3)))


def test_none_and_empty(images):
    assert images.prepare_images(None) is None
    assert images.prepare_images(torch.zeros((0, 64, 64, 3))) is None


@pytest.mark.parametrize("shape", [(64, 64, 3), (1, 64, 64, 2), (1, 64, 64, 1), (1, 3, 64, 64)])
def test_not_an_image_batch(images, shape):
    with pytest.raises(ValueError, match=r"image must be an IMAGE batch \(frames, height, width, 3\)"):
        images.prepare_images(torch.zeros(shape))


def frames(n, h, w, seed=0):
    """`n` 8-bit noise frames (the worst case for the bicubic overshoot), as float32 in 0..1, and as uint8."""
    u8 = (torch.rand((n, h, w, 3), generator=torch.Generator().manual_seed(seed)) * 255).round().to(torch.uint8)
    return u8.float() / 255, u8


@pytest.mark.parametrize("hw", [(100, 150), (61, 47), (300, 200), (700, 500)])
def test_resize_is_pil_bicubic_within_a_level(images, hw):
    from PIL import Image

    x, u8 = frames(2, *hw)
    out = images.prepare_images(x)
    th, tw = images.smart_resize(*hw)
    assert out.shape == (2, th, tw, 3) and out.dtype == torch.float32 and out.device.type == "cpu"
    for i in range(2):
        pil = np.asarray(Image.fromarray(u8[i].numpy()).resize((tw, th), Image.BICUBIC)).astype(np.float32)
        assert float((out[i] * 255 - torch.from_numpy(pil)).abs().max()) < 1.2
    one_pass = torch.nn.functional.interpolate(x[:1].permute(0, 3, 1, 2), size=(th, tw), mode="bicubic",
                                               align_corners=False, antialias=True).clamp(0, 1)[0].permute(1, 2, 0)
    assert float((one_pass - out[0]).abs().max()) * 255 > 10, "one 2-D pass clamped at the end is far off"


def test_frames_are_independent(images):
    x, _ = frames(3, 90, 70)
    out = images.prepare_images(x)
    assert all(torch.equal(out[i], images.prepare_images(x[i:i + 1])[0]) for i in range(3))


def test_a_frame_at_its_size_is_kept(images):
    x, _ = frames(2, 512, 512)
    out = images.prepare_images(x)
    assert torch.equal(out, x) and out.data_ptr() != x.data_ptr()


def test_alpha_is_dropped(images):
    x, _ = frames(2, 100, 80)
    rgba = torch.cat([x, torch.rand((2, 100, 80, 1))], dim=-1)
    assert torch.equal(images.prepare_images(rgba), images.prepare_images(x))
    same = torch.cat([frames(1, 512, 512)[0], torch.zeros((1, 512, 512, 1))], dim=-1)
    assert torch.equal(images.prepare_images(same), same[..., :3])


@pytest.mark.parametrize("hw", [(512, 512), (100, 80)])
def test_values_are_clamped(images, hw):
    x = torch.rand((1, *hw, 3), generator=torch.Generator().manual_seed(4)) * 1.6 - 0.3
    before = x.clone()
    out = images.prepare_images(x)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0 and float(out.max()) == 1.0 and float(out.min()) == 0.0
    assert torch.equal(x, before), "the input is not modified"


@pytest.mark.parametrize("hw", [(100, 80), (512, 512)])
def test_half_precision(images, hw):
    x, _ = frames(2, *hw)
    reference = images.prepare_images(x)
    assert torch.equal(images.prepare_images(x.half()), reference), "float16 8-bit data reads as its float32 source"
    bf16 = x.bfloat16()
    assert torch.equal(images.prepare_images(bf16), images.prepare_images(bf16.float())), "bfloat16: its exact values"
    continuous = torch.rand((2, *hw, 3), generator=torch.Generator().manual_seed(9)).half()
    assert torch.equal(images.prepare_images(continuous), images.prepare_images(continuous.float()))
    assert images.prepare_images(x.half()).dtype == torch.float32

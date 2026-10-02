"""Frequency merge (libs/frequency.py) against hand-written expectations:

  - detail = base, strength 1: base again (low + high of one image is the image);
  - strength 0: the Gaussian low-pass of base, whatever detail is (the convolution written out in
    test_lib_filters.py);
  - the colour comes from base: flat base, flat detail of another colour -> base;
  - the texture comes from detail: a one-pixel checkerboard on detail (no low frequency at sigma 4)
    lands on a flat base as base + strength x checkerboard;
  - clamped to [0, 1]; RGBA in -> RGB out; the output dtype is the inputs' promoted dtype;
  - a batch is merged image by image (each equal to its own single-image merge);
  - another size or image count, and a sigma the image cannot hold, are errors that say what to do.
"""

import pytest
import torch
import torch.nn.functional as F

from test_lib_filters import low_pass

CPU = torch.device("cpu")


@pytest.fixture
def fq(bcnodes):
    return bcnodes["libs.frequency"].frequency_merge


def picture(seed, n=1, h=40, w=56):
    """Smooth shapes plus fine noise, inside (0.1, 0.9)."""
    g = torch.Generator().manual_seed(seed)
    coarse = F.interpolate(torch.rand(n, 3, h // 8, w // 8, generator=g), size=(h, w), mode="bilinear")
    fine = torch.rand(n, 3, h, w, generator=g) - 0.5
    return (0.15 + 0.6 * coarse + 0.1 * fine).permute(0, 2, 3, 1).contiguous()


def test_detail_equal_to_base_gives_base(fq):
    a = picture(1, n=2)
    assert torch.allclose(fq(a, a.clone(), 3.0, 1.0, CPU), a, atol=1e-6)


def test_strength_zero_is_the_low_pass_of_base(fq):
    a = picture(2)
    expected = low_pass(a.permute(0, 3, 1, 2), 2.5)[0].permute(1, 2, 0).clamp(0, 1).float()
    for detail in (picture(3), torch.zeros_like(a), torch.ones_like(a)):
        assert torch.allclose(fq(a, detail, 2.5, 0.0, CPU)[0], expected, atol=1e-6)


def test_colour_comes_from_base(fq):
    base, detail = torch.full((1, 24, 32, 3), 0.3), torch.full((1, 24, 32, 3), 0.9)
    assert torch.allclose(fq(base, detail, 2.0, 1.0, CPU), base, atol=1e-6)


@pytest.mark.parametrize("strength", [0.5, 1.0, 1.5])
def test_texture_comes_from_detail(fq, strength):
    h, w = 32, 40
    checker = (((torch.arange(h).view(-1, 1) + torch.arange(w).view(1, -1)) % 2) * 2 - 1).float() * 0.1  # +-0.1, period 2
    base = torch.full((1, h, w, 3), 0.5)
    detail = (0.4 + checker).view(1, h, w, 1).expand(1, h, w, 3).contiguous()
    expected = (0.5 + strength * checker).view(1, h, w, 1).expand(1, h, w, 3)
    assert torch.allclose(fq(base, detail, 4.0, strength, CPU), expected, atol=1e-3)


def test_clamped_to_unit_range(fq):
    base = torch.full((1, 24, 32, 3), 0.95)
    detail = torch.zeros(1, 24, 32, 3)
    detail[:, ::2, ::2] = 1.0
    out = fq(base, detail, 2.0, 2.0, CPU)
    assert out.min() >= 0.0 and out.max() <= 1.0 and out.max() == 1.0 and out.min() < 0.95


def test_rgba_in_rgb_out(fq):
    a = picture(4)
    rgba = torch.cat([a, torch.full(a.shape[:3] + (1,), 0.25)], dim=-1)
    out = fq(rgba, rgba, 2.0, 1.0, CPU)
    assert out.shape == a.shape and torch.allclose(out, a, atol=1e-6)


@pytest.mark.parametrize("da, db, expected", [
    (torch.float32, torch.float32, torch.float32),
    (torch.float16, torch.float16, torch.float16),
    (torch.float16, torch.float32, torch.float32),
    (torch.float32, torch.float16, torch.float32),
])
def test_output_dtype_is_the_promoted_input_dtype(fq, da, db, expected):
    a = picture(5)
    out = fq(a.to(da), a.to(db), 2.0, 1.0, CPU)
    assert out.dtype == expected and out.shape == a.shape
    assert torch.allclose(out.float(), a.to(da).float(), atol=1e-3)  # within float16 rounding


def test_batch_merged_image_by_image(fq):
    base, detail = picture(6, n=3), picture(7, n=3)
    out = fq(base, detail, 3.0, 0.8, CPU)
    assert out.shape == (3, 40, 56, 3)
    for i in range(3):
        assert torch.equal(out[i:i + 1], fq(base[i:i + 1], detail[i:i + 1], 3.0, 0.8, CPU))


def test_empty_batch_gives_an_empty_batch(fq):
    out = fq(torch.zeros(0, 16, 16, 3), torch.zeros(0, 16, 16, 3), 2.0, 1.0, CPU)
    assert out.shape == (0, 16, 16, 3)


@pytest.mark.parametrize("detail", [torch.zeros(1, 24, 30, 3), torch.zeros(1, 20, 32, 3), torch.zeros(2, 24, 32, 3)])
def test_another_size_or_count_says_what_to_do(fq, detail):
    with pytest.raises(ValueError, match="base is 32x24 \\(1 image\\(s\\)\\).*same size and image count"):
        fq(torch.zeros(1, 24, 32, 3), detail, 2.0, 1.0, CPU)


def test_sigma_the_image_cannot_hold_says_what_to_do(fq):
    for sigma in (7.7, 8.0):  # ceil(3 x sigma) reaches 24 rows
        with pytest.raises(ValueError, match="set split_sigma to 7.6 or less"):
            fq(torch.zeros(1, 24, 32, 3), torch.zeros(1, 24, 32, 3), sigma, 1.0, CPU)
    assert fq(torch.zeros(1, 24, 32, 3), torch.zeros(1, 24, 32, 3), 7.6, 1.0, CPU).shape == (1, 24, 32, 3)


def test_inputs_are_not_modified(fq):
    base, detail = picture(8), picture(9)
    b0, d0 = base.clone(), detail.clone()
    fq(base, detail, 3.0, 1.5, CPU)
    assert torch.equal(base, b0) and torch.equal(detail, d0)

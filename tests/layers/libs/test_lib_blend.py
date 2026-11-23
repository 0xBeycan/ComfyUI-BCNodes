"""Laplacian-pyramid blend (libs/blend.py):

  - weight 1 everywhere gives the patch, weight 0 the base exactly; any size (odd sides too);
  - an image too small for one level (shorter side under 2 x SMALLEST) is the plain alpha blend;
  - multiband: across a step in the weight, a constant difference fades in over several pixels on the
    weight-0 side (its low band's weight is blurred), while a one-pixel checkerboard difference (the
    finest band only) stays on the weight-1 side.
"""

import pytest
import torch


@pytest.fixture
def blend(bcnodes):
    return bcnodes["libs.blend"]


def picture(seed, h, w):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(3, h, w, generator=g) * 0.6 + 0.2


@pytest.mark.parametrize("size", [(64, 64), (37, 53), (130, 97)])
def test_full_and_zero_weight(blend, size):
    base, patch = picture(1, *size), picture(2, *size)
    assert torch.allclose(blend.laplacian_blend(base, patch, torch.ones(size)), patch, atol=1e-5)
    assert torch.equal(blend.laplacian_blend(base, patch, torch.zeros(size)), base)
    assert blend.laplacian_blend(base, patch, torch.rand(size)).shape == (3, *size)


def test_too_small_for_a_level_is_the_alpha_blend(blend):
    base, patch, weight = picture(1, 31, 40), picture(2, 31, 40), torch.rand(31, 40)
    assert torch.allclose(blend.laplacian_blend(base, patch, weight), base + weight * (patch - base), atol=1e-6)


def test_low_frequencies_blend_wider_than_fine_detail(blend):
    h, w = 64, 128
    weight = torch.zeros(h, w)
    weight[:, w // 2:] = 1.0
    base = torch.full((3, h, w), 0.3)
    flat = blend.laplacian_blend(base, base + 0.2, weight) - base
    row = flat[0, h // 2]
    # the constant difference reaches pixels left of the step, which a plain alpha blend leaves at 0
    assert 0.0 < float(row[w // 2 - 3]) < 0.2 and float(row[w // 2 - 6]) > 0.0
    checker = (torch.arange(h)[:, None] + torch.arange(w)[None, :]) % 2 * 0.2 - 0.1
    fine = blend.laplacian_blend(base, base + checker, weight) - base
    assert float(fine[0, h // 2, : w // 2 - 2].abs().max()) < 1e-5
    assert torch.allclose(fine[0, h // 2, w // 2 + 2:], checker[h // 2, w // 2 + 2:], atol=1e-5)

"""gauss_reflect (libs/filters.py) against the separable Gaussian written out here in float64 as a
convolution: kernel exp(-t^2 / 2 sigma^2) over |t| <= ceil(3 sigma), normalised, rows then columns,
each reflect-padded. Batches and channels, odd sizes, a radius one short of the image, a
non-contiguous input; sigma <= 0.2 returns the input itself; the input is not modified."""

import math

import pytest
import torch
import torch.nn.functional as F


def low_pass(x, sigma):
    """The reference blur of a (B, C, H, W) batch, float64."""
    radius = math.ceil(3 * sigma)
    t = torch.arange(-radius, radius + 1, dtype=torch.float64)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    b, c, h, w = x.shape
    y = x.double().reshape(b * c, 1, h, w)
    y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1))
    return y.reshape(b, c, h, w)


@pytest.fixture
def gauss(bcnodes):
    return bcnodes["libs.filters"].gauss_reflect


@pytest.mark.parametrize("shape, sigma", [
    ((2, 3, 40, 56), 0.5), ((2, 3, 40, 56), 2.2), ((1, 1, 37, 129), 5.0), ((3, 1, 31, 64), 3.3),  # radius 10 < 31
    ((1, 2, 31, 50), 10.0),  # radius 30, one short of the 31 rows
])
def test_equals_the_written_out_convolution(gauss, shape, sigma):
    x = torch.rand(shape, generator=torch.Generator().manual_seed(1))
    out = gauss(x, sigma)
    assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous()
    assert torch.allclose(out.double(), low_pass(x, sigma), atol=1e-6)


def test_a_non_contiguous_input(gauss):
    x = torch.rand(1, 40, 56, 3, generator=torch.Generator().manual_seed(2)).permute(0, 3, 1, 2)
    assert torch.allclose(gauss(x, 3.0).double(), low_pass(x.contiguous(), 3.0), atol=1e-6)


def test_a_constant_stays_constant(gauss):
    assert torch.allclose(gauss(torch.full((1, 3, 24, 32), 0.37), 4.0), torch.full((1, 3, 24, 32), 0.37), atol=1e-6)


def test_small_sigma_returns_the_input(gauss):
    x = torch.rand(1, 1, 8, 8)
    assert gauss(x, 0.2) is x


def test_input_not_modified(gauss):
    x = torch.rand(2, 3, 30, 30, generator=torch.Generator().manual_seed(3))
    x0 = x.clone()
    gauss(x, 4.0)
    assert torch.equal(x, x0)

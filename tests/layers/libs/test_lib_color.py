"""Colour match over a band (libs/color.match_band):

  - over the band, every channel of the result has the reference's mean and standard deviation, and the
    affine map found there is applied to the whole patch;
  - a channel flat in the band is only shifted; an empty band leaves the patch as it is; the result is
    clamped to [0, 1].
"""

import pytest
import torch


@pytest.fixture
def match(bcnodes):
    return bcnodes["libs.color"].match_band


def test_band_statistics_become_the_references(match):
    g = torch.Generator().manual_seed(3)
    reference = torch.rand(3, 32, 40, generator=g) * 0.4 + 0.3
    patch = reference * 0.5 + 0.1  # darker, lower contrast, a tint
    band = torch.zeros(32, 40, dtype=torch.bool)
    band[:4], band[-4:] = True, True
    out = match(patch, reference, band)
    for c in range(3):
        assert torch.allclose(out[c, band].mean(), reference[c, band].mean(), atol=1e-5)
        assert torch.allclose(out[c, band].std(correction=0), reference[c, band].std(correction=0), atol=1e-5)
    # the same affine map inside: patch was an affine map of reference, so all of it comes back
    assert torch.allclose(out, reference, atol=1e-5)


def test_flat_channel_is_shifted_and_empty_band_is_a_no_op(match):
    reference = torch.full((3, 8, 8), 0.6)
    patch = torch.full((3, 8, 8), 0.2)
    patch[1, 4:, :] = 0.3  # channel 1 varies outside the band only
    band = torch.zeros(8, 8, dtype=torch.bool)
    band[:2] = True
    out = match(patch, reference, band)
    assert torch.allclose(out[0], torch.full((8, 8), 0.6)) and torch.allclose(out[1, 4:], torch.full((4, 8), 0.7))
    assert match(patch, reference, torch.zeros(8, 8, dtype=torch.bool)) is patch


def test_clamped(match):
    reference = torch.full((3, 4, 4), 0.95)
    patch = torch.linspace(0.0, 1.0, 16).reshape(1, 4, 4).repeat(3, 1, 1)
    band = torch.zeros(4, 4, dtype=torch.bool)
    band[0, 0] = True  # one pixel: flat, shifted by +0.95
    out = match(patch, reference, band)
    assert float(out.max()) == 1.0 and float(out.min()) >= 0.0

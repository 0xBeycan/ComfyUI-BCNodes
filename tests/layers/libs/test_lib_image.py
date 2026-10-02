"""libs/image.py's float -> 8-bit conversion of half-precision values, and output_dtype:

  - tensor_to_u8 of a float16 or bfloat16 copy of every 8-bit level gives the uint8 of its float32
    source k / 255, where the plain `255 * x` cast truncates float16(1 / 255) to 0; a half frame read
    as float32 first (Image Resize's lanczos) and converted with `stored` set to its dtype, the same;
  - on values that are not levels, a half value is truncated after adding its margin
    (floor(255 * x + 1/16) for float16, floor(255 * x + 1/2) for bfloat16, written out here);
  - a float32 frame gives what the expression it replaced gives (`np.clip(255.0 * x, 0,
    255).astype(np.uint8)`, written out here), out-of-range and NaN values included;
  - output_dtype: a half input's own dtype, float32 for anything else.
"""

import numpy as np
import pytest
import torch

LEVELS = torch.arange(256, dtype=torch.float32) / 255  # what a float32 load holds
MARGIN = {torch.float16: 1 / 16, torch.bfloat16: 1 / 2}


@pytest.fixture
def image(bcnodes):
    return bcnodes["libs.image"]


def frame_of_levels():
    """Every 8-bit level in an (8, 32, 3) frame (each channel a shifted copy)."""
    return torch.stack([LEVELS.roll(s).view(8, 32) for s in (0, 85, 170)], dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_every_level_of_a_half_frame_gives_its_float32_source(image, dtype):
    source = frame_of_levels()
    expected = image.tensor_to_u8(source)
    assert np.array_equal(expected[..., 0].reshape(-1), np.arange(256, dtype=np.uint8))  # every level is k
    half = source.to(dtype)
    assert np.array_equal(image.tensor_to_u8(half), expected)
    assert np.array_equal(image.tensor_to_u8(half.float(), stored=dtype), expected)
    plain = np.clip(255.0 * half.float().numpy(), 0, 255).astype(np.uint8)  # without the margin
    assert dtype == torch.bfloat16 or not np.array_equal(plain, expected)  # float16(1/255) x 255 = 0.99998 -> 0


def test_the_margins_hold_every_level_with_room(image):
    for dtype, margin in MARGIN.items():
        error = (LEVELS.to(dtype).double() * 255 - torch.arange(256).double()).abs().max().item()
        assert error < margin < 1 - error, (dtype, error)  # float16: 0.0623 < 1/16; bfloat16: 0.498 < 1/2


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_values_between_levels_are_truncated_after_the_margin(image, dtype):
    x = torch.rand(64, 48, 3, generator=torch.Generator().manual_seed(1)).to(dtype)
    expected = np.floor(np.clip(255.0 * x.double().numpy() + MARGIN[dtype], 0, 255)).astype(np.uint8)
    assert np.array_equal(image.tensor_to_u8(x), expected)


def test_a_float32_frame_is_the_old_expression(image):
    x = torch.rand(16, 24, 3, generator=torch.Generator().manual_seed(3)).mul_(1.4).sub_(0.2)
    x[0, :5, 0] = float("nan")
    x = torch.cat([x, frame_of_levels()[:, :24]], dim=0)
    with np.errstate(invalid="ignore"):
        assert np.array_equal(image.tensor_to_u8(x), np.clip(255.0 * x.numpy(), 0, 255).astype(np.uint8))


def test_output_dtype(image):
    assert image.output_dtype(torch.zeros(1, dtype=torch.float16)) == torch.float16
    assert image.output_dtype(torch.zeros(1, dtype=torch.bfloat16)) == torch.bfloat16
    for dtype in (torch.float32, torch.float64, torch.uint8, torch.bool):
        assert image.output_dtype(torch.zeros(1, dtype=dtype)) == torch.float32

"""libs/image.py's half-precision reads and its float -> 8-bit conversion, on every 8-bit level:

  - requantized gives back exactly the float32 k / 255 of a float32 load from float16 and bfloat16
    copies of it; a float32 tensor is handed back itself;
  - tensor_to_u8 of a half copy gives the uint8 of its float32 source, where the plain
    `255 * x.float()` truncates float16(1 / 255) to 0; a float32 frame gives what the expression it
    replaced gives (`np.clip(255.0 * x, 0, 255).astype(np.uint8)`, written out here), out-of-range
    and NaN values included;
  - output_dtype: a half input's own dtype, float32 for anything else.
"""

import numpy as np
import pytest
import torch

LEVELS = torch.arange(256, dtype=torch.float32) / 255  # what a float32 load holds


@pytest.fixture
def image(bcnodes):
    return bcnodes["libs.image"]


def frame_of_levels():
    """Every 8-bit level in an (8, 32, 3) frame: (each channel a shifted copy)."""
    return torch.stack([LEVELS.roll(s).view(8, 32) for s in (0, 85, 170)], dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_requantized_gives_the_float32_levels_back(image, dtype):
    half = LEVELS.to(dtype)
    assert not torch.equal(half.float(), LEVELS)  # the half copy is not the float32 load
    back = image.requantized(half)
    assert back.dtype == torch.float32 and torch.equal(back, LEVELS)
    out = torch.empty(256)
    assert image.requantized(half, out=out) is out and torch.equal(out, LEVELS)


def test_requantized_hands_float32_back_itself(image):
    x = torch.rand(4, 5)
    assert image.requantized(x) is x


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tensor_to_u8_of_a_half_frame_is_its_float32_source(image, dtype):
    source = frame_of_levels()
    expected = image.tensor_to_u8(source)
    assert np.array_equal(expected[..., 0].reshape(-1), np.arange(256, dtype=np.uint8))  # every level is k
    assert np.array_equal(image.tensor_to_u8(source.to(dtype)), expected)
    if dtype == torch.float16:  # what the conversion without the requantize gives: k - 1 for some levels
        plain = np.clip(255.0 * source.to(dtype).float().numpy(), 0, 255).astype(np.uint8)
        assert not np.array_equal(plain, expected) and plain.reshape(-1).min() == 0 and int(plain[0, 1, 0]) == 0


def test_tensor_to_u8_of_a_float32_frame_is_the_old_expression(image):
    x = torch.rand(16, 24, 3, generator=torch.Generator().manual_seed(3)).mul_(1.4).sub_(0.2)
    x[0, :5, 0] = float("nan")
    x = torch.cat([x, frame_of_levels()[:, :24]], dim=0)
    assert np.array_equal(image.tensor_to_u8(x), np.clip(255.0 * x.numpy(), 0, 255).astype(np.uint8))


def test_output_dtype(image):
    assert image.output_dtype(torch.zeros(1, dtype=torch.float16)) == torch.float16
    assert image.output_dtype(torch.zeros(1, dtype=torch.bfloat16)) == torch.bfloat16
    for dtype in (torch.float32, torch.float64, torch.uint8, torch.bool):
        assert image.output_dtype(torch.zeros(1, dtype=dtype)) == torch.float32

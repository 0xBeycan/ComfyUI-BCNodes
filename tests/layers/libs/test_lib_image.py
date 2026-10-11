"""libs/image.py's half-precision read and its float -> 8-bit conversion:

  - float_frame of a float16 copy of 8-bit data (every level) gives exactly the float32 k / 255; of
    continuous float16 values, their exact values (.float()); of a continuous frame whose values all lie
    within 1/16 of a level, those levels, each value moved by at most 1/16 of a level; of bfloat16, its
    exact values; of a NaN in an 8-bit frame, the exact values; of float32, the frame itself;
  - float16 keeps every level within 1/16 of it, which no other value of 0..1 at float16's step does
    beyond 1/16 (the tolerance holds the worst level, 0.0623);
  - tensor_to_u8 of a float16 or bfloat16 copy of every level gives the uint8 of its float32 source,
    where the plain `255 * x` cast truncates float16(1 / 255) to 0; a float16 frame that is not 8-bit
    data gives the float32 truncation of its own values; bfloat16 values are truncated after
    BF16_U8_MARGIN (floor(255 * x + 1/2), written out here); a float32 frame gives what the expression it
    replaced gives (`np.clip(255.0 * x, 0, 255).astype(np.uint8)`, written out here);
  - output_dtype: a half input's own dtype, float32 for anything else;
  - thumbnail_frame (Captioner's and Prompter's image cap): a frame within max_side is returned as it is (the
    same tensor, any dtype, never enlarged); a larger one is exactly what PIL's Image.thumbnail((max_side,
    max_side)) gives on the frame as 8-bit RGB (alpha dropped), a float32 (1, h, w, 3) frame.
"""

import numpy as np
import pytest
import torch

LEVELS = torch.arange(256, dtype=torch.float32) / 255  # what a float32 load holds


@pytest.fixture
def image(bcnodes):
    return bcnodes["libs.image"]


def frame_of_levels():
    """Every 8-bit level in an (8, 32, 3) frame (each channel a shifted copy)."""
    return torch.stack([LEVELS.roll(s).view(8, 32) for s in (0, 85, 170)], dim=-1)


def test_float16_8_bit_data_is_read_as_its_levels(image):
    source = frame_of_levels()
    half = source.half()
    assert not torch.equal(half.float(), source)
    assert torch.equal(image.float_frame(half), source)
    out = torch.empty(source.shape)
    assert image.float_frame(half, out=out) is out and torch.equal(out, source)
    assert (LEVELS.half().double() * 255 - torch.arange(256)).abs().max() < image.LEVEL_TOLERANCE  # 0.0623 < 1/16


def test_other_half_frames_are_read_as_their_values(image):
    continuous = torch.rand(64, 48, 3, generator=torch.Generator().manual_seed(1)).half()
    assert torch.equal(image.float_frame(continuous), continuous.float())
    bf16 = frame_of_levels().bfloat16()
    assert torch.equal(image.float_frame(bf16), bf16.float())
    with_nan = frame_of_levels().half()
    with_nan[0, 0, 0] = float("nan")
    assert torch.equal(image.float_frame(with_nan)[1:], with_nan[1:].float())
    x = torch.rand(4, 5)
    assert image.float_frame(x) is x


def test_a_continuous_frame_close_to_levels_moves_at_most_the_tolerance(image):
    # values 0.04 of a level off, under 0.25 (where float16's own rounding adds at most 0.016)
    near = (torch.arange(60, dtype=torch.float32) + torch.linspace(-0.04, 0.04, 60)) / 255
    half = near.half()
    read = image.float_frame(half)
    assert torch.equal(read, torch.arange(60, dtype=torch.float32) / 255)  # taken for 8-bit data
    assert float((read.double() - half.double()).abs().max()) * 255 <= image.LEVEL_TOLERANCE


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_every_level_of_a_half_frame_gives_its_float32_uint8(image, dtype):
    source = frame_of_levels()
    expected = image.tensor_to_u8(source)
    assert np.array_equal(expected[..., 0].reshape(-1), np.arange(256, dtype=np.uint8))  # every level is k
    half = source.to(dtype)
    assert np.array_equal(image.tensor_to_u8(half), expected)
    assert np.array_equal(image.tensor_to_u8(image.float_frame(half), stored=dtype), expected)
    plain = np.clip(255.0 * half.float().numpy(), 0, 255).astype(np.uint8)  # the plain cast
    assert dtype == torch.bfloat16 or not np.array_equal(plain, expected)  # float16(1/255) x 255 = 0.99998 -> 0


def test_values_that_are_not_levels(image):
    x = torch.rand(64, 48, 3, generator=torch.Generator().manual_seed(2))
    half = x.half()  # not 8-bit data: truncated as float32 truncates its values
    assert np.array_equal(image.tensor_to_u8(half), image.tensor_to_u8(half.float()))
    bf16 = x.bfloat16()
    expected = np.floor(np.clip(255.0 * bf16.double().numpy() + image.BF16_U8_MARGIN, 0, 255)).astype(np.uint8)
    assert np.array_equal(image.tensor_to_u8(bf16), expected)


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


def thumbnail(frame, max_side):
    """PIL's Image.thumbnail((max_side, max_side)) on the frame as 8-bit RGB, as a float32 (1, h, w, 3) tensor."""
    from PIL import Image

    u8 = np.clip(255.0 * frame[0, :, :, :3].numpy(), 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8)
    pil.thumbnail((max_side, max_side))
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0).unsqueeze(0)


@pytest.mark.parametrize("h, w, size", [(1536, 2048, (768, 1024)), (2048, 1536, (1024, 768)), (1025, 10, (1024, 10)),
                                        (1100, 1100, (1024, 1024))])
def test_a_large_frame_is_pil_thumbnail_of_it(image, h, w, size):
    frame = torch.rand(1, h, w, 3)
    out = image.thumbnail_frame(frame, 1024)
    expected = thumbnail(frame, 1024)
    assert out.dtype == torch.float32 and tuple(out.shape) == (1, *size, 3) == tuple(expected.shape)
    assert torch.equal(out, expected)


def test_the_cap_is_the_max_side_given(image):
    frame = torch.rand(1, 600, 800, 3)
    out = image.thumbnail_frame(frame, 512)
    assert out.shape == (1, 384, 512, 3) and torch.equal(out, thumbnail(frame, 512))


def test_alpha_is_dropped_on_a_large_frame(image):
    frame = torch.rand(1, 1200, 900, 4)
    out = image.thumbnail_frame(frame, 1024)
    assert out.shape == (1, 1024, 768, 3) and torch.equal(out, thumbnail(frame, 1024))


@pytest.mark.parametrize("h, w", [(600, 800), (1024, 1024), (1024, 300), (8, 8)])
def test_a_frame_within_the_cap_is_never_changed_or_enlarged(image, h, w):
    frame = torch.rand(1, h, w, 4, dtype=torch.float16)
    assert image.thumbnail_frame(frame, 1024) is frame

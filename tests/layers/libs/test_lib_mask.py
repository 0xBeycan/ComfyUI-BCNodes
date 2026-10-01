"""Mask operations (libs/mask.py) against hand-made masks and scipy / PIL written out here.

  - fill_holes: 4-connected background (a hole closed only diagonally is filled), a hole that
    touches the border stays open, values below 1/255 are background, and equality with
    scipy.ndimage.binary_fill_holes (its default cross structure) on random blobs;
  - grow_and_blur: equality with the grey morphology + PIL blur formula written out here
    (scipy.ndimage grey_dilation / grey_erosion with the cross footprint on 8-bit levels),
    grow and shrink, invert, blur;
  - draw_mask_on_image: the blend formula by hand, alpha as opacity, RGBA alpha, masks repeated
    and scaled, the colour forms;
  - blockify: blocks of a bounding box by hand, the last block taking the remainder.
"""

import numpy as np
import pytest
import torch
from PIL import Image, ImageFilter

CPU = torch.device("cpu")


@pytest.fixture
def mk(bcnodes):
    return bcnodes["libs.mask"]


def blobs(seed, n=3, h=48, w=64):
    """Random soft blobs with holes and gaps: thresholded smoothed noise, some pixels below 1/255."""
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(n, 1, h // 4, w // 4, generator=g)
    field = torch.nn.functional.interpolate(noise, size=(h, w), mode="bilinear")[:, 0]
    return torch.where(field > 0.5, field, field * 0.002)


def scipy_grow_and_blur(frames, invert_mask, grow, blur):
    import scipy.ndimage

    def to_u8(x):
        return np.clip(255.0 * x, 0, 255).astype(np.uint8)

    out = []
    for frame in frames.numpy():
        if invert_mask:
            frame = 1 - frame
        arr = to_u8(frame).astype(np.float32) / 255.0
        for _ in range(abs(grow)):
            op = scipy.ndimage.grey_erosion if grow < 0 else scipy.ndimage.grey_dilation
            arr = op(arr, footprint=[[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        blurred = Image.fromarray(to_u8(arr)).filter(ImageFilter.GaussianBlur(blur))
        out.append(torch.from_numpy(np.asarray(blurred).astype(np.float32) / 255.0))
    return torch.stack(out)


# --- fill_holes -------------------------------------------------------------------------------


def test_hole_closed_only_diagonally_is_filled(mk):
    mask = torch.zeros(1, 5, 5)
    for y, x in ((1, 2), (2, 1), (2, 3), (3, 2)):
        mask[0, y, x] = 1.0
    out = mk.fill_holes(mask)
    assert out[0, 2, 2] == 1.0 and out.sum() == 5


def test_hole_touching_the_border_stays_open(mk):
    mask = torch.ones(1, 6, 6)
    mask[0, 2:4, 0:3] = 0.0  # opens to the left border
    mask[0, 2:4, 4] = 0.0  # enclosed
    out = mk.fill_holes(mask)
    assert torch.equal(out[0, 2:4, 0:3], torch.zeros(2, 3)) and torch.equal(out[0, 2:4, 4], torch.ones(2))


def test_values_below_one_level_are_background(mk):
    mask = torch.ones(1, 5, 5)
    mask[0, 2, 2] = 0.003  # 255 * 0.003 < 1: background, enclosed, so filled
    mask[0, 0, 0] = 0.003  # background on the border: stays 0
    out = mk.fill_holes(mask)
    assert out[0, 2, 2] == 1.0 and out[0, 0, 0] == 0.0 and out.dtype == torch.float32


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fill_holes_equals_scipy(mk, seed):
    from scipy.ndimage import binary_fill_holes

    frames = blobs(seed)
    ref = np.stack([binary_fill_holes(np.clip(255.0 * f, 0, 255).astype(np.uint8) > 0) for f in frames.numpy()])
    assert torch.equal(mk.fill_holes(frames), torch.from_numpy(ref.astype(np.float32)))


# --- grow_and_blur ----------------------------------------------------------------------------


@pytest.mark.parametrize("invert_mask, grow, blur", [(False, 3, 0), (False, -2, 0), (True, 4, 2), (False, 0, 3), (True, -5, 1)])
def test_grow_and_blur_equals_grey_morphology_and_pil(mk, invert_mask, grow, blur):
    frames = blobs(7)
    assert torch.equal(mk.grow_and_blur(frames, invert_mask, grow, blur), scipy_grow_and_blur(frames, invert_mask, grow, blur))


def test_grow_by_one_is_the_cross(mk):
    mask = torch.zeros(1, 5, 5)
    mask[0, 2, 2] = 1.0
    expected = torch.zeros(1, 5, 5)
    expected[0, 1:4, 2] = 1.0
    expected[0, 2, 1:4] = 1.0
    assert torch.equal(mk.grow_and_blur(mask, False, 1, 0), expected)
    assert torch.equal(mk.grow_and_blur(expected, False, -1, 0), mask)


# --- draw_mask_on_image -----------------------------------------------------------------------


def test_draw_blends_by_mask_times_alpha(mk):
    image = torch.rand(2, 4, 4, 3)
    mask = torch.rand(2, 4, 4)
    out = mk.draw_mask_on_image(image, mask, "255, 0, 0, 0.5", CPU)
    blend = mask.unsqueeze(-1) * 0.5
    assert torch.equal(out, image * (1 - blend) + torch.tensor([1.0, 0.0, 0.0]) * blend)


def test_draw_rgba_keeps_the_larger_alpha(mk):
    image = torch.rand(1, 4, 4, 4)
    mask = torch.rand(1, 4, 4)
    out = mk.draw_mask_on_image(image, mask, "#00ff00", CPU)
    blend = mask.unsqueeze(-1)
    assert torch.equal(out[..., :3], image[..., :3] * (1 - blend) + torch.tensor([0.0, 1.0, 0.0]) * blend)
    assert torch.equal(out[..., 3:], torch.maximum(image[..., 3:], blend))


def test_draw_repeats_fewer_masks_and_scales_them_nearest(mk):
    image = torch.zeros(3, 4, 4, 3)
    mask = torch.zeros(2, 2, 2)
    mask[0, 0, 0] = 1.0  # the top-left 2x2 block of frames 0 and 2
    out = mk.draw_mask_on_image(image, mask, "1, 1, 1", CPU)
    assert out[0, :2, :2].min() == 1.0 and out[0, 2:].max() == 0.0 and out[1].max() == 0.0 and torch.equal(out[2], out[0])


@pytest.mark.parametrize("text, rgb, alpha", [("255, 0, 0", [1.0, 0.0, 0.0], 1.0), ("1, 0.5, 0", [1.0, 0.5, 0.0], 1.0),
                                              ("255, 0, 0, 0.25", [1.0, 0.0, 0.0], 0.25), ("0, 255, 0, 51", [0.0, 1.0, 0.0], 0.2),
                                              ("#ff000080", [1.0, 0.0, 0.0], 128 / 255), ("#f008", [1.0, 0.0, 0.0], 136 / 255),
                                              ("#0f0", [0.0, 1.0, 0.0], 1.0), ("0.5", [0.5], 1.0)])
def test_draw_color_forms(mk, text, rgb, alpha):
    assert mk.parse_draw_color(text) == (rgb, alpha)


@pytest.mark.parametrize("text", ["", "red", "1, 2", "#12345", "#ggg", "1, 2, 3, 4, 5"])
def test_bad_draw_color_is_refused(mk, text):
    with pytest.raises(ValueError, match="color"):
        mk.parse_draw_color(text)


# --- blockify ---------------------------------------------------------------------------------


def test_blockify_fills_the_blocks_of_the_bounding_box_that_hold_the_mask(mk):
    mask = torch.zeros(1, 40, 40)
    mask[0, 5, 5] = 0.2  # bounding box rows / cols 5..24 (20 px): 20 // 8 = 2 blocks of 10
    mask[0, 24, 24] = 1.0
    expected = torch.zeros(1, 40, 40)
    expected[0, 5:15, 5:15] = 1.0
    expected[0, 15:25, 15:25] = 1.0
    assert torch.equal(mk.blockify(mask, 8, CPU), expected)


def test_blockify_last_block_takes_the_remainder(mk):
    mask = torch.zeros(1, 1, 34)
    mask[0, 0, 0] = 1.0
    mask[0, 0, 33] = 1.0  # 34 px, block 8: 4 blocks of 34 // 4 = 8, the last one 10 px
    expected = torch.cat([torch.ones(8), torch.zeros(16), torch.ones(10)])
    assert torch.equal(mk.blockify(mask, 8, CPU)[0, 0], expected)
    mask[0, 0, 12] = 1.0
    assert torch.equal(mk.blockify(mask, 8, CPU)[0, 0], torch.cat([torch.ones(16), torch.zeros(8), torch.ones(10)]))


def test_blockify_box_smaller_than_a_block_is_one_block(mk):
    mask = torch.zeros(2, 16, 16)
    mask[0, 3, 4] = 1.0
    mask[0, 6, 9] = 0.5
    out = mk.blockify(mask, 32, CPU)
    expected = torch.zeros(2, 16, 16)
    expected[0, 3:7, 4:10] = 1.0
    assert torch.equal(out, expected)

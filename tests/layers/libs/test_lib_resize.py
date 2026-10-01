"""Image Resize (libs/resize.py): the size plan worked out by hand for every keep_proportion,
the per-frame work against torch / PIL written out here, the padding modes, the mask and the
pass-through.

  - plan: stretch / resize / total_pixels / crop / the pad modes at each crop_position, the
    divisible_by floor and the padding that grows to the next multiple, RTX VSR's multiple of 8,
    a zero size refused;
  - resize_image: stretch equals F.interpolate on the batch (ComfyUI's common_upscale), lanczos
    equals PIL LANCZOS on 8-bit frames, crop takes the window, pad / pad_edge / pad_edge_pixel /
    pillarbox_blur fill the padding as stated, the mask follows the image, the padding mask
    without one, the 64x64 placeholder, the image at its size passed through as the same tensor;
  - parse_pad_color: every accepted form, a bad one refused.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

CPU = torch.device("cpu")


@pytest.fixture
def rs(bcnodes):
    return bcnodes["libs.resize"]


def run(rs, image, mask=None, width=64, height=64, method="bilinear", keep="stretch", color="0, 0, 0",
        position="center", divisible_by=2):
    return rs.resize_image(image, mask, width, height, method, keep, color, position, divisible_by, CPU)


# --- plan -------------------------------------------------------------------------------------


@pytest.mark.parametrize("width, height, size", [(64, 30, (30, 64)), (0, 30, (30, 96)), (40, 0, (48, 40)), (0, 0, (48, 96))])
def test_stretch_size_zero_keeps_the_source_side(rs, width, height, size):
    p = rs.plan(48, 96, width, height, "stretch", "center", 0, "bilinear")
    assert (p.size, p.crop, p.pad) == (size, None, None)


@pytest.mark.parametrize("width, height, size", [(64, 64, (32, 64)), (0, 30, (30, 60)), (48, 0, (24, 48)), (0, 0, (48, 96))])
def test_resize_keeps_the_aspect_inside_the_box(rs, width, height, size):
    assert rs.plan(48, 96, width, height, "resize", "center", 0, "bilinear").size == size


def test_total_pixels_at_the_source_aspect(rs):
    # 64 * 64 = 4096 px at aspect 2: width int(sqrt(8192)) = 90, height int(sqrt(2048)) = 45, floored to 2.
    assert rs.plan(48, 96, 64, 64, "total_pixels", "center", 2, "bilinear").size == (44, 90)


def test_divisible_by_floors_both_sides(rs):
    assert rs.plan(48, 96, 70, 37, "stretch", "center", 8, "bilinear").size == (32, 64)


@pytest.mark.parametrize("position, pad", [("center", (16, 16, 0, 0)), ("top", (0, 32, 0, 0)), ("bottom", (32, 0, 0, 0)),
                                           ("left", (16, 16, 0, 0)), ("right", (16, 16, 0, 0))])
def test_pad_places_a_wide_frame(rs, position, pad):
    p = rs.plan(48, 96, 64, 64, "pad", position, 2, "bilinear")
    assert (p.size, p.pad, p.output) == ((32, 64), pad, (64, 64))


@pytest.mark.parametrize("position, pad", [("center", (0, 0, 16, 16)), ("left", (0, 0, 0, 32)), ("right", (0, 0, 32, 0)),
                                           ("top", (0, 0, 16, 16))])
def test_pad_places_a_tall_frame(rs, position, pad):
    assert rs.plan(96, 48, 64, 64, "pillarbox_blur", position, 2, "bilinear").pad == pad


def test_pad_grows_to_the_next_multiple(rs):
    # 100x48 into 64x64: 64x31 (round(30.72)), pad 16 / 17; the frame floors to 64x24 and the
    # bottom pad grows from 17 to 24 so the output is 64x64 again.
    p = rs.plan(48, 100, 64, 64, "pad_edge", "center", 8, "bilinear")
    assert (p.size, p.pad, p.output) == ((24, 64), (16, 24, 0, 0), (64, 64))


def test_pad_mode_at_the_target_aspect_has_no_padding(rs):
    assert rs.plan(48, 96, 128, 64, "pad", "center", 2, "bilinear").pad is None


@pytest.mark.parametrize("position, crop", [("center", (0, 24, 48, 48)), ("left", (0, 0, 48, 48)), ("right", (0, 48, 48, 48)),
                                            ("top", (0, 24, 48, 48))])
def test_crop_window_at_the_target_aspect(rs, position, crop):
    p = rs.plan(48, 96, 32, 32, "crop", position, 2, "bilinear")
    assert (p.size, p.crop) == ((32, 32), crop)


def test_crop_of_a_tall_frame_rounds_the_height(rs):
    # 50x101 to 16:9 (32x18): height round(50 / (32 / 18)) = round(28.125) = 28, top at (101 - 28) // 2.
    assert rs.plan(101, 50, 32, 18, "crop", "center", 2, "bilinear").crop == (36, 0, 28, 50)


def test_crop_at_the_source_aspect_is_no_crop(rs):
    assert rs.plan(48, 96, 64, 32, "crop", "center", 2, "bilinear").crop is None


def test_rtx_vsr_rounds_to_a_multiple_of_8(rs):
    # round(100 / 8) = 12 (half to even), round(50 / 8) = 6.
    assert rs.plan(48, 96, 100, 50, "stretch", "center", 0, "nvidia_rtx_vsr").size == (48, 96)


def test_zero_size_is_refused(rs):
    with pytest.raises(ValueError, match="raise width / height"):
        rs.plan(48, 96, 0, 64, "total_pixels", "center", 2, "bilinear")


# --- resize_image -----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nearest-exact", "bilinear", "area", "bicubic"])
def test_stretch_equals_interpolate_on_the_batch(rs, method):
    image = torch.rand(3, 37, 53, 3)
    out, width, height, mask = run(rs, image, width=40, height=24, method=method)
    ref = F.interpolate(image.movedim(-1, 1), size=(24, 40), mode=method).movedim(1, -1)
    assert torch.equal(out, ref) and (width, height) == (40, 24)
    assert torch.equal(mask, torch.zeros(1, 64, 64))


def test_lanczos_equals_pil_on_8_bit_frames(rs):
    image = torch.rand(2, 37, 53, 3)
    out = run(rs, image, width=40, height=24, method="lanczos")[0]
    for frame, got in zip(image, out):
        u8 = Image.fromarray(np.clip(255.0 * frame.numpy(), 0, 255).astype(np.uint8)).resize((40, 24), Image.Resampling.LANCZOS)
        assert torch.equal(got, torch.from_numpy(np.array(u8).astype(np.float32) / 255.0))


def test_lanczos_off_the_cpu_is_refused(rs):
    with pytest.raises(ValueError, match="lanczos runs on the CPU"):
        rs.resize_image(torch.rand(1, 8, 8, 3), None, 4, 4, "lanczos", "stretch", "0, 0, 0", "center", 2, torch.device("cuda"))


def test_image_at_the_output_size_is_passed_through(rs):
    image, mask = torch.rand(2, 32, 48, 3), torch.rand(2, 32, 48)
    out, width, height, out_mask = run(rs, image, mask, width=48, height=32, method="lanczos", keep="crop")
    assert out is image and out_mask.data_ptr() == mask.data_ptr() and torch.equal(out_mask, mask) and (width, height) == (48, 32)


def test_crop_takes_the_window_then_resizes(rs):
    image = torch.rand(1, 48, 96, 3)
    out = run(rs, image, width=24, height=24, method="area", keep="crop", position="right")[0]
    ref = F.interpolate(image[:, :, 48:].movedim(-1, 1), size=(24, 24), mode="area").movedim(1, -1)
    assert torch.equal(out, ref)


def test_pad_color_around_the_resized_frame(rs):
    image = torch.rand(1, 16, 32, 3)
    out, _, _, mask = run(rs, image, width=32, height=32, keep="pad", color="255, 128, 0")
    assert out.shape == (1, 32, 32, 3)
    assert torch.equal(out[0, 8:24], image[0])
    assert torch.equal(out[0, :8], torch.tensor([1.0, 128 / 255, 0.0]).expand(8, 32, 3))
    # No mask given: the mask marks the padding.
    assert torch.equal(mask[0, :8], torch.ones(8, 32)) and torch.equal(mask[0, 8:24], torch.zeros(16, 32))


def test_pad_edge_fills_with_the_edge_means(rs):
    image = torch.tensor([[[[0.1] * 3, [0.3] * 3], [[0.5] * 3, [0.9] * 3]]])  # (1, 2, 2, 3)
    # 2x2 into 2x4: no resize, one row of padding above and below, the mean of the first / last row.
    out = run(rs, image, width=2, height=4, method="nearest-exact", keep="pad_edge", divisible_by=0)[0][0, :, :, 0]
    assert torch.allclose(out, torch.tensor([[0.2, 0.2], [0.1, 0.3], [0.5, 0.9], [0.7, 0.7]]))
    # 2x2 into 5x2: one column left, two right, grown to three by divisible_by 2; the mean of the
    # first / last column.
    out = run(rs, image, width=5, height=2, method="nearest-exact", keep="pad_edge", divisible_by=2)[0][0, :, :, 0]
    assert torch.allclose(out, torch.tensor([[0.3, 0.1, 0.3, 0.6, 0.6, 0.6], [0.3, 0.5, 0.9, 0.6, 0.6, 0.6]]))


def test_pad_edge_pixel_is_replicate_padding(rs):
    image, mask = torch.rand(2, 10, 20, 3), torch.rand(2, 10, 20)
    out, _, _, out_mask = run(rs, image, mask, width=20, height=17, method="nearest-exact", keep="pad_edge_pixel",
                              position="bottom", divisible_by=0)
    assert torch.equal(out, F.pad(image.movedim(-1, 1), (0, 0, 7, 0), mode="replicate").movedim(1, -1))
    assert torch.equal(out_mask, F.pad(mask, (0, 0, 7, 0), mode="replicate"))


def test_pad_mask_is_padded_with_its_edge_values(rs):
    mask = torch.rand(1, 16, 32)
    out_mask = run(rs, torch.rand(1, 16, 32, 3), mask, width=32, height=32, keep="pad")[3]
    assert torch.equal(out_mask, F.pad(mask, (0, 0, 8, 8), mode="replicate"))


def test_pillarbox_blur_background_is_dimmed_and_the_mask_is_1_around(rs):
    image, mask = torch.rand(1, 32, 16, 3), torch.zeros(1, 32, 16)
    out, _, _, out_mask = run(rs, image, mask, width=32, height=32, keep="pillarbox_blur")
    assert torch.equal(out[0, :, 8:24], image[0])
    assert out[0, :, :8].max() <= 0.35 and out[0, :, 24:].max() <= 0.35
    assert torch.equal(out_mask[0, :, :8], torch.ones(32, 8)) and torch.equal(out_mask[0, :, 8:24], torch.zeros(32, 16))


def test_mask_of_another_size_is_fitted_bilinear_then_follows_the_image(rs):
    image, mask = torch.rand(2, 24, 48, 3), torch.rand(2, 12, 24)
    out_mask = run(rs, image, mask, width=16, height=16, keep="crop", method="bilinear")[3]
    fitted = F.interpolate(mask[:, None], size=(24, 48), mode="bilinear")
    ref = F.interpolate(fitted[:, :, :, 12:36], size=(16, 16), mode="bilinear")[:, 0]
    assert torch.equal(out_mask, ref)


def test_lanczos_mask_equals_pil_on_8_bit(rs):
    mask = torch.rand(1, 37, 53)
    out_mask = run(rs, torch.rand(1, 37, 53, 3), mask, width=40, height=24, method="lanczos")[3]
    u8 = Image.fromarray(np.clip(255.0 * mask[0].numpy(), 0, 255).astype(np.uint8)).resize((40, 24), Image.Resampling.LANCZOS)
    assert torch.equal(out_mask[0], torch.from_numpy(np.array(u8).astype(np.float32) / 255.0))


def test_zero_64x64_mask_is_no_mask_but_a_real_one_is_resized(rs):
    image = torch.rand(1, 32, 32, 3)
    assert torch.equal(run(rs, image, torch.zeros(1, 64, 64), width=16, height=16)[3], torch.zeros(1, 64, 64))
    assert run(rs, image, torch.ones(1, 64, 64), width=16, height=16)[3].shape == (1, 16, 16)


@pytest.mark.parametrize("text, rgb", [("1, 0, 0", [255, 0, 0]), ("255, 128, 0", [255, 128, 0]), ("0.5", [127, 127, 127]),
                                       ("200", [200, 200, 200]), ("#ff8000", [255, 128, 0]), ("ff8000", [255, 128, 0]),
                                       ("#ff800080", [255, 128, 0, 128]), ("red", [255, 0, 0]), ("#f00", [255, 0, 0]),
                                       ("300, -5, 10", [255, 0, 10])])
def test_pad_color_forms(rs, text, rgb):
    assert rs.parse_pad_color(text) == rgb


@pytest.mark.parametrize("text", ["", "1, x, 0", "notacolour"])
def test_bad_pad_color_is_refused(rs, text):
    with pytest.raises(ValueError, match="pad_color"):
        rs.parse_pad_color(text)


def test_pad_color_needs_one_value_per_channel(rs):
    with pytest.raises(ValueError, match="has 3 values but the image has 4 channels"):
        run(rs, torch.rand(1, 16, 32, 4), width=32, height=32, keep="pad")


def test_bad_pad_color_is_ignored_without_color_padding(rs):
    assert run(rs, torch.rand(1, 16, 32, 3), width=32, height=32, keep="pad_edge", color="nope")[0].shape == (1, 32, 32, 3)

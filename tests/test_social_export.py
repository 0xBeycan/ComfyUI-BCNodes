"""
Tests for the Social Media Export planner and encoder.

Runnable with plain pytest, no ComfyUI or torch import:

    python -m pytest tests/test_social_export.py

Only ``nodes/social_export_core.py`` (Pillow-only) is exercised here. Specs are defined inline so the
tests do not depend on the (editable, changing) values in social_specs.json,
except for two integration tests that load the shipped file directly.
"""

import io
import os
import sys

import pytest
from PIL import Image, JpegImagePlugin

NODES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes")
sys.path.insert(0, NODES)

import social_export_core as core  # noqa: E402


# --- inline specs -------------------------------------------------------------

FEED = {
    "ar_min": 0.8, "ar_max": 1.91,
    "max_w": 1440, "max_h": 1800, "min_w": 1080,
    "format": "jpg",
}
# Single-point 9:16 band, like a story.
STORY = {
    "ar_min": 0.5625, "ar_max": 0.5625,
    "max_w": 1080, "max_h": 1920,
    "format": "jpg",
}


# --- helpers ------------------------------------------------------------------

def _noise(size):
    """Max-entropy (incompressible) RGB image."""
    return Image.frombytes("RGB", (size, size), os.urandom(size * size * 3))


def _gradient(size):
    """Smooth diagonal gradient — compresses well, so JPEG quality strongly
    affects file size (used for the byte-cap stepping test)."""
    data = bytes(((x + y) % 256) for y in range(size) for x in range(size))
    return Image.frombytes("L", (size, size), data).convert("RGB")


# --- 1. AR inside band -> exactly zero crop ----------------------------------

def test_ar_inside_band_zero_crop():
    (ox, oy, cw, ch), _ = core.plan(1200, 1200, FEED)   # ar 1.0, inside [0.8, 1.91]
    assert (cw, ch) == (1200, 1200)
    assert (ox, oy) == (0, 0)
    assert core.crop_fraction(1200, 1200, cw, ch) == 0.0


# --- 2. AR outside band -> band edge within 0.5%, one axis only ---------------

@pytest.mark.parametrize(
    "w,h,edge,axis",
    [
        (4000, 1000, 1.91, "x"),   # too wide -> trim width to ar_max
        (1000, 4000, 0.80, "y"),   # too tall -> trim height to ar_min
    ],
)
def test_ar_outside_band_edge_single_axis(w, h, edge, axis):
    (ox, oy, cw, ch), _ = core.plan(w, h, FEED, anchor=0.5)
    got = cw / ch
    assert abs(got - edge) / edge <= 0.005          # within 0.5% of the band edge
    if axis == "x":                                 # width trimmed only
        assert ch == h and oy == 0 and cw < w
    else:                                           # height trimmed only
        assert cw == w and ox == 0 and ch < h


# --- 3. anchor 0.0 / 0.5 / 1.0 -> top / center / bottom -----------------------

def test_anchor_top_center_bottom():
    w, h = 1000, 4000                               # too tall for FEED -> vertical crop
    (_, oy_top, _, ch), _ = core.plan(w, h, FEED, anchor=0.0)
    (_, oy_mid, _, _), _ = core.plan(w, h, FEED, anchor=0.5)
    (_, oy_bot, _, _), _ = core.plan(w, h, FEED, anchor=1.0)
    assert oy_top == 0                              # flush to top
    assert oy_bot == h - ch                         # flush to bottom
    assert oy_top < oy_mid < oy_bot                 # strictly increasing


# --- 4. allow_upscale=False never enlarges (and =True can) --------------------

@pytest.mark.parametrize("w,h", [(100, 100), (5000, 3000), (300, 1200), (1920, 1080)])
def test_no_upscale_never_larger_than_input(w, h):
    _, (ow, oh) = core.plan(w, h, FEED, allow_upscale=False)
    assert ow <= w and oh <= h


def test_allow_upscale_can_enlarge_to_envelope():
    _, (ow, oh) = core.plan(200, 200, FEED, allow_upscale=True)
    assert ow > 200 and oh > 200                    # scaled up toward the envelope


# --- 5. crop budget: error / pad / crop --------------------------------------

def test_overflow_error_raises_with_context():
    img = Image.new("RGB", (3840, 2160))
    with pytest.raises(core.OverflowCropError) as excinfo:
        core.render(img, STORY, overflow_strategy="error",
                    max_crop_ratio=0.15, platform="instagram_story")
    msg = str(excinfo.value)
    assert "instagram_story" in msg
    assert "3840x2160" in msg
    assert "%" in msg                               # names the crop percentage


def test_overflow_pad_letterboxes_without_cropping():
    img = Image.new("RGB", (3840, 2160), (10, 120, 200))
    result, meta = core.render(img, STORY, overflow_strategy="pad", max_crop_ratio=0.15)
    assert meta["strategy"] == "pad"
    assert meta["crop"] == (3840, 2160)             # nothing cropped
    assert meta["crop_pct"] == 0.0
    ow, oh = result.size
    assert abs((ow / oh) - 0.5625) / 0.5625 <= 0.005
    assert ow <= 3840 and oh <= 2160                # no upscale
    assert result.getpixel((0, 0)) != (0, 0, 0)     # letterbox is not flat black


def test_overflow_crop_forced_proceeds():
    img = Image.new("RGB", (3840, 2160))
    result, meta = core.render(img, STORY, overflow_strategy="crop", max_crop_ratio=0.15)
    assert meta["strategy"] == "crop!"
    assert meta["over"] is True
    ow, oh = result.size
    assert abs((ow / oh) - 0.5625) / 0.5625 <= 0.005
    assert meta["crop"][0] < 3840                   # width was trimmed


def test_below_budget_is_plain_crop_regardless_of_strategy():
    # 1:1 into a 4:5..1.91 band needs no crop; a mild landscape needs a small crop.
    img = Image.new("RGB", (2000, 1000))            # ar 2.0 -> tiny trim to 1.91
    for strat in ("error", "pad", "crop"):
        _, meta = core.render(img, FEED, overflow_strategy=strat, max_crop_ratio=0.15)
        assert meta["strategy"] == "crop"           # never pads/errors below threshold
        assert meta["crop_pct"] < 15.0


# --- 6. encode: 4:4:4 subsampling + embedded sRGB ICC ------------------------

def test_encode_jpeg_is_444_with_icc():
    data, q, ext, exceeded = core.encode(_noise(64), {"format": "jpg"},
                                          quality=95, chroma_444=True)
    assert ext == "jpg" and exceeded is False
    im = Image.open(io.BytesIO(data))
    im.load()
    assert JpegImagePlugin.get_sampling(im) == 0    # 4:4:4
    assert im.info.get("icc_profile")               # sRGB profile embedded
    assert "exif" not in im.info                     # provenance metadata stripped


def test_encode_jpeg_subsamples_when_not_444():
    data, _, _, _ = core.encode(_noise(64), {"format": "jpg"},
                                quality=95, chroma_444=False)
    im = Image.open(io.BytesIO(data))
    im.load()
    assert JpegImagePlugin.get_sampling(im) != 0    # 4:2:0 (Pillow default)


# --- 7. max_bytes stepping loop: terminates, respects floor ------------------

def test_max_bytes_generous_cap_keeps_quality():
    data, q, _, exceeded = core.encode(_gradient(256),
                                       {"format": "jpg", "max_bytes": 50 * 1024 * 1024},
                                       quality=95)
    assert q == 95 and exceeded is False


def test_max_bytes_impossible_cap_floors_at_70():
    data, q, _, exceeded = core.encode(_noise(384),
                                       {"format": "jpg", "max_bytes": 10},
                                       quality=95)
    assert q == 70                                  # floored (never below)
    assert exceeded is True                         # honestly reported, not silent


def test_max_bytes_steps_down_to_fit():
    img = _gradient(512)
    big, q95, _, _ = core.encode(img, {"format": "jpg"}, quality=95)
    small, q70, _, _ = core.encode(img, {"format": "jpg"}, quality=70)
    assert q95 == 95 and q70 == 70
    assert len(small) < len(big)                    # lower quality -> fewer bytes
    cap = (len(big) + len(small)) // 2
    data, q, _, exceeded = core.encode(img, {"format": "jpg", "max_bytes": cap}, quality=95)
    assert 70 <= q < 95                             # stepped down at least once
    assert not exceeded and len(data) <= cap        # and it fit


# --- 8. specs file: loads, filters metadata keys, all platforms plan ---------

def test_real_specs_load_and_plan_all():
    specs = core.load_specs(os.path.join(NODES, "social_specs.json"))
    names = core.platform_names(specs)
    assert "instagram_feed" in names
    assert "_comment" not in names and "_schema" not in names   # '_' keys filtered
    for name in names:
        spec = core.get_spec(specs, name)
        (_, _, cw, ch), (ow, oh) = core.plan(1920, 1080, spec)
        assert cw > 0 and ch > 0 and ow > 0 and oh > 0


def test_unknown_platform_lists_valid_keys():
    specs = core.load_specs(os.path.join(NODES, "social_specs.json"))
    with pytest.raises(core.UnknownPlatformError) as excinfo:
        core.get_spec(specs, "myspace")
    assert "instagram_feed" in str(excinfo.value)


def test_report_line_is_readable():
    img = Image.new("RGB", (2000, 2000))
    _, meta = core.render(img, FEED, overflow_strategy="crop", platform="instagram_feed")
    line = core.format_report_line(0, meta, 92, 123456, "jpg", False, None)
    assert "instagram_feed" in line
    assert "->" in line and "q92" in line

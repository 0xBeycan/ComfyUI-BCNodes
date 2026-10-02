"""Emulate's profiles of Frequency Merge and SeedVR2 Framing Downscale (pipelines/process_monitor/
profiles.py), worked out by hand from each node's code; the graph helpers are the emulate test's.

  - Frequency Merge: the output at base's size, float16 only when both inputs are; on the CPU the
    per-image float32 working set (detail's high-pass, gauss_reflect's row-padded copy, row sum and
    column-padded copy, a float32 copy of each input that is not float32); on the gpu choice none in
    RAM; inputs of another size, a split_sigma the image cannot hold or one from a link: not counted;
  - SeedVR2 Framing Downscale: no tensor output (two numbers); per group of up to four frames, core
    SAM3_Detect's 1008 x 1008 copy in the image's dtype and its float32 masks, listed and stacked.
"""

import pytest

import test_pipe_process_monitor_emulate as emulate
from test_pipe_process_monitor_emulate import N, loader, rows_of


class Env(emulate.Env):
    TYPES = {**emulate.Env.TYPES, "BC_FrequencyMerge": ["IMAGE"], "BC_SeedVR2FramingDownscale": ["FLOAT", "FLOAT"]}


ENV_VIDEOS = {"v.mp4": emulate.PORTRAIT}
SMALL = N("BC_ImageResize", image=["9", 0], width=64, height=48, upscale_method="bilinear", keep_proportion="stretch",
          pad_color="0, 0, 0", crop_position="center", divisible_by=2)  # 64x48


@pytest.fixture
def em(bcnodes):
    return bcnodes["pipelines.process_monitor.emulate"]


def rows(em, prompt):
    return rows_of(em, prompt, Env(videos=ENV_VIDEOS))


def merge(base, detail, **widgets):
    return N("BC_FrequencyMerge", base=base, detail=detail, **dict(dict(split_sigma=3.0, detail_strength=1.0), **widgets))


def reference(frames):
    """Five 1280x720 float32 frames (h x w) through SeedVR2 Resize at 1x: its float16 reference."""
    return {"1": loader(frame_count=str(frames), precision="fp32"),
            "2": N("BC_SeedVR2Resize", image=["1", 0], upscale_factor=1.0, downscale_factor=1.0, max_resolution=0, emulate_bf16=False)}


def test_frequency_merge_float32_on_the_cpu(em):
    one = 480 * 640 * 3 * 4
    row = rows(em, {"9": N("LoadImage", image="ref.png"), "1": merge(["9", 0], ["9", 0])})["1"]
    assert row["status"] == "counted" and row["outputs"][0]["shape"] == [1, 480, 640, 3] and row["output_bytes"] == one
    # sigma 3: a 9 px reflect pad; detail's high-pass + the row sum, the row-padded and the column-padded copy
    assert row["transient"] == 2 * one + 480 * (640 + 18) * 3 * 4 + (480 + 18) * 640 * 3 * 4
    row = rows(em, {"9": N("LoadImage", image="ref.png"), "1": merge(["9", 0], ["9", 0], split_sigma=32.0)})["1"]
    assert row["transient"] == 2 * one + 480 * (640 + 192) * 3 * 4 + (480 + 192) * 640 * 3 * 4
    row = rows(em, {"9": N("LoadImage", image="ref.png"), "1": merge(["9", 0], ["9", 0], device="gpu")})["1"]
    assert row["output_bytes"] == one and row["transient"] == 0 and "device not counted" in row["note"]


def test_frequency_merge_dtypes(em):
    one = 1280 * 720 * 3 * 4
    pads = 1280 * (720 + 18) * 3 * 4 + (1280 + 18) * 720 * 3 * 4
    p = {**reference(5), "3": merge(["2", 1], ["2", 1])}
    row = rows(em, p)["3"]
    # both float16: a float16 output, each input copied to float32 per image
    assert row["outputs"][0]["shape"] == [5, 1280, 720, 3] and row["output_bytes"] == 5 * 1280 * 720 * 3 * 2
    assert row["transient"] == 2 * one + pads + 2 * one
    p["3"] = merge(["2", 1], ["1", 0])  # float16 base, float32 detail: a float32 output, base copied
    row = rows(em, p)["3"]
    assert row["output_bytes"] == 5 * 1280 * 720 * 3 * 4 and row["transient"] == 2 * one + pads + one


@pytest.mark.parametrize("prompt, why", [
    ({"9": N("LoadImage", image="ref.png"), "1": loader(frame_count="5"), "2": merge(["9", 0], ["1", 0])}, "differ in size"),
    ({"9": N("LoadImage", image="ref.png"), "1": SMALL, "2": merge(["1", 0], ["1", 0], split_sigma=16.0)},  # 48 px pad, 48 rows
     "more than the image holds"),
    ({"9": N("LoadImage", image="ref.png"), "2": merge(["9", 0], ["9", 0], split_sigma=["9", 1])}, "comes from a link"),
])
def test_frequency_merge_not_counted(em, prompt, why):
    row = rows(em, prompt)["2"]
    assert row["status"] == "not counted" and why in row["note"]


def test_seedvr2_framing_downscale(em):
    sam_input = 3 * 1008 * 1008
    row = rows(em, {"9": N("LoadImage", image="ref.png"), "1": N("BC_SeedVR2FramingDownscale", image=["9", 0])})["1"]
    assert row["status"] == "counted" and row["outputs"] == [] and row["output_bytes"] == 0
    assert row["transient"] == sam_input * 4 + 2 * 480 * 640 * 4 and "SAM 3's working set on its device not counted" in row["note"]
    # 81 frames: four at a time
    row = rows(em, {"1": loader(frame_count="81", precision="fp32"), "2": N("BC_SeedVR2FramingDownscale", image=["1", 0])})["2"]
    assert row["transient"] == 4 * sam_input * 4 + 2 * 4 * 1280 * 720 * 4
    # three frames: one group of three
    three = N("BCVLoadVideo", video="v.mp4", model="None", resolution="720p", orientation="auto", force_fps="", start_frame=1,
              frame_count="3", precision="fp32")
    row = rows(em, {"1": three, "2": N("BC_SeedVR2FramingDownscale", image=["1", 0])})["2"]
    assert row["transient"] == 3 * sam_input * 4 + 2 * 3 * 1280 * 720 * 4
    # a float16 batch is scaled to 1008 x 1008 in float16; the masks are float32
    row = rows(em, {**reference(5), "3": N("BC_SeedVR2FramingDownscale", image=["2", 1])})["3"]
    assert row["transient"] == 4 * sam_input * 2 + 2 * 4 * 1280 * 720 * 4

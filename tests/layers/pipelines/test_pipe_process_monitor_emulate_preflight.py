"""Emulate's profiles of the PreFlight nodes (pipelines/process_monitor/profiles.py); the graph helpers are the
emulate test's.

  - PreFlight Observe: not counted. It runs a fixed catalog model (Qwen3.5-9B INT8 ConvRot) that no widget names
    as a file, so Emulate reads no weights for it, and nothing in the prompt sizes the KV cache, the sampled
    frames or their resized copies; its two string outputs alone would read as "counted", 0 bytes. The image
    it reads keeps its own estimate;
  - PreFlight Report: counted; its image output is the image input passed on (shared, no new bytes), absent
    when nothing feeds it; an image input without an estimate leaves it not counted;
  - Outcome and Calibrate output text only: counted without a profile, no tensor.
"""

import pytest

import test_pipe_process_monitor_emulate as emulate
from test_pipe_process_monitor_emulate import N, rows_of


class Env(emulate.Env):
    TYPES = {**emulate.Env.TYPES, "BC_PreFlightObserve": ["STRING", "STRING"],
             "BC_PreFlightReport": ["STRING", "STRING", "STRING", "IMAGE"], "BC_PreFlightOutcome": ["STRING"],
             "BC_PreFlightCalibrate": ["STRING"], "PreviewAny": []}


@pytest.fixture
def em(bcnodes):
    return bcnodes["pipelines.process_monitor.emulate"]


def graph(image=True):
    report = dict(observations_json=["2", 0], caption="", log_prediction=True)
    if image:
        report["image"] = ["9", 0]
    return {"9": N("LoadImage", image="ref.png"),
            "2": N("BC_PreFlightObserve", image=["9", 0], max_frames=6, keep_model_loaded=True),
            "3": N("BC_PreFlightReport", **report),
            "4": N("PreviewImage", images=["3", 3]),
            "5": N("BC_PreFlightOutcome", record="no records yet", platform="instagram", result="clean",
                   record_id_override=""),
            "6": N("BC_PreFlightCalibrate"),
            "7": N("PreviewAny", source=["6", 0])}


def test_observe_is_not_counted_and_report_passes_the_image_on(em):
    r = em.estimate(graph(), Env())
    rows = {row["id"]: row for row in r["rows"]}
    observe = rows["2"]
    assert observe["status"] == "not counted" and observe["outputs"] == [] and observe["weights"] == []
    assert observe["note"].startswith("runs Qwen3.5-9B INT8 ConvRot, which no widget names as a file")
    for part in ("model file's weights", "KV cache core reserves for prompt + 300 tokens", "sampled frames",
                 "images resized"):
        assert part in observe["note"], part
    image = 480 * 640 * 3 * 4
    assert rows["9"]["output_bytes"] == image + 480 * 640 * 4
    report = rows["3"]
    assert report["status"] == "counted" and report["output_bytes"] == 0 and report["transient"] == 0
    assert report["outputs"] == [{"slot": 3, "type": "IMAGE", "shape": [1, 480, 640, 3], "bytes": image, "shared": True}]
    assert rows["4"]["status"] == "counted"  # the passed-on image reaches Preview Image with its estimate
    for nid in ("5", "6"):
        assert (rows[nid]["status"], rows[nid]["outputs"], rows[nid]["output_bytes"]) == ("counted", [], 0)
    assert r["not_counted"] == ["2"]


def test_report_without_an_image(em):
    p = graph(image=False)
    del p["4"]
    row = rows_of(em, p, Env())["3"]
    assert (row["status"], row["outputs"], row["output_bytes"]) == ("counted", [], 0)
    assert row["note"] == "text outputs only: no image to pass on"


def test_report_with_an_image_without_an_estimate(em):
    class Unknown(Env):
        TYPES = {**Env.TYPES, "SomeUnknownNode": ["IMAGE"]}

    p = graph()
    p["9"] = N("SomeUnknownNode")
    row = rows_of(em, p, Unknown())["3"]
    assert row["status"] == "not counted" and "image" in row["note"]

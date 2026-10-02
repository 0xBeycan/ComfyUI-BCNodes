"""SeedVR2 Framing Downscale, the surface: its widgets in order with their defaults and ranges (the
factors on SeedVR2 Resize's downscale_factor range), two FLOAT outputs, the SeedVR2 category, the
default SAM 3 checkpoint offered with none on disk, and each widget handed to the flow under its own
name (a stand-in detector, so a swapped argument shows)."""

import pytest
import torch


@pytest.fixture
def node(bcnodes):
    return bcnodes["seedvr2"].SeedVR2FramingDownscale


def test_surface(bcnodes, node):
    mod = bcnodes["seedvr2"]
    inputs = node.INPUT_TYPES()
    required = inputs["required"]
    assert set(inputs) == {"required"}
    assert list(required) == ["image", "sam3_model", "close_up_min_face", "close_up_factor", "medium_min_face", "medium_factor",
                              "far_factor", "no_face_factor", "detection_threshold"]
    assert required["image"][0] == "IMAGE"
    assert required["sam3_model"][1]["default"] in required["sam3_model"][0]
    defaults = {name: spec[1]["default"] for name, spec in required.items() if spec[0] == "FLOAT"}
    assert defaults == dict(close_up_min_face=0.3, close_up_factor=0.5, medium_min_face=0.18, medium_factor=0.75, far_factor=1.0,
                            no_face_factor=1.0, detection_threshold=0.5)
    resize = mod.SeedVR2Resize.INPUT_TYPES()["required"]["downscale_factor"][1]
    for name in ("close_up_factor", "medium_factor", "far_factor", "no_face_factor"):
        assert (required[name][1]["min"], required[name][1]["max"]) == (resize["min"], resize["max"])
    assert "Uncalibrated" in required["close_up_min_face"][1]["tooltip"] and "Uncalibrated" in required["medium_min_face"][1]["tooltip"]
    assert "tallest face" in required["image"][1]["tooltip"]
    assert (node.RETURN_TYPES, node.RETURN_NAMES, node.CATEGORY) == (("FLOAT", "FLOAT"), ("downscale_factor", "face_fraction"),
                                                                    "BCNodes/seedvr2")
    assert not getattr(node, "OUTPUT_NODE", False) and not hasattr(node, "HEAVY_OUTPUTS")
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["BC_SeedVR2FramingDownscale"] == "SeedVR2 Framing Downscale"


def test_widgets_reach_the_flow_by_name(bcnodes, node, monkeypatch):
    framing = bcnodes["pipelines.seedvr2.framing"]
    seen = {}

    def load(name):
        seen["model"] = name
        return None, None

    def detect(model, clip, image, text, threshold, refine_iterations=2):
        seen["threshold"] = threshold
        return None, [[dict(x=0.0, y=0.0, width=10.0, height=30.0, score=0.9)]] * image.shape[0]  # 0.3 of 100 rows

    monkeypatch.setattr(framing, "load", load)
    monkeypatch.setattr(framing, "detect", detect)
    widgets = dict(sam3_model="a.safetensors", close_up_min_face=0.35, close_up_factor=0.4, medium_min_face=0.2, medium_factor=0.7,
                   far_factor=0.95, no_face_factor=0.85, detection_threshold=0.3)
    assert node().choose(torch.zeros(1, 100, 80, 3), **widgets) == (0.7, pytest.approx(0.3))
    assert seen == dict(model="a.safetensors", threshold=0.3)
    assert node().choose(torch.zeros(1, 100, 80, 3), **dict(widgets, close_up_min_face=0.3)) == (0.4, pytest.approx(0.3))
    assert node().choose(torch.zeros(1, 100, 80, 3), **dict(widgets, medium_min_face=0.31, close_up_min_face=0.4)) == \
        (0.95, pytest.approx(0.3))

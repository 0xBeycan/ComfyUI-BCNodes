"""Detailer and Face Detailer, the surface: their inputs in order with defaults and ranges, the two
outputs, the detailer category, the default SAM 3 checkpoint offered with none on disk; each widget
handed to its flow under its own name (stand-in flows, so a swapped argument shows); an empty image,
and the Detailer without a mask, returned unchanged without sampling."""

import pytest
import torch

SAMPLING = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "refine_denoise"]


@pytest.fixture
def mod(bcnodes):
    return bcnodes["detailer"]


def test_surface(bcnodes, mod):
    common = bcnodes["nodes.common"]
    detailer = mod.Detailer.INPUT_TYPES()
    assert list(detailer["required"]) == ["image", "mask", "model", "vae", "positive", "negative", *SAMPLING, "region_size", "padding"]
    assert set(detailer) == {"required", "hidden"} and detailer["hidden"] == common.LINK_INPUTS
    face = mod.FaceDetailer.INPUT_TYPES()
    assert list(face["required"]) == ["image", "model", "vae", "positive", "negative", *SAMPLING, "face_size", "padding", "sam3_model",
                                      "threshold", "min_face_height", "face", "mouth", "eyes"]
    assert list(face["optional"]) == ["model_no_lora", "mask"] and face["hidden"] == common.LINK_INPUTS
    assert [face["optional"][k][0] for k in ("model_no_lora", "mask")] == ["MODEL", "MASK"]
    required = face["required"]
    defaults = {name: spec[1]["default"] for name, spec in required.items() if len(spec) > 1 and "default" in spec[1]}
    assert defaults == dict(seed=0, steps=20, cfg=8.0, denoise=0.5, refine_denoise=0.2, face_size=640, padding=0.5,
                            sam3_model=bcnodes["models.sam3.checkpoint"].DEFAULT_SAM3, threshold=0.5, min_face_height=64, face=True,
                            mouth=False, eyes=True)
    ranges = {name: tuple(spec[1][k] for k in ("default", "min", "max", "step")) for name, spec in required.items()
              if name in ("face_size", "padding")}
    assert ranges == {"face_size": (640, 64, 8192, 64), "padding": (0.5, 0.0, 4.5, 0.05)}
    # the Detailer's pair: region_size as face_size, the same padding; each with a tooltip of its own
    for ours, theirs in (("region_size", "face_size"), ("padding", "padding")):
        a, b = detailer["required"][ours], required[theirs]
        assert a[0] == b[0] and {k: v for k, v in a[1].items() if k != "tooltip"} == {k: v for k, v in b[1].items() if k != "tooltip"}
        assert a[1]["tooltip"] != b[1]["tooltip"]
    assert required["sam3_model"][1]["default"] in required["sam3_model"][0]
    assert required["sampler_name"][0] == ["euler"] and required["scheduler"][0] == ["simple"]  # comfy.samplers' lists
    for spec in SAMPLING:
        assert detailer["required"][spec] == required[spec]
    for cls in (mod.Detailer, mod.FaceDetailer):
        assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.CATEGORY) == (("IMAGE", "MASK"), ("image", "mask"), "BCNodes/detailer")
        assert not getattr(cls, "OUTPUT_NODE", False)
    assert mod.NODE_DISPLAY_NAME_MAPPINGS == {"BC_Detailer": "Detailer", "BC_FaceDetailer": "Face Detailer"}


WIDGETS = dict(seed=11, steps=7, cfg=3.5, sampler_name="euler", scheduler="simple", denoise=0.6, refine_denoise=0.15, padding=0.75)


def test_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core = bcnodes["pipelines.detailer.core"]
    seen = {}

    def run(image, masks, model, s, want_image=True):
        seen.update(masks=masks, model=model, s=s, want_image=want_image)
        return image, masks

    monkeypatch.setattr(core, "run", run)
    image, mask = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.Detailer().detail(image, mask, "m", "v", ["p"], ["n"], **WIDGETS, region_size=1280)
    assert seen["masks"] is mask and seen["model"] == "m" and seen["want_image"] is True
    assert seen["s"] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 0.15, 1280, 0.75)


def test_face_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core, flow = bcnodes["pipelines.detailer.core"], bcnodes["pipelines.detailer.face"]
    seen = []
    monkeypatch.setattr(flow, "run", lambda *args: seen.append(args) or (args[0], torch.zeros(1, 8, 8)))
    image, select = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.FaceDetailer().detail(image, "m", "v", ["p"], ["n"], **WIDGETS, face_size=1280, sam3_model="s.safetensors", threshold=0.3,
                              min_face_height=40, face=False, mouth=True, eyes=False, model_no_lora="o", mask=select)
    (args,) = seen
    assert args[0] is image and args[1:3] == ("m", "o") and args[4:6] == ("s.safetensors", 0.3) and args[6] is select
    assert args[7:] == (40, False, True, False)
    assert args[3] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 0.15, 1280, 0.75)


def test_empty_image_and_no_mask(mod):
    for out in (mod.Detailer().detail(None, None, None, None, [], [], **WIDGETS, region_size=640),
                mod.FaceDetailer().detail(torch.zeros(0, 8, 8, 3), None, None, [], [], **WIDGETS, face_size=640, sam3_model="s",
                                          threshold=0.5, min_face_height=64, face=True, mouth=False, eyes=True)):
        assert out[0].shape == (0, 64, 64, 3) and out[1].shape == (0, 64, 64)
    image = torch.rand(2, 16, 16, 3)
    out, changed = mod.Detailer().detail(image, None, None, None, [], [], **WIDGETS, region_size=640)
    assert torch.equal(out, image) and out is not image and changed.shape == (2, 16, 16) and not bool(changed.any())

"""Detailer and Face Detailer, the surface: their inputs in order with defaults and ranges, the two
outputs, the detailer category, the default SAM 3 checkpoint offered with none on disk; each widget
handed to its flow under its own name (stand-in flows, so a swapped argument shows); an empty image,
and the Detailer without a mask, returned unchanged without sampling."""

import pytest
import torch

SAMPLING = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "refine_denoise", "resolution", "context"]


@pytest.fixture
def mod(bcnodes):
    return bcnodes["detailer"]


def test_surface(bcnodes, mod):
    common = bcnodes["nodes.common"]
    detailer = mod.Detailer.INPUT_TYPES()
    assert list(detailer["required"]) == ["image", "mask", "model", "vae", "positive", "negative", *SAMPLING]
    assert set(detailer) == {"required", "hidden"} and detailer["hidden"] == common.LINK_INPUTS
    face = mod.FaceDetailer.INPUT_TYPES()
    assert list(face["required"]) == ["image", "model", "vae", "positive", "negative", *SAMPLING, "sam3_model", "threshold",
                                      "min_face_height", "max_face_height", "face", "mouth", "eyes"]
    assert list(face["optional"]) == ["model_no_lora", "mask"] and face["hidden"] == common.LINK_INPUTS
    assert [face["optional"][k][0] for k in ("model_no_lora", "mask")] == ["MODEL", "MASK"]
    required = face["required"]
    defaults = {name: spec[1]["default"] for name, spec in required.items() if len(spec) > 1 and "default" in spec[1]}
    assert defaults == dict(seed=0, steps=20, cfg=8.0, denoise=0.5, refine_denoise=0.2, resolution=1024, context=3.0,
                            sam3_model=bcnodes["models.sam3.checkpoint"].DEFAULT_SAM3, threshold=0.5, min_face_height=64,
                            max_face_height=16384, face=True, mouth=False, eyes=True)
    assert (required["resolution"][1]["min"], required["resolution"][1]["max"], required["resolution"][1]["step"]) == (256, 8192, 64)
    assert (required["context"][1]["min"], required["context"][1]["max"]) == (1.0, 10.0)
    assert required["sam3_model"][1]["default"] in required["sam3_model"][0]
    assert required["sampler_name"][0] == ["euler"] and required["scheduler"][0] == ["simple"]  # comfy.samplers' lists
    for spec in SAMPLING:
        assert detailer["required"][spec] == required[spec]
    for cls in (mod.Detailer, mod.FaceDetailer):
        assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.CATEGORY) == (("IMAGE", "MASK"), ("image", "mask"), "BCNodes/detailer")
        assert not getattr(cls, "OUTPUT_NODE", False)
    assert mod.NODE_DISPLAY_NAME_MAPPINGS == {"BC_Detailer": "Detailer", "BC_FaceDetailer": "Face Detailer"}


WIDGETS = dict(seed=11, steps=7, cfg=3.5, sampler_name="euler", scheduler="simple", denoise=0.6, refine_denoise=0.15,
               resolution=1280, context=2.5)


def test_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core = bcnodes["pipelines.detailer.core"]
    seen = {}

    def run(image, masks, model, s, want_image=True):
        seen.update(masks=masks, model=model, s=s, want_image=want_image)
        return image, masks

    monkeypatch.setattr(core, "run", run)
    image, mask = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.Detailer().detail(image, mask, "m", "v", ["p"], ["n"], **WIDGETS)
    assert seen["masks"] is mask and seen["model"] == "m" and seen["want_image"] is True
    assert seen["s"] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 0.15, 1280, 2.5)


def test_face_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core, flow = bcnodes["pipelines.detailer.core"], bcnodes["pipelines.detailer.face"]
    seen = []
    monkeypatch.setattr(flow, "run", lambda *args: seen.append(args) or (args[0], torch.zeros(1, 8, 8)))
    image, select = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.FaceDetailer().detail(image, "m", "v", ["p"], ["n"], **WIDGETS, sam3_model="s.safetensors", threshold=0.3,
                              min_face_height=40, max_face_height=900, face=False, mouth=True, eyes=False,
                              model_no_lora="o", mask=select)
    (args,) = seen
    assert args[0] is image and args[1:3] == ("m", "o") and args[4:6] == ("s.safetensors", 0.3) and args[6] is select
    assert args[7:] == (40, 900, False, True, False)
    assert args[3] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 0.15, 1280, 2.5)


def test_empty_image_and_no_mask(mod):
    for out in (mod.Detailer().detail(None, None, None, None, [], [], **WIDGETS),
                mod.FaceDetailer().detail(torch.zeros(0, 8, 8, 3), None, None, [], [], **WIDGETS, sam3_model="s", threshold=0.5,
                                          min_face_height=64, max_face_height=16384, face=True, mouth=False, eyes=True)):
        assert out[0].shape == (0, 64, 64, 3) and out[1].shape == (0, 64, 64)
    image = torch.rand(2, 16, 16, 3)
    out, changed = mod.Detailer().detail(image, None, None, None, [], [], **WIDGETS)
    assert torch.equal(out, image) and out is not image and changed.shape == (2, 16, 16) and not bool(changed.any())

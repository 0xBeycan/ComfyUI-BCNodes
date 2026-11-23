"""Detailer and Face Detailer, the surface: their inputs in order with defaults and ranges, the two
outputs, the detailer category, the default SAM 3 checkpoint offered with none on disk; each widget
handed to its flow under its own name (stand-in flows, so a swapped argument shows); an empty image,
and the Detailer without a mask, returned unchanged without sampling."""

import pytest
import torch

SAMPLING = ["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"]


@pytest.fixture
def mod(bcnodes):
    return bcnodes["detailer"]


def test_surface(bcnodes, mod):
    common = bcnodes["nodes.common"]
    detailer = mod.Detailer.INPUT_TYPES()
    assert list(detailer["required"]) == ["image", "mask", "model", "vae", "positive", "negative", *SAMPLING, "region_size",
                                          "region_crop_size", "max_region_size"]
    assert set(detailer) == {"required", "hidden"} and detailer["hidden"] == common.LINK_INPUTS
    face = mod.FaceDetailer.INPUT_TYPES()
    assert list(face["required"]) == ["image", "model", "vae", "positive", "negative", *SAMPLING, "face_size", "face_crop_size",
                                      "max_face_height", "eye_size", "eye_crop_size", "eye_denoise", "sam3_model", "threshold", "face",
                                      "eyes"]
    assert list(face["optional"]) == ["model_no_lora", "mask"] and face["hidden"] == common.LINK_INPUTS
    assert [face["optional"][k][0] for k in ("model_no_lora", "mask")] == ["MODEL", "MASK"]
    required = face["required"]
    defaults = {name: spec[1]["default"] for name, spec in required.items() if len(spec) > 1 and "default" in spec[1]}
    assert defaults == dict(seed=0, steps=20, cfg=8.0, denoise=0.5, face_size=640, face_crop_size=1280, max_face_height=384, eye_size=320,
                            eye_crop_size=640, eye_denoise=0.35, sam3_model=bcnodes["models.sam3.checkpoint"].DEFAULT_SAM3,
                            threshold=0.5, face=True, eyes=True)
    ranges = {name: tuple(spec[1][k] for k in ("default", "min", "max", "step")) for name, spec in required.items()
              if name in ("face_size", "face_crop_size", "max_face_height", "eye_size", "eye_crop_size", "eye_denoise")}
    assert ranges == {"face_size": (640, 64, 8192, 64), "face_crop_size": (1280, 256, 8192, 64), "max_face_height": (384, 64, 16384, 1),
                      "eye_size": (320, 64, 8192, 64), "eye_crop_size": (640, 256, 8192, 64), "eye_denoise": (0.35, 0.0, 1.0, 0.01)}
    # the Detailer's three: as the face's; each with a tooltip of its own
    for ours, theirs in (("region_size", "face_size"), ("region_crop_size", "face_crop_size"), ("max_region_size", "max_face_height")):
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


WIDGETS = dict(seed=11, steps=7, cfg=3.5, sampler_name="euler", scheduler="simple", denoise=0.6)


def test_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core = bcnodes["pipelines.detailer.core"]
    seen = {}

    def run(image, masks, model, s, max_size, want_image=True):
        seen.update(masks=masks, model=model, s=s, max_size=max_size, want_image=want_image)
        return image, masks

    monkeypatch.setattr(core, "run", run)
    image, mask = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.Detailer().detail(image, mask, "m", "v", ["p"], ["n"], **WIDGETS, region_size=1280, region_crop_size=1920, max_region_size=500)
    assert seen["masks"] is mask and seen["model"] == "m" and seen["want_image"] is True and seen["max_size"] == 500
    assert seen["s"] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 1280, 1920)


def test_face_detailer_widgets_reach_the_flow_by_name(bcnodes, mod, monkeypatch):
    core, flow = bcnodes["pipelines.detailer.core"], bcnodes["pipelines.detailer.face"]
    seen = []
    monkeypatch.setattr(flow, "run", lambda *args: seen.append(args) or (args[0], torch.zeros(1, 8, 8)))
    image, select = torch.rand(1, 8, 8, 3), torch.ones(1, 8, 8)
    mod.FaceDetailer().detail(image, "m", "v", ["p"], ["n"], **WIDGETS, face_size=1280, face_crop_size=1920, max_face_height=300,
                              eye_size=384, eye_crop_size=768, eye_denoise=0.3, sam3_model="s.safetensors", threshold=0.3, face=False,
                              eyes=True, model_no_lora="o", mask=select)
    (args,) = seen
    assert args[0] is image and args[1:3] == ("m", "o") and args[5:7] == ("s.safetensors", 0.3) and args[7] is select
    assert args[8:] == (300, False, True)
    assert args[3] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.6, 1280, 1920)
    assert args[4] == core.Settings("v", ["p"], ["n"], 11, 7, 3.5, "euler", "simple", 0.3, 384, 768)  # eye_denoise


def test_empty_image_and_no_mask(mod):
    for out in (mod.Detailer().detail(None, None, None, None, [], [], **WIDGETS, region_size=640, region_crop_size=1280,
                                      max_region_size=384),
                mod.FaceDetailer().detail(torch.zeros(0, 8, 8, 3), None, None, [], [], **WIDGETS, face_size=640, face_crop_size=1280,
                                          max_face_height=384, eye_size=320, eye_crop_size=640, eye_denoise=0.35, sam3_model="s",
                                          threshold=0.5, face=True, eyes=True)):
        assert out[0].shape == (0, 64, 64, 3) and out[1].shape == (0, 64, 64)
    image = torch.rand(2, 16, 16, 3)
    out, changed = mod.Detailer().detail(image, None, None, None, [], [], **WIDGETS, region_size=640, region_crop_size=1280,
                                         max_region_size=384)
    assert torch.equal(out, image) and out is not image and changed.shape == (2, 16, 16) and not bool(changed.any())

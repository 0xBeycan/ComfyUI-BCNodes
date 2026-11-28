"""DyPE Patcher, DyPE Config and Dual Sampler (nodes/dype.py, nodes/dual_sampler.py), node side:

- the widgets, their order and defaults: Patcher mode auto, dype_start_sigma 1.0 in [0.001, 1]; Config's `model`
  (the dype family's names, the first by default, each one's automatic values under `automatic`), then
  models/common/dype.ConfigValues' keys in order, their defaults the first model's automatic values, then `edited`;
  Dual Sampler: Z-Image Turbo's values (9 steps, cfg 1.0, dpmpp_sde, beta both passes; denoise 0.45; bislerp x2.0;
  blend 0.5, lanczos; increment; dype auto);
- DyPE Config: the model always; nothing edited -> no values (every value automatic); only the edited and the linked
  fields leave it, typed; an unknown edited name is an error naming the fields;
- DyPE Patcher: no model -> None; otherwise models/common/dype.patch with the mode, start sigma and config;
- Dual Sampler: the pipeline gets the two passes (the first at denoise 1.0) and the settings as given, no DyPE Config
  as None; model_2 as the second pass's model, or model when model_2 is not linked.
"""

import sys
import types

import pytest


@pytest.fixture
def mod(bcnodes):
    return bcnodes["dype"]


@pytest.fixture
def samplers(monkeypatch, comfy_stubs):
    module = types.ModuleType("comfy.samplers")
    module.KSampler = types.SimpleNamespace(SAMPLERS=["euler", "dpmpp_sde"], SCHEDULERS=["simple", "beta"])
    monkeypatch.setitem(sys.modules, "comfy.samplers", module)
    monkeypatch.setattr(sys.modules["comfy"], "samplers", module, raising=False)


def test_patcher_widgets(mod):
    req = mod.DyPEPatcher.INPUT_TYPES()
    assert list(req["required"]) == ["model", "mode", "dype_start_sigma"] and list(req["optional"]) == ["dype_config"]
    assert req["required"]["mode"][0] == ["enable", "disable"] and req["required"]["mode"][1]["default"] == "enable"
    start = req["required"]["dype_start_sigma"][1]
    assert (start["default"], start["min"], start["max"]) == (1.0, 0.001, 1.0)
    assert req["optional"]["dype_config"][0] == "BC_DYPE_CONFIG"


def test_config_widgets(mod, bcnodes):
    req = mod.DyPEConfig.INPUT_TYPES()
    fields = list(bcnodes["models.common.dype"].ConfigValues.__annotations__)
    assert list(req["required"]) == ["model"] + fields + ["edited"] == [
        "model", "method", "base_resolution", "dype_scale", "dype_exponent", "yarn_alt_scaling", "edited"]
    names, options = req["required"]["model"]
    assert names == ["Z-Image", "Qwen-Image"] and options["default"] == "Z-Image"
    assert options["automatic"] == {
        "Z-Image": {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0},
        "Qwen-Image": {"method": "vision_yarn", "base_resolution": 1328, "dype_scale": 2.0, "yarn_alt_scaling": False}}
    assert req["required"]["method"][0] == ["vision_yarn", "yarn", "ntk", "pi", "base"]
    assert req["required"]["yarn_alt_scaling"][0] == "BOOLEAN"
    defaults = {name: req["required"][name][1]["default"] for name in fields}
    # the default model's, as shown (Z-Image has no yarn_alt_scaling: the widget's own default)
    assert defaults == {**options["automatic"]["Z-Image"], "dype_exponent": 1.0, "yarn_alt_scaling": False}
    assert req["hidden"] == {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}


def build(mod, edited, model="Qwen-Image", **kw):
    values = dict(method="pi", base_resolution=1280, dype_scale=1.5, dype_exponent=2.0, yarn_alt_scaling=1)
    values.update(kw)
    return mod.DyPEConfig().build(model, **values, edited=edited)[0]


def test_config_only_the_edited_and_linked_fields(mod):
    assert build(mod, "") == {"model": "Qwen-Image", "values": {}}
    assert build(mod, "", model="Z-Image") == {"model": "Z-Image", "values": {}}
    out = build(mod, "dype_scale, method, yarn_alt_scaling", dype_scale=3)["values"]
    assert out == {"method": "pi", "dype_scale": 3.0, "yarn_alt_scaling": True} and isinstance(out["dype_scale"], float)
    prompt = {"9": {"inputs": {"base_resolution": ["4", 0], "method": "pi"}}}
    assert build(mod, "", prompt_graph=prompt, unique_id="9")["values"] == {"base_resolution": 1280}
    assert list(build(mod, "dype_exponent", prompt_graph=prompt, unique_id="9")["values"]) == ["dype_exponent",
                                                                                              "base_resolution"]
    with pytest.raises(ValueError, match=r"DyPE Config: edited names unknown field\(s\) model; the fields are method, "
                                         r"base_resolution, dype_scale, dype_exponent, yarn_alt_scaling"):
        build(mod, "model")


def test_patcher_calls_the_one_dype(mod, bcnodes, monkeypatch):
    seen = []
    monkeypatch.setattr(bcnodes["models.common.dype"], "patch", lambda *a: seen.append(a) or "patched")
    assert mod.DyPEPatcher().patch(None, "enable", 1.0) == (None,) and not seen
    config = {"model": "Z-Image", "values": {"method": "ntk"}}
    assert mod.DyPEPatcher().patch("model", "enable", 0.7, config) == ("patched",)
    assert seen == [("model", "enable", 0.7, config)]


def test_dual_sampler_widgets(bcnodes, samplers):
    cls = bcnodes["dual_sampler"].DualSampler
    req = cls.INPUT_TYPES()
    assert list(req["required"]) == [
        "model", "positive", "negative", "latent_image", "vae", "seed",
        "steps_1", "cfg_1", "sampler_1", "scheduler_1", "steps_2", "cfg_2", "sampler_2", "scheduler_2", "denoise_2",
        "upscale_method", "scale_by", "blend_factor", "pixel_upscale_method", "second_seed", "dype"]
    defaults = {name: spec[1]["default"] for name, spec in req["required"].items() if len(spec) > 1 and "default" in spec[1]}
    assert defaults == {"seed": 0, "steps_1": 9, "cfg_1": 1.0, "sampler_1": "dpmpp_sde", "scheduler_1": "beta", "steps_2": 9,
                        "cfg_2": 1.0, "sampler_2": "dpmpp_sde", "scheduler_2": "beta", "denoise_2": 0.45,
                        "upscale_method": "bislerp", "scale_by": 2.0, "blend_factor": 0.5, "pixel_upscale_method": "lanczos",
                        "second_seed": "increment", "dype": "auto"}
    assert req["required"]["upscale_method"][0] == ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    assert req["required"]["pixel_upscale_method"][0] == ["lanczos", "bicubic"]
    assert list(req["optional"]) == ["model_2", "dype_config", "dype_auto_scale"] and req["optional"]["model_2"][0] == "MODEL"
    assert req["required"]["dype"][0] == ["auto", "enable", "disable"]
    auto_scale = req["optional"]["dype_auto_scale"]
    assert auto_scale[0] == "FLOAT" and [auto_scale[1][k] for k in ("default", "min", "max", "step")] == [1.25, 0.0, 8.0, 0.01]
    assert req["hidden"] == {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES) == (("IMAGE", "IMAGE", "LATENT"), ("first_image", "second_image", "latent"))


@pytest.mark.parametrize("model_2, second", [({"model_2": "m2"}, "m2"), ({}, "m1")])
def test_dual_sampler_hands_the_settings_on(bcnodes, monkeypatch, model_2, second):
    flow = bcnodes["pipelines.dual_sampler"]
    seen = []
    monkeypatch.setattr(flow, "run", lambda *a, **kw: seen.append((a, kw)) or ("first", "second", "latent"))
    out = bcnodes["dual_sampler"].DualSampler().sample(
        "m1", "pos", "neg", "lat", "vae", 3, 9, 1.0, "dpmpp_sde", "beta", 7, 2.0, "euler", "simple", 0.4, "bislerp",
        1.5, 0.25, "bicubic", "fixed", "enable", **model_2)
    assert out == ("first", "second", "latent")
    (args, kw), = seen
    assert args[:6] == ("m1", second, "pos", "neg", "lat", "vae") and kw == {"want_first": True, "want_second": True}
    assert args[6] == flow.Settings(3, flow.Pass(9, 1.0, "dpmpp_sde", "beta", 1.0), flow.Pass(7, 2.0, "euler", "simple", 0.4),
                                    "bislerp", 1.5, 0.25, "bicubic", "fixed", "enable", 1.25, None)

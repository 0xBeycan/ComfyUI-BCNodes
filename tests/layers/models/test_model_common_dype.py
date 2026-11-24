"""DyPE on a model (models/common/dype.py):

- dype_exponent from the scale: 2.0 at 4x and above, 1.0 from 2x, 0.5 below;
- the settings: the adapter's values (auto_threshold among them: each model's own; yarn_alt_scaling where the model's
  yarn has it, off otherwise) and dype_exponent from the scale, each replaced by a DyPE Config field it holds;
  start_sigma clamped to [0.001, 1]; the console names yarn_alt_scaling as on only for a model whose yarn has it;
- patch: disable returns the model itself; a model no adapter takes is an error naming the supported models, and so
  is a DyPE Config set for another model; a clone carries one CALC_COND_BATCH wrapper under its key (a second patch
  replaces the first's), the model's other wrappers and its object patches (model_sampling: the shift) untouched;
- the wrapper, around one sampling step (core's calc_cond_batch: the ControlNets' get_control, then the model): auto
  at or below the adapter's auto_threshold, and any mode at or below base_resolution, run the step with the own
  embedders and the arguments unchanged; above, the step runs with the adapter's embedder (the latent's size in px,
  the scale, the sigma over sigma_max clamped to [0, 1], the exponent, the settings) on the diffusion model, on each
  multi-GPU clone's, and on every ControlNet model of the step's conditioning (its chain, its multi-GPU copies) that
  carries its own embedder, each once; ControlNet models without one are left alone; every embedder is back after
  the step, also when it raises.

The registry is the pack's, with a test adapter registered first; the patcher is a stand-in with ModelPatcher's
clone semantics for wrappers and object patches.
"""

import sys
import types

import pytest
import torch

KEY = "bcnodes_dype"
CALC = "calc_cond_batch"


class Adapter:
    name = "Test"
    attribute = "rope_embedder"
    latent_scale = 8
    defaults = {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0, "auto_threshold": 1600,
                "yarn_alt_scaling": False}

    def matches(self, model):
        return getattr(model, "kind", None) == "test"

    def own_embedder(self, control_model):
        return getattr(control_model, "own", False)

    def embedder(self, module, s, height, width, scale, t, dype_exponent):
        return ("embedder", module.name, s.method, height, width, scale, t, s.start_sigma, s.dype_scale, dype_exponent,
                s.yarn_alt_scaling)


class Patcher:
    def __init__(self, kind="test"):
        self.kind = kind
        self.model = types.SimpleNamespace(model_config=object())
        self.object_patches = {"model_sampling": "the shift node's"}
        self.wrappers = {}

    def clone(self):
        n = Patcher(self.kind)
        n.model = self.model
        n.object_patches = dict(self.object_patches)
        n.wrappers = {k: {k1: list(w) for k1, w in v.items()} for k, v in self.wrappers.items()}
        return n

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


@pytest.fixture
def dype(bcnodes, monkeypatch, comfy_stubs):
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = types.SimpleNamespace(CALC_COND_BATCH=CALC)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    models = types.ModuleType("comfy.supported_models")  # what the pack's adapters check a model against
    models.ZImage = type("ZImage", (), {})
    models.QwenImage = type("QwenImage", (), {})
    monkeypatch.setitem(sys.modules, "comfy.supported_models", models)
    registry = bcnodes["models.common.registry"]
    families = dict(registry._FAMILIES)
    registry._FAMILIES["dype"] = {"Test": Adapter(), **registry._FAMILIES.get("dype", {})}
    yield bcnodes["models.common.dype"]
    registry._FAMILIES.clear()
    registry._FAMILIES.update(families)


def config(model="Test", **values):
    return {"model": model, "values": values}


def test_the_exponent_follows_the_scale(dype):
    assert [dype.exponent(s) for s in (8.0, 4.0, 3.99, 3.0, 2.0, 1.99, 1.2)] == [2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5]


def test_the_settings(dype):
    s = dype.settings(Adapter(), "auto", 0.6944, {})
    assert (s.mode, s.method, s.base_resolution, s.dype_scale, s.dype_exponent, s.auto_threshold, s.yarn_alt_scaling,
            s.start_sigma) == ("auto", "vision_yarn", 1024, 2.0, None, 1600, False, 0.6944)
    s = dype.settings(Adapter(), "enable", 5.0, {"method": "pi", "dype_exponent": 0.5, "auto_threshold": 2048,
                                                 "yarn_alt_scaling": True})
    assert (s.method, s.dype_exponent, s.auto_threshold, s.yarn_alt_scaling, s.base_resolution, s.start_sigma) == (
        "pi", 0.5, 2048, True, 1024, 1.0)
    assert dype.settings(Adapter(), "auto", 0.0, None).start_sigma == 0.001


def test_every_adapter_has_every_automatic_value(dype, bcnodes):
    registry = bcnodes["models.common.registry"]
    fields = {"method", "base_resolution", "dype_scale", "auto_threshold"}
    for name in registry.names("dype"):
        assert set(registry.get("dype", name).defaults) - {"yarn_alt_scaling"} == fields, name
    assert "yarn_alt_scaling" in registry.get("dype", "Qwen-Image").defaults
    assert "yarn_alt_scaling" not in registry.get("dype", "Z-Image").defaults
    assert dype.settings(registry.get("dype", "Z-Image"), "auto", 1.0, {}).yarn_alt_scaling is False


def test_the_console_names_yarn_alt_scaling_only_where_it_acts(dype, bcnodes, caplog):
    registry = bcnodes["models.common.registry"]
    for name, note in (("Test", "yarn (yarn_alt_scaling),"), ("Z-Image", "yarn (yarn_alt_scaling not used: Z-Image's yarn "
                                                                        "has no such option),")):
        entry = registry.get("dype", name)
        caplog.clear()
        with caplog.at_level("INFO"):
            dype.Wrapper(entry, dype.settings(entry, "enable", 1.0, {"method": "yarn", "yarn_alt_scaling": True})).report(
                2048, 2048, 2.0, 1.0)
        assert note in caplog.text, caplog.text


def test_patch(dype):
    model = Patcher()
    model.add_wrapper_with_key(CALC, "another", "theirs")
    assert dype.patch(model, "disable", 1.0, config("Z-Image")) is model
    with pytest.raises(ValueError, match="DyPE supports Test, Z-Image, Qwen-Image models; this one is object. Set the "
                                         "DyPE mode to disable"):
        dype.patch(Patcher("other"), "auto", 1.0, None)
    with pytest.raises(ValueError, match="DyPE Config is set for Qwen-Image, the model is Test. Set DyPE Config's model "
                                         "to Test"):
        dype.patch(model, "auto", 1.0, config("Qwen-Image"))
    out = dype.patch(model, "auto", 0.7, config(method="ntk"))
    assert out is not model and model.wrappers[CALC] == {"another": ["theirs"]}
    (w,) = out.wrappers[CALC][KEY]
    assert out.wrappers[CALC]["another"] == ["theirs"] and out.object_patches == {"model_sampling": "the shift node's"}
    assert w.settings.method == "ntk" and w.settings.start_sigma == 0.7
    again = dype.patch(out, "enable", 1.0, None)
    (w2,) = again.wrappers[CALC][KEY]
    assert w2 is not w and w2.settings.mode == "enable" and w2.settings.method == "vision_yarn"
    with pytest.raises(ValueError, match="unknown method 'nope'"):
        dype.patch(model, "auto", 1.0, config(method="nope"))


def module(name, **kw):
    return types.SimpleNamespace(name=name, rope_embedder="own", **kw)


def base_model(name="dm"):
    """A BaseModel as calc_cond_batch gets it: its diffusion model's embedder, sigma_max 1."""
    return types.SimpleNamespace(diffusion_model=module(name), model_sampling=types.SimpleNamespace(sigma_max=torch.tensor(1.0)))


def control(name, own, previous=None, clones=()):
    """A ControlBase: its control model (own: carries its own embedder), its chain and its multi-GPU copies."""
    return types.SimpleNamespace(control_model=module(name, own=own), previous_controlnet=previous,
                                 multigpu_clones={f"cuda:{i}": c for i, c in enumerate(clones)})


class Step:
    """core's _calc_cond_batch: records the embedder of each module it is given at the time it runs."""

    def __init__(self, watched, fail=False):
        self.watched = watched
        self.fail = fail
        self.calls = []

    def __call__(self, *args):
        self.calls.append((args, {m.name: m.rope_embedder for m in self.watched}))
        if self.fail:
            raise RuntimeError("the step failed")
        return "out"


def step(dype, mode, height, width, sigma=0.5, cfg=None, conds=None, model=None, options=None, watched=(), fail=False):
    """-> {module name: the embedder it ran with} for one step, every embedder checked back after it."""
    (w,) = dype.patch(Patcher(), mode, 0.7, cfg).wrappers[CALC][KEY]
    model = model or base_model()
    watched = [model.diffusion_model, *watched]
    ex = Step(watched, fail)
    x, t = torch.zeros(1, 16, height, width), torch.tensor([sigma, sigma], dtype=torch.float64)
    conds = conds if conds is not None else [[{"model_conds": {}}], None]
    options = options if options is not None else {"transformer_options": {}}
    assert w(ex, model, conds, x, t, options) == "out"
    (args, seen), = ex.calls
    assert args[0] is model and args[1] is conds and args[2] is x and args[3] is t and args[4] is options
    assert all(m.rope_embedder == "own" for m in watched)  # put back
    return seen


def test_auto_leaves_a_latent_at_or_below_the_threshold_alone(dype):
    assert step(dype, "auto", 160, 200)["dm"] == "own"   # 1280 x 1600 px: the adapter's auto_threshold
    assert step(dype, "auto", 128, 192)["dm"] == "own"   # 1024 x 1536 px
    assert step(dype, "auto", 192, 288, cfg=config(auto_threshold=2304))["dm"] == "own"
    assert step(dype, "enable", 128, 128)["dm"] == "own"  # 1024 px: not larger than base_resolution


def test_above_it_the_step_runs_with_the_adapters_embedder(dype):
    # 2048 x 3072 px: scale 3, exponent 1.0
    assert step(dype, "auto", 256, 384, sigma=0.6)["dm"] == (
        "embedder", "dm", "vision_yarn", 2048, 3072, 3.0, 0.6, 0.7, 2.0, 1.0, False)
    # 1600 x 1608 px: just over the adapter's auto_threshold; a sigma over sigma_max clamped
    assert step(dype, "auto", 200, 201, sigma=1.4)["dm"][3:7] == (1600, 1608, 1608 / 1024, 1.0)
    assert step(dype, "enable", 160, 160, cfg=config(dype_exponent=3.0, base_resolution=512, yarn_alt_scaling=True))["dm"] == (
        "embedder", "dm", "vision_yarn", 1280, 1280, 2.5, 0.5, 0.7, 2.0, 3.0, True)
    seen = step(dype, "enable", 512, 128)  # 4096 px: exponent 2.0
    assert seen["dm"][5] == 4.0 and seen["dm"][9] == 2.0


def test_a_controlnet_with_its_own_embedder_takes_dypes_in_the_same_step(dype):
    plain = control("plain", own=False)
    copy = control("copy", own=True)
    chained = control("chained", own=True, previous=plain, clones=[copy])
    first = control("first", own=True, previous=chained)
    conds = [[{"control": first}, {"control": first}], [{"control": chained}], None]  # one control in several conds
    seen = step(dype, "enable", 256, 256, conds=conds, watched=[m.control_model for m in (first, chained, plain, copy)])
    assert {name: e[1] if isinstance(e, tuple) else e for name, e in seen.items()} == {
        "dm": "dm", "first": "first", "chained": "chained", "plain": "own", "copy": "copy"}
    below = step(dype, "auto", 128, 128, conds=conds, watched=[first.control_model])
    assert below == {"dm": "own", "first": "own"}


def test_each_multi_gpu_clone_takes_it_once(dype):
    model = base_model()
    other = base_model("dm1")
    options = {"multigpu_clones": {"cuda:0": types.SimpleNamespace(model=model), "cuda:1": types.SimpleNamespace(model=other)}}
    seen = step(dype, "enable", 256, 256, model=model, options=options, watched=[other.diffusion_model])
    assert seen["dm"][1] == "dm" and seen["dm1"][1] == "dm1"
    w = dype.patch(Patcher(), "enable", 1.0, None).wrappers[CALC][KEY][0]
    assert w.modules(model, [], options) == [model.diffusion_model, other.diffusion_model]


def test_every_embedder_is_back_after_a_failed_step(dype):
    model, first = base_model(), control("first", own=True)
    with pytest.raises(RuntimeError, match="the step failed"):
        step(dype, "enable", 256, 256, conds=[[{"control": first}]], model=model, watched=[first.control_model], fail=True)
    assert model.diffusion_model.rope_embedder == "own" and first.control_model.rope_embedder == "own"

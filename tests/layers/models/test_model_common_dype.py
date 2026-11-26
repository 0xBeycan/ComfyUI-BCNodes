"""DyPE on a model (models/common/dype.py):

- dype_exponent from the scale: 2.0 at 4x and above, 1.0 from 2x, 0.5 below;
- the settings: the adapter's values (auto_threshold among them: each model's own; yarn_alt_scaling where the model's
  yarn has it, off otherwise) and dype_exponent from the scale, each replaced by a DyPE Config field it holds;
  start_sigma clamped to [0.001, 1]; the console names yarn_alt_scaling as on only for a model whose yarn has it;
- patch: disable returns the model itself; a model no adapter takes is an error naming the supported models, and so
  is a DyPE Config set for another model; a clone carries one CALC_COND_BATCH wrapper and one APPLY_MODEL wrapper
  under its key (a second patch replaces the first's), the model's other wrappers and its object patches
  (model_sampling: the shift) untouched;
- the window, one sampling step (core's calc_cond_batch: the ControlNets' get_control, then the model): auto at or
  below the adapter's auto_threshold, and any mode at or below base_resolution, run the step with the modules' own
  embedders and the arguments unchanged; above, the diffusion model (each multi-GPU clone's too) and every ControlNet
  model of the step's conditioning (its chain, its multi-GPU copies) that carries its own embedder compute, each call,
  with the adapter's embedder for the size of that call (the step's sigma over sigma_max clamped to [0, 1], the
  exponent, the settings) or with their own when DyPE leaves that size alone: a model call sized from its latent (the
  APPLY_MODEL wrapper), a ControlNet call from its control model's `x`, anything else from the step's latent; an
  area crop smaller than the latent therefore runs at the crop's size, a ControlNet reading the diffusion model's
  embedder (Qwen-Image's Fun) at its own input's; ControlNet models without an embedder of their own keep it; every
  embedder is back and every hook removed after the step, also when it raises; the APPLY_MODEL wrapper outside a step
  runs the call as it is; a start sigma of None is the sigma of each run's first model call (a run known by its
  sample_sigmas), kept for the run's later steps and read again for the next run.

The registry is the pack's, with a test adapter registered first; the patcher is a stand-in with ModelPatcher's
clone semantics for wrappers and object patches.
"""

import sys
import types

import pytest
import torch

KEY = "bcnodes_dype"
CALC = "calc_cond_batch"
APPLY = "apply_model"


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

    def embedder(self, module, rope, s, height, width, scale, t, dype_exponent):
        assert rope.name == module.name  # the module's own embedder
        return lambda ids: ("embedder", module.name, s.method, height, width, scale, t, s.start_sigma, s.dype_scale,
                            dype_exponent, s.yarn_alt_scaling)


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
    extension.WrappersMP = types.SimpleNamespace(CALC_COND_BATCH=CALC, APPLY_MODEL=APPLY)
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
    assert out is not model and model.wrappers[CALC] == {"another": ["theirs"]} and APPLY not in model.wrappers
    (w,) = out.wrappers[CALC][KEY]
    assert out.wrappers[APPLY][KEY] == [w.apply_model]
    assert out.wrappers[CALC]["another"] == ["theirs"] and out.object_patches == {"model_sampling": "the shift node's"}
    assert w.settings.method == "ntk" and w.settings.start_sigma == 0.7
    again = dype.patch(out, "enable", 1.0, None)
    (w2,) = again.wrappers[CALC][KEY]
    assert w2 is not w and w2.settings.mode == "enable" and w2.settings.method == "vision_yarn"
    assert again.wrappers[APPLY][KEY] == [w2.apply_model]
    with pytest.raises(ValueError, match="unknown method 'nope'"):
        dype.patch(model, "auto", 1.0, config(method="nope"))


class Own(torch.nn.Module):
    """A module's own rotary embedder (core's EmbedND)."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, ids):
        return ("own", self.name)


def module(name):
    return types.SimpleNamespace(name=name, rope_embedder=Own(name))


def base_model(name="dm"):
    """A BaseModel as calc_cond_batch gets it: its diffusion model's embedder, sigma_max 1."""
    return types.SimpleNamespace(diffusion_model=module(name), model_sampling=types.SimpleNamespace(sigma_max=torch.tensor(1.0)))


class ControlModel(torch.nn.Module):
    """A ControlNet's control model: its forward (given `x` as core's ControlNet gives it) returns what the rotary
    embedder it reads computes: its own (own) or the diffusion model's it is given as base_model (the Fun ControlNet)."""

    def __init__(self, name, own):
        super().__init__()
        self.name = name
        self.own = own
        self.rope_embedder = Own(name)

    def forward(self, x=None, base_model=None):
        return (self.rope_embedder if self.own else base_model.rope_embedder)(None)


def control(name, own, previous=None, clones=()):
    """A ControlBase: its control model, its chain and its multi-GPU copies."""
    return types.SimpleNamespace(control_model=ControlModel(name, own), previous_controlnet=previous,
                                 multigpu_clones={f"cuda:{i}": c for i, c in enumerate(clones)})


def latent(height, width):
    """A latent of height x width px (8 px cells)."""
    return torch.zeros(1, 16, height // 8, width // 8)


class Step:
    """core's _calc_cond_batch: runs `calls` in order, each ("model", x): a model call through the APPLY_MODEL wrapper
    (the model computing with its diffusion model's embedder), ("control", control, x): its control model's forward,
    or ("other", module): a module's embedder called outside both; -> what each computed."""

    def __init__(self, wrapper, calls, fail=False):
        self.wrapper = wrapper
        self.calls = calls
        self.fail = fail
        self.args = None

    def __call__(self, model, conds, x_in, timestep, options):
        self.args = (model, conds, x_in, timestep, options)
        out = []
        for call in self.calls:
            if call[0] == "model":
                out.append(self.wrapper.apply_model(lambda x, t: model.diffusion_model.rope_embedder(None), call[1], timestep))
            elif call[0] == "control":
                out.append(call[1].control_model(x=call[2], base_model=model.diffusion_model))
            else:
                out.append(call[1].rope_embedder(None))
        if self.fail:
            raise RuntimeError("the step failed")
        return out


def step(dype, mode, size, calls=None, sigma=0.5, cfg=None, conds=None, model=None, options=None, watched=(), fail=False):
    """One step of a `size` (height, width) px latent -> what each call computed (`calls`: by default one model call at
    the latent's size); every embedder checked back after it."""
    (w,) = dype.patch(Patcher(), mode, 0.7, cfg).wrappers[CALC][KEY]
    model = model or base_model()
    x_in = latent(*size)
    ex = Step(w, calls if calls is not None else [("model", x_in)], fail)
    t = torch.tensor([sigma, sigma], dtype=torch.float64)
    conds = conds if conds is not None else [[{"model_conds": {}}], None]
    options = options if options is not None else {"transformer_options": {}}
    held = [(m, m.rope_embedder) for m in (model.diffusion_model, *watched)]
    out = w(ex, model, conds, x_in, t, options)
    assert ex.args[0] is model and ex.args[1] is conds and ex.args[2] is x_in and ex.args[3] is t and ex.args[4] is options
    assert all(m.rope_embedder is rope for m, rope in held)  # put back
    return out


def on(name, height, width, scale, sigma=0.5, exponent=1.0, method="vision_yarn", alt=False):
    return ("embedder", name, method, height, width, scale, sigma, 0.7, 2.0, exponent, alt)


def test_auto_leaves_a_latent_at_or_below_the_threshold_alone(dype):
    assert step(dype, "auto", (1280, 1600)) == [("own", "dm")]   # the adapter's auto_threshold
    assert step(dype, "auto", (1024, 1536)) == [("own", "dm")]
    assert step(dype, "auto", (1536, 2304), cfg=config(auto_threshold=2304)) == [("own", "dm")]
    assert step(dype, "enable", (1024, 1024)) == [("own", "dm")]  # not larger than base_resolution


def test_above_it_a_call_computes_with_the_adapters_embedder(dype):
    assert step(dype, "auto", (2048, 3072), sigma=0.6) == [on("dm", 2048, 3072, 3.0, 0.6)]  # scale 3, exponent 1.0
    # just over the adapter's auto_threshold; a sigma over sigma_max clamped
    assert step(dype, "auto", (1600, 1608), sigma=1.4) == [on("dm", 1600, 1608, 1608 / 1024, 1.0, 0.5)]
    assert step(dype, "enable", (1280, 1280), cfg=config(dype_exponent=3.0, base_resolution=512, yarn_alt_scaling=True)) == [
        on("dm", 1280, 1280, 2.5, exponent=3.0, alt=True)]
    assert step(dype, "enable", (4096, 1024)) == [on("dm", 4096, 1024, 4.0, exponent=2.0)]


def test_each_call_is_sized_from_its_own_input(dype):
    full = latent(2048, 2048)
    calls = [("model", full), ("model", latent(1024, 2048)), ("model", latent(1024, 1024)), ("other", None)]
    model = base_model()
    calls[3] = ("other", model.diffusion_model)  # a call nothing sizes: the step's latent
    assert step(dype, "enable", (2048, 2048), calls, model=model) == [
        on("dm", 2048, 2048, 2.0), on("dm", 1024, 2048, 2.0), ("own", "dm"), on("dm", 2048, 2048, 2.0)]
    # auto: an area crop at or under the adapter's auto_threshold (1600 px) runs with the model's own embedder
    assert step(dype, "auto", (2048, 2048), [("model", latent(1024, 1600)), ("model", full)]) == [
        ("own", "dm"), on("dm", 2048, 2048, 2.0)]


def test_a_controlnet_computes_at_its_own_inputs_size(dype):
    plain = control("plain", own=False)   # reads the diffusion model's embedder (the Fun ControlNet)
    copy = control("copy", own=True)
    chained = control("chained", own=True, previous=plain, clones=[copy])
    first = control("first", own=True, previous=chained)
    conds = [[{"control": first}, {"control": first}], [{"control": chained}], None]  # one control in several conds
    crop = latent(1024, 2048)
    calls = [("control", first, latent(2048, 2048)), ("control", plain, crop), ("control", chained, crop),
             ("control", copy, latent(1024, 1024)), ("model", crop)]
    watched = [c.control_model for c in (first, chained, plain, copy)]
    assert step(dype, "enable", (2048, 2048), calls, conds=conds, watched=watched) == [
        on("first", 2048, 2048, 2.0), on("dm", 1024, 2048, 2.0), on("chained", 1024, 2048, 2.0), ("own", "copy"),
        on("dm", 1024, 2048, 2.0)]
    assert plain.control_model.rope_embedder.name == "plain" and isinstance(plain.control_model.rope_embedder, Own)
    assert all(not c.control_model._forward_pre_hooks and not c.control_model._forward_hooks for c in (first, chained, plain, copy))
    assert step(dype, "auto", (1536, 1536), [("control", first, latent(1536, 1536))], conds=conds) == [("own", "first")]


def test_a_control_model_given_no_x_keeps_the_size_it_is_called_in(dype):
    first = control("first", own=True)
    assert step(dype, "enable", (2048, 2048), [("control", first, None)], conds=[[{"control": first}]],
                watched=[first.control_model]) == [on("first", 2048, 2048, 2.0)]


def test_each_multi_gpu_clone_takes_it_once(dype):
    model = base_model()
    other = base_model("dm1")
    options = {"multigpu_clones": {"cuda:0": types.SimpleNamespace(model=model), "cuda:1": types.SimpleNamespace(model=other)}}
    assert step(dype, "enable", (2048, 2048), [("model", latent(2048, 2048)), ("other", other.diffusion_model)], model=model,
                options=options, watched=[other.diffusion_model]) == [on("dm", 2048, 2048, 2.0), on("dm1", 2048, 2048, 2.0)]
    w = dype.patch(Patcher(), "enable", 1.0, None).wrappers[CALC][KEY][0]
    assert w.modules(model, [], options) == ([model.diffusion_model, other.diffusion_model], [])


def test_every_embedder_is_back_and_every_hook_gone_after_a_failed_step(dype):
    model, first = base_model(), control("first", own=True)
    with pytest.raises(RuntimeError, match="the step failed"):
        step(dype, "enable", (2048, 2048), [("control", first, latent(2048, 2048))], conds=[[{"control": first}]],
             model=model, watched=[first.control_model], fail=True)
    assert isinstance(model.diffusion_model.rope_embedder, Own) and isinstance(first.control_model.rope_embedder, Own)
    assert not first.control_model._forward_pre_hooks and not first.control_model._forward_hooks


def test_the_model_call_outside_a_step_runs_as_it_is(dype):
    (w,) = dype.patch(Patcher(), "enable", 1.0, None).wrappers[CALC][KEY]
    assert w.apply_model(lambda x, t, *a, **k: (x, t, a, k), "x", "t", 1, extra=2) == ("x", "t", (1,), {"extra": 2})


def test_no_start_sigma_takes_each_runs_first(dype):
    (w,) = dype.patch(Patcher(), "enable", None, None).wrappers[CALC][KEY]
    assert w.settings.start_sigma is None
    model, x_in = base_model(), latent(2048, 2048)

    def one(sigma, run):
        ex = Step(w, [("model", x_in)])
        return w(ex, model, [[{}]], x_in, torch.tensor([sigma]), {"transformer_options": {"sample_sigmas": run}})[0]

    first, second = torch.tensor([0.83, 0.5, 0.0]), torch.tensor([0.83, 0.5, 0.0])
    assert one(0.83, first)[7] == pytest.approx(0.83) and one(0.5, first)[7] == pytest.approx(0.83)  # the run's first
    assert one(0.62, second)[7] == pytest.approx(0.62)  # another run (an equal tensor, another object): read again
    assert one(1.4, torch.tensor([1.4]))[7] == 1.0      # clamped to [0.001, 1]
    assert w.settings.start_sigma is None

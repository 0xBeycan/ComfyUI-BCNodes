"""DyPE on a model (models/common/dype.py):

- dype_exponent from the scale: 2.0 at 4x and above, 1.0 from 2x, 0.5 below;
- the settings: the adapter's values, auto_threshold 1920 and dype_exponent from the scale, each replaced by a DyPE
  Config field it holds; start_sigma clamped to [0.001, 1];
- patch: disable returns the model itself; a model no adapter takes is an error naming the supported models; a
  clone carries one APPLY_MODEL wrapper under its key (a second patch replaces the first's), the model's other
  wrappers and its object patches (model_sampling: the shift) untouched;
- the wrapper: auto at or below auto_threshold, and any mode at or below base_resolution, call the model with its
  own embedder and the arguments unchanged; above, the call runs with the adapter's embedder for that call (the
  scale from the latent in px, the sigma over sigma_max clamped to [0, 1], the exponent), and the model's own
  embedder is back after the call, also when it raises.

The registry is the pack's, with a test adapter registered first; the patcher is a stand-in with ModelPatcher's
clone semantics for wrappers and object patches.
"""

import sys
import types

import pytest
import torch

KEY = "bcnodes_dype"
APPLY = "apply_model"


class Adapter:
    name = "Test"
    attribute = "rope_embedder"
    latent_scale = 8
    defaults = {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0}

    def matches(self, model):
        return getattr(model, "kind", None) == "test"

    def embedder(self, diffusion_model, method, scale, t, start_sigma, dype_scale, dype_exponent):
        return ("embedder", method, scale, t, start_sigma, dype_scale, dype_exponent)


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
    extension.WrappersMP = types.SimpleNamespace(APPLY_MODEL=APPLY)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    models = types.ModuleType("comfy.supported_models")  # what the Z-Image adapter checks a model against
    models.ZImage = type("ZImage", (), {})
    monkeypatch.setitem(sys.modules, "comfy.supported_models", models)
    registry = bcnodes["models.common.registry"]
    families = dict(registry._FAMILIES)
    registry._FAMILIES["dype"] = {"Test": Adapter(), **registry._FAMILIES.get("dype", {})}
    yield bcnodes["models.common.dype"]
    registry._FAMILIES.clear()
    registry._FAMILIES.update(families)


def test_the_exponent_follows_the_scale(dype):
    assert [dype.exponent(s) for s in (8.0, 4.0, 3.99, 3.0, 2.0, 1.99, 1.2)] == [2.0, 2.0, 1.0, 1.0, 1.0, 0.5, 0.5]


def test_the_settings(dype):
    s = dype.settings(Adapter(), "auto", 0.6944, {})
    assert (s.mode, s.method, s.base_resolution, s.dype_scale, s.dype_exponent, s.auto_threshold, s.start_sigma) == (
        "auto", "vision_yarn", 1024, 2.0, None, 1920, 0.6944)
    s = dype.settings(Adapter(), "enable", 5.0, {"method": "pi", "dype_exponent": 0.5, "auto_threshold": 2048})
    assert (s.method, s.dype_exponent, s.auto_threshold, s.base_resolution, s.start_sigma) == ("pi", 0.5, 2048, 1024, 1.0)
    assert dype.settings(Adapter(), "auto", 0.0, None).start_sigma == 0.001


def test_patch(dype):
    model = Patcher()
    model.add_wrapper_with_key(APPLY, "another", "theirs")
    assert dype.patch(model, "disable", 1.0, {}) is model
    with pytest.raises(ValueError, match="DyPE supports Test(, Z-Image)? models; this one is object. Set the DyPE mode to disable"):
        dype.patch(Patcher("other"), "auto", 1.0, {})
    out = dype.patch(model, "auto", 0.7, {"method": "ntk"})
    assert out is not model and model.wrappers[APPLY] == {"another": ["theirs"]}
    (w,) = out.wrappers[APPLY][KEY]
    assert out.wrappers[APPLY]["another"] == ["theirs"] and out.object_patches == {"model_sampling": "the shift node's"}
    assert w.settings.method == "ntk" and w.settings.start_sigma == 0.7
    again = dype.patch(out, "enable", 1.0, {})
    (w2,) = again.wrappers[APPLY][KEY]
    assert w2 is not w and w2.settings.mode == "enable"
    with pytest.raises(ValueError, match="unknown method 'nope'"):
        dype.patch(model, "auto", 1.0, {"method": "nope"})


class Model:
    """The BaseModel an APPLY_MODEL executor holds: its diffusion model's embedder, sigma_max 1."""

    def __init__(self):
        self.diffusion_model = types.SimpleNamespace(rope_embedder="own")
        self.model_sampling = types.SimpleNamespace(sigma_max=torch.tensor(1.0))


class Executor:
    def __init__(self, fail=False):
        self.class_obj = Model()
        self.fail = fail
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs, self.class_obj.diffusion_model.rope_embedder))
        if self.fail:
            raise RuntimeError("the model failed")
        return "out"


def call(dype, mode, height, width, sigma=0.5, config=None, fail=False):
    """-> (the executor, the embedder the model ran with)."""
    (w,) = dype.patch(Patcher(), mode, 0.7, config or {}).wrappers[APPLY][KEY]
    ex, x, t = Executor(fail), torch.zeros(1, 16, height, width), torch.tensor([sigma, sigma], dtype=torch.float64)
    assert w(ex, x, t, None, None, None, {"opts": 1}, extra=2) == "out"
    (args, kwargs, rope), = ex.calls
    assert args[0] is x and args[1] is t and args[2:] == (None, None, None, {"opts": 1}) and kwargs == {"extra": 2}
    assert ex.class_obj.diffusion_model.rope_embedder == "own"  # put back
    return ex, rope


def test_auto_leaves_a_latent_at_or_below_the_threshold_alone(dype):
    assert call(dype, "auto", 160, 240)[1] == "own"   # 1280 x 1920 px
    assert call(dype, "auto", 128, 192)[1] == "own"   # 1024 x 1536 px
    assert call(dype, "auto", 192, 288, config={"auto_threshold": 2304})[1] == "own"
    assert call(dype, "enable", 128, 128)[1] == "own"  # 1024 px: not larger than base_resolution


def test_above_it_the_call_runs_with_the_adapters_embedder(dype):
    _, rope = call(dype, "auto", 256, 384, sigma=0.6)  # 2048 x 3072 px: scale 3, exponent 1.0
    assert rope == ("embedder", "vision_yarn", 3.0, 0.6, 0.7, 2.0, 1.0)
    _, rope = call(dype, "auto", 192, 288, sigma=1.4)  # 1536 x 2304 px: scale 2.25; a sigma over sigma_max clamped
    assert rope == ("embedder", "vision_yarn", 2.25, 1.0, 0.7, 2.0, 1.0)
    _, rope = call(dype, "enable", 160, 160, config={"dype_exponent": 3.0, "base_resolution": 512})
    assert rope == ("embedder", "vision_yarn", 2.5, 0.5, 0.7, 2.0, 3.0)
    _, rope = call(dype, "enable", 512, 128)  # 4096 px: exponent 2.0
    assert rope[2] == 4.0 and rope[6] == 2.0


def test_the_models_embedder_is_back_after_a_failed_call(dype):
    (w,) = dype.patch(Patcher(), "enable", 1.0, {}).wrappers[APPLY][KEY]
    ex = Executor(fail=True)
    with pytest.raises(RuntimeError, match="the model failed"):
        w(ex, torch.zeros(1, 16, 256, 256), torch.tensor([0.5]))
    assert ex.calls[0][2][0] == "embedder" and ex.class_obj.diffusion_model.rope_embedder == "own"

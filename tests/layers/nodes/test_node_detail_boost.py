"""Detail Boost (nodes/detail_boost.py): the output is always a clone and the input model is never
changed; weight 0 or eta 0 leave the clone unpatched; otherwise the clone carries one PREDICT_NOISE
wrapper beside the wrappers and the model_function_wrapper the model already had; end_step at or under
start_step is an error. The wrapper hands every call on, with the timestep scaled to the lowered sigma
inside the step range and the very same timestep tensor outside it.

The patcher is a stand-in with ModelPatcher's clone semantics for wrappers (each list copied);
comfy.patcher_extension and comfy.model_sampling are stubs. The wrapper chain through ComfyUI's own
sampler is in tests/test_runtime.py (detail_boost).
"""

import sys
import types

import pytest
import torch


class Patcher:
    def __init__(self):
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}

    def clone(self):
        n = Patcher()
        n.model_options = {k: (dict(v) if isinstance(v, dict) else v) for k, v in self.model_options.items()}
        n.wrappers = {k: {k1: list(w) for k1, w in v.items()} for k, v in self.wrappers.items()}
        return n

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)


class CONST:
    sigma_max = torch.tensor(1.0)


@pytest.fixture
def comfy_api(monkeypatch, comfy_stubs):
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = types.SimpleNamespace(PREDICT_NOISE="predict_noise")
    sampling = types.ModuleType("comfy.model_sampling")
    sampling.CONST = CONST
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", sampling)


@pytest.fixture
def node(bcnodes):
    return bcnodes["detail_boost"].DetailBoost()


@pytest.fixture
def db(bcnodes):
    return bcnodes["libs.detail_boost"]


def run(node, model, weight=1.0, mode="hard", eta=0.5, start_step=3, end_step=10):
    return node.patch(model, weight, mode, eta, start_step, end_step)[0]


def test_widgets_and_defaults(node, db):
    req = node.INPUT_TYPES()["required"]
    assert list(req) == ["model", "weight", "mode", "eta", "start_step", "end_step"]
    assert req["mode"][0] == list(db.MODES) and req["mode"][1]["default"] == "hard"
    assert [req[k][1]["default"] for k in ("weight", "eta", "start_step", "end_step")] == [1.0, 0.5, 3, 10]


def test_off_leaves_the_clone_unpatched(node, comfy_api):
    model = Patcher()
    for weight, eta in ((0.0, 0.5), (1.0, 0.0)):
        out = run(node, model, weight=weight, eta=eta)
        assert out is not model and out.wrappers == {} and out.model_options == model.model_options


def test_patch_keeps_the_input_and_the_other_patches(node, comfy_api):
    model = Patcher()
    unet_wrapper = object()
    other = object()
    model.model_options["model_function_wrapper"] = unet_wrapper
    model.add_wrapper_with_key("predict_noise", "another_pack", other)
    out = run(node, model)
    assert model.wrappers == {"predict_noise": {"another_pack": [other]}}
    assert out.wrappers["predict_noise"]["another_pack"] == [other]
    assert len(out.wrappers["predict_noise"]["bcnodes_detail_boost"]) == 1
    assert out.model_options["model_function_wrapper"] is unet_wrapper
    # a second node on the clone chains another wrapper under the same key
    twice = run(node, out)
    assert len(twice.wrappers["predict_noise"]["bcnodes_detail_boost"]) == 2


def test_step_range_error(node, comfy_api):
    with pytest.raises(ValueError, match=r"end_step \(3\) must be greater than start_step \(3\)"):
        run(node, Patcher(), start_step=3, end_step=3)
    assert run(node, Patcher(), start_step=3, end_step=-1).wrappers


def test_none_model(node):
    assert run(node, None) is None


class Executor:
    def __init__(self):
        self.class_obj = types.SimpleNamespace(inner_model=types.SimpleNamespace(model_sampling=CONST()))
        self.calls = []

    def __call__(self, x, timestep, model_options, seed):
        self.calls.append((x, timestep, model_options, seed))
        return "out"


def test_wrapper_scales_the_timestep_in_range(node, comfy_api, db):
    boost = run(node, Patcher()).wrappers["predict_noise"]["bcnodes_detail_boost"][0]
    schedule = torch.tensor([1.0 - i / 20 for i in range(21)])
    options = {"transformer_options": {"sample_sigmas": schedule}}
    x = torch.zeros(2, 4)
    for step in (2, 3, 9, 10):
        ex = Executor()
        timestep = schedule[step] * torch.ones(2)
        assert boost(ex, x, timestep, options, 7) == "out"
        got_x, got_t, got_options, seed = ex.calls[0]
        assert got_x is x and got_options is options and seed == 7
        if 3 <= step < 10:
            sigma, prev = schedule[step].item(), schedule[step - 1].item()
            expected = db.lowered_sigma("hard", sigma, prev, 0.5, True, 1.0)
            assert torch.allclose(got_t, torch.full((2,), expected)) and got_t.dtype == timestep.dtype
        else:
            assert got_t is timestep

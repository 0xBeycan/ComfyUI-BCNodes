"""The pieces of libs/flow_shift.py that Qwen Image Shift and Dual Sampling Shift share: flow_sampling
(ModelSamplingAuraFlow's object: core's ModelSamplingDiscreteFlow + CONST on the model config, multiplier 1.0, the
model's noise_scale kept when its model_sampling has one, the model untouched) and the OUTER_SAMPLE wrapper pieces:
refine_pass (a first sigma below the full-noise value and nonzero noise, nothing else), check_model_sampling
(the run's model_sampling must be the node's object) and run_at_shift (model_sampling at the shift for the run,
then at the value after it, also when the run fails; the executor's result and arguments passed through).
Both nodes through these pieces: tests/layers/nodes/test_node_qwen_image_shift.py and
test_node_dual_sampling_shift.py. comfy.model_sampling is a stub.
"""

import sys
import types

import pytest
import torch


@pytest.fixture
def fs(bcnodes):
    return bcnodes["libs.flow_shift"]


class Sampling:
    def __init__(self):
        self.shift = 1.0
        self.calls = []

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.calls.append((shift, multiplier))
        self.shift = shift


class FlowSampling:
    """Stands in for core's ModelSamplingDiscreteFlow: noise_scale from the model config; records set_parameters."""

    def __init__(self, model_config=None):
        self.model_config = model_config
        self.noise_scale = float(model_config.sampling_settings.get("noise_scale", 1.0))
        self.calls = []

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.calls.append((shift, multiplier))
        self.shift = shift

    def set_noise_scale(self, noise_scale):
        self.noise_scale = float(noise_scale)


class CONST:
    pass


@pytest.fixture
def model_sampling(monkeypatch, comfy_stubs):
    stub = types.ModuleType("comfy.model_sampling")
    stub.ModelSamplingDiscreteFlow = FlowSampling
    stub.CONST = CONST
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", stub)


@pytest.mark.parametrize("own, noise_scale", [
    ({"noise_scale": 8.0}, 8.0),  # a flow model's model_sampling: its noise_scale kept
    ({"shift": 1.15}, 0.5),  # ModelSamplingFlux has none: the model config's
])
def test_flow_sampling(fs, model_sampling, own, noise_scale):
    original = types.SimpleNamespace(**own)
    config = types.SimpleNamespace(sampling_settings={"noise_scale": 0.5})
    inner = types.SimpleNamespace(model_config=config, model_sampling=original)
    model = types.SimpleNamespace(model=inner, get_model_object=lambda name: getattr(inner, name))
    sampling = fs.flow_sampling(model, 2.5)
    assert type(sampling).__bases__ == (FlowSampling, CONST) and sampling.model_config is config
    assert sampling.calls == [(2.5, 1.0)] and sampling.noise_scale == noise_scale
    assert inner.model_sampling is original and vars(original) == own


def test_refine_pass(fs):
    noise, zeros = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0)), torch.zeros(1, 4, 8, 8)
    one_value = torch.zeros(1, 4, 8, 8)
    one_value[0, 1, 2, 3] = -0.25
    tail = torch.tensor([0.6, 0.3, 0.0])
    assert fs.refine_pass(noise, tail, 1.0) and fs.refine_pass(one_value, tail, 1.0)
    assert not fs.refine_pass(zeros, tail, 1.0)  # core's empty noise: the later part of a split run
    assert not fs.refine_pass(noise, torch.tensor([1.0, 0.5, 0.0]), 1.0)  # from full noise
    assert not fs.refine_pass(noise, tail, 0.6) and fs.refine_pass(noise, tail, 0.6000001)


def test_check_model_sampling(fs):
    ours, other = Sampling(), Sampling()
    patcher = types.SimpleNamespace(get_model_object=lambda name: {"model_sampling": ours}[name])
    executor = types.SimpleNamespace(class_obj=types.SimpleNamespace(model_patcher=patcher))
    assert fs.check_model_sampling(executor, ours, "[prefix]") is None
    with pytest.raises(ValueError, match=r"^\[prefix\] a node after it replaced the model's model_sampling "
                                         r"\(ModelSamplingAuraFlow or another shift node\), so its shift would be "
                                         r"applied twice\. Keep one of the two\.$"):
        fs.check_model_sampling(executor, other, "[prefix]")


def test_run_at_shift(fs):
    sampling = Sampling()
    seen = []

    def executor(*args, **kwargs):
        seen.append((args, kwargs, sampling.shift))
        return "out"

    assert fs.run_at_shift(sampling, 2.5, 3.0, 1000, executor, ["a", 1], {"latent_shapes": [(1, 2)]}) == "out"
    assert seen == [(("a", 1), {"latent_shapes": [(1, 2)]}, 2.5)]
    assert sampling.calls == [(2.5, 1000), (3.0, 1000)] and sampling.shift == 3.0

    def failing(*args, **kwargs):
        raise RuntimeError("the sampler failed")

    with pytest.raises(RuntimeError, match="the sampler failed"):
        fs.run_at_shift(sampling, 1.5, 1.0, 1.0, failing, [], {})
    assert sampling.calls[2:] == [(1.5, 1.0), (1.0, 1.0)] and sampling.shift == 1.0

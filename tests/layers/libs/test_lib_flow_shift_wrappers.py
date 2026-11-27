"""The OUTER_SAMPLE wrapper pieces of libs/flow_shift.py that Qwen Image Shift and Dual Sampling Shift share:
refine_pass (a first sigma below the full-noise value and nonzero noise, nothing else), check_model_sampling
(the run's model_sampling must be the node's object) and run_at_shift (model_sampling at the shift for the run,
then at the value after it, also when the run fails; the executor's result and arguments passed through).
Both nodes through these pieces: tests/layers/nodes/test_node_qwen_image_shift.py and
test_node_dual_sampling_shift.py.
"""

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

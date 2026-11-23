"""Qwen Image Shift (nodes/qwen_image_shift.py): the output is a clone whose model_sampling is
ModelSamplingAuraFlow's object (flow, multiplier 1.0, the model's noise_scale), at `shift` in fixed mode and
at 1 in dynamic mode; the input model is never changed. Fixed mode with shift_terminal off carries no
wrapper; otherwise one OUTER_SAMPLE wrapper, a second node replacing the first one's, the model's other
wrappers kept. The wrapper hands the run on with its schedule shifted for the run's latent (dynamic) and
stretched to the terminal sigma, model_sampling at the run's shift for the run only (back at 1 after it,
also when the run fails); it refuses a model_sampling another node put after it, and a terminal stretch on
a run that does not end at 0 or whose only sigma is 1.

The patcher is a stand-in with ModelPatcher's clone semantics for wrappers and object patches;
comfy.patcher_extension and comfy.model_sampling are stubs (time_snr_shift is core's formula). The node
through ComfyUI's own sampler and against core's ModelSamplingAuraFlow is in tests/test_runtime.py
(qwen_image_shift).
"""

import logging
import math
import sys
import types

import pytest
import torch

from _diffusers_sigmas import ATOL, DIFFUSERS, OFFICIAL

KEY = "bcnodes_qwen_image_shift"
OUTER = "outer_sample"


class FlowSampling:
    """Stands in for core's ModelSamplingDiscreteFlow: records set_parameters."""

    def __init__(self, model_config=None):
        self.model_config = model_config
        self.noise_scale = 1.0
        self.calls = []
        self.shift = None

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.calls.append((shift, multiplier))
        self.shift = shift

    def set_noise_scale(self, noise_scale):
        self.noise_scale = float(noise_scale)


class CONST:
    pass


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class Patcher:
    def __init__(self):
        self.model = types.SimpleNamespace(model_config="config", model_sampling=types.SimpleNamespace(noise_scale=0.5))
        self.object_patches = {}
        self.wrappers = {}

    def clone(self):
        n = Patcher()
        n.model = self.model
        n.object_patches = dict(self.object_patches)
        n.wrappers = {k: {k1: list(w) for k1, w in v.items()} for k, v in self.wrappers.items()}
        return n

    def get_model_object(self, name):
        return self.object_patches.get(name, getattr(self.model, name))

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)


class Executor:
    """The OUTER_SAMPLE executor: records each call and the model_sampling shift it ran with."""

    def __init__(self, patcher, fail=False):
        self.class_obj = types.SimpleNamespace(model_patcher=patcher)
        self.fail = fail
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs, self.class_obj.model_patcher.get_model_object("model_sampling").shift))
        if self.fail:
            raise RuntimeError("the sampler failed")
        return "out"


@pytest.fixture
def comfy_api(monkeypatch, comfy_stubs):
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE=OUTER)
    sampling = types.ModuleType("comfy.model_sampling")
    sampling.ModelSamplingDiscreteFlow = FlowSampling
    sampling.CONST = CONST
    sampling.time_snr_shift = time_snr_shift
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", sampling)


@pytest.fixture
def node(bcnodes):
    return bcnodes["qwen_image_shift"].QwenImageShift()


@pytest.fixture
def fs(bcnodes):
    return bcnodes["libs.flow_shift"]


def run(node, model, mode="dynamic", shift=3.0, multiplier=1.0, shift_terminal=True, terminal=0.02,
        base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9):
    return node.patch(model, mode, shift, multiplier, shift_terminal, terminal, base_seq_len, max_seq_len,
                      base_shift, max_shift)[0]


def wrapper(out):
    (w,) = out.wrappers[OUTER][KEY]
    return w


def sample(w, patcher, sigmas, latent=(1, 16, 1, 166, 166), fail=False):
    """-> (the executor, the sigmas it got). The args as CFGGuider.sample hands them to OUTER_SAMPLE."""
    ex = Executor(patcher, fail)
    noise, latent_image = torch.zeros(latent), torch.zeros(latent)
    w(ex, noise, latent_image, "sampler", sigmas, None, None, False, 7, latent_shapes=[latent])
    args, kwargs, _ = ex.calls[0]
    assert args[0] is noise and args[1] is latent_image and args[2:3] + args[4:] == ("sampler", None, None, False, 7)
    assert kwargs == {"latent_shapes": [latent]}
    return ex, args[3]


def grid(steps):
    """The scheduler's sigmas at shift 1 (simple, steps dividing 1000): t from 1 down, then 0."""
    return torch.tensor([1 - i / steps for i in range(steps)] + [0.0])


def test_widgets_and_defaults(node):
    req = node.INPUT_TYPES()["required"]
    assert list(req) == ["model", "mode", "shift", "multiplier", "shift_terminal", "terminal", "base_seq_len",
                         "max_seq_len", "base_shift", "max_shift"]
    assert req["mode"][0] == ["dynamic", "fixed"] and req["mode"][1]["default"] == "dynamic"
    assert [req[k][1]["default"] for k in ("shift", "multiplier", "shift_terminal", "terminal")] == [3.0, 1.0, True, 0.02]
    assert [req[k][1]["default"] for k in ("base_seq_len", "max_seq_len", "base_shift", "max_shift")] == [
        OFFICIAL["base_seq_len"], OFFICIAL["max_seq_len"], OFFICIAL["base_shift"], OFFICIAL["max_shift"]]
    assert OFFICIAL["terminal"] == 0.02


def test_none_model(node):
    assert run(node, None) is None


def test_fixed_is_aura_flows_model_sampling_and_no_wrapper(node, comfy_api):
    model = Patcher()
    out = run(node, model, mode="fixed", shift=3.0, shift_terminal=False)
    assert out is not model and model.object_patches == {} and model.wrappers == {}
    sampling = out.object_patches["model_sampling"]
    assert isinstance(sampling, FlowSampling) and isinstance(sampling, CONST)
    assert sampling.model_config == "config" and sampling.calls == [(3.0, 1.0)] and sampling.noise_scale == 0.5
    assert out.wrappers.get(OUTER, {}).get(KEY) is None


def test_fixed_with_terminal_stretches_only(node, comfy_api, fs):
    out = run(node, Patcher(), mode="fixed", shift=2.0, shift_terminal=True, terminal=0.05)
    sampling = out.object_patches["model_sampling"]
    sigmas = torch.tensor([1.0, 0.8, 0.5, 0.25, 0.0])
    ex, got = sample(wrapper(out), out, sigmas)
    assert torch.equal(got, fs.stretch_to_terminal(sigmas, 0.05))
    assert ex.calls[0][2] == 2.0 and sampling.calls == [(2.0, 1.0)]  # model_sampling left alone


def test_dynamic_matches_diffusers_and_restores_the_shift(node, comfy_api, caplog):
    out = run(node, Patcher())
    sampling = out.object_patches["model_sampling"]
    assert sampling.calls == [(1.0, 1.0)]  # outside a run: the scheduler's plain grid
    width, height, steps, seq, mu, expected = DIFFUSERS["1328x1328_8_terminal"]
    with caplog.at_level(logging.INFO):
        ex, got = sample(wrapper(out), out, grid(steps), latent=(1, 16, 1, height // 8, width // 8))
    assert torch.allclose(got, torch.tensor(expected), rtol=0, atol=ATOL), (got - torch.tensor(expected)).abs().max()
    assert math.isclose(ex.calls[0][2], math.exp(mu), rel_tol=1e-12)
    assert sampling.calls == [(1.0, 1.0), (math.exp(mu), 1.0), (1.0, 1.0)] and sampling.shift == 1.0
    assert f"[BCNodes] Qwen Image Shift: latent 166x166, {seq} tokens, mu 0.8343, shift {math.exp(mu):.4f}" in caplog.text


def test_dynamic_without_terminal_and_with_multiplier(node, comfy_api):
    width, height, steps, _, mu, expected = DIFFUSERS["1328x1328_8_no_terminal"]
    out = run(node, Patcher(), shift_terminal=False)
    _, got = sample(wrapper(out), out, grid(steps), latent=(1, 16, 1, height // 8, width // 8))
    assert torch.allclose(got, torch.tensor(expected), rtol=0, atol=ATOL)
    out = run(node, Patcher(), shift_terminal=False, multiplier=1.5)
    ex, got = sample(wrapper(out), out, grid(steps), latent=(1, 16, 1, height // 8, width // 8))
    assert math.isclose(ex.calls[0][2], 1.5 * math.exp(mu), rel_tol=1e-12)
    assert torch.allclose(got, time_snr_shift(1.5 * math.exp(mu), grid(steps)))


def test_each_run_gets_its_own_shift(node, comfy_api):
    """One patched model, two runs: the 1.25x upscaled pass gets a larger shift."""
    out = run(node, Patcher())
    w = wrapper(out)
    first, _ = sample(w, out, grid(8), latent=(1, 16, 1, 166, 166))
    second, _ = sample(w, out, grid(8)[-5:], latent=(1, 16, 1, 208, 208))
    assert 2.30 < first.calls[0][2] < 2.31 and 2.80 < second.calls[0][2] < 2.81
    assert out.object_patches["model_sampling"].shift == 1.0


def test_shift_restored_when_the_run_fails(node, comfy_api):
    out = run(node, Patcher())
    with pytest.raises(RuntimeError, match="the sampler failed"):
        sample(wrapper(out), out, grid(8), fail=True)
    assert out.object_patches["model_sampling"].shift == 1.0


def test_other_wrappers_stay_and_a_second_node_replaces_the_first(node, comfy_api):
    model = Patcher()
    other = object()
    model.add_wrapper_with_key(OUTER, "another_pack", other)
    out = run(node, model)
    assert model.wrappers == {OUTER: {"another_pack": [other]}}
    assert out.wrappers[OUTER]["another_pack"] == [other] and len(out.wrappers[OUTER][KEY]) == 1
    again = run(node, out, mode="fixed", shift=3.0, shift_terminal=True)
    assert again.wrappers[OUTER]["another_pack"] == [other] and len(again.wrappers[OUTER][KEY]) == 1
    assert wrapper(again).dynamic is None and wrapper(again).sampling is again.object_patches["model_sampling"]
    plain = run(node, out, mode="fixed", shift=3.0, shift_terminal=False)
    assert KEY not in plain.wrappers[OUTER] and plain.wrappers[OUTER]["another_pack"] == [other]


def test_refuses_a_model_sampling_put_after_it(node, comfy_api):
    out = run(node, Patcher())
    later = out.clone()
    later.add_object_patch("model_sampling", FlowSampling())
    with pytest.raises(ValueError, match="replaced the model's model_sampling"):
        sample(wrapper(later), later, grid(8))
    assert out.object_patches["model_sampling"].calls == [(1.0, 1.0)]


def test_terminal_needs_a_run_ending_at_zero(node, comfy_api):
    out = run(node, Patcher())
    sampling = out.object_patches["model_sampling"]
    # 0.5 at the run's shift (2.3032 for 166 x 166) is 0.6973
    with pytest.raises(ValueError, match="needs a run that ends at sigma 0; this one stops at 0.6973"):
        sample(wrapper(out), out, torch.tensor([1.0, 0.75, 0.5]))
    with pytest.raises(ValueError, match="only sigma is 1"):
        sample(wrapper(out), out, torch.tensor([1.0, 0.0]))
    assert sampling.calls == [(1.0, 1.0)]  # refused before the run: model_sampling untouched
    # off: the same runs go through
    off = run(node, Patcher(), shift_terminal=False)
    _, got = sample(wrapper(off), off, torch.tensor([1.0, 0.75, 0.5]))
    assert got.shape == (3,) and got[0].item() == 1.0


def test_seq_len_line_needs_two_points(node, comfy_api):
    with pytest.raises(ValueError, match=r"max_seq_len \(256\) must differ from base_seq_len \(256\)"):
        run(node, Patcher(), max_seq_len=256)
    assert run(node, Patcher(), mode="fixed", max_seq_len=256).object_patches["model_sampling"].calls == [(3.0, 1.0)]

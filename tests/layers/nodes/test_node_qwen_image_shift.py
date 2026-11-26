"""Qwen Image Shift (nodes/qwen_image_shift.py): the output is a clone whose model_sampling is
ModelSamplingAuraFlow's object (flow, multiplier 1.0, the model's noise_scale), at `shift` in fixed mode and
at 1 in dynamic mode; the input model is never changed. Fixed mode with shift_terminal off carries no
wrapper; otherwise one OUTER_SAMPLE wrapper, a second node replacing the first one's, the model's other
wrappers kept. The wrapper hands the run on with its schedule shifted for the run's latent (dynamic) and
stretched to the terminal sigma, model_sampling at the run's shift for the run only (back at 1 after it,
also when the run fails); it refuses a model_sampling another node put after it, and a terminal stretch on
a run that does not end at 0 or whose only sigma is 1. A refine pass (first sigma below 1, nonzero noise)
above refine_threshold tokens takes mu at threshold + rate x (tokens - threshold); a run from noise, a split
run's later part (zero noise), a refine pass at or below the threshold and fixed mode are unchanged, bit for
bit; the two widgets are optional (an API prompt without them gets their defaults).

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
        base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9, **refine):
    return node.patch(model, mode, shift, multiplier, shift_terminal, terminal, base_seq_len, max_seq_len,
                      base_shift, max_shift, **refine)[0]


def wrapper(out):
    (w,) = out.wrappers[OUTER][KEY]
    return w


def sample(w, patcher, sigmas, latent=(1, 16, 1, 166, 166), fail=False, add_noise=True):
    """-> (the executor, the sigmas it got). The args as CFGGuider.sample hands them to OUTER_SAMPLE: the noise
    random (KSampler's), or all zeros as core's empty noise (add_noise off)."""
    ex = Executor(patcher, fail)
    noise = torch.randn(latent, generator=torch.Generator().manual_seed(0)) if add_noise else torch.zeros(latent)
    latent_image = torch.zeros(latent)
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
    opt = node.INPUT_TYPES()["optional"]
    assert list(opt) == ["refine_threshold", "refine_rate"]
    assert opt["refine_threshold"][0] == "INT" and opt["refine_rate"][0] == "FLOAT"
    assert [opt["refine_threshold"][1][k] for k in ("default", "min", "max")] == [9600, 0, 1000000]
    assert [opt["refine_rate"][1][k] for k in ("default", "min", "max", "step")] == [0.17, -1.0, 2.0, 0.01]


def test_refine_widgets_missing_take_their_defaults(node, comfy_api):
    """An API prompt saved before the two widgets: patch gets the ten required inputs only."""
    out = node.patch(Patcher(), "dynamic", 3.0, 1.0, True, 0.02, 256, 8192, 0.5, 0.9)[0]
    assert wrapper(out).dynamic == (1.0, 256, 8192, 0.5, 0.9, 9600, 0.17)


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
    assert (f"[BCNodes] Qwen Image Shift: latent 166x166, refine: no, tokens {seq}, mu 0.8343, shift {math.exp(mu):.4f}"
            in caplog.text)


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
    """One patched model, two runs: the 1.25x upscaled pass (6400 tokens, below refine_threshold) gets a larger
    shift."""
    out = run(node, Patcher())
    w = wrapper(out)
    first, _ = sample(w, out, grid(8), latent=(1, 16, 1, 128, 128))
    second, _ = sample(w, out, grid(8)[-5:], latent=(1, 16, 1, 160, 160))
    assert 2.00 < first.calls[0][2] < 2.01 and 2.24 < second.calls[0][2] < 2.25
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


TWO_X = (1, 16, 1, 384, 256)          # 2048 x 3072 px (1024 x 1536 upscaled 2x): 24576 tokens
AT_THRESHOLD = (1, 16, 1, 240, 160)   # 1280 x 1920 px (1024 x 1536 upscaled 1.25x): 9600 tokens


def own(fs, sigmas, tokens, multiplier=1.0, terminal=0.02):
    """-> (shift, sigmas) as the node gave them before refine passes: the latent's own shift, then the stretch."""
    shift = math.exp(fs.mu(tokens, 256, 8192, 0.5, 0.9)) * multiplier
    out = time_snr_shift(shift, sigmas)
    return shift, (out if terminal is None else fs.stretch_to_terminal(out, terminal))


def test_from_noise_unchanged_bit_for_bit(node, comfy_api, fs):
    """A full schedule (denoise 1), with or without noise: the latent's own shift, as before, at every size."""
    for latent in ((1, 16, 1, 128, 128), (1, 16, 1, 166, 166), (1, 16, 1, 192, 128), AT_THRESHOLD, TWO_X,
                   (1, 16, 1, 512, 512), (1, 16, 1, 93, 61)):
        tokens = fs.image_seq_len(*latent[-2:])
        for terminal in (True, False):
            out = run(node, Patcher(), shift_terminal=terminal)
            for steps in (8, 50):
                for add_noise in (True, False):
                    ex, got = sample(wrapper(out), out, grid(steps), latent=latent, add_noise=add_noise)
                    shift, expected = own(fs, grid(steps), tokens, terminal=0.02 if terminal else None)
                    assert ex.calls[0][2] == shift and torch.equal(got, expected), (latent, terminal, steps, add_noise)


def test_split_runs_later_part_is_no_refine(node, comfy_api, fs):
    """KSamplerAdvanced split in two at step 4 of 8: the first part starts at sigma 1, the later one gets no noise
    (add_noise off): both keep the latent's own shift. The later part with add_noise on is a refine pass."""
    out = run(node, Patcher(), shift_terminal=False)
    shift, expected = own(fs, grid(8), 24576, terminal=None)
    first, got_first = sample(wrapper(out), out, grid(8)[:5], latent=TWO_X)
    later, got_later = sample(wrapper(out), out, grid(8)[4:], latent=TWO_X, add_noise=False)
    assert first.calls[0][2] == shift and later.calls[0][2] == shift
    assert torch.equal(got_first, expected[:5]) and torch.equal(got_later, expected[4:])
    noised, _ = sample(wrapper(out), out, grid(8)[4:], latent=TWO_X)
    assert round(noised.calls[0][2], 4) == 3.0020


def test_refine_at_or_below_the_threshold_unchanged_bit_for_bit(node, comfy_api, fs):
    out = run(node, Patcher())
    for latent in (AT_THRESHOLD, (1, 16, 1, 160, 160)):
        ex, got = sample(wrapper(out), out, grid(20)[-10:], latent=latent)
        shift, expected = own(fs, grid(20)[-10:], fs.image_seq_len(*latent[-2:]))
        assert ex.calls[0][2] == shift and torch.equal(got, expected)
    assert round(math.exp(fs.mu(9600, 256, 8192, 0.5, 0.9)), 4) == 2.6405


@pytest.mark.parametrize("rate, rounded", [(0.17, 3.0020), (1.0, 5.6170), (0.0, 2.6405), (-1.0, 1.2413)])
def test_refine_above_the_threshold(node, comfy_api, fs, rate, rounded):
    """2048 x 3072, denoise 0.5: mu = mu(9600) + rate x (mu(24576) - mu(9600)); rate 1 is the latent's own shift
    and rate 0 the threshold's, bit for bit; the stretch on top; model_sampling at that shift for the run only."""
    out = run(node, Patcher(), refine_rate=rate)
    sigmas = grid(20)[-11:]
    ex, got = sample(wrapper(out), out, sigmas, latent=TWO_X)
    shift = ex.calls[0][2]
    mu_t, mu_h = (fs.mu(n, 256, 8192, 0.5, 0.9) for n in (9600, 24576))
    assert round(shift, 4) == rounded and math.isclose(math.log(shift), mu_t + rate * (mu_h - mu_t), rel_tol=1e-12)
    assert torch.equal(got, fs.stretch_to_terminal(time_snr_shift(shift, sigmas), 0.02))
    if rate == 1.0:
        own_shift, own_sigmas = own(fs, sigmas, 24576)
        assert shift == own_shift and torch.equal(got, own_sigmas)
    if rate == 0.0:
        assert shift == math.exp(mu_t)
    assert out.object_patches["model_sampling"].calls == [(1.0, 1.0), (shift, 1.0), (1.0, 1.0)]


def test_refine_with_its_widgets_and_the_multiplier(node, comfy_api, fs):
    out = run(node, Patcher(), multiplier=1.5, shift_terminal=False, refine_threshold=4096, refine_rate=0.5)
    ex, got = sample(wrapper(out), out, grid(8)[-5:], latent=TWO_X)
    shift = 1.5 * math.exp(fs.mu(4096 + 0.5 * (24576 - 4096), 256, 8192, 0.5, 0.9))
    assert ex.calls[0][2] == shift and torch.equal(got, time_snr_shift(shift, grid(8)[-5:]))


def test_fixed_mode_has_no_refine(node, comfy_api, fs):
    for terminal in (True, False):
        out = run(node, Patcher(), mode="fixed", shift=3.0, shift_terminal=terminal, refine_rate=0.0)
        sigmas = grid(20)[-11:]
        if terminal:
            ex, got = sample(wrapper(out), out, sigmas, latent=TWO_X)
            assert ex.calls[0][2] == 3.0 and torch.equal(got, fs.stretch_to_terminal(sigmas, 0.02))
        else:
            assert KEY not in out.wrappers.get(OUTER, {})
        assert out.object_patches["model_sampling"].calls == [(3.0, 1.0)]


def test_console_line_names_the_refine_state(node, comfy_api, caplog):
    out = run(node, Patcher())
    with caplog.at_level(logging.INFO):
        sample(wrapper(out), out, grid(8), latent=TWO_X)
        sample(wrapper(out), out, grid(8)[-5:], latent=TWO_X)
        sample(wrapper(out), out, grid(8)[-5:], latent=AT_THRESHOLD)
    assert [r.getMessage() for r in caplog.records] == [
        "[BCNodes] Qwen Image Shift: latent 256x384, refine: no, tokens 24576, mu 1.7258, shift 5.6170",
        "[BCNodes] Qwen Image Shift: latent 256x384, refine: yes, tokens 24576, threshold 9600, rate 0.17, mu 1.0993, "
        "shift 3.0020",
        "[BCNodes] Qwen Image Shift: latent 160x240, refine: yes, tokens 9600 <= threshold 9600, mu 0.9710, "
        "shift 2.6405"]

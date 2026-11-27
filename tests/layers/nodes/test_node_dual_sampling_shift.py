"""Dual Sampling Shift (nodes/dual_sampling_shift.py): the output is a clone whose model_sampling is a new object
of the model's own class at shift_1, with the model's multiplier and noise_scale; the input model is never
changed. A run from full noise (simple and beta, KSampler's sigmas at denoise 1) has exactly the sigmas
ModelSamplingAuraFlow (multiplier 1) / ModelSamplingSD3 (multiplier 1000) give at shift_1, also at shifts whose
top sigma is a float32 step off 1 and with a first sigma a few ulps below the top, and the wrapper hands that
very tensor on with model_sampling at shift_1. A refine pass (denoise below 1, nonzero noise) gets the shift_2
schedule (simple and beta, to float32 rounding) with model_sampling at shift_2 for the run only (back at shift_1
after it, also when it fails). The later part of a split run (zero noise) stays on shift_1. A model_sampling that
is not a one-shift flow one, a Qwen Image Shift wrapper on the input model and a model_sampling another node put
after it are errors; the model's other wrappers stay and a second node replaces the first one's wrapper.

The patcher is a stand-in with ModelPatcher's clone semantics for wrappers and object patches;
comfy.patcher_extension and comfy.model_sampling are stubs whose ModelSamplingDiscreteFlow and time_snr_shift
are core's formulas, and simple / beta / KSampler's denoise slice are core's code (comfy/samplers.py).
"""

import sys
import types

import pytest
import torch

KEY = "bcnodes_dual_sampling_shift"
OUTER = "outer_sample"


def time_snr_shift(alpha, t):
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


class CONST:
    pass


class VELOCITY(CONST):
    """Stands in for another prediction mixin of a flow model (core's IMG_TO_IMG_VELOCITY)."""


class EPS:
    pass


class ModelSamplingDiscreteFlow:
    """Core's ModelSamplingDiscreteFlow, its formulas; records the set_parameters calls made after __init__."""

    def __init__(self, model_config=None):
        settings = model_config.sampling_settings if model_config is not None else {}
        self.calls = []
        self.set_noise_scale(settings.get("noise_scale", 1.0))
        self.set_parameters(shift=settings.get("shift", 1.0), multiplier=settings.get("multiplier", 1000))
        self.calls = []

    def set_parameters(self, shift=1.0, timesteps=1000, multiplier=1000):
        self.calls.append((shift, multiplier))
        self.shift = shift
        self.multiplier = multiplier
        self.sigmas = self.sigma((torch.arange(1, timesteps + 1, 1) / timesteps) * multiplier)

    def set_noise_scale(self, noise_scale):
        self.noise_scale = float(noise_scale)

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def sigma(self, timestep):
        return time_snr_shift(self.shift, timestep / self.multiplier)


class ModelSamplingAV(ModelSamplingDiscreteFlow):
    pass


class ModelSamplingFlux:
    def __init__(self, model_config=None):
        self.shift = 1.15


class ModelSamplingDiscrete:
    def __init__(self, model_config=None):
        pass


class Patcher:
    def __init__(self, sampling, config):
        self.model = types.SimpleNamespace(model_config=config, model_sampling=sampling)
        self.object_patches = {}
        self.wrappers = {}

    def clone(self):
        n = Patcher(self.model.model_sampling, self.model.model_config)
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

    def get_wrappers(self, wrapper_type, key):
        return self.wrappers.get(wrapper_type, {}).get(key, [])


def model_with(bases, multiplier=1.0, noise_scale=0.5):
    """A model whose model_sampling is built as core's model_base.model_sampling builds it: class
    ModelSampling(s, c) on the model config's sampling_settings."""
    settings = {"shift": 1.15, "multiplier": multiplier, "noise_scale": noise_scale}
    config = types.SimpleNamespace(sampling_settings=settings)
    sampling = type("ModelSampling", bases, {})(config)
    return Patcher(sampling, config)


def flow_model(multiplier=1.0, mixin=CONST):
    return model_with((ModelSamplingDiscreteFlow, mixin), multiplier)


def aura_flow(model, shift, multiplier):
    """The model_sampling core's ModelSamplingSD3.patch puts on the model (ModelSamplingAuraFlow: multiplier 1.0)."""

    class ModelSamplingAdvanced(ModelSamplingDiscreteFlow, CONST):
        pass

    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(shift=shift, multiplier=multiplier)
    sampling.set_noise_scale(model.get_model_object("model_sampling").noise_scale)
    return sampling


def simple(ms, steps):
    """Core's simple_scheduler."""
    ss = len(ms.sigmas) / steps
    return torch.FloatTensor([float(ms.sigmas[-(1 + int(x * ss))]) for x in range(steps)] + [0.0])


def beta(ms, steps, alpha=0.6, beta=0.6):
    """Core's beta_scheduler."""
    import numpy
    import scipy.stats

    total_timesteps = len(ms.sigmas) - 1
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    sigs, last_t = [], -1
    for t in ts:
        if t != last_t:
            sigs += [float(ms.sigmas[int(t)])]
        last_t = t
    return torch.FloatTensor(sigs + [0.0])


def ksampler_sigmas(ms, scheduler, steps, denoise=1.0):
    """KSampler.set_steps: denoise below 1 takes the tail of a longer schedule."""
    if denoise > 0.9999:
        return scheduler(ms, steps)
    return scheduler(ms, int(steps / denoise))[-(steps + 1):]


class Executor:
    """The OUTER_SAMPLE executor: records each call with the shift, multiplier and table of model_sampling."""

    def __init__(self, patcher, fail=False):
        self.class_obj = types.SimpleNamespace(model_patcher=patcher)
        self.fail = fail
        self.calls = []

    def __call__(self, *args, **kwargs):
        ms = self.class_obj.model_patcher.get_model_object("model_sampling")
        self.calls.append((args, kwargs, ms.shift, ms.multiplier, ms.sigmas.clone()))
        if self.fail:
            raise RuntimeError("the sampler failed")
        return "out"


@pytest.fixture
def comfy_api(monkeypatch, comfy_stubs):
    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = types.SimpleNamespace(OUTER_SAMPLE=OUTER)
    sampling = types.ModuleType("comfy.model_sampling")
    sampling.ModelSamplingDiscreteFlow = ModelSamplingDiscreteFlow
    sampling.ModelSamplingAV = ModelSamplingAV
    sampling.CONST = CONST
    sampling.time_snr_shift = time_snr_shift
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", sampling)


@pytest.fixture
def mod(bcnodes):
    return bcnodes["dual_sampling_shift"]


@pytest.fixture
def node(mod):
    return mod.DualSamplingShift()


def wrapper(out):
    (w,) = out.wrappers[OUTER][KEY]
    return w


def sample(out, sigmas, add_noise=True, fail=False, latent=(1, 16, 64, 64)):
    """-> (the executor, the sigmas it got). The args as CFGGuider.sample hands them to OUTER_SAMPLE: the noise
    random (KSampler's), or all zeros as core's empty noise (add_noise off)."""
    ex = Executor(out, fail)
    noise = torch.randn(latent, generator=torch.Generator().manual_seed(0)) if add_noise else torch.zeros(latent)
    latent_image = torch.zeros(latent)
    result = wrapper(out)(ex, noise, latent_image, "sampler", sigmas, None, None, False, 7, latent_shapes=[latent])
    assert result == "out"
    args, kwargs = ex.calls[0][:2]
    assert args[0] is noise and args[1] is latent_image and args[2:3] + args[4:] == ("sampler", None, None, False, 7)
    assert kwargs == {"latent_shapes": [latent]}
    return ex, args[3]


SCHEDULERS = [simple, beta]
RUNS = [(8, 0.5), (20, 0.3), (4, 0.75), (30, 0.95)]  # (steps, denoise) of a refine pass


def test_widgets_and_registration(mod):
    cls = mod.DualSamplingShift
    types_ = cls.INPUT_TYPES()
    assert list(types_) == ["required"]
    req = types_["required"]
    assert list(req) == ["model", "shift_1", "shift_2"] and req["model"][0] == "MODEL"
    for name in ("shift_1", "shift_2"):
        assert req[name][0] == "FLOAT"
        assert [req[name][1][k] for k in ("default", "min", "max", "step")] == [3.0, 0.0, 100.0, 0.01]
    assert all(spec[1]["tooltip"] for spec in req.values())
    assert cls.RETURN_TYPES == ("MODEL",) and cls.RETURN_NAMES == ("MODEL",) and cls.FUNCTION == "patch"
    assert cls.CATEGORY == "BCNodes/model" and cls.SEARCH_ALIASES[0] == "BCNodes" and cls.DESCRIPTION
    assert not getattr(cls, "OUTPUT_NODE", False)
    assert mod.NODE_CLASS_MAPPINGS == {"BC_DualSamplingShift": cls}
    assert mod.NODE_DISPLAY_NAME_MAPPINGS == {"BC_DualSamplingShift": "Dual Sampling Shift"}


def test_none_model(node):
    assert node.patch(None, 3.0, 2.0) == (None,)


@pytest.mark.parametrize("multiplier, mixin", [(1.0, CONST), (1000, CONST), (1.0, VELOCITY)])
def test_clone_keeps_the_models_class_multiplier_and_noise_scale(node, comfy_api, multiplier, mixin):
    model = flow_model(multiplier, mixin)
    original = model.get_model_object("model_sampling")
    (out,) = node.patch(model, 2.5, 1.5)
    assert out is not model and model.object_patches == {} and model.wrappers == {}
    assert original.calls == [] and original.shift == 1.15
    sampling = out.object_patches["model_sampling"]
    assert sampling is not original and type(sampling) is type(original)
    assert sampling.calls == [(2.5, multiplier)] and sampling.shift == 2.5 and sampling.multiplier == multiplier
    assert sampling.noise_scale == 0.5
    assert torch.equal(sampling.sigmas, aura_flow(model, 2.5, multiplier).sigmas)
    assert len(out.wrappers[OUTER][KEY]) == 1


@pytest.mark.parametrize("scheduler", SCHEDULERS)
@pytest.mark.parametrize("multiplier", [1.0, 1000])
def test_from_full_noise_is_aura_flow_at_shift_1(node, comfy_api, scheduler, multiplier):
    """1.18 and 1.43: the top sigma is 0.99999988, below 1; 0.54: 1.0000001, above it."""
    for shift_1 in (3.0, 1.18, 1.43, 0.54, 6.0):
        model = flow_model(multiplier)
        (out,) = node.patch(model, shift_1, 1.5)
        reference = aura_flow(model, shift_1, multiplier)
        sampling = out.object_patches["model_sampling"]
        for steps in (1, 4, 8, 20, 30):
            sigmas = ksampler_sigmas(sampling, scheduler, steps)
            assert torch.equal(sigmas, ksampler_sigmas(reference, scheduler, steps)), (shift_1, steps)
            for add_noise in (True, False):
                ex, got = sample(out, sigmas, add_noise=add_noise)
                assert got is sigmas and ex.calls[0][2:4] == (shift_1, multiplier), (shift_1, steps, add_noise)
        assert sampling.calls == [(shift_1, multiplier)]


def test_a_first_sigma_a_few_ulps_off_the_top_is_full_noise(node, comfy_api):
    """normal, sgm_uniform and karras compute their first sigma from the top one and land slightly around it."""
    (out,) = node.patch(flow_model(), 1.18, 1.5)
    top = out.object_patches["model_sampling"].sigma_max
    assert float(top) < 1.0
    for start in (top, torch.nextafter(torch.nextafter(top, torch.tensor(0.0)), torch.tensor(0.0)), torch.tensor(1.0)):
        sigmas = torch.cat([start.reshape(1), torch.tensor([0.5, 0.0])])
        ex, got = sample(out, sigmas)
        assert got is sigmas and ex.calls[0][2] == 1.18


@pytest.mark.parametrize("scheduler", SCHEDULERS)
@pytest.mark.parametrize("multiplier", [1.0, 1000])
@pytest.mark.parametrize("shift_1, shift_2", [(3.0, 2.0), (1.18, 6.0), (6.0, 3.0), (2.5, 3.5)])
def test_refine_pass_gets_the_shift_2_schedule(node, comfy_api, scheduler, multiplier, shift_1, shift_2):
    model = flow_model(multiplier)
    (out,) = node.patch(model, shift_1, shift_2)
    sampling = out.object_patches["model_sampling"]
    at_1, at_2 = aura_flow(model, shift_1, multiplier), aura_flow(model, shift_2, multiplier)
    for steps, denoise in RUNS:
        sigmas = ksampler_sigmas(sampling, scheduler, steps, denoise)
        expected = ksampler_sigmas(at_2, scheduler, steps, denoise)
        ex, got = sample(out, sigmas)
        # the map goes through t: float32 rounding, a few ulps
        assert got.shape == expected.shape and torch.allclose(got, expected, rtol=0, atol=5e-7), (
            steps, denoise, (got - expected).abs().max().item())
        assert got[-1].item() == 0.0 and not torch.allclose(got, sigmas, rtol=0, atol=1e-4)
        # model_sampling at shift_2 for the run (percent_to_sigma, ControlNet ranges), shift_1 after it
        assert ex.calls[0][2:4] == (shift_2, multiplier) and torch.equal(ex.calls[0][4], at_2.sigmas)
        assert sampling.shift == shift_1 and torch.equal(sampling.sigmas, at_1.sigmas)
    assert sampling.calls == [(shift_1, multiplier)] + [(shift_2, multiplier), (shift_1, multiplier)] * len(RUNS)


def test_split_runs_later_part_keeps_shift_1(node, comfy_api):
    """KSamplerAdvanced split at step 4 of 8: the first part starts at full noise, the later one gets zero noise
    (add_noise off): both are shift_1's sigmas, handed on as they are. The later part with add_noise on is a
    refine pass."""
    model = flow_model()
    (out,) = node.patch(model, 3.0, 1.5)
    full = simple(out.object_patches["model_sampling"], 8)
    first, got_first = sample(out, full[:5])
    later, got_later = sample(out, full[4:], add_noise=False)
    assert torch.equal(got_first, full[:5]) and torch.equal(got_later, full[4:])
    assert first.calls[0][2] == 3.0 and later.calls[0][2] == 3.0
    noised, got = sample(out, full[4:])
    assert noised.calls[0][2] == 1.5
    assert torch.allclose(got, simple(aura_flow(model, 1.5, 1.0), 8)[4:], rtol=0, atol=5e-7)


def test_shift_1_restored_when_the_run_fails(node, comfy_api):
    (out,) = node.patch(flow_model(), 3.0, 1.5)
    sampling = out.object_patches["model_sampling"]
    with pytest.raises(RuntimeError, match="the sampler failed"):
        sample(out, ksampler_sigmas(sampling, simple, 8, 0.5), fail=True)
    assert sampling.shift == 3.0 and sampling.calls == [(3.0, 1.0), (1.5, 1.0), (3.0, 1.0)]


@pytest.mark.parametrize("bases, names", [
    ((ModelSamplingFlux, CONST), "ModelSamplingFlux + CONST"),
    ((ModelSamplingDiscrete, EPS), "ModelSamplingDiscrete + EPS"),
    ((ModelSamplingAV, CONST), "ModelSamplingAV + CONST"),
])
def test_not_a_one_shift_flow_model(node, comfy_api, bases, names):
    model = model_with(bases)
    with pytest.raises(ValueError, match=r"needs a flow model with one shift") as error:
        node.patch(model, 3.0, 1.5)
    assert f"this model's is {names}." in str(error.value)
    assert "put ModelSamplingAuraFlow before this node" in str(error.value)
    assert model.object_patches == {} and model.wrappers == {}


def test_qwen_image_shift_on_the_input_model(bcnodes, mod, node, comfy_api):
    qwen = bcnodes["qwen_image_shift"]
    assert mod.QWEN_IMAGE_SHIFT_KEY == qwen.WRAPPER_KEY

    def shifted(mode, shift_terminal):
        return qwen.QwenImageShift().patch(flow_model(), mode, 3.0, 1.0, shift_terminal, 0.02, 256, 8192, 0.5, 0.9)[0]

    for mode, shift_terminal in (("dynamic", True), ("dynamic", False), ("fixed", True)):
        with pytest.raises(ValueError, match="the model has a Qwen Image Shift: both nodes set the model's shift"):
            node.patch(shifted(mode, shift_terminal), 3.0, 1.5)
    # fixed without shift_terminal carries no wrapper: its model_sampling is replaced, as ModelSamplingAuraFlow's
    before = shifted("fixed", False)
    (out,) = node.patch(before, 2.0, 1.5)
    assert out.object_patches["model_sampling"] is not before.object_patches["model_sampling"]
    assert out.object_patches["model_sampling"].shift == 2.0


def test_refuses_a_model_sampling_put_after_it(node, comfy_api):
    model = flow_model()
    (out,) = node.patch(model, 3.0, 1.5)
    later = out.clone()
    later.add_object_patch("model_sampling", aura_flow(model, 3.0, 1.0))
    for denoise in (1.0, 0.5):
        with pytest.raises(ValueError, match=r"^\[BCNodes\] Dual Sampling Shift: a node after it replaced the model's"):
            sample(later, ksampler_sigmas(later.get_model_object("model_sampling"), simple, 8, denoise))
    assert out.object_patches["model_sampling"].calls == [(3.0, 1.0)]


def test_other_wrappers_stay_and_a_second_node_replaces_the_first(node, comfy_api):
    model = flow_model()
    other = object()
    model.add_wrapper_with_key(OUTER, "another_pack", other)
    (out,) = node.patch(model, 3.0, 1.5)
    assert model.wrappers == {OUTER: {"another_pack": [other]}}
    assert out.wrappers[OUTER]["another_pack"] == [other] and len(out.wrappers[OUTER][KEY]) == 1
    (again,) = node.patch(out, 2.0, 1.0)
    assert again.wrappers[OUTER]["another_pack"] == [other] and len(again.wrappers[OUTER][KEY]) == 1
    w = wrapper(again)
    assert w.sampling is again.object_patches["model_sampling"] and (w.shift_1, w.shift_2) == (2.0, 1.0)
    assert type(w.sampling) is type(out.object_patches["model_sampling"]) and w.sampling.shift == 2.0

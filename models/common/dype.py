"""DyPE (Dynamic Position Extrapolation) on a model: the one implementation DyPE Patcher and Dual Sampler call.

A model of a registered DyPE adapter (the dype family, models/common/registry.py: Z-Image, Qwen-Image) is cloned
with two of ComfyUI's patcher wrappers. The window is a CALC_COND_BATCH wrapper: core's calc_cond_batch runs, for
one sampling step, each ControlNet's get_control and then the model, so the window holds both. For the step it sets
a step embedder (StepEmbedder) in place of the rotary embedder of the diffusion model (of each multi-GPU clone too)
and of every ControlNet model in the step's conditioning that carries its own (the adapter's own_embedder), and puts
each back when the step returns (or raises). A ControlNet that reads the diffusion model's embedder (Qwen-Image's Fun
ControlNet) and a model patch running inside the model (DiffSynth's) take it with the model.

Each call of the step is sized from its own input: a model call from the latent the APPLY_MODEL wrapper is given, a
ControlNet from the `x` its control model's forward is given (a forward hook on every control model of the step,
removed with the window); a call nothing sizes takes the step's latent. So an area-conditioned crop (Conditioning
Set Area) runs, model and ControlNet, at the crop's size, as the original sizes each forward from the positions it
gets. The step embedder computes with the adapter's embedder for that size and the step's sigma, or with the
module's own when DyPE is off for that size.

Nothing is left on a model between steps: an object patch would stay on the shared diffusion model after a run,
since core detaches a clone of the same model with unpatch_all=False (comfy/model_management.py load_models_gpu),
and the next run of another clone (the first pass's model) would run with it. The model's model_sampling, and so
its shift, is never touched.

When DyPE is on, per call: the scale is the latent's longer side in px over base_resolution; auto mode leaves a
latent whose longer side is auto_threshold px or less alone, and any mode one not larger than base_resolution.
dype_exponent, unless given, follows the scale (exponent()). The values come from the adapter (defaults: method,
base_resolution, dype_scale, auto_threshold, and yarn_alt_scaling where the model's yarn has it), each replaced by DyPE
Config's edited or linked field when one is given; a DyPE Config set for another model is an error.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, TypedDict

import torch

from ...libs.dype.rope import METHODS
from . import registry

MODES = ("auto", "enable", "disable")
WRAPPER_KEY = "bcnodes_dype"
_CALLS = threading.local()  # per thread (multi-GPU: a thread a device): {id(wrapper): the sizes of the calls in progress}
PREFIX = "[BCNodes] DyPE:"


class ConfigValues(TypedDict, total=False):
    """DyPE Config's edited and linked fields; {} = every value automatic. The keys are its fields in widget order,
    each with the type its value leaves the node with."""
    method: str
    base_resolution: int
    dype_scale: float
    dype_exponent: float
    auto_threshold: int
    yarn_alt_scaling: bool


class Config(TypedDict):
    """The BC_DYPE_CONFIG value: the model DyPE Config is set for (a dype entry's name) and its values."""
    model: str
    values: ConfigValues


@dataclass(frozen=True)
class Settings:
    mode: str                         # auto or enable
    method: str
    base_resolution: int              # px
    dype_scale: float                 # DyPE's lambda_s
    dype_exponent: Optional[float]    # DyPE's lambda_t; None: from the scale
    auto_threshold: int               # px
    start_sigma: float                # in [0.001, 1]
    yarn_alt_scaling: bool = False    # yarn: each image axis stretched to its own length (a model whose defaults have it)


class Adapter(Protocol):
    """An entry of the dype family: one model's DyPE."""
    name: str
    attribute: str       # the rotary embedder attribute of the diffusion model (and of a ControlNet that has its own)
    latent_scale: int    # px per latent cell
    defaults: dict       # method, base_resolution, dype_scale, auto_threshold for this model, and yarn_alt_scaling for a
                         # model whose yarn has that option

    def matches(self, model) -> bool:
        """Whether the ModelPatcher `model` is this adapter's model."""

    def own_embedder(self, control_model) -> bool:
        """Whether a ControlNet's model (a ControlBase's control_model) carries its own rotary embedder under
        `attribute`, which DyPE then replaces too."""

    def embedder(self, module, rope, settings: Settings, height, width, scale, t, dype_exponent):
        """The rotary embedder (a torch module) of one call of `module` (the diffusion model, or a ControlNet model of
        own_embedder), from `rope`, the module's own: the latent height x width px, scale its longer side over
        base_resolution, t the step's sigma over sigma_max."""


def exponent(scale):
    """dype_exponent from the scale (the longer side over base_resolution): 2.0 at 4x and above, 1.0 from 2x, 0.5 below.
    DyPE's README advises 2.0 for 4K, 1.0 for 2K to 3K and 0.5 just above the base resolution, for a 1024 px model; the
    tiers are kept in scale terms for every model (Qwen-Image at 1328: 2.0 from 5312 px, 1.0 from 2656 px)."""
    return 2.0 if scale >= 4.0 else 1.0 if scale >= 2.0 else 0.5


def adapter(model):
    """The first dype adapter that takes the ModelPatcher `model`; ValueError naming the supported ones."""
    for name in registry.names(registry.DYPE):
        entry = registry.get(registry.DYPE, name)
        if entry.matches(model):
            return entry
    config = type(getattr(model.model, "model_config", None)).__name__
    raise ValueError(f"DyPE supports {', '.join(registry.names(registry.DYPE))} models; this one is {config}. Set the "
                     "DyPE mode to disable for it.")


def settings(entry, mode, start_sigma, values):
    """The Settings of `entry`'s model for `mode`, with DyPE Config's values (ConfigValues) over the automatic ones.
    start_sigma is clamped to [0.001, 1], as the original."""
    merged = {**entry.defaults, "dype_exponent": None, **(values or {})}
    return Settings(mode=mode, start_sigma=max(0.001, min(1.0, float(start_sigma))), **merged)


class Wrapper:
    """DyPE on a model: the CALC_COND_BATCH wrapper (__call__: (executor, model, conds, x_in, timestep, model_options),
    as core's calc_cond_batch; model the BaseModel, x_in the step's latent, timestep its sigma), the APPLY_MODEL
    wrapper (apply_model: (executor, x, t, ...), as BaseModel.apply_model) and the control models' forward hooks."""

    def __init__(self, entry, settings):
        self.adapter = entry
        self.settings = settings
        self.reported = set()            # the (height, width) already logged
        self.step = None                 # (sigma over sigma_max, the step's latent size): inside the window

    def size(self, x):
        """(height, width) px of a latent."""
        return x.shape[-2] * self.adapter.latent_scale, x.shape[-1] * self.adapter.latent_scale

    def scale(self, height, width):
        """The scale of a call of height x width px, or None when DyPE leaves it alone."""
        s = self.settings
        longer = max(height, width)
        scale = longer / s.base_resolution
        if (s.mode == "auto" and longer <= s.auto_threshold) or scale <= 1.0:
            return None
        return scale

    def sizes(self):
        """This thread's stack of call sizes (a size, or None: not known)."""
        return _CALLS.__dict__.setdefault(id(self), [])

    def current(self):
        """The size of the call in progress on this thread; the step's latent when nothing sized it."""
        sizes = self.sizes()
        return (sizes[-1] if sizes else None) or self.step[1]

    def __call__(self, executor, model, conds, x_in, timestep, model_options):
        height, width = self.size(x_in)
        if self.scale(height, width) is None:  # no call of a step is larger than its latent
            self.report(height, width, None, None)
            return executor(model, conds, x_in, timestep, model_options)
        sigma = min(max(float(timestep.flatten()[0]) / float(model.model_sampling.sigma_max), 0.0), 1.0)
        attribute = self.adapter.attribute
        modules, control_models = self.modules(model, conds, model_options)
        held = [(module, getattr(module, attribute)) for module in modules]
        hooks = []
        outer, self.step = self.step, (sigma, (height, width))
        self.sizes().clear()
        try:
            for module, rope in held:
                setattr(module, attribute, StepEmbedder(self, module, rope))
            for control_model in control_models:
                hooks.append(control_model.register_forward_pre_hook(self.enter_control, with_kwargs=True))
                hooks.append(control_model.register_forward_hook(self.leave_control, always_call=True))
            return executor(model, conds, x_in, timestep, model_options)
        finally:
            for hook in hooks:
                hook.remove()
            for module, rope in held:
                setattr(module, attribute, rope)
            self.step = outer
            self.sizes().clear()

    def apply_model(self, executor, x, t, *args, **kwargs):
        """The APPLY_MODEL wrapper: the model call is sized from its latent `x` (inside the window)."""
        if self.step is None:
            return executor(x, t, *args, **kwargs)
        sizes = self.sizes()
        sizes.append(self.size(x))
        try:
            return executor(x, t, *args, **kwargs)
        finally:
            sizes.pop()

    def enter_control(self, module, args, kwargs):
        """A control model's forward pre-hook: the ControlNet call is sized from its input `x` (core's ControlNet
        gives it as a keyword; a control model given none keeps the size it is called in)."""
        x = kwargs.get("x", args[0] if args else None)
        self.sizes().append(self.size(x) if isinstance(x, torch.Tensor) and x.ndim >= 4 else None)

    def leave_control(self, module, args, output):
        sizes = self.sizes()
        if sizes:
            sizes.pop()

    def modules(self, model, conds, model_options):
        """(the modules whose rotary embedder a step replaces, each once: the diffusion model and each multi-GPU
        clone's, then the ControlNet models of `conds` that carry their own; every ControlNet model of `conds`, each
        once), the ControlNets walked through their chains and multi-GPU copies."""
        found = {}
        for base in (model, *(clone.model for clone in model_options.get("multigpu_clones", {}).values())):
            found.setdefault(id(base.diffusion_model), base.diffusion_model)
        control_models = {}
        controls = [c.get("control") for cond in conds if cond is not None for c in cond]
        seen = set()
        while controls:
            control = controls.pop()
            if control is None or id(control) in seen:
                continue
            seen.add(id(control))
            controls += [control.previous_controlnet, *control.multigpu_clones.values()]
            control_model = getattr(control, "control_model", None)  # a T2I adapter holds t2i_model
            if isinstance(control_model, torch.nn.Module):
                control_models.setdefault(id(control_model), control_model)
                if self.adapter.own_embedder(control_model):
                    found.setdefault(id(control_model), control_model)
        return list(found.values()), list(control_models.values())

    def report(self, height, width, scale, dype_exponent):
        """One console line per call size: whether DyPE is on for it and with what."""
        if (height, width) in self.reported:
            return
        self.reported.add((height, width))
        s = self.settings
        if scale is None:
            why = (f"longer side not over auto_threshold {s.auto_threshold} px (auto)"
                   if s.mode == "auto" and max(height, width) <= s.auto_threshold
                   else f"not larger than base_resolution {s.base_resolution} px")
            logging.info("%s %s %dx%d px: off, %s", PREFIX, self.adapter.name, width, height, why)
            return
        source = "from the scale" if s.dype_exponent is None else "set"
        method = s.method
        if s.method == "yarn" and s.yarn_alt_scaling:
            method += (" (yarn_alt_scaling)" if "yarn_alt_scaling" in self.adapter.defaults
                       else f" (yarn_alt_scaling not used: {self.adapter.name}'s yarn has no such option)")
        logging.info("%s %s %dx%d px: on, scale %.3f of %d px, %s, dype_scale %g, dype_exponent %g (%s), start sigma "
                     "%.4f", PREFIX, self.adapter.name, width, height, scale, s.base_resolution, method, s.dype_scale,
                     dype_exponent, source, s.start_sigma)


class StepEmbedder(torch.nn.Module):
    """A module's rotary embedder for one sampling step: each call computes with the adapter's DyPE embedder for the
    size of the call in progress (Wrapper.current) and the step's sigma, built once per size, or with the module's
    own embedder when DyPE leaves that size alone. The module and its embedder are held in a tuple, not as child
    modules: the module tree stays as core built it."""

    def __init__(self, wrapper, module, rope):
        super().__init__()
        self.wrapper = wrapper
        self.held = (module, rope)
        self.built = {}  # (height, width) -> the DyPE embedder of that size

    def forward(self, ids):
        w = self.wrapper
        module, rope = self.held
        height, width = w.current()
        scale = w.scale(height, width)
        if scale is None:
            w.report(height, width, None, None)
            return rope(ids)
        embedder = self.built.get((height, width))
        if embedder is None:
            s = w.settings
            dype_exponent = exponent(scale) if s.dype_exponent is None else s.dype_exponent
            w.report(height, width, scale, dype_exponent)
            embedder = w.adapter.embedder(module, rope, s, height, width, scale, w.step[0], dype_exponent)
            self.built[(height, width)] = embedder
        return embedder(ids)


def patch(model, mode, start_sigma, config=None):
    """`model` (a ModelPatcher) itself for mode disable; otherwise a clone carrying DyPE (one DyPE: an earlier one on
    the model is replaced). `config`: DyPE Config's value (Config) or None. ValueError for a model no adapter takes
    and for a DyPE Config set for another model."""
    if mode == "disable":
        return model
    if mode not in MODES:
        raise ValueError(f"DyPE: unknown mode {mode!r}; the modes are {', '.join(MODES)}")
    entry = adapter(model)
    if config is not None and config["model"] != entry.name:
        raise ValueError(f"DyPE: DyPE Config is set for {config['model']}, the model is {entry.name}. Set DyPE Config's "
                         f"model to {entry.name}.")
    s = settings(entry, mode, start_sigma, config["values"] if config is not None else None)
    if s.method not in METHODS:
        raise ValueError(f"DyPE: unknown method {s.method!r}; the methods are {', '.join(METHODS)}")
    from comfy.patcher_extension import WrappersMP

    patched = model.clone()
    wrapper = Wrapper(entry, s)
    for kind, function in ((WrappersMP.CALC_COND_BATCH, wrapper), (WrappersMP.APPLY_MODEL, wrapper.apply_model)):
        patched.remove_wrappers_with_key(kind, WRAPPER_KEY)
        patched.add_wrapper_with_key(kind, WRAPPER_KEY, function)
    return patched

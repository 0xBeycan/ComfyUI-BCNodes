"""DyPE (Dynamic Position Extrapolation) on a model: the one implementation DyPE Patcher and Dual Sampler call.

A model of a registered DyPE adapter (the dype family, models/common/registry.py: Z-Image, Qwen-Image) is cloned
with a CALC_COND_BATCH wrapper (ComfyUI's patcher wrappers). core's calc_cond_batch runs, for one sampling step,
each ControlNet's get_control and then the model, so the wrapper's window holds both: at each step it reads the
latent's size and the step's sigma, and when DyPE is on for that size it sets the adapter's embedder for that step
in place of the rotary embedder of the diffusion model (of each multi-GPU clone too) and of every ControlNet model
in the step's conditioning that carries its own (the adapter's own_embedder), and puts each back when the step
returns (or raises). A ControlNet that reads the diffusion model's embedder (Qwen-Image's Fun ControlNet) and a
model patch running inside the model (DiffSynth's) take DyPE's with it. Nothing is left on a model between steps:
an object patch would stay on the shared diffusion model after a run, since core detaches a clone of the same
model with unpatch_all=False (comfy/model_management.py load_models_gpu), and the next run of another clone (the
first pass's model) would run with it. The model's model_sampling, and so its shift, is never touched.

When DyPE is on, per step: the scale is the latent's longer side in px over base_resolution; auto mode leaves a
latent whose longer side is auto_threshold px or less alone, and any mode one not larger than base_resolution.
dype_exponent, unless given, follows the scale (exponent()). The values come from the adapter (defaults: method,
base_resolution, dype_scale, auto_threshold, and yarn_alt_scaling where the model's yarn has it), each replaced by DyPE
Config's edited or linked field when one is given; a DyPE Config set for another model is an error.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, TypedDict

from ...libs.dype.rope import METHODS
from . import registry

MODES = ("auto", "enable", "disable")
WRAPPER_KEY = "bcnodes_dype"
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

    def embedder(self, module, settings: Settings, height, width, scale, t, dype_exponent):
        """The rotary embedder (a torch module) of one step for `module` (the diffusion model, or a ControlNet model
        of own_embedder), from the one it holds: the latent height x width px, scale its longer side over
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
    """The CALC_COND_BATCH wrapper: (executor, model, conds, x_in, timestep, model_options), as core's
    calc_cond_batch; model the BaseModel, x_in the step's latent, timestep its sigma."""

    def __init__(self, entry, settings):
        self.adapter = entry
        self.settings = settings
        self.reported = set()  # the (height, width) already logged

    def __call__(self, executor, model, conds, x_in, timestep, model_options):
        s = self.settings
        height, width = x_in.shape[-2] * self.adapter.latent_scale, x_in.shape[-1] * self.adapter.latent_scale
        longer = max(height, width)
        scale = longer / s.base_resolution
        if (s.mode == "auto" and longer <= s.auto_threshold) or scale <= 1.0:
            self.report(height, width, None, None)
            return executor(model, conds, x_in, timestep, model_options)
        sigma = min(max(float(timestep.flatten()[0]) / float(model.model_sampling.sigma_max), 0.0), 1.0)
        dype_exponent = exponent(scale) if s.dype_exponent is None else s.dype_exponent
        self.report(height, width, scale, dype_exponent)
        attribute = self.adapter.attribute
        modules = self.modules(model, conds, model_options)
        swaps = [(module, getattr(module, attribute), self.adapter.embedder(module, s, height, width, scale, sigma,
                                                                             dype_exponent)) for module in modules]
        try:
            for module, _, embedder in swaps:
                setattr(module, attribute, embedder)
            return executor(model, conds, x_in, timestep, model_options)
        finally:
            for module, held, _ in swaps:
                setattr(module, attribute, held)

    def modules(self, model, conds, model_options):
        """The modules whose rotary embedder a step swaps, each once: the diffusion model and each multi-GPU
        clone's, then the ControlNet models in `conds` (each one's chain and multi-GPU copies) that carry their
        own."""
        found = {}
        for base in (model, *(clone.model for clone in model_options.get("multigpu_clones", {}).values())):
            found.setdefault(id(base.diffusion_model), base.diffusion_model)
        controls = [c.get("control") for cond in conds if cond is not None for c in cond]
        seen = set()
        while controls:
            control = controls.pop()
            if control is None or id(control) in seen:
                continue
            seen.add(id(control))
            controls += [control.previous_controlnet, *control.multigpu_clones.values()]
            control_model = getattr(control, "control_model", None)
            if control_model is not None and self.adapter.own_embedder(control_model):
                found.setdefault(id(control_model), control_model)
        return list(found.values())

    def report(self, height, width, scale, dype_exponent):
        """One console line per latent size: whether DyPE is on for it and with what."""
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
    patched.remove_wrappers_with_key(WrappersMP.CALC_COND_BATCH, WRAPPER_KEY)
    patched.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, Wrapper(entry, s))
    return patched

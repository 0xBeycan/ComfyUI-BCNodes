"""DyPE (Dynamic Position Extrapolation) on a model: the one implementation DyPE Patcher and ZiT Dual Sampler call.

A model of a registered DyPE adapter (the dype family, models/common/registry.py: Z-Image today) is cloned with an
APPLY_MODEL wrapper (ComfyUI's patcher wrappers). At each model call the wrapper reads the latent's size and the
call's sigma, and when DyPE is on for that size it sets the adapter's embedder for that call on the diffusion
model, in place of its rotary embedder, and puts the embedder back when the call returns (or raises). Nothing is
left on the model between calls: an object patch would stay on the shared diffusion model after a run, since core
detaches a clone of the same model with unpatch_all=False (comfy/model_management.py load_models_gpu), and the
next run of another clone (the first pass's model) would run with it. The model's model_sampling, and so its
shift, is never touched.

When DyPE is on, per call: the scale is the latent's longer side in px over base_resolution; auto mode leaves a
latent whose longer side is auto_threshold px or less alone, and any mode one not larger than base_resolution.
dype_exponent, unless given, follows the scale (exponent()). The values come from the adapter (method,
base_resolution, dype_scale) and the module (auto_threshold), each replaced by DyPE Config's edited or linked
field when one is given.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Protocol, TypedDict

from ...libs.dype.rope import METHODS
from . import registry

MODES = ("auto", "enable", "disable")
AUTO_THRESHOLD = 1920  # px: auto mode turns DyPE on for a latent whose longer side is over it
WRAPPER_KEY = "bcnodes_dype"
PREFIX = "[BCNodes] DyPE:"


class ConfigValues(TypedDict, total=False):
    """The BC_DYPE_CONFIG value: DyPE Config's edited and linked fields only; {} = every value automatic. The
    keys are its fields in widget order, each with the type its value leaves the node with."""
    method: str
    base_resolution: int
    dype_scale: float
    dype_exponent: float
    auto_threshold: int


class Adapter(Protocol):
    """An entry of the dype family: one model's DyPE."""
    name: str
    attribute: str       # the diffusion model's rotary embedder attribute, swapped for one call
    latent_scale: int    # px per latent cell
    defaults: dict       # method, base_resolution, dype_scale for this model

    def matches(self, model) -> bool:
        """Whether the ModelPatcher `model` is this adapter's model."""

    def embedder(self, diffusion_model, method, scale, t, start_sigma, dype_scale, dype_exponent):
        """The rotary embedder (a torch module) of one call of `diffusion_model`."""


@dataclass(frozen=True)
class Settings:
    mode: str                         # auto or enable
    method: str
    base_resolution: int              # px
    dype_scale: float                 # DyPE's lambda_s
    dype_exponent: Optional[float]    # DyPE's lambda_t; None: from the scale
    auto_threshold: int               # px
    start_sigma: float                # in [0.001, 1]


def exponent(scale):
    """dype_exponent from the scale (the longer side over base_resolution), as DyPE's README advises: 2.0 at 4x
    (4K at 1024) and above, 1.0 from 2x (2K to 3K), 0.5 below (just above the base resolution)."""
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


def settings(entry, mode, start_sigma, config):
    """The Settings of `entry`'s model for `mode`, with DyPE Config's values (ConfigValues) over the automatic ones.
    start_sigma is clamped to [0.001, 1], as the original."""
    values = {**entry.defaults, "dype_exponent": None, "auto_threshold": AUTO_THRESHOLD, **(config or {})}
    return Settings(mode=mode, start_sigma=max(0.001, min(1.0, float(start_sigma))), **values)


class Wrapper:
    """The APPLY_MODEL wrapper: (executor, x, t, c_concat, c_crossattn, control, transformer_options, ...), as
    BaseModel.apply_model; x the latent of the call, t its sigma."""

    def __init__(self, entry, settings):
        self.adapter = entry
        self.settings = settings
        self.reported = set()  # the (height, width) already logged

    def __call__(self, executor, x, t, *args, **kwargs):
        s = self.settings
        height, width = x.shape[-2] * self.adapter.latent_scale, x.shape[-1] * self.adapter.latent_scale
        longer = max(height, width)
        scale = longer / s.base_resolution
        if (s.mode == "auto" and longer <= s.auto_threshold) or scale <= 1.0:
            self.report(height, width, None, None)
            return executor(x, t, *args, **kwargs)
        model = executor.class_obj  # the BaseModel
        sigma = min(max(float(t.flatten()[0]) / float(model.model_sampling.sigma_max), 0.0), 1.0)
        dype_exponent = exponent(scale) if s.dype_exponent is None else s.dype_exponent
        self.report(height, width, scale, dype_exponent)
        diffusion_model = model.diffusion_model
        held = getattr(diffusion_model, self.adapter.attribute)
        setattr(diffusion_model, self.adapter.attribute, self.adapter.embedder(
            diffusion_model, s.method, scale, sigma, s.start_sigma, s.dype_scale, dype_exponent))
        try:
            return executor(x, t, *args, **kwargs)
        finally:
            setattr(diffusion_model, self.adapter.attribute, held)

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
        logging.info("%s %s %dx%d px: on, scale %.3f of %d px, %s, dype_scale %g, dype_exponent %g (%s), start sigma "
                     "%.4f", PREFIX, self.adapter.name, width, height, scale, s.base_resolution, s.method, s.dype_scale,
                     dype_exponent, source, s.start_sigma)


def patch(model, mode, start_sigma, config):
    """`model` (a ModelPatcher) itself for mode disable; otherwise a clone carrying DyPE (one DyPE: an earlier one on
    the model is replaced). ValueError for a model no adapter takes."""
    if mode == "disable":
        return model
    if mode not in MODES:
        raise ValueError(f"DyPE: unknown mode {mode!r}; the modes are {', '.join(MODES)}")
    entry = adapter(model)
    s = settings(entry, mode, start_sigma, config)
    if s.method not in METHODS:
        raise ValueError(f"DyPE: unknown method {s.method!r}; the methods are {', '.join(METHODS)}")
    from comfy.patcher_extension import WrappersMP

    patched = model.clone()
    patched.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, WRAPPER_KEY)
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, WRAPPER_KEY, Wrapper(entry, s))
    return patched

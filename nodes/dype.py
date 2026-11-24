"""BC_DyPEPatcher (DyPE Patcher), BC_DyPEConfig (DyPE Config)

DyPE Patcher: MODEL -> a clone with DyPE (Dynamic Position Extrapolation) on its rotary position embedding, against
the content repeating itself when a model samples above its training resolution. The model is recognised and its own
values taken (Z-Image: base_resolution 1024; Qwen-Image: 1328; vision_yarn); the image size is read from the latent
of each sampling step, and a ControlNet of the step takes the same DyPE; the model's shift is never changed. The work
is models/common/dype.py, which Dual Sampler's second pass calls too.

DyPE Config: the values DyPE Patcher and Dual Sampler take automatically for the model selected on it, replaced by
the fields changed on this node or fed by a link (the others stay automatic): a BC_DYPE_CONFIG. A model of another
kind than the one selected is an error.
"""

from .common import LINK_INPUTS, edited_fields

MODE_TOOLTIP = ("auto: DyPE only for a latent whose longer side is over the model's auto_threshold (1920 px for Z-Image "
                "and Qwen-Image; DyPE Config can set it); below it the model runs unchanged. enable: DyPE for every "
                "latent larger than base_resolution. disable: the model is returned as it is.")


class DyPEPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        from ..models.common.dype import MODES

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model (Z-Image or Qwen-Image). The input model is not changed; the "
                                    "output is a patched clone. Its shift (ModelSamplingAuraFlow or the model's own) is "
                                    "kept."}),
                "mode": (list(MODES), {"default": "auto", "tooltip": MODE_TOOLTIP}),
                "dype_start_sigma": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip":
                                     "The sigma below which DyPE's attention scale starts to decay: the first sigma of "
                                     "the sampler run this model is used in (for a second pass at a denoise below 1, "
                                     "the first value of that run's schedule). 1.0: from the start of a full run."}),
            },
            "optional": {
                "dype_config": ("BC_DYPE_CONFIG", {"tooltip": "DyPE Config: the automatic values it changes. Its model must "
                                "be this model's kind."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "dype", "dynamic position extrapolation", "rope", "high resolution", "hires",
                      "z-image", "qwen-image", "repetition"]
    DESCRIPTION = ("DyPE (Dynamic Position Extrapolation) for sampling above the model's training resolution: the "
                   "rotary positions are extrapolated with a strength that follows the sigma, which stops the scene "
                   "repeating itself in a large image or a 2x hires pass. Takes the model's own method and values "
                   "(vision_yarn; base_resolution Z-Image 1024, Qwen-Image 1328; dype_exponent from the scale) and the "
                   "image size from each sampling step's latent; a ControlNet in the conditioning takes the same DyPE. "
                   "The shift is never changed. Z-Image and Qwen-Image (2512).")

    def patch(self, model, mode, dype_start_sigma, dype_config=None):
        # as the pack's other MODEL nodes: nothing to patch
        if model is None:
            return (model,)
        from ..models.common import dype

        return (dype.patch(model, mode, dype_start_sigma, dype_config),)


class DyPEConfig:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs.dype.rope import METHODS
        from ..models.common import registry

        names = registry.names(registry.DYPE)
        # each model's automatic values: web/js/dype.js shows the selected model's in the fields not changed
        automatic = {name: dict(registry.get(registry.DYPE, name).defaults) for name in names}
        return {
            "required": {
                "model": (names, {"default": names[0], "automatic": automatic, "tooltip":
                          "The model kind these values are for: the fields not changed show its automatic values. "
                          "DyPE Patcher or Dual Sampler with a model of another kind stops with an error."}),
                "method": (list(METHODS), {"default": METHODS[0], "tooltip":
                           "vision_yarn: YaRN's interpolation per frequency band with DyPE's sigma-driven ramps and "
                           "attention scale (the models' default). yarn: classic YaRN at the stretched length, with "
                           "the same modulation. ntk: NTK-aware scaling, strongest at high sigma. pi: position "
                           "interpolation, the positions squeezed most at high sigma (DY-PI). base: static NTK "
                           "scaling, no change over the run."}),
                "base_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip":
                                    "The model's training resolution, px; the scale is the latent's longer side "
                                    "over it (automatic: Z-Image 1024, Qwen-Image 1328)."}),
                "dype_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip":
                               "DyPE's magnitude (lambda_s): the modulation exponent is dype_scale x t^dype_exponent."}),
                "dype_exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1, "tooltip":
                                  "How fast DyPE's effect decays over the run (lambda_t; 2: quadratic). Not changed: "
                                  "from the scale, 2.0 at 4x base_resolution and above, 1.0 from 2x, 0.5 below "
                                  "(Z-Image: 4096 / 2048 px; Qwen-Image: 5312 / 2656 px)."}),
                "auto_threshold": ("INT", {"default": 1920, "min": 0, "max": 16384, "step": 16, "tooltip":
                                   "auto mode: DyPE turns on for a latent whose longer side is over this many px "
                                   "(automatic: the model's own, 1920 for Z-Image and Qwen-Image)."}),
                "yarn_alt_scaling": ("BOOLEAN", {"default": False, "tooltip":
                                     "yarn only, Qwen-Image only: each image axis stretched to its own length "
                                     "(anisotropic; an axis not over base_resolution stays plain) in place of both to "
                                     "the longer side. Z-Image's yarn has no such option, as the original."}),
                "edited": ("STRING", {"default": "",
                                      "tooltip": "The fields changed on this node, comma-separated (kept by the node, "
                                                 "hidden there). Only these, and the fields fed by a link, are "
                                                 "applied; the others keep the automatic values."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("BC_DYPE_CONFIG",)
    RETURN_NAMES = ("dype_config",)
    FUNCTION = "build"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "dype config", "dype", "rope", "vision yarn", "yarn", "ntk"]
    DESCRIPTION = ("DyPE's values for DyPE Patcher and Dual Sampler, for the model selected in `model`. Only the "
                   "fields you change, or feed by a link, are applied; every other field keeps that model's automatic "
                   "value (vision_yarn, base_resolution Z-Image 1024 / Qwen-Image 1328, dype_scale 2.0, dype_exponent "
                   "from the scale, auto_threshold 1920), shown greyed on the classic canvas; selecting another model "
                   "shows its values. The node's menu item Reset to automatic values clears the changes. A model of "
                   "another kind than the one selected is an error. The shift is not a DyPE value: it stays the "
                   "model's.")

    def build(self, model, method, base_resolution, dype_scale, dype_exponent, auto_threshold, yarn_alt_scaling, edited,
              prompt_graph=None, unique_id=None):
        from ..models.common.dype import Config, ConfigValues

        fields = ConfigValues.__annotations__  # in widget order, each with the type its value leaves with
        values = {"method": method, "base_resolution": base_resolution, "dype_scale": dype_scale,
                  "dype_exponent": dype_exponent, "auto_threshold": auto_threshold, "yarn_alt_scaling": yarn_alt_scaling}
        names = edited_fields("DyPE Config", edited, fields, prompt_graph, unique_id)
        return (Config(model=model, values=ConfigValues(**{name: fields[name](values[name]) for name in names})),)


NODE_CLASS_MAPPINGS = {
    "BC_DyPEPatcher": DyPEPatcher,
    "BC_DyPEConfig": DyPEConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DyPEPatcher": "DyPE Patcher",
    "BC_DyPEConfig": "DyPE Config",
}

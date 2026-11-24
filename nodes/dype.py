"""BC_DyPEPatcher (DyPE Patcher), BC_DyPEConfig (DyPE Config)

DyPE Patcher: MODEL -> a clone with DyPE (Dynamic Position Extrapolation) on its rotary position embedding, against
the content repeating itself when a model samples above its training resolution. The model is recognised and its own
values taken (Z-Image: base_resolution 1024, vision_yarn); the image size is read from the latent of each sampler
run; the model's shift is never changed. The work is models/common/dype.py, which ZiT Dual Sampler's second pass
calls too.

DyPE Config: the values DyPE Patcher and ZiT Dual Sampler take automatically, replaced by the fields changed on this
node or fed by a link (the others stay automatic): a BC_DYPE_CONFIG.
"""

from .common import LINK_INPUTS, edited_fields

MODE_TOOLTIP = ("auto: DyPE only for a latent whose longer side is over auto_threshold (1920 px unless DyPE Config "
                "sets it); below it the model runs unchanged. enable: DyPE for every latent larger than "
                "base_resolution. disable: the model is returned as it is.")


class DyPEPatcher:
    @classmethod
    def INPUT_TYPES(cls):
        from ..models.common.dype import MODES

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model (Z-Image). The input model is not changed; the output is a "
                                    "patched clone. Its shift (ModelSamplingAuraFlow or the model's own) is kept."}),
                "mode": (list(MODES), {"default": "auto", "tooltip": MODE_TOOLTIP}),
                "dype_start_sigma": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1.0, "step": 0.001, "tooltip":
                                     "The sigma below which DyPE's attention scale starts to decay: the first sigma of "
                                     "the sampler run this model is used in (for a second pass at a denoise below 1, "
                                     "the first value of that run's schedule). 1.0: from the start of a full run."}),
            },
            "optional": {
                "dype_config": ("BC_DYPE_CONFIG", {"tooltip": "DyPE Config: the automatic values it changes."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "dype", "dynamic position extrapolation", "rope", "high resolution", "hires",
                      "z-image", "repetition"]
    DESCRIPTION = ("DyPE (Dynamic Position Extrapolation) for sampling above the model's training resolution: the "
                   "rotary positions are extrapolated with a strength that follows the sigma, which stops the scene "
                   "repeating itself in a large image or a 2x hires pass. Takes the model's own method and values "
                   "(Z-Image: vision_yarn, base_resolution 1024; dype_exponent from the scale) and the image size from "
                   "each sampler run's latent. The shift is never changed. Z-Image only for now.")

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
        from ..models.common.dype import AUTO_THRESHOLD

        return {
            "required": {
                "method": (list(METHODS), {"default": METHODS[0], "tooltip":
                           "vision_yarn: YaRN's interpolation per frequency band with DyPE's sigma-driven ramps and "
                           "attention scale (Z-Image's default). yarn: classic YaRN at the stretched length, with the "
                           "same modulation. ntk: NTK-aware scaling, strongest at high sigma. pi: position "
                           "interpolation, the positions squeezed most at high sigma (DY-PI). base: static NTK "
                           "scaling, no change over the run."}),
                "base_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 16, "tooltip":
                                    "The model's training resolution, px; the scale is the latent's longer side "
                                    "over it (Z-Image: 1024)."}),
                "dype_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 8.0, "step": 0.1, "tooltip":
                               "DyPE's magnitude (lambda_s): the modulation exponent is dype_scale x t^dype_exponent."}),
                "dype_exponent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1, "tooltip":
                                  "How fast DyPE's effect decays over the run (lambda_t; 2: quadratic). Not changed: "
                                  "from the scale, 2.0 at 4x base_resolution and above (4K), 1.0 from 2x (2K to 3K), "
                                  "0.5 below."}),
                "auto_threshold": ("INT", {"default": AUTO_THRESHOLD, "min": 0, "max": 16384, "step": 16, "tooltip":
                                   "auto mode: DyPE turns on for a latent whose longer side is over this many px."}),
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
    DESCRIPTION = ("DyPE's values for DyPE Patcher and ZiT Dual Sampler. Only the fields you change, or feed by a "
                   "link, are applied; every other field keeps the model's automatic value (Z-Image: vision_yarn, "
                   "base_resolution 1024, dype_scale 2.0, dype_exponent from the scale, auto_threshold 1920), drawn "
                   "greyed on the classic canvas. The node's menu item Reset to automatic values clears the changes. "
                   "The shift is not a DyPE value: it stays the model's.")

    def build(self, method, base_resolution, dype_scale, dype_exponent, auto_threshold, edited, prompt_graph=None,
              unique_id=None):
        from ..models.common.dype import ConfigValues

        fields = ConfigValues.__annotations__  # in widget order, each with the type its value leaves with
        values = {"method": method, "base_resolution": base_resolution, "dype_scale": dype_scale,
                  "dype_exponent": dype_exponent, "auto_threshold": auto_threshold}
        names = edited_fields("DyPE Config", edited, fields, prompt_graph, unique_id)
        return (ConfigValues(**{name: fields[name](values[name]) for name in names}),)


NODE_CLASS_MAPPINGS = {
    "BC_DyPEPatcher": DyPEPatcher,
    "BC_DyPEConfig": DyPEConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DyPEPatcher": "DyPE Patcher",
    "BC_DyPEConfig": "DyPE Config",
}

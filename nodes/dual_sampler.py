"""BC_DualSampler (Dual Sampler)

Two sampling passes in one node, for any model: a first pass, the latent upscaled (core's latent upscale, a
decode -> resize -> encode, or a blend of the two), and a second pass at a denoise with DyPE on its model only (the
model's own DyPE: Z-Image, Qwen-Image; dype disable for another model), its start sigma read from the second pass's
own schedule. Outputs both decoded images and the second pass's latent. The defaults are Z-Image Turbo's. The flow is
pipelines/dual_sampler.py; DyPE is models/common/dype.py, as DyPE Patcher's.
"""

from .common import LINK_INPUTS, drop_unwanted, heavy_wanted, wants

DEFAULT_SAMPLER = "dpmpp_sde"
DEFAULT_SCHEDULER = "beta"


def pass_inputs(n, sampler_names, schedulers):
    """steps / cfg / sampler / scheduler of pass n, Z-Image Turbo's values as defaults."""
    which = "first" if n == 1 else "second"
    return {
        f"steps_{n}": ("INT", {"default": 9, "min": 1, "max": 10000, "tooltip": f"Steps of the {which} pass."}),
        f"cfg_{n}": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                               "tooltip": f"CFG of the {which} pass (Z-Image Turbo: 1.0, no negative pass)."}),
        f"sampler_{n}": (sampler_names, {"default": DEFAULT_SAMPLER, "tooltip": f"Sampler of the {which} pass."}),
        f"scheduler_{n}": (schedulers, {"default": DEFAULT_SCHEDULER, "tooltip": f"Scheduler of the {which} pass."}),
    }


class DualSampler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        from ..models.common.dype import MODES

        sampler_names, schedulers = comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
        return {
            "required": {
                "model_1": ("MODEL", {"tooltip": "The first pass's model (e.g. with a ControlNet model patch)."}),
                "model_2": ("MODEL", {"tooltip": "The second pass's model; DyPE is put on a clone of it for that pass."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {"tooltip": "The first pass's latent (e.g. an empty latent)."}),
                "vae": ("VAE", {"tooltip": "Decodes both passes; encodes the pixel way of the upscale."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True,
                                 "tooltip": "The first pass's seed; second_seed sets the second's."}),
                **pass_inputs(1, sampler_names, schedulers),
                **pass_inputs(2, sampler_names, schedulers),
                "denoise_2": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                              "The second pass's denoise. The first pass always runs at 1.0."}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"], {"default": "bislerp",
                                   "tooltip": "The latent way: core's Upscale Latent By with this method."}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip":
                             "The upscale factor; each side is then rounded to the nearest multiple of 16 px."}),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip":
                                 "The pixel way's weight: 0 is the latent upscale alone (no decode or encode), 1 the "
                                 "decode -> resize -> encode alone, between them core's Latent Blend of the two."}),
                "pixel_upscale_method": (["lanczos", "bicubic"], {"default": "lanczos", "tooltip":
                                         "The pixel way's resize (lanczos as core's: PIL on 8 bits)."}),
                "second_seed": (["fixed", "increment"], {"default": "increment", "tooltip":
                                "The second pass's seed: the seed itself, or seed + 1."}),
                "dype": (list(MODES), {"default": "auto", "tooltip":
                         "DyPE on the second pass only (DyPE Patcher's modes; Z-Image and Qwen-Image), its start sigma "
                         "the first sigma of that pass's schedule. auto: only for a second latent whose longer side is "
                         "over the model's auto_threshold (1920 px for both; DyPE Config can set it). disable: model_2 "
                         "as it is, for any model; any other mode on a model DyPE does not take stops the run before "
                         "the first pass."}),
            },
            "optional": {
                "dype_config": ("BC_DYPE_CONFIG", {"tooltip": "DyPE Config: the automatic DyPE values it changes. Its model "
                                "must be model_2's kind."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("first_image", "second_image", "latent")
    # nodes/common.py: an image nothing links is not decoded (the first one still is for the pixel way)
    HEAVY_OUTPUTS = ("first_image", "second_image")
    FUNCTION = "sample"
    CATEGORY = "BCNodes/sampling"
    SEARCH_ALIASES = ["BCNodes", "dual sampler", "zit dual sampler", "z-image turbo", "qwen-image", "hires fix",
                      "two pass", "ksampler", "dype", "upscale"]
    DESCRIPTION = ("A two-pass hires fix in one node, for any model: the first pass (denoise 1.0), the latent upscaled "
                   "by scale_by to a multiple of 16 px (core's latent upscale, a decode -> resize -> encode, or "
                   "blend_factor of the two), and the second pass at denoise_2 with DyPE on its model only (Z-Image, "
                   "Qwen-Image; start sigma from that pass's schedule; dype disable for other models). The same as the "
                   "core chain KSampler -> Upscale Latent By -> KSampler -> VAE Decode x2 at blend_factor 0. Outputs "
                   "both images and the second latent. The defaults are Z-Image Turbo's.")

    def sample(self, model_1, model_2, positive, negative, latent_image, vae, seed, steps_1, cfg_1, sampler_1,
               scheduler_1, steps_2, cfg_2, sampler_2, scheduler_2, denoise_2, upscale_method, scale_by, blend_factor,
               pixel_upscale_method, second_seed, dype, dype_config=None, prompt_graph=None, unique_id=None):
        from ..pipelines.dual_sampler import Pass, Settings, run

        settings = Settings(seed, Pass(steps_1, cfg_1, sampler_1, scheduler_1, 1.0),
                            Pass(steps_2, cfg_2, sampler_2, scheduler_2, denoise_2), upscale_method, scale_by,
                            blend_factor, pixel_upscale_method, second_seed, dype, dype_config)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), run(model_1, model_2, positive, negative, latent_image, vae, settings,
                                             want_first=wants(wanted, "first_image"),
                                             want_second=wants(wanted, "second_image")), wanted)


NODE_CLASS_MAPPINGS = {
    "BC_DualSampler": DualSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DualSampler": "Dual Sampler",
}

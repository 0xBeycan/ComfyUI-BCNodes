"""BC_DualSampler (Dual Sampler)

Two sampling passes in one node, for any model: a first pass with `model`, the latent upscaled (core's latent upscale,
a decode -> resize -> encode, or a blend of the two), and a second pass at a denoise with `model_2` (`model` when it is
not linked) and DyPE on that pass only (the model's own DyPE: Z-Image, Qwen-Image; dype disable for another model), its
start sigma the sigma of the second pass's first model call (the run's schedule, after a per-run shift such as Qwen
Image Shift's dynamic one). Outputs both decoded images and the second pass's latent. The defaults are Z-Image
Turbo's. The flow is pipelines/dual_sampler.py; DyPE is models/common/dype.py, as DyPE Patcher's.
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
                "model": ("MODEL", {"tooltip": "The first pass's model, and the second pass's too when model_2 is not "
                          "linked."}),
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
                "dype": (["auto", *MODES], {"default": "auto", "tooltip":
                         "DyPE on the second pass only (Z-Image and Qwen-Image), its start sigma the sigma of that "
                         "pass's first model call (after a shift the model applies per run, such as Qwen Image Shift's "
                         "dynamic one). auto: enable when scale_by is over dype_auto_scale, disable otherwise. enable: "
                         "DyPE for a second latent larger than the model's base_resolution, as DyPE Patcher. disable: "
                         "the second pass's model as it is, for any model; enable on a model DyPE does not take stops "
                         "the run before the first pass."}),
            },
            "optional": {
                "model_2": ("MODEL", {"tooltip": "Optional: the second pass's own model (e.g. other LoRAs or "
                            "patches than model's). Not linked: model runs the second pass too. DyPE is put on a clone "
                            "of the second pass's model, never on model's first pass."}),
                "dype_config": ("BC_DYPE_CONFIG", {"tooltip": "DyPE Config: the automatic DyPE values it changes. Its "
                                "model must be the second pass's model's kind (model_2, or model when model_2 is not "
                                "linked)."}),
                # added after the node's release, optional so an API-format prompt saved without it still validates
                "dype_auto_scale": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 8.0, "step": 0.01, "tooltip":
                                    "dype auto: DyPE on the second pass when scale_by is over this. 1.25: Z-Image "
                                    "Turbo's 1.25x pass stays clean without DyPE, a 1.5x one repeats."}),
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
    DESCRIPTION = ("A two-pass hires fix in one node, for any model: the first pass with model (denoise 1.0), the "
                   "latent upscaled by scale_by to a multiple of 16 px (core's latent upscale, a decode -> resize -> "
                   "encode, or blend_factor of the two), and the second pass at denoise_2 with model_2 (model when it "
                   "is not linked) and DyPE on that pass only (Z-Image, Qwen-Image; start sigma: that pass's first "
                   "model call's; auto: when scale_by is over dype_auto_scale; dype disable for other models). The "
                   "same as the core chain KSampler -> Upscale Latent By -> KSampler -> VAE Decode x2 at blend_factor "
                   "0. Outputs both images and the second latent. The defaults are Z-Image Turbo's.")

    def sample(self, model, positive, negative, latent_image, vae, seed, steps_1, cfg_1, sampler_1, scheduler_1,
               steps_2, cfg_2, sampler_2, scheduler_2, denoise_2, upscale_method, scale_by, blend_factor,
               pixel_upscale_method, second_seed, dype, model_2=None, dype_config=None, dype_auto_scale=1.25,
               prompt_graph=None, unique_id=None):
        from ..pipelines.dual_sampler import Pass, Settings, run

        settings = Settings(seed, Pass(steps_1, cfg_1, sampler_1, scheduler_1, 1.0),
                            Pass(steps_2, cfg_2, sampler_2, scheduler_2, denoise_2), upscale_method, scale_by,
                            blend_factor, pixel_upscale_method, second_seed, dype, dype_auto_scale, dype_config)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), run(model, model if model_2 is None else model_2, positive, negative,
                                             latent_image, vae, settings, want_first=wants(wanted, "first_image"),
                                             want_second=wants(wanted, "second_image")), wanted)


NODE_CLASS_MAPPINGS = {
    "BC_DualSampler": DualSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DualSampler": "Dual Sampler",
}

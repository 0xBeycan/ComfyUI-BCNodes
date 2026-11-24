"""BC_LatentUpscale (Latent Upscale)

A latent upscaled by a weights file of a model of the latent upscale family (models/common/registry.py),
for a second sampling pass. Wan 2.1 2x: the latents of Wan 2.1, Wan 2.2 A14B and Qwen-Image, doubled on
height and width by a small network whose weights file is picked from models/latent_upscale_models. The
other fields of the latent are kept, as core's Upscale Latent keeps them.
"""


class LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        from ..models.common import registry
        from ..models.common.registry import LATENT_UPSCALE

        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to upscale, e.g. the first sampling pass's output."}),
                "model": (registry.names(LATENT_UPSCALE), {"default": "Wan 2.1 2x",
                                                           "tooltip": "The model the weights file is. Wan 2.1 2x: the latents "
                                                                      "of Wan 2.1, Wan 2.2 A14B and Qwen-Image, doubled on height "
                                                                      "and width. Sample the result again at a low denoise for "
                                                                      "the detail."}),
                "upscale_model_name": (folder_paths.get_filename_list("latent_upscale_models"),
                                       {"tooltip": "Weights file under models/latent_upscale_models."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "BCNodes/latent"
    SEARCH_ALIASES = ["BCNodes", "latent upscale", "upscale latent", "hires fix", "wan latent upscale", "qwen latent upscale"]
    DESCRIPTION = ("Upscales a latent with a model's weights file from models/latent_upscale_models: Wan 2.1 2x doubles "
                   "a Wan 2.1, Wan 2.2 A14B or Qwen-Image latent on height and width with a small network, much closer "
                   "to a real latent than bilinear or bislerp. Sample the result again at a low denoise (hires fix). A "
                   "file of another model is refused with an error.")

    def upscale(self, samples, model, upscale_model_name):
        import folder_paths

        from ..models.common import registry
        from ..models.common.registry import LATENT_UPSCALE

        path = folder_paths.get_full_path_or_raise("latent_upscale_models", upscale_model_name)
        out = samples.copy()  # as core's Upscale Latent: the other fields as they are
        out["samples"] = registry.get(LATENT_UPSCALE, model)(samples["samples"], path)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "BC_LatentUpscale": LatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_LatentUpscale": "Latent Upscale",
}

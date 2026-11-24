"""BC_VAELoader (VAE Loader)

A VAE file loaded for the model picked: each model of the VAE family (models/common/registry.py)
loads its files through core and does what that model's special VAEs need. Wan 2.1 (the latents of
Wan 2.1, Wan 2.2 A14B and Qwen-Image): a decoder that packs a larger image into its channels (the
Wan 2.1 upscale 2x VAE) decodes at that size through core's VAE Decode nodes. One console line when
the file needed anything.
"""


class VAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        from ..models.common import registry
        from ..models.common.registry import VAE

        return {
            "required": {
                "model": (registry.names(VAE), {"default": "Wan 2.1",
                                                "tooltip": "The model whose VAE the file is. Wan 2.1: the latents of Wan 2.1, "
                                                           "Wan 2.2 A14B and Qwen-Image; a decoder that packs a larger image "
                                                           "into its channels (Wan 2.1 upscale 2x) decodes at that size."}),
                "vae_name": (folder_paths.get_filename_list("vae"), {"tooltip": "VAE file under models/vae."}),
            },
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load"
    CATEGORY = "BCNodes/loaders"
    SEARCH_ALIASES = ["BCNodes", "vae loader", "load vae", "upscale vae", "2x vae", "wan vae", "qwen vae"]
    DESCRIPTION = ("Loads a VAE file for the model picked, as ComfyUI's Load VAE does, and handles that model's special "
                   "VAEs. Wan 2.1 (Wan 2.1, Wan 2.2 A14B, Qwen-Image): a decoder that packs a larger image into its "
                   "channels, such as the Wan 2.1 upscale 2x VAE, decodes at that size through ComfyUI's VAE Decode and "
                   "VAE Decode (Tiled). A file of another model is refused with an error.")

    def load(self, model, vae_name):
        import logging

        import folder_paths

        from ..models.common import registry
        from ..models.common.registry import VAE

        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae, note = registry.get(VAE, model)(path)
        if note:
            logging.info(f"[BCNodes] VAE Loader: {vae_name}: {note}")
        return (vae,)


NODE_CLASS_MAPPINGS = {
    "BC_VAELoader": VAELoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_VAELoader": "VAE Loader",
}

"""BC_LoraLoaderKeyFix (Lora Loader (Key Fix))

MODEL + one LoRA file -> MODEL, applied through core's own loader (`load_lora_for_models`, no CLIP)
after the keys core would leave out are renamed (libs/lora_keys.py: lightx2v's `.diff_m`
modulation differences, PEFT keys without the `diffusion_model.` prefix), by pipelines/lora.py, the
path Power Lora Loader's rows take too. The LoRA goes through core's format conversion first, so
the renames see the keys core's loader sees. One console line says how many keys were renamed and
how many still match no module of the model.
"""


class LoraLoaderKeyFix:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA is applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file under models/loras."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                       "tooltip": "How strongly the LoRA changes the model; 0 returns the model untouched."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "BCNodes/loaders"
    SEARCH_ALIASES = ["BCNodes", "lora loader", "lora key fix", "diff_m", "peft lora", "lightx2v", "svi"]
    DESCRIPTION = ("Applies a LoRA to the model with every tensor core can map. Keys ComfyUI's own loader leaves out "
                   "are renamed first: lightx2v's .diff_m modulation differences become .modulation.diff, and PEFT keys "
                   "without the diffusion_model. prefix (official SVI 2.0 files) get it. Keys core already maps are "
                   "untouched. The console says how many keys were renamed and how many still match no module of "
                   "the model (core lists those as 'lora key not loaded').")

    def load(self, model, lora_name, strength):
        # as core's LoRA loader and Power Lora Loader: nothing to apply to, or nothing to apply
        if model is None or strength == 0:
            return (model,)
        import folder_paths

        from ..pipelines.lora import apply_lora

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        model = apply_lora(model, path, strength, "Lora Loader (Key Fix)", lora_name, log_clean=True, with_metadata=True)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "BC_LoraLoaderKeyFix": LoraLoaderKeyFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_LoraLoaderKeyFix": "Lora Loader (Key Fix)",
}

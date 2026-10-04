"""BC_LoraLoaderKeyFix (Lora Loader (Key Fix))

MODEL + one LoRA file -> MODEL, applied through core's own loader (`load_lora_for_models`, no CLIP)
after the keys core would leave out are renamed (libs/lora_keys.py: lightx2v's `.diff_m`
modulation differences, PEFT keys without the `diffusion_model.` prefix). The LoRA goes through
core's format conversion first, so the renames see the keys core's loader sees. One console line
says how many keys were renamed and how many still match no module of the model.
"""

import logging


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
        import comfy.lora
        import comfy.lora_convert
        import comfy.sd
        import comfy.utils
        import folder_paths

        from ..libs.lora_keys import fix_keys

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        lora = comfy.lora_convert.convert_lora(lora)
        fix = fix_keys(lora.keys(), comfy.lora.model_lora_keys_unet(model.model, {}))
        lora = {fix.renamed.get(k, k): v for k, v in lora.items()}
        line = (f"[BCNodes] Lora Loader (Key Fix): {lora_name}: {len(lora)} keys, {len(fix.renamed)} renamed "
                f"({fix.modulation} .diff_m -> .modulation.diff, {fix.prefixed} given the diffusion_model. prefix), "
                f"{len(fix.unmapped)} match no module of this model")
        if fix.unmapped:
            shown = ", ".join(fix.unmapped[:5]) + (", ..." if len(fix.unmapped) > 5 else "")
            logging.warning("%s and are not applied: %s", line, shown)
        else:
            logging.info(line)
        model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0, lora_metadata=metadata)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "BC_LoraLoaderKeyFix": LoraLoaderKeyFix,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_LoraLoaderKeyFix": "Lora Loader (Key Fix)",
}

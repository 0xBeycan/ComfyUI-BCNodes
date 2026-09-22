"""BC_PowerLoraLoader (Power Lora Loader)

A MODEL in, any number of LoRA rows, a MODEL out. Each row is
{"on": bool, "lora": file name, "strength": float}; the rows are edited on
the node (web/js/power_lora_loader.js) and arrive as inputs lora_N through
the flexible optional mapping, in row order (the prompt keeps the widget
order and the numbers may have gaps). There is no CLIP in or out: LoRAs are
applied to the model only.
"""

import re

from .common import ANY, FlexibleOptionalInputType

_ROW = re.compile(r"^lora_(\d+)$")


def _lora_path(name):
    """Full path of a LoRA by its listed name, tolerant to a missing extension."""
    import folder_paths

    if not name:
        return None
    path = folder_paths.get_full_path("loras", name)
    if path is not None:
        return path
    for candidate in folder_paths.get_filename_list("loras"):
        if candidate == name or candidate.rsplit(".", 1)[0] == name or candidate.endswith("/" + name):
            return folder_paths.get_full_path("loras", candidate)
    return None


class PowerLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY, {"model": ("MODEL",)}),
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "load_loras"
    CATEGORY = "BCNodes/loaders"
    SEARCH_ALIASES = ['BCNodes', 'power lora loader', 'lora loader', 'multiple loras']

    def load_loras(self, model=None, **kwargs):
        if model is None:
            return (None,)
        rows = [v for k, v in kwargs.items() if _ROW.match(k) and isinstance(v, dict)]
        for row in rows:
            if not row.get("on") or not row.get("lora"):
                continue
            strength = float(row.get("strength", 1.0))
            if strength == 0:
                continue
            path = _lora_path(row["lora"])
            if path is None:
                print(f"[BCNodes] Power Lora Loader: LoRA not found, skipping: {row['lora']}")
                continue
            import comfy.sd
            import comfy.utils

            lora = comfy.utils.load_torch_file(path, safe_load=True)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
        return (model,)


NODE_CLASS_MAPPINGS = {
    "BC_PowerLoraLoader": PowerLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PowerLoraLoader": "Power Lora Loader",
}

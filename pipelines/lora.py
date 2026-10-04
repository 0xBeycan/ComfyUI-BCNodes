"""A LoRA file applied to a MODEL through ComfyUI core's own loader (`load_lora_for_models`, no
CLIP), after the keys core would leave out are renamed (libs/lora_keys.py). The LoRA goes through
core's format conversion first, so the renames see the keys core's loader sees. Lora Loader (Key
Fix) and each row of Power Lora Loader apply their LoRA here. ComfyUI is imported inside the
function.
"""

import logging

from ..libs.lora_keys import fix_keys


def apply_lora(model, path, strength, node, lora_name, *, log_clean, with_metadata):
    """`model` (a ModelPatcher) patched with the LoRA file at `path` at `strength`. With nothing
    to rename, core's loader gets the keys and tensors a plain core load hands it.

    One console line, `[BCNodes] <node>: <lora_name>: ...`, says how many keys were renamed and how
    many still match no module of the model (a warning naming the first five when there are any);
    with `log_clean` False it is printed only when a key was renamed or matches no module.
    `with_metadata` hands the file's metadata to core, which attaches it to the patched model (core's
    Get IC-LoRA Parameters reads it there); the patches do not depend on it."""
    import comfy.lora
    import comfy.lora_convert
    import comfy.sd
    import comfy.utils

    lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    lora = comfy.lora_convert.convert_lora(lora)
    fix = fix_keys(lora.keys(), comfy.lora.model_lora_keys_unet(model.model, {}))
    lora = {fix.renamed.get(k, k): v for k, v in lora.items()}
    if log_clean or fix.renamed or fix.unmapped:
        line = (f"[BCNodes] {node}: {lora_name}: {len(lora)} keys, {len(fix.renamed)} renamed "
                f"({fix.modulation} .diff_m -> .modulation.diff, {fix.prefixed} given the diffusion_model. prefix), "
                f"{len(fix.unmapped)} match no module of this model")
        if fix.unmapped:
            shown = ", ".join(fix.unmapped[:5]) + (", ..." if len(fix.unmapped) > 5 else "")
            logging.warning("%s and are not applied: %s", line, shown)
        else:
            logging.info(line)
    model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0,
                                             lora_metadata=metadata if with_metadata else None)
    return model

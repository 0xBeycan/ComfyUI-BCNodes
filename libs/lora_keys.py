"""LoRA keys ComfyUI core's loader would leave out, renamed to the names it maps.

Core applies a LoRA tensor only when its key is an entry of the key map it builds for the model
(`comfy.lora.model_lora_keys_unet`) followed by a suffix one of its formats reads
(`.lora_A.weight`, `.lora_B.default.weight`, `.diff`, ...); any other key is left out with a
"lora key not loaded" warning. Two exports miss the map:

- lightx2v's distill LoRAs store a block's modulation difference as `<block>.diff_m`; core reads a
  modulation difference as `<block>.modulation.diff` (the parameter is mapped under its own name);
- PEFT exports made outside ComfyUI (DiffSynth's, e.g. SVI 2.0) name the modules without the
  `diffusion_model.` prefix of core's map.

`first_names` is how a LoRA loader's warning or error names the keys or modules it is about (this
one's unmapped keys, libs/lm_lora.py's offenders).

Strings only: the caller passes the LoRA's keys and the key map; nothing here imports torch or
ComfyUI.
"""

from dataclasses import dataclass

SHOWN = 5  # names a message gives before ", ..."
MODEL_PREFIX = "diffusion_model."
DIFF_M = ".diff_m"
MODULATION_DIFF = ".modulation.diff"
# The one format of core whose suffix does not start with a dot: diffusers' attention-processor
# LoRA, `<entry>_lora.up.weight` / `<entry>_lora.down.weight`.
_PROCESSOR = "_lora."


@dataclass(frozen=True)
class KeyFix:
    renamed: dict    # old key -> new key, in the LoRA's order
    unmapped: list   # keys, after the renames, that no entry of the key map addresses
    modulation: int  # renames of .diff_m to .modulation.diff
    prefixed: int    # renames that added the diffusion_model. prefix


def addressed(key, key_map):
    """True when `key` is an entry of `key_map` followed by `.<suffix>` (or `_lora.<suffix>`).
    Whether core reads that suffix is its own call: a key of a mapped module in a format core does
    not know is still left out, with core's own warning."""
    cut = key.rfind(".")
    while cut > 0:
        if key[:cut] in key_map:
            return True
        cut = key.rfind(".", 0, cut)
    cut = key.find(_PROCESSOR)
    while cut > 0:
        if key[:cut] in key_map:
            return True
        cut = key.find(_PROCESSOR, cut + 1)
    return False


def fix_keys(keys, key_map):
    """The renames that let core map the keys it would leave out. A key `key_map` already
    addresses is never touched; `.diff_m` becomes `.modulation.diff`, and a key still unmapped
    without the `diffusion_model.` prefix gets it. A rename is kept only when the new key is
    mapped and no other tensor holds that name already (the LoRA's own or an earlier rename's), so
    no tensor is overwritten. The keys left unmapped are listed."""
    keys = list(keys)
    taken = set(keys)
    renamed, unmapped = {}, []
    modulation = prefixed = 0
    for key in keys:
        if addressed(key, key_map):
            continue
        new = key[:-len(DIFF_M)] + MODULATION_DIFF if key.endswith(DIFF_M) else key
        if not addressed(new, key_map) and not new.startswith(MODEL_PREFIX):
            new = MODEL_PREFIX + new
        if new == key or new in taken or not addressed(new, key_map):
            unmapped.append(key)
            continue
        renamed[key] = new
        taken.add(new)
        if key.endswith(DIFF_M):
            modulation += 1
        if not key.startswith(MODEL_PREFIX):
            prefixed += 1
    return KeyFix(renamed, unmapped, modulation, prefixed)


def first_names(names):
    """The first SHOWN of `names` (a list) joined by ", ", then ", ..." when there are more."""
    return ", ".join(names[:SHOWN]) + (", ..." if len(names) > SHOWN else "")

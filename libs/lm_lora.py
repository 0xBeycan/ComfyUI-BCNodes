"""LoRA files for the LM nodes, turned into the tensors ComfyUI core applies to a CLIP
(`comfy.sd.load_lora_for_models(None, clip, tensors, 0, strength)`).

Core's CLIP key map (`comfy.lora.model_lora_keys_clip`) addresses each LM weight as
`text_encoders.transformer.model.<module>` and each vision weight as
`text_encoders.transformer.visual.<module>` (also `text_encoders.<clip>.transformer.*` and
`lora_te_<module, dots as underscores>`). For an addressed module core reads `<module>.lora_B.weight`
(up), `<module>.lora_A.weight` (down) and the scale from a `<module>.alpha` tensor: alpha / rank, rank
= lora_A's first dimension; with no alpha tensor the scale is 1.0. Core checks nothing more: a module
the model lacks is skipped with a warning, a shape that does not fit only logs an error at patch time.
So the alpha is resolved here (a PEFT LoRA's is never guessed), `plan_lora` refuses an adapter a plain
patch does not reproduce (UNSUPPORTED, BASE_REWRITING_INITS) and `check_base` refuses a LoRA of another
base model, and one on embed_tokens / lm_head of a model whose head core ties to the embedding.

FORMATS lists the layouts read here; a new layout is one entry:
- `peft`: what PEFT saves (Unsloth, TRL, Axolotl, LLaMA-Factory and ms-swift all save it). Module
  names `base_model.model.<HF module>`; alpha in adapter_config.json, never in the file.
- `comfy`: the names core's key map holds already; module names pass through. With no alpha anywhere
  the alpha is the rank, core's own convention (scale 1.0; core's LoRA extraction writes no alpha).

Torch at module level; safetensors inside the function that reads a file.
"""

import json
import math
import os
import re
from dataclasses import dataclass

import torch

from .files import walk_once
from .lora_keys import first_names

PEFT_CONFIG = "adapter_config.json"
PEFT_WEIGHTS = "adapter_model.safetensors"
PATTERN_MATCH = r"(.*\.)?({})$"  # an alpha_pattern / rank_pattern key as PEFT matches it against a module name
RANK_ALPHA = "rank"  # LoraPlan.alpha_source of a file in core's names without alpha: alpha = rank, core's scale 1.0
# A tied output head (check_base): the embedding's state-dict key with no head key beside it; a plan module
# for the head has no key map entry then, so it is told by these name endings (core's dotted and lora_te_ forms).
EMBED_WEIGHT, HEAD_WEIGHT = ".transformer.model.embed_tokens.weight", ".transformer.model.lm_head.weight"
HEAD_MODULES = (".model.lm_head", "lora_te_lm_head")

# Roles of a stored tensor. A BASE tensor is the frozen base weight PEFT saves next to a LoRA on
# embed_tokens / lm_head (save_embedding_layers="auto"): a copy of the model's own, left out unread.
UP, DOWN, ALPHA, BASE = "up", "down", "alpha", "base"
CORE_SUFFIX = {UP: "lora_B.weight", DOWN: "lora_A.weight", ALPHA: "alpha"}

# adapter_config.json fields whose adapters a plain lora_A / lora_B patch does not reproduce. A field
# is off when false, null or empty, as PEFT tests it; a SUB_CONFIGS field is on for any value but null
# or false (PEFT builds the sub-config from it: {} is its defaults).
UNSUPPORTED = (
    ("use_dora", "DoRA (use_dora)"),
    ("modules_to_save", "fully trained modules (modules_to_save)"),
    ("trainable_token_indices", "trained token rows (trainable_token_indices)"),
    ("lora_bias", "LoRA biases (lora_bias)"),
    ("fan_in_fan_out", "transposed weights (fan_in_fan_out)"),
    ("use_qalora", "QA-LoRA (use_qalora)"),
    ("layer_replication", "replicated layers (layer_replication)"),
    ("alora_invocation_tokens", "activated LoRA (alora_invocation_tokens)"),
    ("use_bdlora", "block-diagonal factors (use_bdlora)"),
    ("arrow_config", "Arrow routing over several LoRAs (arrow_config)"),
    ("kasa_config", "KaSA, which also truncates the base weights (kasa_config)"),
)
SUB_CONFIGS = {"use_bdlora", "arrow_config", "kasa_config"}
# init_lora_weights values (prefixes, as PEFT tests them) whose initialisation rewrites the base weights:
# the trained factors fit only that rewritten model until PEFT converts them to a plain LoRA.
BASE_REWRITING_INITS = ("pissa", "olora", "corda", "astra", "lora_ga", "loftq")


class LoraError(ValueError):
    """What is wrong with a LoRA and how to fix it."""


@dataclass(frozen=True)
class LoraFormat:
    name: str
    marks: tuple    # key prefixes that make a key this format's
    modules: tuple  # (stored prefix, core prefix) on `<module>.`; the first that matches is replaced
    tensors: tuple  # (stored suffix, role, transposed)


FORMATS = (
    LoraFormat(
        "peft",
        marks=("base_model.model.",),
        modules=(("base_model.model.model.language_model.", "text_encoders.transformer.model."),  # ...ForConditionalGeneration
                 ("base_model.model.model.visual.", "text_encoders.transformer.visual."),
                 ("base_model.model.model.", "text_encoders.transformer.model."),  # ...ForCausalLM
                 ("base_model.model.lm_head.", "text_encoders.transformer.model.lm_head.")),
        tensors=(("lora_A.weight", DOWN, False), ("lora_B.weight", UP, False),
                 ("lora_A.default.weight", DOWN, False), ("lora_B.default.weight", UP, False),
                 # nn.Embedding: A (rank, vocab), B (hidden, rank), delta (B @ A).T = A.T @ B.T
                 ("lora_embedding_A", UP, True), ("lora_embedding_B", DOWN, True),
                 ("lora_embedding_A.default", UP, True), ("lora_embedding_B.default", DOWN, True),
                 ("base_layer.weight", BASE, False),
                 ("alpha", ALPHA, False))),
    LoraFormat(
        "comfy",
        marks=("text_encoders.", "lora_te_"),
        modules=(("text_encoders.", "text_encoders."), ("lora_te_", "lora_te_")),
        tensors=(("lora_A.weight", DOWN, False), ("lora_B.weight", UP, False),
                 ("lora_down.weight", DOWN, False), ("lora_up.weight", UP, False),
                 ("lora_A.default.weight", DOWN, False), ("lora_B.default.weight", UP, False),
                 ("alpha", ALPHA, False))),
)


@dataclass(frozen=True)
class LoraSource:
    path: str          # the .safetensors file or the PEFT folder
    tensors: dict      # key -> torch.Tensor as stored; None for a full base weight (left out, never read)
    metadata: dict     # safetensors __metadata__, {} when none
    config: "dict | None"  # sidecar / adapter_config.json, None when absent
    config_path: "str | None"


@dataclass(frozen=True)
class LoraPlan:
    tensors: dict      # core-ready: `<module>.lora_A.weight` / `.lora_B.weight` + `<module>.alpha` float32 scalars
    modules: tuple     # core module keys, e.g. 'text_encoders.transformer.model.layers.0.mlp.down_proj'
    format: str        # 'peft' | 'comfy'
    alpha_source: str  # 'tensors' | 'metadata' | the config file name | RANK_ALPHA
    notes: tuple       # human-readable lines for the console


def read_lora(path):
    """The LoRA at `path`: a .safetensors file, or a PEFT folder holding adapter_model.safetensors
    (and adapter_config.json). The config of a single file is `<stem>.json` next to it, or, for an
    adapter_model.safetensors, the adapter_config.json beside it."""
    if os.path.isdir(path):
        weights = os.path.join(path, PEFT_WEIGHTS)
        if not os.path.isfile(weights):
            pickled = os.path.isfile(os.path.join(path, "adapter_model.bin"))
            raise LoraError(f"{path}: this folder holds no {PEFT_WEIGHTS}"
                            + (" (only adapter_model.bin: save the adapter again with safe_serialization=True)"
                               if pickled else "; a LoRA is a .safetensors file or a PEFT folder holding "
                               f"{PEFT_WEIGHTS} and {PEFT_CONFIG}"))
        config_path = os.path.join(path, PEFT_CONFIG)
    elif os.path.isfile(path) and path.lower().endswith(".safetensors"):
        weights = path
        config_path = os.path.splitext(path)[0] + ".json"
        if not os.path.isfile(config_path) and os.path.basename(path) == PEFT_WEIGHTS:
            config_path = os.path.join(os.path.dirname(path), PEFT_CONFIG)
    elif not os.path.exists(path):
        raise LoraError(f"LoRA not found: {path}")
    else:
        raise LoraError(f"{path}: not a LoRA; pick a .safetensors file or a PEFT folder holding "
                        f"{PEFT_WEIGHTS} and {PEFT_CONFIG}")
    from safetensors import SafetensorError, safe_open
    try:
        with safe_open(weights, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
            tensors = {}
            for key in f.keys():  # noqa: SIM118 (a safe_open handle is not iterable)
                part = _parse(key)
                # a base weight is a copy of the model's own (an embedding's: 1 G values on a 9B model): not read
                tensors[key] = None if part and part[1] == BASE else f.get_tensor(key)
    except SafetensorError as e:
        raise LoraError(f"{weights}: not a readable safetensors file ({e}); download or export it again") from e
    if not os.path.isfile(config_path):
        return LoraSource(path, tensors, metadata, None, None)
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except ValueError as e:
        raise LoraError(f"{config_path}: not valid JSON ({e}); fix or remove it") from e
    if not isinstance(config, dict):
        raise LoraError(f"{config_path}: holds {type(config).__name__}, not a JSON object; fix or remove it")
    return LoraSource(path, tensors, metadata, config, config_path)


def plan_lora(source):
    """The tensors core applies for `source`: modules renamed to core's key map, the factors as
    lora_A (down) / lora_B (up), and one float32 `<module>.alpha` per module. Raises LoraError on a
    feature a plain LoRA patch does not reproduce, a tensor no format reads, an incomplete module, or
    no alpha outside core's names (never guessed; in core's names alpha = rank, core's convention)."""
    meta_config = _metadata_config(source.metadata, source.path)
    if meta_config is not None:
        _check_supported(meta_config[1], f"{source.path} metadata {meta_config[0]}")
    if source.config is not None:
        _check_supported(source.config, source.config_path)
    fmt, modules, stored, left_out = _modules(source)
    ranks = {m: parts[DOWN].shape[0] for m, parts in modules.items()}
    alphas, alpha_source, alpha_notes = _alphas(source, fmt, modules, stored, ranks, meta_config)
    tensors = {}
    for module, parts in modules.items():
        tensors[f"{module}.{CORE_SUFFIX[UP]}"] = parts[UP]
        tensors[f"{module}.{CORE_SUFFIX[DOWN]}"] = parts[DOWN]
        tensors[f"{module}.{CORE_SUFFIX[ALPHA]}"] = torch.tensor(alphas[module], dtype=torch.float32)
    scales = [alphas[m] / ranks[m] for m in modules]
    notes = [(f"{fmt.name} LoRA: {len(modules)} modules, rank {_span(ranks.values())}, alpha "
              f"{_span(alphas.values())} from {alpha_source} (scale alpha / rank {_span(scales)})")] + alpha_notes
    if left_out:
        notes.append(f"left out: {len(left_out)} full base weights PEFT saves next to a LoRA on embed_tokens / "
                     f"lm_head ({first_names(left_out)}); the model's own weights are used")
    return LoraPlan(tensors, tuple(modules), fmt.name, alpha_source, tuple(notes))


def check_base(plan, key_map, weight_shapes):
    """Raises LoraError unless every module of `plan` is addressed by `key_map` (core's CLIP key map,
    alias -> state-dict key) and its lora_B @ lora_A has the shape of that weight (`weight_shapes`:
    state-dict key -> shape). A name match alone proves nothing: Qwen models of other sizes share
    every module name. Also refused: a LoRA on embed_tokens or lm_head of a model whose weights hold
    `<clip>.transformer.model.embed_tokens.weight` and no `...lm_head.weight` beside it, which core
    ties (its output head is the embedding), so the patch would change both."""
    tied = {key for key in weight_shapes if key.endswith(EMBED_WEIGHT)
            and key[:-len(EMBED_WEIGHT)] + HEAD_WEIGHT not in weight_shapes}
    missing, wrong, seen, head = [], [], {}, []
    for module in plan.modules:
        target = key_map.get(module)
        if tied and target is None and module.endswith(HEAD_MODULES):
            head.append(module)  # no weight of its own here: core's head is the embedding
            continue
        if target in tied:
            head.append(module)
        if target is None or target not in weight_shapes:
            missing.append(module)
            continue
        if target in seen:
            raise LoraError(f"LoRA modules {seen[target]} and {module} both patch {target}; keep one of them")
        seen[target] = module
        up = plan.tensors[f"{module}.{CORE_SUFFIX[UP]}"]
        down = plan.tensors[f"{module}.{CORE_SUFFIX[DOWN]}"]
        delta = (up.shape[0], *down.shape[1:])
        shape = tuple(weight_shapes[target])
        if delta != shape:
            wrong.append(f"{module}: LoRA {delta}, model {shape}")
    if missing or wrong:
        parts = []
        if missing:
            parts.append(f"{len(missing)} of {len(plan.modules)} modules are not in this model ({first_names(missing)})")
        if wrong:
            parts.append(f"{len(wrong)} of {len(plan.modules)} have another shape ({first_names(wrong)})")
        raise LoraError(f"this LoRA does not fit this model: {'; '.join(parts)}. It was trained on another base "
                        "model: pick the model it was trained on, or a LoRA trained on this one")
    if head:
        raise LoraError(f"this model ties lm_head to embed_tokens in ComfyUI core (it has no lm_head weight: the output "
                        f"head is embed_tokens.weight, as in Qwen3.5-4B), so a LoRA on embed_tokens / lm_head would "
                        f"patch both ({first_names(head)}): use a LoRA trained without embed_tokens and lm_head")


def list_loras(root):
    """The LoRAs under `root`, relative paths, sorted: every .safetensors file that is not inside a
    PEFT folder, and every folder holding adapter_config.json and adapter_model.safetensors (root
    itself excluded: its files are listed). Links are followed, each real folder once. [] when `root`
    does not exist."""
    entries, inside = [], set()
    for dirpath, _dirnames, filenames in walk_once(root):
        rel = os.path.relpath(dirpath, root)
        peft = rel != "." and PEFT_CONFIG in filenames and PEFT_WEIGHTS in filenames
        if peft:
            entries.append(rel)
        if peft or os.path.dirname(dirpath) in inside:  # a PEFT folder inside one (a checkpoint) is listed, its files are not
            inside.add(dirpath)
            continue
        entries += [os.path.relpath(os.path.join(dirpath, name), root) for name in filenames
                    if name.lower().endswith(".safetensors")]
    return sorted(entries)


def _modules(source):
    """(format, {core module: {UP, DOWN[, ALPHA]: tensor}}, {core module: stored module name},
    [stored BASE keys]) in the file's order; the embedding pair transposed into core's up / down."""
    formats, unknown, modules, stored, left_out = set(), [], {}, {}, []
    for key, tensor in source.tensors.items():
        part = _parse(key)
        if part is None:
            unknown.append(key)
            continue
        fmt, role, name, transposed = part
        formats.add(fmt.name)
        module = _core_module(fmt, name)
        if module is None:
            raise LoraError(f"{source.path}: {fmt.name} module {name} is not a module of a Qwen LM (its name starts "
                            f"with none of {', '.join(p for p, _ in fmt.modules)}); this LoRA is for another model")
        if stored.setdefault(module, name) != name:
            raise LoraError(f"{source.path}: modules {stored[module]} and {name} are the same module {module}; "
                            "keep one of them")
        if role == BASE:
            left_out.append(key)
            modules.setdefault(module, {})
            continue
        parts = modules.setdefault(module, {})
        if role in parts:
            raise LoraError(f"{source.path}: {module} has its {CORE_SUFFIX[role]} twice (as {key} and another key); "
                            "keep one of them")
        parts[role] = tensor.t().contiguous() if transposed else tensor
    if unknown:
        raise LoraError(f"{source.path}: {len(unknown)} of {len(source.tensors)} tensors are not LoRA tensors this "
                        f"loader reads ({first_names(unknown)}). It reads PEFT LoRAs (base_model.model.*: lora_A / lora_B, "
                        "lora_embedding_A / lora_embedding_B) and ComfyUI's names (text_encoders.* or lora_te_*: "
                        "lora_A / lora_B or lora_down / lora_up, alpha). Other trained tensors (biases trained with "
                        "PEFT's bias \"all\" / \"lora_only\", DoRA magnitudes, ...) are not reproduced by a plain LoRA "
                        "patch: use a LoRA trained without them")
    if not modules:
        raise LoraError(f"{source.path}: holds no tensors; export the LoRA again")
    if len(formats) > 1:
        raise LoraError(f"{source.path}: mixes the {' and '.join(sorted(formats))} layouts; export it again in one")
    fmt = next(f for f in FORMATS if f.name in formats)
    for module, parts in modules.items():
        lacking = [factor for role, factor in ((DOWN, "lora_A"), (UP, "lora_B")) if role not in parts]
        if lacking:
            raise LoraError(f"{source.path}: {stored[module]} has no {' and no '.join(lacking)}; the file is "
                            "incomplete, export it again")
        up, down = parts[UP], parts[DOWN]
        if up.dim() < 2 or down.dim() < 2 or math.prod(up.shape[1:]) != down.shape[0]:
            raise LoraError(f"{source.path}: {stored[module]}: lora_B {tuple(up.shape)} and lora_A "
                            f"{tuple(down.shape)} do not share a rank; the file is damaged, export it again")
        if ALPHA in parts and parts[ALPHA].numel() != 1:
            raise LoraError(f"{source.path}: {stored[module]}.alpha holds {parts[ALPHA].numel()} values, not one; the "
                            "file is damaged, export it again")
    return fmt, modules, stored, left_out


def _parse(key):
    """(format, role, stored module name, transposed) of a stored key by FORMATS, None when no
    format reads it."""
    fmt = next((f for f in FORMATS if key.startswith(f.marks)), None)
    return fmt and next(((fmt, role, key[:-len(s) - 1], t) for s, role, t in fmt.tensors if key.endswith("." + s)),
                        None)


def _core_module(fmt, name):
    """`name` with the first matching prefix of `fmt` replaced, None when none matches."""
    dotted = name + "."
    for old, new in fmt.modules:
        if dotted.startswith(old):
            return (new + dotted[len(old):])[:-1]
    return None


def _alphas(source, fmt, modules, stored, ranks, meta_config):
    """({core module: alpha core reads}, source name, notes). Sources in order: the file's .alpha
    tensors, its metadata, its config file; with none of them a file in core's names takes alpha = rank
    (core reads no alpha as scale 1.0), any other raises."""
    with_alpha = [m for m, parts in modules.items() if ALPHA in parts]
    if with_alpha and len(with_alpha) < len(modules):
        lacking = [stored[m] for m in modules if m not in with_alpha]
        raise LoraError(f"{source.path}: {len(with_alpha)} of {len(modules)} modules carry an .alpha tensor and "
                        f"{len(lacking)} do not ({first_names(lacking)}); the file is incomplete, export it again")
    if with_alpha:
        return {m: modules[m][ALPHA].item() for m in modules}, "tensors", []
    if meta_config is not None:
        return _config_alphas(meta_config[1], "metadata", f"{source.path} metadata {meta_config[0]}", stored, ranks)
    if source.config is not None:
        name = os.path.basename(source.config_path)
        if "lora_alpha" not in source.config:
            raise LoraError(f"{source.config_path}: no lora_alpha in it; add \"lora_alpha\": <alpha> "
                            "(the value used in training)")
        return _config_alphas(source.config, name, source.config_path, stored, ranks)
    if fmt.name == "comfy":
        return ({m: float(rank) for m, rank in ranks.items()}, RANK_ALPHA,
                ["no alpha in the file, its metadata or a config file: alpha = rank on every module, core's convention "
                 "for a LoRA without alpha (scale 1.0; core's own LoRA extraction saves none)"])
    seen = set(ranks.values())
    fill = (f"{{\"lora_alpha\": <alpha>, \"r\": {seen.pop()}}} (the values used in training)" if len(seen) == 1 else
            "{\"lora_alpha\": <alpha>} (the value used in training; the ranks differ by module and are read from "
            "the file)")
    if os.path.isdir(source.path):
        raise LoraError(f"{source.path}: no LoRA alpha: the folder has no {PEFT_CONFIG} and the file carries none. "
                        f"Put the {PEFT_CONFIG} saved with this adapter next to {PEFT_WEIGHTS}, or write one "
                        f"containing {fill}")
    stem = os.path.splitext(os.path.basename(source.path))[0]
    raise LoraError(f"{source.path}: no LoRA alpha (no .alpha tensors, no lora_alpha in its metadata, no {stem}.json "
                    f"next to it); without it the LoRA runs at the wrong strength. Write {stem}.json next to it "
                    f"containing {fill}"
                    + (f", or keep {PEFT_CONFIG} next to {PEFT_WEIGHTS} and pick that folder" if fmt.name == "peft" else ""))


def _config_alphas(config, name, where, stored, ranks):
    """Alphas from a PEFT-style config: `lora_alpha`, per module `alpha_pattern` by PEFT's rule, and
    `use_rslora` written as alpha * sqrt(rank) so that core's alpha / rank is alpha / sqrt(rank).
    The rank comes from the tensors; a config `r` (or `rank_pattern`) that disagrees with them is
    another adapter's config."""
    alpha = _number(config.get("lora_alpha"), "lora_alpha", where)
    alpha_pattern = _patterns(config.get("alpha_pattern"), "alpha_pattern", where)
    rank_pattern = _patterns(config.get("rank_pattern"), "rank_pattern", where)
    r = config.get("r")
    if r is not None and (isinstance(r, bool) or not isinstance(r, int) or r < 1):
        raise LoraError(f"{where}: r is {r!r}, not a positive whole number; fix it")
    rslora = config.get("use_rslora", False)
    if not isinstance(rslora, bool):
        raise LoraError(f"{where}: use_rslora is {rslora!r}, not true or false; fix it")
    alphas, patterned, off = {}, 0, []
    for module, rank in ranks.items():
        peft_name = stored[module].removeprefix("base_model.model.")
        key = _pattern_key(alpha_pattern, peft_name)
        a = alpha if key is None else _number(alpha_pattern[key], f"alpha_pattern[{key!r}]", where)
        patterned += key is not None
        if r is not None:
            key = _pattern_key(rank_pattern, peft_name)
            expected = r if key is None else rank_pattern[key]
            if expected != rank:
                off.append(f"{stored[module]}: {rank} in the file, {expected} in the config")
        alphas[module] = a * math.sqrt(rank) if rslora else a
    if off:
        raise LoraError(f"{where}: its ranks disagree with the file's for {len(off)} of {len(ranks)} modules "
                        f"({first_names(off)}), so it is another adapter's config. Use the config saved with this "
                        "adapter")
    notes = []
    if patterned:
        notes.append(f"alpha_pattern: {patterned} of {len(ranks)} modules take their own alpha")
    if rslora:
        notes.append("use_rslora: alpha written as alpha * sqrt(rank), so the scale is alpha / sqrt(rank)")
    return alphas, name, notes


def _pattern_key(patterns, name):
    """PEFT's get_pattern_key: the first pattern that matches `name` as `(.*\\.)?<pattern>$`."""
    return next((key for key in patterns if re.match(PATTERN_MATCH.format(key), name)), None)


def _metadata_config(metadata, path):
    """(metadata key, config) for the LoRA config the safetensors metadata holds: `lora_alpha` (+
    `r`) as values of their own, or a JSON string value holding a PEFT config with `lora_alpha`
    (preferred: it carries the patterns too). None when there is none."""
    found = []
    for key, value in metadata.items():
        if isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                config = json.loads(value)
            except ValueError:
                continue  # some other text, not a config
            if isinstance(config, dict) and "lora_alpha" in config:
                found.append((key, config))
    if len({json.dumps(c, sort_keys=True) for _, c in found}) > 1:
        raise LoraError(f"{path}: its metadata holds different LoRA configs ({', '.join(k for k, _ in found)}); "
                        "keep one")
    if "lora_alpha" in metadata:
        direct = {"lora_alpha": _parsed(metadata["lora_alpha"], float, "lora_alpha", path)}
        if "r" in metadata:
            direct["r"] = _parsed(metadata["r"], int, "r", path)
        if found and _number(found[0][1]["lora_alpha"], "lora_alpha", path) != direct["lora_alpha"]:
            raise LoraError(f"{path}: its metadata gives lora_alpha {direct['lora_alpha']} and {found[0][0]} gives "
                            f"{found[0][1]['lora_alpha']}; keep one")
        if not found:
            return "lora_alpha", direct
    return found[0] if found else None


def _check_supported(config, where):
    for field, feature in UNSUPPORTED:
        value = config.get(field)
        if (value is not None and value is not False) if field in SUB_CONFIGS else value:
            raise LoraError(f"{where}: sets {feature}, which a plain LoRA patch does not reproduce; only lora_A / "
                            "lora_B LoRAs are applied. Use a LoRA trained or exported without it")
    peft_type = config.get("peft_type")
    if peft_type is not None and str(peft_type).upper() != "LORA":
        raise LoraError(f"{where}: peft_type is {peft_type}, not LORA; only LoRA adapters are applied")
    init = config.get("init_lora_weights")
    if isinstance(init, str) and init.lower().startswith(BASE_REWRITING_INITS):
        raise LoraError(f"{where}: init_lora_weights is {init!r}, an initialisation that rewrites the base model's "
                        "weights, so the adapter fits only the rewritten model. Save it again converted to a plain "
                        "LoRA (PEFT save_pretrained with path_initial_model_for_weight_conversion; a LoftQ adapter "
                        "cannot be converted)")


def _number(value, field, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LoraError(f"{where}: {field} is {value!r}, not a number; fix it")
    return float(value)


def _parsed(text, kind, field, where):
    try:
        return kind(text)
    except ValueError:
        raise LoraError(f"{where}: metadata {field} is {text!r}, not a number; fix it") from None


def _patterns(value, field, where):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LoraError(f"{where}: {field} is {value!r}, not an object; fix it")
    for key in value:
        try:
            re.compile(PATTERN_MATCH.format(key))  # as matched: a flag group such as (?i) is an error inside it
        except re.error as e:
            raise LoraError(f"{where}: {field} key {key!r} is not a valid pattern ({e}); fix it") from None
    return value


def _span(values):
    values = sorted(set(values))
    fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v)
    return fmt(values[0]) if len(values) == 1 else f"{fmt(values[0])}..{fmt(values[-1])}"

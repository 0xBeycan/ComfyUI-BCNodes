"""The LM flow: one chat-model run of an LM node (Qwen LM today), and the catalog, LoRA list and catalog
route data the node layer shows.

run(request), in this order:
  1. the family (models/common/registry.py LM_FAMILY), its catalog (the built-in models.yaml merged with
     the user's models.yaml of each model folder), the model, the backend it names (LM_BACKEND) and the
     precision;
  2. validation: the family's build_prompt refuses what its templates cannot take (an empty user text
     without an image, thinking on a model without it, thinking with an assistant prefill, an image
     placeholder typed in a text), so the prompt is built here and kept for step 7;
  3. the model file: found by name in each model folder (the folder, then below it) and then below each
     text_encoders folder, first hit wins; else downloaded into the first model folder from the
     catalog's url (models/common/download.fetch_with_progress); no url: an error saying where to put it;
  4. the LoRA ("None", or strength 0 as core's LoRA loader, means none): its path in the LoRA folders,
     read, planned (libs/lm_lora.py; the notes go to the console) and handed to the backend as a
     LoraSpec, whose build checks the LoRA against the model's key map and weight shapes;
  5. sampling: the model's defaults for the mode (thinking on / off) with the config node's edited and
     linked fields over them; mtp is off on a model without an MTP head (a warning when the config set it);
     seed and max_new_tokens from the node;
  6. the images as the family sizes them;
  7. the backend: load (one slot, kept between runs), generate, the family's split of the output into
     (text, thinking);
  8. keep_model_loaded off: the backend unloads in a `finally` around everything after the backend is
     known (the precision, steps 2 - 7), so an error in any of them unloads too, a model an earlier run
     kept included; only an error found before (the catalog, the model name) leaves it held.

Folder paths come in from the node layer; nothing here imports folder_paths or ComfyUI.
"""

import logging
import os
from dataclasses import asdict, dataclass
from typing import TypedDict

from ..libs.files import walk_once
from ..libs.lm_lora import PEFT_CONFIG, PEFT_WEIGHTS, check_base, list_loras, plan_lora, read_lora
from ..models.common import registry
from ..models.common.download import fetch_with_progress
from ..models.common.lm.backend import MTP_CHOICES, GenerateParams, LoraSpec
from ..models.common.lm.catalog import SAMPLING_FIELDS, CatalogError, load_catalog
from ..models.common.lm.family import PromptParts
from ..models.common.registry import LM_BACKEND, LM_FAMILY

USER_CATALOG = "models.yaml"  # the user's catalog file in each model folder
NO_LORA = "None"
_CATALOGS = {}  # (family key, catalog files) -> (their (mtime_ns, size) stamps, {model name: CatalogModel})


class LMConfigValues(TypedDict, total=False):
    """The BC_LM_CONFIG value: the LM Config node's edited and linked fields only; {} = the model's defaults.
    The keys are its widgets in their order, each with the type its value leaves the node with (nodes/lm.py
    LMConfig.build casts by them), the type of the catalog's Sampling field it replaces. A model's defaults
    per mode in the catalog route give every field."""
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    mtp: str  # one of backend.MTP_CHOICES


class ModelInfo(TypedDict):
    """One model in the catalog route, keys in the order catalog_json builds them."""
    thinking: bool
    mtp: bool
    precisions: list[str]  # the model's own, in catalog order
    defaults: dict[str, LMConfigValues]  # 'thinking_off' (and 'thinking_on' on a thinking model) -> every field


class FamilyInfo(TypedDict):
    """One LM family in the catalog route, keys in the order catalog_json builds them."""
    node: str  # the LM node class of the family
    error: "str | None"  # why the user models.yaml files cannot be used; models is then the built-in catalog
    models: dict[str, ModelInfo]  # model name -> its info, in catalog order


class CatalogPayload(TypedDict):
    """GET /bcnodes/lm/catalog's body (nodes/lm.catalog_payload), read by web/js/lm.js."""
    families: dict[str, FamilyInfo]  # family key -> catalog_json's FamilyInfo
    config_node: str  # the LM config node class


@dataclass(frozen=True)
class LMRequest:
    family: str             # the LM family's registry key, e.g. 'qwen_lm'
    node: str               # the node's display name, for the console lines
    model: str              # a catalog model name
    precision: str          # one of that model's precision names
    lora: str               # NO_LORA, or a path relative to a LoRA folder (lora_choices)
    lora_strength: float
    thinking: bool
    max_new_tokens: int
    seed: int
    keep_model_loaded: bool
    system: str             # "" = no system turn
    user: str
    assistant: str          # the prefill: the answer starts with it
    image: object           # an IMAGE batch (N, H, W, C), one image per frame, or None
    config: LMConfigValues  # the config node's edited and linked fields {field: value}; {} = the model's defaults
    model_folders: tuple    # the family's folders (folder_paths order): models, user catalogs, LoRA folders
    text_encoder_folders: tuple  # ComfyUI's text_encoders folders, searched after the model folders


@dataclass(frozen=True)
class LMResult:
    text: str      # the answer: the prefill and its continuation
    thinking: str  # the reasoning; "" with thinking off


@dataclass(frozen=True)
class FamilyCatalog:
    models: dict         # model name -> CatalogModel, in catalog order
    precisions: tuple    # every precision name: the built-in ones in catalog order, then those the user files add
    error: "str | None"  # why the user models.yaml files cannot be used (models is then the built-in catalog)


def run(request: LMRequest) -> LMResult:
    """One generation; raises with what to change on any input the family, the files or the backend
    cannot take."""
    # 1. family, catalog, model, its backend, precision
    family = registry.get(LM_FAMILY, request.family)
    models = _load(family, user_catalogs(request.model_folders))
    if request.model not in models:
        raise ValueError(f"model {request.model!r} is not in the {family.folder} catalog; the models are: "
                         f"{', '.join(models)}")
    model = models[request.model]
    backend = registry.get(LM_BACKEND, model.backend)
    try:  # from here on an error unloads too (8.): a model an earlier run kept is not left held
        precision = model.precisions.get(request.precision)
        if precision is None:
            raise ValueError(f"{model.name} has no precision {request.precision!r}; its precisions are: "
                             f"{', '.join(model.precisions)}")

        # 2. validation: the family's prompt builder refuses what its templates cannot take
        image = request.image
        parts = PromptParts(system=request.system, user=request.user, assistant=request.assistant,
                            n_images=0 if image is None else int(image.shape[0]), thinking=request.thinking)
        prompt = family.build_prompt(model, parts)

        # 3. - 6. the model file, the LoRA, the sampling, the images
        path = model_path(precision, request.model_folders, request.text_encoder_folders)
        lora = lora_spec(request.lora, request.lora_strength, lora_roots(family, request.model_folders),
                         request.node)
        params = generate_params(model, request.thinking, request.config, request.seed, request.max_new_tokens,
                                 request.node)
        images = family.prepare_images(image)

        # 7. generate
        raw = backend.generate(backend.load(path, lora), prompt, images, params)
        text, thinking = family.split_output(model, raw, parts)
    finally:
        # 8. unload unless the model is kept
        if not request.keep_model_loaded:
            backend.unload()
    return LMResult(text=text, thinking=thinking)


def user_catalogs(model_folders):
    """The user catalog (USER_CATALOG) of each model folder that has one, in folder order."""
    return [p for p in (os.path.join(f, USER_CATALOG) for f in model_folders) if os.path.isfile(p)]


def _load(family, user_files):
    """{model name: CatalogModel}: the family's built-in catalog merged with `user_files`, in their order;
    read again only when one of the files changed (so its console lines come once per change, not on every
    widget refresh). Raises CatalogError naming the file and the entry."""
    files = (family.catalog_path, *user_files)
    stamp = tuple((st.st_mtime_ns, st.st_size) for st in map(os.stat, files))
    cached = _CATALOGS.get((family.key, files))
    if cached is None or cached[0] != stamp:
        models = load_catalog(family.catalog_path, user_files, family.templates, default_backend=family.default_backend)
        cached = _CATALOGS[(family.key, files)] = (stamp, models)
    return cached[1]


def catalog(family_key, model_folders):
    """The FamilyCatalog the node widgets and the catalog route show. A user models.yaml that cannot be
    used leaves the built-in catalog with the error (a run raises it)."""
    family = registry.get(LM_FAMILY, family_key)
    builtin = _load(family, [])
    user_files = user_catalogs(model_folders)
    try:
        models, error = (_load(family, user_files) if user_files else builtin), None
    except CatalogError as e:
        models, error = builtin, str(e)
    names = {}
    for entries in (builtin, models):
        for entry in entries.values():
            names.update(dict.fromkeys(entry.precisions))
    return FamilyCatalog(models=models, precisions=tuple(names), error=error)


def catalog_json(families) -> dict[str, FamilyInfo]:
    """{family key: FamilyInfo} for the catalog route: `families` maps each LM node class to (its family key,
    the family's model folders); each model gives thinking, mtp, its precision names and its sampling
    defaults per mode, in catalog order."""
    out: dict[str, FamilyInfo] = {}
    for node, (family_key, model_folders) in families.items():
        view = catalog(family_key, model_folders)
        out[family_key] = {
            "node": node,
            "error": view.error,
            "models": {name: {"thinking": m.thinking, "mtp": m.mtp, "precisions": list(m.precisions),
                              "defaults": {mode: asdict(sampling) for mode, sampling in m.defaults.items()}}
                       for name, m in view.models.items()},
        }
    return out


def lora_roots(family, model_folders):
    """The LoRA folders: the family's LoRA subfolder of each model folder, in folder order."""
    return [os.path.join(folder, family.lora_subfolder) for folder in model_folders]


def lora_choices(family_key, model_folders):
    """The lora widget's list: NO_LORA, then every LoRA of the LoRA folders (libs/lm_lora.list_loras),
    sorted, a name in two folders once."""
    roots = lora_roots(registry.get(LM_FAMILY, family_key), model_folders)
    return [NO_LORA] + sorted({name for root in roots for name in list_loras(root)})


def model_path(precision, model_folders, text_encoder_folders):
    """The file of `precision` (a catalog Precision): found by name in each model folder (the folder, then
    below it), then below each text_encoders folder; else downloaded into the first model folder."""
    for folder in (*model_folders, *text_encoder_folders):
        found = _find_under(folder, precision.file)
        if found is not None:
            return found
    if not model_folders:
        raise ValueError("no model folder is registered for this family; the node registers models/<family folder>")
    target = os.path.join(model_folders[0], precision.file)
    if precision.url is None:
        raise FileNotFoundError(f"{precision.file} is not in {', '.join(model_folders)} or under "
                                f"{', '.join(text_encoder_folders) or 'a text_encoders folder'}, and the catalog "
                                f"entry has no url to download it from: put the file in {model_folders[0]}")
    return fetch_with_progress(precision.url, target, precision.file)


def _find_under(folder, name):
    """The file `name` in `folder` or below it (libs/files.walk_once: top-down, subfolders sorted, links
    followed, each real folder once), None when there is none."""
    for dirpath, _dirnames, filenames in walk_once(folder):
        path = os.path.join(dirpath, name)
        if name in filenames and os.path.isfile(path):
            return path
    return None


def lora_spec(name, strength, roots, node):
    """The LoraSpec of the LoRA `name` (a path relative to one of `roots`, the first that holds it), None
    for NO_LORA or strength 0. Its key changes when the weights file or its config file (a PEFT folder's
    adapter_config.json, a single file's <stem>.json) changes, so the backend re-plans an edited LoRA."""
    if name == NO_LORA or strength == 0:
        return None
    path = next((os.path.abspath(p) for p in (os.path.join(root, name) for root in roots) if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"LoRA {name} is not in {', '.join(roots)}; pick another, or put it back")
    # The weights and every file read_lora may take the config from, stamped before the read: a file replaced
    # during the read leaves a key older than the tensors read, which the next run finds changed (stamped
    # after it, the old tensors would stay under the new file's stamp). A file missing here has no stamp:
    # read_lora says what is wrong when it is the weights; one that appeared during the read gives None.
    folder = os.path.isdir(path)
    weights_file = os.path.join(path, PEFT_WEIGHTS) if folder else path
    configs = ((os.path.join(path, PEFT_CONFIG),) if folder else
               (os.path.splitext(path)[0] + ".json", os.path.join(os.path.dirname(path), PEFT_CONFIG)))
    stamps = {p: os.stat(p) for p in (weights_file, *configs) if os.path.isfile(p)}
    source = read_lora(path)
    weights, config = stamps.get(weights_file), stamps.get(source.config_path)
    plan = plan_lora(source)
    for note in plan.notes:
        logging.info("[BCNodes] %s: LoRA %s: %s", node, name, note)

    def build(key_map, weight_shapes):
        check_base(plan, key_map, weight_shapes)
        return plan.tensors

    return LoraSpec(key=(path, (weights and weights.st_mtime_ns, config and config.st_mtime_ns),
                         weights and weights.st_size, strength), strength=strength, build=build)


def generate_params(model, thinking, config: LMConfigValues, seed, max_new_tokens, node):
    """GenerateParams: the catalog `model`'s sampling defaults for the mode, `config`'s fields over them;
    mtp off on a model without an MTP head (one console warning when `config` set it on)."""
    unknown = [str(field) for field in config if field not in SAMPLING_FIELDS]
    if unknown:
        raise ValueError(f"LM Config: unknown field(s) {', '.join(unknown)}; the fields are {', '.join(SAMPLING_FIELDS)}")
    sampling = {**asdict(model.defaults["thinking_on" if thinking else "thinking_off"]), **config}
    if sampling["mtp"] not in MTP_CHOICES:
        raise ValueError(f"LM Config: mtp is {sampling['mtp']!r}; use one of {', '.join(MTP_CHOICES)}")
    if not model.mtp and config.get("mtp", "off") != "off":
        logging.warning("[BCNodes] %s: mtp is %s in LM Config, but %s has no MTP head: it runs without MTP",
                        node, config["mtp"], model.name)
    return GenerateParams(do_sample=sampling["do_sample"], temperature=sampling["temperature"], top_k=sampling["top_k"],
                          top_p=sampling["top_p"], min_p=sampling["min_p"],
                          repetition_penalty=sampling["repetition_penalty"],
                          presence_penalty=sampling["presence_penalty"], seed=seed, max_new_tokens=max_new_tokens,
                          mtp=sampling["mtp"] if model.mtp else "off")

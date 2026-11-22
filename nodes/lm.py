"""LM nodes: chat models run by ComfyUI core, one node per model family, one config node for all, and an
LM Loader whose model a wrapper node runs.

    BC_QwenLM    (Qwen LM)    Qwen3.5 / Qwen3.8 / Qwen3-VL, one LoRA, images -> text, thinking
    BC_LMConfig  (LM Config)  sampling settings for an LM node: only the fields edited on it or fed by a link
    BC_LMLoader  (LM Loader)  family, model, precision, LoRA, keep -> BC_LM (a pipelines/lm.LMSpec), no load

The flow is pipelines/lm.py: run for an LM node, prepare + generate for a wrapper (PreFlight Observe,
nodes/preflight.py); the catalog, families and backend are
models/common/lm and models/qwen_lm. A wrapper lists no model and names no family: it gets an LMSpec on its
lm input and reaches the family only through pipelines/lm. This module holds the surface: the widgets
(model_inputs: the model, precision, lora and lora_strength widgets Qwen LM and LM Loader share, with
MODEL_DEFAULTS), each family's folder_paths key (models/Qwen-LM, which extra_model_paths.yaml can redirect or
extend; the LoRAs live in the loras/ subfolder of each of its folders, no key of their own) and
GET /bcnodes/lm/catalog, the catalog web/js/lm.js reads, registered by register_routes() (called from the root
__init__) only inside a running ComfyUI server. folder_paths, server and aiohttp are imported inside the
functions.
"""

import logging

from .common import LINK_INPUTS, _is_link, lm_folders

# The model widgets' initial values, Qwen LM's and LM Loader's (both INPUT_TYPES read them here): a model of the
# Qwen LM family, LM Loader's initial family (its family widget, which Qwen LM does not have, holds the family's
# name, LMFamily.name). lora is pipelines/lm.NO_LORA. A wrapper node has none: its model comes from an LM Loader.
MODEL_DEFAULTS = {"family": "Qwen LM", "model": "Qwen3.5-9B", "precision": "INT8 ConvRot", "lora": "None",
                  "lora_strength": 1.0, "keep_model_loaded": True}


def family_folders(family_key):
    """(model folders, text_encoders folders) of the LM family `family_key`: nodes/common.lm_folders over
    the family's folder (registered as models/<its folder> first). An unknown key is an error naming the
    families."""
    from ..pipelines.lm import lm_family

    return lm_folders(lm_family(family_key).folder)


def model_inputs(family_keys, node, lora_note):
    """The model, precision, lora and lora_strength widgets of an LM node over the LM families `family_keys`
    (Qwen LM: its family; LM Loader: every registered one): each family's catalog models and precisions, in
    family order, and NO_LORA then the LoRAs of each family's LoRA folders; MODEL_DEFAULTS. web/js/lm.js narrows
    the precision list to the selected model's. `lora_note` ends the lora tooltip with what the node itself
    does. A user models.yaml that cannot be used leaves the built-in catalog, with one console error naming
    `node` (the node's display name)."""
    from ..pipelines.lm import catalog, lora_choices

    models, precisions, loras = {}, {}, {}
    for family_key in family_keys:
        model_folders, _ = family_folders(family_key)
        view = catalog(family_key, model_folders)
        if view.error:
            logging.error("[BCNodes] %s: %s. The model list holds the built-in models until it is fixed; "
                          "a run raises this error.", node, view.error)
        models.update(dict.fromkeys(view.models))
        precisions.update(dict.fromkeys(view.precisions))
        loras.update(dict.fromkeys(lora_choices(family_key, model_folders)))
    return {
        "model": (list(models), {"default": MODEL_DEFAULTS["model"],
                                 "tooltip": "The chat model (the built-in ones all see images): the built-in list plus "
                                            "the models.yaml in models/Qwen-LM. The file is looked for by name in "
                                            "models/Qwen-LM (subfolders too), then anywhere under "
                                            "models/text_encoders, and downloaded into models/Qwen-LM when "
                                            "missing."}),
        "precision": (list(precisions), {"default": MODEL_DEFAULTS["precision"],
                                         "tooltip": "The model file's weights: BF16 (full), INT8 ConvRot (8-bit, "
                                                    "about half the size), W4A8 (4-bit, Qwen3.8-27B only). Only the "
                                                    "selected model's precisions are shown; another one is an "
                                                    "error."}),
        "lora": (list(loras), {"default": MODEL_DEFAULTS["lora"],
                               "tooltip": "A LoRA from models/Qwen-LM/loras: a .safetensors file (PEFT or ComfyUI "
                                          "key names) or a PEFT folder (adapter_config.json + "
                                          "adapter_model.safetensors). Its alpha is read from the file, its "
                                          "metadata or its config file (a file in ComfyUI's names without one: "
                                          "alpha = rank, core's convention); a LoRA trained on another base model "
                                          "is refused." + (" " + lora_note if lora_note else "")}),
        "lora_strength": ("FLOAT", {"default": MODEL_DEFAULTS["lora_strength"], "min": -100.0, "max": 100.0,
                                    "step": 0.01,
                                    "tooltip": "How strongly the LoRA changes the model: strength x alpha / rank "
                                               "x B @ A, core's LoRA math. 0 runs the model without it."}),
    }


class QwenLM:
    FAMILY = "qwen_lm"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **model_inputs((cls.FAMILY,), NODE_DISPLAY_NAME_MAPPINGS["BC_QwenLM"], lora_note=""),
                "thinking": ("BOOLEAN", {"default": False,
                                         "tooltip": "The model reasons before it answers: the reasoning comes out on "
                                                    "thinking, the answer on text, and sampling takes the model's "
                                                    "thinking-on defaults. Only on models with a thinking mode (not "
                                                    "Qwen3-VL), and not with an assistant prefill. Qwen3.8-27B thinks "
                                                    "at its official default effort, xhigh."}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 262144,
                                           "tooltip": "The most tokens the model writes (reasoning and answer); it stops "
                                                      "earlier at its end token. 32768 is the official length for most "
                                                      "queries. Core reserves the KV cache for prompt + max_new_tokens "
                                                      "before the first token: at 32768, about 1 GiB of VRAM for "
                                                      "Qwen3.5-4B / 9B, 2 GiB for Qwen3.8-27B and 4.5 GiB for "
                                                      "Qwen3-VL-8B (bf16), in proportion to the value."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "The sampling seed, used when do_sample is on (the model defaults have it "
                                            "on). Its control starts at fixed, so queueing again gives the cached "
                                            "answer instead of a new run; set it to randomize for a new answer "
                                            "each time."}),
                "keep_model_loaded": ("BOOLEAN", {"default": MODEL_DEFAULTS["keep_model_loaded"],
                                                  "tooltip": "On: the model (with its LoRA) stays loaded for the next "
                                                             "run; one model file is held at a time, another one "
                                                             "replaces it. Off: it is unloaded when the run ends, "
                                                             "after an error too, except an unusable models.yaml or "
                                                             "an unknown model name (raised before the model is "
                                                             "known)."}),
                "system": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "The system message. Empty: no system turn (Qwen3.8-27B with thinking "
                                                 "on still has one, holding its reasoning-effort line). Qwen3.5 / 3.8 "
                                                 "strip the whitespace around it, Qwen3-VL takes it as written."}),
                "user": ("STRING", {"default": "", "multiline": True,
                                    "tooltip": "The user message, the prompt. Empty only when an image is "
                                               "connected; the images come before this text."}),
                "assistant": ("STRING", {"default": "", "multiline": True,
                                         "tooltip": "Prefill: the answer starts with this text, and text includes it "
                                                    "(a prefill of { gives complete JSON). Empty: none. Not with "
                                                    "thinking on."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Images the model sees, one per frame of the batch, before the user "
                                               "text. Each is resized as the official Qwen processor does (sides "
                                               "multiples of 32, 65536 to 16777216 pixels, bicubic); core caps an "
                                               "image at 12845056 pixels. MTP does not run with images."}),
                "config": ("BC_LM_CONFIG", {"tooltip": "Sampling from an LM Config node: the fields edited there, "
                                                       "or fed by a link, replace the model's defaults for the mode "
                                                       "(thinking on / off); the others keep them. Not connected: the "
                                                       "model's official defaults."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    FUNCTION = "generate"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "qwen", "qwen lm", "llm", "vlm", "language model", "text generation", "caption",
                      "prompt enhancer", "chat"]
    DESCRIPTION = ("Runs a Qwen chat model (Qwen3.5-4B / 9B, Qwen3.8-27B, Qwen3-VL-8B Instruct, and abliterated "
                   "builds of Qwen3.5-9B and Qwen3-VL-8B) through ComfyUI core: system, user and an optional "
                   "assistant prefill go in as the official chat template renders them, with images from the image "
                   "input. text is the answer (prefill included), thinking the reasoning when thinking is on. "
                   "Sampling is the model's official defaults unless an LM Config node changes fields. The model "
                   "file is found in models/Qwen-LM or models/text_encoders, or downloaded on first use; one LoRA "
                   "from models/Qwen-LM/loras can be applied.")

    def generate(self, model, precision, lora, lora_strength, thinking, max_new_tokens, seed, keep_model_loaded,
                 system, user, assistant, image=None, config=None):
        from ..pipelines.lm import LMRequest, run

        model_folders, text_encoder_folders = family_folders(self.FAMILY)
        result = run(LMRequest(
            family=self.FAMILY, node=NODE_DISPLAY_NAME_MAPPINGS["BC_QwenLM"], model=model, precision=precision,
            lora=lora, lora_strength=lora_strength, thinking=thinking, max_new_tokens=max_new_tokens, seed=seed,
            keep_model_loaded=keep_model_loaded, system=system, user=user, assistant=assistant, image=image,
            config=config or {}, model_folders=model_folders, text_encoder_folders=text_encoder_folders))
        return (result.text, result.thinking)


class LMConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "do_sample": ("BOOLEAN", {"default": True,
                                          "tooltip": "On: sample with the settings below and the LM node's seed. Off: "
                                                     "greedy, the most likely token each step; core then applies no "
                                                     "temperature, filter or penalty."}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 0.01,
                                          "tooltip": "Divides the logits: lower is more focused, higher more varied."}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 1000,
                                  "tooltip": "Keep only the k most likely tokens; 0 = off."}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Keep the most likely tokens up to this total probability; 1.0 = off. "
                                               "Core leaves out the token that crosses it, a slightly smaller set "
                                               "than the official implementation keeps."}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Drop the tokens less likely than min_p x the most likely one; 0 = off."}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01,
                                                 "tooltip": "Tokens already generated (never the prompt or the prefill) "
                                                            "have their logits divided by it (multiplied when "
                                                            "negative); 1.0 = off."}),
                "presence_penalty": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.01,
                                               "tooltip": "Subtracted from the logits of tokens already generated "
                                                          "(never the prompt or the prefill); 0 = off. Official "
                                                          "advice: 0 to 2 against endless repetition; a higher value "
                                                          "can mix languages and cost some quality."}),
                "mtp": (["auto", "off", "2", "3", "4", "5"], {"default": "auto",
                                                              "tooltip": "Multi-token prediction: faster decoding with "
                                                                         "the file's MTP head. auto adapts the draft "
                                                                         "depth, 2-5 pins it, off disables it. Runs only "
                                                                         "for text-only prompts (no image) on files with "
                                                                         "an MTP head (Qwen3.5-4B / 9B, Qwen3.8-27B); "
                                                                         "elsewhere the run goes without it. Sampled "
                                                                         "output differs from non-MTP output for the "
                                                                         "same seed."}),
                "edited": ("STRING", {"default": "",
                                      "tooltip": "The fields changed on this node, comma-separated (kept by the node, "
                                                 "hidden there). Only these, and the fields fed by a link, are "
                                                 "applied; the others keep the LM node's model defaults."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("BC_LM_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "lm config", "llm settings", "sampling", "temperature", "top_p", "mtp"]
    DESCRIPTION = ("Sampling settings for an LM node (Qwen LM). Only the fields you change, or feed by a link, are "
                   "applied; every other field keeps the linked model's official default for its mode (thinking on / "
                   "off), shown on the node (greyed on the classic canvas; the Vue renderer has no greyed look that "
                   "still takes input). The node's menu item Reset to model defaults clears the changes.")

    def build(self, do_sample, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, mtp, edited,
              prompt_graph=None, unique_id=None):
        """The fields named in `edited`, then every other field whose input in the prompt (`prompt_graph`, this
        node `unique_id`) is a link: a linked value is always applied, whatever `edited` says. No prompt (a direct
        call, a node made by expansion): `edited` alone. A link is part of ComfyUI's cache key like `edited`."""
        from ..pipelines.lm import LMConfigValues

        fields = LMConfigValues.__annotations__  # in widget order, each with the type its value leaves with
        values = {"do_sample": do_sample, "temperature": temperature, "top_k": top_k, "top_p": top_p, "min_p": min_p,
                  "repetition_penalty": repetition_penalty, "presence_penalty": presence_penalty, "mtp": mtp}
        names = list(dict.fromkeys(name for name in (part.strip() for part in edited.split(",")) if name))
        unknown = [name for name in names if name not in fields]
        if unknown:
            raise ValueError(f"LM Config: edited names unknown field(s) {', '.join(unknown)}; the fields are "
                             f"{', '.join(fields)}")
        node = prompt_graph.get(unique_id) if isinstance(prompt_graph, dict) else None
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            names += [name for name in fields if name not in names and _is_link(inputs.get(name))]
        return (LMConfigValues(**{name: fields[name](values[name]) for name in names}),)


class LMLoader:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.lm import family_keys

        families = family_keys()  # {name: registry key}: the widget shows the names
        return {
            "required": {
                "family": (list(families), {"default": MODEL_DEFAULTS["family"],
                                            "tooltip": "The LM family: the catalog, folder and chat templates the "
                                                       "model comes from. The model, precision and lora lists show "
                                                       "the selected family's; a value of another family is an "
                                                       "error that names this family's."}),
                **model_inputs(tuple(families.values()), NODE_DISPLAY_NAME_MAPPINGS["BC_LMLoader"],
                               lora_note="Only the selected family's LoRAs are shown."),
                "keep_model_loaded": ("BOOLEAN", {"default": MODEL_DEFAULTS["keep_model_loaded"],
                                                  "tooltip": "On: the model (with its LoRA) stays loaded after the "
                                                             "node that runs it, for the next run; one LM model is "
                                                             "held at a time, shared with the other LM nodes. Off: "
                                                             "that node unloads it once when its call ends, after "
                                                             "an error too."}),
            },
        }

    RETURN_TYPES = ("BC_LM",)
    RETURN_NAMES = ("lm",)
    OUTPUT_TOOLTIPS = ("The model for a wrapper node's lm input (PreFlight Observe): a "
                       "description, nothing is loaded here.",)
    FUNCTION = "load"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "lm loader", "llm loader", "model loader", "chat model", "qwen", "lora"]
    DESCRIPTION = ("Picks the chat model a wrapper node runs (PreFlight Observe): the family, "
                   "its model and precision, one LoRA and its strength, and whether the model stays loaded. It "
                   "loads nothing: lm describes the model, and the node it feeds finds the file (or downloads it on "
                   "first use), reads the LoRA and loads them when it runs. The lists follow the selected family and "
                   "model.")

    def load(self, family, model, precision, lora, lora_strength, keep_model_loaded):
        """`family` is the family's name (the widget); the LMSpec holds its registry key."""
        from ..pipelines.lm import family_key_of, lm_spec

        key = family_key_of(family)
        return (lm_spec(key, model, precision, lora, lora_strength, keep_model_loaded, family_folders(key)[0]),)


def catalog_payload():
    """GET /bcnodes/lm/catalog's body, a pipelines/lm.CatalogPayload: {"families": {family key: FamilyInfo
    (pipelines/lm.catalog_json, with the family's name, which LM Loader's family widget holds)}, "config_node",
    "loader_node"}."""
    from ..pipelines.lm import CatalogPayload, catalog_json

    families = {node: (family, family_folders(family)[0]) for node, family in FAMILY_NODES.items()}
    return CatalogPayload(families=catalog_json(families), config_node="BC_LMConfig", loader_node="BC_LMLoader")


def register_routes():
    """GET /bcnodes/lm/catalog, served off the event loop. Only inside a running ComfyUI server; returns at
    once without one."""
    try:
        from server import PromptServer
    except ImportError:  # not under a ComfyUI server (tests, the import gate)
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    import asyncio

    from aiohttp import web

    @server.routes.get("/bcnodes/lm/catalog")
    async def _catalog(request):
        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, catalog_payload))


NODE_CLASS_MAPPINGS = {
    "BC_QwenLM": QwenLM,
    "BC_LMConfig": LMConfig,
    "BC_LMLoader": LMLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_QwenLM": "Qwen LM",
    "BC_LMConfig": "LM Config",
    "BC_LMLoader": "LM Loader",
}

# Each LM node and its family (models/common/registry.py LM_FAMILY), for the catalog route: every registered
# family has its node, so the route covers every family LM Loader offers.
FAMILY_NODES = {"BC_QwenLM": QwenLM.FAMILY}

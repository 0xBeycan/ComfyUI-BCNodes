"""LM nodes: chat models run by ComfyUI core, one node per model family, one config node for all, and an
LM Loader whose model a wrapper node runs.

    BC_QwenLM    (Qwen LM)    Qwen3.5 / Qwen3.8 / Qwen3-VL, one LoRA, images -> text, thinking
    BC_LMConfig  (LM Config)  sampling settings for an LM node: only the fields edited on it or fed by a link
    BC_LMLoader  (LM Loader)  family, model, precision, LoRA, keep -> BC_LM (a pipelines/lm.LMSpec), no load
    BC_Captioner (Captioner)  one training caption per image frame, with the LM Loader's model
    BC_Prompter  (Prompter)   one text-to-image prompt from a brief and / or an image, with the LM Loader's model;
                              the prompt encoded by a connected CLIP

The flow is pipelines/lm.py: run for an LM node, prepare + generate for a wrapper (Captioner's flow is
pipelines/captioner, Prompter's pipelines/prompter, each handed a start callable over them); the catalog,
families and backend are
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

from .common import LINK_INPUTS, _is_link, _links, lm_folders

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
    OUTPUT_TOOLTIPS = ("The model for a wrapper node's lm input (Captioner, Prompter): a description, nothing is "
                       "loaded here.",)
    FUNCTION = "load"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "lm loader", "llm loader", "model loader", "chat model", "qwen", "lora"]
    DESCRIPTION = ("Picks the chat model a wrapper node runs (Captioner, Prompter): the family, its model and "
                   "precision, one LoRA and its strength, and whether the model stays loaded. It loads nothing: lm "
                   "describes the model, and the node it feeds finds the file (or downloads it on first use), reads "
                   "the LoRA and loads them when it runs. The lists follow the selected family and model.")

    def load(self, family, model, precision, lora, lora_strength, keep_model_loaded):
        """`family` is the family's name (the widget); the LMSpec holds its registry key."""
        from ..pipelines.lm import family_key_of, lm_spec

        key = family_key_of(family)
        return (lm_spec(key, model, precision, lora, lora_strength, keep_model_loaded, family_folders(key)[0]),)


class Captioner:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.captioner.caption import MAX_NEW_TOKENS, MAX_SIDE, STYLES

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The images to caption, one caption per frame: a single image, a "
                                               "batch, or a list of batches (each frame of each, in order). The "
                                               f"model sees each frame with its long side capped at {MAX_SIDE} "
                                               "(never enlarged), then sized as its family sizes images."}),
                "lm": ("BC_LM", {"tooltip": "The chat model, from an LM Loader: its model, precision and LoRA (at "
                                            "its strength), and whether it stays loaded after this node (off: "
                                            "unloaded once after the last frame, after an error too)."}),
                "style": (list(STYLES), {"default": "prose",
                                         "tooltip": "The built-in prompt. tags: one line of attribute groups (class "
                                                    "word, shot type, pose, expression, clothing, accessory, "
                                                    "setting, lighting, background), each field only when the image "
                                                    "shows it. prose: two or three sentences, 30 to 50 words. Both "
                                                    "open with the class word, woman or man, read off the image, "
                                                    "and never describe hair or identity. none: no prompt, the "
                                                    "model gets the image alone (a LoRA trained to caption without "
                                                    "one). A connected instruction replaces it. Any style works "
                                                    "with a LoRA."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "The sampling seed, the same for every frame. Its control starts at "
                                            "fixed, so queueing again gives the cached captions instead of a new "
                                            "run; set it to randomize for new captions each time."}),
            },
            "optional": {
                "instruction": ("STRING", {"forceInput": True,
                                           "tooltip": "Your own prompt in place of the style's built-in one, with any "
                                                      "style; the model gets the image, then this text. Not "
                                                      "connected, or only whitespace: the style decides. "
                                                      f"The answer is cut at {MAX_NEW_TOKENS} tokens."}),
                "config": ("BC_LM_CONFIG", {"tooltip": "Sampling from an LM Config node in place of the Captioner's "
                                                       "own (temperature 0.7, top_p 0.8, top_k 20, no presence "
                                                       "penalty): the model's thinking-off defaults with the fields "
                                                       "edited there, or fed by a link, over them. Not connected: the "
                                                       "Captioner's own."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "caption")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_TOOLTIPS = ("Every frame of the input as it came in (not the resized copy the model saw), one list item "
                       "per frame.",
                       "Each frame's caption, in the same order: connect both outputs to Save Image With Caption to "
                       "save every frame with its caption.")
    FUNCTION = "caption"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "captioner", "caption", "captioning", "auto caption", "dataset", "lora training",
                      "tags"]
    DESCRIPTION = ("Writes one training caption per image: each frame of the batch (or of each batch in a list) goes "
                   "on its own to the chat model of the connected LM Loader, with the style's built-in prompt (tags "
                   "or prose) or your instruction, and comes out as two lists in the same order: image (the frame as "
                   "it came in) and caption. Connect both to Save Image With Caption to write a dataset folder. For "
                   "captions: thinking off, no system turn, at most 600 tokens, one seed for every frame, each "
                   "frame's long side capped at 1024; sampling temperature 0.7, top_p 0.8, top_k 20, no presence "
                   "penalty, or a connected LM Config's (the model's own defaults with the fields edited there). "
                   "Style none sends the image alone, for an image-to-caption LoRA of your own (chosen on the LM "
                   "Loader), with or without an instruction.")

    def caption(self, image, lm, style, seed, instruction=None, config=None):
        """The flow (pipelines/captioner/caption) with start: the model of `lm` (an LMSpec) prepared once per node
        call, then one generate per frame: the frame then the user text as the user turn, no system turn, no
        prefill, thinking off, MAX_NEW_TOKENS, `seed`, and DECODING, or `config` (an LM Config's value) as it is
        when one is connected. The model stays loaded from one frame to the next; with the spec's
        keep_model_loaded off it is unloaded once afterwards, after an error too."""
        from ..pipelines.captioner import caption as flow
        from ..pipelines.lm import generate, prepare, unload_backends

        def start():
            model_folders, text_encoder_folders = family_folders(lm.family)
            prepared = prepare(lm, node=NODE_DISPLAY_NAME_MAPPINGS["BC_Captioner"], thinking=False,
                               max_new_tokens=flow.MAX_NEW_TOKENS, seed=seed,
                               config=dict(flow.DECODING) if config is None else config,
                               model_folders=model_folders, text_encoder_folders=text_encoder_folders)
            return lambda frame, user: generate(prepared, "", user, "", frame).text

        try:
            return flow.caption(image, style, instruction, start)
        finally:
            if not lm.keep_model_loaded:
                unload_backends()


# The error of a linked conditioning output without a CLIP: raised before the model runs when the prompt links it,
# and the message of the ExecutionBlocker the output carries otherwise (a node linked to it after a cached run).
NO_CLIP = ("Prompter: its conditioning output is linked, but no CLIP is connected: connect a CLIP (from a checkpoint "
           "or CLIP loader) to Prompter's clip input, or encode the text output with a node of your own")


class Prompter:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.prompter.write import INSTRUCTIONS, MAX_NEW_TOKENS, MAX_SIDE

        return {
            "required": {
                "lm": ("BC_LM", {"tooltip": "The chat model, from an LM Loader: its model, precision and LoRA (at "
                                            "its strength), and whether it stays loaded after this node (off: "
                                            "unloaded once the prompt is written, before the clip encode, after an "
                                            "error too)."}),
                "text": ("STRING", {"default": "", "multiline": True,
                                    "tooltip": "The brief: the picture you want, short and informal, in any "
                                               "language. With include / exclude identity it follows the built-in "
                                               "brief prompt and a connected image is not used; with none it is the "
                                               "user text itself, after the image when one is connected. Empty (or "
                                               "only whitespace): the image alone is used."}),
                "instructions": (list(INSTRUCTIONS), {"default": "include identity",
                                                      "tooltip": "The built-in prompt. include identity: the "
                                                                 "person's appearance is written (as the brief gives "
                                                                 "it, or read off the image). exclude identity: "
                                                                 "nothing about how the person looks, for a "
                                                                 "character LoRA that carries it; the class word, "
                                                                 "woman or man, comes from the brief (woman when it "
                                                                 "says nothing) or the image. Both write one "
                                                                 "paragraph that opens with the shot type and camera "
                                                                 "height. A brief uses the brief prompt, an image "
                                                                 "alone the image prompt. none: no prompt, the model "
                                                                 "gets the brief and / or the image alone (a "
                                                                 "prompter LoRA of your own). A connected instruction "
                                                                 "replaces it."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "The sampling seed. Its control starts at fixed, so queueing again gives "
                                            "the cached prompt instead of a new run; set it to randomize for a new "
                                            "prompt each time."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "A reference photograph: one image (a batch of more frames is an "
                                               "error). Used when the brief is empty, or with instructions none (the "
                                               "image, then the brief); with include / exclude identity and a brief "
                                               "it is not used. The model sees it with its long side capped at "
                                               f"{MAX_SIDE} (never enlarged), then sized as its family sizes "
                                               "images."}),
                "instruction": ("STRING", {"forceInput": True,
                                           "tooltip": "Your own prompt in place of the built-in one, with any "
                                                      "instructions value: before the brief, a blank line between, "
                                                      "or after the image when the brief is empty. Not connected, "
                                                      "or only whitespace: instructions decides. The answer is cut "
                                                      f"at {MAX_NEW_TOKENS} tokens."}),
                "clip": ("CLIP", {"tooltip": "A CLIP (text encoder) for the conditioning output: the prompt is "
                                             "encoded as the standard CLIP Text Encode encodes it. Not connected: "
                                             "leave conditioning unlinked (linked, it is an error asking for a "
                                             "CLIP); text works either way."}),
                "config": ("BC_LM_CONFIG", {"tooltip": "Sampling from an LM Config node in place of Prompter's own "
                                                       "(temperature 0.7, top_p 0.8, top_k 20, no presence "
                                                       "penalty): the model's thinking-off defaults with the fields "
                                                       "edited there, or fed by a link, over them. Not connected: "
                                                       "Prompter's own."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("STRING", "CONDITIONING")
    RETURN_NAMES = ("text", "conditioning")
    OUTPUT_TOOLTIPS = ("The prompt the model wrote, its surrounding whitespace stripped: the positive prompt. Take "
                       "it to your own encode node when the workflow needs another one.",
                       "The prompt encoded by the connected CLIP, as CLIP Text Encode encodes it: the positive "
                       "conditioning only (the negative comes from elsewhere). Needs the clip input.")
    FUNCTION = "write"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "prompter", "prompt", "prompt enhancer", "prompt writer", "image to prompt",
                      "enhance prompt", "positive prompt"]
    DESCRIPTION = ("Writes one text-to-image prompt with the chat model of the connected LM Loader, from a short "
                   "brief (text), a reference image, or both. include identity and exclude identity (the person's "
                   "appearance written or left to a character LoRA) choose a built-in prompt: with a brief the brief "
                   "prompt, the brief after it and the image not used; with an image alone the image prompt, after "
                   "the image. none sends the brief and / or the image (the image first) with no prompt, for a "
                   "prompter LoRA of your own. A connected instruction replaces the built-in prompt. Thinking off, "
                   "no system turn, at most 600 tokens, the image's long side capped at 1024; sampling temperature "
                   "0.7, top_p 0.8, top_k 20, no presence penalty, or a connected LM Config's (the model's own "
                   "defaults with the fields edited there). With a CLIP connected, conditioning is the prompt "
                   "encoded as CLIP Text Encode encodes it, the positive only, after the model is unloaded when the "
                   "LM Loader does not keep it.")

    def write(self, lm, text, instructions, seed, image=None, instruction=None, clip=None, config=None,
              prompt_graph=None, unique_id=None):
        """The flow (pipelines/prompter/write) with start: the model of `lm` (an LMSpec) prepared once, then one
        generate: no system turn, no prefill, thinking off, MAX_NEW_TOKENS, `seed`, and DECODING, or `config` (an
        LM Config's value) as it is when one is connected. With the spec's keep_model_loaded off the model is
        unloaded afterwards, after an error too, before `clip` encodes the prompt (core's CLIPTextEncode:
        tokenize, then encode_from_tokens_scheduled). Without a clip: an error before the model runs when the
        prompt (`prompt_graph`, this node `unique_id`) links conditioning, else conditioning is an
        ExecutionBlocker with that error, which only a node linked to it after a cached run meets."""
        from ..pipelines.lm import generate, prepare, unload_backends
        from ..pipelines.prompter import write as flow

        def start():
            model_folders, text_encoder_folders = family_folders(lm.family)
            prepared = prepare(lm, node=NODE_DISPLAY_NAME_MAPPINGS["BC_Prompter"], thinking=False,
                               max_new_tokens=flow.MAX_NEW_TOKENS, seed=seed,
                               config=dict(flow.DECODING) if config is None else config,
                               model_folders=model_folders, text_encoder_folders=text_encoder_folders)
            return lambda frame, user: generate(prepared, "", user, "", frame).text

        try:
            if clip is None and isinstance(prompt_graph, dict) and _links(
                    prompt_graph, {unique_id: {self.RETURN_NAMES.index("conditioning"): "conditioning"}})[unique_id]:
                raise ValueError(NO_CLIP)
            prompt = flow.write(text, image, instructions, instruction, start)
        finally:
            if not lm.keep_model_loaded:
                unload_backends()
        if clip is None:
            from comfy_execution.graph import ExecutionBlocker

            return (prompt, ExecutionBlocker(NO_CLIP))
        return (prompt, clip.encode_from_tokens_scheduled(clip.tokenize(prompt)))


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
    "BC_Captioner": Captioner,
    "BC_Prompter": Prompter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_QwenLM": "Qwen LM",
    "BC_LMConfig": "LM Config",
    "BC_LMLoader": "LM Loader",
    "BC_Captioner": "Captioner",
    "BC_Prompter": "Prompter",
}

# Each LM node and its family (models/common/registry.py LM_FAMILY), for the catalog route: every registered
# family has its node, so the route covers every family LM Loader offers.
FAMILY_NODES = {"BC_QwenLM": QwenLM.FAMILY}

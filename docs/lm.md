# LM

Chat models run by ComfyUI core: one node per model family (Qwen LM today) and one config node, LM Config, for all of them. Text and images in, text out. The prompt is built as the model's official chat template renders it, sampling starts from the official defaults of each model and mode, and the model files come from a public repository on first use. Type `qwen`, `llm`, `vlm`, `caption` or `prompt enhancer` in the node search to find the node.

## `BC_QwenLM` — Qwen LM

`model`, `precision`, `lora`, `lora_strength`, `thinking`, `max_new_tokens`, `seed`, `keep_model_loaded`, `system`, `user`, `assistant`; optional `image` (`IMAGE`), `config` (`BC_LM_CONFIG`) → `text` (`STRING`), `thinking` (`STRING`).

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `model` | `Qwen3.5-9B` | the [built-in models](#models-and-files) plus those a [`models.yaml`](#modelsyaml) adds | The chat model. |
| `precision` | `INT8 ConvRot` | `BF16` (full weights), `INT8 ConvRot` (8-bit, about half the size), `W4A8` (4-bit, Qwen3.8-27B only), then the names a `models.yaml` adds | The model file. The node lists only the selected model's precisions; a precision the model does not have (kept from a saved workflow, or sent through the API) is an error that lists its precisions. |
| `lora` | `None` | `None` and the [LoRAs](#lora) in `models/Qwen-LM/loras` | One LoRA applied to the model. |
| `lora_strength` | `1.0` | `-100`–`100`, step `0.01` (core's LoRA loader's range) | The patch is strength × alpha / rank × (lora_B @ lora_A), core's LoRA math. `0` runs the model without the LoRA, which is then not read at all (as core's LoRA loader). |
| `thinking` | off | | The model reasons before it answers: the reasoning comes out on `thinking`, the answer on `text`, and sampling takes the model's thinking-on defaults. Only on a model with a thinking mode (not Qwen3-VL: choosing such a model turns the toggle off and disables it); not with an `assistant` prefill. See [Prompt and thinking](#prompt-and-thinking). |
| `max_new_tokens` | `32768` | `1`–`262144` | The most tokens the model writes, reasoning and answer together; it stops earlier at its end token. 32768 is the official length for most queries. It sets the VRAM of the [KV cache](#memory). |
| `seed` | `0` | `0`–`2^64 − 1` | The sampling seed, used only when `do_sample` is on (every model's default). Its *control after generate* starts at `fixed`, not `randomize` (below). |
| `keep_model_loaded` | on | | On: the model (with its LoRA) stays loaded for the next run. Off: it is unloaded when the run ends, after an error too, except one found before the model is known (a `models.yaml` that cannot be used, a model the catalog does not have; [Memory](#memory)). |
| `system` | empty | multiline | The system message; empty = no system turn. |
| `user` | empty | multiline | The prompt. May be empty only when an image is connected. |
| `assistant` | empty | multiline | Prefill: the answer starts with this text, and `text` includes it. Empty = none. |

| Input / output | Type | Meaning |
| --- | --- | --- |
| `image` | `IMAGE`, optional | Images the model sees: every frame of the batch is one image (no separate video input), placed before the user text. See [Images](#images). |
| `config` | `BC_LM_CONFIG`, optional | An [LM Config](#bc_lmconfig--lm-config) node: the fields edited there, or fed by a link, replace the model's defaults for the mode. Not connected: the model's official defaults. |
| `text` | `STRING` | The answer: the prefill and its continuation. |
| `thinking` | `STRING` | The reasoning; empty with `thinking` off. |

Seed: an LM run is long and loads gigabytes, so the seed's control starts at `fixed`: queueing the same workflow again returns ComfyUI's cached answer instead of running the model again. Set the control to `randomize` (or `increment`) for a new answer on every queue. With `do_sample` off the seed is not used.

The frontend (`web/js/lm.js`, the classic canvas and the Vue node renderer) keeps the node consistent with the catalog: the `precision` list shows only the selected model's precisions, and `thinking` is disabled while it is off on a model without thinking. Only a model you choose changes the values: the current precision is kept when the new model has it, otherwise it becomes the model's first, and `thinking` turns off for a model without it. A saved workflow's values come back as saved: a precision its model does not have, or `thinking` on for a model without it, is kept (`thinking` then stays enabled, so it can be turned off) and the run stops with an error that says what to change. While a `models.yaml` cannot be used (the lists then hold the built-in models only), choosing a model does not change the precision or `thinking` either, so a precision the file adds is not lost.

### Models and files

The built-in catalog (`models/qwen_lm/models.yaml`). All six see images. The files are in the public Hugging Face repository [`beycanai/Qwen-LM`](https://huggingface.co/beycanai/Qwen-LM) (no token), each at `https://huggingface.co/beycanai/Qwen-LM/resolve/main/<file>`.

| `model` | Template | Thinking | MTP head | `BF16` | `INT8 ConvRot` | `W4A8` |
| --- | --- | --- | --- | --- | --- | --- |
| `Qwen3.5-4B` | `qwen3.5` | yes | yes | `Qwen3.5-4B_bf16.safetensors` | `Qwen3.5-4B_int8_convrot.safetensors` | — |
| `Qwen3.5-9B` | `qwen3.5` | yes | yes | `Qwen3.5-9B_bf16.safetensors` | `Qwen3.5-9B_int8_convrot.safetensors` | — |
| `Qwen3.5-9B Abliterated` | `qwen3.5` | yes | no | `Qwen3.5-9B-abliterated_bf16.safetensors` | `Qwen3.5-9B-abliterated_int8_convrot.safetensors` | — |
| `Qwen3-VL-8B Instruct` | `qwen3-vl` | no | no | `Qwen3-VL-8B-Instruct_bf16.safetensors` | `Qwen3-VL-8B-Instruct_int8_convrot.safetensors` | — |
| `Qwen3-VL-8B Instruct Abliterated` | `qwen3-vl` | no | no | `Qwen3-VL-8B-Instruct-abliterated_bf16.safetensors` | `Qwen3-VL-8B-Instruct-abliterated_int8_convrot.safetensors` | — |
| `Qwen3.8-27B` | `qwen3.8` | yes | yes | `Qwen3.8-27B_bf16.safetensors` | `Qwen3.8-27B_int8_convrot.safetensors` | `Qwen3.8-27B_w4a8.safetensors` |

Where the file comes from, first hit wins (by exact file name, links followed):

1. each `Qwen-LM` folder in ComfyUI's folder order (`ComfyUI/models/Qwen-LM/`, and any folder `extra_model_paths.yaml` gives the `Qwen-LM` key): the folder itself, then its subfolders in name order;
2. each `text_encoders` folder, with its subfolders, so a file already there is used where it is;
3. otherwise it is downloaded into the first `Qwen-LM` folder on the first run that needs it, through `<file>.part` (an interrupted download resumes on the next run), with a console line every 10%. A catalog entry without a URL (only a user `models.yaml` can have one) is an error instead, naming the folders searched and the folder to put the file in.

A Hugging Face token saved in [Auto Model Downloader](loaders.md#bc_automodeldownloader--auto-model-downloader) is sent with a huggingface.co URL, so a `models.yaml` entry may point at a gated repository; the built-in files need none.

ComfyUI core loads the file with its text-encoder loader (`comfy.sd.load_clip`, no embedding directory), which tells the architecture from the weights, and generates with the model's own `generate`. One model file is held at a time: another file unloads the one held first; another LoRA, or none, drops only the LoRA'd copy and keeps the base model.

### Prompt and thinking

The prompt is the model's official chat template, built by the node: core's own Qwen templates differ from the official ones and are not used, and the node tokenizes the prompt itself, as the official tokenizer does (below). Qwen3.5-9B with thinking off, a system text, one image and `Describe this.`:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Describe this.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

| Template | Models | Texts | The assistant turn opens with |
| --- | --- | --- | --- |
| `qwen3.5` | Qwen3.5-4B, 9B, 9B Abliterated | `system` and `assistant` stripped of surrounding whitespace; `user` stripped together with its images (below) | thinking off: an empty, closed think block (`<think>\n\n</think>\n\n`), then the prefill; thinking on: `<think>\n` |
| `qwen3.8` | Qwen3.8-27B | as `qwen3.5` | as `qwen3.5`; with thinking on the system turn also carries the reasoning-effort line (below) |
| `qwen3-vl` | Qwen3-VL-8B Instruct and its abliterated build | verbatim, nothing stripped | no think block (the model has no thinking mode); the prefill verbatim |

- `system`: empty, or whitespace only, means no system turn, for every template. Exception: Qwen3.8-27B with thinking on always has a system turn holding its official default reasoning effort, `xhigh`: `Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.`, then a blank line and the system text when there is one. There is no widget for another effort.
- `user`: the images come first, one `<|vision_start|><|image_pad|><|vision_end|>` block per frame with nothing between them, then the text. Qwen3.5 / 3.8 strip the user content as the official template trims it, images and text together: without an image the text loses the whitespace around it, after an image only the trailing whitespace (`\n\nDescribe the scene.` gives `<|vision_end|>\n\nDescribe the scene.<|im_end|>`); Qwen3-VL takes it verbatim. An empty `user` without an image is an error: `user is empty: write the prompt in user, or connect an image`.
- `assistant` (prefill): the answer starts with it and `text` is the prefill plus the generated continuation, so a prefill of `{` gives complete JSON. Qwen3.5 / 3.8 take it stripped, after the closed think block; Qwen3-VL verbatim. Of the continuation only the end is stripped after a prefill (its leading space belongs between the prefill's last word and the next); without a prefill both ends are. Nothing else is cleaned.
- `thinking`: off by default here, although the official Qwen3.5 / 3.8 templates think by default. On, the prompt ends inside the think block (`<think>\n`) and the output is split at the first `</think>`: before it is `thinking`, after it `text`, both stripped. When the output has no `</think>` (`max_new_tokens` ran out while the model was thinking), `text` is empty, `thinking` holds all of the output and the console warns: `[BCNodes] Qwen LM: <model> stopped while thinking (no </think>; max_new_tokens reached?): text is empty and thinking holds all of the output`.
- Thinking with a non-empty `assistant` is an error (officially a prefill closes an empty think block, so the two cannot coexist): `assistant (a prefill) closes the think block, so it cannot be used with thinking on: clear assistant or turn thinking off`. Thinking on a model without thinking (kept from a saved workflow, or sent through the API) is an error: `<model> has no thinking mode: turn thinking off`. These checks, the empty `user` and a typed `<|image_pad|>` (below) come before any file is looked for.
- The output is decoded with the special tokens (`<|im_end|>`, `<|endoftext|>`) left out.
- The text goes to the model as written: the node tokenizes the whole prompt with the model's own tokenizer, not through ComfyUI's prompt parser, so `(word:1.2)` weights, `embedding:` names and `\(` escapes are plain text. Chat tokens typed into a text (`<|im_start|>`, `<think>`, …) are read as those tokens; a `<|image_pad|>` typed into `system`, `user` or `assistant` is an error, because each one must stand for an image: `<|image_pad|> in user: it stands for an image (the prompt gets one per frame of the image input); remove <|image_pad|> from the system, user and assistant text`.
- The ids are the official tokenizer's. Core builds its Qwen3.5 / 3.8 tokenizer as transformers' `Qwen2Tokenizer`, which splits words with Qwen2's pre-tokenizer pattern (`\p{L}+`) and ignores the official one that core's own `tokenizer_config.json` gives (`[\p{L}\p{M}]+`, as in the official `tokenizer.json` of Qwen3.5-9B and Qwen3.8-27B). So a word with combining marks (Hindi and the other Indic scripts, Thai, Arabic with harakat, …) gets other ids from core: `नमस्ते दुनिया, यह एक परीक्षण है।` is 21 ids there and 14 officially. The node encodes these models' prompts with a copy of core's tokenizer that splits with the official pattern (core's own tokenizer is left as it is), and gets exactly the official ids; Latin, Turkish, CJK and Korean text gives the same ids either way. Qwen3-VL's official tokenizer splits with Qwen2's pattern, so core's is used as it is. Where the copy cannot be built (a transformers 4.x install, whose `Qwen2Tokenizer` is the slow tokenizer), the console warns once, naming the cause and the transformers and tokenizers versions, and core's ids are used.

### Images

Every frame of the `image` batch is one image, in batch order (a video's frames given as an IMAGE batch are that many images). Each is sized as the official Qwen image processor sizes it (the `preprocessor_config` of Qwen3.5, Qwen3.8 and Qwen3-VL), not with core's defaults (3,136 to 12,845,056 pixels, bilinear):

- each side rounded to a multiple of 32 (patch 16 × merge 2), then scaled with the aspect kept into 65,536 to 16,777,216 pixels; the same size for every frame;
- resampled bicubic with antialiasing in float32, a pass along the width and then one along the height, each clamped to [0, 1], as the official processor's 8-bit resize does (within about one 8-bit level of it); a frame already at its size is only clamped;
- an alpha channel is dropped (the official processor converts to RGB); an image more than 200 times wider than tall (or taller than wide) is refused, as the official processor refuses it; a half-precision batch (float16 / bfloat16) is read a frame at a time, as the pack's other image nodes read it ([Image](image.md));
- core's own resize then keeps that size, up to 12,845,056 pixels (see [Known limits](#known-limits)).

| Input (width × height) | The model sees |
| --- | --- |
| 128 × 128 | 256 × 256 |
| 300 × 200 | 320 × 224 |
| 512 × 512 | 512 × 512 |
| 1920 × 1080 | 1920 × 1088 |
| 4096 × 4096 | 4096 × 4096 here, 3584 × 3584 after core's cap |

An empty batch (0 frames) counts as no image. MTP does not run with images ([MTP](#mtp)).

### LoRA

One LoRA per node, from the `loras` folder inside each `Qwen-LM` folder (`ComfyUI/models/Qwen-LM/loras/`; it has no folder key of its own, so it follows the `Qwen-LM` folders, `extra_model_paths.yaml` included).

The `lora` list: `None`, then, sorted, every `.safetensors` file in the folder or below it (its relative path) and every folder below it that holds both `adapter_config.json` and `adapter_model.safetensors` (a PEFT folder: its relative folder path; a PEFT folder inside one, such as a checkpoint, is listed too, the files inside PEFT folders are not). Links are followed, each real folder once. A name in two `Qwen-LM` folders is listed once, and the first folder that holds it is used. The list is read with the node definitions: a new LoRA appears after *Refresh Node Definitions* (`R`) or a restart.

Formats (`libs/lm_lora.py` `FORMATS`; a file holds one of them, a mix is refused):

| Format | Tensor names | Saved by |
| --- | --- | --- |
| PEFT | `base_model.model.<module>.lora_A.weight` / `.lora_B.weight` (also with PEFT's `.default` adapter name), `lora_embedding_A` / `lora_embedding_B` for `embed_tokens`. Modules: `model.layers.N.…` (a causal LM's names) and `model.language_model.layers.N.…` (a vision-language model's), `embed_tokens`, `lm_head`, and the vision tower `model.visual.…` | PEFT, so Unsloth, TRL, Axolotl, LLaMA-Factory and ms-swift |
| ComfyUI names | `text_encoders.…` or `lora_te_…`, with `lora_A` / `lora_B` or `lora_down` / `lora_up` and `.alpha` (optional, as in core) | the names core's key map holds; passed through |

Either comes as a single `.safetensors` file or as a PEFT folder (`adapter_model.safetensors` and `adapter_config.json`).

When a LoRA trains `embed_tokens` or `lm_head`, PEFT also saves a full copy of that base weight (`base_layer.weight`; about a billion values on a 9B model): it is left out unread, and the model's own weight is used.

**Alpha.** Core patches a weight with strength × alpha / rank × (lora_B @ lora_A), with rank = lora_A's first dimension, read from the file. Core reads alpha only from a `<module>.alpha` tensor, and with none it uses scale 1.0; PEFT keeps alpha in `adapter_config.json`, never in the file. So the node resolves the alpha and writes the `.alpha` tensors itself, and never guesses a PEFT LoRA's: an adapter trained with r 16 and alpha 32 runs at scale 2.0, where without its alpha it would run at 1.0. Sources, the first that has an alpha wins:

1. `.alpha` tensors in the file (on every module or on none; some is an error);
2. the safetensors metadata: `lora_alpha` (and `r`) as values of their own, or a text value holding a PEFT config with `lora_alpha` (preferred, it carries the patterns too; when both are there they must agree);
3. the config file: `adapter_config.json` in a PEFT folder; `<stem>.json` next to a single file (for a file named `adapter_model.safetensors` with no `adapter_model.json`, the `adapter_config.json` beside it);
4. a file with ComfyUI names and none of the above: alpha = rank on every module, so scale 1.0, core's own convention for a LoRA without alpha (core's loader reads no alpha as scale 1.0, and core's *Extract and Save Lora* writes none). The console's plan line says so: `alpha 16 from rank (scale alpha / rank 1)`, then `no alpha in the file, its metadata or a config file: alpha = rank on every module, core's convention for a LoRA without alpha (scale 1.0; core's own LoRA extraction saves none)`.

A PEFT-named LoRA with none of them stops the run with the exact fix (PEFT's own default alpha is 8, not the rank, so it cannot be assumed), e.g. for `my_lora.safetensors` with rank 16:

```
.../loras/my_lora.safetensors: no LoRA alpha (no .alpha tensors, no lora_alpha in its metadata, no my_lora.json next to it); without it the LoRA runs at the wrong strength. Write my_lora.json next to it containing {"lora_alpha": <alpha>, "r": 16} (the values used in training), or keep adapter_config.json next to adapter_model.safetensors and pick that folder
```

When the ranks differ by module it asks for `{"lora_alpha": <alpha>}` alone (the ranks are read from the file). A folder holding `adapter_model.safetensors` without `adapter_config.json` is not listed as a PEFT folder: the list shows its `adapter_model.safetensors` as a file, which gets the message above. The folder name itself is not in the list, so a prompt sent through the API with it is refused by ComfyUI's validation (`Value not in list`) before the node runs; only a name ComfyUI does not check against the list (one that comes over a link from an any-type `*` output, or one that code hands `pipelines/lm.run` directly) gets the LoRA reader's `Put the adapter_config.json saved with this adapter next to adapter_model.safetensors, or write one containing {"lora_alpha": <alpha>, "r": 16} (the values used in training)`. A config file without `lora_alpha` gets `add "lora_alpha": <alpha> (the value used in training)`, for a file with ComfyUI names too (the config file is there to give the alpha).

From a PEFT config the node also takes:

- `alpha_pattern`: a module's own alpha, by PEFT's rule (the first key that matches the module name, without `base_model.model.`, as `(.*\.)?<key>$`);
- `use_rslora: true`: scale alpha / √rank instead of alpha / rank (written as alpha × √rank, so core's alpha / rank gives it);
- the rank from the tensors, so `rank_pattern` is followed as trained; when the config gives `r`, it (with its `rank_pattern`) must agree with the file's ranks, otherwise it is another adapter's config and an error.

Refused, with an error naming the feature, because a plain lora_A / lora_B patch does not reproduce it (checked in the metadata's config and in the config file whenever they are there, whichever source gives the alpha):

| Config field | Feature |
| --- | --- |
| `use_dora` | DoRA |
| `modules_to_save` (non-empty) | fully trained modules |
| `trainable_token_indices` | trained token rows |
| `lora_bias` | LoRA biases |
| `fan_in_fan_out` | transposed weights |
| `use_qalora` | QA-LoRA |
| `layer_replication` | replicated layers |
| `alora_invocation_tokens` | activated LoRA |
| `use_bdlora` | block-diagonal factors |
| `arrow_config` | Arrow routing over several LoRAs |
| `kasa_config` | KaSA, which also truncates the base weights |
| `peft_type` other than `LORA` | another adapter type |
| `init_lora_weights` `pissa…`, `olora`, `corda`, `astra`, `lora_ga`, `loftq` | an initialisation that rewrites the base model's weights, so the adapter fits only the rewritten model: save it again converted to a plain LoRA (PEFT `save_pretrained` with `path_initial_model_for_weight_conversion`); a LoftQ adapter cannot be converted |

A field counts as set when it is true or non-empty, as PEFT tests it; `use_bdlora`, `arrow_config` and `kasa_config` count for any value but `null` or `false` (`{}` is PEFT's default sub-config). Tensors no format reads are refused too: biases trained with PEFT's `bias` `"all"` / `"lora_only"`, DoRA magnitudes and the like. A PEFT folder holding only `adapter_model.bin` is not listed, so through the API ComfyUI's validation refuses its name (`Value not in list`); a name that comes over a link from an any-type `*` output, or that code hands `pipelines/lm.run` directly, gets "save the adapter again with safe_serialization=True".

**Base-model check.** Core applies whatever matches and checks nothing more: a module the model lacks is skipped with a console warning, and a shape that does not fit only logs an error while patching. Qwen3.5 / 3.8 models of different sizes share every module name, so the node checks that every module of the LoRA is a module of this model and that lora_B @ lora_A has the shape of its weight (the logical shape for INT8 and W4A8 weights), and otherwise stops: `this LoRA does not fit this model: <n> of <m> modules are not in this model (…); <k> of <m> have another shape (…). It was trained on another base model: pick the model it was trained on, or a LoRA trained on this one`, naming the first five. A Qwen3.5-9B adapter of 200 modules on Qwen3-VL-8B: 72 not in the model, 8 of another shape (core alone would apply the other 120); on Qwen3.5-4B or Qwen3.8-27B every name matches and no shape fits.

**Tied head.** Qwen3.5-4B ties its output head to its embedding (official config `tie_word_embeddings: true`), and core's build has no `lm_head` weight: the logits, and the MTP head's, are computed with `embed_tokens.weight`. A LoRA on `embed_tokens` would therefore patch the output head too (PEFT's model patches only the embedding), and one on `lm_head` has no weight of its own to patch. So on a model whose weights hold `embed_tokens` and no `lm_head` (told from the weights, not the model name) a LoRA touching either is refused: `this model ties lm_head to embed_tokens in ComfyUI core (it has no lm_head weight: the output head is embed_tokens.weight, as in Qwen3.5-4B), so a LoRA on embed_tokens / lm_head would patch both (…): use a LoRA trained without embed_tokens and lm_head`. A LoRA that also does not fit the model gets the base-model message above instead.

The console gets the plan of each run (an info line), e.g. `[BCNodes] Qwen LM: LoRA my_adapter: peft LoRA: 200 modules, rank 16, alpha 32 from adapter_config.json (scale alpha / rank 2)`, plus a line for `alpha_pattern`, `use_rslora` or left-out base weights when they apply. The LoRA is applied with core's own loader (`load_lora_for_models` on a copy that shares the base model's weights); the LoRA'd model is kept by the LoRA's path, its file and config file times, size and the strength, so an edited LoRA is applied again on the next run. The file is read and planned on every run, before the model loads, so those errors come first; the base-model check needs the loaded model's key map and runs when the LoRA'd copy is made.

### Memory

Core reserves the KV cache for prompt + `max_new_tokens` before the first token is generated: at 32768, about 1 GiB of VRAM for Qwen3.5-4B / 9B, 2 GiB for Qwen3.8-27B and 4.5 GiB for Qwen3-VL-8B (bf16), in proportion to the value. Lower `max_new_tokens` when the answers are short (a caption, a rewritten prompt) to keep that VRAM free.

`keep_model_loaded` on keeps the model, and its LoRA'd copy, loaded for the next run (ComfyUI's model management may still move it out of VRAM when another model needs the room); one model file is held at a time and another one replaces it. Off unloads it in a `finally` around everything after the model is known (its precision, the prompt checks, the file, the LoRA, the sampling, the images, loading and generating), so an error in any of them unloads too, a model an earlier run kept included; only an error found before that (a `models.yaml` that cannot be used, a model name the catalog does not have) leaves held what an earlier run kept. Unloading goes through core (`unload_model_and_clones` on the base model), which frees the base model and every LoRA'd copy of it; dropping only the copy would leave the model loaded.

The Process Monitor's [full clear](process-monitor.md) unloads it too (its `pack_models` step: the pack's hook unloads every LM backend), reported as the model file's name and the bytes of its weights; the next run loads the model again.

## `BC_LMConfig` — LM Config

`do_sample`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `presence_penalty`, `mtp`, `edited` (hidden) → `config` (`BC_LM_CONFIG`), for the `config` input of an LM node (Qwen LM). Only the fields you change, or feed by a link, are applied; every other field keeps the linked model's official default for its mode (thinking on / off). The widget defaults are Qwen3.5-9B's thinking-off values.

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `do_sample` | on | | On: sample with the settings below and the LM node's seed. Off: greedy, the most likely token at each step; core then applies no temperature, filter or penalty. |
| `temperature` | `0.7` | `0.01`–`2.0`, step `0.01` | Divides the logits: lower is more focused, higher more varied. |
| `top_k` | `20` | `0`–`1000` | Keep only the k most likely tokens; `0` = off. |
| `top_p` | `0.8` | `0.0`–`1.0`, step `0.01` | Keep the most likely tokens up to this total probability; `1.0` = off. Core keeps a slightly smaller set than the official implementation ([Known limits](#known-limits)). |
| `min_p` | `0.0` | `0.0`–`1.0`, step `0.01` | Drop the tokens less likely than `min_p` × the most likely one; `0` = off. |
| `repetition_penalty` | `1.0` | `0.0`–`5.0`, step `0.01` | The logit of a token already generated is divided by it (multiplied when negative); `1.0` = off. |
| `presence_penalty` | `1.5` | `0.0`–`5.0`, step `0.01` | Subtracted from the logit of a token already generated; `0` = off. Official advice: 0 to 2 against endless repetition; a higher value can mix languages and cost some quality. |
| `mtp` | `auto` | `auto`, `off`, `2`, `3`, `4`, `5` | Multi-token prediction ([MTP](#mtp)). |
| `edited` | empty | hidden | The fields changed on this node, comma-separated. Only these, and the fields fed by a link, are applied. |

**What is applied.** The `config` output is `{field: value}` for the fields named in `edited`, then every other field whose input is fed by a link, nothing else. The node reads the links from the prompt it runs in (ComfyUI's hidden `PROMPT` and `UNIQUE_ID` inputs), so a linked field is always applied: with `edited` empty, after *Reset to model defaults*, and in an API prompt whose `edited` does not name it. A call without a prompt (code calling the node directly, a node made by another node's expansion) applies `edited` alone. The frontend keeps `edited`: a change you make to a field's value adds it (a value the frontend sets itself, or renaming the field, does not). The node's menu item *Reset to model defaults* (in both renderers' node menu) clears `edited` and, when an LM node is linked, shows its defaults again. Through the API, set `edited` yourself, e.g. `"temperature, top_p"`: names are trimmed, empty ones skipped, a repeated one counted once, and an unknown name is an error that lists the fields. `edited` is an ordinary widget value and a link is part of the node's inputs, so both are part of ComfyUI's cache key; so is the node's id (ComfyUI adds it for a node with a hidden `UNIQUE_ID` input), so a new LM Config node with the same values runs the LM node again.

**What is shown.** The fields not in `edited` show the defaults of the first LM node the `config` output is linked to (the oldest link), for that node's model and thinking mode: refreshed when that link changes, when that node's model or `thinking` changes, on *Reset to model defaults* and on a node-definition refresh, never on load (a saved workflow's values come back as saved; only the edited and linked ones reach the model anyway). On the classic canvas they are drawn greyed and still take input; the Vue node renderer has no greyed look that still takes input, so there they look like edited fields. A default outside a widget's range (a `models.yaml` may give `temperature: 0`) is shown at the nearest end, since ComfyUI refuses a prompt with a widget value out of range; the model still gets the real default. One LM Config linked to several LM nodes shows the first one's defaults; each LM node applies its own model's defaults to the fields not edited.

## Sampling

Without an LM Config, sampling is the model's defaults for the mode (`thinking` on or off). With one, the fields named in its `edited` and the fields fed by a link replace those defaults and the others keep them. `seed` and `max_new_tokens` always come from the LM node.

The defaults (`models/qwen_lm/models.yaml`, each block with its source). `do_sample` is on for all (the official `generation_config` `do_sample: true`, the cards' `greedy=false`), `mtp` is `auto`:

| Model | Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B, 9B, 9B Abliterated | thinking off | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 | Qwen/Qwen3.5-9B model card, general tasks (README lines 833-836); the 4B card is identical; the abliterated build keeps its base model's |
| Qwen3.5-4B, 9B, 9B Abliterated | thinking on | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 | as above |
| Qwen3.8-27B | thinking off | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 | Qwen/Qwen3.8-27B model card (README lines 246-264) |
| Qwen3.8-27B | thinking on | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 | as above |
| Qwen3-VL-8B Instruct, Abliterated | thinking off (no thinking mode) | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 | Qwen/Qwen3-VL-8B-Instruct model card (README lines 134-153); it gives no min_p, so 0.0 (off) |

Core's sampling (`comfy/text_encoders/llama.py`), at every token:

- `do_sample` off, or a temperature of exactly 0 (only a `models.yaml` can give one; LM Config's minimum is 0.01): the most likely token, with no penalty, temperature or filter, and the seed unused.
- `do_sample` on, in this order: the penalties, the temperature, `top_k` (when above 0), `min_p` (when above 0), `top_p` (when below 1), then a draw with the seed.
- The penalties count only the tokens generated in this run (the reasoning included), never the prompt or the prefill. They are active when `repetition_penalty` is not 1 or `presence_penalty` is not 0; `repetition_penalty` divides a positive logit and multiplies a negative one, `presence_penalty` is subtracted.

An edited field the model cannot use gets one console warning naming it, and the run goes on: today that is `mtp` set to `auto` or `2`–`5` on a model without an MTP head, e.g. `[BCNodes] Qwen LM: mtp is auto in LM Config, but Qwen3-VL-8B Instruct has no MTP head: it runs without MTP`.

### MTP

Multi-token prediction: core's speculative decoding with the multi-token-prediction head a model file carries, for faster decoding. `auto` lets core adapt the draft depth, `2`–`5` pins it, `off` disables it. It runs only on a model the catalog marks as having an MTP head (Qwen3.5-4B, Qwen3.5-9B, Qwen3.8-27B) and only for a text-only prompt: with an image connected core decodes without MTP. Everywhere else the run goes without it (the node passes `off` for a model without the head; Qwen3-VL's generation ignores it). Sampled output with MTP differs from non-MTP output for the same seed; core keeps it correctly distributed. Core verifies the drafted tokens in one pass over several positions, which is not numerically identical to decoding one token at a time, so even greedy text (`do_sample` off) can part from the text with `mtp` `off` after some hundred tokens. Measured on an RTX 3090 with Qwen3.5-9B `INT8 ConvRot`: 35.5 against 21.6 tokens/s (about 1.65×), the same text up to about 190 tokens in that run.

## `models.yaml`

The built-in catalog is `models/qwen_lm/models.yaml` in the pack. Your changes go into a `models.yaml` in a `Qwen-LM` folder (`ComfyUI/models/Qwen-LM/models.yaml`, or in a folder `extra_model_paths.yaml` gives the key); a pack update leaves that file alone. Every `Qwen-LM` folder may hold one; they apply in folder order, a later one over an earlier one. Same schema for both:

```yaml
models:
  <display name>:
    template: qwen3.5 | qwen3.8 | qwen3-vl   # the chat template
    thinking: true | false                    # the model has a thinking mode
    mtp: true | false                         # the file carries an MTP head
    backend: core                             # optional; the family's default (core) when absent
    defaults:                                 # sampling per mode; thinking_on only on a thinking model
      thinking_off: {do_sample: true, temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: auto}
      thinking_on: {...}
    precisions:                               # in the order the precision widget lists them
      <precision name>: {file: <file name>, url: <download URL>}
```

`mtp` in a defaults block is `auto`, `off`, `2`, `3`, `4` or `5` (YAML reads a bare `off` as false, which counts as `off`). `file` is a plain file name, no folder; `url` an `http(s)` address. A built-in precision has a URL; one of yours may leave it out, and the file must then be in a `Qwen-LM` or `text_encoders` folder (it is searched as above).

Merge rules:

- An entry for a model the catalog has overrides it field by field: each field it gives replaces that field, `defaults` per mode and per sampling field, `precisions` per name (a name the model has is replaced, a new one is added after its others). `thinking: false` on a thinking model also drops its `thinking_on` defaults.
- An entry for a new model gives every field but `backend`; it is listed after the built-in models.
- An empty file, or one whose entries are all commented out, changes nothing.
- One console line per overridden model names what changed, e.g. `[BCNodes] LM catalog: .../models/Qwen-LM/models.yaml overrides Qwen3.5-9B: defaults.thinking_off.temperature, defaults.thinking_off.presence_penalty` and `... overrides Qwen3.5-4B: precisions.W4A8 (added)`.
- The files are read again only when one changes. New models and precisions reach the node's lists after *Refresh Node Definitions* (`R`) or a restart; a run reads the changed file at once.

An example, three changes in one file:

```yaml
models:
  # Override fields of a built-in model: only what is written here changes.
  Qwen3.5-9B:
    defaults:
      thinking_off: {temperature: 0.6, presence_penalty: 1.0}

  # Add a precision to a built-in model: a file you made yourself, so no url.
  Qwen3.5-4B:
    precisions:
      W4A8: {file: Qwen3.5-4B_w4a8.safetensors}

  # A new model: every field but backend.
  Qwen3.5-9B Finetune:
    template: qwen3.5
    thinking: true
    mtp: false
    defaults:
      thinking_off: {do_sample: true, temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: off}
      thinking_on: {do_sample: true, temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: off}
    precisions:
      BF16: {file: Qwen3.5-9B-finetune_bf16.safetensors, url: "https://huggingface.co/your-name/your-repo/resolve/main/Qwen3.5-9B-finetune_bf16.safetensors"}
```

The model file must be one core's text-encoder loader reads; the catalog does not check that, nor whether `mtp: true` matches the file.

Errors name the file and the entry and say what to change: a file that cannot be read, is not UTF-8 text (`save it as UTF-8`), is not valid YAML, is nested too deeply to read or holds a plain value YAML reads as a date or a number but cannot convert (such as `2026-13-01`: put it in quotes); a file that is not one `models:` key mapping names to entries; an unknown key (`models.Qwen3.5-9B.defaults.thinking_off: unknown key(s) temprature; the keys are do_sample, temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, mtp`); a new model missing a field (`a new model needs thinking, mtp, defaults, precisions (only backend may be left out)`); a template the family does not have (`template 'qwen4' is not one of this family's: qwen3.5, qwen3.8, qwen3-vl`); an unknown backend; a value of the wrong type (`thinking`, the model's `mtp` and `do_sample` true or false, `top_k` a whole number, a defaults `mtp` one of the six values, the other sampling fields finite numbers); a `file` with a folder in it; a `url` that is not http(s); `thinking_on` defaults on a model with `thinking: false`; a mode or a sampling field missing after the merge. A new model is checked complete in the file that adds it, so its errors name that file; an error only the merge shows (an override with `thinking: true` on a model that has no `thinking_on` defaults, say) names every file that wrote to the entry, in folder order. The catalog does not check the sampling values against LM Config's ranges.

A `models.yaml` with any of these errors, one that is not UTF-8 text included, does not hide the node: the console logs the error (`[BCNodes] Qwen LM: <error>. The model list holds the built-in models until it is fixed; a run raises this error.`), the lists show the built-in models, and every run raises the error until the file is fixed.

### Folders

```
ComfyUI/models/
  Qwen-LM/                                  the folder_paths key Qwen-LM (.safetensors)
    models.yaml                             your catalog changes (optional)
    Qwen3.5-9B_int8_convrot.safetensors     downloaded here on first use
    any/subfolder/                          model files are found in subfolders too
    loras/
      my_lora.safetensors                   a single-file LoRA
      my_lora.json                          its alpha, when the file has none: {"lora_alpha": 32, "r": 16}
      my_peft_adapter/                      a PEFT folder
        adapter_config.json
        adapter_model.safetensors
  text_encoders/                            searched for the model file (subfolders too); never downloaded into
```

`extra_model_paths.yaml` can redirect or extend `Qwen-LM` like any model folder:

```yaml
my_models:
  base_path: /mnt/models
  Qwen-LM: Qwen-LM
```

ComfyUI reads `extra_model_paths.yaml` before the node adds `models/Qwen-LM`, so such a folder comes first: it is searched first, downloads go into it, its `models.yaml` is applied first and its `loras/` is listed. A `Qwen-LM` key the YAML created without file extensions gets `.safetensors`.

## Known limits

- **top_p in core.** Core drops the token whose cumulative probability crosses `top_p`, a smaller nucleus than the official implementation (Hugging Face transformers) keeps: probabilities [0.5, 0.3, 0.2] with `top_p` 0.7 — core keeps {0.5}, transformers keeps {0.5, 0.3}. It is core's sampler and is not changed here.
- **Very large images.** An image the official sizing puts between 12,845,056 and 16,777,216 pixels is still capped by core's own resize to at most 12,845,056 pixels, bilinearly (4096 × 4096 → 3584 × 3584). Below that, core keeps the official size.
- **The Vue node renderer** shows LM Config's un-edited fields without the greyed look; they still show the defaults, and `edited` and the links still decide what is applied.
- **No prompt syntax.** Token weights such as `(word:1.2)`, `embedding:` names and `\(` escapes are not interpreted; the text is tokenized as written.
- **Silent cases.** MTP with an image connected, and edited sampling fields while `do_sample` is off, are not used and not warned about.
- **LM Config's shown defaults** do not follow a link through a legacy Reroute node or across a subgraph boundary; a `model` or `thinking` turned into an input shows the defaults of the widget's last value.
- **LoRAs.** One per node. Refused, besides the features above: keys without the `base_model.model.` prefix (as transformers' own adapter integration saves them); VeLoRA and MonteCLoRA adapters (extra tensors); a LoRA on `embed_tokens` or `lm_head` for a model whose core build ties the two (Qwen3.5-4B; [Base-model check](#lora)). When the metadata gives the alpha, a config file next to the LoRA is not read for `alpha_pattern` or `use_rslora` (it is still checked for refused features).
- **Reasoning effort.** Qwen3.8-27B with thinking on always uses the official default effort, `xhigh`; there is no widget for another.
- **Quantized kernels.** `INT8 ConvRot` and `W4A8` run on comfy-kitchen's CUDA kernels only with a torch built for CUDA 13.0 or later (cu130). With a cu12x torch ComfyUI 0.37 turns that backend off (`comfy/quant_ops.py`, with a console warning) and the slower fallback runs.
- **cuDNN attention.** Core runs attention on a large enough input (from 32 prompt tokens on Qwen3-VL-8B) with PyTorch's SDPA in the order flash, cuDNN, memory-efficient, math (`SDPA_BACKEND_PRIORITY` in `comfy/ops.py`); with an attention mask flash is out, so cuDNN runs, and on some GPU and cuDNN builds it builds no plan (seen with Qwen3-VL-8B on an RTX 3090, torch 2.11 cu130, cuDNN 9.19: `cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built.`). The first such error in an LM node's generation logs `[BCNodes] LM: cuDNN attention failed (<the error>); LM generation runs without cuDNN attention in this process from now on (core's other models keep it), and this one runs again`, and the run is repeated without cuDNN attention, from the same state (the failed attempt's prefetch state is dropped as ComfyUI does after a node; the KV cache and the seeded sampler are new in every run). Every later LM generation in that ComfyUI process runs without it from the start. It covers LM generation only: the list is changed for the length of an LM node's generate and put back after it, so core's other models keep cuDNN attention; no core file is changed and no launch flag is needed. Should it fail again, the run stops with `start ComfyUI with --use-split-cross-attention`, which takes core's attention off PyTorch's SDPA. Any other error passes through as it is.
- **Downloads** report in the console every 10%, not on the node's progress bar; core's generation does update the progress bar.
- **Process Monitor.** The full clear's report counts the model file's weights, not the LoRA's patch tensors. Emulate lists Qwen LM as not counted: its `model` widget names a catalog entry, not a file, so the model file's weights, the LoRA, the KV cache for prompt + `max_new_tokens` and the resized images are not in the estimate. LM Config is counted, with no tensor (a dict of the edited sampling fields).

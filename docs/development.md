# Development

The repository layout (the four layers `nodes/ -> pipelines/ -> models/ -> libs/`, what each module holds, the layer rule, how to add a node or a model) is in [CLAUDE.md](../CLAUDE.md), the one place it is kept.

## Install

The pack goes into a ComfyUI checkout's `custom_nodes/` ([Installation](../README.md#installation); ComfyUI 0.44.0 or later), or next to a ComfyUI checkout for the tests alone. Its requirements go into the Python ComfyUI runs on, from the repo root:

```
pip install -r requirements.txt
```

The tests need `postfx`, `caption-audit` and `anthropic` importable: `pip install -r requirements.txt`, or `PYTHONPATH=/path/to/postfx:/path/to/caption-audit` for local checkouts of the first two. Without `anthropic` the runtime test fails at its Claude sections (`claude_chat_routes` stops the run with an import error), and pytest skips the Claude tests (`tests/layers/pipelines/test_pipe_claude_messages.py`, `test_pipe_claude_chat.py`).

## Tests

The four gates, from the repo root. `COMFYUI_DIR=../ComfyUI` is a ComfyUI checkout next to the pack; with the pack inside `ComfyUI/custom_nodes/` it is `COMFYUI_DIR=../..`.

```
python tests/test_import_time.py                      # import budget, no heavy module, every module cold
python tests/test_nodes.py                            # every node with None / empty inputs, plus spot checks
COMFYUI_DIR=../ComfyUI python tests/test_runtime.py   # nodes through ComfyUI's validate_prompt + PromptExecutor; a SKIP is red
python -m pytest tests -q                             # layer rule, unit tests
```

Unit tests live in `tests/layers/<layer>/`, next to the layer of the code they call directly (`nodes/`, `pipelines/`, `models/`, `libs/`); `tests/pytest.ini` anchors pytest's rootdir in `tests/`, since the repo root is itself a package.

The runtime test needs a ComfyUI checkout (`$COMFYUI_DIR`, or `ComfyUI` next to this pack) with its requirements installed in the same Python; without one it prints SKIP, which counts as a failure. It starts no server and uses only core CPU nodes plus test-only stand-ins, and covers what only the real executor can prove. `main()` first runs the core checks: canvas-only slots (`In3`, `any_03`) reaching the node, wildcard sockets validating in both directions, list outputs fanning out, Select Switch running only the selected lazy branch, a genuine type mismatch still rejected, and each node's caching on a second queue. Then come the sections, one function each, in this order: `cond_pipe`, `lora_key_fix`, `vae_loader`, `latent_upscale`, `sage_attention`, `detail_boost`, `qwen_image_shift`, `dype_models`, `detailer_zimage_patch`, `detailer_qwen_fun_controlnet`, `detailer_qwen21_patch`, `detailer_controlnet_kinds`, `detailer_model_patch_kinds`, `lm_core`, `lm_nodes`, `claude_nodes`, `claude_chat_routes`, `preflight_nodes`, `unused_outputs`, `process_monitor_hook` and `process_monitor_clear` (`process_monitor_default` runs before the core checks). Each one's docstring says what it proves. The LM sections run against core's Qwen code without weights: `lm_core` (the prompt's tokens against core's own Qwen3.5 and Qwen3-VL tokenizers and the official Qwen3.5 tokenizer's ids where core's differ; every LoRA key form in the key maps of core models built on the meta device; the base-model and tied-head checks) and `lm_nodes` (the catalog route, the `Qwen-LM` folder, LM Config -> Qwen LM through `validate_prompt` + `PromptExecutor` with a stand-in backend).

Two scripts outside the suite:

```
python tests/parity_seedvr2_video.py --comfy ../ComfyUI --vae /path/to/seedvr2_vae.safetensors   # SeedVR2 VAE Encode / Decode / PostProcess against ComfyUI's nodes (GPU for the VAE)
python tests/compare_sage_attention.py                # SageAttention kernels against torch SDPA (CUDA GPU; SKIP without one)
```

## Import rule

Every module in `nodes/`, `pipelines/`, `models/` and `libs/` imports only `torch`, `numpy`, the standard library and relative pack modules at module level. Everything else (`PIL`, `cv2`, `safetensors`, `torchvision`, `yaml`, `av`, `folder_paths`, `comfy.*`, `comfy_extras.*`, the pip packages `postfx`, `caption_audit` and `anthropic`, and the optional packages) is imported inside the function that uses it, so the pack adds nothing to ComfyUI's startup. The one exception is the downloader's `server` / `aiohttp`, imported at module level inside a guarded `try` (ComfyUI has loaded both already). A node module also imports its pipeline, model and lib modules inside the methods that use them (a widget list such as the model names inside `INPUT_TYPES`), so the package import reads and compiles little more than `nodes/`; the Process Monitor's own modules load at import, where it registers and starts. `python tests/test_import_time.py` checks both, and `tests/test_layers.py` checks the layer rule and the module-level imports statically.

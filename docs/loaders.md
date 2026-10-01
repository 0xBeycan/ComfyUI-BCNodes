# Loaders

## `BC_PowerLoraLoader` — Power Lora Loader

`model` (`MODEL`) + LoRA rows → `MODEL`. No `CLIP` in or out: each LoRA is applied to the model only (`load_lora_for_models(model, None, …)`), which is what you want for models that do not take a CLIP LoRA anyway.

Each row is one line on the node: a toggle dot, the LoRA file (click to pick from `models/loras` — the list has a filter box), and one strength (`◀ ▶` steps by 0.05, click the number to type). `➕ Add LoRA` appends a row; a right click on a row opens its menu: *Toggle On/Off*, *Move Up*, *Move Down*, *Remove*. Rows are saved with the workflow as `{on, lora, strength}` and reach the backend as `lora_N` in row order; a row that is off, at strength 0, or whose file is missing is skipped (missing files are reported in the console).

## `BC_AutoModelDownloader` — Auto Model Downloader

One line per model: a URL, a directory under `ComfyUI/models` and two switches, *HF token needed?* and *Civitai token needed?*. `Add line` adds a line, `Remove last line` drops the last one. Two boxes at the top take a Hugging Face token and a Civitai token.

| Field | Example | Meaning |
| --- | --- | --- |
| `model_N` | `https://huggingface.co/owner/repo/resolve/main/model.safetensors` | Direct download URL, from `huggingface.co` or `civitai.com` (or their subdomains) only — a shared workflow carries its URLs, so any other host is refused. Hugging Face `blob/` links are rewritten to `resolve/`. |
| `dir_N` | `diffusion_models` or `sam3/nested` | Directory under `models/`; created if missing. Write `dir/name.safetensors` to save the file under another name. |
| `hf_N`, `civitai_N` | `HF token needed?`, `Civitai token needed?` | Mark files that need a token: gated Hugging Face repos, Civitai downloads. |
| `HF token`, `Civitai token` | — | Paste, press enter. Saved on the server (`user/BCNodes/downloader_tokens.json`, mode 600), never into the workflow, never shown again — the box reads `(saved)`. An empty value clears it. |

The file name comes from the URL; when the URL has none (Civitai-style links) write it after the directory.

What happens:

- **First open.** When a workflow with this node is opened and some of its models are missing, one dialog lists them with a `Download` / `Not now` choice. Files marked as needing a token are tagged, and the dialog shows the token boxes with the files each one is needed for; a download does not start while a required token is missing. `Download` runs the downloads with a progress bar per file. The answer is remembered on the server (`user/BCNodes/downloader_seen.json`, keyed by the model list), so it is asked once per list, on any browser or URL. Nothing is asked when every file is already there.
- **On the node.** The button reads `Download all models`, `Download missing models (n)` or `All models downloaded`; the status row names the missing files, or the token that is still needed (`needs Civitai token: x.safetensors`) — the button stays disabled until it is entered. A download started from the button reports to the browser console and to the status row, and ends with a small done dialog.
- **When queued.** The node is an output node: running the workflow — from the UI or through the API with no browser — downloads whatever is still missing before finishing, and fails with a clear message if a required token is not stored.

Downloads stream to `name.part` and are renamed when complete; an interrupted download resumes. The Hugging Face token is sent as a bearer header to huggingface.co, the Civitai token is appended to civitai.com links.

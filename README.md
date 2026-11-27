# ComfyUI-BCNodes

Utility nodes for ComfyUI, mostly for image work: nodes that prepare, generate, refine, check and save images, and nodes that keep a workflow simple. Every node is built to be fast and light on RAM and VRAM. The nodes sit under `BCNodes/` in the node menu, and every registration key starts with `BC_`, so the pack installs next to other node packs without a clash.

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/0xBeycan/ComfyUI-BCNodes
pip install -r ComfyUI-BCNodes/requirements.txt
```

Run `pip` with the Python ComfyUI runs on, then restart ComfyUI. Needs ComfyUI 0.44.0 or later.

`requirements.txt` holds the four packages the pack uses that ComfyUI does not ship. Each one is imported only when a node uses it, so the pack loads without any of them:

| Package | Used by | Without it |
| --- | --- | --- |
| `opencv-python` | MaskGrow, Mask Fill Holes, Image Quality Gate, PostFx Apply's mask | those nodes stop with Python's import error for `cv2` when run (the `opencv-python-headless` that `postfx` installs provides `cv2` too) |
| `anthropic` (Anthropic's official SDK) | Claude, Claude Chat | Claude stops with Python's import error when run; a Claude Chat message shows that error |
| [`postfx`](https://github.com/0xBeycan/postfx) | the five PostFx nodes | PostFx Apply, PostFx Theme and PostFx Signature Sheet are missing from the node list (the console logs why); PostFx Custom Look and PostFx LUT stop with an install message when run |
| [`caption-audit`](https://github.com/0xBeycan/caption-audit) | Caption Audit | Caption Audit draws an error card that names the package and returns `critical` 1 |

Optional packages, in no requirements file:

| Package | Enables | Without it |
| --- | --- | --- |
| SageAttention 2.2.x ([install](docs/model.md#install)) | Patch Sage Attention | the model keeps ComfyUI's attention; the console says why |
| `sageattn3` | Patch Sage Attention's `sageattn3 (experimental)` mode | the same |
| `nvidia-vfx` (with an NVIDIA RTX GPU) | Image Resize's `nvidia_rtx_vsr` | that method stops with an install message |
| `pillow-avif-plugin`, `pillow-jxl-plugin` | Save Image's `.avif` and `.jxl` | those extensions are not offered |
| `nvidia-ml-py` | the Process Monitor's GPU load meter | no GPU load meter; on CUDA the device-wide VRAM comes from `cudaMemGetInfo` |

## Documentation

Each node has a section in the page of its menu category, under a heading with its registration key.

- [Logic](docs/logic.md) — booleans, arithmetic, switches and the seed
- [Mask](docs/mask.md) — mask checks, fill / grow / draw / blockify / repeat, BiRefNet background removal
- [Image](docs/image.md) — scaling and resizing, image lists, depth maps, social media export, saving, skin texture, frequency merge
- [PostFx](docs/postfx.md) — film-emulation looks: themes, custom looks, LUTs, signature sheets
- [Analysis](docs/analysis.md) — the caption set audit and the image quality gate
- [Text](docs/text.md) — prompt lists and showing text
- [LM](docs/lm.md) — chat models through ComfyUI core: Qwen LM, LM Config, LM Loader, the models and their files, sampling, LoRAs, `models.yaml`; Claude through Anthropic's API with your own key: the request, images, refusals, Claude Config, and Claude Chat, the Claude tab in the left sidebar
- [PreFlight](docs/preflight.md) — how Instagram, TikTok and X would likely treat an image or video before it is published: the observation sensor, the rules engine, reading a verdict, the feedback loop and calibration
- [Loaders](docs/loaders.md) — LoRA loading (with the key fix for LoRAs core loads only in part), VAE loading per model (the Wan 2.1 upscale 2x VAE) and the model downloader
- [Latent](docs/latent.md) — latent upscaling with a model (Wan 2.1 2x) for a second sampling pass
- [Model](docs/model.md) — per-model patches: Patch Sage Attention (the modes, `qk_quant`, long sequences, the calls that run the original attention, `nan_guard`, the console, install, an A/B test), Detail Boost, Qwen Image Shift, Dual Sampling Shift, DyPE Patcher and DyPE Config
- [Sampling](docs/sampling.md) — Dual Sampler: a two-pass hires fix for any model, an optional second model for the second pass, the upscale (latent, pixel or a blend of the two), DyPE on the second pass (Z-Image, Qwen-Image), the chain of core nodes it replaces
- [Detailer](docs/detailer.md) — a mask region or every face sampled again at the model's size and pasted back: the crop, the sampling, ControlNet hints, the paste; the character and `model_no_lora`, the size gate, the face and eye passes
- [SeedVR2](docs/seedvr2.md) — the downscale factor from the framing, the SeedVR2 input stage, streaming VAE encode / decode, the chunk size and the VAE tile for the card, post-processing
- [Workflow](docs/workflow.md) — image comparer, Anything Everywhere, group bypass, the conditioning pipe, Auto Bypass, and the Align buttons in the selection toolbox (not a node)
- [Unused outputs](docs/unused-outputs.md) — a whole-batch image or mask output that nothing is connected to comes out empty, and its step does not run where it is a step of its own
- [Process Monitor](docs/process-monitor.md) — not a node: RAM / VRAM live in the top bar, an estimate before a run, per-node measurement, the report of a killed run, the full clear back to the startup memory, and the drop stale outputs setting
- [Development](docs/development.md) — installing for development, the tests and gates, the import rule

## Third-party code and license

MIT ([`LICENSE`](LICENSE)). These folders hold code from other projects and keep their own license:

- `models/birefnet/arch/` — the BiRefNet architecture and its Swin v1 backbone; MIT, notices in [`models/birefnet/arch/LICENSE`](models/birefnet/arch/LICENSE).
- `models/depth_anything_v2/arch/` — the Depth Anything V2 DPT head and its DINOv2 ViT-S backbone; Apache-2.0, notices and license text in [`models/depth_anything_v2/arch/LICENSE`](models/depth_anything_v2/arch/LICENSE).
- `models/wan_latent_upscale/arch/` — spacepxl's Wan 2.1 latent upscaler; MIT, in [`models/wan_latent_upscale/arch/LICENSE`](models/wan_latent_upscale/arch/LICENSE).
- `libs/dype/`, `models/z_image/dype/` and `models/qwen_image/dype/` — DyPE's rotary tables and the Z-Image and Qwen-Image embedders, derived from [ComfyUI-DyPE](https://github.com/wildminder/ComfyUI-DyPE) (wildminder); Apache-2.0, the license text and the list of changes in each folder's `LICENSE` and `NOTICE`.

Everything else in this pack is BCNodes' own code.

# ComfyUI-BCNodes

Utility nodes for ComfyUI, in one small pack.

| Registration key | Display name | What it does |
| --- | --- | --- |
| `BC_AutoBypass` | Auto Bypass | Bypasses its targets automatically when a watched source is empty |
| `BC_LogicBoolean` | Logic Boolean | 0–1 float → `BOOLEAN` / `NUMBER` / `INT` / `FLOAT` |
| `BC_IsMaskEmpty` | Is Mask Empty | `MASK` → `BOOLEAN` |
| `BC_MaskFillHoles` | Mask Fill Holes | Fills enclosed holes in a mask |
| `BC_MaskGrow` | MaskGrow | Grows / shrinks a mask, then blurs it |
| `BC_ImageScaleByAspectRatio` | Image Scale By Aspect Ratio | Scales an image / mask to an aspect ratio and side length; `MASK` output is never `None` |
| `BC_JoinImageLists` | Join Image Lists | Concatenates image lists, unlimited inputs |
| `BC_MathExpression` | Math Expression | Arithmetic over `a`, `b`, `c` without `eval()` |
| `BC_PromptList` | Prompt List | One prompt per line, as a list |
| `BC_AnySwitch` | Any Switch | First connected non-None input, any type, unlimited inputs |
| `BC_Seed` | Seed | Seed widget; `-1` draws a new random seed on every run |
| `BC_ShowText` | Show Text | Shows incoming text on the node, passes it on |
| `BC_ImageComparer` | Image Comparer | Two images on the node, compared with a sliding divider |
| `BC_VideoComparer` | Video Comparer | Two frame batches (+ optional audio) played as one clip, compared with a sliding divider |
| `BC_PowerLoraLoader` | Power Lora Loader | `MODEL` + any number of LoRA rows → `MODEL`, no CLIP |
| `BC_AnythingEverywhere` | Anything Everywhere | Feeds unconnected inputs of a type at prompt time |
| `BC_FastGroupsBypasser` | Fast Groups Bypasser | One bypass toggle per group |
| `BC_BiRefNetRemoveBackground` | BiRefNet Remove Background | Background removal with BiRefNet, plain torch |
| `BC_SeedVR2Resize` | SeedVR2 Resize | Original image → the padded frame SeedVR2 encodes (lanczos downscale, shortest-edge antialiased bicubic, pad 16, 4n+1 frames) plus the colour reference |
| `BC_SeedVR2VAEEncode` | SeedVR2 VAE Encode | SeedVR2 VAE encode with the frames streamed from RAM slice by slice, so VRAM does not grow with the frame count |
| `BC_SeedVR2VAEDecode` | SeedVR2 VAE Decode | SeedVR2 VAE decode with every decoded slice streamed to RAM, so VRAM does not grow with the frame count |
| `BC_SeedVR2PostProcess` | SeedVR2 PostProcess | Post-Process SeedVR2 Output one frame at a time into one float16 output, no full-size temporaries |
| `BC_AutoModelDownloader` | Auto Model Downloader | Lists a workflow's models and fetches the missing ones into `models/` |
| `BC_PostFxApply` | PostFx Apply | Applies a `postfx` film-emulation look (theme + condition + strength) to an image batch, optional look override and mask |
| `BC_PostFxTheme` | PostFx Theme | Built-in `postfx` theme → `POSTFX_LOOK`, to start a chain from a named look |
| `BC_PostFxCustomLook` | PostFx Custom Look | Builds a look from common controls, or overrides them on top of an incoming look |
| `BC_PostFxLut` | PostFx LUT | 3D `.cube` LUT as a look, standalone or layered on a theme |
| `BC_PostFxSignatureSheet` | PostFx Signature Sheet | Labeled contact sheet of every theme in a category applied to one image |
| `BC_CaptionAudit` | Caption Audit | Runs `caption-audit` over a LoRA caption folder and draws the report card on the node |
| `BC_SocialMediaExport` | Social Media Export | One platform-ready derivative per ticked platform, minimum crop, spec-driven |
| `BC_ImageQualityGate` | Image Quality Gate | Blur / sharpness / noise / clipping / entropy → `PASS` / `SO-SO` / `FAIL` badge, verdict and scores |
| `BC_SaveImage` | Save Image | Saves images with folder / file names built from prompt widget values, any Pillow format, prompt + workflow embedded; preview only in the gallery, never under the node |
| `BC_SkinTexture` | Skin Texture | Micro-texture on skin inside a SAM 3 mask: boosts the image's own detail and multiplies in a synthetic pore field, in linear light |
| — | Align | Align / distribute buttons in the selection toolbox |

Registration keys are BCNodes' own, so the packages above can be installed side by side without a clash. Type `BCNodes` in the node library to see them all; in the menu they sit in these groups:

| Category | Nodes |
| --- | --- |
| `BCNodes/logic` | Logic Boolean, Math Expression, Any Switch, Seed |
| `BCNodes/mask` | Mask Fill Holes, MaskGrow, Is Mask Empty, BiRefNet Remove Background |
| `BCNodes/image` | Image Scale By Aspect Ratio, Join Image Lists, Social Media Export, Save Image, Skin Texture |
| `BCNodes/postfx` | PostFx Apply, Theme, Custom Look, LUT, Signature Sheet |
| `BCNodes/analysis` | Image Quality Gate, Caption Audit |
| `BCNodes/text` | Prompt List, Show Text |
| `BCNodes/loaders` | Power Lora Loader, Auto Model Downloader |
| `BCNodes/seedvr2` | SeedVR2 Resize, VAE Encode, VAE Decode, PostProcess |
| `BCNodes/workflow` | Image Comparer, Video Comparer, Anything Everywhere, Fast Groups Bypasser, Auto Bypass |

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/0xBeycan/ComfyUI-BCNodes
```

```
pip install -r ComfyUI-BCNodes/requirements.txt
```

Restart ComfyUI. `requirements.txt` holds the two packages that back the PostFx and Caption Audit nodes, [`postfx`](https://github.com/0xBeycan/postfx) and [`caption-audit`](https://github.com/0xBeycan/caption-audit); everything else ships with ComfyUI. Without them the pack still loads and only those nodes are absent. If this repository is still present under its old name `ComfyUI-AutoBypass`, delete that folder — its `AutoBypass` node is this pack's `BC_AutoBypass`.

The Align buttons are not a node; they appear in the toolbox above a multi-selection.

## Nodes

### `BC_LogicBoolean` — Logic Boolean

A `FLOAT` widget in `[0, 1]` (default `1`) is rounded to a boolean.

| Output | Value for widget `0.7` |
| --- | --- |
| `BOOLEAN` | `True` |
| `NUMBER` | `1` |
| `INT` | `1` |
| `FLOAT` | `0.7` (unrounded) |

### `BC_IsMaskEmpty` — Is Mask Empty

`MASK` → `BOOLEAN`. `True` when the mask is missing or every pixel is `0`.

### `BC_MaskFillHoles` — Mask Fill Holes

`masks` (`MASK`, optional) → `MASKS`. Fills every fully enclosed hole of each mask in the batch (`scipy.ndimage.binary_fill_holes`); a gap touching the border is not enclosed and stays open. Output is hard `0/1`, `float32`, shape `(B, H, W)`. A missing mask (`None`, an empty tensor, or nothing wired) returns `torch.zeros((1, 64, 64))`.

### `BC_MaskGrow` — MaskGrow

`mask` (`MASK`, optional), widgets `invert_mask` (default `False`), `grow` (`-999..999`, default `4`), `blur` (`0..999`, default `4`) → `mask`.

Optional inversion, then `|grow|` iterations of grey dilation (positive) or erosion (negative) with a cross-shaped 3×3 kernel, then a Gaussian blur of radius `blur`. A missing mask returns `torch.zeros((1, 64, 64))`.

### `BC_ImageScaleByAspectRatio` — Image Scale By Aspect Ratio

`image` (`IMAGE`, optional), `mask` (`MASK`, optional), widgets `aspect_ratio` (`original` / `custom` / `1:1` / `3:2` / `4:3` / `16:9` / `2:3` / `3:4` / `9:16`), `proportional_width` and `proportional_height` (the `custom` ratio), `fit` (`letterbox` / `crop` / `fill`), `method` (`lanczos` / `bicubic` / `hamming` / `bilinear` / `box` / `nearest`), `round_to_multiple` (`8` … `512` / `None`), `scale_to_side` (`None` / `longest` / `shortest` / `width` / `height` / `total_pixel(kilo pixel)`), `scale_to_length` (default `1024`), `background_color` (default `#000000`, the letterbox fill).

| Output | Type | Value |
| --- | --- | --- |
| `image` | `IMAGE` | `(B, H, W, 3)` |
| `mask` | `MASK` | `(B, H, W)`; zeros of the output size when no usable mask came in |
| `original_size` | `BOX` | `[width, height]` of the input |
| `width` | `INT` | output width |
| `height` | `INT` | output height |

The aspect ratio fixes the shape, `scale_to_side` and `scale_to_length` fix the size (fractions truncated), then both sides are rounded **up** to `round_to_multiple`. `letterbox` fits the whole image inside and pads with `background_color`, `crop` centre-crops to the target ratio, `fill` stretches.

Two things worth knowing: the `MASK` output is never `None` — with no mask wired, or ComfyUI's 64×64 placeholder mask, it is `torch.zeros((B, H, W))` — and a mask that does not match the image size, or no image and no mask at all, raises instead of returning `None` on every output.

### `BC_JoinImageLists` — Join Image Lists

Concatenates image lists (`INPUT_IS_LIST`) and returns the joined list plus each input's length:

| Output | Type | Meaning |
| --- | --- | --- |
| `Joined` | `IMAGE` list | all inputs, in slot order |
| `Sizes` | `INT` list | length of each connected input |

Slots grow on their own: the node starts with `In1` and `In2`; when both are connected `In3` appears, when `In3` is connected `In4` appears, and so on with no limit. Disconnecting a slot in the middle removes it and renumbers the rest. `In1` and `In2` are required.

The backend accepts any number of `InN` inputs, so API prompts can list `In3`, `In4`, … directly; the frontend part only manages the slots on the canvas.

### `BC_BiRefNetRemoveBackground` — BiRefNet Remove Background

`image` (`IMAGE`), `model` (combo) → `IMAGE` (RGBA with alpha = matte, or RGB over a solid colour), `MASK` (the matte, `(B, H, W)`), `MASK_IMAGE` (the matte as an RGB image).

Options, applied to the matte in this order: `sensitivity` (below 1 the matte is amplified, so faint areas count as foreground), `mask_blur` (Gaussian blur of the edges, pixels), `mask_offset` (grow / shrink by one pixel per step), `invert_output` (keep the background instead), `refine_foreground` (hardens the matte edge and scales the colours by it, for cleaner cut-outs on transparent output), `background` (`Alpha` → RGBA output; `Color` → RGB output over `background_color`), `background_color` (colour picker, `#rrggbb`; the picker needs frontend 1.28 or newer — older frontends show the input as a socket and the default `#222222` is used).

The BiRefNet architecture lives in [`birefnet/`](birefnet/) as a plain `nn.Module` and the weights are loaded straight from safetensors — no `transformers`, no `trust_remote_code`, no `timm`.

| Model | Backbone | Inference size |
| --- | --- | --- |
| `BiRefNet-general` (default) | swin_v1_l | 1024 |
| `BiRefNet_512x512` | swin_v1_l | 512 |
| `BiRefNet-HR` | swin_v1_l | 2048 |
| `BiRefNet-portrait` | swin_v1_l | 1024 |
| `BiRefNet-matting` | swin_v1_l | 1024 |
| `BiRefNet-HR-matting` | swin_v1_l | 2048 |
| `BiRefNet_lite` | swin_v1_t | 1024 |
| `BiRefNet_lite-2K` | swin_v1_t | 2048 |
| `BiRefNet_dynamic` | swin_v1_l | 1024 |
| `BiRefNet_lite-matting` | swin_v1_t | 1024 |
| `Lucida` | swin_v1_l | 1024 |

Weights are fetched from Hugging Face on the node's first run — never at import — into `ComfyUI/models/background_removal/<name>.safetensors` (ComfyUI's own folder for its core BiRefNet nodes, so the weights are shared), through the pack's own downloader (resumable). The model stays loaded between runs and is swapped when a different one is selected. fp16 on CUDA, fp32 elsewhere.

Licenses: the BiRefNet and Swin Transformer code is MIT (see [`birefnet/LICENSE`](birefnet/LICENSE)); the checkpoints are published under MIT by their authors.

### `BC_MathExpression` — Math Expression

`expression` (multiline) plus optional `a`, `b`, `c` (`INT`, `FLOAT`, `IMAGE` or `LATENT`) → `INT`, `FLOAT` (the same value, truncated and as a float). The result is also drawn on the node. Re-evaluated on every run.

The expression is parsed with `ast` and walked with a whitelist — there is no `eval()`. Allowed: numbers; `+ - * / // % **`, `& | ^ << >>`; unary `- + ~ not`; `and` / `or`; comparisons (`== != < <= > >=`, chained) which yield `1` / `0`; the names `a`, `b`, `c`; `a.width` / `a.height` for an `IMAGE` or `LATENT` input (latent sizes are multiplied by 8); `NodeTitle.widget` to read another node's widget by node title or type; and the functions `min max abs round int float pow sqrt floor ceil randomint randomchoice iif`. Anything else — subscripts, strings, lambdas, other attributes or functions — is a `ValueError` naming the offending piece. An empty expression evaluates to `0`; a referenced input that is not connected is an error.

### `BC_PromptList` — Prompt List

`prepend_text`, `multiline_text`, `append_text`, `start_index`, `max_rows` → `prompt` (list), `body_text` (list), `show_help` (string). One entry per line of `multiline_text`, windowed by `start_index` / `max_rows`; `prompt` wraps each line in the prepend / append texts, `body_text` is the bare line. Both lists are `OUTPUT_IS_LIST`, so downstream nodes run once per line.

### `BC_AnySwitch` — Any Switch

Wildcard inputs `any_01`, `any_02`, … → the first one that is connected and not `None`. Nothing connected → `None`. Slots grow as they are connected (one empty slot always waits at the end; empty slots in the middle are removed and the rest renumbered), and the socket type follows whatever is connected so the canvas shows and checks the real type. Useful with an optional branch: wire the optional source first and a fallback second.

### `BC_Seed` — Seed

`seed` (`INT`, `-1 … 0xffffffffffffffff`, default `0`) → `SEED`. Three buttons under the widget:

| Button | Effect |
| --- | --- |
| `🎲 Randomize Each Time` | Sets the widget to `-1`: every queued run gets a fresh random seed. |
| `🎲 New Fixed Random` | Writes a concrete random seed into the widget; it then stays fixed. |
| `♻️ (Use Last Queued Seed)` | Puts the seed of the last queued run back into the widget. Greyed out and in parentheses until a run has used a seed that differs from the widget; after a random run it reads `♻️ <seed>`. |

`-1` never travels: right before a prompt is sent, the frontend replaces it with a drawn seed in the API prompt and in the workflow copy that ends up in image metadata, so dropping a saved image onto the canvas brings the real seed back. A prompt queued through the API with `-1` gets the same treatment on the server. Random seeds are drawn below 2⁵³ so they survive the round trip through JavaScript exactly. The node re-evaluates every run; downstream nodes re-run only when the seed actually changes.

### `BC_ShowText` — Show Text

`text` (`STRING`, input only) → `STRING`. Shows the text in a read-only, growing box on the node and passes it on. A list input shows one box per element and is passed on as a list. The shown text is also written into the workflow metadata of saved images.

### `BC_ImageComparer` — Image Comparer

`image_a`, `image_b` (both optional) → shown on the node, drawn on the canvas itself so it moves with the node. A fills the node; while the pointer is over it, B is painted from the left edge up to the pointer with a divider line and A / B tags; leave the node and A shows alone. With more than one image per side a row of `A1 A2 B1 …` labels above the image picks the pair. The node keeps the size you give it; the image is letterboxed inside. The images are written to ComfyUI's temp folder like Preview Image does; the comparison lives with the run (it survives a tab switch, not a restart) and nothing is saved into the workflow file. Output node, no outputs.

### `BC_VideoComparer` — Video Comparer

`video_a`, `video_b` as **`IMAGE` frame batches** (optional), `audio` (`AUDIO`, optional) plus `fps` (default 24). Feed it what *Load Video → Get Video Components* gives, or any frames you generated — no video pack needed. Both batches are written into **one** H.264 MP4 in ComfyUI's `temp/` folder, A and B side by side in the same frame (PyAV, which ComfyUI already depends on), so the browser decodes a single stream and the two sides cannot drift apart. Clips are cut to the shorter one, a smaller frame is letterboxed into the larger one, odd sizes lose one row / column, and the audio is muxed in as AAC, cut to the clip.

On the node: the two halves layered with the same divider as the image comparer — the divider follows the pointer only while it is over the video. Nothing plays until asked: the ▶ / ⏸ button in the top row or a click on the video toggles playback, the seek bar at the bottom scrubs, the clip loops. With audio connected a speaker button in the top row mutes it. Clips of different length show a note. Output node, no outputs.

### `BC_PowerLoraLoader` — Power Lora Loader

`model` (`MODEL`) + LoRA rows → `MODEL`. No `CLIP` in or out: each LoRA is applied to the model only (`load_lora_for_models(model, None, …)`), which is what you want for models that do not take a CLIP LoRA anyway.

Each row is one line on the node: a toggle dot, the LoRA file (click to pick from `models/loras` — the list has a filter box), and one strength (`◀ ▶` steps by 0.05, click the number to type). `➕ Add LoRA` appends a row; a right click on a row opens its menu: *Toggle On/Off*, *Move Up*, *Move Down*, *Remove*. Rows are saved with the workflow as `{on, lora, strength}` and reach the backend as `lora_N` in row order; a row that is off, at strength 0, or whose file is missing is skipped (missing files are reported in the console).

### `BC_AnythingEverywhere` — Anything Everywhere

One node, any number of sources. Each source wired into the node is handed to **every unconnected input of the same type** in the workflow when the prompt is built — root graph and subgraph nodes alike. A connected slot takes the type and colour of its link (`VAE`, `CLIP`, …) and an empty `anything` slot is always kept at the bottom for the next one. The node properties `title_regex` and `input_regex` (right click → *Properties Panel*) narrow the targets by node title and input name; a node with a regex takes precedence over one without. A bypassed or muted Anything Everywhere does nothing.

On the canvas the node shows what it reaches: inputs it feeds get a glowing ring in the link's colour, every other free input a small dot (it could be fed), the node itself a green badge in its title bar (yellow when a regex restricts it, dim when it feeds nothing). The translucent phantom links from the node to the inputs it feeds are drawn when the node or the target is selected or under the pointer. Settings → *BCNodes › Anything Everywhere*: *Show links* (all off / selected nodes / mouseover node / selected and mouseover nodes / all on) and *Highlight connected and connectable inputs*.

Implementation: the API prompt that `graphToPrompt` returns is patched with the extra links, so queueing and *Export (API)* both contain them; no real links are drawn. Limits: the Anything Everywhere node and its source must sit in the root graph; a source that is a subgraph node's output is not resolved (a console warning says so).

**API / serverless mode:** the links only exist because the frontend wrote them into the prompt. A prompt exported from the frontend (*Export (API)*) carries them and runs anywhere. A prompt assembled without the frontend has no such links, so any input that depended on Anything Everywhere is simply missing and validation fails with `Required input is missing`. The Python side of the node is a no-op that is never executed.

### `BC_FastGroupsBypasser` — Fast Groups Bypasser

One toggle per group — the groups of the graph the node sits in and the groups inside every subgraph: on = the group's nodes are active, off = bypassed (subgraph nodes inside a group are switched together with their contents). The list follows the graph on a half-second tick — new, renamed and removed groups, and modes changed by other means. Right-click menu: *Bypass all*, *Enable all*, *Toggle all*.

Properties: `sort` (`position`, `alphanumeric` or `custom alphabet`), `customSortAlphabet` (letters, or comma-separated prefixes, that order the `custom alphabet` sort), `matchColors` (comma-separated group colours — names such as `red` or hex values — only matching groups are listed), `matchTitle` (regex; same), `showAllGraphs` (off = only the groups of the graph on screen), `toggleRestriction` (`default`, `max one` — switching a group on switches the others off — or `always one` — same, and the last active group cannot be switched off).

**API / serverless mode:** bypass is a frontend concept. When the frontend builds the prompt, bypassed nodes are already left out and their links routed around them, so an exported prompt reflects the toggles at export time and runs anywhere. A prompt assembled without the frontend cannot be switched by this node; it contains whatever nodes it was given. The Python side of the node is a no-op that is never executed.

### `BC_SeedVR2Resize` — SeedVR2 Resize

`image` (`IMAGE`, the original frames), widgets `upscale_factor` (default `2`), `downscale_factor` (default `0.5`, `1` = none), `max_resolution` (cap on the longest edge, `0` = none, default `4096`), `emulate_bf16` (default on).

| Output | Type | Value |
| --- | --- | --- |
| `image` | `IMAGE` | the frames SeedVR2 encodes: downscaled, resized, clamped to `[0, 1]`, zero-padded to a multiple of 16, frame count padded to 4n+1 by repeating the last frame; stored as float16 (VAE Encode casts to float16 anyway) — wire to *SeedVR2 VAE Encode* (video) or *VAE Encode (Tiled)* (single image) |
| `reference` | `IMAGE` | the colour-correction reference: float32 resize of the same frames stored as float16, cropped to even width / height, original frame count — wire to *SeedVR2 PostProcess* `original_resized_images` |

One node for the whole input stage of the SeedVR2 graph: `ImageScaleBy(lanczos, downscale_factor)` (PIL LANCZOS on 8-bit, exactly ComfyUI's), then a shortest-edge resize with `resolution = shortest edge of the original × upscale_factor` — `torchvision` `resize` with the shortest edge at `resolution` (the long edge is floored), `BICUBIC` with antialias, a second resize to `round(edge × max_resolution / longest)` when the longest edge exceeds `max_resolution` — then the clamp, the pad to a multiple of 16 and the 4n+1 frame padding. ComfyUI's own scale nodes cannot match the middle step: `bicubic` there is `F.interpolate` without antialias (a different cubic kernel) and *Resize Image* rounds the long edge. The resize runs in bfloat16 on the GPU for the frame that gets encoded and again in float32 on the CPU for the colour reference; with `emulate_bf16` on and a CUDA device, `image` carries the bfloat16-rounded values, otherwise it is a float32 resize. Both outputs are stored as float16 (VAE Encode casts `image` to float16 anyway; the float16 rounding of `reference` moves the colour transfer by ~0.1/255), which halves the RAM a long clip takes. Frames go through the GPU four at a time, so a long video batch works.

### `BC_SeedVR2VAEEncode` — SeedVR2 VAE Encode

`pixels` (`IMAGE`, the frames from *SeedVR2 Resize* `image`), `vae` (`VAE`, the SeedVR2 VAE), widgets `tile_size` (default `1024`), `overlap` (default `256`), `temporal_size`, `temporal_overlap` (the four of *VAE Encode (Tiled)*; the temporal two are ignored, as they are there for this VAE) → `LATENT` `(1, 16, T', H/8, W/8)`, float32.

ComfyUI's *VAE Encode (Tiled)* moves the whole clip to the GPU first (after a full float32 `x × 2 − 1` copy in RAM), then encodes it slice by slice, so VRAM grows with the frame count on top of the encoder's fixed working set (over 28 GB for a whole 1080p frame, measured on a 32 GB card): 897 frames at 1080p do not fit a 32 GB card in one tile. This node runs the same slice loop with the same causal memory cache and the same `x × 2 − 1`, but builds each 4-frame input slice on the GPU only when its turn comes; the posterior mode, the crop to the latent size and the `× scaling_factor` follow the native path. VRAM is the encoder's working set whatever the length, and the float32 copy in RAM is gone. Spatial tiling is the native one: the same tile grid, cosine blend on the latent grid and count normalisation as `tiled_vae`, tile by tile with one causal cache at a time, the blend accumulated in float32 in RAM; a `tile_size` that covers the frame means one tile and no blend. At 1080p a whole-frame tile (2048) just fits a 32 GB card; a 1024 tile needs about half. `tests/parity_seedvr2_video.py --vae … --tile …` checks it against `VAE.encode_tiled` (expected: identical latent) and prints both peak VRAMs.

### `BC_SeedVR2VAEDecode` — SeedVR2 VAE Decode

`samples` (`LATENT`), `vae` (`VAE`, the SeedVR2 VAE), widgets `tile_size` (default `1024`), `overlap` (default `256`), `temporal_size`, `temporal_overlap` (the four of *VAE Decode (Tiled)*; the temporal two are ignored, as they are there for this VAE) → `IMAGE` `(B×T, H, W, 3)`, float16, `[0, 1]`.

ComfyUI's *VAE Decode (Tiled)* on the SeedVR2 VAE decodes the whole clip in one call and keeps every decoded frame on the GPU until the end (`slicing_decode` collects the slices and `torch.cat`s them), so VRAM grows with the frame count — a 32 GB card tops out near 190 frames at 1080p. This node runs the same slice loop with the same causal memory cache, the same `/ scaling_factor`, even crop and `(x + 1) / 2` clamp, but moves each decoded slice to RAM as soon as it exists. VRAM is then the decoder's working set for one tile (over 31 GB for a whole 1080p frame, which no 32 GB card fits — 2048 fails at the first slice on a 5090; ~11 GB with a 1024 tile), whatever the length; the clip length is bounded by RAM instead. The frames are stored as float16, which is what the float16 VAE produced — the native node only upcasts them. Spatial tiling is the native one (same latent-grid tiles, same pixel-grid cosine blend and count normalisation as `tiled_vae`), tile by tile with one causal cache at a time; the blend is accumulated in float16 in RAM (12.5 MB per 1080p frame while the node runs) and normalised in float32 per 4-frame chunk, so the result equals the native tiled decode within float16 rounding. `tests/parity_seedvr2_video.py --vae … --tile …` checks it against `VAE.decode_tiled` (expected: equal after rounding to float16) and prints both peak VRAMs.

Both VAE nodes check the free VRAM against the working set first and, when it is short, unload every other model (the DiT is staged in RAM by ComfyUI and reloads on demand); ComfyUI's own loader will not evict one "dynamic" model for another, which is what made the native decode fail next to a resident DiT. With enough VRAM nothing is unloaded.

Measured on a 5090 (32 GB, 92 GB RAM), 720p → 1080p, 7B int8 DiT, both tiles 1024/256: 901 frames (30 s at 30 fps) in one piece, 18 min end to end, linear at ~36 s per second of video; the native tiled decode topped out near 190 frames on the same card. VRAM is flat over the length, RAM is what bounds it (about 61 MB per 1080p frame across the graph's kept outputs, plus a fixed ~13 GB).

Both log their progress next to the progress bar: one line at the start (`897 frames, 6 tile(s) x 224 slice(s)` — the tile count shows at once whether `tile_size` covers the frame) and then every 10 s and at the end (`slice 48/1344, tile 1/6, frame 193/897, VRAM 11.2 GiB used, 41 s elapsed, ~1100 s left`; the VRAM figure is the driver's, what `nvidia-smi` shows).

### `BC_SeedVR2PostProcess` — SeedVR2 PostProcess

`images` (`IMAGE`), `original_resized_images` (`IMAGE`, the reference), widget `color_correction_method` (`lab` / `wavelet` / `adain` / `none`) → `images` (`IMAGE`, float16).

*Post-Process SeedVR2 Output* builds five full-size float32 copies of the clip on the way through (the raw range conversion, the flattening reshape, the result buffer and the add / div / clamp chain), which is what runs a 30-second 1080p clip out of RAM. This node does the same operations in the same order — frame count and size cropped to the reference, `x × 2 − 1`, the colour transfer from `comfy/ldm/seedvr/color_fix.py` on the VAE device, `(x + 1) / 2` clamp, alpha taken from the reference, even crop — one frame at a time into a single preallocated float16 output. The colour maths itself stays float32 per frame, exactly as in the native node. `tests/parity_seedvr2_video.py` checks every method against the native node: equal after rounding to float16 (max difference 2.4e-4, a sixteenth of an 8-bit step).

### `BC_AutoModelDownloader` — Auto Model Downloader

One line per model: a URL, a directory under `ComfyUI/models` and two switches, *HF token needed?* and *Civitai token needed?*. `Add line` adds a line, `Remove last line` drops the last one. Two boxes at the top take a Hugging Face token and a Civitai token.

| Field | Example | Meaning |
| --- | --- | --- |
| `model_N` | `https://huggingface.co/owner/repo/resolve/main/model.safetensors` | Direct download URL. Hugging Face `blob/` links are rewritten to `resolve/`. |
| `dir_N` | `diffusion_models` or `sam3/nested` | Directory under `models/`; created if missing. Write `dir/name.safetensors` to save the file under another name. |
| `hf_N`, `civitai_N` | `HF token needed?`, `Civitai token needed?` | Mark files that need a token: gated Hugging Face repos, Civitai downloads. |
| `HF token`, `Civitai token` | — | Paste, press enter. Saved on the server (`user/BCNodes/downloader_tokens.json`, mode 600), never into the workflow, never shown again — the box reads `(saved)`. An empty value clears it. |

The file name comes from the URL; when the URL has none (Civitai-style links) write it after the directory.

What happens:

- **First open.** When a workflow with this node is opened and some of its models are missing, one dialog lists them with a `Download` / `Not now` choice. Files marked as needing a token are tagged, and the dialog shows the token boxes with the files each one is needed for; a download does not start while a required token is missing. `Download` runs the downloads with a progress bar per file. The answer is remembered on the server (`user/BCNodes/downloader_seen.json`, keyed by the model list), so it is asked once per list, on any browser or URL. Nothing is asked when every file is already there.
- **On the node.** The button reads `Download all models`, `Download missing models (n)` or `All models downloaded`; the status row names the missing files, or the token that is still needed (`needs Civitai token: x.safetensors`) — the button stays disabled until it is entered. A download started from the button reports to the browser console and to the status row, and ends with a small done dialog.
- **When queued.** The node is an output node: running the workflow — from the UI or through the API with no browser — downloads whatever is still missing before finishing, and fails with a clear message if a required token is not stored.

Downloads stream to `name.part` and are renamed when complete; an interrupted download resumes. The Hugging Face token is sent as a bearer header to huggingface.co, the Civitai token is appended to civitai.com links.

### `BC_PostFxApply` — PostFx Apply, and the look nodes

Film-emulation looks from the [`postfx`](https://github.com/0xBeycan/postfx) pipeline: film stocks, cinematic grades and `.cube` LUTs defined in YAML, running on the CPU. Pick a look, dial a shooting condition and a global strength, and every image gets the same reusable visual identity.

| Node | In → Out | What it does |
| --- | --- | --- |
| PostFx Apply | `IMAGE` (+ `look`, `mask`) → `IMAGE` | The core node. Applies a **theme** + **condition** + **strength** to an image batch. `theme = none` passes the image through untouched. A connected `look` overrides the theme dropdown; an optional `mask` limits the effect to the masked region (an all-black mask is ignored). |
| PostFx Theme | → `POSTFX_LOOK` | Emits a built-in theme as a look, to start a chain from a named theme. |
| PostFx Custom Look | (`look`) → `POSTFX_LOOK` | Builds a look from common controls (white balance, exposure, contrast, vibrance / saturation, grain, vignette, halation, clarity). With a `look` input, only the knobs moved off neutral override it. |
| PostFx LUT | (`look`) → `POSTFX_LOOK` | Attaches a 3D `.cube` LUT. Standalone by default; connect a `look` to layer the LUT on top of a theme. |
| PostFx Signature Sheet | `IMAGE` → `IMAGE` | Labeled contact-sheet grid of every theme in a category, for side-by-side comparison. |

`POSTFX_LOOK` is the link type between the look-producing nodes and PostFx Apply; every look node has an optional `look` input, so they chain in any order:

```
PostFx Theme (portra_400) → PostFx LUT (my_look.cube) → PostFx Custom Look (grain ↑) → PostFx Apply (condition = neon_night) → Save Image
```

- **Theme** = the look (colour / grain / lens): 3 texture-only `grain` finishes (fine per-pixel grain, colour untouched — the default social-still finish; chain a signature theme before one for a look), 15 `signature` film stocks and industry grades, 15 `experimental`.
- **Condition** scales only the texture (grain, chroma noise, halation) for the shooting situation — `neutral`, `day_outdoor`, `overcast`, `indoor_evening`, `neon_night`, `night_flash` — and leaves colour alone.
- **Strength** `0–1.5` blends the whole effect with the original. **Seed** makes grain deterministic; `batch_seed = increment` gives each frame of a batch its own grain.
- **LUTs**: drop `.cube` 3D LUTs into this repo's [`luts/`](luts/) folder to see them in the PostFx LUT dropdown, or point `lut_path` at any absolute path. A LUT applies mid-pipeline, so a theme's grade runs before it and grain / vignette / sharpen finish on top.

### `BC_CaptionAudit` — Caption Audit

Audits a LoRA caption set before a training run, with the [`caption-audit`](https://github.com/0xBeycan/caption-audit) package. A token that appears in nearly every caption stops being a describable attribute: the model cannot tell it apart from the trigger word, so it bakes the concept into the identity — it can no longer be prompted in or out. That is a caption distribution problem, and nothing downstream of the dataset fixes it.

A caption set is a **folder**: images plus `.txt` sidecars sharing each image's basename. The node takes the path, runs the audit in-process and draws the report card as a preview inside the node.

| Input | Default | Meaning |
| --- | --- | --- |
| `directory` | *(empty = ComfyUI's cwd)* | The caption set to audit. Confined to the ComfyUI tree and the roots named in `BC_CAPTION_ROOTS` — see below. |
| `trigger` | *(empty)* | The trigger word this LoRA owns. Empty = inferred and marked **[INFERRED]** on the card. |
| `class_words` | — | Comma-separated, e.g. `woman, car`. Shown as `EXPECTED`, never flagged; coverage is measured per word. |
| `fuse` | — | Comma-separated attributes you *want* welded to the trigger, e.g. `red scarf`. Shown as `INTENDED`, excluded from `critical`. |
| `critical_threshold` / `warn_threshold` / `info_threshold` | `0.85` / `0.60` / `0.35` | Document-frequency cut-offs for fused / strong bias / not reported. |
| `ngram_max` | `3` | Longest phrase analysed; phrases never cross a comma. |
| `no_stopwords` | `false` | On shows function and relational words too. |
| `recursive` | `false` | Descend into subdirectories. |
| `table_rows` | `12` | Rows in the card's term table — the only input that changes the card's size, so the node never resizes between runs. |
| `images_dir` *(optional)* | — | Where the images live when captions are kept apart. With no images anywhere the audit runs in caption-only mode. |

`directory` and `images_dir` are widget values, and a widget value comes out of the workflow JSON — a graph you downloaded could otherwise point the node at any folder on the machine and read the caption text back out through `report_text` / `report_json`. So both paths are resolved with `realpath` and must land inside the ComfyUI tree (the ComfyUI root, `input/`, `output/`, `user/`). Datasets normally live somewhere else, and the way to allow one is the **`BC_CAPTION_ROOTS`** environment variable — `:`-separated roots on Linux and macOS, `;`-separated on Windows — read from the environment ComfyUI starts in, never from the graph:

```bash
BC_CAPTION_ROOTS=/path/to/datasets python main.py
```

A path outside those roots is refused on the error card with the roots listed, the same as a missing folder.

Outputs: `report_image` (the card), `report_text` (the complete terminal report), `report_json` (identical to `caption-audit --format json`), `critical` and `warning` counts. `critical` exists to stop a workflow: wire it into a compare node ahead of the training branch. If the audit cannot run (missing folder, no `.txt` files, thresholds out of order) the node renders an error card and returns `critical = 1` instead of raising — a set that could not be checked has not been cleared.

The card ends with the question the tool refuses to answer: every flagged term has two possible causes with opposite fixes — caption hygiene (delete the word where it is not the point) or dataset composition (collect contrast data) — and telling them apart means looking at the images, which neither the node nor the package does.

### `BC_SocialMediaExport` — Social Media Export

An output node: a full-quality master image in, one platform-ready derivative per ticked platform out, with the **minimum possible cropping** and a controlled, high-fidelity encode — so the platform serves the file as-is instead of re-cropping and re-compressing it. Wire the master into your normal save node *and* into this one; the input passes through untouched on the `images` output and `report` lists one aligned line per (image, platform).

Each platform is an **aspect-ratio band** inside a pixel envelope (`nodes/social_specs.json`, re-read on every execution): a master inside the band is scaled only; outside it, `resize_mode` decides — `crop` trims the minimum on one axis, `pad` keeps every pixel on a blurred cover-scaled copy of itself. `quality` (default 92) is the starting JPEG / WebP quality; a platform with a byte cap steps it down to fit. `allow_upscale` enlarges small masters toward the envelope. Files land under `output/` as `<filename_prefix>_<platform>_00001_.<ext>`. 4:4:4 chroma, progressive encoding and a slightly top-weighted crop anchor are always on.

### `BC_ImageQualityGate` — Image Quality Gate

Single-node quality control for AI-generated images, built for filtering LoRA training sets. Five metrics — blur (block-wise Laplacian variance, optionally **center-weighted** so background bokeh does not inflate the score), sharpness (Laplacian + Tenengrad), noise (Gaussian difference), highlight / shadow clipping and Shannon entropy — each scored against a threshold, folded into a three-tier verdict:

| Verdict | `verdict` | Meaning |
| --- | --- | --- |
| PASS | `2` | every metric within its threshold |
| SO-SO | `1` | no hard failure, but a metric sits in the margin zone (within 1.4× of its threshold) |
| FAIL | `0` | a metric is past the margin zone |

`shot_type` presets (`close-up`, `medium`, `wide / full-body`) scale the sliders — a close-up is judged stricter on blur and sharpness, a wide shot looser — and `custom` uses the raw values. `blur_var_threshold` is the per-block Laplacian variance that counts as blurry: 20–50 for AI images (distilled models have inherently lower variance), 80–150 for photographs. Outputs: a colour-coded `badge` image with the per-metric breakdown, the integer `verdict` for a Switch node, a text `report` and the five raw scores.

### `BC_SaveImage` — Save Image

`images` (`IMAGE`), optional `positive_text_opt` / `negative_text_opt` (`STRING`, saved into the job data), no outputs. Widgets, in this order: `filename_prefix` (default `ComfyUI`), `filename_keys` (default `sampler_name, cfg, steps, %F %H-%M-%S`), `foldername_prefix`, `foldername_keys` (default `ckpt_name`), `delimiter` (one character, default `-`), `save_job_data` (`disabled` / `prompt` / `basic, prompt` / `basic, sampler, prompt` / `basic, models, sampler, prompt`), `job_data_per_image`, `job_custom_text`, `save_metadata`, `counter_digits` (0–8), `counter_position` (`last` / `first`), `one_counter_per_folder` (unused, kept for widget order), `image_preview`, `output_ext`, `quality` (0–100), `named_keys`.

Folder and file names are built from comma-separated keys. Each key is one of:

| Key | Gives |
| --- | --- |
| `cfg`, `sampler_name`, `ckpt_name`, any widget name | that widget's value; when several nodes have it, the highest-numbered node wins |
| `13.cfg` | the widget of node 13 (falls back to the search above when node 13 is absent) |
| `ckpt_path`, `lora_path`, `control_net_path` | the folder part of the matching `*_name` widget |
| `resolution` | `WxH` of the first image |
| `%F %H-%M-%S` | `strftime` of the run's timestamp |
| `'text'` | a fixed string, quotes kept |
| `/key`, `./key`, `../key` | steps into a subfolder before `key`; a bare `/` is a separator |
| anything else | kept as a literal |

Model names lose their `.safetensors` / `.ckpt` / `.pt` / `.bin` / `.pth` extension, floats are trimmed to 10 significant digits, `named_keys` writes `seed=123` instead of `123`, and `*?:"<>|` are dropped. The counter continues from the highest number already in the folder for that name and extension; `counter_digits` `0` writes a fixed name and overwrites.

`output_ext` lists `.webp` (default), `.png`, `.jpg`, `.jpeg`, `.j2k`, `.jp2`, `.gif`, `.tiff`, `.bmp`, plus `.avif` and `.jxl` when `pillow-avif-plugin` / `pillow-jxl-plugin` are installed. `quality` is the encoder quality for the lossy formats (`100` = lossless for WebP / AVIF / JXL) and maps to PNG compression level 0–9. With `save_metadata` on, the prompt and workflow go into PNG text chunks or, for the other formats, into the EXIF `Make` and `ImageDescription` tags (BMP has neither); ComfyUI loads PNG and WebP back into the editor.

`save_job_data` appends an entry per run to `jobs.json` in the folder (or one `<image>.json` per image with `job_data_per_image`): `basic` = prefix + resolution, `models` = checkpoint / LoRAs / VAE / upscale model, `sampler` = seed / steps / cfg / sampler / scheduler / denoise, `prompt` = the two `*_text_opt` inputs or, when neither is wired, the `text` widgets behind a KSampler's positive / negative links.

`image_preview` only decides whether the saved images are listed in the queue / history gallery. Nothing is ever drawn under the node, so the node keeps the size it was given (`web/js/save_image.js` switches the frontend's output preview off for this node type). Errors while writing raise.

### `BC_SkinTexture` — Skin Texture

Rendered skin comes out as a smooth gradient, and grain laid on top of it reads as noise over plastic. This node puts a surface under the grain. Inside a skin mask it does two things, both in linear light and both as ratios, so colour is untouched and nothing moves:

- **detail** — boosts the image's own high-frequency luminance (a 2 px high-pass at a 1024 px long edge, scaled with the image) so the fine structure the model did render stops being flat.
- **texture** — multiplies in a synthetic pore field: two band-passed noise octaves at pore scale plus sparse darker pits, unit variance, `1.0` = ±6% luminance modulation. `pore_scale` sets the pore size (1 = about one pixel at 1024 px, scaled with the image).

The mask comes from SAM 3: `sam3_model` is a checkpoint under `models/checkpoints` (the default `sam3.1_multiplex_fp16.safetensors` is downloaded on first use), prompted with `skin` minus `eyes, eyebrows, lips, teeth`; a `face` detection sets the strength — full when the face spans about a third of the frame height, fading as it gets smaller, because pore-scale detail has nowhere to live on a small face. Connect `mask` to skip the detection (body masks from your own SAM 3 prompts, for instance); `exclude_mask` is subtracted either way; `feather` softens the edge. Highlights get 30% of the effect and black none. The mask that was used comes out as `skin_mask`.

Order in a still pipeline: Skin Texture → upscale → PostFx grain last. Keep the clean image for I2V; texture and grain are for the published still.

### Align

With two or more items selected (nodes, groups, reroutes, subgraph nodes), the selection toolbox gains eight buttons: align left / horizontal centers / right / top / vertical centers / bottom, and — from three items — distribute horizontally / vertically. A group moves with its contents. Each action is one undo step.

For keeping things tidy while dragging, ComfyUI's own **Settings → LiteGraph → Canvas → Always snap to grid** does the job; the toolbox's **Arrange** menu re-stacks a selection vertically, horizontally or as a grid.

### `BC_AutoBypass` — Auto Bypass

A frontend-only node that watches a source (LoadImage, VHS_LoadVideo, ...) and flips its targets between **ACTIVE** and **BYPASS** automatically: empty source → targets bypassed, source loaded → targets active.

No Python execution. The node is virtual — it never appears in the prompt sent to the backend; it only rewrites `mode` on the nodes wired into it. It lives entirely in [`web/js/auto_bypass.js`](web/js/auto_bypass.js).

#### The problem it solves

Nodes such as `ImageResizeKJv2` have a **required** `image` input. When the upstream loader is bypassed, the link is gone and prompt validation fails with `Required input is missing: image`. In a workflow with an optional branch (a reference image that is sometimes there, sometimes not) you end up opening the subgraph and bypassing the resize node by hand every run. `Auto Bypass` does that for you.

#### Inputs

| Slot | Type | Meaning |
| --- | --- | --- |
| `watch` | `*` | Output of the source to observe (e.g. `LoadImage.IMAGE`). |
| `force` | `BOOLEAN`, optional | Overrides the watch check: `true` → ACTIVE, `false` → BYPASS. See the note below. |
| `mode` | `COMBO`, optional | Socket of the `mode` widget. Wire it, or promote the widget out of a subgraph so it is set from the parent graph. See the note below. |
| `target_1..N` | `*` | Output of each node to control. Dynamic: connecting the last slot opens a new empty one, empty slots in the middle are removed. |

#### Widgets

| Widget | Values | Meaning |
| --- | --- | --- |
| `mode` | `auto` / `force_enable` / `force_bypass` | `auto` follows the decision below. The two `force_*` values pin the targets regardless of inputs. |
| `status` | read-only | Current result and why, e.g. `BYPASS (image empty) -> 2 targets`. |

#### Decision order

1. `mode` is `force_enable` / `force_bypass` → that.
2. `force` input is connected and resolves to a boolean → that.
3. Otherwise the `watch` source is **empty** when any of these holds:
   - `watch` is not connected;
   - the source node's mode is MUTE (2) or BYPASS (4);
   - the source's file widget (`image`, `video`, `audio`, `file`, `filename`, `model_file`, `path`, `url`) is an empty string / `None`.

Empty → every target is set to BYPASS. Not empty → every target is set to ACTIVE.

#### Behavior

- Reroutes and virtual pass-through nodes (KJNodes Set/Get and the like) are followed to the real source, with cycle protection.
- Links that enter a subgraph through its input panel are followed out to the parent graph, so the node can live inside a subgraph while the loader sits outside.
- Every instance in the root graph and in every subgraph is evaluated together, on a 500 ms tick, on connection changes, and once more right before the prompt is built — so what gets queued always reflects the current state.
- A target that is itself a subgraph node gets its inner nodes set as well (the frontend does not propagate a subgraph node's mode into its body on its own).
- If `Auto Bypass` itself is muted or bypassed it stops touching its targets and says so in `status`.
- A source that is also listed as a target is not treated as "empty" because of its own mode — otherwise it would lock itself in BYPASS.

#### Note on `mode`

`mode` is a widget with a socket. Inside a subgraph you can promote it (or drag its socket to the subgraph's input panel); the value then lives on the subgraph node in the parent graph and wins over the inner widget. The same works through nested subgraphs. When the socket is linked to a node instead, the value is read from that node's widget if it carries one of the three mode strings; otherwise the inner widget value applies.

#### Note on `force`

The frontend cannot see values computed during execution. `force` only resolves when the connected node carries the boolean as a widget — a Primitive node, a BOOL constant node, and similar. If it comes from a node that computes the value at run time (e.g. `Is Mask Empty`) it cannot be read; `status` reports `force unresolved` and the watch check applies instead.

## Development

```
ComfyUI-BCNodes/
  __init__.py              assembles the mappings, nothing else
  nodes/
    logic.py               BC_LogicBoolean, BC_IsMaskEmpty
    mask.py                BC_MaskFillHoles, BC_MaskGrow
    image_scale.py         BC_ImageScaleByAspectRatio
    lists.py               BC_JoinImageLists
    math_expression.py     BC_MathExpression
    prompt_list.py         BC_PromptList
    any_switch.py          BC_AnySwitch
    seed.py                BC_Seed
    show_text.py           BC_ShowText
    image_comparer.py      BC_ImageComparer
    video_comparer.py      BC_VideoComparer
    power_lora_loader.py   BC_PowerLoraLoader
    everywhere.py          BC_AnythingEverywhere, BC_FastGroupsBypasser (no-ops)
    seedvr2.py             BC_SeedVR2Resize, BC_SeedVR2VAEEncode, BC_SeedVR2VAEDecode, BC_SeedVR2PostProcess
    common.py              wildcard type + flexible optional inputs
    birefnet.py            BC_BiRefNetRemoveBackground (download, load, run)
    downloader.py          BC_AutoModelDownloader + its HTTP routes
    postfx.py              BC_PostFxApply, BC_PostFxTheme, BC_PostFxCustomLook, BC_PostFxLut, BC_PostFxSignatureSheet
    caption_audit.py       BC_CaptionAudit (audit plumbing, the fixed-size card, the node)
    social_media_export.py BC_SocialMediaExport (the ComfyUI adapter)
    social_export_core.py  its geometry / encoding engine, no ComfyUI or torch imports
    social_specs.json      the platform table it reads on every run
    image_quality_gate.py  BC_ImageQualityGate
    save_image.py          BC_SaveImage
    skin_texture.py        BC_SkinTexture (SAM 3 skin mask + texture engine)
  birefnet/                BiRefNet + Swin v1 architecture (see LICENSE in the folder)
  luts/                    drop .cube LUTs here for PostFx LUT (gitignored)
  web/js/
    auto_bypass.js         the BC_AutoBypass virtual node
    join_image_lists.js    unlimited slots for Join Image Lists
    any_switch.js          unlimited slots + type following for Any Switch
    math_expression.js     result overlay for Math Expression
    seed.js                Seed buttons + prompt rewrite of -1
    show_text.js           Show Text boxes
    comparer.js            image / video comparer widget
    power_lora_loader.js   LoRA rows
    fast_groups_bypasser.js  group toggles
    anything_everywhere.js prompt-time input filling
    auto_model_downloader.js  node UI, first-open dialog, progress
    align.js               toolbox align / distribute buttons
    save_image.js          no output preview under Save Image
  locales/en/main.json     tooltips for the Align buttons
  tests/
    test_import_time.py    import gate
    test_nodes.py          every node with None / empty input, plus numeric goldens
                           for MaskGrow and Image Scale By Aspect Ratio
    test_runtime.py        headless ComfyUI: real validation + execution
    test_social_export.py  the Social Media Export planner and encoder, plain pytest
    parity_seedvr2_video.py  BC_SeedVR2VAEEncode / VAEDecode / PostProcess vs ComfyUI's own nodes, numerically
```

Tests:

```
python tests/test_import_time.py                  # import budget and heavy-module ban
python tests/test_nodes.py                        # None / empty input never raises unexpectedly
COMFYUI_DIR=../ComfyUI python tests/test_runtime.py   # nodes through ComfyUI's validate_prompt + PromptExecutor
python -m pytest tests/test_social_export.py      # Social Media Export geometry / encoding
python tests/parity_seedvr2_video.py --comfy ../ComfyUI --vae ../ComfyUI/models/vae/seedvr2_ema_vae_fp16.safetensors   # SeedVR2 VAE Encode / Decode / PostProcess against ComfyUI's nodes (GPU for the VAE)
```

The runtime test needs a ComfyUI checkout with its requirements installed in the same Python; it starts no server. It covers the things that only the real executor can prove: canvas-only slots (`In3`, `any_03`) reaching the node, wildcard sockets validating in both directions, list outputs fanning out, and that a genuine type mismatch is still rejected.

Node modules import only `torch`, `numpy` and the standard library at module level; `scipy`, `PIL`, `cv2`, `safetensors`, `torchvision`, `folder_paths`, `comfy.*` and the pip packages `postfx` / `caption_audit` are imported inside the functions that use them (the downloader also touches `server` / `aiohttp`, which ComfyUI has loaded already), so the pack adds nothing to ComfyUI's startup. `python tests/test_import_time.py` checks that.

## Third-party code

- `birefnet/` — the BiRefNet architecture and its Swin v1 backbone, MIT, notices in [`birefnet/LICENSE`](birefnet/LICENSE).

Everything else in this pack is BCNodes' own code.

## License

MIT. The code in `birefnet/` keeps its own MIT notice, in `birefnet/LICENSE`.

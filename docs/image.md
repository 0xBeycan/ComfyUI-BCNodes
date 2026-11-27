# Image

Half-precision input (float16 or bfloat16, e.g. BCVideoNodes' Load Video or SeedVR2 PostProcess): Image Resize, Depth Anything, Skin Texture and Frequency Merge read it a frame at a time as float32 and work in float32; float32 input works as before. A float16 frame whose values all lie within 1/16 of an 8-bit level is 8-bit data (Load Video's) and is read as those levels, exactly its float32 source; any other half frame as its exact values. Image Scale By Aspect Ratio, Image Resize, Depth Anything and Skin Texture give their outputs in the input's dtype, Frequency Merge in its two inputs' promoted dtype ([below](#bc_frequencymerge--frequency-merge)).

The 8-bit conversions (Save Image, Save Image With Caption, Social Media Export, the lanczos resizes, Image Scale By Aspect Ratio) read a half frame the same way, then convert as for float32, so an 8-bit clip given as float16 gives exactly the uint8 of its float32 source: float16(1/255) x 255 is 0.99998, which a plain cast truncates to 0. A bfloat16 value gets 1/2 added before the truncation (its error on a level is up to 0.498).

Image Scale By Aspect Ratio, Image Resize and Skin Texture leave out an image or mask output that nothing is connected to: [Unused outputs](unused-outputs.md).

## `BC_ImageScaleByAspectRatio` — Image Scale By Aspect Ratio

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

## `BC_ImageResize` — Image Resize

`image` (`IMAGE`), widgets `width` and `height` (`0..16384`, default `512`), `upscale_method` (`nearest-exact` / `bilinear` / `area` / `bicubic` / `lanczos` / `nvidia_rtx_vsr`), `keep_proportion` (default `stretch`, below), `pad_color` (default `0, 0, 0`), `crop_position` (`center` / `top` / `bottom` / `left` / `right`), `divisible_by` (`0..512`, default `2`); optional `mask` (`MASK`) and `device` (`cpu` / `gpu`).

| Output | Type | Value |
| --- | --- | --- |
| `IMAGE` | `IMAGE` | the resized batch |
| `width` | `INT` | output width |
| `height` | `INT` | output height |
| `mask` | `MASK` | the mask resized with the image; without one, the padding (`1` = padding) or a `(1, 64, 64)` zero mask |

| `keep_proportion` | Output |
| --- | --- |
| `stretch` | exactly `width × height`; a `0` keeps that side of the source |
| `resize` | the largest size inside `width × height` at the source aspect; a `0` side is free |
| `total_pixels` | `width × height` pixels at the source aspect |
| `crop` | `width × height`; the source is first cropped to that aspect, the window placed at `crop_position` |
| `pad` | as `resize`, then padded out to `width × height` with `pad_color`, the image placed at `crop_position` |
| `pad_edge` | as `pad`; the padding is the mean of the image's first / last row and column |
| `pad_edge_pixel` | as `pad`; the edge pixels are repeated outwards |
| `pillarbox_blur` | as `pad`; the padding is the frame itself scaled to cover, blurred, 20 % desaturated and dimmed to 35 % |

Both sides are then floored to a multiple of `divisible_by` (`0` or `1` = off); a padded image instead grows its right / bottom padding up to the next multiple. `nearest-exact`, `bilinear`, `area` and `bicubic` resample through torch on `device`; `lanczos` through PIL on 8-bit frames, as ComfyUI's own lanczos does, on the CPU only (`gpu` with `lanczos` raises); `nvidia_rtx_vsr` through NVIDIA RTX Video Super Resolution, which needs the `nvidia-vfx` package and an NVIDIA RTX GPU and rounds the size to a multiple of 8 (not exercised by the tests, which run without one). `pad_color` takes `r, g, b` (all values in 0–1 are scaled by 255, otherwise 0–255), `#rrggbb` / `#rrggbbaa` (the `#` optional), a colour name or one grey value, with one value per image channel (four for RGBA).

The mask follows the image: scaled bilinearly to the image's size first when it differs, then cropped and resampled with it, padded with its own edge values (`1` around the frame for `pillarbox_blur`). ComfyUI's all-zero 64×64 placeholder counts as no mask.

Frames are processed one at a time into one preallocated output; nothing is split into sub-batches and joined again. Deliberate choices:

- An image already at the output size is returned as is, without a copy; an image or crop window that needs no resampling is not resampled, so `lanczos` does not round it through 8 bit.
- Only an all-zero 64×64 mask is taken as the placeholder; a real 64×64 mask is resized like any other.
- `pillarbox_blur` returns one mask per incoming mask frame, not one per image frame.
- A `pad_color` that cannot be read, or whose value count does not match the image's channels, raises with what to fix instead of padding black; it is read only when `pad` actually pads.
- Settings that would give a side of 0 pixels raise.

## `BC_JoinImageLists` — Join Image Lists

Concatenates image lists (`INPUT_IS_LIST`) and returns the joined list plus each input's length:

| Output | Type | Meaning |
| --- | --- | --- |
| `Joined` | `IMAGE` list | all inputs, in slot order |
| `Sizes` | `INT` list | length of each connected input |

Slots grow on their own: the node starts with `In1` and `In2`; when both are connected `In3` appears, when `In3` is connected `In4` appears, and so on with no limit. Disconnecting a slot in the middle removes it and renumbers the rest. `In1` and `In2` are required.

The backend accepts any number of `InN` inputs, so API prompts can list `In3`, `In4`, … directly; the frontend part only manages the slots on the canvas.

## `BC_DepthAnythingV2` — Depth Anything

`image` (`IMAGE`), `resolution` (`INT`, default `518`, 14–2044, step 14); optional `model` (default `v2-small`) and `width`, `height` (`INT` sockets, both or neither) → `depth` (`IMAGE`, grayscale in all three channels).

| `model` | Model | Weights, fetched on first use |
| --- | --- | --- |
| `v2-small` | Depth-Anything-V2-Small, this pack's own code | `depth_anything_v2_vits.pth` (~99 MB) into `ComfyUI/models/geometry_estimation/` |
| `v3-small` | DA3-Small, through ComfyUI core | `depth_anything_3_small.safetensors` (137,254,980 B) |
| `v3-base` | DA3-Base, through ComfyUI core | `depth_anything_3_base.safetensors` (541,524,124 B) |
| `v3-mono-large` | DA3Mono-Large, through ComfyUI core | `depth_anything_3_mono_large.safetensors` (1,336,748,056 B) |
| `v3-metric-large` | DA3Metric-Large, through ComfyUI core | `depth_anything_3_metric_large.safetensors` (1,336,748,056 B) |

Size: with `width` and `height` not connected, the depth map's short side is `resolution` and its long side keeps the input's aspect, each side rounded half to even — the size comfyui_controlnet_aux's Depth Anything preprocessor gives for that resolution. Connected, it is exactly `width` × `height` (a ControlNet hint at the latent's pixel size, say): a frame of another aspect is covered with its aspect kept and centre-cropped, as core's ControlNet cuts a hint (`center`). The prediction is resampled once, bilinearly, straight to that size. `resolution` is also the short side the model sees, rounded to a multiple of 14: `518` is the size V2 was trained at; a higher value gives finer edges and costs more memory and time.

Values: relative, normalised per frame (min–max): nearest = white, farthest = black — the convention ControlNet depth models expect, so no invert is needed. V2 predicts relative inverse depth, used as it is. DA3 predicts depth: it is inverted (1 / depth) and clipped to its 2nd–98th percentiles, as the DA3 authors' `visualize_depth` does; the metric model gives no metres here (its scale cancels in the normalisation). The Mono and Metric models' sky (probability ≥ 0.3) is set far, as the authors' forward does.

V2: each frame is preprocessed as the authors' `image2tensor` (resized with its aspect kept so the short side is `resolution` and both sides are multiples of 14, bicubic, ImageNet-normalised). Only Depth-Anything-V2-**Small** (ViT-S) is offered: it is the only Depth Anything V2 size released under Apache-2.0 (Base, Large and Giant are CC-BY-NC). The architecture (DPT head + DINOv2 ViT-S) lives in [`models/depth_anything_v2/arch/`](../models/depth_anything_v2/arch/) as a plain `nn.Module` — no `transformers`, no xFormers. Attention runs through torch's `scaled_dot_product_attention`, with the same formula and scale as upstream, so the `[heads, N, N]` attention matrix is never built (the memory-efficient kernel on CUDA in fp32, the flash kernel on the CPU). At resolution 1288 on a 9:16 frame (15,089 tokens) that matrix is 5.1 GiB in fp32, and the explicit path held two of them, the scores and their softmax: about 10 GiB (computed). The authors' `depth_anything_v2_vits.pth` (from `depth-anything/Depth-Anything-V2-Small`; downloaded from [`beycanai/Depth-Anything`](https://huggingface.co/beycanai/Depth-Anything), an unchanged copy) is fetched on the node's first run — never at import — through the pack's own downloader, loaded with `torch.load(weights_only=True)`, and kept loaded between runs. fp32 on every device.

DA3: the v3 models run ComfyUI core's Depth Anything 3 — its loader, its preprocessing (`lower_bound_resize`) and its forward, in the dtype core loads the model in, one frame at a time — so they need a ComfyUI with core DA3 (commit 5ece24e7, 2026-06-10); without it the node says to update ComfyUI or choose `v2-small`. The weights are Comfy-Org's repackage (`Comfy-Org/Depth-Anything-3`), downloaded from [`beycanai/Depth-Anything`](https://huggingface.co/beycanai/Depth-Anything) (unchanged copies) into `ComfyUI/models/geometry_estimation/`, the folder core's Load Depth Anything 3 node reads, so the two share them. One model is kept loaded at a time.

Licenses: the Depth Anything V2 and DINOv2 code is Apache-2.0 (see [`models/depth_anything_v2/arch/LICENSE`](../models/depth_anything_v2/arch/LICENSE)); the V2 Small checkpoint is published under Apache-2.0 by its authors. DA3-Small, DA3-Base, DA3Mono-Large and DA3Metric-Large are Apache-2.0 (the authors' model cards and the Comfy-Org repackage); DA3-Large, DA3-Giant and DA3Nested are CC-BY-NC-4.0 and not offered. No DA3 code is vendored: it is core's, imported at run time.

## `BC_SocialMediaExport` — Social Media Export

An output node: a full-quality master image in, one platform-ready derivative per ticked platform out, with the **minimum possible cropping** and a controlled, high-fidelity encode — so the platform serves the file as-is instead of re-cropping and re-compressing it. Wire the master into your normal save node *and* into this one; the input passes through untouched on the `images` output and `report` lists one aligned line per (image, platform).

`images` (`IMAGE`), one checkbox per platform (below; `instagram_feed` and `instagram_story` start ticked), widgets `resize_mode` (`crop` / `pad`, default `crop`), `quality` (1–100, default `92`), `allow_upscale` (default off), `filename_prefix` (default `social/export`) → `images` (`IMAGE`, the input as it came in), `report` (`STRING`). A run with no platform ticked stops with an error.

Each platform is an **aspect-ratio band** inside a pixel envelope (`nodes/social_specs.json`, re-read on every execution): a master inside the band is scaled only; outside it, `resize_mode` decides. `crop` trims the minimum on one axis to reach the nearest band edge: a width cut is centred, a height cut takes 40 % from the top and 60 % from the bottom (slightly top-weighted, for faces). `pad` keeps every pixel: the image is centred on a canvas of the nearest band-edge aspect whose background is a cover-scaled copy of the image, blurred by 4 % of the canvas width. Without `allow_upscale` the output is scaled down to fit the envelope, never enlarged. With it the output is scaled to fit the envelope, up or down; where the platform sets `min_w`, a scaled or cropped output is also at least `min_w` wide, past the envelope's height if need be (a padded one is not).

Encoding: JPEG at `quality` with 4:4:4 chroma, progressive and optimized; WebP at `quality`; PNG lossless. Every file carries the sRGB ICC profile and no other metadata (no prompt, no workflow). A platform with a byte cap steps the quality down by 5 until the file fits, to no lower than 70 (a `quality` of 70 or less is not stepped); a file still over the cap is written anyway and its report line says so. An RGBA image is flattened onto white, a one-channel image made RGB. Files land under `output/` as `<filename_prefix>_<platform>_00001_.<ext>`, numbered by ComfyUI's counter; a `/` in the prefix makes subfolders (the default writes `output/social/export_instagram_feed_00001_.jpg`). `report` is a header plus one line per (image, platform): the image's index, the platform, input → crop → output size, the share cropped, the mode (`scale`, `crop` or `pad`), the final quality, the format and the file size.

The fields of a platform in `nodes/social_specs.json` (keys starting with `_`, such as `_comment` and `_schema`, are not platforms):

| Field | Meaning |
| --- | --- |
| `ar_min`, `ar_max` | The band: the smallest and largest width / height kept without a crop. Equal for a fixed shape. |
| `max_w`, `max_h` | The envelope: the largest output, px. |
| `min_w` | Optional. The smallest width of a scaled or cropped output when `allow_upscale` is on. |
| `format` | `jpg` (when absent), `webp` or `png`; any other value writes JPEG. |
| `max_bytes` | Optional. The byte cap. |
| `_source`, `_verified`, `_note` | Not read: where the numbers come from, the date they were checked, which ones are estimates. |

The platforms the file ships with, all JPEG, checked on 2026-07-23:

| Platform | Band (w / h) | Envelope (px) | `min_w` | Byte cap | Status |
| --- | --- | --- | --- | --- | --- |
| `instagram_feed` | 0.8–1.91 | 1440 × 1800 | 1080 | 8 MiB | Verified. `min_w` is a quality floor (the platform's minimum is 320); 1800 is 4:5 at 1440 wide. |
| `instagram_story` | 0.5625 (9:16) | 1080 × 1920 | 1080 | 8 MiB | Verified. The cap is Instagram's image limit. |
| `x_timeline` | 0.75–2.0 | 2048 × 2048 | – | 5 MiB | Band verified; envelope and cap estimated. |
| `tiktok_photo` | 0.4545–2.2 | 1080 × 1920 | – | none | Band verified; envelope estimated. |
| `tiktok_vertical` | 0.5625 (9:16) | 1080 × 1920 | 1080 | none | A design choice: the full-screen feed shape. Envelope estimated, as `tiktok_photo`. |
| `pinterest_pin` | 0.476–0.6667 | 1000 × 1500 | 1000 | 20 MiB | Verified (2:3 at 1000 × 1500 recommended, pins taller than 0.476 truncated, 20 MB). The band, truncation floor to 2:3, is a design choice. |
| `facebook_feed` | 0.8–1.91 | 1440 × 1800 | 1080 | 30 MiB | Verified. `min_w` is a quality floor (the platform's minimum is 600). |
| `facebook_story` | 0.5625 (9:16) | 1080 × 1920 | 1080 | 30 MiB | Verified. |
| `threads_feed` | 0.01–10.0 | 1080 × 1920 | – | 8 MiB | Band and cap verified; envelope estimated. |
| `bluesky` | 0.5–2.0 | 2000 × 2000 | – | 1,000,000 B | Cap verified; band and envelope estimated. |
| `reddit` | 0.5–1.91 | 1200 × 1600 | – | none | Unverified: every number is an estimate. |

The sources and the reasoning behind each number are in the file's `_source` and `_note` fields. Edit the file to change a platform or add one: a changed value applies on the next run; a new platform gets its checkbox once the page is reloaded, and until then ComfyUI refuses a queued prompt with "Required input is missing".

## `BC_SaveImage` — Save Image

`images` (`IMAGE`), optional `positive_text_opt` / `negative_text_opt` (`STRING`, saved into the job data), no outputs. Widgets, in this order:

| Widget | Default | Meaning |
| --- | --- | --- |
| `filename_prefix` | `ComfyUI` | Fixed text the file name starts with; with a `%` in it, a `strftime` format. |
| `filename_keys` | `sampler_name, cfg, steps, %F %H-%M-%S` | The keys of the file name (below). |
| `foldername_prefix` | empty | Fixed text the folder name starts with, as `filename_prefix`. |
| `foldername_keys` | `ckpt_name` | The keys of the folder name. |
| `delimiter` | `-` | The character between the parts; only the first character is used. |
| `save_job_data` | `disabled` | `disabled` / `prompt` / `basic, prompt` / `basic, sampler, prompt` / `basic, models, sampler, prompt` (below). |
| `job_data_per_image` | off | One `<image>.json` per image instead of `jobs.json`. |
| `job_custom_text` | empty | Free text saved with the job data. |
| `save_metadata` | on | The prompt and workflow embedded in the file (below). |
| `counter_digits` | `4` | 0–8 digits of the counter; `0` writes a fixed name and overwrites. |
| `counter_position` | `last` | `last` / `first`: the counter after or before the name. |
| `one_counter_per_folder` | on | Unused, kept for widget order. |
| `image_preview` | on | The saved images listed in the queue / history gallery. |
| `output_ext` | `.webp` | The file format (below). |
| `quality` | `90` | 0–100, used per format (below); `0` is replaced by `90`. |
| `named_keys` | off | `seed=123` instead of `123`. |

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

`output_ext` lists `.webp` (default), `.png`, `.jpg`, `.jpeg`, `.j2k`, `.jp2`, `.gif`, `.tiff`, `.bmp`, plus `.avif` and `.jxl` when `pillow-avif-plugin` / `pillow-jxl-plugin` are installed. What each format does with `quality` (after a `0` is replaced by `90`):

| Format | `quality` |
| --- | --- |
| JPEG | The encoder quality (4:4:4 chroma, optimized). |
| WebP, AVIF, JXL | The encoder quality; `100` = lossless. |
| JPEG 2000 | Not read by Pillow's JPEG 2000 writer: below `100` the lossy (irreversible) wavelet at Pillow's default rate, so every value below `100` gives the same file; `100` lossless. |
| PNG | Compression level `round(quality / 10)`, at most `6`, so level `0` takes a quality of 1–5. PNG is lossless at every level, and above `6` zlib takes many times longer for a few percent smaller files. |
| TIFF, GIF, BMP | Not used. |

With `save_metadata` on, the prompt and workflow go into PNG text chunks or, for WebP, AVIF, JXL, JPEG and TIFF, into the EXIF `Make` and `ImageDescription` tags; JPEG 2000, GIF and BMP get neither. JPEG's EXIF holds at most 64 KB: a larger prompt and workflow stop the save with an error (turn `save_metadata` off or save as PNG / WebP). ComfyUI loads PNG and WebP back into the editor.

`save_job_data` appends an entry per run to `jobs.json` in the folder (or one `<image>.json` per image with `job_data_per_image`): `basic` = prefix + resolution, `models` = checkpoint / LoRAs / VAE / upscale model, `sampler` = seed / steps / cfg / sampler / scheduler / denoise, `prompt` = the two `*_text_opt` inputs or, when neither is wired, the texts found in the prompt: the `positive` / `negative` widgets of an `Efficient Loader` or `Eff. Loader SDXL` node, and the `text` widget behind the `positive` / `negative` links of a `KSampler`, `KSamplerAdvanced` or `UltimateSDUpscale` node (a `text` fed by a link is skipped; with several such nodes, the last in the prompt wins).

`image_preview` only decides whether the saved images are listed in the queue / history gallery. Nothing is ever drawn under the node, so the node keeps the size it was given (`web/js/save_image.js` switches the frontend's output preview off for this node type). Errors while writing raise.

## `BC_SaveImageWithCaption` — Save Image With Caption

`images` (`IMAGE`), optional `caption` (`STRING` input) → `filename` (the last image's file name). Saves each image as a PNG with ComfyUI's own naming, `prefix_00001_.png`, and, with `caption` connected, the caption next to it under the same name, `prefix_00001_.txt`: the layout a training dataset needs. A small node on purpose; Save Image is the one with the name grammar, the formats and the job data.

| Widget | Default | Meaning |
| --- | --- | --- |
| `filename_prefix` | `ComfyUI` | The name before the counter. Takes `%date:yyyy-MM-dd%` and `%Node.widget%` (filled in by the frontend when the prompt is queued, as for ComfyUI's Save Image), `%year%` … `%second%`, `%width%`, `%height%`, `%batch_num%` (the image's index in the batch) and `sub/` folders. |
| `output_folder` | `output` | `output` is ComfyUI's output folder; `output/my_dataset` or `my_dataset` a folder inside it; an absolute path any folder, created when missing. A relative path that leads out of the output folder is refused. |
| `caption_file_extension` | `.txt` | One of `.txt`, `.caption`, `.json`, `.yaml`, `.yml`, `.md`, `.csv`, `.tsv`, `.xml`, `.log`, `.ini`, `.toml` (the dot may be left out); anything else is an error. |

The counter continues from the files in the folder and never overwrites one: a name whose image or caption file exists already is passed over. The prompt and the workflow are embedded in the PNG unless ComfyUI runs with `--disable-metadata`. Nothing is drawn under the node.

## `BC_SkinTexture` — Skin Texture

`image` (`IMAGE`); optional `mask` and `exclude_mask` (`MASK`) → `image` (`IMAGE`), `skin_mask` (`MASK`).

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `sam3_model` | `sam3.1_multiplex_fp16.safetensors` | a file under `models/checkpoints` | The SAM 3 checkpoint. Unused when `mask` is connected. |
| `texture` | `0.35` | 0–1 | The pore field's strength: `0` off, `1` ±6% luminance modulation. |
| `detail` | `0.45` | 0–2 | The gain on the image's own fine detail: `0` off. |
| `pore_scale` | `1.0` | 0.5–4 | The pore size: `1` is about one pixel at a 1024 px long edge, scaled with the image. The detail high-pass scales with it. |
| `feather` | `8` | 0–64 | The mask edge's softening (a Gaussian sigma), px at a 1024 px long edge, scaled with the image; `0` off. |
| `seed` | `0` | 0–2^64 − 1 | The seed of the pore field's noise. |
| `threshold` | `0.5` | 0–1 | SAM 3's detection threshold, for the skin, the exclusions and the face. |

Rendered skin comes out as a smooth gradient, and grain laid on top of it reads as noise over plastic. This node puts a surface under the grain. Inside a skin mask it does two things, both in linear light and both as ratios, so colour is untouched and nothing moves:

- **detail** — boosts the image's own high-frequency luminance (a 2 px high-pass at a 1024 px long edge, scaled with the image) so the fine structure the model did render stops being flat.
- **texture** — multiplies in a synthetic pore field: two band-passed noise octaves at pore scale plus sparse darker pits, unit variance, `1.0` = ±6% luminance modulation. `pore_scale` sets the pore size (1 = about one pixel at 1024 px, scaled with the image).

The mask comes from ComfyUI core's SAM 3 (its checkpoint loader and its SAM 3 detection). The default `sam3.1_multiplex_fp16.safetensors` is downloaded into `models/checkpoints` when missing, from [`beycanai/utility-models`](https://huggingface.co/beycanai/utility-models) (`checkpoints/`, an unchanged copy of `Comfy-Org/sam3.1`'s file); any other checkpoint must already be there. SAM 3 is prompted with `skin` minus `eyes, eyebrows, lips, teeth`; a `face` detection sets the strength — full when the faces span 30% of the frame height or more, fading linearly as they get smaller, full with no face, because pore-scale detail has nowhere to live on a small face; a batch takes its frames' lowest strength. Connect `mask` to skip both detections (body masks from your own SAM 3 prompts, for instance; the strength is then full); `exclude_mask` is subtracted either way; `feather` softens the edge. Highlights get 30% of the effect and black none. The mask that was used, after `exclude_mask` and `feather`, comes out as `skin_mask`.

Order in a still pipeline: Skin Texture → upscale → PostFx grain last. Keep the clean image for I2V; texture and grain are for the published still.

## `BC_FrequencyMerge` — Frequency Merge

`base` (`IMAGE`), `detail` (`IMAGE`, the same size and image count), widgets `split_sigma` (px, default `3`, `0.5`–`64`), `detail_strength` (default `1`, `0`–`2`), optional `device` (`cpu` / `gpu`, default `cpu`) → `image` (`IMAGE`).

The structure, shading and colour of `base` with the fine detail of `detail`:

`image = G(base) + detail_strength × (detail − G(detail))`, clamped to `[0, 1]`,

where `G` is a Gaussian blur with sigma `split_sigma` pixels (reflect-padded at the borders), image by image. Made for two upscales of one image, which line up pixel for pixel — SeedVR2 at `downscale_factor` 1 as `base` (the face stays right, the skin looks plastic) and at a lower factor as `detail` (more skin texture, but a face can change). Two images that do not line up give double edges.

- `split_sigma` sets what counts as fine: a pattern that repeats every 5.3 × `split_sigma` pixels is split half and half, anything finer comes from `detail`, anything coarser from `base`. At `3` the split sits at about 16 px. Keep 5.3 × `split_sigma` well below the size of what must not change (eyes, mouth); raise it to take more of `detail`.
- `detail_strength` `1` takes `detail`'s fine detail as it is; `0` takes none, which leaves the blurred `base`; above `1` it is sharper than `detail`.
- `detail` = `base` gives `base` back (up to float rounding): low + high of one image is the image.
- A Gaussian because it does not ring (an ideal frequency cut-off rings around every edge), it is the same in every direction, and the split is exact.

The merge runs in float32, one image at a time, into one output in the two inputs' promoted dtype: float16 when both are float16 (SeedVR2 PostProcess gives float16), bfloat16 when both are bfloat16, float32 otherwise. A half-precision input is read as the top of this page describes: a float16 image of 8-bit data as its levels (exactly its float32 source), any other as its exact values. Alpha is dropped. Inputs of another size or image count stop the run with an error that says so. A `split_sigma` whose blur (3 × sigma) does not fit inside the image is refused with the largest value that fits. On a CPU, a 3840x2160 image takes 0.3 s at `split_sigma` 3 and 2.5 s at 32, with about 0.5 GB of working memory besides the inputs and the output (Apple M-series CPU).

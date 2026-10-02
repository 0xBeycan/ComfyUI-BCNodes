# SeedVR2

## `BC_SeedVR2FramingDownscale` — SeedVR2 Framing Downscale

`image` (`IMAGE`, the original image or batch), `sam3_model` (a SAM 3 checkpoint under `models/checkpoints`; the default `sam3.1_multiplex_fp16.safetensors` is downloaded on first use), widgets below → `downscale_factor` (`FLOAT`, wire to *SeedVR2 Resize* or *SeedVR2 Preprocess (Compact)* `downscale_factor`), `face_fraction` (`FLOAT`, the measured face size, to see why).

A lower `downscale_factor` gives SeedVR2 more to restore, so more skin texture, but it also shrinks the face SeedVR2 sees, and a face that is small in the frame comes out changed. This node picks the factor from the face size: SAM 3 finds the faces (prompt `face`, up to four per frame, boxes only), and the height of the tallest box divided by the image height is `face_fraction`.

| `face_fraction` | shot | `downscale_factor` |
| --- | --- | --- |
| `close_up_min_face` (`0.3`) or more | close-up | `close_up_factor` (`0.5`) |
| `medium_min_face` (`0.18`) or more | medium | `medium_factor` (`0.75`) |
| below `medium_min_face` | far | `far_factor` (`1`: no downscale, the face is left as it is) |
| no face found | — | `no_face_factor` (`1`), with a warning in the log |

`detection_threshold` (default `0.5`) is SAM 3's. The node logs the fraction, the band and the factor on every run. A batch gets one factor, from its tallest face over every frame, because Resize takes one factor; the frames are detected four at a time.

The factors follow what was seen on SeedVR2 image upscales: a close-up at `0.5` looks best (`1` looks plastic), a medium shot at `0.75`, and a far shot at `0.5` changes the face. The two bounds are uncalibrated estimates. SAM 3's face box measured `0.37` of the image height on the chest-up shot of those observations, `0.24` on the thigh-up one and `0.14` on the knee-up one, and each bound sits between two neighbours (their geometric mean, rounded). On an image 1024 px tall the bounds keep the face at 138 px or more after the downscale (`0.18 × 1024 × 0.75`; `0.3 × 1024 × 0.5` = 154), between the 70 px face that changed and the 184 px one that held. The fraction does not see the resolution: the same fraction is twice the pixels at twice the height, so a high-resolution far shot may stand a lower factor than a low-resolution one.

## `BC_SeedVR2Resize` — SeedVR2 Resize

`image` (`IMAGE`, the original frames), widgets `upscale_factor` (default `2`), `downscale_factor` (default `0.5`, `1` = none), `max_resolution` (cap on the longest edge, `0` = none, default `4096`), `emulate_bf16` (default on).

| Output | Type | Value |
| --- | --- | --- |
| `image` | `IMAGE` | the frames SeedVR2 encodes: downscaled, resized, clamped to `[0, 1]`, zero-padded to a multiple of 16, frame count padded to 4n+1 by repeating the last frame; stored as float16 (VAE Encode casts to float16 anyway) — wire to *SeedVR2 VAE Encode* (video) or *VAE Encode (Tiled)* (single image) |
| `reference` | `IMAGE` | the colour-correction reference: float32 resize of the same frames stored as float16, cropped to even width / height, original frame count — wire to *SeedVR2 PostProcess* `original_resized_images` |

One node for the whole input stage of the SeedVR2 graph: `ImageScaleBy(lanczos, downscale_factor)` (PIL LANCZOS on 8-bit, exactly ComfyUI's), then a shortest-edge resize with `resolution = shortest edge of the original × upscale_factor` — `torchvision` `resize` with the shortest edge at `resolution` (the long edge is floored), `BICUBIC` with antialias, a second resize to `round(edge × max_resolution / longest)` when the longest edge exceeds `max_resolution` — then the clamp, the pad to a multiple of 16 and the 4n+1 frame padding. ComfyUI's own scale nodes cannot match the middle step: `bicubic` there is `F.interpolate` without antialias (a different cubic kernel) and *Resize Image* rounds the long edge. The resize runs in bfloat16 on the GPU for the frame that gets encoded and again in float32 on the CPU for the colour reference; with `emulate_bf16` on and a CUDA device, `image` carries the bfloat16-rounded values, otherwise it is a float32 resize. Both outputs are stored as float16 (VAE Encode casts `image` to float16 anyway; the float16 rounding of `reference` moves the colour transfer by ~0.1/255), which halves the RAM a long clip takes. Frames are downscaled and resized four at a time, so a long video batch works and the downscaled clip is never held whole: the two outputs are the only clip-sized tensors. A float16 `image` (8-bit frames stored as float16) is requantized four frames at a time to the float32 `k / 255` it stands for before anything else, so it gives exactly the frames the same clip gives as float32 (float16 `k / 255` is not float32 `k / 255`, and the lanczos, the bfloat16 cast and the reference resize would round it differently); a float32 `image` takes the same path as before.

## `BC_SeedVR2VAEEncode` — SeedVR2 VAE Encode

`pixels` (`IMAGE`, the frames from *SeedVR2 Resize* `image`), `vae` (`VAE`, the SeedVR2 VAE), widgets `tile_size` (default `1024`), `overlap` (default `256`), `temporal_size`, `temporal_overlap` (the four of *VAE Encode (Tiled)*; the temporal two are ignored, as they are there for this VAE) → `LATENT` `(1, 16, T', H/8, W/8)`, float32.

ComfyUI's *VAE Encode (Tiled)* moves the whole clip to the GPU first (after a full float32 `x × 2 − 1` copy in RAM), then encodes it slice by slice, so VRAM grows with the frame count on top of the encoder's fixed working set (over 28 GB for a whole 1080p frame, measured on a 32 GB card): 897 frames at 1080p do not fit a 32 GB card in one tile. This node runs the same slice loop with the same causal memory cache and the same `x × 2 − 1`, but builds each 4-frame input slice on the GPU only when its turn comes; the posterior mode, the crop to the latent size and the `× scaling_factor` follow the native path. VRAM is the encoder's working set whatever the length, and the float32 copy in RAM is gone. Spatial tiling is the native one: the same tile grid, cosine blend on the latent grid and count normalisation as `tiled_vae`, tile by tile with one causal cache at a time, the blend accumulated in float32 in RAM in the latent the node returns; a `tile_size` that covers the frame means one tile and no blend. At 1080p a whole-frame tile (2048) needs 27.7 GiB (driver level; 24.3 GiB torch peak) and just fits a 32 GB card with nothing else loaded; a 1024 tile needs 14 GiB, about 21 GiB free next to other resident memory (measured on an RTX PRO 4500, 81 frames at 1088x1920). `tests/parity_seedvr2_video.py --vae … --tile …` checks it against `VAE.encode_tiled` (expected: identical latent) and prints both peak VRAMs.

## `BC_SeedVR2VAEDecode` — SeedVR2 VAE Decode

`samples` (`LATENT`), `vae` (`VAE`, the SeedVR2 VAE), widgets `tile_size` (default `1024`), `overlap` (default `256`), `temporal_size`, `temporal_overlap` (the four of *VAE Decode (Tiled)*; the temporal two are ignored, as they are there for this VAE) → `IMAGE` `(B×T, H, W, 3)`, float16, `[0, 1]`.

ComfyUI's *VAE Decode (Tiled)* on the SeedVR2 VAE decodes the whole clip in one call and keeps every decoded frame on the GPU until the end (`slicing_decode` collects the slices and `torch.cat`s them), so VRAM grows with the frame count — a 32 GB card tops out near 190 frames at 1080p. This node runs the same slice loop with the same causal memory cache, the same `/ scaling_factor`, even crop and `(x + 1) / 2` clamp, but moves each decoded slice to RAM as soon as it exists, and keeps the latent in RAM too: each slice of each tile goes to the GPU only when its turn comes. VRAM is then the decoder's working set for one tile (over 31 GB for a whole 1080p frame, which no 32 GB card fits — 2048 fails at the first slice on a 5090; 25-28 GiB with a 1024 tile, about the whole of a 32 GB card; 18 GiB with 768, which runs with 20 GiB free; 10.5 GiB with 512, which runs with 13; measured on an RTX PRO 4500, 81 frames at 1088x1920), whatever the length; the clip length is bounded by RAM instead. The frames are stored as float16, which is what the float16 VAE produced — the native node only upcasts them. Spatial tiling is the native one (same latent-grid tiles, same pixel-grid cosine blend and count normalisation as `tiled_vae`), tile by tile with one causal cache at a time; the blend is accumulated in float16 in the output itself and normalised in place in float32 per 4-frame chunk, so the clip is in RAM once and the result equals the native tiled decode within float16 rounding. `tests/parity_seedvr2_video.py --vae … --tile …` checks it against `VAE.decode_tiled` (expected: equal after rounding to float16) and prints both peak VRAMs.

Both VAE nodes check the free VRAM against the working set first and, when it is short, unload every other model (the DiT is staged in RAM by ComfyUI and reloads on demand); ComfyUI's own loader will not evict one "dynamic" model for another, which is what made the native decode fail next to a resident DiT. With enough VRAM nothing is unloaded.

Measured on a 5090 (32 GB, 92 GB RAM), 720p → 1080p, 7B int8 DiT, both tiles 1024/256: 901 frames (30 s at 30 fps) in one piece, 18 min end to end, linear at ~36 s per second of video; the native tiled decode topped out near 190 frames on the same card. VRAM is flat over the length, RAM is what bounds it (about 61 MB per 1080p frame across the graph's kept outputs, plus a fixed ~13 GB).

Both log their progress next to the progress bar: one line at the start (`81 frames, 6 tile(s) x 20 slice(s)` — the tile count shows at once whether `tile_size` covers the frame) and then every 10 s and at the end (`slice 7/120, tile 1/6, frame 29/81, VRAM 26.0 GiB used, 11 s elapsed, ~174 s left`; the VRAM figure is the driver's, what `nvidia-smi` shows).

## `BC_SeedVR2PostProcess` — SeedVR2 PostProcess

`images` (`IMAGE`), `original_resized_images` (`IMAGE`, the reference), widget `color_correction_method` (`lab` / `wavelet` / `adain` / `none`) → `images` (`IMAGE`, float16).

*Post-Process SeedVR2 Output* builds five full-size float32 copies of the clip on the way through (the raw range conversion, the flattening reshape, the result buffer and the add / div / clamp chain), which is what runs a 30-second 1080p clip out of RAM. This node does the same operations in the same order — frame count and size cropped to the reference, `x × 2 − 1`, the colour transfer from `comfy/ldm/seedvr/color_fix.py` on the VAE device, `(x + 1) / 2` clamp, alpha taken from the reference, even crop — one frame at a time into a single preallocated float16 output. The colour maths itself stays float32 per frame, exactly as in the native node. `tests/parity_seedvr2_video.py` checks every method against the native node: equal after rounding to float16 (max difference 2.4e-4, a sixteenth of an 8-bit step).

## Compact: `BC_SeedVR2PreprocessCompact`, `BC_SeedVR2PostProcessCompact`

Menu `BCNodes/seedvr2/compact`. The same chain as *SeedVR2 Resize* → *SeedVR2 VAE Encode* → sampler → *SeedVR2 VAE Decode* → *SeedVR2 PostProcess*, in two nodes with no clip between them. ComfyUI keeps every node's outputs until the prompt ends, so today's chain holds Resize's `image` and `reference` and Decode's frames to the end next to the result; the compact pair holds the result only.

### `BC_SeedVR2PreprocessCompact` — SeedVR2 Preprocess (Compact)

`image` (`IMAGE`, the original frames), `vae` (`VAE`, the SeedVR2 VAE), the widgets of *SeedVR2 Resize* (`upscale_factor`, `downscale_factor`, `max_resolution`, `emulate_bf16`) and of *SeedVR2 VAE Encode* (`tile_size`, `overlap`; the two temporal widgets, which this VAE ignores, are left out), with the same names and defaults.

| Output | Type | Value |
| --- | --- | --- |
| `latent` | `LATENT` | what *SeedVR2 VAE Encode* gives for *SeedVR2 Resize* `image` — wire to the sampler |
| `plan` | `SEEDVR2_PLAN` | the resize settings and the output size, a few numbers — wire to *SeedVR2 PostProcess (Compact)* `plan` |

Resize's `image` (the padded float16 clip) exists only inside the node, while it is encoded; the colour reference is not made here.

### `BC_SeedVR2PostProcessCompact` — SeedVR2 PostProcess (Compact)

`samples` (`LATENT`, the sampler's), `vae` (`VAE`), `image` (`IMAGE`, the same original frames), `plan` (`SEEDVR2_PLAN`), widgets `color_correction_method` (`lab` / `wavelet` / `adain` / `none`), `tile_size`, `overlap` → `images` (`IMAGE`, float16).

The decode writes only the frames, rows and columns the output keeps into the output buffer (the 4n+1 frames and the pad to 16 are decoded, never stored), and each frame is colour-corrected in place there. Its reference is rebuilt from `image` four frames at a time with Resize's own code, so it is Resize's `reference`; `none` builds none. A float16 `image` is requantized to `k / 255` first, as in Resize. The node stops with an error when `image` is not the batch the plan was made from.

### Same output, less RAM

The pair gives the frames today's chain gives, bit for bit (`tests/layers/pipelines/test_pipe_seedvr2_compact.py`: every colour-correction method, downscale 0.5 / 0.75 / 1, upscale 1.5 / 2, padded frame counts and sizes, one tile and tiled; at 720p → 1080p with ComfyUI's own colour transfers as well).

What the output cache holds at the end, per frame of 720p → 1080p (`upscale_factor` 1.5, original frames float32 as *Load Video* gives them): today 11.1 MB of original + 12.5 (Resize `image`) + 12.4 (`reference`) + 12.5 (Decode) + 12.4 (PostProcess) + 1.0 (the encoded and the sampled latent) = 62 MB; compact 11.1 + 12.4 + 1.0 = 24.5 MB. For 902 frames that is 56 GB against 22 GB. The peak inside Preprocess (Compact) is the original, the padded clip and the latent: 24.4 MB per frame.

With `downscale_factor` below 1 the lanczos downscale runs twice per frame (once in each node; today Resize runs it once for both of its outputs), about 20 ms per 720p frame of CPU work. PostProcess (Compact) builds each four-frame reference chunk on a worker thread while the frames before it are decoded or colour-corrected, so that pass costs no measurable wall time: at 720p → 1080p on the CPU (stand-in VAE, ComfyUI's colour transfers) the compact chain measured within 2% of today's time or faster with `lab`, `wavelet` and `adain`, and 11–19% faster with `none`, which builds no reference.


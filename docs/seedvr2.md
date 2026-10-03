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

ComfyUI's *VAE Encode (Tiled)* moves the whole clip to the GPU first (after a full float32 `x × 2 − 1` copy in RAM), then encodes it slice by slice, so VRAM grows with the frame count on top of the encoder's fixed working set (over 28 GB for a whole 1080p frame, measured on a 32 GB card): 897 frames at 1080p do not fit a 32 GB card in one tile. This node runs the same slice loop with the same causal memory cache and the same `x × 2 − 1`, but builds each 4-frame input slice on the GPU only when its turn comes; the posterior mode, the crop to the latent size and the `× scaling_factor` follow the native path. VRAM is the encoder's working set whatever the length, and the float32 copy in RAM is gone. Spatial tiling is the native one: the same tile grid, cosine blend on the latent grid and count normalisation as `tiled_vae`, tile by tile with one causal cache at a time, the blend accumulated in float32 in RAM in the latent the node returns; a `tile_size` that covers the frame means one tile and no blend. At 1080p a whole-frame tile (2048) needs 27.7 GiB (driver level; 24.3 GiB torch peak) on 81 frames and fits a 32 GB card with nothing else loaded (a 5090 peaked at 27.89 of its 31.36 GiB on the driver over a 901-frame clip, the same at every length, with the Process Monitor off); a 1024 tile needs 14 GiB, about 21 GiB free next to other resident memory (measured on an RTX PRO 4500, 81 frames at 1088x1920). `tests/parity_seedvr2_video.py --vae … --tile …` checks it against `VAE.encode_tiled` (expected: identical latent) and prints both peak VRAMs.

## `BC_SeedVR2ChunkSize` — SeedVR2 Chunk Size

`latent` (`LATENT`, the encoded clip: *SeedVR2 Preprocess (Compact)* or *SeedVR2 VAE Encode* `latent`), widget `safety_margin` (default `0.64`) → `frames_per_chunk` (`INT`).

The longest chunk ComfyUI's SeedVR2 sampler can take on this card. Set *Split SeedVR2 Latent*'s `chunking_mode` to `manual` and wire `frames_per_chunk` into its `frames_per_chunk` socket (it appears with `manual`). Split's own `auto` budgets the memory free when it runs with a law fitted on the 3B, and ran out of memory on both cards measured (at 1080p, 281 frames on an RTX PRO 6000).

The pick is the largest 4n+1 chunk, at most the clip's frames, with

    8.36 GiB + 0.6075 GiB × megapixels × latent frames × (1 + safety_margin)  ≤  the card's total memory

where the megapixels are the DiT's frame (8 × 8 per latent cell, so 2.09 for 1088x1920) and a chunk of `n` latent frames is `4(n − 1) + 1` pixel frames. The line is the 5090's KSampler allocated peaks at 21 to 57 frames per chunk (7B int8, 1080p, dynamic VRAM, cudaMallocAsync). The card is ComfyUI's total for its device (on CUDA the driver's total, what an out-of-memory message calls the device limit), not the memory free when the node runs, so the pick is the same on every run of a card. With the Process Monitor off (on, it inflated every figure of those runs) the line holds: its part per frame is the allocator's reserve to within 3% (13.5 GiB at most at 41 frames over the 23 chunks of a 901-frame clip, 18.44 at 57, 19.72 at 61), its fixed part the DiT's resident weights and the CUDA context. `safety_margin` 0.64 picks the longest chunks measured on a long clip, 41 at 1080p on a 32 GB card and 161 on a 96 GB one. On the 5090, 0.2 would pick 57 and 0.1 61, both measured to run on an 81-frame clip; the same margins give 221 and 245 on the 96 GB card, not measured yet. The sampler is about a tenth of the upscale's time (95 s of 1017 for 901 frames at 41) and its cost grows with the frames, not the chunks, so a longer chunk saves little. Raise the margin if a chunk runs out of memory; lower it to try longer chunks.

| card (device total) | 720p → 1080p (DiT 1088x1920) | 1080p → 4K (DiT 2160x3840) |
| --- | --- | --- |
| 24 GB (23.65 GiB) | 25 frames | 1 frame |
| 32 GB (31.36 GiB, 5090) | 41 | 5 |
| 48 GB (47.50 GiB) | 69 | 13 |
| 96 GB (94.97 GiB, RTX PRO 6000) | 161 | 37 |

41 and 161 are the longest chunks measured to run on those cards' long clips. The law is the 7B's; the 3B is narrower, so it gets the same, shorter-than-needed chunks. The node logs the card, the megapixels, the margin and the pick. A card that holds no latent frame gets 1 frame and a warning.

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

`image` (`IMAGE`, the original frames), `vae` (`VAE`, the SeedVR2 VAE), the widgets of *SeedVR2 Resize* (`upscale_factor`, `downscale_factor`, `max_resolution`, `emulate_bf16`) and of *SeedVR2 VAE Encode* (`tile_size`, `overlap`; the two temporal widgets, which this VAE ignores, are left out), with the same names and defaults, except that `tile_size` takes `0` (auto, the default; see *Auto tile* below).

| Output | Type | Value |
| --- | --- | --- |
| `latent` | `LATENT` | what *SeedVR2 VAE Encode* gives for *SeedVR2 Resize* `image` — wire to the sampler |
| `plan` | `SEEDVR2_PLAN` | the resize settings and the output size, a few numbers — wire to *SeedVR2 PostProcess (Compact)* `plan` |

Resize's `image` (the padded float16 clip) exists only inside the node, while it is encoded; the colour reference is not made here.

### `BC_SeedVR2PostProcessCompact` — SeedVR2 PostProcess (Compact)

`samples` (`LATENT`, the sampler's), `vae` (`VAE`), `image` (`IMAGE`, the same original frames), `plan` (`SEEDVR2_PLAN`), widgets `color_correction_method` (`lab` / `wavelet` / `adain` / `none`), `tile_size` (`0`, auto, by default), `overlap` → `images` (`IMAGE`, float16).

The decode writes only the frames, rows and columns the output keeps into the output buffer (the 4n+1 frames and the pad to 16 are decoded, never stored), and each frame is colour-corrected in place there. Its reference is rebuilt from `image` four frames at a time with Resize's own code, so it is Resize's `reference`; `none` builds none. A float16 `image` is requantized to `k / 255` first, as in Resize. The node stops with an error when `image` is not the batch the plan was made from.

### Auto tile

`tile_size` `0` (the default of a new compact node) picks the tiling that computes the fewest pixels and fits the card. The tile's two sides are chosen separately (multiples of 32), so a 1088-high frame can run in full-height strips instead of two rows of squares; the work counted is every tile in full, the overlaps computed twice included. Among tilings with the same work it takes the fewest tiles, then the smallest working set. The working set is the VAE's own estimate for the largest tile, the one both nodes compare with the free VRAM before they run (a fixed part plus bytes per tile pixel; the encoder 6.36 GB + 15,440 bytes, the decoder 7.95 GB + 22,940), and the card is its total less 768 MiB the driver never hands out (a 5090 shows 30.7 of its 31.36 GiB free with every model unloaded). The `overlap` stays the one set: a side that does not cover the frame is at least twice it (*VAE Decode (Tiled)* would cut the overlap of a tile under four overlaps to a quarter; the auto tile never does). The tiles are the ones `tiled_vae` makes for that (rows, columns) tile and blend the same way; when no tiling keeping the overlap fits, the largest square that does, as a typed `tile_size` runs. Any other `tile_size` is a square tile as before, so a saved workflow runs as it did.

| card | frame | encode: tile, tiles, pixels computed | decode: tile, tiles, pixels computed | the square tile's decode |
| --- | --- | --- | --- | --- |
| 24 GB (23.65 GiB) | 1088x1920 | 832 x 1088, 3, 2.65 M | 608 x 1088, 5, 3.20 M | 832, 6, 3.03 M (overlap cut to 208) |
| 24 GB | 2160x3840 | 1152 x 896, 12, 12.3 M | 768 x 896, 21, 14.4 M | 832, 24, 13.6 M (overlap 208) |
| 32 GB (31.36 GiB) | 1088x1920 | 1088 x 1088, 2, 2.37 M | 832 x 1088, 3, 2.65 M | 1024, 6, 3.27 M |
| 32 GB | 2160x3840 | 1152 x 1216, 8, 11.1 M | 1152 x 896, 12, 12.3 M | 1024, 15, 13.0 M |
| 48 GB (47.50 GiB) | 1088x1920 | one tile, 2.09 M | 1088 x 1088, 2, 2.37 M | 1664, 2, 2.37 M |
| 48 GB | 2160x3840 | 2048 x 1216, 4, 9.90 M | 1472 x 1216, 6, 10.5 M | 1344, 8, 11.1 M |
| 96 GB (94.97 GiB) | 1088x1920 | one tile, 2.09 M | one tile, 2.09 M | one tile |
| 96 GB | 2160x3840 | 2048 x 2176, 2, 8.85 M | 1472 x 2176, 3, 9.40 M | 1984, 6, 10.5 M |

Tiles are width x height. On a 5090 at 720p → 1080p (Process Monitor off, 901 frames) the decode was 63% of the upscale's time, 637.6 s with the 1024 square's six tiles; the three strips compute a fifth less and their working set is smaller (26.7 against 29.8 GiB estimated; the 1024 square peaked at 25.72 GiB on the driver). The encode's two strips compute what the 1568 tile's two did (223.6 s); the whole frame, which ran at 27.89 GiB on the driver and took 213.0 s, is not picked on 32 GB: its estimate (36.0 GiB) is over the card. The encoder's estimate is 1.34 to 1.40 times its driver peaks with the monitor off; it stays until the bigger cards are measured the same way.

### Same output, less RAM

The pair gives the frames today's chain gives, bit for bit (`tests/layers/pipelines/test_pipe_seedvr2_compact.py`: every colour-correction method, downscale 0.5 / 0.75 / 1, upscale 1.5 / 2, padded frame counts and sizes, one tile and tiled; at 720p → 1080p with ComfyUI's own colour transfers as well).

What the output cache holds at the end, per frame of 720p → 1080p (`upscale_factor` 1.5, original frames float32 as *Load Video* gives them): today 11.1 MB of original + 12.5 (Resize `image`) + 12.4 (`reference`) + 12.5 (Decode) + 12.4 (PostProcess) + 1.0 (the encoded and the sampled latent) = 62 MB; compact 11.1 + 12.4 + 1.0 = 24.5 MB. For 902 frames that is 56 GB against 22 GB. The peak inside Preprocess (Compact) is the original, the padded clip and the latent: 24.4 MB per frame.

With `downscale_factor` below 1 the lanczos downscale runs twice per frame (once in each node; today Resize runs it once for both of its outputs), about 20 ms per 720p frame of CPU work. PostProcess (Compact) builds each four-frame reference chunk on a worker thread while the frames before it are decoded or colour-corrected, so that pass costs no measurable wall time: at 720p → 1080p on the CPU (stand-in VAE, ComfyUI's colour transfers) the compact chain measured within 2% of today's time or faster with `lab`, `wavelet` and `adain`, and 11–19% faster with `none`, which builds no reference.


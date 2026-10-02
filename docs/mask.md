# Mask

Half-precision input (float16 or bfloat16; BCVideoNodes' Load Video gives float16 by default): Mask Fill Holes, MaskGrow, Blockify Mask, Repeat Mask Batch, Draw Mask On Image and BiRefNet Remove Background read it a frame at a time as the float32 8-bit levels a float32 load holds, work in float32 and give their outputs in the input's dtype (Draw Mask On Image and BiRefNet: the image's); any other input gives float32 outputs, as before.

## `BC_IsMaskEmpty` — Is Mask Empty

`MASK` → `BOOLEAN`. `True` when the mask is missing or every pixel is `0`.

## `BC_MaskFillHoles` — Mask Fill Holes

`masks` (`MASK`, optional) → `MASKS`. Fills every fully enclosed hole of each mask in the batch: the mask is quantised to 8 bit, and the background regions that cannot reach the border in 4-connected steps (OpenCV connected components) become foreground — the same result as the earlier scipy `binary_fill_holes` implementation, bit for bit. A hole closed only by diagonal steps is filled; a gap touching the border is not enclosed and stays open. Output is hard `0/1`, `float32`, shape `(B, H, W)`. A missing mask (`None`, an empty tensor, or nothing wired) returns `torch.zeros((1, 64, 64))`.

## `BC_MaskGrow` — MaskGrow

`mask` (`MASK`, optional), widgets `invert_mask` (default `False`), `grow` (`-999..999`, default `4`), `blur` (`0..999`, default `4`) → `mask`.

Optional inversion, then `|grow|` iterations of grey dilation (positive) or erosion (negative) with a cross-shaped 3×3 kernel on the 8-bit mask (OpenCV), then a Gaussian blur of radius `blur` (PIL). The result is the same as the earlier scipy implementation, bit for bit, with the morphology about ten times faster (measured at `grow` 10); frames are written into one preallocated output. A missing mask returns `torch.zeros((1, 64, 64))`.

## `BC_DrawMaskOnImage` — Draw Mask On Image

`image` (`IMAGE`, RGB or RGBA), `mask` (`MASK`), `color` (`STRING`, default `0, 0, 0`), `device` (`cpu` / `gpu`, optional) → `images`.

Each frame is blended towards `color` by `m = mask × alpha`: `rgb × (1 − m) + color × m`; an RGBA frame keeps the larger of its own alpha and `m`. `color` is 1 (grey), 3 (RGB) or 4 (RGBA) comma-separated values — each value above 1 is read as 0–255, any other as 0–1 — or `#rgb` / `#rgba` / `#rrggbb` / `#rrggbbaa`; the alpha is the opacity, `1` when absent. A mask of another size is scaled to the image (nearest), fewer masks than frames repeat in order, extra masks are ignored.

Frames are blended one at a time on `device` into one preallocated output; the inputs are never copied. A missing or empty mask returns the image unchanged, an empty image batch passes through. A colour in another form, and an image that is neither RGB nor RGBA, raise with what to fix.

## `BC_BlockifyMask` — Blockify Mask

`masks` (`MASK`), `block_size` (`8..512`, default `32`), `device` (`cpu` / `gpu`, optional) → `mask`.

Per mask: the bounding box of its pixels above `0` is cut into `side // block_size` blocks per axis (at least one) of equal size, the last row / column of blocks taking the remainder; a block is `1` when any of its pixels is above `0`, everything else is `0`. Integer counts per frame into one preallocated output, returned on the CPU. A missing or empty mask returns `torch.zeros((1, 64, 64))`.

## `BC_RepeatMaskBatch` — Repeat Mask Batch

`mask` (`MASK`), `amount` (`1..4096`, default `1`) → `mask`. The whole batch repeated `amount` times in order (`m0 m1 m0 m1 …`), as ComfyUI's Repeat Image Batch does for images: one allocation, and `amount = 1` passes the input on without a copy. A missing or empty mask returns `torch.zeros((1, 64, 64))`.

## `BC_BiRefNetRemoveBackground` — BiRefNet Remove Background

`image` (`IMAGE`), `model` (combo) → `IMAGE` (RGBA with alpha = matte, or RGB over a solid colour), `MASK` (the matte, `(B, H, W)`), `MASK_IMAGE` (the matte as an RGB image).

Options, applied to the matte in this order: `sensitivity` (below 1 the matte is amplified, so faint areas count as foreground), `mask_blur` (Gaussian blur of the edges, pixels), `mask_offset` (grow / shrink by one pixel per step), `invert_output` (keep the background instead), `refine_foreground` (hardens the matte edge and scales the colours by it, for cleaner cut-outs on transparent output), `background` (`Alpha` → RGBA output; `Color` → RGB output over `background_color`), `background_color` (colour picker, `#rrggbb`; the picker needs frontend 1.28 or newer — older frontends show the input as a socket and the default `#222222` is used).

The BiRefNet architecture lives in [`models/birefnet/arch/`](../models/birefnet/arch/) as a plain `nn.Module` and the weights are loaded straight from safetensors — no `transformers`, no `trust_remote_code`, no `timm`.

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

Licenses: the BiRefNet and Swin Transformer code is MIT (see [`models/birefnet/arch/LICENSE`](../models/birefnet/arch/LICENSE)); the checkpoints are published under MIT by their authors.

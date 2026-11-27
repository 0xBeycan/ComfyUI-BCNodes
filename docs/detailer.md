# Detailer

Both nodes run one core: a region of the image is cut out with its surroundings, enlarged until the region's longer side is the size you set, sampled once, shrunk back and pasted through the region's own shape. The core knows nothing of what a region shows and nothing of the model: any `MODEL`, `VAE` and `CONDITIONING` go through ComfyUI's own sampler, VAE and ControlNet code.

## `BC_Detailer` — Detailer

`image` (`IMAGE`), `mask` (`MASK`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), then the sampling widgets, `region_size`, `region_crop_size` and `max_region_size` (below) → `image` (`IMAGE`), `mask` (`MASK`).

Each mask of the `mask` batch is one region (SAM 3 Detect with `individual_masks` on gives one mask per object), detailed in batch order on every frame of `image`, each on the frame the previous one left. The `mask` output holds the pixels that changed.

### Sampling widgets (both nodes)

| Widget | Default | What it does |
| --- | --- | --- |
| `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` | KSampler's | as on KSampler |
| `denoise` | `0.5` | how far the region may be redrawn; Face Detailer's face pass (its eye pass takes `eye_denoise`) |

Each region is sampled once. At `denoise` `0` the Detailer details no region: nothing is encoded, sampled, decoded or pasted, every pixel keeps its exact value and the `mask` output is empty. On the Face Detailer `denoise` `0` turns the face pass off and `eye_denoise` `0` the eye pass, as their toggles do.

### Size, crop size and the largest region (both nodes)

The Detailer has `region_size`, `region_crop_size` and `max_region_size`; the Face Detailer `face_size`, `face_crop_size` and `max_face_height` for the face pass, and `eye_size` and `eye_crop_size` for the eye pass.

| Widget | Default | What it does |
| --- | --- | --- |
| `region_size` / `face_size` / `eye_size` | `640` / `640` / `320` | the px the region's longer side is redrawn at inside its crop (`64`–`8192`, step `64`); its height for an upright face. A region whose longer side is already this size or more is left as it is: there is nothing to enlarge |
| `region_crop_size` / `face_crop_size` / `eye_crop_size` | `1280` / `1280` / `640` | **the model samples at this size**: the crop's longer side, in px (`256`–`8192`, step `64`). The crop holds the region at its size and the surroundings around it; its shorter side follows the region's aspect, rounded up to a multiple of 64. Under the size it is the size: the region alone, no surroundings. Only for context: pixels outside the mask never change |
| `max_region_size` / `max_face_height` | `384` | the largest region detailed, in px (`64`–`16384`, step `1`): the Detailer's on the region's longer side, the Face Detailer's on the face's height. A larger one is left as it is |

The rule: scale = size / the region's longer side; the crop is crop size / scale image px on its longer side, around the region's centre, and the model samples it at exactly the crop size. With `face_size` `512` and `face_crop_size` `1280`, a 100 px face is redrawn at 512 px inside a 1280 px sampling of a 250 px crop. Raising the crop size shows the model more of the scene at the same scale for the region; raising the size redraws the region larger in the same sampling size, with less around it.

The size and the largest region are apart on purpose: lowering the largest region skips mid-size regions without changing how much a small region is enlarged.

### How a region is detailed

1. **The crop.** The region's box at its own aspect, grown around its centre until, at the scale that puts the region's longer side at the size, the crop's longer side is the crop size; cut to the image. Each sampling side is rounded up to a multiple of 64 (so no model pads its latent), the crop is widened to that aspect at the same scale (the rounding only adds surroundings) and moved inside the image, its scale kept. Where the image is smaller than the crop, the crop is cut to it and sampled smaller than the crop size; only where the rounding would need more than the image does the scale rise a little to fit. **A region whose longer side is already the size or more is left as it is.**
2. **Sampling.** The model gets ComfyUI's Differential Diffusion (on a clone; the input model is not changed). The crop is encoded as ComfyUI's Inpaint Model Conditioning does it, with the region's mask, softened at its edge (a Gaussian, sigma 2% of the crop's shorter side, about 20 px at 1024), as the noise mask: a model with inpaint channels also gets the masked image, any other model ignores it. One sampling at `denoise`.
3. **The paste.** The decoded crop is shrunk back (antialiased bicubic), its colour matched to the image's in the mask's edge band (each channel's mean and standard deviation within about two sigmas inside the edge), and blended in with a Laplacian pyramid (Burt and Adelson) under the mask softened inward, so fine detail switches over a short distance and colour over a long one. **Only the pixels inside the mask change**; every other pixel keeps its exact value.

### ControlNets

Each crop is sampled with every ControlNet's control image cut to the same part of the image, so the hint lines up with the crop; ComfyUI then scales the cut to the crop's sampling size, and encodes it where the ControlNet does, as it always does. Every other setting of the ControlNet (strength, start and end percent, VAE) stays as it was applied. This holds for every model the nodes sample with: Face Detailer's `model` and `model_no_lora`, and its face and eye passes. The conditioning and the model you connect are not changed.

- **In `positive` / `negative`**: every ControlNet ComfyUI's Apply ControlNet and the nodes built on it put there: ControlNet, Control LoRA, T2I-Adapter, the SD3.5 ControlNets and Qwen-Image's Fun ControlNet Union (for example Qwen-Image-2512's), with the mask of Apply ControlNet Inpainting (AliMama). Each link of the chain is copied and its hint, and its mask, cut. A hint of another size than the image is mapped the way ComfyUI maps it onto the whole image (its centre cut to the image's aspect).
- **On the `MODEL`**: the ControlNets ComfyUI applies as a model patch: Apply Fun ControlNet (Z-Image Turbo's and Qwen-Image 2.1's Fun ControlNet Union, with the inpaint image and mask when they are given), Apply Qwen Image DiffSynth ControlNet, Apply Anima LLLite, Apply Wan Uni3C ControlNet and Apply MiniMax H3 Fun ControlNet. For each crop the detailer samples a copy of the model whose patch holds its images and mask cut the same way, each mapped the way that patch lays it onto the image (through the centre cut, or scaled as it is); the patch then computes its control from the cut at the crop's sampling size.

A ControlNet the detailer cannot cut stops the node with an error that names it, before the crop is encoded or sampled: one in the conditioning that is not one of ComfyUI's ControlNet objects, and Apply SUPIR Patch on the model (it holds the latent of the whole image, encoded when it is applied).

Not cut: area or mask conditioning, reference latents (Qwen-Image Edit), and the model patches of other node packs, which are sampled as they are. A control latent the conditioning holds where the inpaint latent goes (`concat_latent_image`: MOSS-Image Color/Depth Control, WanFunControlToVideo, Wan22FunControlToVideo) is replaced by the crop's inpaint conditioning.

## `BC_FaceDetailer` — Face Detailer

`image` (`IMAGE`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), the sampling widgets, `face_size`, `face_crop_size`, `max_face_height`, `eye_size`, `eye_crop_size` (above), `eye_denoise`, `sam3_model`, `threshold`, `face`, `eyes`; optional `model_no_lora` (`MODEL`) and `mask` (`MASK`) → `image` (`IMAGE`), `mask` (`MASK`, the pixels that changed).

SAM 3 finds every face of each frame (`sam3_model` is a checkpoint under `models/checkpoints`; the default `sam3.1_multiplex_fp16.safetensors` is downloaded on first use; `threshold` is its detection threshold). The prompts are fixed: `face` (every face), `eyes`.

**The character and the models.** `model` is the pipeline's main model, with the character LoRA. The character's face is detailed with it: the face `mask` overlaps most when `mask` is connected (no face is the character when it overlaps none), the face of the largest area otherwise. The character is chosen among every face found, before the size gate. With `model_no_lora` connected, every other face is detailed with it; without it, the other faces are left as they are. The character goes first, so the model changes once per frame.

**The passes.** Each face gets two passes of the core, each a region of its own:

| Toggle | Default | Region | Settings |
| --- | --- | --- | --- |
| `face` | on | the face's mask | `face_size`, `face_crop_size`, `max_face_height`, `denoise` |
| `eyes` | on | both eyes as one region, on one crop | `eye_size`, `eye_crop_size`, `eye_denoise` |

A pass at `0` (`denoise` for the face, `eye_denoise` for the eyes) is off, as its toggle off. With both passes off the image passes through as it is and the `mask` is empty: SAM 3 is not loaded and no face is searched for.

The eyes are found on the face's box of the frame the face pass left, and cut to the face's mask (another face's eye inside the box is not taken). Both are detailed as one region, the union of their masks, on one crop in one sampling: the model sees both eyes, so their gazes match (an eye on a crop of its own never sees the other one). When only one eye is found it is detailed alone. The eye pass uses the face's model and the sampling widgets but `denoise`: `eye_denoise` (default `0.35`, `0`–`1`) sets how far the eyes may be redrawn; at `0` the eyes are not searched for and keep their pixels. The box of the eyes has its longer side redrawn at `eye_size` px inside a crop sampled at `eye_crop_size` px, by the same rule as the face.

**The size gate.** A face's height is the height of its SAM 3 mask's box, its longer side the longer side of that box (the height, for an upright face), in px.

- Height under 64 px (fixed): no pass at all, not even the eyes. At that size an eye is about 13 px and the model would invent the face rather than detail it.
- Height over `max_face_height`: the face pass is skipped; the eyes still run, judged on their own.
- Longer side at `face_size` or more (a face up to `max_face_height`, when `face_size` is the smaller): the face pass is skipped, there is nothing to enlarge; the eyes still run.
- The eye pass is skipped when the box of the eyes has its longer side at `eye_size` or more. At the defaults a face up to 384 px tall is redrawn, and its eyes while the box of both is under 320 px: a face too large for the face pass keeps its own pixels while its eyes are still detailed.

The image and the mask are computed together (the eyes are found on the detailed face): an unconnected one is dropped when the node returns.

Frames one at a time; a half-precision image (float16 / bfloat16) is read a frame at a time as float32 and both outputs come out in its dtype.

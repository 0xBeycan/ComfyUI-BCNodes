# Detailer

Both nodes run one core: a region of the image is cut out with its surroundings, enlarged to the size the model samples best at, sampled again, shrunk back and pasted through the region's own shape. The core knows nothing of what a region shows and nothing of the model: any `MODEL`, `VAE` and `CONDITIONING` go through ComfyUI's own sampler, VAE and ControlNet code.

## `BC_Detailer` — Detailer

`image` (`IMAGE`), `mask` (`MASK`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), then the sampling widgets (below) → `image` (`IMAGE`), `mask` (`MASK`).

Each mask of the `mask` batch is one region (SAM 3 Detect with `individual_masks` on gives one mask per object), detailed in batch order on every frame of `image`, each on the frame the previous one left. The `mask` output holds the pixels that changed.

### Sampling widgets (both nodes)

| Widget | Default | What it does |
| --- | --- | --- |
| `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` | KSampler's | as on KSampler; `steps` for each pass |
| `denoise` | `0.5` | the structure pass: how far the region may be redrawn |
| `refine_denoise` | `0.2` | the refine pass, run on the structure pass's result: fine detail only |
| `resolution` | `1024` | the long side each crop is sampled at, in px (`256`–`8192`, step `64`): the model's native size (SD 1.5 512, SDXL 1024, Qwen-Image 1328) |
| `context` | `3.0` | the crop around a region as a multiple of the region's box (`1`–`10`): more context, but fewer pixels on the region itself |

A pass at `0` does not run. The refine pass uses `seed + 1`.

### How a region is detailed

1. **The crop.** The region's box times `context`, around its centre, cut to the image, widened to the aspect of the sampling size (long side `resolution`, both sides multiples of 64, so no model pads its latent) and moved inside the image. **A crop that would not be enlarged is left as it is**: sampling a region at its own size or smaller adds no detail.
2. **Sampling.** The model gets ComfyUI's Differential Diffusion (on a clone; the input model is not changed). The crop is encoded as ComfyUI's Inpaint Model Conditioning does it, with the region's mask, softened at its edge (a Gaussian, sigma 2% of the crop's shorter side, about 20 px at 1024), as the noise mask: a model with inpaint channels also gets the masked image, any other model ignores it. The structure pass samples at `denoise`, then the refine pass samples its latent again at `refine_denoise`; there is no decode between them.
3. **The paste.** The decoded crop is shrunk back (antialiased bicubic), its colour matched to the image's in the mask's edge band (each channel's mean and standard deviation within about two sigmas inside the edge), and blended in with a Laplacian pyramid (Burt and Adelson) under the mask softened inward, so fine detail switches over a short distance and colour over a long one. **Only the pixels inside the mask change**; every other pixel keeps its exact value.

### ControlNets

A ControlNet applied with ComfyUI's Apply ControlNet (for example Qwen-Image-2512 Fun ControlNet Union) sits in `positive` / `negative`. For each crop, every ControlNet of the chain is copied and its hint image (and any image-sized extra input) cut to the same part of the image, so the hint lines up with the crop; ComfyUI then scales the cut hint to the crop's latent as it always does. A hint of another size than the image is mapped the way ComfyUI maps it onto the whole image (its centre cut to the image's aspect). The conditioning you connect is not changed.

Not cut: a ControlNet applied as a model patch (ComfyUI's Qwen-Image DiffSynth ControlNet node, for one), area or mask conditioning, and reference latents (Qwen-Image Edit).

## `BC_FaceDetailer` — Face Detailer

`image` (`IMAGE`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), the sampling widgets (above), `sam3_model`, `threshold`, `min_face_height`, `max_face_height`, `face`, `mouth`, `eyes`; optional `model_no_lora` (`MODEL`) and `mask` (`MASK`) → `image` (`IMAGE`), `mask` (`MASK`, the pixels that changed).

SAM 3 finds every face of each frame (`sam3_model` is a checkpoint under `models/checkpoints`; the default `sam3.1_multiplex_fp16.safetensors` is downloaded on first use; `threshold` is its detection threshold). The prompts are fixed: `face` (every face), `teeth, lips` (the mouth), `eyes`.

**The character and the models.** `model` is the pipeline's main model, with the character LoRA. The character's face is detailed with it: the face `mask` overlaps most when `mask` is connected (no face is the character when it overlaps none), the face of the largest area otherwise. The character is chosen among every face found, before the size gate. With `model_no_lora` connected, every other face is detailed with it; without it, the other faces are left as they are. The character goes first, so the model changes once per frame.

**The passes.** Each face gets up to three passes of the core, each a region of its own:

| Toggle | Default | Region |
| --- | --- | --- |
| `face` | on | the face's mask |
| `mouth` | off | `teeth` and `lips` as one region |
| `eyes` | on | each eye on a crop of its own |

The mouth and the eyes are found on the face's box of the frame the face pass left, and cut to the face's mask (another face's eye inside the box is not taken). They use the face's model and the same widgets; their crops are smaller because their regions are.

**The size gate.** A face's height is the height of its SAM 3 mask, in px.

- Below `min_face_height` (default `64`): no pass at all. At that size an eye is about 13 px and the model would invent the face rather than detail it.
- Above `max_face_height` (default `16384`, no limit): the face pass is skipped; the mouth and eyes still run.
- Every pass also skips itself when its crop would not be enlarged (step 1 above). At `resolution` 1024 and `context` 3 the face pass gains up to a face about 340 px tall, the mouth up to about 850 px of face width, each eye (a fifth of the face's width) up to about 1700 px: a large face keeps its own pixels while its eyes are still detailed.

The image and the mask are computed together (the mouth and the eyes are found on the detailed face): an unconnected one is dropped when the node returns.

Frames one at a time; a half-precision image (float16 / bfloat16) is read a frame at a time as float32 and both outputs come out in its dtype.

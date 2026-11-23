# Detailer

Both nodes run one core: a region of the image is cut out with its surroundings, enlarged until the region's longer side is the size you set, sampled again, shrunk back and pasted through the region's own shape. The core knows nothing of what a region shows and nothing of the model: any `MODEL`, `VAE` and `CONDITIONING` go through ComfyUI's own sampler, VAE and ControlNet code.

## `BC_Detailer` — Detailer

`image` (`IMAGE`), `mask` (`MASK`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), then the sampling widgets, `region_size` and `padding` (below) → `image` (`IMAGE`), `mask` (`MASK`).

Each mask of the `mask` batch is one region (SAM 3 Detect with `individual_masks` on gives one mask per object), detailed in batch order on every frame of `image`, each on the frame the previous one left. The `mask` output holds the pixels that changed.

### Sampling widgets (both nodes)

| Widget | Default | What it does |
| --- | --- | --- |
| `seed`, `steps`, `cfg`, `sampler_name`, `scheduler` | KSampler's | as on KSampler; `steps` for each pass |
| `denoise` | `0.5` | the structure pass: how far the region may be redrawn |
| `refine_denoise` | `0.2` | the refine pass, run on the structure pass's result: fine detail only |

A pass at `0` does not run. The refine pass uses `seed + 1`.

### Size and padding (both nodes)

Both are stated in the region's own pixels. The Detailer calls the size `region_size`, the Face Detailer `face_size`; the rule is the same.

| Widget | Default | What it does |
| --- | --- | --- |
| `region_size` / `face_size` | `640` | the px the region's longer side is redrawn at (`64`–`8192`, step `64`); its height for an upright face. A region whose longer side is already this size or more is left as it is: sampling it adds no detail |
| `padding` | `0.5` | the surroundings the model sees around the region's box, on every side, as a fraction of the box (`0`–`4.5`): `0.5` adds half the box's width left and right and half its height above and below, a crop twice the box. Only for context: pixels outside the mask never change |

A crop is sampled at about (1 + 2 × `padding`) × size px on its longer side: at the defaults, a 200 px face is cropped at 400 px around it and sampled at 1280 px. Raising `padding` shows the model more of the scene without shrinking the region; raising the size redraws the region larger.

### How a region is detailed

1. **The crop.** The region's box grown by `padding` times its width on the left and on the right and its height above and below, around its centre, cut to the image. The crop is enlarged so the region's longer side is the size; each sampling side is rounded up to a multiple of 64 (so no model pads its latent), the crop is widened to that aspect at the same scale (the rounding only adds surroundings) and moved inside the image. Only where that crop would be larger than the image does the scale rise a little to fit. **A region whose longer side is already the size or more is left as it is.**
2. **Sampling.** The model gets ComfyUI's Differential Diffusion (on a clone; the input model is not changed). The crop is encoded as ComfyUI's Inpaint Model Conditioning does it, with the region's mask, softened at its edge (a Gaussian, sigma 2% of the crop's shorter side, about 20 px at 1024), as the noise mask: a model with inpaint channels also gets the masked image, any other model ignores it. The structure pass samples at `denoise`, then the refine pass samples its latent again at `refine_denoise`; there is no decode between them.
3. **The paste.** The decoded crop is shrunk back (antialiased bicubic), its colour matched to the image's in the mask's edge band (each channel's mean and standard deviation within about two sigmas inside the edge), and blended in with a Laplacian pyramid (Burt and Adelson) under the mask softened inward, so fine detail switches over a short distance and colour over a long one. **Only the pixels inside the mask change**; every other pixel keeps its exact value.

### ControlNets

A ControlNet applied with ComfyUI's Apply ControlNet (for example Qwen-Image-2512 Fun ControlNet Union) sits in `positive` / `negative`. For each crop, every ControlNet of the chain is copied and its hint image (and any image-sized extra input) cut to the same part of the image, so the hint lines up with the crop; ComfyUI then scales the cut hint to the crop's latent as it always does. A hint of another size than the image is mapped the way ComfyUI maps it onto the whole image (its centre cut to the image's aspect). The conditioning you connect is not changed.

Z-Image Turbo's Fun ControlNet Union (ComfyUI's ModelPatchLoader, then its ZImageFunControlnet node) patches the `MODEL` instead. For each crop the detailer samples a copy of the model whose ControlNet patch holds the hint, and the inpaint image and mask when they are given, cut to the crop the same way; ComfyUI then encodes the cut hint at the crop's sampling size. This holds for every model the nodes sample with: Face Detailer's `model` and `model_no_lora`, and its face, mouth and eye passes. The model you connect is not changed.

Not cut: other ControlNets applied as a model patch (ComfyUI's Qwen-Image DiffSynth ControlNet node, for one), area or mask conditioning, and reference latents (Qwen-Image Edit).

## `BC_FaceDetailer` — Face Detailer

`image` (`IMAGE`), `model` (`MODEL`), `vae` (`VAE`), `positive` and `negative` (`CONDITIONING`), the sampling widgets, `face_size` and `padding` (above), `sam3_model`, `threshold`, `min_face_height`, `face`, `mouth`, `eyes`; optional `model_no_lora` (`MODEL`) and `mask` (`MASK`) → `image` (`IMAGE`), `mask` (`MASK`, the pixels that changed).

SAM 3 finds every face of each frame (`sam3_model` is a checkpoint under `models/checkpoints`; the default `sam3.1_multiplex_fp16.safetensors` is downloaded on first use; `threshold` is its detection threshold). The prompts are fixed: `face` (every face), `teeth, lips` (the mouth), `eyes`.

**The character and the models.** `model` is the pipeline's main model, with the character LoRA. The character's face is detailed with it: the face `mask` overlaps most when `mask` is connected (no face is the character when it overlaps none), the face of the largest area otherwise. The character is chosen among every face found, before the size gate. With `model_no_lora` connected, every other face is detailed with it; without it, the other faces are left as they are. The character goes first, so the model changes once per frame.

**The passes.** Each face gets up to three passes of the core, each a region of its own:

| Toggle | Default | Region |
| --- | --- | --- |
| `face` | on | the face's mask |
| `mouth` | off | `teeth` and `lips` as one region |
| `eyes` | on | each eye on a crop of its own |

The mouth and the eyes are found on the face's box of the frame the face pass left, and cut to the face's mask (another face's eye inside the box is not taken). They use the face's model, the same widgets and the same rule: the mouth's or the eye's box grown by `padding`, its longer side redrawn at `face_size` px. An eye is sampled the way a face is; only its crop holds less of the scene.

**The size gate.** A face's height is the height of its SAM 3 mask's box, its longer side the longer side of that box (the height, for an upright face), in px.

- Height under `min_face_height` (default `64`): no pass at all, not even the mouth or the eyes. At that size an eye is about 13 px and the model would invent the face rather than detail it.
- Longer side at `face_size` or more: the face pass is skipped (nothing to gain); the mouth and eyes still run.
- The mouth and each eye skip their own pass the same way, when their longer side is `face_size` or more. At the default `640` a face up to 639 px is redrawn, and its eyes (about a fifth of the face's width each) until the face is about 3200 px wide: a large face keeps its own pixels while its eyes are still detailed.

The image and the mask are computed together (the mouth and the eyes are found on the detailed face): an unconnected one is dropped when the node returns.

Frames one at a time; a half-precision image (float16 / bfloat16) is read a frame at a time as float32 and both outputs come out in its dtype.

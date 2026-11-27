# Sampling

## `BC_DualSampler` — Dual Sampler

`model` (`MODEL`) + `positive`, `negative` + `latent_image` + `vae` + `seed` + the two passes' settings + the upscale's + `second_seed` + `dype` + optional `model_2` (`MODEL`) and `dype_config` → `first_image`, `second_image` (`IMAGE`) + `latent` (the second pass's `LATENT`): a two-pass hires fix in one node, for any model. DyPE takes Z-Image and Qwen-Image (2512); for another model set `dype` to `disable`. The defaults are Z-Image Turbo's.

### What it runs

1. DyPE first, before any sampling: DyPE (DyPE Patcher's code, [docs](model.md#bc_dypepatcher--dype-patcher)) goes on a clone of the second pass's model (`model_2`, or `model` when `model_2` is not linked), in the `dype` mode, with `dype_config`'s values; its start sigma is the sigma of the second pass's first model call, read by DyPE in the run: the schedule as that pass gets it, after any shift the model's wrappers apply per run (Qwen Image Shift's dynamic shift from the second pass's latent size, with its refine-pass rule, and its `shift_terminal` stretch). A start sigma taken from the schedule before the run would miss what those wrappers apply: in Qwen Image Shift's dynamic mode the model is at shift 1 outside a run. A model DyPE does not take (with `dype` not `disable`), or a DyPE Config of another model, stops the run here, before the first pass. At `denoise_2` 0 no DyPE is put on.
2. The first pass: ComfyUI's KSampler with `model`, `seed`, `steps_1`, `cfg_1`, `sampler_1`, `scheduler_1`, always at denoise 1.0.
3. The upscale to the first latent's size × `scale_by`, each side rounded to the nearest multiple of 16 px (8 px VAE cells × Z-Image's and Qwen-Image's 2 × 2 patches; 1024 × 1536 × 2.0 = 2048 × 3072; one cell for a VAE of 16 px or more):
   - the latent way: ComfyUI's *Upscale Latent By* with `upscale_method` (no crop);
   - the pixel way: the first latent decoded, the image resized to the same size with `pixel_upscale_method` (`lanczos`, as ComfyUI's: PIL on 8 bits; or `bicubic`) and encoded with `vae`;
   - `blend_factor`, the pixel way's weight, mixes them as ComfyUI's *Latent Blend* mixes samples1 (pixel) and samples2 (latent): pixel × b + latent × (1 − b). At 0 only the latent way runs (no decode, no encode); at 1 only the pixel way. The other fields of the latent (batch index, noise mask) are kept. Qwen-Image's latent, with its frame axis (16 channels × 1 frame), goes through both ways and the blend as it is; its pixel way encodes each image of a batch on its own, since ComfyUI's VAE Encode with Qwen-Image's VAE takes an image batch for the frames of one video (one latent for the whole batch).
4. The second pass: KSampler with the DyPE clone of `model_2` (of `model` when `model_2` is not linked) at `denoise_2`, the seed itself (`second_seed` fixed) or the next one (increment; after the last seed, 0). A ControlNet in the conditioning takes DyPE's positions in this pass too. At `denoise_2` 0 the pass does not sample.
5. The decodes: ComfyUI's VAE Decode of each pass's latent, for the outputs that are linked. The pixel way's decode is also `first_image` (one decode). Without the pixel way both decodes run after the second pass, so the VAE is loaded once.

`model` runs both passes when `model_2` is not linked, with every patch on it (a ControlNet model patch included). `model_2` gives the second pass a model of its own: another LoRA, another patch, or `model` without the first pass's ControlNet model patch. The shift is each model's own (ModelSamplingAuraFlow or the model's default); the node never changes it. To give the second pass a shift of its own, link `model_2` with that shift, or put [Dual Sampling Shift](model.md#bc_dualsamplingshift--dual-sampling-shift) (`BC_DualSamplingShift`) on `model` and leave `model_2` unlinked: a run from full noise (the first pass, at denoise 1.0) samples at its `shift_1`, a run that adds new noise below full noise (the second pass, at a `denoise_2` under 1) at its `shift_2`.

### Defaults

Both passes 9 steps, cfg 1.0, `dpmpp_sde`, `beta`; `denoise_2` 0.45; `bislerp` × 2.0; `blend_factor` 0.5, `lanczos`; `second_seed` increment; `dype` auto (on for a second latent whose longer side is over the model's `auto_threshold`: 1920 px for Z-Image and Qwen-Image, or DyPE Config's).

### Against the chain it replaces

At `blend_factor` 0 the node gives what this chain of core nodes gives, bit for bit: KSampler (`model`, denoise 1.0) → Upscale Latent By → KSampler (DyPE Patcher on the second pass's model with `dype_start_sigma` set to the second pass's first sigma, seed + 1) → VAE Decode for each pass. At 1 the upscale is VAE Decode → Upscale Image (lanczos) → VAE Encode, between them Latent Blend of the two. `tests/test_runtime.py` (`dype_model`) checks all three on a tiny Z-Image and a tiny Qwen-Image.

A VAE whose latent does not match the model's (another channel count, size or batch) stops the run with an error when the pixel way runs.

# Sampling

## `BC_DualSampler` — Dual Sampler

`model_1`, `model_2` (`MODEL`) + `positive`, `negative` + `latent_image` + `vae` + `seed` + the two passes' settings + the upscale's + `second_seed` + `dype` + optional `dype_config` → `first_image`, `second_image` (`IMAGE`) + `latent` (the second pass's `LATENT`): a two-pass hires fix in one node, for any model. DyPE takes Z-Image and Qwen-Image (2512); for another model set `dype` to `disable`. The defaults are Z-Image Turbo's.

### What it runs

1. DyPE first, before any sampling: DyPE (DyPE Patcher's code, [docs](model.md#bc_dypepatcher--dype-patcher)) goes on a clone of `model_2` for the second pass, in the `dype` mode, with `dype_config`'s values; its start sigma is the first sigma of the second pass's own schedule (the steps, sampler, scheduler and denoise given, the shift on `model_2`), read at each run. A model DyPE does not take (with `dype` not `disable`), or a DyPE Config of another model, stops the run here, before the first pass. At `denoise_2` 0 no DyPE is put on.
2. The first pass: ComfyUI's KSampler with `model_1`, `seed`, `steps_1`, `cfg_1`, `sampler_1`, `scheduler_1`, always at denoise 1.0.
3. The upscale to the first latent's size × `scale_by`, each side rounded to the nearest multiple of 16 px (8 px VAE cells × Z-Image's and Qwen-Image's 2 × 2 patches; 1024 × 1536 × 2.0 = 2048 × 3072; one cell for a VAE of 16 px or more):
   - the latent way: ComfyUI's *Upscale Latent By* with `upscale_method` (no crop);
   - the pixel way: the first latent decoded, the image resized to the same size with `pixel_upscale_method` (`lanczos`, as ComfyUI's: PIL on 8 bits; or `bicubic`) and encoded with `vae`;
   - `blend_factor`, the pixel way's weight, mixes them as ComfyUI's *Latent Blend* mixes samples1 (pixel) and samples2 (latent): pixel × b + latent × (1 − b). At 0 only the latent way runs (no decode, no encode); at 1 only the pixel way. The other fields of the latent (batch index, noise mask) are kept. Qwen-Image's latent, with its frame axis (16 channels × 1 frame), goes through both ways and the blend as it is; its pixel way encodes each image of a batch on its own, since ComfyUI's VAE Encode with Qwen-Image's VAE takes an image batch for the frames of one video (one latent for the whole batch).
4. The second pass: KSampler with `model_2`'s DyPE clone at `denoise_2`, the seed itself (`second_seed` fixed) or the next one (increment; after the last seed, 0). A ControlNet in the conditioning takes DyPE's positions in this pass too. At `denoise_2` 0 the pass does not sample.
5. The decodes: ComfyUI's VAE Decode of each pass's latent, for the outputs that are linked. The pixel way's decode is also `first_image` (one decode). Without the pixel way both decodes run after the second pass, so the VAE is loaded once.

`model_1` and `model_2` may be the same model, or differently patched clones of it (a ControlNet model patch on the first pass only). The shift is each model's own (ModelSamplingAuraFlow or the model's default); the node never changes it.

### Defaults

Both passes 9 steps, cfg 1.0, `dpmpp_sde`, `beta`; `denoise_2` 0.45; `bislerp` × 2.0; `blend_factor` 0.5, `lanczos`; `second_seed` increment; `dype` auto (on for a second latent whose longer side is over the model's `auto_threshold`: 1920 px for Z-Image and Qwen-Image, or DyPE Config's).

### Against the chain it replaces

At `blend_factor` 0 the node gives what this chain of core nodes gives, bit for bit: KSampler (`model_1`, denoise 1.0) → Upscale Latent By → KSampler (DyPE Patcher on `model_2` with `dype_start_sigma` set to the second pass's first sigma, seed + 1) → VAE Decode for each pass. At 1 the upscale is VAE Decode → Upscale Image (lanczos) → VAE Encode, between them Latent Blend of the two. `tests/test_runtime.py` (`dype_model`) checks all three on a tiny Z-Image and a tiny Qwen-Image.

A VAE whose latent does not match the model's (another channel count, size or batch) stops the run with an error when the pixel way runs.

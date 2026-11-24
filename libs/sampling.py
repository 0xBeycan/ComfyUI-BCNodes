"""One sampler run as core's KSampler node makes it: common_ksampler (ComfyUI's nodes.py, which this pack may not
import), restated. comfy.sample, comfy.utils and latent_preview are imported on first use."""


def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
    """core's common_ksampler on the LATENT dict `latent`: its empty-latent channel fix, its noise (seed, the
    latent's batch_index), its noise_mask, the preview callback and progress bar -> the LATENT dict it returns
    (`latent`'s other fields kept, the downscale ratios dropped)."""
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent_image = comfy.sample.fix_empty_latent_channels(model, latent["samples"], latent.get("downscale_ratio_spacial", None),
                                                          latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, seed, latent.get("batch_index", None))
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, noise_mask=latent.get("noise_mask", None),
                                  callback=latent_preview.prepare_callback(model, steps),
                                  disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out

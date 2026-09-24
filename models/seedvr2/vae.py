"""The SeedVR2 VAE: its latent channels, the working set per tile pixel, the check that
a VAE is the SeedVR2 one, and the VRAM room made before it runs."""

import logging

import torch

LATENT_CHANNELS = 16
# Working set of the SeedVR2 VAE per tile pixel, in bytes, independent of the
# frame count. Measured on a 5090 at 1088x1920 (float16, one tile): the
# encoder's first slice needs more than 28.5 GB (13.6 KB/px; it ran out of
# memory there in a process holding nothing else), the decoder is in the same
# class and not measured. Biased 20% high: an under-estimate kills the run,
# an over-estimate costs a DiT reload.
DECODER_BYTES_PER_PIXEL = 17_000
ENCODER_BYTES_PER_PIXEL = 16_300


def vae_model(vae):
    model = getattr(vae, "first_stage_model", None)
    if type(model).__name__ != "VideoAutoencoderKLWrapper":
        raise ValueError(f"SeedVR2 VAE Encode/Decode: needs the SeedVR2 VAE, got {type(model).__name__}")
    return model


def _free_vram(device):
    """Free VRAM in bytes as the driver reports it, after handing cached blocks back.

    ComfyUI's get_free_memory adds `reserved - active` from torch.cuda.memory_stats,
    and under the cudaMallocAsync backend (ComfyUI's default on CUDA) `active` stays
    0, so everything the pool holds — live model weights included — reads as free.
    After a trim, mem_get_info is honest under both allocators."""
    import comfy.model_management as mm

    mm.soft_empty_cache()
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    return mm.get_free_memory(device)


def make_room_for_vae(vae, needed):
    """Load the VAE with `needed` bytes of VRAM free for its working set.

    ComfyUI's load_models_gpu will not evict one "dynamic" model (the DiT) for
    another (the VAE): free_memory(for_dynamic=True) skips them
    (comfy/model_management.py). At 1080p the decoder's working set plus a
    resident DiT does not fit a 32 GB card, so when the free VRAM is short
    everything is unloaded first; the DiT is staged in RAM and comes back on
    demand, the VAE is reloaded right here."""
    import comfy.model_management as mm

    free = _free_vram(vae.device)
    if free < needed:
        mm.unload_all_models()
        after = _free_vram(vae.device)
        logging.info("SeedVR2 VAE: %.1f GiB free, working set needs %.1f GiB -> unloaded all models, %.1f GiB free now",
                     free / 2 ** 30, needed / 2 ** 30, after / 2 ** 30)
    else:
        logging.info("SeedVR2 VAE: %.1f GiB free, working set needs %.1f GiB -> models stay", free / 2 ** 30, needed / 2 ** 30)
    mm.load_models_gpu([vae.patcher], memory_required=needed, force_full_load=vae.disable_offload)

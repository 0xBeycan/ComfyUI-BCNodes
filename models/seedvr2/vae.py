"""The SeedVR2 VAE: its latent channels, the working set per tile pixel, the check that
a VAE is the SeedVR2 one, the tile tile_size 0 (auto) picks for the card, and the VRAM room
made before it runs."""

import logging

import torch

LATENT_CHANNELS = 16
# Working set of the SeedVR2 VAE in bytes, independent of the frame count: a fixed part plus a part
# per pixel of the largest spatial tile. It is the free VRAM, as the driver reports it, that the VAE
# needs, which is what make_room_for_vae compares it with. Measured on an RTX PRO 4500 (32 GB) under
# ComfyUI's own setup (cudaMallocAsync, DynamicVRAM on), 81 frames of 1088x1920, overlap 256, one
# encode or decode per process:
#   - the VRAM growth with nothing else loaded runs well above torch's own peak, because the async
#     pool keeps the blocks it freed (decoder, 1024 tile: 17.9 GiB allocated, 25 to 28.4 GiB on the
#     driver). Encoder, tile 512 / 768 / 1024 / 1280 / 1536 / 2048: 7.9 / 13.7 / 14.0 / 18.6 / 25.9 /
#     27.7 GiB; decoder, tile 512 / 768 / 1024: 10.5 / 18.1 / 28.4 GiB (1280 and up do not fit 32 GB);
#   - with the rest of the card held, the free VRAM at which the op ran to the end: decoder 512: fails
#     at 11 GiB, runs at 13; 768: fails at 16, runs at 20; 1024: fails at 29, runs at 31 (the whole
#     card); encoder 512: fails at 5, runs at 7; 1024: runs at 21 (failed once at 19).
# The decoder's figure is the line through its 512 and 768 points that ran (13 and 20 GiB), which gives
# 29.8 GiB at 1024, between the 29 that failed and the 31 that ran. The encoder's is the line through
# 5% over its 768 growth and the 21 GiB that ran at 1024; it lies above every other measured point. An
# under-estimate kills the run, an over-estimate costs a DiT reload. The old figures (17,000 and
# 16,300 bytes per pixel, no fixed part, the decoder never measured) put the decoder's 1024 tile at
# 16.6 GiB, and a 32 GB card with the DiT resident ran out of memory with 20.7 GiB free.
#
# Checked on an RTX 5090 (device limit 31.36 GiB, 30.7 GiB free with every model unloaded), 81 frames of
# 1088x1920, overlap 256. Encoder, tile 1024: estimate 21.0 GiB, NVML peak 15.99; whole frame (2048):
# estimate 36.0, NVML peak 30.08, and the same tile on 901 frames ran out of memory in its first slice
# (28.67 GiB allocated + 2.49 asked). Decoder, tile 1024: estimate 29.8, NVML peak 31.22 (30.0 reserved,
# the driver's own ~1.2 on top); whole frame: estimate 52.0, out of memory at its first slice. The
# encoder's estimate is 5 to 6 GiB over its NVML peak on a free card, the margin the hold test above
# showed it needs on a tight one and the 901-frame failure shows on a free one; the decoder's matches.
ENCODER_FIXED_BYTES = 6_360_000_000
ENCODER_BYTES_PER_PIXEL = 15_440
DECODER_FIXED_BYTES = 7_950_000_000
DECODER_BYTES_PER_PIXEL = 22_940
# What the driver never hands out with no model loaded (the CUDA context, the libraries' handles): the 5090
# above had 30.7 of its 31.36 GiB free with every model unloaded. tile_size 0 (auto) fits the working set
# into the card's total less this.
CONTEXT_BYTES = 768 * 2 ** 20
TILE_STEP = 32  # the tile_size widget's step; an auto tile is a value it could hold
MIN_TILE = 64  # the tile_size widget's smallest tile


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


def tile_for(tile_size, height, width, fixed, per_pixel, device, label):
    """The spatial tile a VAE flow runs with: `tile_size` itself, or for 0 (auto) the largest tile (a
    multiple of TILE_STEP) whose working set, `fixed` + `per_pixel` x the tile's pixels on a `height` x
    `width` frame, fits the card's total less CONTEXT_BYTES. That is the smallest tile covering the frame
    (one tile, no blend) when it fits, else the largest one that does."""
    import comfy.model_management as mm

    if tile_size != 0:
        return tile_size
    budget = mm.get_total_memory(device) - CONTEXT_BYTES
    tile = -(-max(height, width) // TILE_STEP) * TILE_STEP

    def working_set(t):
        return fixed + per_pixel * min(height, t) * min(width, t)

    while tile > MIN_TILE and working_set(tile) > budget:
        tile -= TILE_STEP
    if working_set(tile) > budget:
        logging.warning("%s: even a %d tile needs %.1f GiB, more than the %.1f GiB this card gives: it may run out of memory",
                        label, tile, working_set(tile) / 2 ** 30, budget / 2 ** 30)
    logging.info("%s: tile_size auto -> %d (%dx%d frame, %.1f GiB of %.1f GiB)", label, tile, width, height,
                 working_set(tile) / 2 ** 30, budget / 2 ** 30)
    return tile


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

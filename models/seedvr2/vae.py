"""The SeedVR2 VAE: its latent channels, the working set per tile pixel, the check that
a VAE is the SeedVR2 one, the tiling a tile_size runs (0, auto: the one computing the least that fits
the card), and the VRAM room made before it runs."""

import logging

import torch

from .tiling import spans

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
# Checked on an RTX 5090 (device limit 31.36 GiB, 30.7 GiB free with every model unloaded), 1088x1920,
# overlap 256, with the Process Monitor off (on, it kept the running node's frames alive and inflated every
# figure, the ones above likely included). Encoder: a 1088 x 1568 tile, estimate 30.5 GiB, driver peak
# 22.77 on 901 frames (allocated 16.81, reserved 21.59); the whole frame, estimate 36.0, driver peak 27.89
# (19.54 / 26.72), and a standalone run allocated the same 19.56 GiB at its peak slice on 81 and on 901
# frames. Decoder: 1024 tile, estimate 29.8, driver peak 25.72 on 901 frames (17.90 / 24.50); 1088 x
# 1568, estimate 43.9, out of memory (24.04 allocated + 2.03 asked); the whole frame, estimate 52.0, out
# of memory. Both estimates stay: each is over every peak that ran and the decoder's above its failures.
# The decoder's is 1.2x its driver peak at 1024; the encoder's is 1.34x to 1.40x (1.2x would be about
# 0.62 GB fixed + 16,300 bytes per pixel), left as it is until the bigger cards are measured with the
# monitor off.
ENCODER_FIXED_BYTES = 6_360_000_000
ENCODER_BYTES_PER_PIXEL = 15_440
DECODER_FIXED_BYTES = 7_950_000_000
DECODER_BYTES_PER_PIXEL = 22_940
# What the driver never hands out with no model loaded (the CUDA context, the libraries' handles): the 5090
# above had 30.7 of its 31.36 GiB free with every model unloaded. tile_size 0 (auto) fits the working set
# into the card's total less this.
CONTEXT_BYTES = 768 * 2 ** 20
TILE_STEP = 32  # the tile_size widget's step; an auto tile's sides are values it could hold
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


def tile_for(tile_size, height, width, overlap, axis, cell, fixed, per_pixel, device, label):
    """The tiling a VAE flow runs: ((rows, overlap), (columns, overlap)) in its grid's cells (`cell` pixels
    a side: 1 for the encoder, 8 for the decoder's latent) on a `height` x `width` pixel frame.

    A typed `tile_size` is the same on both axes, its overlap cut by `axis` (tiling.encode_axis /
    decode_axis), as before. 0 (auto) picks the tile sides (multiples of TILE_STEP) that compute the fewest
    pixels (every tile in full, overlaps counted again) with a working set, `fixed` + `per_pixel` x the
    largest tile's pixels, inside the card's total less CONTEXT_BYTES; then the fewest tiles, then the
    smallest working set. The overlap stays the one given: a side that does not cover the frame is at least
    twice it. Without such a tiling, the largest square tile that fits, as a typed one would run."""
    import comfy.model_management as mm

    if tile_size != 0:
        square = axis(tile_size, overlap)
        return square, square
    budget = mm.get_total_memory(device) - CONTEXT_BYTES

    def sides(length):
        """Per tile count along an axis of `length` pixels, the smallest side with that count:
        (side, count, pixels its tiles cover together, pixels of its largest tile)."""
        best = {}
        for side in range(max(MIN_TILE, -(-length // TILE_STEP) * TILE_STEP), MIN_TILE - 1, -TILE_STEP):
            if side < length and side < 2 * overlap:
                break
            row = spans(length // cell, side // cell, overlap // cell)
            best[len(row)] = (side, len(row), sum(end - start for start, end in row) * cell, min(length // cell, side // cell) * cell)
        return best.values()

    chosen = None
    for rows in sides(height):
        for cols in sides(width):
            working_set = fixed + per_pixel * rows[3] * cols[3]
            key = (rows[2] * cols[2], rows[1] * cols[1], working_set)
            if working_set <= budget and (chosen is None or key < chosen[0]):
                chosen = (key, rows, cols)
    if chosen is not None:
        (computed, count, working_set), rows, cols = chosen
        logging.info("%s: tile_size auto -> %dx%d (%d x %d tiles on a %dx%d frame, %.2f Mpx computed, %.1f GiB of %.1f GiB)",
                     label, cols[0], rows[0], cols[1], rows[1], width, height, computed / 1e6, working_set / 2 ** 30, budget / 2 ** 30)
        return (rows[0] // cell, overlap // cell), (cols[0] // cell, overlap // cell)

    tile = max(MIN_TILE, -(-max(height, width) // TILE_STEP) * TILE_STEP)

    def square_set(t):
        return fixed + per_pixel * min(height, t) * min(width, t)

    while tile > MIN_TILE and square_set(tile) > budget:
        tile -= TILE_STEP
    if square_set(tile) > budget:
        logging.warning("%s: even a %d tile needs %.1f GiB, more than the %.1f GiB this card gives: it may run out of memory",
                        label, tile, square_set(tile) / 2 ** 30, budget / 2 ** 30)
    logging.info("%s: tile_size auto -> %d (no tiling keeping the overlap %d fits; %dx%d frame, %.1f GiB of %.1f GiB)", label,
                 tile, overlap, width, height, square_set(tile) / 2 ** 30, budget / 2 ** 30)
    square = axis(tile, overlap)
    return square, square


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

"""Chunk Size — the frames per chunk ComfyUI's Split SeedVR2 Latent gets in its `manual` mode: the
largest 4n+1 chunk whose DiT working set fits the card, at most the clip.

The working set is the DiT's law (models/seedvr2/dit.py): a fixed part plus a part per latent frame
per megapixel of the DiT's frame, the latter times 1 + safety_margin. The card is ComfyUI's total for
its torch device (on CUDA the driver's total, the "device limit" of an out-of-memory message), not the
free memory when the node runs: what is loaded then (the VAE, a DiT from the last run) is unloaded or
evicted by the time the sampler needs the room, so the pick is the same on every run of a card.
Split SeedVR2 Latent's own `auto` budgets the free memory with a law fitted on the 3B and ran out of
memory on both cards measured (281 frames at 1080p on a 96 GB card).
"""

import logging

from ...models.seedvr2.dit import FIXED_GIB, GIB_PER_MPX_LATENT_FRAME
from ...models.seedvr2.vae import LATENT_CHANNELS

GIB = 2 ** 30
NODE = "BC_SeedVR2ChunkSize"


def frames_per_chunk(latent, safety_margin):
    """-> (frames,): the largest 4n+1 pixel-frame chunk of `latent` the card holds, at most its frames."""
    import comfy.model_management as mm

    z = latent["samples"]
    if z.ndim != 5 or z.shape[1] != LATENT_CHANNELS:
        raise ValueError(f"{NODE}: expected a SeedVR2 latent (B, {LATENT_CHANNELS}, T, H, W), got {tuple(z.shape)}; "
                         "connect the latent of SeedVR2 Preprocess (Compact) or SeedVR2 VAE Encode")
    if safety_margin < 0:
        raise ValueError(f"{NODE}: safety_margin must be 0 or more, got {safety_margin}")
    b, _, t_latent, h, w = z.shape
    clip = 4 * (t_latent - 1) + 1
    device = mm.get_torch_device()
    total = mm.get_total_memory(device)
    mpx = b * h * w * 64 / 1e6  # the DiT's pixels per frame: 8 x 8 per latent cell, as core's own sizing counts them
    per_frame = GIB_PER_MPX_LATENT_FRAME * mpx * (1 + safety_margin)
    latent_frames = int((total / GIB - FIXED_GIB) // per_frame)
    if latent_frames < 1:
        logging.warning("%s: %.1f GiB on %s holds no latent frame of %.2f Mpx under safety_margin %.2f "
                        "(%.2f GiB fixed + %.2f GiB per frame): 1 frame per chunk, which may run out of memory; "
                        "lower the resolution", NODE, total / GIB, device, mpx, safety_margin, FIXED_GIB, per_frame)
        latent_frames = 1
    frames = min(4 * (latent_frames - 1) + 1, clip)
    logging.info("%s: %.2f GiB on %s, %.2f Mpx per frame, safety_margin %.2f -> %d frames per chunk (the clip has %d)",
                 NODE, total / GIB, device, mpx, safety_margin, frames, clip)
    return (frames,)

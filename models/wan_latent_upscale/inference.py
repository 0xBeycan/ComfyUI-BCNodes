"""upscale(samples, path): the LATENT_UPSCALE entry "Wan 2.1 2x"."""

import torch

from .loader import load

CHANNELS = 16


@torch.no_grad()
def upscale(samples, path):
    """A Wan 2.1 latent, (B, 16, F, H, W) or (B, 16, H, W) -> the same with H and W doubled by the weights file at
    path, in the input's dtype on ComfyUI's intermediate device. Frames are upscaled one by one (the
    convolutions are spatial), the channel attention pools each batch item whole, as the model was trained."""
    import comfy.model_management as mm

    if samples.shape[1] != CHANNELS:
        raise ValueError(f"Wan 2.1 2x takes {CHANNELS}-channel latents (Wan 2.1, Wan 2.2 A14B, Qwen-Image); "
                         f"this latent has {samples.shape[1]}.")
    model, device = load(path)
    x = samples if samples.ndim == 5 else samples.unsqueeze(2)
    out = model(x.to(device=device, dtype=torch.float32))
    if samples.ndim == 4:
        out = out.squeeze(2)
    return out.to(device=mm.intermediate_device(), dtype=samples.dtype)

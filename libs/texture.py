"""The skin-texture engine: a unit-std pore field and the image's own detail, applied in
linear light inside a mask."""

import torch
import torch.nn.functional as F

from .color import linear_to_srgb, srgb_to_linear
from .filters import gauss_reflect

# Pore field geometry, in px at a 1024 px long edge (scaled with the image).
PORE_SIGMA = 0.9          # fine octave
PORE_SIGMA_COARSE = 2.2   # coarse octave
DETAIL_SIGMA = 2.0        # high-pass radius for the self-detail boost
MAX_AMPLITUDE = 0.06      # luminance modulation at texture = 1.0 (unit-std field)


def _smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _unit(x):
    return x / (x.flatten(1).std(dim=1).view(-1, 1, 1, 1) + 1e-8)


def pore_field(shape, scale, seed, device):
    """Unit-std pore field (B, 1, H, W): two band-passed noise octaves plus
    sparse darker pits. `scale` is px per reference px (long edge / 1024)."""
    b, _, h, w = shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((b, 1, h, w), generator=gen).to(device)
    fine = _unit(gauss_reflect(noise, PORE_SIGMA * scale * 0.55) - gauss_reflect(noise, PORE_SIGMA * scale * 1.6))
    coarse = _unit(gauss_reflect(noise, PORE_SIGMA_COARSE * scale * 0.55) - gauss_reflect(noise, PORE_SIGMA_COARSE * scale * 1.6))
    field = _unit(0.65 * fine + 0.35 * coarse)
    pits = -F.relu(coarse - 1.25) * 1.5
    return _unit(field + pits)


def apply_texture(image, mask, texture=0.5, detail=0.6, pore_scale=1.0, seed=0, gate=1.0):
    """image (B, H, W, 3) sRGB, mask (B, H, W) 0..1 -> textured image, same shape."""
    b, h, w, _ = image.shape
    device = image.device
    scale = max(h, w) / 1024.0 * float(pore_scale)
    x = image[..., :3].permute(0, 3, 1, 2).float()
    lin = srgb_to_linear(x.clamp(0, 1))
    lum = (lin * torch.tensor([0.2126, 0.7152, 0.0722], device=device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    m = mask.to(device).float().unsqueeze(1).clamp(0, 1)

    # where the texture lands: the mask, less in highlights, gone in the black
    weight = m * _smoothstep(0.01, 0.08, lum) * (1.0 - 0.7 * _smoothstep(0.55, 0.95, lum)) * float(gate)

    # (a) the image's own detail, boosted (a ratio, so colour is untouched)
    hf = lum - gauss_reflect(lum, DETAIL_SIGMA * scale)
    gain = (1.0 + float(detail) * hf / (lum + 0.02)).clamp(0.5, 1.5)
    lin = lin * (1.0 + (gain - 1.0) * weight)

    # (b) the synthetic pore field, multiplicative in linear light
    if texture > 0:
        field = pore_field(lum.shape, scale, seed, device)
        lin = lin * (1.0 + MAX_AMPLITUDE * float(texture) * field * weight)

    out = linear_to_srgb(lin.clamp(0, 1)).clamp(0, 1)
    return out.permute(0, 2, 3, 1).to(image.dtype)

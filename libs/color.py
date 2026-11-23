"""Colour helpers: the hex colour parser, the sRGB <-> linear transfer functions and the colour match of
a patch to a reference over a band of pixels (the detailer's paste)."""

import torch


def parse_hex_color(value):
    """'#rrggbb' or '#rrggbbaa' (also the 3-digit short form) -> (r, g, b, a) in 0..1."""
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"background_color must be #rrggbb or #rrggbbaa, got {value!r}")
    channels = [int(text[i:i + 2], 16) / 255.0 for i in range(0, len(text), 2)]
    if len(channels) == 3:
        channels.append(1.0)
    return channels


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0) ** (1 / 2.4) - 0.055)


def match_band(patch, reference, band):
    """`patch` (C, H, W) with each channel shifted and scaled so that its mean and standard deviation over
    the pixels of `band` (H, W bool) are `reference`'s there, clamped to [0, 1]; a channel flat in the band
    is only shifted. An empty band leaves `patch` as it is."""
    if not bool(band.any()):
        return patch
    p, r = patch[:, band], reference[:, band]
    p_std, r_std = p.std(dim=1, correction=0), r.std(dim=1, correction=0)
    scale = torch.where(p_std > 1e-6, r_std / p_std, torch.ones_like(p_std))
    return ((patch - p.mean(dim=1)[:, None, None]) * scale[:, None, None] + r.mean(dim=1)[:, None, None]).clamp(0, 1)

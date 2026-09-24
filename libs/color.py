"""Colour helpers: the hex colour parser and the sRGB <-> linear transfer functions."""

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

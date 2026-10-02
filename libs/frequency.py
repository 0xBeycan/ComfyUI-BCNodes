"""Frequency merge: the low frequencies of one image batch with the high frequencies of another.

The split is a Gaussian low-pass (gauss_reflect): low = G(x), high = x - G(x), so low + high is x
again. A Gaussian has no ringing (an ideal FFT cut-off rings around every edge) and is the same in
every direction; the reflect padding keeps the borders free of a dark or bright rim.
"""

import math

import torch

from .filters import gauss_reflect


def frequency_merge(base, detail, sigma, strength, device):
    """G(base) + strength * (detail - G(detail)), clamped to [0, 1]: `base` gives the structure and
    colour, `detail` the texture. `base` and `detail` are (B, H, W, C) batches of one size, C 3 or 4
    (alpha is dropped); `sigma` is the Gaussian's, in pixels. One image at a time in float32 on
    `device`, written into one (B, H, W, 3) output on the CPU in the inputs' promoted dtype."""
    if tuple(base.shape[:3]) != tuple(detail.shape[:3]):
        raise ValueError(
            f"Frequency Merge: base is {base.shape[2]}x{base.shape[1]} ({base.shape[0]} image(s)), detail is "
            f"{detail.shape[2]}x{detail.shape[1]} ({detail.shape[0]} image(s)); connect two batches of the same size and "
            "image count (resize one to the other's size first, e.g. with Image Resize)")
    batch, height, width = base.shape[:3]
    radius = int(math.ceil(3.0 * sigma))
    if batch and radius >= min(height, width):
        raise ValueError(
            f"Frequency Merge: split_sigma {sigma} blurs over {radius} px, which the {width}x{height} image does not "
            f"hold; set split_sigma to {math.floor((min(height, width) - 1) / 3 * 10) / 10} or less")
    out = torch.empty((batch, height, width, 3), dtype=torch.promote_types(base.dtype, detail.dtype))
    for i in range(batch):
        b = base[i, ..., :3].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        d = detail[i, ..., :3].to(device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        merged = d - gauss_reflect(d, sigma)  # a new tensor: detail's high frequencies
        merged.mul_(strength).add_(gauss_reflect(b, sigma)).clamp_(0, 1)
        out[i].copy_(merged[0].permute(1, 2, 0))
    return out

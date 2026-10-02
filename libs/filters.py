"""Separable Gaussian blurs.

gauss_reflect (skin texture) and gauss_replicate (BiRefNet matte) are two different filters,
kept apart on purpose: padding mode, pad order, the sigma cut-off and the kernel build differ,
so they do not give the same output. Do not merge them.
"""

import math

import torch
import torch.nn.functional as F


def gauss_reflect(x, sigma):
    """Separable Gaussian blur of (B, C, H, W), reflect-padded; sigma in px. Rows first, then
    columns, each pass a weighted sum of shifted slices of the padded image into one accumulator:
    conv2d on a CPU without oneDNN unfolds the kernel into an image-sized buffer per tap (6 GB for a
    3840x2160 image at sigma 32), and the sum is faster there too."""
    if sigma <= 0.2:
        return x
    radius = int(math.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    weights = (k / k.sum()).tolist()
    height, width = x.shape[-2:]
    padded = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    rows = padded.new_zeros(x.shape)
    for i, weight in enumerate(weights):
        rows.add_(padded[..., i:i + width], alpha=weight)
    padded = F.pad(rows, (0, 0, radius, radius), mode="reflect")
    out = rows.zero_()
    for i, weight in enumerate(weights):
        out.add_(padded[..., i:i + height, :], alpha=weight)
    return out


def gauss_replicate(mask, radius):
    """Gaussian blur of a (B, 1, H, W) matte; `radius` is the sigma in pixels."""
    sigma = float(radius)
    size = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(size, dtype=mask.dtype, device=mask.device) - size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    pad = size // 2
    out = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(out, kernel.view(1, 1, 1, size))
    return F.conv2d(out, kernel.view(1, 1, size, 1))

"""Separable Gaussian blurs.

gauss_reflect (skin texture) and gauss_replicate (BiRefNet matte) are two different filters,
kept apart on purpose: padding mode, pad order, the sigma cut-off and the kernel build differ,
so they do not give the same output. Do not merge them.
"""

import math

import torch
import torch.nn.functional as F


def gauss_reflect(x, sigma):
    """Separable Gaussian blur of (B, C, H, W), reflect-padded; sigma in px."""
    if sigma <= 0.2:
        return x
    radius = int(math.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


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

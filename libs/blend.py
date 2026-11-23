"""Multiband blending (Burt and Adelson, "A Multiresolution Spline With Application to Image Mosaics",
1983): each frequency band of two images is mixed under its own blur of the weight, so fine detail
switches over a short distance and colour over a long one. Plain torch.
"""

import math

import torch.nn.functional as F

SMALLEST = 16  # px: the pyramid halves while its shorter side stays at least this


def _down(x):
    return F.interpolate(x, size=((x.shape[-2] + 1) // 2, (x.shape[-1] + 1) // 2), mode="bilinear",
                         align_corners=False, antialias=True)


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def laplacian_blend(base, patch, weight):
    """`base` and `patch` (C, H, W) blended under `weight` (H, W) in [0, 1] (1 = patch), not clamped: the
    Laplacian pyramid of patch - base, each level scaled by the same level of the weight's Gaussian
    pyramid, added back onto base (the pyramid is linear, so this is the blend of the two pyramids)."""
    diff = (patch - base)[None]
    w = weight[None, None].to(diff.dtype)
    levels = max(0, int(math.log2(min(base.shape[-2:]) / SMALLEST)))
    bands = []
    for _ in range(levels):
        low = _down(diff)
        bands.append((diff - _up(low, diff.shape[-2:])) * w)
        diff, w = low, _down(w)
    out = diff * w
    for band in reversed(bands):
        out = _up(out, band.shape[-2:]) + band
    return base + out[0]

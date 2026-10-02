"""Depth Anything: a depth map per frame from a model of the depth family, near = white.

Output size:
  - no target size: the short side is `resolution`, the long side keeps the input's aspect, each
    side rounded half to even (libs/geometry.short_side_size), the size a ControlNet
    preprocessor's detect resolution gives;
  - a target size (height, width): exactly that size; on an aspect mismatch the depth map covers
    it with its aspect kept and is centre-cropped.
The model's inverse depth is resampled once, straight to that size (the covering size when
cropped), normalised per frame over the whole frame (min-max, nearest = 1, a flat frame 0), then
cropped and written on all three channels into one preallocated output: in the input's dtype when
that is half precision (each frame read with a plain .float(), libs/image.py), float32 otherwise.
"""

import torch

from ..libs.geometry import short_side_size
from ..libs.image import output_dtype
from ..models.common import registry
from ..models.common.registry import DEPTH


def _cover(h, w, height, width):
    """An h x w frame scaled to cover height x width with its aspect kept: (its height, its width,
    the top and left of the centred height x width window)."""
    k = max(height / h, width / w)
    ch, cw = max(height, int(round(h * k))), max(width, int(round(w * k)))
    return ch, cw, (ch - height) // 2, (cw - width) // 2


def _normalize(depth):
    """(H, W) inverse depth -> 0..1, nearest = 1; a flat frame -> 0."""
    lo = depth.amin()
    span = depth.amax() - lo
    return torch.where(span > 0, (depth - lo) / span.clamp_min(1e-12), torch.zeros_like(depth))


def estimate(rgb, model, resolution, size=None):
    """`rgb` is (B, H, W, 3) in 0..1 on any device; `size` is (height, width) or None. Returns the
    depth IMAGE (B, height, width, 3) on the CPU, in rgb's dtype when half precision, else float32."""
    b, h, w = rgb.shape[:3]
    if size is None:
        height, width = short_side_size(h, w, resolution)
        ch, cw, top, left = height, width, 0, 0
    else:
        height, width = size
        ch, cw, top, left = _cover(h, w, height, width)

    predict = registry.get(DEPTH, model)()
    out = torch.empty((b, height, width, 3), dtype=output_dtype(rgb))
    for i in range(b):
        depth = _normalize(predict(rgb[i].float(), resolution, (ch, cw)))
        out[i].copy_(depth[top:top + height, left:left + width].cpu().unsqueeze(-1))
    return out

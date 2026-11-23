"""Where a detail pass samples: the crop around a region and the size it is sampled at."""

from dataclasses import dataclass

import torch

# px: both sampling sides are multiples of this. A UNet halves its latent three times (8 x 8), a DiT pads
# its latent to its patch itself (16 or 32 px with an 8x or 16x VAE); 64 needs neither.
MULTIPLE = 64


@dataclass(frozen=True)
class Crop:
    """A region's window in the image, px (x0, y0 inclusive, x1, y1 exclusive), and the size it is
    sampled at (width, height)."""
    x0: int
    y0: int
    x1: int
    y1: int
    width: int
    height: int


def bounds(hard):
    """(x0, y0, x1, y1) of the True pixels of `hard` (H, W bool), x1 and y1 exclusive; None when there
    are none."""
    rows = torch.where(hard.any(dim=1))[0]
    if rows.numel() == 0:
        return None
    cols = torch.where(hard.any(dim=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def plan(box, image_width, image_height, context, resolution):
    """The Crop of the region whose bounds are `box`: `context` times the box around its centre, cut to
    the image, widened to the aspect of the sampling size (its long side `resolution`, both sides
    multiples of MULTIPLE) and moved inside the image. None when the crop would not be enlarged for
    sampling: sampling it at its own size or smaller adds no detail."""
    x0, y0, x1, y1 = box
    want_w = min(image_width, (x1 - x0) * context)
    want_h = min(image_height, (y1 - y0) * context)
    scale = resolution / max(want_w, want_h)
    width = max(MULTIPLE, round(want_w * scale / MULTIPLE) * MULTIPLE)
    height = max(MULTIPLE, round(want_h * scale / MULTIPLE) * MULTIPLE)
    # sampling px per image px: the crop at the sampling aspect covers want_w x want_h, and fits the image
    zoom = max(min(width / want_w, height / want_h), width / image_width, height / image_height)
    if zoom <= 1.0:
        return None
    crop_w, crop_h = min(image_width, round(width / zoom)), min(image_height, round(height / zoom))
    left = min(max(round((x0 + x1 - crop_w) / 2), 0), image_width - crop_w)
    top = min(max(round((y0 + y1 - crop_h) / 2), 0), image_height - crop_h)
    return Crop(left, top, left + crop_w, top + crop_h, width, height)

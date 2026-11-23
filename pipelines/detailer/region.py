"""Where a detail pass samples: the crop around a region and the size it is sampled at."""

import math
from dataclasses import dataclass

import torch

from ...libs.geometry import round_up_to_multiple

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


def plan(box, image_width, image_height, padding, size):
    """The Crop of the region whose bounds are `box`, sampled with the region's longer side at `size` px; None
    when that side is `size` px or more (sampling it adds no detail).

    The crop is the box grown by `padding` times its width on the left and on the right and its height above
    and below, cut to the image. Its sampling size is that crop at the region's zoom (size / the longer
    side), each side rounded up to a multiple of MULTIPLE; the crop is then taken at the sampling size's
    aspect and that zoom (so the rounding only adds surroundings), around the box's centre and moved inside
    the image. Only where that crop would be larger than the image does the zoom rise until it fits."""
    x0, y0, x1, y1 = box
    longer = max(x1 - x0, y1 - y0)
    if longer >= size:
        return None
    grow = 1 + 2 * padding
    want_w, want_h = min(image_width, (x1 - x0) * grow), min(image_height, (y1 - y0) * grow)
    zoom = size / longer
    # round(..., 6): a float error (1280.0000000002 for 1280) is not rounded up a whole MULTIPLE
    width, height = (max(MULTIPLE, round_up_to_multiple(math.ceil(round(px * zoom, 6)), MULTIPLE)) for px in (want_w, want_h))
    zoom = max(zoom, width / image_width, height / image_height)
    crop_w, crop_h = min(image_width, round(width / zoom)), min(image_height, round(height / zoom))
    left = min(max(round((x0 + x1 - crop_w) / 2), 0), image_width - crop_w)
    top = min(max(round((y0 + y1 - crop_h) / 2), 0), image_height - crop_h)
    return Crop(left, top, left + crop_w, top + crop_h, width, height)

"""The image batch at the size the official Qwen image processor gives it (the preprocessor_config of
Qwen3.5, Qwen3.8 and Qwen3-VL: patch 16 x merge 2 = factor 32, min_pixels 65536, max_pixels 16777216,
bicubic, antialiased), not core's defaults (min 3136, max 12845056, bilinear). Core's own resize then
keeps every frame as it is, up to 12845056 pixels; a frame the official sizing puts between 12845056 and
16777216 pixels is still capped by core.

The official processor resizes 8-bit images (PIL's bicubic, or torchvision's uint8 kernel that copies it): a
pass along the width, then one along the height, each stored in 8 bits, so each clamps the overshoot of the
bicubic kernel to [0, 1]. The resize here makes the same two passes in float32 and clamps after each: on
8-bit frames it stays within about one level of the official result, where one 2-D pass clamped only at
the end differs by up to 26 levels next to sharp edges. A frame kept at its size is only clamped.
"""

import math

import torch
import torch.nn.functional as F

from ...libs.image import float_frame, is_half

FACTOR = 32
MIN_PIXELS = 65536
MAX_PIXELS = 16777216


def smart_resize(height, width):
    """(h, w): the official smart_resize: each side rounded to a multiple of FACTOR, then scaled (aspect kept)
    into [MIN_PIXELS, MAX_PIXELS]."""
    if min(height, width) < 1:
        raise ValueError(f"an image of {width}x{height} has no pixels: connect an image with at least one pixel per "
                         f"side, or disconnect image")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"an image of {width}x{height} is more than 200 times wider than tall (or taller than "
                         f"wide), which the Qwen image processor refuses: crop it")
    h_bar = round(height / FACTOR) * FACTOR
    w_bar = round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def prepare_images(image):
    """The IMAGE batch (N, H, W, 3 or 4) as a float32 (N, h, w, 3) batch at smart_resize's size, one image per
    frame; None for None or an empty batch. A half-precision batch is read a frame at a time (float_frame);
    an alpha channel is dropped, as the official processor converts to RGB."""
    if image is None:
        return None
    if image.ndim != 4 or image.shape[-1] not in (3, 4):
        raise ValueError(f"image must be an IMAGE batch (frames, height, width, 3), got shape {tuple(image.shape)}")
    if image.shape[0] == 0:
        return None
    n, h, w = image.shape[:3]
    th, tw = smart_resize(h, w)
    out = torch.empty((n, th, tw, 3), dtype=torch.float32)
    buf = torch.empty((h, w, 3), dtype=torch.float32, device=image.device) if is_half(image) else None
    for i in range(n):
        x = float_frame(image[i, :, :, :3], out=buf).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
        for size in ((h, tw), (th, tw)):  # the width pass, then the height pass
            if size != tuple(x.shape[2:]):
                x = F.interpolate(x, size=size, mode="bicubic", align_corners=False, antialias=True).clamp_(0.0, 1.0)
        out[i].copy_(x[0].permute(1, 2, 0))
    return out.clamp_(0.0, 1.0)

"""Matting flow: the matte of the one matting implementation (models/birefnet), then the
matte options (finish).

The matte can be post-processed before it is applied: sensitivity, blur,
grow / shrink, invert, an edge refinement of the foreground colours, and a
solid background colour instead of alpha. All of it is plain torch on the
(B, 1, H, W) matte, so the whole batch goes through at once.
"""

import torch

from ..libs import mask as mask_ops
from ..libs.color import parse_hex_color
from ..libs.filters import gauss_replicate
from ..models.birefnet.inference import matte


def finish(rgb, mask, sensitivity=1.0, mask_blur=0, mask_offset=0, invert_output=False,
           refine_foreground=False, background="Alpha", background_color="#222222"):
    """Apply the options to a raw matte and build the three outputs.
    `rgb` is (B, H, W, 3) in 0..1, `mask` (B, H, W) in 0..1, both on the CPU."""
    m = mask.unsqueeze(1).float()
    if sensitivity < 1.0:
        m = (m * (1 + (1 - sensitivity))).clamp_(0, 1)
    if mask_blur > 0:
        m = gauss_replicate(m, mask_blur)
    if mask_offset != 0:
        m = mask_ops.offset_matte(m, mask_offset)
    if invert_output:
        m = 1 - m
    color = rgb.float()
    if refine_foreground:
        color = mask_ops.refine_foreground(color, m)
    alpha = m.permute(0, 2, 3, 1)  # (B, H, W, 1)
    if background == "Alpha":
        image = torch.cat((color, alpha), dim=-1)
    else:
        r, g, b, a = parse_hex_color(background_color)
        bg = torch.tensor([r, g, b], dtype=color.dtype).view(1, 1, 1, 3)
        # Straight-alpha "over" onto a background that may itself be
        # translucent, then the alpha is dropped.
        weight = a * (1 - alpha)
        total = alpha + weight
        image = torch.where(total > 0, (color * alpha + bg * weight) / total.clamp(min=1e-6), torch.zeros_like(color))
    mask_out = m[:, 0]
    mask_image = mask_out.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    return image, mask_out, mask_image


def remove_background(image, model, sensitivity, mask_blur, mask_offset, invert_output,
                      refine_foreground, background, background_color):
    """IMAGE (B, H, W, C) and a checkpoint name -> (image, mask, mask_image) as finish builds
    them. An unknown name raises KeyError(name) inside the load, after the cached model is
    evicted."""
    rgb = image[..., :3]
    mask = matte(model, rgb)
    return finish(rgb.cpu(), mask, sensitivity, mask_blur, mask_offset, invert_output,
                  refine_foreground, background, background_color)

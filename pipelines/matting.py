"""Matting flow: the matte of the one matting implementation (models/birefnet), then the
matte options (finish).

The matte can be post-processed before it is applied: sensitivity, blur,
grow / shrink, invert, an edge refinement of the foreground colours, and a
solid background colour instead of alpha. All of it is plain torch on one
(1, 1, H, W) matte at a time: each frame's matte, options and composite are
made and written into the preallocated outputs before the next frame's, in the
image's dtype when that is half precision (its frames read as float32 levels,
libs/image.requantized), float32 otherwise.
"""

import torch

from ..libs import mask as mask_ops
from ..libs.color import parse_hex_color
from ..libs.filters import gauss_replicate
from ..libs.image import output_dtype, requantized
from ..models.birefnet.inference import matte


def finish(rgb, mattes, sensitivity=1.0, mask_blur=0, mask_offset=0, invert_output=False,
           refine_foreground=False, background="Alpha", background_color="#222222", want_image=True,
           want_mask_image=True):
    """Apply the options to each raw matte and build the three outputs, a frame at a time.
    `rgb` is (B, H, W, 3) in 0..1 on the CPU, `mattes` the raw mattes (H, W) in 0..1 of its frames,
    in order (an iterable; a (B, H, W) tensor will do). `want_image` / `want_mask_image` False: that
    output is built from no frame (0 frames); the mask does not read either."""
    b, h, w = rgb.shape[:3]
    dtype = output_dtype(rgb)
    alpha_out = background == "Alpha"
    if not alpha_out:
        r, g, bl, a = parse_hex_color(background_color)
        bg = torch.tensor([r, g, bl], dtype=torch.float32).view(1, 1, 1, 3)
    image = torch.empty((b if want_image else 0, h, w, 4 if alpha_out else 3), dtype=dtype)
    direct = dtype == torch.float32  # the composite made straight in the output
    mask_out = torch.empty((b, h, w), dtype=dtype)
    mask_image = torch.empty((b if want_mask_image else 0, h, w, 3), dtype=dtype)
    for i, raw in enumerate(mattes):
        m = raw[None, None].float()
        if sensitivity < 1.0:
            m = (m * (1 + (1 - sensitivity))).clamp_(0, 1)
        if mask_blur > 0:
            m = gauss_replicate(m, mask_blur)
        if mask_offset != 0:
            m = mask_ops.offset_matte(m, mask_offset)
        if invert_output:
            m = 1 - m
        mask_out[i] = m[0, 0]
        if want_mask_image:
            mask_image[i] = m[0, 0, :, :, None]  # the matte on all three channels
        if not want_image:
            continue
        color = requantized(rgb[i:i + 1]).float()
        if refine_foreground:
            color = mask_ops.refine_foreground(color, m)
        alpha = m.permute(0, 2, 3, 1)  # (1, H, W, 1)
        if alpha_out:
            image[i, :, :, :3] = color[0]
            image[i, :, :, 3:] = alpha[0]
        else:
            # Straight-alpha "over" onto a background that may itself be
            # translucent, then the alpha is dropped.
            weight = a * (1 - alpha)
            total = alpha + weight
            over = torch.mul(color, alpha, out=image[i:i + 1]) if direct else color * alpha
            over += bg * weight
            over /= total.clamp(min=1e-6)
            over.masked_fill_(~(total > 0), 0.0)
            if not direct:
                image[i] = over[0]
    return image, mask_out, mask_image


def remove_background(image, model, sensitivity, mask_blur, mask_offset, invert_output,
                      refine_foreground, background, background_color, want_image=True, want_mask_image=True):
    """IMAGE (B, H, W, C) and a checkpoint name -> (image, mask, mask_image) as finish builds
    them. An unknown name raises KeyError(name) inside the load, after the cached model is
    evicted."""
    rgb = image[..., :3]
    mask = matte(model, rgb)
    return finish(rgb.cpu(), mask, sensitivity, mask_blur, mask_offset, invert_output,
                  refine_foreground, background, background_color, want_image, want_mask_image)

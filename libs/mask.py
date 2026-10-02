"""Mask operations.

fill_holes and grow_and_blur work on 8-bit quantised (B, H, W) masks and import cv2 and PIL
inside the functions; draw_mask_on_image paints a colour through a mask onto an image;
blockify turns each mask into the blocks of its bounding box that hold any of it;
offset_matte and refine_foreground work on a (B, 1, H, W) matte; fit_mask_frame fits one frame of
a MASK to an image. Frame loops write into one preallocated output, in a half-precision input's
dtype (each frame read through libs/image.float_frame) or float32.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .image import float_frame, is_half, output_dtype, tensor_to_u8

# The 4-neighbourhood: the grow / shrink step and the connectivity of fill_holes' background.
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
# 8-bit level -> float32 in 0..1, the same values as `u8.astype(np.float32) / 255.0`.
U8_TO_FLOAT = np.arange(256, dtype=np.float32) / 255.0


def fill_holes(frames):
    """Background regions the border cannot reach (4-connected) become foreground; hard 0/1.
    `frames` (B, H, W) on the CPU; the output in its dtype when half, float32 otherwise."""
    import cv2

    out = torch.empty(frames.shape, dtype=output_dtype(frames))
    for i in range(frames.shape[0]):
        # Quantised to 8-bit first, so anything below 1/255 is background.
        background = (tensor_to_u8(frames[i]) == 0).view(np.uint8)
        count, labels = cv2.connectedComponents(background, connectivity=4)
        outside = np.zeros(count, dtype=bool)
        for edge in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
            outside[edge] = True
        outside[0] = False  # label 0 is the foreground
        out[i] = torch.from_numpy(~outside[labels])
    return out


def grow_and_blur(frames, invert_mask, grow, blur):
    """Invert (optional), quantise to 8-bit, dilate (grow > 0) or erode (grow < 0) |grow| times
    with CROSS, then PIL Gaussian blur of radius `blur`. `frames` (B, H, W) on the CPU; the output
    (8-bit levels) in its dtype when half, float32 otherwise."""
    import cv2
    from PIL import Image, ImageFilter

    morph = cv2.dilate if grow > 0 else cv2.erode
    out = torch.empty(frames.shape, dtype=output_dtype(frames))
    direct = out.dtype == torch.float32  # each frame's levels written straight into the output
    levels = None if direct else np.empty(frames.shape[1:], dtype=np.float32)
    for i in range(frames.shape[0]):
        frame = float_frame(frames[i])
        u8 = tensor_to_u8(1 - frame if invert_mask else frame, stored=frames.dtype)
        if grow:
            u8 = morph(u8, CROSS, iterations=abs(grow))
        blurred = np.asarray(Image.fromarray(u8).filter(ImageFilter.GaussianBlur(blur)))
        np.take(U8_TO_FLOAT, blurred, out=out[i].numpy() if direct else levels, mode="clip")
        if not direct:
            out[i] = torch.from_numpy(levels)
    return out


def parse_draw_color(text):
    """`color` -> (channels, alpha) in 0..1. "#rgb", "#rgba", "#rrggbb", "#rrggbbaa", or 1, 3 or 4
    comma-separated values, each one above 1 read as 0-255 and any other as 0-1; one value is a
    grey, a 4th value the alpha (the opacity), 1 when absent."""
    text = str(text).strip()
    hint = "use 'r, g, b' or 'r, g, b, a' (0-255 or 0-1), or '#rrggbb' / '#rrggbbaa'"
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) in (3, 4):
            digits = "".join(c * 2 for c in digits)
        if len(digits) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in digits):
            raise ValueError(f"Draw Mask On Image: color {text!r} is not a hex colour; {hint}")
        values = [int(digits[i:i + 2], 16) / 255.0 for i in range(0, len(digits), 2)]
    else:
        try:
            values = [float(v.strip()) for v in text.split(",")]
        except ValueError as e:
            raise ValueError(f"Draw Mask On Image: color {text!r} is not a colour; {hint}") from e
        values = [v / 255.0 if v > 1.0 else v for v in values]
    if len(values) not in (1, 3, 4):
        raise ValueError(f"Draw Mask On Image: color {text!r} has {len(values)} values; {hint}")
    return values[:3] if len(values) == 4 else values, values[3] if len(values) == 4 else 1.0


def draw_mask_on_image(image, mask, color, device):
    """Each frame blended towards `color` by mask x alpha: rgb * (1 - m) + color * m; an RGBA
    frame keeps the larger of its alpha and m. `image` (B, H, W, 3 or 4), `mask` (M, h, w): a
    mask of another size is scaled to the image (nearest-exact), fewer masks than frames repeat.
    Runs on `device` in float32 (or wider; a half frame or mask read through float_frame); returns on the
    CPU, in the image's dtype when that is half precision."""
    channels_rgb, alpha = parse_draw_color(color)
    batch, height, width, channels = image.shape
    if channels not in (3, 4):
        raise ValueError(f"Draw Mask On Image: the image has {channels} channels; it needs 3 (RGB) or 4 (RGBA)")
    fill = torch.tensor(channels_rgb, dtype=torch.float32, device=device)
    scale = tuple(mask.shape[-2:]) != (height, width)
    dtype = torch.promote_types(torch.promote_types(image.dtype, mask.dtype), torch.float32)  # float32 for half
    out = torch.empty((batch, height, width, channels), dtype=image.dtype if is_half(image) else dtype)
    direct = device.type == "cpu" and out.dtype == dtype  # each frame blended straight into the output
    for i in range(batch):
        m = float_frame(mask[i % mask.shape[0]].to(device))
        if scale:
            m = F.interpolate(m[None, None], size=(height, width), mode="nearest-exact")[0, 0]
        blend = m.unsqueeze(-1) * alpha
        frame = float_frame(image[i].to(device))
        dst = out[i] if direct else torch.empty(out.shape[1:], dtype=dtype, device=device)
        rgb = dst[..., :3] if channels == 4 else dst
        torch.mul(frame[..., :3] if channels == 4 else frame, 1 - blend, out=rgb)
        rgb += fill * blend
        if channels == 4:
            torch.maximum(frame[..., 3:], blend, out=dst[..., 3:])
        if not direct:
            out[i].copy_(dst)
    return out


def _block_ids(length, block_size, device):
    """Block index of each of `length` pixels: length // block_size blocks (at least one) of
    equal size, the last one taking the remainder; and the block count."""
    blocks = max(1, length // block_size)
    ids = torch.arange(length, device=device) // (length // blocks)
    return ids.clamp_(max=blocks - 1), blocks


def blockify(masks, block_size, device):
    """Per mask: its bounding box cut into blocks of about `block_size` px; a block is 1 when any
    pixel of it is above 0, everything else 0. `masks` (B, H, W); runs on `device`, returns on the
    CPU in the masks' dtype."""
    out = torch.zeros(masks.shape, dtype=masks.dtype)
    for i in range(masks.shape[0]):
        present = masks[i].to(device) > 0
        rows = torch.nonzero(present.any(dim=1))
        if rows.numel() == 0:
            continue
        cols = torch.nonzero(present.any(dim=0))
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        row_ids, row_blocks = _block_ids(y1 - y0, block_size, device)
        col_ids, col_blocks = _block_ids(x1 - x0, block_size, device)
        counts = torch.zeros((row_blocks, x1 - x0), dtype=torch.int32, device=device)
        counts.index_add_(0, row_ids, present[y0:y1, x0:x1].to(torch.int32))
        counts = torch.zeros((row_blocks, col_blocks), dtype=torch.int32, device=device).index_add_(1, col_ids, counts)
        out[i, y0:y1, x0:x1] = (counts > 0)[row_ids[:, None], col_ids[None, :]]
    return out


def offset_matte(mask, steps):
    """Grow (steps > 0) or shrink (steps < 0) a matte by one pixel per step."""
    for _ in range(abs(steps)):
        mask = F.max_pool2d(mask, 3, 1, 1) if steps > 0 else -F.max_pool2d(-mask, 3, 1, 1)
    return mask


def refine_foreground(rgb, mask):
    """Edge refinement: the matte is hardened at 0.45, its transition band is
    blended with a 3x3-blurred copy and slightly darkened, and the colours are
    scaled by that refined matte. `rgb` is (B, H, W, 3), `mask` (B, 1, H, W)."""
    binary = (mask > 0.45).to(mask.dtype)
    k = torch.tensor([0.25, 0.5, 0.25], dtype=mask.dtype, device=mask.device)
    edge = F.pad(binary, (1, 1, 1, 1), mode="replicate")
    edge = F.conv2d(edge, k.view(1, 1, 1, 3))
    edge = F.conv2d(edge, k.view(1, 1, 3, 1))
    refined = torch.where((mask > 0.05) & (mask < 0.95), 0.85 * mask + 0.15 * edge, binary)
    refined = torch.where((mask > 0.2) & (mask < 0.8), refined * 0.98, refined)
    return rgb * refined.permute(0, 2, 3, 1)


def fit_mask_frame(x, i, h, w, b):
    """The mask of frame i of a batch of b images from the MASK `x` (any MASK shape), float32 (h, w):
    a single mask serves every frame, otherwise frame i of it (fewer than b raise); bilinear resize
    when the size differs."""
    m = x.reshape(-1, x.shape[-2], x.shape[-1])
    if 1 < m.shape[0] < b:
        raise ValueError(f"the mask has {m.shape[0]} frames for {b} images: connect one mask, or one per image")
    m = float_frame(m[min(i, m.shape[0] - 1)]).float()
    if m.shape != (h, w):
        m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return m

"""Mask operations.

fill_holes and grow_and_blur work on 8-bit quantised (B, H, W) masks and import scipy and PIL
inside the functions; offset_matte and refine_foreground work on a (B, 1, H, W) matte;
fit_mask_batch fits a MASK to an image batch.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_u8(frame):
    return np.clip(255.0 * frame, 0, 255).astype(np.uint8)


def fill_holes(frames):
    from scipy.ndimage import binary_fill_holes

    out = np.empty(frames.shape, dtype=np.float32)
    for i, frame in enumerate(frames.numpy()):
        # Quantised to 8-bit first, so anything below 1/255 is background.
        out[i] = binary_fill_holes(_to_u8(frame) > 0)
    return torch.from_numpy(out)


def grow_and_blur(frames, invert_mask, grow, blur):
    import scipy.ndimage
    from PIL import Image, ImageFilter

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    out = []
    for frame in frames.numpy():
        if invert_mask:
            frame = 1 - frame
        # 8-bit round trip before the morphology and again before the blur.
        arr = np.asarray(Image.fromarray(_to_u8(frame))).astype(np.float32) / 255.0
        for _ in range(abs(grow)):
            if grow < 0:
                arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
            else:
                arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
        blurred = Image.fromarray(_to_u8(arr)).filter(ImageFilter.GaussianBlur(blur))
        out.append(torch.from_numpy(np.asarray(blurred).astype(np.float32) / 255.0))
    return torch.stack(out, dim=0)


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


def fit_mask_batch(x, h, w, b):
    """Any MASK shape -> float (b, h, w): bilinear resize when the size differs, then the
    first b masks; fewer than b are expanded, which works for one mask and raises otherwise."""
    m = x.float().reshape(-1, x.shape[-2], x.shape[-1])
    if m.shape[-2:] != (h, w):
        m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
    return m[:b] if m.shape[0] >= b else m.expand(b, -1, -1)

"""Depth Anything V2 Small inference, preprocessing as the authors' image2tensor: each frame
resized with its aspect kept so the short side is `resolution` and both sides are multiples of
14 (Resize lower_bound, ensure_multiple_of=14), bicubic, ImageNet normalised; the prediction is
resized back bilinearly (align_corners=True, as infer_image) and min-max normalised per frame."""

import torch
import torch.nn.functional as F

from .loader import load

PATCH = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def round_resolution(resolution):
    """The nearest multiple of 14 to `resolution`, at least 14."""
    return max(PATCH, round(resolution / PATCH) * PATCH)


def _constrain(x, min_val):
    y = round(x / PATCH) * PATCH
    if y < min_val:
        y = -(-x // PATCH) * PATCH
    return int(y)


def net_size(h, w, resolution):
    """(height, width) of the net input for an h x w frame: the authors' Resize with
    keep_aspect_ratio, resize_method="lower_bound", ensure_multiple_of=14 and `resolution` as
    both target sides, so the short side is `resolution` and the long side keeps the aspect."""
    scale = max(resolution / h, resolution / w)
    return _constrain(scale * h, resolution), _constrain(scale * w, resolution)


def normalize(depth):
    """(B, H, W) relative inverse depth -> 0..1 per frame, nearest = 1; a flat frame -> 0."""
    lo = depth.amin(dim=(1, 2), keepdim=True)
    span = depth.amax(dim=(1, 2), keepdim=True) - lo
    return torch.where(span > 0, (depth - lo) / span.clamp_min(1e-12), torch.zeros_like(depth))


def estimate(rgb, resolution):
    """`rgb` is (B, H, W, 3) in 0..1 on any device; returns the depth map (B, H, W) float32 in
    0..1 on the CPU, near = 1 (white), far = 0 (black)."""
    net, device = load()
    size = round_resolution(resolution)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    frames = []
    with torch.no_grad():
        for img in rgb:
            x = img[..., :3].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
            h, w = x.shape[-2:]
            x = F.interpolate(x, size=net_size(h, w, size), mode="bicubic", align_corners=False)
            x = (x - mean) / std
            pred = net(x).float()
            pred = F.interpolate(pred[:, None], size=(h, w), mode="bilinear", align_corners=True)
            frames.append(pred[0, 0].cpu())

    return normalize(torch.stack(frames, dim=0))

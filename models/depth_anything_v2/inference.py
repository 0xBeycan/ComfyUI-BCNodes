"""Depth Anything V2 Small inference, preprocessing as the authors' image2tensor: each frame
resized with its aspect kept so the short side is `resolution` and both sides are multiples of
14 (Resize lower_bound, ensure_multiple_of=14), bicubic, ImageNet normalised; the prediction
(relative inverse depth) is resized bilinearly (align_corners=True, as infer_image) to the size
the caller asks for."""

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


def predictor():
    """The DEPTH entry: loads the model and returns predict(frame, resolution, size) (see
    models/common/registry.py)."""
    net, device = load()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def predict(frame, resolution, size):
        x = frame[..., :3].permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=net_size(h, w, round_resolution(resolution)), mode="bicubic", align_corners=False)
        x = (x - mean) / std
        pred = net(x).float()
        return F.interpolate(pred[:, None], size=size, mode="bilinear", align_corners=True)[0, 0]

    return predict

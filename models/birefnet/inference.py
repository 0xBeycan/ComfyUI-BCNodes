"""BiRefNet inference: each frame resized to the checkpoint's square input, ImageNet
normalised, run under torch.no_grad(), and the matte resized back."""

import torch
import torch.nn.functional as F

from ...libs.image import float_frame
from ..common import registry
from ..common.registry import MATTING
from .loader import load

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def matte(model, rgb):
    """`rgb` is (B, H, W, 3) in 0..1 on any device (a half frame read through libs/image.float_frame);
    yields each frame's matte (H, W) float32 in 0..1 on the CPU, in order."""
    net, device, dtype = load(model)
    res = registry.get(MATTING, model).res
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    for img in rgb:
        x = float_frame(img).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        x = ((x - mean) / std).to(dtype)
        pred = net(x).float().sigmoid()
        pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
        yield pred[0, 0].cpu()

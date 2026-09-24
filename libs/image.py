"""Image conversions and fitting shared by the nodes. PIL is imported inside the functions."""

import numpy as np
import torch


def tensor_to_pil_u8(frame):
    """float (H, W) or (H, W, C) tensor in 0..1 -> 8-bit PIL image (clipped, truncated)."""
    from PIL import Image

    return Image.fromarray(np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8))


def pil_to_tensor_hwc(pil):
    """PIL image -> float32 (H, W) or (H, W, C) tensor in 0..1."""
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def fit_image(image, target_width, target_height, fit, sampler, background):
    """letterbox: whole image inside the target, `background` around it.
    crop: centre crop to the target ratio, then resize. fill: plain resize."""
    from PIL import Image

    orig_width, orig_height = image.size
    if fit == "letterbox":
        if orig_width / orig_height > target_width / target_height:
            fit_width = target_width
            fit_height = int(target_width / orig_width * orig_height)
        else:
            fit_height = target_height
            fit_width = int(target_height / orig_height * orig_width)
        resized = image.resize((fit_width, fit_height), sampler)
        out = Image.new(image.mode, (target_width, target_height), color=background)
        out.paste(resized, box=((target_width - fit_width) // 2, (target_height - fit_height) // 2))
        return out
    if fit == "crop":
        if orig_width / orig_height > target_width / target_height:
            fit_width = int(orig_height * target_width / target_height)
            left = (orig_width - fit_width) // 2
            image = image.crop((left, 0, left + fit_width, orig_height))
        else:
            fit_height = int(orig_width * target_height / target_width)
            top = (orig_height - fit_height) // 2
            image = image.crop((0, top, orig_width, top + fit_height))
    return image.resize((target_width, target_height), sampler)

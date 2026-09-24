"""Image scale node.

    BC_ImageScaleByAspectRatio (Image Scale By Aspect Ratio)

Five outputs, with integer-truncation size arithmetic and round-up to the
multiple. The MASK output is never None — with no usable mask it is a zero
mask of the output size — and the two failure cases raise instead of
returning `(None, None, None, 0, 0)`. PIL is imported inside the
function.
"""

import numpy as np
import torch

from ..libs.geometry import round_up_to_multiple, target_size
from ..libs.image import fit_image, pil_to_tensor_hwc

RATIOS = ["original", "custom", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"]
FITS = ["letterbox", "crop", "fill"]
METHODS = ["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"]
MULTIPLES = ["8", "16", "32", "64", "128", "256", "512", "None"]
SIDES = ["None", "longest", "shortest", "width", "height", "total_pixel(kilo pixel)"]

# ComfyUI hands a (1, 64, 64) zero mask to nodes whose mask input carries no
# real mask; a zero frame of exactly that shape is treated as "no mask".
PLACEHOLDER_MASK_SHAPE = (64, 64)


def _to_pil(frame):
    """float (H, W) or (H, W, C) in 0..1 -> 8-bit PIL image (truncated, as the
    original does)."""
    from PIL import Image

    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    return Image.fromarray(np.clip(255.0 * frame.float().cpu().numpy(), 0, 255).astype(np.uint8))


class ImageScaleByAspectRatio:
    """Scale an image and / or mask to an aspect ratio and a side length."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (RATIOS,),
                "proportional_width": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "proportional_height": ("INT", {"default": 1, "min": 1, "max": 1e8, "step": 1}),
                "fit": (FITS,),
                "method": (METHODS,),
                "round_to_multiple": (MULTIPLES,),
                "scale_to_side": (SIDES,),
                "scale_to_length": ("INT", {"default": 1024, "min": 4, "max": 1e8, "step": 1}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height")
    FUNCTION = "scale"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "image scale by aspect ratio", "scale image", "resize image", "aspect ratio", "letterbox", "crop"]

    def scale(self, aspect_ratio, proportional_width, proportional_height, fit, method, round_to_multiple,
              scale_to_side, scale_to_length, background_color, image=None, mask=None):
        from PIL import Image

        frames = []
        if isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[0] > 0:
            frames = [f for f in image]
        mask_frames = []
        if isinstance(mask, torch.Tensor) and mask.ndim >= 2 and mask.numel() > 0:
            for m in mask.reshape(-1, mask.shape[-2], mask.shape[-1]):
                if tuple(m.shape) == PLACEHOLDER_MASK_SHAPE and not bool(torch.any(m != 0)):
                    continue
                mask_frames.append(m)

        orig_width = orig_height = 0
        if frames:
            orig_height, orig_width = frames[0].shape[0], frames[0].shape[1]
        if mask_frames:
            mask_height, mask_width = mask_frames[0].shape
            if frames and (orig_width != mask_width or orig_height != mask_height):
                raise ValueError(
                    f"BC_ImageScaleByAspectRatio: mask is {mask_width}x{mask_height} but image is "
                    f"{orig_width}x{orig_height}; they must match"
                )
            if not frames:
                orig_width, orig_height = mask_width, mask_height
        if orig_width + orig_height == 0:
            raise ValueError("BC_ImageScaleByAspectRatio: connect an image or a mask")

        if aspect_ratio == "original":
            ratio = orig_width / orig_height
        elif aspect_ratio == "custom":
            ratio = proportional_width / proportional_height
        else:
            a, b = aspect_ratio.split(":")
            ratio = int(a) / int(b)

        target_width, target_height = target_size(orig_width, orig_height, ratio, scale_to_side, scale_to_length)
        if round_to_multiple != "None":
            multiple = int(round_to_multiple)
            target_width = round_up_to_multiple(target_width, multiple)
            target_height = round_up_to_multiple(target_height, multiple)

        sampler = {
            "bicubic": Image.Resampling.BICUBIC,
            "hamming": Image.Resampling.HAMMING,
            "bilinear": Image.Resampling.BILINEAR,
            "box": Image.Resampling.BOX,
            "nearest": Image.Resampling.NEAREST,
        }.get(method, Image.Resampling.LANCZOS)

        out_images = None
        if frames:
            out_images = torch.stack([
                pil_to_tensor_hwc(fit_image(_to_pil(f).convert("RGB"), target_width, target_height, fit, sampler, background_color))
                for f in frames
            ])
        if mask_frames:
            out_masks = torch.stack([
                pil_to_tensor_hwc(fit_image(_to_pil(m).convert("L"), target_width, target_height, fit, sampler, "black"))
                for m in mask_frames
            ])
        else:
            out_masks = torch.zeros((len(frames), target_height, target_width), dtype=torch.float32)

        return (out_images, out_masks, [orig_width, orig_height], target_width, target_height)


NODE_CLASS_MAPPINGS = {
    "BC_ImageScaleByAspectRatio": ImageScaleByAspectRatio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ImageScaleByAspectRatio": "Image Scale By Aspect Ratio",
}

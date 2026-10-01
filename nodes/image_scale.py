"""Image scale nodes.

    BC_ImageScaleByAspectRatio (Image Scale By Aspect Ratio)
    BC_ImageResize             (Image Resize)

Image Scale By Aspect Ratio: five outputs, with integer-truncation size arithmetic and round-up
to the multiple. The MASK output is never None — with no usable mask it is a zero mask of the
output size — and the two failure cases raise instead of returning `(None, None, None, 0, 0)`.
PIL is imported inside the function.

Image Resize: stretch / keep proportion / pad / crop to a width and height, frame by frame into
one preallocated output (libs/resize.py).
"""

import numpy as np
import torch

from ..libs.geometry import aspect_ratio as ratio_of, round_up_to_multiple, target_size
from ..libs.image import fit_image, pil_to_tensor_hwc
from ..libs.resize import resize_image
from .common import DEVICES, LINK_INPUTS, compute_device, drop_unwanted, heavy_wanted, wants

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
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "original_size", "width", "height")
    # not computed when nothing links them (nodes/common.py): the other outputs do not read them
    HEAVY_OUTPUTS = ("image", "mask")
    FUNCTION = "scale"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "image scale by aspect ratio", "scale image", "resize image", "aspect ratio", "letterbox", "crop"]

    def scale(self, aspect_ratio, proportional_width, proportional_height, fit, method, round_to_multiple,
              scale_to_side, scale_to_length, background_color, image=None, mask=None, prompt_graph=None, unique_id=None):
        from PIL import Image

        wanted = heavy_wanted(type(self), prompt_graph, unique_id)

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

        ratio = ratio_of(aspect_ratio, orig_width, orig_height, proportional_width, proportional_height)
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
        if frames and not wants(wanted, "image"):
            out_images = torch.empty((0, target_height, target_width, 3), dtype=torch.float32)  # not fitted
        elif frames:
            out_images = torch.stack([
                pil_to_tensor_hwc(fit_image(_to_pil(f).convert("RGB"), target_width, target_height, fit, sampler, background_color))
                for f in frames
            ])
        if not wants(wanted, "mask"):
            out_masks = torch.empty((0, target_height, target_width), dtype=torch.float32)  # not fitted
        elif mask_frames:
            out_masks = torch.stack([
                pil_to_tensor_hwc(fit_image(_to_pil(m).convert("L"), target_width, target_height, fit, sampler, "black"))
                for m in mask_frames
            ])
        else:
            out_masks = torch.zeros((len(frames), target_height, target_width), dtype=torch.float32)

        return drop_unwanted(type(self), (out_images, out_masks, [orig_width, orig_height], target_width, target_height), wanted)


# ComfyUI core's nodes.MAX_RESOLUTION.
MAX_RESOLUTION = 16384
UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "nvidia_rtx_vsr"]
KEEP_PROPORTIONS = ["stretch", "resize", "pad", "pad_edge", "pad_edge_pixel", "crop", "pillarbox_blur", "total_pixels"]
CROP_POSITIONS = ["center", "top", "bottom", "left", "right"]


class ImageResize:
    """Resize an image batch (and a mask) to a width and height: stretch, keep the proportion, pad or crop."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                  "tooltip": "0 = the source width (or free, when the proportion is kept)."}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                   "tooltip": "0 = the source height (or free, when the proportion is kept)."}),
                "upscale_method": (UPSCALE_METHODS, {"tooltip": "lanczos runs on the CPU (PIL, 8-bit); nvidia_rtx_vsr "
                                                                "needs the nvidia-vfx package and an NVIDIA RTX GPU and "
                                                                "rounds the size to a multiple of 8."}),
                "keep_proportion": (KEEP_PROPORTIONS, {"default": "stretch",
                                                       "tooltip": "stretch: exactly width x height. resize: the largest size "
                                                                  "inside width x height at the source aspect. pad / pad_edge / "
                                                                  "pad_edge_pixel / pillarbox_blur: resize, then pad out to "
                                                                  "width x height with pad_color / the edge mean / the edge "
                                                                  "pixels / a blurred cover of the frame. crop: crop to the "
                                                                  "target aspect, then resize. total_pixels: width x height "
                                                                  "pixels at the source aspect."}),
                "pad_color": ("STRING", {"default": "0, 0, 0",
                                         "tooltip": "Padding colour for pad: 'r, g, b' in 0-255 or 0-1, #rrggbb, a colour name "
                                                    "or one grey value."}),
                "crop_position": (CROP_POSITIONS, {"default": "center",
                                                   "tooltip": "Where the crop window sits (crop), or where the image sits "
                                                              "inside the padding (pad modes)."}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 512, "step": 1,
                                         "tooltip": "Floor both sides to a multiple of this; padding grows to the next "
                                                    "multiple instead. 0 or 1 = off."}),
            },
            "optional": {
                "mask": ("MASK",),
                "device": (DEVICES, {"tooltip": "Device the frames are processed on (not for lanczos)."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "MASK")
    RETURN_NAMES = ("IMAGE", "width", "height", "mask")
    # not computed when nothing links them (nodes/common.py): the sizes do not read them
    HEAVY_OUTPUTS = ("IMAGE", "mask")
    FUNCTION = "resize"
    CATEGORY = "BCNodes/image"
    DESCRIPTION = ("Resizes the image (and mask) to width x height. The mask follows the image; without a mask, a "
                   "padded image returns the padding as the mask (1 = padding). An image already at the output size "
                   "is passed through untouched.")
    SEARCH_ALIASES = ["BCNodes", "image resize", "resize image", "scale image", "pad image", "crop image", "letterbox",
                      "pillarbox"]

    def resize(self, image, width, height, upscale_method, keep_proportion, pad_color, crop_position, divisible_by,
               mask=None, device="cpu", prompt_graph=None, unique_id=None):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Image Resize: connect an image batch (B, H, W, C)")
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), resize_image(
            image, mask, width, height, upscale_method, keep_proportion, pad_color, crop_position, divisible_by,
            compute_device(device), want_image=wants(wanted, "IMAGE"), want_mask=wants(wanted, "mask")), wanted)


NODE_CLASS_MAPPINGS = {
    "BC_ImageScaleByAspectRatio": ImageScaleByAspectRatio,
    "BC_ImageResize": ImageResize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ImageScaleByAspectRatio": "Image Scale By Aspect Ratio",
    "BC_ImageResize": "Image Resize",
}

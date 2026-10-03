"""Mask nodes.

    BC_MaskFillHoles    (Mask Fill Holes)
    BC_MaskGrow         (MaskGrow)
    BC_DrawMaskOnImage  (Draw Mask On Image)
    BC_BlockifyMask     (Blockify Mask)
    BC_RepeatMaskBatch  (Repeat Mask Batch)

Fill Holes and Grow work on 8-bit quantised masks and return (B, H, W). Every output keeps a half-
precision input's dtype (float16 / bfloat16, read a frame at a time as float32 levels: libs/image.py)
and is float32 otherwise. A missing or empty mask returns a blank (1, 64, 64) mask instead of raising
(Draw Mask On Image returns the image unchanged). The mask operations are in libs/mask.py, which
imports cv2 and PIL inside the functions that use them.
"""

import torch

from .common import DEVICES, compute_device

EMPTY_MASK_SHAPE = (1, 64, 64)


def _empty_mask():
    return torch.zeros(EMPTY_MASK_SHAPE, dtype=torch.float32)


def _frames(mask):
    """Any of (H, W) / (B, H, W) / (B, 1, H, W) -> (B, H, W) on the CPU, half precision kept and any
    other dtype as float32, or None when there is nothing to process."""
    from ..libs.image import is_half

    if not isinstance(mask, torch.Tensor) or mask.ndim < 2 or mask.numel() == 0:
        return None
    mask = mask.detach().cpu()
    if not is_half(mask):
        mask = mask.float()
    return mask.reshape(-1, mask.shape[-2], mask.shape[-1])


class MaskFillHoles:
    """Fill the enclosed holes of every mask in the batch. Output is hard 0/1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"optional": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)
    FUNCTION = "fill_region"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ['BCNodes', 'mask fill holes', 'fill holes']

    def fill_region(self, masks=None):
        from ..libs.mask import fill_holes

        frames = _frames(masks)
        if frames is None:
            return (_empty_mask(),)

        return (fill_holes(frames),)


class MaskGrow:
    """Grow (dilate) or shrink (erode) a mask by `grow` pixels with a cross
    kernel, then Gaussian-blur it by `blur`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "invert_mask": ("BOOLEAN", {"default": False}),
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_grow"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ['BCNodes', 'mask grow', 'grow mask', 'shrink mask', 'erode', 'dilate']

    def mask_grow(self, invert_mask, grow, blur, mask=None):
        from ..libs.mask import grow_and_blur

        frames = _frames(mask)
        if frames is None:
            return (_empty_mask(),)

        return (grow_and_blur(frames, invert_mask, grow, blur),)


class DrawMaskOnImage:
    """Paint a colour over the image where the mask is set, blended by mask x alpha."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "color": ("STRING", {"default": "0, 0, 0",
                                     "tooltip": "RGB or RGBA, comma-separated, each value 0-255 or 0-1 (a value above 1 is read "
                                                "as 0-255), or #rgb / #rgba / #rrggbb / #rrggbbaa. Alpha is the opacity, "
                                                "1 when absent. Ex: 255, 0, 0, 128"}),
            },
            "optional": {
                "device": (DEVICES, {"default": "cpu", "tooltip": "Device the frames are blended on."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "BCNodes/mask"
    DESCRIPTION = ("Blends each frame towards `color` by mask x alpha; an RGBA image keeps the larger of its alpha and "
                   "the mask. A mask of another size is scaled to the image (nearest), fewer masks than frames repeat.")
    SEARCH_ALIASES = ["BCNodes", "draw mask on image", "mask overlay", "fill mask", "paint mask", "composite mask"]

    def apply(self, image, mask, color, device="cpu"):
        from ..libs.mask import draw_mask_on_image

        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("Draw Mask On Image: connect an image batch (B, H, W, C)")
        frames = _frames(mask)
        if frames is None or image.shape[0] == 0:
            return (image,)
        return (draw_mask_on_image(image, frames, color, compute_device(device)),)


class BlockifyMask:
    """Each mask as the blocks of its bounding box that hold any of it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 512, "step": 1,
                                       "tooltip": "Block size in pixels (smaller = smaller blocks)."}),
            },
            "optional": {
                "device": (DEVICES, {"default": "cpu", "tooltip": "Device the masks are processed on."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "BCNodes/mask"
    DESCRIPTION = ("Divides the bounding box of each mask into blocks of about block_size pixels (the last row and "
                   "column take the remainder) and fills every block that holds any part of the mask.")
    SEARCH_ALIASES = ["BCNodes", "blockify mask", "block mask", "pixelate mask", "grid mask"]

    def process(self, masks, block_size, device="cpu"):
        from ..libs.mask import blockify

        frames = _frames(masks)
        if frames is None:
            return (_empty_mask(),)
        return (blockify(frames, block_size, compute_device(device)),)


class RepeatMaskBatch:
    """The whole mask batch repeated `amount` times, as Repeat Image Batch does for images."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "repeat"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ["BCNodes", "repeat mask batch", "duplicate mask", "repeat masks", "clone mask"]

    def repeat(self, mask, amount):
        frames = _frames(mask)
        if frames is None:
            return (_empty_mask(),)
        return (frames if amount == 1 else frames.repeat((amount, 1, 1)),)


NODE_CLASS_MAPPINGS = {
    "BC_MaskFillHoles": MaskFillHoles,
    "BC_MaskGrow": MaskGrow,
    "BC_DrawMaskOnImage": DrawMaskOnImage,
    "BC_BlockifyMask": BlockifyMask,
    "BC_RepeatMaskBatch": RepeatMaskBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_MaskFillHoles": "Mask Fill Holes",
    "BC_MaskGrow": "MaskGrow",
    "BC_DrawMaskOnImage": "Draw Mask On Image",
    "BC_BlockifyMask": "Blockify Mask",
    "BC_RepeatMaskBatch": "Repeat Mask Batch",
}

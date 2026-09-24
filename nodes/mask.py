"""Mask nodes.

    BC_MaskFillHoles (Mask Fill Holes)
    BC_MaskGrow      (MaskGrow)

Both work on 8-bit quantised masks and return float32 (B, H, W). A missing or
empty mask returns a blank (1, 64, 64) mask instead of raising. The mask
operations are in libs/mask.py, which imports scipy and PIL inside the
functions that use them.
"""

import torch

from ..libs.mask import fill_holes, grow_and_blur

EMPTY_MASK_SHAPE = (1, 64, 64)


def _empty_mask():
    return torch.zeros(EMPTY_MASK_SHAPE, dtype=torch.float32)


def _frames(mask):
    """Any of (H, W) / (B, H, W) / (B, 1, H, W) -> float32 (B, H, W) on the CPU,
    or None when there is nothing to process."""
    if not isinstance(mask, torch.Tensor) or mask.ndim < 2 or mask.numel() == 0:
        return None
    mask = mask.detach().cpu().float()
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
        frames = _frames(mask)
        if frames is None:
            return (_empty_mask(),)

        return (grow_and_blur(frames, invert_mask, grow, blur),)


NODE_CLASS_MAPPINGS = {
    "BC_MaskFillHoles": MaskFillHoles,
    "BC_MaskGrow": MaskGrow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_MaskFillHoles": "Mask Fill Holes",
    "BC_MaskGrow": "MaskGrow",
}

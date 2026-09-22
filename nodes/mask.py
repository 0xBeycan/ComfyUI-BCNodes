"""Mask nodes.

    BC_MaskFillHoles (Mask Fill Holes)
    BC_MaskGrow      (MaskGrow)

Both work on 8-bit quantised masks and return float32 (B, H, W). A missing or
empty mask returns a blank (1, 64, 64) mask instead of raising. scipy and PIL
are imported inside the functions that use them.
"""

import numpy as np
import torch

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


def _to_u8(frame):
    return np.clip(255.0 * frame, 0, 255).astype(np.uint8)


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

        from scipy.ndimage import binary_fill_holes

        out = np.empty(frames.shape, dtype=np.float32)
        for i, frame in enumerate(frames.numpy()):
            # Quantised to 8-bit first, so anything below 1/255 is background.
            out[i] = binary_fill_holes(_to_u8(frame) > 0)
        return (torch.from_numpy(out),)


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
        return (torch.stack(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "BC_MaskFillHoles": MaskFillHoles,
    "BC_MaskGrow": MaskGrow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_MaskFillHoles": "Mask Fill Holes",
    "BC_MaskGrow": "MaskGrow",
}

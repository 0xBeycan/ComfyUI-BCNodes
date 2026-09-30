"""Depth Anything V2 depth estimation.

    BC_DepthAnythingV2 (Depth Anything V2)
"""

import torch

from ..models.depth_anything_v2.inference import estimate


class DepthAnythingV2:
    """IMAGE in -> a grayscale depth map, near = white, far = black (the ControlNet depth
    convention), at the input size. Depth-Anything-V2-Small (Apache-2.0); the weights are
    fetched from Hugging Face into models/depthanything on first use and kept loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 518, "min": 14, "max": 2044, "step": 14,
                                       "tooltip": "Short side the model sees, rounded to a multiple of 14. "
                                                  "518 is what the model was trained at; higher gives finer detail and costs more memory."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "estimate_depth"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ['BCNodes', 'depth anything', 'depth map', 'depth', 'controlnet depth']

    def estimate_depth(self, image, resolution=518):
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)),)
        depth = estimate(image, resolution)
        return (depth.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous(),)


NODE_CLASS_MAPPINGS = {
    "BC_DepthAnythingV2": DepthAnythingV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DepthAnythingV2": "Depth Anything V2",
}

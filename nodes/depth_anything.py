"""Depth Anything depth estimation (V2 Small, and the Apache-2.0 Depth Anything 3 models).

    BC_DepthAnythingV2 (Depth Anything)
"""

import torch


class DepthAnythingV2:
    """IMAGE in -> a grayscale depth map, near = white, far = black (the ControlNet depth
    convention), normalised per frame. Short side `resolution` (the size a ControlNet
    preprocessor gives), or exactly width x height when both are connected. v2-small is
    Depth-Anything-V2-Small, the v3 models are ComfyUI core's Depth Anything 3; the weights are
    fetched from Hugging Face on first use and kept loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..models.common import registry
        from ..models.common.registry import DEPTH

        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 518, "min": 14, "max": 2044, "step": 14,
                                       "tooltip": "Short side the model sees, rounded to a multiple of 14, and the short side "
                                                  "of the depth map when width / height are not connected. Higher gives finer "
                                                  "detail and costs more memory."}),
            },
            "optional": {
                "model": (registry.names(DEPTH), {"default": "v2-small",
                                                  "tooltip": "v2-small: Depth Anything V2 Small, this pack's own code. "
                                                             "v3-*: Depth Anything 3 through ComfyUI core, weights in "
                                                             "models/geometry_estimation (small 137 MB, base 542 MB, "
                                                             "mono-large and metric-large 1.3 GB). All Apache-2.0."}),
                "width": ("INT", {"forceInput": True, "min": 1, "max": 16384,
                                  "tooltip": "With height: the depth map comes out at exactly width x height (covering it "
                                             "with its aspect kept, centre-cropped), e.g. the latent's pixel size."}),
                "height": ("INT", {"forceInput": True, "min": 1, "max": 16384,
                                   "tooltip": "With width: the depth map comes out at exactly width x height."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "estimate_depth"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ['BCNodes', 'depth anything', 'depth anything 3', 'da3', 'depth map', 'depth', 'controlnet depth']

    def estimate_depth(self, image, resolution=518, model="v2-small", width=None, height=None):
        from ..pipelines.depth_anything import estimate

        if (width is None) != (height is None):
            raise ValueError("Depth Anything: connect both width and height, or neither "
                             "(neither gives a depth map with the short side `resolution`).")
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)),)
        size = None if width is None else (int(height), int(width))
        return (estimate(image, model, resolution, size),)


NODE_CLASS_MAPPINGS = {
    "BC_DepthAnythingV2": DepthAnythingV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DepthAnythingV2": "Depth Anything",
}

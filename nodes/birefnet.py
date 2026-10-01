"""BiRefNet background removal.

    BC_BiRefNetRemoveBackground (BiRefNet Remove Background)
"""

import torch

from ..models.common import registry
from ..models.common.registry import MATTING
from ..pipelines import matting
from .common import LINK_INPUTS, drop_unwanted, heavy_wanted, wants


class BiRefNetRemoveBackground:
    """IMAGE in -> image (RGBA with alpha = matte, or RGB over a solid colour),
    MASK and the mask as an RGB image. The model is fetched from Hugging Face
    on first use and kept loaded between runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (registry.names(MATTING), {"default": "BiRefNet-general"}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Below 1 the matte is amplified, so faint areas count as foreground."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Gaussian blur of the matte edges, in pixels."}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1,
                                        "tooltip": "Grow (positive) or shrink (negative) the matte, one pixel per step."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Keep the background instead of the subject."}),
                "refine_foreground": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Harden the matte edge and scale the colours by it, for cleaner cut-outs on transparent output."}),
                "background": (["Alpha", "Color"], {"default": "Alpha",
                                                    "tooltip": "Alpha: RGBA output with the matte as alpha. Color: RGB output over background_color."}),
                "background_color": ("COLOR", {"default": "#222222", "tooltip": "Used when background is Color."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    # nodes/common.py: IMAGE and MASK_IMAGE are not built when nothing links them; MASK, the matte
    # both are built from, is dropped at return
    HEAVY_OUTPUTS = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "remove_background"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ['BCNodes', 'birefnet', 'remove background', 'rmbg', 'matting']

    def remove_background(self, image, model, sensitivity=1.0, mask_blur=0, mask_offset=0, invert_output=False,
                          refine_foreground=False, background="Alpha", background_color="#222222", prompt_graph=None,
                          unique_id=None):
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 4)), torch.zeros((0, 64, 64)), torch.zeros((0, 64, 64, 3)))
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), matting.remove_background(
            image, model, sensitivity, mask_blur, mask_offset, invert_output, refine_foreground, background, background_color,
            want_image=wants(wanted, "IMAGE"), want_mask_image=wants(wanted, "MASK_IMAGE")), wanted)


NODE_CLASS_MAPPINGS = {
    "BC_BiRefNetRemoveBackground": BiRefNetRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_BiRefNetRemoveBackground": "BiRefNet Remove Background",
}

"""Skin Texture — micro-texture on skin, masked by SAM 3.

    BC_SkinTexture (Skin Texture)
"""

import torch

from ..models.sam3.checkpoint import DEFAULT_SAM3, choices
from ..pipelines import skin_texture


class SkinTexture:
    """Micro-texture inside a SAM 3 skin mask (or a connected mask)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam3_model": (choices(), {"default": DEFAULT_SAM3,
                                          "tooltip": "SAM 3 checkpoint under models/checkpoints; the default is downloaded when missing. "
                                                     "Unused when a mask is connected."}),
                "texture": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Synthetic pore field strength. 0 = off; 1 = ±6% luminance modulation."}),
                "detail": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "Gain on the image's own fine detail inside the mask (0 = off)."}),
                "pore_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.05,
                                         "tooltip": "Pore size multiplier. 1 = ~1 px pores at a 1024 px long edge, scaled with the image."}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 64,
                                    "tooltip": "Mask edge softness in px (at a 1024 px long edge)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "SAM 3 detection threshold."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Where to texture. Connected, it replaces the SAM 3 skin detection."}),
                "exclude_mask": ("MASK", {"tooltip": "Subtracted from the skin mask (added to the eyes / lips exclusion)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "run"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "skin texture", "pores", "micro texture", "skin detail", "sam3"]
    DESCRIPTION = ("Adds micro-texture to skin: boosts the image's own fine detail and multiplies in a synthetic pore "
                   "field, in linear light, only inside a SAM 3 skin mask (eyes, eyebrows, lips and teeth excluded). "
                   "Put it before upscaling and before grain. Connect a mask to skip the detection.")

    def run(self, image, sam3_model, texture, detail, pore_scale, feather, seed, threshold, mask=None, exclude_mask=None):
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        return skin_texture.run(image, sam3_model, texture, detail, pore_scale, feather, seed, threshold, mask, exclude_mask)


NODE_CLASS_MAPPINGS = {
    "BC_SkinTexture": SkinTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_SkinTexture": "Skin Texture",
}

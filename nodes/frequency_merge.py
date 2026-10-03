"""Frequency Merge — the structure and colour of one image with the fine detail of another.

    BC_FrequencyMerge (Frequency Merge)

The merge is libs/frequency.py: a Gaussian low-pass of `base` plus the high-pass of `detail`.
"""

import torch

from .common import DEVICES, compute_device


class FrequencyMerge:
    """`base`'s low frequencies + `detail`'s high frequencies."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("IMAGE", {"tooltip": "Gives the low frequencies: shapes, shading, colour (e.g. the SeedVR2 output "
                                              "made at downscale_factor 1, whose face is right)."}),
                "detail": ("IMAGE", {"tooltip": "Gives the high frequencies: skin texture, fine detail (e.g. the SeedVR2 output "
                                                "made at a lower downscale_factor). Same size and image count as base."}),
                "split_sigma": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 64.0, "step": 0.1,
                                          "tooltip": "Sigma in pixels of the Gaussian that splits low from high: a pattern that repeats "
                                                     "every 5.3 x sigma px is split half and half, finer comes from detail, coarser "
                                                     "from base. Raise it to take more of detail; keep 5.3 x sigma well below the size "
                                                     "of what must not change (eyes, mouth)."}),
                "detail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                              "tooltip": "Gain on detail's high frequencies. 1 = as in detail; 0 = none (the blurred "
                                                         "base); above 1 sharper than detail."}),
            },
            "optional": {
                "device": (DEVICES, {"default": "cpu", "tooltip": "Device the images are filtered on."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_TOOLTIPS = ("base's low frequencies + detail's high frequencies, clamped to [0, 1], RGB; float16 only when both "
                       "inputs are float16.",)
    FUNCTION = "merge"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "frequency merge", "frequency separation", "high pass", "low pass", "detail transfer",
                      "texture transfer", "seedvr2"]
    DESCRIPTION = ("Takes the structure and colour of `base` (a Gaussian low-pass) and the fine detail of `detail` (what "
                   "the same Gaussian removes from it), image by image. For two upscales of the same image, e.g. SeedVR2 "
                   "at two downscale factors: the face of one, the skin texture of the other.")

    def merge(self, base, detail, split_sigma, detail_strength, device="cpu"):
        from ..libs.frequency import frequency_merge

        if not isinstance(base, torch.Tensor) or base.ndim != 4 or not isinstance(detail, torch.Tensor) or detail.ndim != 4:
            raise ValueError("Frequency Merge: connect two image batches (B, H, W, C) to base and detail")
        return (frequency_merge(base, detail, split_sigma, detail_strength, compute_device(device)),)


NODE_CLASS_MAPPINGS = {
    "BC_FrequencyMerge": FrequencyMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_FrequencyMerge": "Frequency Merge",
}

"""Image Quality Gate — blur / sharpness / noise / clipping / entropy verdict.

    BC_ImageQualityGate (Image Quality Gate)

Single-node quality control for AI-generated images, built for LoRA dataset
filtering: five metrics, each scored against a threshold scaled by a shot-type
preset, folded into a three-tier verdict (PASS / SO-SO / FAIL) with a badge
image and a text report. The metrics are in libs/image_metrics.py, the verdict,
report and badge in pipelines/quality_gate.py; cv2 and PIL are imported there,
inside the functions that use them.
"""

from ..libs.image import pil_to_tensor_hwc
from ..pipelines import quality_gate


class ImageQualityGate:
    """Analyzes Blur, Sharpness, Noise, Clipping and Entropy; PASS (green),
    SO-SO (yellow) or FAIL (red)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shot_type": (["custom", "close-up", "medium", "wide / full-body"],),
                "blur_mode": (["full image", "center-weighted"],),
                "blur_threshold": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Max blur ratio. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "blur_var_threshold": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 5.0,
                        "max": 500.0,
                        "step": 5.0,
                        "tooltip": "Laplacian variance per block. AI-generated: 20-50, photos: 80-150.",
                    },
                ),
                "sharpness_threshold": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 0.0,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": "Min sharpness. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "noise_threshold": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Max noise level. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "clipping_threshold": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Max clipping ratio. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "entropy_threshold": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Min entropy in bits. Presets scale this; 'custom' uses as-is.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("badge", "verdict", "report", "blur_score", "sharpness_score", "noise_score", "clipping_score", "entropy_score")
    FUNCTION = "analyze"
    CATEGORY = "BCNodes/analysis"
    SEARCH_ALIASES = ["BCNodes", "image quality", "blur", "sharpness", "dataset filter"]

    def analyze(self, image, shot_type, blur_mode, blur_threshold, blur_var_threshold,
                sharpness_threshold, noise_threshold, clipping_threshold, entropy_threshold):
        img, verdict_int, report, blur_score, sharpness_score, noise_score, clipping_score, entropy_score = quality_gate.analyze(
            image, shot_type, blur_mode, blur_threshold, blur_var_threshold,
            sharpness_threshold, noise_threshold, clipping_threshold, entropy_threshold)
        return (pil_to_tensor_hwc(img).unsqueeze(0), verdict_int, report, blur_score, sharpness_score,
                noise_score, clipping_score, entropy_score)


NODE_CLASS_MAPPINGS = {
    "BC_ImageQualityGate": ImageQualityGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ImageQualityGate": "Image Quality Gate",
}

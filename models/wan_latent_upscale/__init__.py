"""The Wan 2.1 latent upscaler (spacepxl, MIT): a small convolution network that doubles a Wan 2.1 latent
(Wan 2.1, Wan 2.2 A14B, Qwen-Image) on height and width, for a second sampling pass. The vendored
architecture (arch/), the model cache of the weights file the node picks (loader.py) and the inference
(inference.py), registered as "Wan 2.1 2x" in the latent_upscale family. Everything beyond torch (comfy,
safetensors) is imported on the first run.
"""

from ..common.registry import LATENT_UPSCALE, register
from .inference import upscale

register(LATENT_UPSCALE, "Wan 2.1 2x", upscale)

"""The Wan 2.1 VAE, registered as "Wan 2.1" in the VAE family: every VAE file of the Wan 2.1 latent
space (Wan 2.1, Wan 2.2 A14B, Qwen-Image), loaded by core; a decoder that packs a larger image into
its channels decodes at that size (loader.py). comfy is imported on the first load.
"""

from ..common.registry import VAE, register
from .loader import load

register(VAE, "Wan 2.1", load)

"""BiRefNet: the vendored architecture (arch/, MIT), the checkpoint table registered under
the matting family (checkpoints.py), the weights (weights.py), the model cache (loader.py) and
the inference (inference.py).

The architecture lives in ./arch and the weights are loaded
straight from safetensors into a plain nn.Module — no transformers, no
trust_remote_code, no timm. Weights come from the model authors' Hugging Face
repos through the pack's own downloader. Everything beyond torch (safetensors,
folder_paths, comfy.model_management) is imported on the first run, never at
import time.
"""

from . import checkpoints  # noqa: F401  (registers the checkpoints under MATTING)

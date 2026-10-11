"""Where the BiRefNet weights live (ComfyUI's background_removal folder) and their download
from the checkpoint's Hugging Face repo."""

from ..common import registry
from ..common.registry import MATTING

FOLDER_KEY = "background_removal"


def weights_path(name):
    from ..common.download import model_file

    ckpt = registry.get(MATTING, name)
    return model_file(FOLDER_KEY, f"{name}.safetensors", f"https://huggingface.co/{ckpt.repo}/resolve/main/{ckpt.file}")

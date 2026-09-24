"""Where the BiRefNet weights live (ComfyUI's background_removal folder) and their download
from the checkpoint's Hugging Face repo."""

import os

from ..common import registry
from ..common.registry import MATTING

FOLDER_KEY = "background_removal"


def model_dir():
    import folder_paths

    path = os.path.join(folder_paths.models_dir, FOLDER_KEY)
    # ComfyUI registers this key itself (2026-05, nodes_bg_removal); on older
    # builds the call adds it, on current ones it is a no-op.
    folder_paths.add_model_folder_path(FOLDER_KEY, path)
    return path


def weights_path(name):
    import folder_paths

    from ..common.download import fetch_with_progress

    primary = model_dir()
    filename = f"{name}.safetensors"
    found = folder_paths.get_full_path(FOLDER_KEY, filename)
    if found is not None:
        return found

    ckpt = registry.get(MATTING, name)
    url = f"https://huggingface.co/{ckpt.repo}/resolve/main/{ckpt.file}"
    path = os.path.join(primary, filename)
    return fetch_with_progress(url, path, filename)

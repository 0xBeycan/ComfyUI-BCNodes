"""Where the BiRefNet weights live (ComfyUI's background_removal folder) and their download
from beycanai/BiRefNet, which holds every checkpoint as <name>.safetensors."""

FOLDER_KEY = "background_removal"
REPO = "beycanai/BiRefNet"


def weights_path(name):
    from ..common.download import model_file

    filename = f"{name}.safetensors"
    return model_file(FOLDER_KEY, filename, f"https://huggingface.co/{REPO}/resolve/main/{filename}")

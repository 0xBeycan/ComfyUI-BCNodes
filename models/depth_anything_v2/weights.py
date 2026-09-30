"""Where the Depth Anything V2 Small weights live (ComfyUI's models/depthanything folder) and
their download from the authors' Hugging Face repo."""

import os

FOLDER_KEY = "depthanything"
REPO = "depth-anything/Depth-Anything-V2-Small"
FILENAME = "depth_anything_v2_vits.pth"


def model_dir():
    import folder_paths

    path = os.path.join(folder_paths.models_dir, FOLDER_KEY)
    folder_paths.add_model_folder_path(FOLDER_KEY, path)
    return path


def weights_path():
    import folder_paths

    from ..common.download import fetch_with_progress

    primary = model_dir()
    found = folder_paths.get_full_path(FOLDER_KEY, FILENAME)
    if found is not None:
        return found

    url = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"
    path = os.path.join(primary, FILENAME)
    return fetch_with_progress(url, path, FILENAME)

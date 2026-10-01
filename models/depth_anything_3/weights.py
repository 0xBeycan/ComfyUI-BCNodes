"""Where the Depth Anything 3 weights live and their download.

The files are ComfyUI's own repackage of the authors' checkpoints (Comfy-Org/Depth-Anything-3, the
layout core's loader reads), kept in core's `geometry_estimation` folder, so core's Load Depth
Anything 3 node and this pack share them. Only the four Apache-2.0 models are offered (DA3-Large,
DA3-Giant and DA3Nested are CC-BY-NC)."""

import os

FOLDER_KEY = "geometry_estimation"
REPO = "Comfy-Org/Depth-Anything-3"
FILES = {
    "v3-small": "depth_anything_3_small.safetensors",
    "v3-base": "depth_anything_3_base.safetensors",
    "v3-mono-large": "depth_anything_3_mono_large.safetensors",
    "v3-metric-large": "depth_anything_3_metric_large.safetensors",
}


def weights_path(name):
    """The file of model `name`, fetched into models/geometry_estimation when no folder of that
    key holds it."""
    import folder_paths

    from ..common.download import fetch_with_progress

    filename = FILES[name]
    primary = os.path.join(folder_paths.models_dir, FOLDER_KEY)
    folder_paths.add_model_folder_path(FOLDER_KEY, primary)
    found = folder_paths.get_full_path(FOLDER_KEY, filename)
    if found is not None:
        return found

    url = f"https://huggingface.co/{REPO}/resolve/main/{FOLDER_KEY}/{filename}"
    return fetch_with_progress(url, os.path.join(primary, filename), filename)

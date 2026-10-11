"""Where the Depth Anything 3 weights live and their download.

The files are ComfyUI's own repackage of the authors' checkpoints (Comfy-Org/Depth-Anything-3, the
layout core's loader reads), kept in core's `geometry_estimation` folder, so core's Load Depth
Anything 3 node and this pack share them. Only the four Apache-2.0 models are offered (DA3-Large,
DA3-Giant and DA3Nested are CC-BY-NC)."""

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
    from ..common.download import model_file

    filename = FILES[name]
    return model_file(FOLDER_KEY, filename, f"https://huggingface.co/{REPO}/resolve/main/{FOLDER_KEY}/{filename}")

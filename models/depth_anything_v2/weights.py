"""Where the Depth Anything V2 Small weights live (core's geometry_estimation folder, where the
Depth Anything 3 weights live too) and their download from the authors' Hugging Face repo."""

FOLDER_KEY = "geometry_estimation"
REPO = "depth-anything/Depth-Anything-V2-Small"
FILENAME = "depth_anything_v2_vits.pth"


def weights_path():
    from ..common.download import model_file

    return model_file(FOLDER_KEY, FILENAME, f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}")

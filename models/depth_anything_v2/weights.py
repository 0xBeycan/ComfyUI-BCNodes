"""Where the Depth Anything V2 Small weights live (core's geometry_estimation folder, where the
Depth Anything 3 weights live too) and their download from beycanai/Depth-Anything, an unchanged
copy of the authors' file (depth-anything/Depth-Anything-V2-Small)."""

FOLDER_KEY = "geometry_estimation"
REPO = "beycanai/Depth-Anything"
FILENAME = "depth_anything_v2_vits.pth"


def weights_path():
    from ..common.download import model_file

    return model_file(FOLDER_KEY, FILENAME, f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}")

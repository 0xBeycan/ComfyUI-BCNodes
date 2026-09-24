"""The SAM 3 checkpoint: the default file and its URL, the combo list, and the path of a chosen
file (the default is downloaded into models/checkpoints when missing). folder_paths is
imported inside the functions."""

import os

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
DEFAULT_SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"


def choices():
    import folder_paths

    names = list(folder_paths.get_filename_list("checkpoints"))
    if DEFAULT_SAM3 not in names:
        names.insert(0, DEFAULT_SAM3)
    return names


def path(name):
    import folder_paths

    found = folder_paths.get_full_path("checkpoints", name)
    if found is not None:
        return found
    if name != DEFAULT_SAM3:
        raise FileNotFoundError(f"SAM 3 checkpoint {name} is not in models/checkpoints")

    from ..common.download import fetch_with_progress

    target = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
    return fetch_with_progress(DEFAULT_SAM3_URL, target, name)

"""The SAM 3 checkpoint: the default file and its URL, the combo list, and the path of a chosen
file (the default is downloaded into models/checkpoints when missing). folder_paths is
imported inside the functions."""

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

    from ..common.download import model_file

    if name == DEFAULT_SAM3:
        return model_file("checkpoints", name, DEFAULT_SAM3_URL)
    found = folder_paths.get_full_path("checkpoints", name)
    if found is None:
        raise FileNotFoundError(f"SAM 3 checkpoint {name} is not in models/checkpoints")
    return found

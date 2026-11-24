"""The Wan 2.1 latent upscaler's model cache: one weights file loaded at a time (2.6M parameters, float32),
kept between runs, keyed by its path."""

import os

import torch

from ...libs.tensor_census import module_bytes


class _Loaded:
    path = None
    model = None
    device = None


def unload():
    """Drops the cached model (it loads again on its node's next run): {its file name: bytes of its
    weights}, {} when none was held."""
    held = {os.path.basename(_Loaded.path): module_bytes(_Loaded.model)} if _Loaded.model is not None else {}
    _Loaded.path, _Loaded.model, _Loaded.device = None, None, None
    return held


def load(path):
    """The model with the weights file at path; ValueError naming the file when its weights do not fit."""
    if _Loaded.path == path and _Loaded.model is not None:
        return _Loaded.model, _Loaded.device

    import comfy.model_management as mm
    from safetensors.torch import load_file

    from .arch import LatentModel3d

    if unload():
        mm.soft_empty_cache()

    model = LatentModel3d()
    try:
        model.load_state_dict(load_file(path), strict=True)
    except RuntimeError as e:  # missing / unexpected keys or other shapes
        raise ValueError(f"{os.path.basename(path)} is not a Wan 2.1 2x latent upscaler (its weights do not fit the "
                         f"model): pick its model, or another file.") from e
    model.eval()

    device = mm.get_torch_device()
    model.to(device=device, dtype=torch.float32)

    _Loaded.path, _Loaded.model, _Loaded.device = path, model, device
    return model, device

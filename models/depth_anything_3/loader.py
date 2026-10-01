"""The Depth Anything 3 model cache: one model at a time, keyed by its name, loaded by ComfyUI
core's loader as its Load Depth Anything 3 node does with weight_dtype "default"
(comfy.sd.load_diffusion_model on the geometry_estimation file). The ModelPatcher it returns is
moved between devices by ComfyUI's model management."""

from .weights import weights_path


class _Loaded:
    name = None
    patcher = None


def load(name):
    if _Loaded.name == name and _Loaded.patcher is not None:
        return _Loaded.patcher

    import comfy.model_management as mm
    import comfy.sd

    if _Loaded.patcher is not None:
        _Loaded.patcher = None
        _Loaded.name = None
        mm.soft_empty_cache()

    path = weights_path(name)
    patcher = comfy.sd.load_diffusion_model(path)
    _Loaded.name, _Loaded.patcher = name, patcher
    return patcher

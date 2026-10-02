"""The SAM 3 model cache: one checkpoint (model and clip) held at a time. comfy.sd is
imported on the first load."""

from ...libs.tensor_census import module_bytes
from .checkpoint import path


class _Sam3:
    name = None
    model = None
    clip = None


def unload():
    """Drops the cached checkpoint, model and text encoder (it loads again on its node's next run):
    {its name: bytes of their weights}, {} when none was held."""
    held = {}
    if _Sam3.model is not None:
        held[_Sam3.name] = module_bytes(_Sam3.model.model, getattr(_Sam3.clip, "cond_stage_model", None))
    _Sam3.name, _Sam3.model, _Sam3.clip = None, None, None
    return held


def load(name):
    if _Sam3.name == name and _Sam3.model is not None:
        return _Sam3.model, _Sam3.clip
    import comfy.sd

    unload()
    loaded = comfy.sd.load_checkpoint_guess_config(path(name), output_vae=False, output_clip=True)
    _Sam3.name, _Sam3.model, _Sam3.clip = name, loaded[0], loaded[1]
    return _Sam3.model, _Sam3.clip

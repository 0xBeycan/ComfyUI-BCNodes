"""The SAM 3 model cache: one checkpoint (model and clip) held at a time. comfy.sd is
imported on the first load."""

from .checkpoint import path


class _Sam3:
    name = None
    model = None
    clip = None


def unload():
    """Drops the cached checkpoint (it loads again on its node's next run); returns its name, or
    None when none was held."""
    name = _Sam3.name if _Sam3.model is not None else None
    _Sam3.name, _Sam3.model, _Sam3.clip = None, None, None
    return name


def load(name):
    if _Sam3.name == name and _Sam3.model is not None:
        return _Sam3.model, _Sam3.clip
    import comfy.sd

    unload()
    loaded = comfy.sd.load_checkpoint_guess_config(path(name), output_vae=False, output_clip=True)
    _Sam3.name, _Sam3.model, _Sam3.clip = name, loaded[0], loaded[1]
    return _Sam3.model, _Sam3.clip

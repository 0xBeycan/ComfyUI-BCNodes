"""The BiRefNet model cache: one model loaded at a time, keyed by the checkpoint name."""

import torch

from ..common import registry
from ..common.registry import MATTING
from .weights import weights_path


class _Loaded:
    name = None
    model = None
    device = None
    dtype = None


def unload():
    """Drops the cached model (it loads again on its node's next run); returns its name, or None
    when none was held."""
    name = _Loaded.name if _Loaded.model is not None else None
    _Loaded.name, _Loaded.model, _Loaded.device, _Loaded.dtype = None, None, None, None
    return name


def load(name):
    if _Loaded.name == name and _Loaded.model is not None:
        return _Loaded.model, _Loaded.device, _Loaded.dtype

    import comfy.model_management as mm
    from safetensors.torch import load_file

    from .arch import BiRefNet

    if unload() is not None:
        mm.soft_empty_cache()

    ckpt = registry.get(MATTING, name)
    path = weights_path(name)
    model = BiRefNet(ckpt.backbone)
    model.load_state_dict(load_file(path), strict=True)
    model.eval()

    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)

    _Loaded.name, _Loaded.model, _Loaded.device, _Loaded.dtype = name, model, device, dtype
    return model, device, dtype

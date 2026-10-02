"""The Depth Anything V2 Small model cache: loaded once, kept between runs."""

import torch

from .weights import weights_path


class _Loaded:
    model = None
    device = None


def unload():
    """Drops the cached model (it loads again on its node's next run); returns its name, or None
    when none was held."""
    held = "Depth Anything V2 Small" if _Loaded.model is not None else None
    _Loaded.model, _Loaded.device = None, None
    return held


def load():
    if _Loaded.model is not None:
        return _Loaded.model, _Loaded.device

    import comfy.model_management as mm

    from .arch import DepthAnythingV2

    path = weights_path()
    model = DepthAnythingV2()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()

    device = mm.get_torch_device()
    model.to(device=device, dtype=torch.float32)

    _Loaded.model, _Loaded.device = model, device
    return model, device

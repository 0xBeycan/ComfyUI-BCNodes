"""Depth Anything 3 over ComfyUI core's implementation: the weights (weights.py), the model
cache (loader.py) and the per-frame inference (inference.py), registered in the depth family as
v3-small, v3-base, v3-mono-large and v3-metric-large, the four Apache-2.0 models. Nothing of DA3
is vendored: the architecture, the loader and the preprocessing are core's, imported on the
first run (comfy.sd, comfy.model_management, comfy.ldm.depth_anything_3, folder_paths).
"""

from functools import partial

from ..common.registry import DEPTH, register
from .inference import predictor
from .weights import FILES

for _name in FILES:
    register(DEPTH, _name, partial(predictor, _name))

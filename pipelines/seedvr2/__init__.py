"""SeedVR2 streaming flows, one module per node: resize, encode, decode, postprocess;
progress is their progress bar and log. The constants and the IMAGE-batch check they
share live here.

The encode, decode and post-process nodes follow ComfyUI's own
comfy/ldm/seedvr/vae.py and comfy_extras/nodes_seedvr.py (GPL-3.0) step for
step; what they change is where the frames live, not what is computed. Encode
and decode unload every other model first when the free VRAM is below the
VAE's working set (ComfyUI will not evict one dynamic model for another).
"""

import torch

FRAMES_PER_CHUNK = 4
# Slices between allocator-pool trims in the VAE slice loops: fragmentation built up over
# ~100 slices in the run that showed it; a trim costs a free/re-alloc of the working set.
TRIM_EVERY = 16


def require_image_batch(x, message):
    """ValueError(message) unless `x` is a non-empty 4-D tensor (an IMAGE batch)."""
    if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] == 0:
        raise ValueError(message)

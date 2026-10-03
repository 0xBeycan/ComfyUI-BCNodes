"""SeedVR2 adapters over ComfyUI core's SeedVR2 VAE: the VAE check, its auto tile and its
VRAM room (vae.py), the spatial tile plan and blend weights (tiling.py), the frame-shape rules
(frames.py); and the DiT's VRAM law, what a temporal chunk of the sampler needs (dit.py).

The encode, decode and post-process nodes follow ComfyUI's own
comfy/ldm/seedvr/vae.py and comfy_extras/nodes_seedvr.py (GPL-3.0) step for
step; what they change is where the frames live, not what is computed. Encode
and decode unload every other model first when the free VRAM is below the
VAE's working set (ComfyUI will not evict one dynamic model for another).
"""

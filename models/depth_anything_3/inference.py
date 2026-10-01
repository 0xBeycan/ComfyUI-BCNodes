"""Depth Anything 3 inference over ComfyUI core's model (comfy/ldm/depth_anything_3): core's
loader, core's preprocess_image and the core model's forward, mono mode, one frame at a time.

Core's Run Depth Anything 3 node is not called: it resizes depth, confidence and sky of the whole
batch back to the input size, where this pack resizes the depth once, straight to the output
size. Per frame, as core's node does:
  - preprocess_image with method "lower_bound_resize", so `resolution` is the short side the model
    sees (rounded to a multiple of 14), as for V2;
  - the forward in the model's own dtype;
  - the Mono / Metric models' sky (probability >= 0.3) set to the 99th percentile of the other
    depths, core's apply_sky_aware_clip, at the model's resolution: what the authors' forward does
    for every model with a sky head;
  - the depth resized bilinearly (align_corners=False, as core's node) and inverted: DA3 predicts
    depth (far = larger), the depth family returns inverse depth (near = larger), as V2. The
    metric model's scale cancels in the per-frame normalisation that follows;
  - the inverse depth clipped to its 2nd..98th percentiles (np.percentile's linear interpolation,
    from exact order statistics), as the authors' visualize_depth does before normalising
    (ByteDance-Seed/Depth-Anything-3 3d835ec, src/depth_anything_3/utils/visualize.py:49-66).
    The small and base models put a few pixels at a near-zero depth, whose inverse would
    otherwise take a quarter of the range and dim the rest of the map.
"""

import torch
import torch.nn.functional as F

from .loader import load

CLIP_PERCENTILES = (2.0, 98.0)


def _percentile(flat, q):
    """np.percentile(flat, q) (linear interpolation between the order statistics around
    q / 100 * (n - 1)), exact and deterministic."""
    pos = q / 100 * (flat.numel() - 1)
    lo = int(pos)
    below = flat.kthvalue(lo + 1).values
    if pos == lo:
        return below
    return below + (flat.kthvalue(lo + 2).values - below) * (pos - lo)


def predictor(name):
    """The DEPTH entry of model `name` (with the name bound): loads the model onto the compute
    device and returns predict(frame, resolution, size) (see models/common/registry.py)."""
    import comfy.model_management as mm

    try:
        from comfy.ldm.depth_anything_3.preprocess import apply_sky_aware_clip, preprocess_image
    except ImportError as e:
        raise RuntimeError(f"Depth Anything {name} runs on ComfyUI core's Depth Anything 3, which this ComfyUI "
                           "does not have: update ComfyUI, or choose v2-small.") from e

    patcher = load(name)
    mm.load_model_gpu(patcher)
    net = patcher.model.diffusion_model
    device = mm.get_torch_device()
    dtype = net.dtype if net.dtype is not None else torch.float32

    @torch.no_grad()
    def predict(frame, resolution, size):
        x = preprocess_image(frame[None, ..., :3].to(device), process_res=resolution, method="lower_bound_resize")
        out = net(x.to(dtype=dtype))
        depth = out["depth"][0].float()
        if "sky" in out:
            depth = apply_sky_aware_clip(depth, out["sky"][0].float())
        inverse = F.interpolate(depth[None, None], size=size, mode="bilinear", align_corners=False)[0, 0].reciprocal()
        flat = inverse.flatten()
        return inverse.clamp(*(_percentile(flat, q) for q in CLIP_PERCENTILES))

    return predict

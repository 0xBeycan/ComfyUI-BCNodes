"""PostProcess — Post-Process SeedVR2 Output builds five full-size float32
copies of the clip on the way through (raw range conversion, the flattening
reshape, the result buffer and the add/div/clamp chain), which is what runs a
30-second 1080p clip out of RAM. This flow does the same operations in the
same order — crop to the reference, `x * 2 - 1`, the colour transfer from
comfy/ldm/seedvr/color_fix.py on the VAE device, `(x + 1) / 2` clamp, alpha
from the reference, even crop — one frame at a time into a single preallocated
float16 output. The colour maths itself stays float32 per frame, exactly as in
the native node, which also processes lab frame by frame.
"""

import torch

from ...models.seedvr2.frames import resize_reference
from . import require_image_batch

METHODS = ["lab", "wavelet", "adain", "none"]


def color_transfer(method, node):
    """The colour transfer of `method` from comfy/ldm/seedvr/color_fix.py, None for "none"; `node`
    names the node in the error for an unknown method."""
    from comfy.ldm.seedvr.color_fix import adain_color_transfer, lab_color_transfer, wavelet_color_transfer

    if method not in METHODS:
        raise ValueError(f"{node}: unknown color_correction_method {method!r}")
    return {"lab": lab_color_transfer, "wavelet": wavelet_color_transfer, "adain": adain_color_transfer}.get(method)


def corrected(decoded, ref, transfer, device):
    """One frame of Post-Process SeedVR2 Output: `decoded` (h, w, 3) colour-transferred from its
    reference frame `ref` (resized to h x w when its size differs) on `device`, float32 in [0, 1]."""
    if ref.shape[0] != decoded.shape[0] or ref.shape[1] != decoded.shape[1]:
        ref = resize_reference(ref, decoded.shape[0], decoded.shape[1])
    decoded_raw = decoded.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
    reference_raw = ref.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
    return transfer(decoded_raw, reference_raw)[0].permute(1, 2, 0).add(1.0).div(2.0).clamp(0.0, 1.0)


def process(images, original_resized_images, color_correction_method):
    import comfy.model_management as mm

    transfer = color_transfer(color_correction_method, "BC_SeedVR2PostProcess")
    for name, x in (("images", images), ("original_resized_images", original_resized_images)):
        require_image_batch(x, f"BC_SeedVR2PostProcess: {name} must be an image batch (B, H, W, C)")

    # The same alignment as Post-Process SeedVR2 Output on 4-D inputs: alpha from the
    # reference, common frame count, crop to the common size, even output size.
    alpha = original_resized_images[..., 3:4] if original_resized_images.shape[-1] == 4 else None
    reference = original_resized_images[..., :3]
    t = min(images.shape[0], reference.shape[0])
    target_h = min(images.shape[1], reference.shape[1])
    target_w = min(images.shape[2], reference.shape[2])
    out_h, out_w = target_h - target_h % 2, target_w - target_w % 2
    out = torch.empty((t, out_h, out_w, 3 + (1 if alpha is not None else 0)), dtype=torch.float16)
    device = mm.vae_device()
    from comfy.utils import ProgressBar

    progress = ProgressBar(t)
    for i in range(t):
        decoded = images[i, :target_h, :target_w, :3]
        frame = decoded if transfer is None else corrected(decoded, reference[i], transfer, device)
        out[i, :, :, :3] = frame[:out_h, :out_w].to(device="cpu", dtype=torch.float16)
        if alpha is not None:
            out[i, :, :, 3] = alpha[i, :out_h, :out_w, 0].to(torch.float16)
        progress.update(1)
    return (out,)

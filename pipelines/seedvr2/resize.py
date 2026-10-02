"""Resize — from the original image to the frame the VAE encodes:

  1. downscale by `downscale_factor`, the way ComfyUI's ImageScaleBy(lanczos)
     does it: `round(side * factor)`, PIL LANCZOS on 8-bit (comfy/utils.py
     `lanczos`); 1.0 skips it;
  2. `resolution = min(width, height) * upscale_factor` from the ORIGINAL
     image;
  3. shortest-edge resize plus clamp:
     `TVF.resize` to `resolution` on the shortest edge (torchvision floors the
     long edge), BICUBIC with antialias, a second resize to
     `round(edge * max_resolution / longest)` when the longest edge exceeds
     `max_resolution`, then clamp to [0, 1];
  4. zero-pad bottom / right to a multiple of 16 and, for a frame batch,
     repeat the last frame up to a 4n+1 frame count (the same arithmetic as
     comfy_extras/nodes_seedvr.py:75-90), so `image` goes straight to
     VAE Encode.

Step 3 runs twice: on the VAE device in bfloat16 for the frame that gets
encoded, and on the CPU in float32 for the colour-correction reference, which
is then cropped to the even output size. `image` is the first (padded),
`reference` the second, with the original frame count — the post-process node drops the
repeated frames against it. Steps 1 to 3 run a few frames at a time, so a long
video batch never has to fit on the GPU at once and the downscaled clip never
exists whole: only the two outputs are clip-sized. Both outputs are stored as
float16: VAE Encode casts `image` to the VAE's float16 anyway (comfy/sd.py
process_input), and `reference` only feeds the colour transfer, where a
float16 rounding of the reference moves the result by ~0.1/255. PIL is
imported inside lanczos_scale_by, torchvision inside models/seedvr2/frames.py.

A float16 IMAGE (8-bit frames stored as float16) is requantized a chunk at a
time to the float32 k/255 it stands for before step 1: float16(k/255) is not
float32(k/255), and the bfloat16 cast, the lanczos `255 * x` and the reference
resize would round it differently, so the same 8-bit clip would give other
frames as float16 than as float32. A float32 input goes through untouched.
"""

import logging

import torch

from ...libs.image import pil_to_tensor_hwc, tensor_to_pil_u8
from ...models.seedvr2.frames import divisible_pad, frames_to_4n1, side_resize
from . import FRAMES_PER_CHUNK, require_image_batch


def _torch_device():
    import comfy.model_management

    return comfy.model_management.get_torch_device()


def lanczos_scale_by(image_bhwc, factor):
    """ImageScaleBy(lanczos, factor): comfy.utils.lanczos on 8-bit PIL frames, each written into one
    batch of the input's dtype and device."""
    from PIL import Image

    height, width = image_bhwc.shape[1], image_bhwc.shape[2]
    size = (round(width * factor), round(height * factor))
    out = torch.empty((image_bhwc.shape[0], size[1], size[0], image_bhwc.shape[3]), dtype=image_bhwc.dtype, device=image_bhwc.device)
    for i, frame in enumerate(image_bhwc):
        out[i] = pil_to_tensor_hwc(tensor_to_pil_u8(frame).resize(size, resample=Image.Resampling.LANCZOS))
    return out


def _downscaled(frames, downscale_factor):
    """Step 1 on (B, H, W, C) frames, as (B, C, H, W); float16 frames requantized to k/255 first (module doc)."""
    if frames.dtype == torch.float16:
        frames = frames.float().mul_(255).round_().div_(255)
    if downscale_factor != 1.0:
        frames = lanczos_scale_by(frames, downscale_factor)
    return frames.permute(0, 3, 1, 2)


def _encoded(chunk, device, dtype, resolution, max_resolution):
    """Step 3 on `device` in `dtype`, then the pad of step 4: the frames VAE Encode gets, float16
    (B, H, W, C) on the CPU."""
    resized = divisible_pad(side_resize(chunk.to(device=device, dtype=dtype), resolution, max_resolution))
    return resized.to(device="cpu", dtype=torch.float16).permute(0, 2, 3, 1)


def _reference(chunk, resolution, max_resolution):
    """Step 3 on the CPU in float32, cropped to the even size: the colour-correction reference,
    float16 (B, H, W, C)."""
    reference = side_resize(chunk.to(device="cpu", dtype=torch.float32), resolution, max_resolution)
    h, w = reference.shape[-2:]
    return reference[:, :, :(h // 2) * 2, :(w // 2) * 2].permute(0, 2, 3, 1).to(torch.float16)


def resize(image, upscale_factor, downscale_factor, max_resolution, emulate_bf16, want_image=True, want_reference=True):
    """(image, reference) as the module doc describes. `want_image` / `want_reference` False: that
    output's step runs on no frame, its batch has 0 frames (the other output does not read it)."""
    require_image_batch(image, "BC_SeedVR2Resize: connect an image batch (B, H, W, C)")
    image = image[..., :3]
    resolution = int(round(min(image.shape[1], image.shape[2]) * upscale_factor))

    device = torch.device("cpu")
    vae_dtype = torch.float32
    if emulate_bf16:
        device = _torch_device()
        if device.type == "cuda":
            vae_dtype = torch.bfloat16
        else:
            logging.warning("BC_SeedVR2Resize: emulate_bf16 needs CUDA (device is %s), resizing in float32", device)
            device = torch.device("cpu")

    t = image.shape[0]
    extra = frames_to_4n1(t)
    # an output not wanted is its step on no frame, on the CPU: the empty batch of its size
    empty = _downscaled(image[:0], downscale_factor)
    out_image = None if want_image else _encoded(empty, torch.device("cpu"), torch.float32, resolution, max_resolution)
    out_reference = None if want_reference else _reference(empty, resolution, max_resolution)
    for start in range(0, t, FRAMES_PER_CHUNK):
        chunk = _downscaled(image[start:start + FRAMES_PER_CHUNK], downscale_factor)
        if want_image:
            resized = _encoded(chunk, device, vae_dtype, resolution, max_resolution)
            if out_image is None:
                out_image = torch.empty((t + extra,) + tuple(resized.shape[1:]), dtype=torch.float16)
            out_image[start:start + resized.shape[0]] = resized
        if want_reference:
            reference = _reference(chunk, resolution, max_resolution)
            if out_reference is None:
                out_reference = torch.empty((t,) + tuple(reference.shape[1:]), dtype=torch.float16)
            out_reference[start:start + reference.shape[0]] = reference
    if extra and want_image:
        out_image[t:] = out_image[t - 1]
    return (out_image, out_reference)

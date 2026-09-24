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
repeated frames against it. Frames are processed a few at a time so a long
video batch never has to fit on the GPU at once. Both outputs are stored as
float16: VAE Encode casts `image` to the VAE's float16 anyway (comfy/sd.py
process_input), and `reference` only feeds the colour transfer, where a
float16 rounding of the reference moves the result by ~0.1/255. PIL is
imported inside lanczos_scale_by, torchvision inside models/seedvr2/frames.py.
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
    """ImageScaleBy(lanczos, factor): comfy.utils.lanczos on 8-bit PIL frames."""
    from PIL import Image

    height, width = image_bhwc.shape[1], image_bhwc.shape[2]
    size = (round(width * factor), round(height * factor))
    frames = []
    for frame in image_bhwc:
        pil = tensor_to_pil_u8(frame)
        frames.append(pil_to_tensor_hwc(pil.resize(size, resample=Image.Resampling.LANCZOS)))
    return torch.stack(frames).to(image_bhwc.device, image_bhwc.dtype)


def resize(image, upscale_factor, downscale_factor, max_resolution, emulate_bf16):
    require_image_batch(image, "BC_SeedVR2Resize: connect an image batch (B, H, W, C)")
    image = image[..., :3]
    resolution = int(round(min(image.shape[1], image.shape[2]) * upscale_factor))
    if downscale_factor != 1.0:
        image = lanczos_scale_by(image, downscale_factor)
    frames = image.permute(0, 3, 1, 2)

    device = torch.device("cpu")
    vae_dtype = torch.float32
    if emulate_bf16:
        device = _torch_device()
        if device.type == "cuda":
            vae_dtype = torch.bfloat16
        else:
            logging.warning("BC_SeedVR2Resize: emulate_bf16 needs CUDA (device is %s), resizing in float32", device)
            device = torch.device("cpu")

    t = frames.shape[0]
    extra = frames_to_4n1(t)
    out_image = out_reference = None
    for start in range(0, t, FRAMES_PER_CHUNK):
        chunk = frames[start:start + FRAMES_PER_CHUNK]
        resized = divisible_pad(side_resize(chunk.to(device=device, dtype=vae_dtype), resolution, max_resolution))
        resized = resized.to(device="cpu", dtype=torch.float16).permute(0, 2, 3, 1)
        reference = side_resize(chunk.to(device="cpu", dtype=torch.float32), resolution, max_resolution)
        h, w = reference.shape[-2:]
        reference = reference[:, :, :(h // 2) * 2, :(w // 2) * 2].permute(0, 2, 3, 1).to(torch.float16)
        if out_image is None:
            out_image = torch.empty((t + extra,) + tuple(resized.shape[1:]), dtype=torch.float16)
            out_reference = torch.empty((t,) + tuple(reference.shape[1:]), dtype=torch.float16)
        out_image[start:start + resized.shape[0]] = resized
        out_reference[start:start + reference.shape[0]] = reference
    if extra:
        out_image[t:] = out_image[t - 1]
    return (out_image, out_reference)

"""Dual Sampler's flow: a first pass, the latent upscaled, a second pass with DyPE, as the chain of core nodes it
replaces runs it. Any model: DyPE takes the models of its registered adapters (Z-Image, Qwen-Image); with its mode
at disable the second pass runs model_2 as it is.

- DyPE first: models/common/dype.patch puts it on a clone of model_2 for the second pass (with DyPE Config's
  values), before any sampling, so a model DyPE does not take (or a DyPE Config of another model) stops the run
  before the first pass. Its start sigma is the sigma of the second pass's first model call, read by DyPE in the run:
  the schedule as the run gets it, after a shift the model's wrappers apply per run (Qwen Image Shift's dynamic one,
  from that pass's latent size). Not at denoise 0: no second pass runs.
- First pass: core's KSampler (libs/sampling.ksampler) with model_1 at denoise 1.0.
- Upscale to the first latent's size times scale_by, each side rounded to the nearest multiple of 16 px, at least one
  latent cell (the VAE's px per cell from vae.spacial_compression_encode()). Two ways, mixed by blend_factor (the
  pixel way's weight): the latent way is core's Upscale Latent By (comfy.utils.common_upscale, upscale_method, no
  crop); the pixel way decodes the first latent (core's VAE Decode), resizes the image (libs/resize.resize_image:
  lanczos through PIL on 8 bits as core's, or bicubic) and encodes it (core's VAE Encode). They are mixed as core's
  Latent Blend mixes samples1 (pixel) and samples2 (latent): pixel * blend_factor + latent * (1 - blend_factor). At 0
  the pixel way does not run (the first pass's latent goes only through core's Upscale Latent By); at 1 the latent
  way does not. A 5-D latent (Qwen-Image's, a frame axis before height and width) is upscaled and mixed as it is; its
  pixel way encodes each image of the batch on its own (core's VAE Encode would take the batch for one video's frames).
- Second pass: core's KSampler with model_2 (DyPE's clone) at denoise, the seed itself or the next one.
- Images: core's VAE Decode of each pass's latent, only for the outputs wanted. A pixel way's decode of the first
  latent is the first image too, never decoded twice. With no pixel way both decodes run after the second pass.
"""

from dataclasses import dataclass
from typing import Optional

import torch

PIXEL_MULTIPLE = 16  # px: the upscaled latent's sides (8 px VAE cells x Z-Image's and Qwen-Image's 2x2 patches)
SEED_SPAN = 0xFFFFFFFFFFFFFFFF + 1  # KSampler's seed range: the next seed after the last is 0


@dataclass(frozen=True)
class Pass:
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float


@dataclass(frozen=True)
class Settings:
    seed: int
    first: Pass
    second: Pass
    upscale_method: str         # core's Upscale Latent By methods
    scale_by: float
    blend_factor: float         # the pixel way's weight, 0 to 1
    pixel_upscale_method: str   # lanczos or bicubic
    second_seed: str            # fixed or increment
    dype: str                   # auto, enable or disable (models/common/dype.MODES)
    dype_config: Optional[dict]  # models/common/dype.Config, or None


def upscaled_size(height, width, scale_by, cell):
    """(height, width) in latent cells: each side times scale_by, rounded to the nearest multiple of PIXEL_MULTIPLE
    px (half to even; of one cell for a VAE whose cell is larger), at least one multiple. `cell`: px per latent
    cell."""
    step = max(1, PIXEL_MULTIPLE // cell)
    return tuple(max(step, round(side * scale_by / step) * step) for side in (height, width))


def decode(vae, latent):
    """core's VAE Decode of a LATENT dict: (B, H, W, C), a video VAE's frames joined into the batch."""
    samples = latent["samples"]
    if samples.is_nested:
        samples = samples.unbind()[0]
    images = vae.decode(samples)
    if len(images.shape) == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def undecoded(vae, latent):
    """The 0-frame IMAGE a decode of `latent` would fill: its size, channels and dtype, for an output not wanted."""
    ratio = vae.spacial_compression_decode()
    height, width = latent["samples"].shape[-2:]
    return torch.empty((0, height * ratio, width * ratio, vae.output_channels), dtype=vae.vae_output_dtype())


def pixel_way(vae, image, height, width, method, frames):
    """The first image resized to height x width px and encoded: (B, C, h, w), or (B, C, 1, h, w) for a latent with a
    frame axis (`frames`). There each image is encoded on its own, a one-frame clip, into a preallocated latent: core's
    VAE Encode with a video VAE (Qwen-Image's Wan 2.1 VAE) takes an image batch for the frames of one video."""
    from ..libs.resize import resize_image

    resized = resize_image(image, None, width, height, method, "stretch", "0", "center", 0, torch.device("cpu"),
                           want_mask=False)[0]
    if not frames or resized.shape[0] == 1:
        return vae.encode(resized)
    out = None
    for i in range(resized.shape[0]):
        one = vae.encode(resized[i:i + 1])
        if out is None:
            out = one.new_empty((resized.shape[0], *one.shape[1:]))
        out[i:i + 1] = one
    return out


def run(model_1, model_2, positive, negative, latent, vae, s, want_first=True, want_second=True):
    """-> (first image, second image, the second pass's LATENT dict). An image not wanted is a 0-frame tensor."""
    import comfy.utils

    from ..libs.sampling import ksampler
    from ..models.common import dype

    if s.second.denoise > 0:
        model_2 = dype.patch(model_2, s.dype, None, s.dype_config)  # start sigma: the second pass's first

    first = ksampler(model_1, s.seed, s.first.steps, s.first.cfg, s.first.sampler_name, s.first.scheduler, positive,
                     negative, latent, 1.0)
    cell = vae.spacial_compression_encode()
    height, width = upscaled_size(*first["samples"].shape[-2:], s.scale_by, cell)
    first_image = None
    if s.blend_factor > 0:
        first_image = decode(vae, first)
        pixel = pixel_way(vae, first_image, height * cell, width * cell, s.pixel_upscale_method, first["samples"].ndim == 5)
        expected = (*first["samples"].shape[:-2], height, width)
        if tuple(pixel.shape) != expected:
            raise ValueError(f"Dual Sampler: the VAE encodes the upscaled images to a latent of {tuple(pixel.shape)} "
                             f"(batch, channels, ..., height, width), the model's latent is {expected}. Connect the "
                             "model's own VAE.")
    if s.blend_factor == 0:
        samples = comfy.utils.common_upscale(first["samples"], width, height, s.upscale_method, "disabled")
    elif s.blend_factor == 1:
        samples = pixel.to(first["samples"])
    else:
        latent_way = comfy.utils.common_upscale(first["samples"], width, height, s.upscale_method, "disabled")
        samples = pixel.to(latent_way) * s.blend_factor + latent_way * (1 - s.blend_factor)
    upscaled = first.copy()  # as core's Upscale Latent By: the other fields as they are
    upscaled["samples"] = samples

    seed = s.seed if s.second_seed == "fixed" else (s.seed + 1) % SEED_SPAN
    second = ksampler(model_2, seed, s.second.steps, s.second.cfg, s.second.sampler_name, s.second.scheduler, positive,
                      negative, upscaled, s.second.denoise)

    if first_image is None:
        first_image = decode(vae, first) if want_first else undecoded(vae, first)
    second_image = decode(vae, second) if want_second else undecoded(vae, second)
    return first_image, second_image, second

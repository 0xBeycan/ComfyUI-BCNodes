"""The detailer core: a region of an image cut out with its surroundings, enlarged until the region's
longer side is `size` px, sampled, shrunk back and pasted through the region's own shape.

- The crop: region.plan (sampled at `crop_size` px on its longer side with the region's longer side at
  `size` px inside it, both sampling sides multiples of 64); a region whose longer side is `size` px or
  more is left as it is.
- Sampling: the model with core's Differential Diffusion (a clone), core's inpaint conditioning of the
  crop with the region's mask, softened at its edge, as the noise mask, every ControlNet's hint cut to the
  crop (conditioning.py: those in the conditioning, and core's model-patch ControlNets on a clone of the model
  per region; a ControlNet it cannot cut is an error). One sampling at `denoise`; at 0 no region is detailed
  (run): nothing is encoded, sampled, decoded or pasted, and no region is marked.
- The paste: the decoded crop shrunk back, its colour matched to the image's in the mask's edge band
  (libs/color.match_band), Laplacian-pyramid blended under the mask softened inward (libs/blend), and
  only the pixels inside the mask taken: every other pixel keeps its exact value.

It knows nothing of what a mask shows and nothing of the model: MODEL, VAE and CONDITIONING go through
core's sampler, VAE and ControlNet objects (the sampler run: libs/sampling.ksampler). comfy.* and comfy_extras.*
are imported on first use.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...libs.blend import laplacian_blend
from ...libs.color import match_band
from ...libs.filters import gauss_reflect
from ...libs.image import float_frame, output_dtype
from ...libs.mask import fit_mask_frame
from ...libs.sampling import ksampler
from .conditioning import crop_conditioning, crop_model, inpaint
from .region import bounds, plan

FEATHER = 0.02    # sigma of the mask's soft edge, a fraction of the crop's shorter side (~20 px at 1024)
EDGE_BAND = 0.98  # the colour is matched inside the mask where its blur is below this: within ~2 sigma of the edge


@dataclass
class Settings:
    """What every region of one node run is sampled with, but the model."""
    vae: object
    positive: list
    negative: list
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    size: int        # px the region's longer side is sampled at (region_size, face_size or eye_size)
    crop_size: int   # px the crop's longer side is sampled at, the region at `size` inside it (at least size)


def differential(model):
    """A clone of `model` with core's Differential Diffusion at its node's default strength (1.0)."""
    from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion

    return DifferentialDiffusion.execute(model).args[0]


def working_frame(frame):
    """A float32 copy of one IMAGE frame (H, W, C) to detail in place; a half frame read through
    libs/image.float_frame."""
    out = torch.empty(frame.shape, dtype=torch.float32)
    read = float_frame(frame, out=out)
    return out if read is out else out.copy_(read)


def locate(hard, s):
    """(Crop, the region's pixels inside it (h, w bool)) of the region `hard` (H, W bool); None when it
    is empty or its longer side is s.size px or more."""
    box = bounds(hard)
    crop = None if box is None else plan(box, hard.shape[1], hard.shape[0], s.size, s.crop_size)
    return None if crop is None else (crop, hard[crop.y0:crop.y1, crop.x0:crop.x1])


def _resize(x, height, width):
    """(N, C, h, w) -> (N, C, height, width), bicubic (antialiased when shrinking), clamped to [0, 1]."""
    shrink = height < x.shape[-2] or width < x.shape[-1]
    return F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=shrink).clamp(0, 1)


def _soft(hard):
    """`hard` (h, w bool) blurred: its soft edge (0.5 on the edge), sigma FEATHER of the shorter side."""
    return gauss_reflect(hard[None, None].float(), FEATHER * min(hard.shape))[0, 0]


def _sample(base, region, crop, model, s, image_width, image_height):
    """The crop `base` (3, h, w) of an image_width x image_height image sampled at crop.width x
    crop.height with the region `region` (h, w bool) as its mask, shrunk back to (3, h, w)."""
    pixels = _resize(base[None], crop.height, crop.width).movedim(1, -1)
    enlarged = F.interpolate(region[None, None].float(), size=(crop.height, crop.width), mode="bilinear") > 0.5
    noise_mask = _soft(enlarged[0, 0])[None, None]
    done = {}
    positive = crop_conditioning(s.positive, crop, image_width, image_height, done)
    negative = crop_conditioning(s.negative, crop, image_width, image_height, done)
    model = crop_model(model, crop, image_width, image_height)  # before the encode: a ControlNet it cannot cut raises
    positive, negative, latent = inpaint(positive, negative, s.vae, pixels, noise_mask)
    latent = ksampler(model, s.seed, s.steps, s.cfg, s.sampler_name, s.scheduler, positive, negative,
                      {"samples": latent, "noise_mask": noise_mask}, s.denoise)["samples"]
    images = s.vae.decode(latent)
    image = images.reshape(-1, *images.shape[-3:])[:1]  # a video VAE decodes (B, T, H, W, C)
    return _resize(image[..., :3].movedim(-1, 1).float().cpu(), base.shape[-2], base.shape[-1])[0]


def paste(base, patch, hard):
    """`patch` into `base` (both (3, h, w)) through `hard` (h, w bool): the patch's colour matched to
    base's in the edge band, the two Laplacian-blended under the mask softened inward (0 on its edge),
    and base's exact values everywhere outside the mask."""
    soft = _soft(hard)
    weight = (2 * soft - 1).clamp(0, 1) * hard
    patch = match_band(patch, base, hard & (soft < EDGE_BAND))
    return torch.where(hard, laplacian_blend(base, patch, weight).clamp(0, 1), base)


def detail(frame, located, model, s):
    """Samples the region `located` (from locate) of `frame` (H, W, C float32) with `model` (made by
    differential) and pastes it into `frame` in place."""
    crop, region = located
    window = frame[crop.y0:crop.y1, crop.x0:crop.x1, :3]
    base = window.permute(2, 0, 1)
    patch = _sample(base, region, crop, model, s, frame.shape[1], frame.shape[0])
    window.copy_(paste(base, patch, region).permute(1, 2, 0))


def mark(changed, located):
    """Sets the pixels the region `located` changes in `changed` (H, W)."""
    crop, region = located
    changed[crop.y0:crop.y1, crop.x0:crop.x1][region] = 1


def run(image, masks, model, s, max_size, want_image=True):
    """BC_Detailer: (image with every region detailed, the pixels changed). Each mask of `masks` is one
    region, detailed in batch order on every frame of `image`, each on the frame the previous ones left; a
    region whose longer side is over `max_size` px is left as it is, as is one s.size px or more (locate), and
    every region at s.denoise 0 (it would redraw nothing). Frames one at a time into outputs of the image's
    dtype when that is half precision, float32 otherwise. `want_image` False: nothing is sampled, the image is
    [0, H, W, 3] (the mask does not read it)."""
    b, h, w, c = image.shape
    count = masks.reshape(-1, *masks.shape[-2:]).shape[0]
    regions = []
    for n in range(count if s.denoise > 0 else 0):
        hard = fit_mask_frame(masks, n, h, w, count) > 0.5
        box = bounds(hard)
        located = None if box is None or max(box[2] - box[0], box[3] - box[1]) > max_size else locate(hard, s)
        if located is not None:
            regions.append(located)
    changed = torch.zeros((b, h, w), dtype=output_dtype(image))
    for located in regions:
        mark(changed[0], located)
    changed[1:] = changed[0]
    out = torch.empty((b if want_image else 0, h, w, c), dtype=output_dtype(image))
    if want_image:
        sampler = differential(model) if regions else model
        for i in range(b):
            frame = working_frame(image[i])
            for located in regions:
                detail(frame, located, sampler, s)
            out[i] = frame
    return out, changed

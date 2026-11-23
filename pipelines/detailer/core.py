"""The detailer core: a region of an image cut out with its context, enlarged to the sampling size,
sampled in two passes, shrunk back and pasted through the region's own shape.

- The crop: region.plan (`context` times the region's box, the long side sampled at `resolution`); a
  region whose crop would not be enlarged is left as it is.
- Sampling: the model with core's Differential Diffusion (a clone), core's inpaint conditioning of the
  crop with the region's mask, softened at its edge, as the noise mask, every ControlNet's hint cut to the
  crop (conditioning.py: those in the conditioning, and Z-Image's Fun ControlNet on a clone of the model per
  region). The structure pass samples at `denoise`, the refine pass samples its latent
  again at `refine_denoise` (seed + 1); a pass at 0 does not run.
- The paste: the decoded crop shrunk back, its colour matched to the image's in the mask's edge band
  (libs/color.match_band), Laplacian-pyramid blended under the mask softened inward (libs/blend), and
  only the pixels inside the mask taken: every other pixel keeps its exact value.

It knows nothing of what a mask shows and nothing of the model: MODEL, VAE and CONDITIONING go through
core's sampler, VAE and ControlNet objects. comfy.*, comfy_extras.* and latent_preview are imported on
first use.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...libs.blend import laplacian_blend
from ...libs.color import match_band
from ...libs.filters import gauss_reflect
from ...libs.image import float_frame, output_dtype
from ...libs.mask import fit_mask_frame
from .conditioning import crop_conditioning, crop_model, inpaint
from .region import bounds, plan

FEATHER = 0.02    # sigma of the mask's soft edge, a fraction of the crop's shorter side (~20 px at 1024)
EDGE_BAND = 0.98  # the colour is matched inside the mask where its blur is below this: within ~2 sigma of the edge
SEED_MAX = 0xFFFFFFFFFFFFFFFF


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
    refine_denoise: float
    resolution: int
    context: float


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
    is empty or its crop would not be enlarged for sampling."""
    box = bounds(hard)
    crop = None if box is None else plan(box, hard.shape[1], hard.shape[0], s.context, s.resolution)
    return None if crop is None else (crop, hard[crop.y0:crop.y1, crop.x0:crop.x1])


def _resize(x, height, width):
    """(N, C, h, w) -> (N, C, height, width), bicubic (antialiased when shrinking), clamped to [0, 1]."""
    shrink = height < x.shape[-2] or width < x.shape[-1]
    return F.interpolate(x, size=(height, width), mode="bicubic", align_corners=False, antialias=shrink).clamp(0, 1)


def _soft(hard):
    """`hard` (h, w bool) blurred: its soft edge (0.5 on the edge), sigma FEATHER of the shorter side."""
    return gauss_reflect(hard[None, None].float(), FEATHER * min(hard.shape))[0, 0]


def _ksampler(model, s, positive, negative, latent, noise_mask, denoise, seed):
    """One run of core's common_ksampler (nodes.py, which this pack may not import) on the latent tensor."""
    import comfy.sample
    import comfy.utils
    import latent_preview

    latent = comfy.sample.fix_empty_latent_channels(model, latent)
    noise = comfy.sample.prepare_noise(latent, seed)
    return comfy.sample.sample(model, noise, s.steps, s.cfg, s.sampler_name, s.scheduler, positive, negative, latent,
                               denoise=denoise, noise_mask=noise_mask, callback=latent_preview.prepare_callback(model, s.steps),
                               disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)


def _sample(base, region, crop, model, s, image_width, image_height):
    """The crop `base` (3, h, w) of an image_width x image_height image sampled at crop.width x
    crop.height with the region `region` (h, w bool) as its mask, shrunk back to (3, h, w)."""
    pixels = _resize(base[None], crop.height, crop.width).movedim(1, -1)
    enlarged = F.interpolate(region[None, None].float(), size=(crop.height, crop.width), mode="bilinear") > 0.5
    noise_mask = _soft(enlarged[0, 0])[None, None]
    done = {}
    positive = crop_conditioning(s.positive, crop, image_width, image_height, done)
    negative = crop_conditioning(s.negative, crop, image_width, image_height, done)
    positive, negative, latent = inpaint(positive, negative, s.vae, pixels, noise_mask)
    model = crop_model(model, crop, image_width, image_height)
    for denoise, seed in ((s.denoise, s.seed), (s.refine_denoise, (s.seed + 1) & SEED_MAX)):
        if denoise > 0:
            latent = _ksampler(model, s, positive, negative, latent, noise_mask, denoise, seed)
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


def run(image, masks, model, s, want_image=True):
    """BC_Detailer: (image with every region detailed, the pixels changed). Each mask of `masks` is one
    region, detailed in batch order on every frame of `image`, each on the frame the previous ones left.
    Frames one at a time into outputs of the image's dtype when that is half precision, float32 otherwise.
    `want_image` False: nothing is sampled, the image is [0, H, W, 3] (the mask does not read it)."""
    b, h, w, c = image.shape
    count = masks.reshape(-1, *masks.shape[-2:]).shape[0]
    regions = [r for r in (locate(fit_mask_frame(masks, n, h, w, count) > 0.5, s) for n in range(count)) if r is not None]
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

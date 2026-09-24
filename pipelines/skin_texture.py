"""Rendered skin comes out as a smooth gradient; grain on top of it reads as
noise over plastic. This flow puts a surface under the grain: inside a skin
mask it (a) boosts the image's own high-frequency detail and (b) multiplies
in a synthetic pore field — band-passed noise at pore scale with sparse
darker pits — in linear light, so the texture follows the shading instead
of sitting on it like a sticker. Highlights and deep shadows get less, and
the whole effect fades when the face is small in the frame (pore-scale
detail has nowhere to live there).

The mask comes from SAM 3 (the multiplex checkpoint under models/checkpoints,
downloaded on first use when missing): "skin" minus "eyes / eyebrows / lips
/ teeth". A connected `mask` replaces the skin detection; `exclude_mask` is
subtracted either way. Nothing here moves geometry — moles and marks live
in the low band the flow never touches. Everything beyond torch (SAM 3,
folder_paths, comfy.*) is imported on the first run.
"""

import torch

from ..libs.filters import gauss_reflect
from ..libs.mask import fit_mask_batch
from ..libs.texture import apply_texture
from ..models.sam3.detect import detect
from ..models.sam3.loader import load

SKIN_PROMPT = "skin:6"
EXCLUDE_PROMPT = "eyes:4, eyebrows:4, lips:2, teeth:2"
FACE_PROMPT = "face:2"

FACE_FULL_FRACTION = 0.30  # face height / frame height at which texture is at full strength


def face_gate(face_mask, height):
    """1.0 when the tallest face box spans FACE_FULL_FRACTION of the frame,
    fading linearly to 0 as the face gets smaller; 1.0 when there is no face
    (a body-only frame keeps its texture)."""
    if face_mask is None or not bool(face_mask.any()):
        return 1.0
    rows = face_mask.flatten(0, 0).any(dim=-1).any(dim=0) if face_mask.ndim == 3 else face_mask.any(dim=-1)
    ys = torch.where(rows)[0]
    face_h = float(ys.max() - ys.min() + 1) / float(height)
    return float(min(1.0, face_h / FACE_FULL_FRACTION))


def run(image, sam3_model, texture, detail, pore_scale, feather, seed, threshold, mask, exclude_mask):
    b, h, w, _ = image.shape
    gate = 1.0

    if mask is not None:
        skin = fit_mask_batch(mask, h, w, b)
    else:
        model, clip = load(sam3_model)
        skin, _ = detect(model, clip, image, SKIN_PROMPT, threshold)
        drop, _ = detect(model, clip, image, EXCLUDE_PROMPT, threshold)
        face, _ = detect(model, clip, image, FACE_PROMPT, threshold)
        skin = (skin - drop).clamp(0, 1)
        gate = min(face_gate(face[i], h) for i in range(b))

    if exclude_mask is not None:
        ex = fit_mask_batch(exclude_mask, h, w, b)
        skin = (skin - ex.to(skin.device)).clamp(0, 1)

    skin = skin.to(image.device)
    if feather > 0:
        skin = gauss_reflect(skin.unsqueeze(1), feather * max(h, w) / 1024.0).squeeze(1).clamp(0, 1)

    out = apply_texture(image, skin, texture=texture, detail=detail, pore_scale=pore_scale, seed=seed, gate=gate)
    return (out, skin.cpu())

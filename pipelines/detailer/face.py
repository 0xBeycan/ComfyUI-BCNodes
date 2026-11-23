"""Face Detailer: SAM 3 finds every face of a frame; the character's face is detailed with the main
model (the one with the character LoRA), every other face with `model_no_lora` when it is connected and
not at all when it is not. Each face gets up to three passes of the detailer core (core.py), each a
region of its own: the face, then the mouth, then each eye, the mouth and the eyes detected by SAM 3 on
the face's box of the frame the face pass left, cut to the face's mask.

- The character: the face the external mask overlaps most (none when it overlaps none); without the
  mask, the face of the largest area. Chosen among every face found, before the size gate.
- The size gate: a face shorter than `min_height` px is left as it is (its eyes are a few pixels); a
  face taller than `max_height` skips only the face pass, the mouth and eyes still run. Every pass also
  skips itself when its crop would not be enlarged for sampling (core.locate): a large face is left
  alone while its eyes, a fifth of its width each, are still detailed.
- Order: the character first, with its model, then the other faces with theirs, so the model changes
  once per frame.

SAM 3 is the multiplex checkpoint under models/checkpoints (models/sam3, downloaded on first use when
missing); its prompts are fixed here. Everything beyond torch is imported on first use.
"""

import torch

from ...libs.image import output_dtype
from ...libs.mask import fit_mask_frame
from ...models.sam3.detect import detect, text_condition
from ...models.sam3.loader import load
from .core import detail, differential, locate, mark, working_frame
from .region import bounds

FACE_PROMPT = "face:200"      # every face: ":N" keeps up to N detections; SAM 3's detector has 200 queries
MOUTH_PROMPT = "teeth, lips"  # one detection of each, detailed as one region
EYES_PROMPT = "eyes:2"        # each eye a detection, and a region, of its own


def find_faces(sam, cond, frame, threshold):
    """The faces SAM 3 finds on `frame` (H, W, C): their masks (H, W bool), the empty ones left out."""
    masks, _ = detect(sam, cond, frame[None], threshold, individual_masks=True)
    return [m for m in (mask > 0.5 for mask in masks) if bool(m.any())]


def character(faces, select):
    """The index in `faces` (H, W bool masks) of the character's face: the face `select` (H, W bool)
    overlaps most, None when it overlaps none; `select` None: the face of the largest area. None when
    there is no face."""
    if not faces:
        return None
    if select is None:
        areas = [int(face.sum()) for face in faces]
        return areas.index(max(areas))
    overlaps = [int((face & select).sum()) for face in faces]
    return overlaps.index(max(overlaps)) if max(overlaps) > 0 else None


def find_parts(sam, cond, frame, face, threshold, separate):
    """The regions SAM 3 finds for `cond` on the box of `face` (H, W bool) of `frame`, each cut to the
    face's mask (an empty one left out): one per detection when `separate`, otherwise their union as
    one region (none when nothing is found)."""
    x0, y0, x1, y1 = bounds(face)
    masks, _ = detect(sam, cond, frame[None, y0:y1, x0:x1], threshold, individual_masks=True)
    parts = []
    for mask in masks:
        part = torch.zeros_like(face)
        part[y0:y1, x0:x1] = mask > 0.5
        part &= face
        if bool(part.any()):
            parts.append(part)
    if separate or not parts:
        return parts
    return [torch.stack(parts).any(dim=0)]


def _detail(frame, region, sampler, s, changed):
    """One pass of the core on `region` (H, W bool) of `frame`, its pixels marked in `changed`; nothing
    when its crop would not be enlarged."""
    located = locate(region, s)
    if located is not None:
        detail(frame, located, sampler, s)
        mark(changed, located)


def run(image, model, model_no_lora, s, sam3_model, threshold, select, min_height, max_height, face_pass, mouth_pass,
        eyes_pass):
    """BC_FaceDetailer: (image with the faces detailed, the pixels changed), frame by frame into outputs of
    the image's dtype when that is half precision, float32 otherwise. `select`: the optional MASK that
    picks the character; `s`: core.Settings."""
    b, h, w, c = image.shape
    sam, clip = load(sam3_model)
    face_cond = text_condition(clip, FACE_PROMPT)
    mouth_cond = text_condition(clip, MOUTH_PROMPT) if mouth_pass else None
    eyes_cond = text_condition(clip, EYES_PROMPT) if eyes_pass else None
    main, other = differential(model), None if model_no_lora is None else differential(model_no_lora)
    out = torch.empty((b, h, w, c), dtype=output_dtype(image))
    changed = torch.zeros((b, h, w), dtype=output_dtype(image))
    for i in range(b):
        frame = working_frame(image[i])
        faces = find_faces(sam, face_cond, frame, threshold)
        chosen = character(faces, None if select is None else fit_mask_frame(select, i, h, w, b) > 0.5)
        order = [] if chosen is None else [(faces[chosen], main)]
        if other is not None:
            order += [(face, other) for n, face in enumerate(faces) if n != chosen]
        for face, sampler in order:
            _, top, _, bottom = bounds(face)
            if bottom - top < min_height:
                continue
            if face_pass and bottom - top <= max_height:
                _detail(frame, face, sampler, s, changed[i])
            if mouth_pass:
                for part in find_parts(sam, mouth_cond, frame, face, threshold, separate=False):
                    _detail(frame, part, sampler, s, changed[i])
            if eyes_pass:
                for part in find_parts(sam, eyes_cond, frame, face, threshold, separate=True):
                    _detail(frame, part, sampler, s, changed[i])
        out[i] = frame
    return out, changed

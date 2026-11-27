"""Face Detailer: SAM 3 finds every face of a frame; the character's face is detailed with the main
model (the one with the character LoRA), every other face with `model_no_lora` when it is connected and
not at all when it is not. Each face gets two passes of the detailer core (core.py): the face, then its
eyes as one region (the union of SAM 3's eye detections on the face's box of the frame the face pass
left, cut to the face's mask), so the model sees both eyes in one crop and their gazes match; one eye
alone when only one is found.

- The character: the face the external mask overlaps most (none when it overlaps none); without the
  mask, the face of the largest area. Chosen among every face found, before the size gate.
- The size gate: a face shorter than MIN_FACE_HEIGHT px is left as it is (its eyes are a few pixels); a
  face taller than `max_height` px skips the face pass. Every pass samples its region by one rule, core's:
  the region's longer side at s.size px inside a crop sampled at s.crop_size px; a region whose longer
  side is s.size px or more skips its pass (core.locate). The face pass with the face's settings, the eye
  pass with the eyes' (their own size, crop size and denoise). So a face too large for the face pass keeps
  its own pixels while its eyes are still detailed.
- Order: the character first, with its model, then the other faces with theirs, so the model changes
  once per frame.
- A pass at denoise 0 is off, as its toggle off: it would redraw nothing, so its regions keep their pixels and
  are not marked (the eye pass off: no eye search).
- Face and eyes both off: the image passes through as it is and SAM 3 is not loaded.

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

FACE_PROMPT = "face:200"  # every face: ":N" keeps up to N detections; SAM 3's detector has 200 queries
EYES_PROMPT = "eyes:2"    # each eye a detection; the two detailed as one region
MIN_FACE_HEIGHT = 64      # px: a shorter face gets no pass (an eye about 13 px; the model would invent the face)


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


def find_part(sam, cond, frame, face, threshold):
    """The region SAM 3 finds for `cond` on the box of `face` (H, W bool) of `frame`: the union of its
    detections, cut to the face's mask; None when none of it is inside the face."""
    x0, y0, x1, y1 = bounds(face)
    masks, _ = detect(sam, cond, frame[None, y0:y1, x0:x1], threshold, individual_masks=True)
    part = torch.zeros_like(face)
    part[y0:y1, x0:x1] = (masks > 0.5).any(dim=0)
    part &= face
    return part if bool(part.any()) else None


def _detail(frame, region, sampler, s, changed):
    """One pass of the core on `region` (H, W bool) of `frame`, its pixels marked in `changed`; nothing
    when its longer side is s.size px or more."""
    located = locate(region, s)
    if located is not None:
        detail(frame, located, sampler, s)
        mark(changed, located)


def run(image, model, model_no_lora, face_s, eye_s, sam3_model, threshold, select, max_height, face_pass, eyes_pass):
    """BC_FaceDetailer: (image with the faces detailed, the pixels changed), frame by frame into outputs of
    the image's dtype when that is half precision, float32 otherwise. `select`: the optional MASK that
    picks the character; `face_s`, `eye_s`: the core.Settings of the face pass and of the eye pass;
    `max_height`: px, a taller face skips the face pass. A pass at denoise 0 is off."""
    b, h, w, c = image.shape
    face_pass, eyes_pass = face_pass and face_s.denoise > 0, eyes_pass and eye_s.denoise > 0
    if not (face_pass or eyes_pass):  # nothing to detail: SAM 3 is not loaded, no model is cloned
        return image.to(output_dtype(image)), torch.zeros((b, h, w), dtype=output_dtype(image))
    sam, clip = load(sam3_model)
    face_cond = text_condition(clip, FACE_PROMPT)
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
            if bottom - top < MIN_FACE_HEIGHT:
                continue
            if face_pass and bottom - top <= max_height:
                _detail(frame, face, sampler, face_s, changed[i])
            if eyes_pass:
                eyes = find_part(sam, eyes_cond, frame, face, threshold)
                if eyes is not None:
                    _detail(frame, eyes, sampler, eye_s, changed[i])
        out[i] = frame
    return out, changed

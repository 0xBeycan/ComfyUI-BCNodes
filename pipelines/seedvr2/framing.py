"""Framing — SeedVR2 Resize's `downscale_factor` from how large the face is in the frame.

A lower downscale factor gives SeedVR2 more to restore, so more skin texture, but it also shrinks
the face SeedVR2 sees: a face that is small in the frame comes out changed. The tallest SAM 3
`face` box over the batch, as a fraction of the frame height, picks one of three bands (close-up,
medium, far), each with its factor; no face gives a factor of its own. One factor for the whole
batch, because Resize takes one.

The faces are detected a few frames at a time (core's SAM3_Detect scales its whole input to
1008x1008 first, in the dtype it gets: a half-precision chunk is read as float32 first, a frame at a time
through libs/image.float_frame) and without the
mask refinement (only the boxes are read); the prompt is encoded once.
"""

import logging

import torch

from ...libs.image import float_frame, is_half
from ...models.sam3.detect import detect, text_condition
from ...models.sam3.loader import load
from . import FRAMES_PER_CHUNK, require_image_batch

FACE_PROMPT = "face:4"  # up to four faces per frame, the highest-scoring


def tallest_face(image, sam3_model, threshold):
    """The tallest SAM 3 face box over every frame of `image` (B, H, W, C), cut to the frame, as a
    fraction of the frame height; 0.0 when no frame has a face."""
    model, clip = load(sam3_model)
    cond = text_condition(clip, FACE_PROMPT)
    height = image.shape[1]
    tallest = 0.0
    for start in range(0, image.shape[0], FRAMES_PER_CHUNK):
        chunk = image[start:start + FRAMES_PER_CHUNK]
        if is_half(chunk):  # no float16 arithmetic on the CPU in core's scaling
            read = torch.empty(chunk.shape, dtype=torch.float32)
            for j in range(chunk.shape[0]):
                float_frame(chunk[j], out=read[j])
            chunk = read
        _, boxes = detect(model, cond, chunk, threshold, refine_iterations=0)
        for frame in boxes:
            for box in frame:
                top, bottom = max(0.0, box["y"]), min(float(height), box["y"] + box["height"])
                tallest = max(tallest, bottom - top)
    return tallest / height


def downscale_factor(image, sam3_model, threshold, close_up_min_face, close_up_factor, medium_min_face, medium_factor, far_factor,
                     no_face_factor):
    """-> (the downscale factor, the face fraction it was chosen by): `close_up_factor` when the
    tallest face is at least `close_up_min_face` of the frame height, `medium_factor` when at least
    `medium_min_face`, `far_factor` below that, `no_face_factor` (fraction 0.0) when no face is found."""
    require_image_batch(image, "SeedVR2 Framing Downscale: connect an image batch (B, H, W, C)")
    if medium_min_face >= close_up_min_face:
        raise ValueError(f"SeedVR2 Framing Downscale: medium_min_face ({medium_min_face}) must be below close_up_min_face "
                         f"({close_up_min_face}); lower medium_min_face or raise close_up_min_face")
    fraction = tallest_face(image, sam3_model, threshold)
    if fraction == 0.0:
        logging.warning("SeedVR2 Framing Downscale: no face found in %d frame(s), downscale_factor %s (no_face_factor)",
                        image.shape[0], no_face_factor)
        return (no_face_factor, 0.0)
    if fraction >= close_up_min_face:
        factor, band = close_up_factor, "close-up"
    elif fraction >= medium_min_face:
        factor, band = medium_factor, "medium"
    else:
        factor, band = far_factor, "far"
    logging.info("SeedVR2 Framing Downscale: tallest face %.3f of the frame height (%s), downscale_factor %s", fraction, band, factor)
    return (factor, fraction)

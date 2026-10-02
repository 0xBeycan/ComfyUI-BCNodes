"""Image conversions and fitting shared by the nodes. PIL is imported inside the functions.

Half-precision inputs: an IMAGE or MASK may come as float16 or bfloat16 (BCVideoNodes' Load Video
gives float16 8-bit frames; SeedVR2 PostProcess gives float16 continuous values). The nodes read such
an input a frame (or a few frames) at a time through float_frame, work in float32, and write into a
preallocated output of the input's own dtype (output_dtype). No float16 arithmetic runs on the CPU,
and the input is never widened as a whole (that would hold both copies)."""

import numpy as np
import torch

# float16 keeps an 8-bit level k / 255 within 255 * 2^-12 = 0.0623 of k once multiplied by 255 (its step
# below 1.0 is 2^-11): a float16 frame whose values all lie within 1/16 of a level is 8-bit data.
LEVEL_TOLERANCE = 1 / 16
# bfloat16 keeps a level only within 255 * 2^-9 = 0.498 of it (its step is 2^-8), which no tolerance can tell
# from continuous values: its frames are read as they are, and tensor_to_u8 adds this before truncating,
# above that error and below 1 minus it, so an 8-bit level given as bfloat16 comes back as k.
BF16_U8_MARGIN = 1 / 2


def is_half(x):
    """Whether the tensor `x` is float16 or bfloat16."""
    return x.dtype in (torch.float16, torch.bfloat16)


def output_dtype(x):
    """The dtype a node's IMAGE or MASK output made from `x` takes: `x`'s own when it is half
    precision, float32 otherwise."""
    return x.dtype if is_half(x) else torch.float32


def _on_levels(x):
    """Whether every value of `x` (the float32 values of a float16 frame) lies within LEVEL_TOLERANCE of
    an 8-bit level: |255 x - round(255 x)| <= 1/16, as | |frac(255 x)| - 1/2 | >= 1/2 - 1/16 (exact in
    float32: 255 x of a float16 value needs 19 bits); a NaN or an infinity is no level. A quick look at
    the first values first: a continuous frame fails there."""
    for part in (x.reshape(-1)[:4096], x):
        distance = (part * 255).frac_().abs_().sub_(0.5).abs_()
        if not float(distance.min()) >= 0.5 - LEVEL_TOLERANCE:
            return False
    return True


def float_frame(frame, out=None):
    """`frame` (one frame of an IMAGE or MASK) ready for float32 arithmetic. A float16 frame whose values
    all lie within 1/16 of an 8-bit level is 8-bit data: it comes back as those levels k / 255 in float32,
    exactly its float32 source. Any other half frame comes back as its exact values (.float()), anything
    else as itself. A continuous float16 frame that passed for 8-bit data (all of its values that close to
    a level) moves each value by at most 1/16 of a level. `out`: a float32 tensor of the frame's shape for
    a half frame."""
    if not is_half(frame):
        return frame
    x = frame.float() if out is None else out.copy_(frame)
    if frame.dtype == torch.float16 and x.numel() and _on_levels(x):
        x.mul_(255).round_().div_(255)
    return x


def tensor_to_u8(frame, stored=None):
    """float tensor in 0..1 -> uint8 numpy array of its shape (x 255, clipped, truncated). A half frame is
    read through float_frame first (float16 8-bit data as its exact levels), so it gives its float32
    source's uint8: float16(1/255) x 255 is 0.99998, which a plain cast truncates to 0. Values stored as
    bfloat16 (`stored`: the dtype they were stored in, the frame's own by default; a float32 frame read
    from a bfloat16 one passes it) get BF16_U8_MARGIN before the truncation. A float32 frame is converted
    as it always was. One float buffer of its own, clipped in place; it is freed when this returns,
    before the caller builds anything on the result."""
    frame = frame.cpu()
    if is_half(frame):
        buf = float_frame(frame).numpy()  # a float32 frame of its own: scaled in place
        buf *= 255.0
    else:
        buf = 255.0 * frame.numpy()
    if (frame.dtype if stored is None else stored) == torch.bfloat16:
        buf += BF16_U8_MARGIN
    np.clip(buf, 0, 255, out=buf)
    return buf.astype(np.uint8)


def tensor_to_pil_u8(frame, stored=None):
    """float (H, W) or (H, W, C) tensor in 0..1 -> 8-bit PIL image (clipped, truncated; `stored` as
    tensor_to_u8 takes it)."""
    from PIL import Image

    return Image.fromarray(tensor_to_u8(frame, stored))


def pil_to_tensor_hwc(pil):
    """PIL image -> float32 (H, W) or (H, W, C) tensor in 0..1."""
    return torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)


def fit_image(image, target_width, target_height, fit, sampler, background):
    """letterbox: whole image inside the target, `background` around it.
    crop: centre crop to the target ratio, then resize. fill: plain resize."""
    from PIL import Image

    orig_width, orig_height = image.size
    if fit == "letterbox":
        if orig_width / orig_height > target_width / target_height:
            fit_width = target_width
            fit_height = int(target_width / orig_width * orig_height)
        else:
            fit_height = target_height
            fit_width = int(target_height / orig_height * orig_width)
        resized = image.resize((fit_width, fit_height), sampler)
        out = Image.new(image.mode, (target_width, target_height), color=background)
        out.paste(resized, box=((target_width - fit_width) // 2, (target_height - fit_height) // 2))
        return out
    if fit == "crop":
        if orig_width / orig_height > target_width / target_height:
            fit_width = int(orig_height * target_width / target_height)
            left = (orig_width - fit_width) // 2
            image = image.crop((left, 0, left + fit_width, orig_height))
        else:
            fit_height = int(orig_width * target_height / target_width)
            top = (orig_height - fit_height) // 2
            image = image.crop((0, top, orig_width, top + fit_height))
    return image.resize((target_width, target_height), sampler)

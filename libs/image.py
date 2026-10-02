"""Image conversions and fitting shared by the nodes. PIL is imported inside the functions.

Half-precision clips: a loader may give an IMAGE or MASK clip as float16 (8-bit frames, every level
k / 255 kept within 2^-12 of it). The nodes read such a clip a frame (or a few frames) at a time
through `requantized`, which gives back exactly the float32 values a float32 load holds, compute
in float32, and write into a preallocated output of the clip's own dtype. No float16 arithmetic
runs on the CPU, and the clip is never widened as a whole (that would hold both copies)."""

import numpy as np
import torch


def is_half(x):
    """Whether the tensor `x` is float16 or bfloat16."""
    return x.dtype in (torch.float16, torch.bfloat16)


def requantized(frames, out=None):
    """`frames` (a frame or a few frames of an IMAGE or MASK clip) ready for float32 arithmetic: a
    half-precision tensor as a new float32 one (or written into `out`, a float32 tensor of its
    shape), every value rounded to the nearest 8-bit level k / 255; any other tensor itself.
    float16 keeps every level within 2^-12 of it and bfloat16 within 2^-9, both under half a level
    (1 / 510), so a half clip of 8-bit frames comes back as exactly the float32 values of a float32
    load: float16(k / 255) * 255 is not k (1 / 255 gives 0.99998, which an 8-bit cast truncates to
    0)."""
    if not is_half(frames):
        return frames
    widened = frames.float() if out is None else out.copy_(frames)
    return widened.mul_(255).round_().div_(255)


def output_dtype(x):
    """The dtype a node's IMAGE or MASK output made from `x` takes: `x`'s own when it is half
    precision, float32 otherwise."""
    return x.dtype if is_half(x) else torch.float32


def tensor_to_u8(frame):
    """float tensor in 0..1 -> uint8 numpy array of its shape (x 255, clipped, truncated), a half
    one requantized first (requantized). One float buffer, clipped in place; it is freed when this
    returns, before the caller builds anything on the result."""
    frame = frame.cpu()
    if is_half(frame):
        buf = requantized(frame).numpy()  # a float32 frame of its own: scaled in place
        buf *= 255.0
    else:
        buf = 255.0 * frame.numpy()
    np.clip(buf, 0, 255, out=buf)
    return buf.astype(np.uint8)


def tensor_to_pil_u8(frame):
    """float (H, W) or (H, W, C) tensor in 0..1 -> 8-bit PIL image (clipped, truncated)."""
    from PIL import Image

    return Image.fromarray(tensor_to_u8(frame))


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

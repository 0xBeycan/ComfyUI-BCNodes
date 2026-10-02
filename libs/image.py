"""Image conversions and fitting shared by the nodes. PIL is imported inside the functions.

Half-precision inputs: an IMAGE or MASK may come as float16 or bfloat16 (BCVideoNodes' Load Video
gives float16 8-bit frames; SeedVR2 PostProcess gives float16 continuous values). The nodes read such
an input a frame (or a few frames) at a time with a plain `.float()`, the exact value of each half
number, work in float32, and write into a preallocated output of the input's own dtype
(output_dtype). No float16 arithmetic runs on the CPU, and the input is never widened as a whole
(that would hold both copies). A half value is not assumed to be an 8-bit level k / 255, except at
the 8-bit boundary (tensor_to_u8)."""

import numpy as np
import torch

# What tensor_to_u8 adds to x 255 before it truncates a half value. float16 keeps an 8-bit level
# k / 255 within 255 * 2^-12 = 0.0623 of k once multiplied (its step below 1.0 is 2^-11), bfloat16
# within 255 * 2^-9 = 0.498 (step 2^-8): adding a margin above that and below 1 - it brings every
# level that fell under k back to k, and leaves the levels that lie above k at k. On values that are not
# levels it moves the truncation by the margin (a float16 value within 1/16 of the next level goes up).
U8_MARGIN = {torch.float16: 1 / 16, torch.bfloat16: 1 / 2}


def is_half(x):
    """Whether the tensor `x` is float16 or bfloat16."""
    return x.dtype in (torch.float16, torch.bfloat16)


def output_dtype(x):
    """The dtype a node's IMAGE or MASK output made from `x` takes: `x`'s own when it is half
    precision, float32 otherwise."""
    return x.dtype if is_half(x) else torch.float32


def tensor_to_u8(frame, stored=None):
    """float tensor in 0..1 -> uint8 numpy array of its shape (x 255, clipped, truncated). Values stored
    in half precision (`stored`, the dtype they were stored in: the frame's own by default; a float32
    frame computed from a half one, such as an inverted mask, passes the half dtype) get U8_MARGIN
    added before the truncation, so an 8-bit level comes back as itself: float16(1/255) x 255 is
    0.99998, which a plain cast truncates to 0. A float32 frame is converted as it always was. One
    float buffer of its own, clipped in place; it is freed when this returns, before the caller builds
    anything on the result."""
    frame = frame.cpu()
    buf = 255.0 * (frame.float() if is_half(frame) else frame).numpy()
    margin = U8_MARGIN.get(frame.dtype if stored is None else stored)
    if margin:
        buf += margin
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

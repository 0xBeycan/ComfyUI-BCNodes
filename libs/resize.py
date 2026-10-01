"""Image Resize: an image batch, and optionally a mask, to a width and height, frame by frame into
one preallocated output.

Size (`plan`), from the source size and the widgets:
  stretch        width x height; a 0 keeps that side of the source.
  resize         the largest size inside width x height at the source aspect (a 0 side is free).
  total_pixels   width * height pixels at the source aspect.
  crop           width x height; the source is first cropped to that aspect at `crop_position`.
  pad, pad_edge, pad_edge_pixel, pillarbox_blur
                 as resize, then padded out to width x height with the frame placed at
                 `crop_position`; the padding is `pad_color`, the mean of the frame's edge row /
                 column, the edge pixels repeated, or a blurred, desaturated and dimmed cover of
                 the frame itself.
Both sides are then floored to a multiple of `divisible_by`; a padded frame instead grows its
right / bottom padding up to the next multiple.

Resampling: nearest-exact, bilinear, area and bicubic through torch; lanczos through PIL on
8-bit frames, as ComfyUI's lanczos does; nvidia_rtx_vsr through NVIDIA RTX Video Super
Resolution (the nvidia-vfx package, CUDA only), whose output is the size rounded to a multiple
of 8.

Mask: resized with the image (bilinear to the image's size first when it differs), padded with
its own edge values, or with 1 around the frame for pillarbox_blur. Without a mask, a padded
run returns the padding as the mask (1 = padding) and any other run a (1, 64, 64) zero mask.

An image that already has the output size is returned as is. PIL and nvidia-vfx are imported
inside the functions that use them.
"""

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .image import pil_to_tensor_hwc, tensor_to_pil_u8

VSR = "nvidia_rtx_vsr"
PADDED = ("pad", "pad_edge", "pad_edge_pixel", "pillarbox_blur")


@dataclass(frozen=True)
class ResizePlan:
    size: tuple  # (height, width) the frame is resampled to
    crop: Optional[tuple]  # (y, x, height, width) of the source frame; None = the whole frame
    pad: Optional[tuple]  # (top, bottom, left, right); None = no padding

    @property
    def output(self):
        """(height, width) of an output frame."""
        if self.pad is None:
            return self.size
        top, bottom, left, right = self.pad
        return self.size[0] + top + bottom, self.size[1] + left + right


def _offset(free_width, free_height, position):
    """(x, y) of a frame placed with `free_width` x `free_height` pixels to spare."""
    x = {"left": 0, "right": free_width}.get(position, free_width // 2)
    y = {"top": 0, "bottom": free_height}.get(position, free_height // 2)
    return x, y


def _proportional_size(src_width, src_height, width, height, keep_proportion):
    if keep_proportion == "total_pixels":
        total = width * height
        aspect = src_width / src_height
        return int(math.sqrt(total * aspect)), int(math.sqrt(total / aspect))
    if width == 0 and height == 0:
        return src_width, src_height
    if width == 0:
        return round(src_width * (height / src_height)), height
    if height == 0:
        return width, round(src_height * (width / src_width))
    ratio = min(width / src_width, height / src_height)
    return round(src_width * ratio), round(src_height * ratio)


def plan(src_height, src_width, width, height, keep_proportion, crop_position, divisible_by, upscale_method):
    """The resample size, the source crop and the padding for one run (see the module doc)."""
    padded = keep_proportion in PADDED
    top = bottom = left = right = 0
    if padded or keep_proportion in ("resize", "total_pixels"):
        new_width, new_height = _proportional_size(src_width, src_height, width, height, keep_proportion)
        if padded:
            left, top = _offset(width - new_width, height - new_height, crop_position)
            right, bottom = width - new_width - left, height - new_height - top
        width, height = new_width, new_height
    else:
        width, height = width or src_width, height or src_height
    if divisible_by > 1:
        width -= width % divisible_by
        height -= height % divisible_by
    if width < 1 or height < 1:
        raise ValueError(f"Image Resize: these settings give a {width}x{height} image; raise width / height "
                         f"or lower divisible_by")

    crop = None
    if keep_proportion == "crop":
        aspect = width / height
        if src_width / src_height > aspect:
            crop_width, crop_height = round(src_height * aspect), src_height
        else:
            crop_width, crop_height = src_width, round(src_width / aspect)
        x, y = _offset(src_width - crop_width, src_height - crop_height, crop_position)
        if (crop_width, crop_height) != (src_width, src_height):
            crop = (y, x, crop_height, crop_width)

    if upscale_method == VSR:
        width, height = max(8, round(width / 8) * 8), max(8, round(height / 8) * 8)

    pad = None
    if padded and max(top, bottom, left, right) > 0:
        if divisible_by > 1:
            right += -(width + left + right) % divisible_by
            bottom += -(height + top + bottom) % divisible_by
        pad = (top, bottom, left, right)
    return ResizePlan((height, width), crop, pad)


def _crop(frames, crop, channels_last):
    """The crop window of [..., H, W] frames, or of [..., H, W, C] ones with `channels_last`; a view."""
    if crop is None:
        return frames
    y, x, height, width = crop
    dim = -2 if channels_last else -1
    return frames.narrow(dim, x, width).narrow(dim - 1, y, height)


def _lanczos(frame, size):
    """ComfyUI's lanczos on one (H, W) or (H, W, C) frame: PIL LANCZOS on 8-bit."""
    from PIL import Image

    grey = frame.ndim == 3 and frame.shape[-1] == 1
    pil = tensor_to_pil_u8(frame[..., 0] if grey else frame).resize((size[1], size[0]), resample=Image.Resampling.LANCZOS)
    out = pil_to_tensor_hwc(pil)
    return (out[..., None] if grey else out).to(frame.device, frame.dtype)


def _resample_image(frame, size, method, sr):
    """A (1, H, W, C) batch of one frame to `size`. Kept a batch, not unsqueezed from (H, W, C):
    torch then picks the same channels-last kernel and output layout as for a whole batch."""
    if method == "lanczos":
        return _lanczos(frame[0], size)[None]
    if method == VSR:
        # Copied off the SR buffer at once: the next run overwrites it.
        return torch.from_dlpack(sr.run(frame[0].movedim(-1, 0).cuda().contiguous()).image).movedim(0, -1)[None].to(frame.device, copy=True)
    return F.interpolate(frame.movedim(-1, 1), size=size, mode=method).movedim(1, -1)


def _resample_mask(frame, size, method):
    """One (H, W) mask frame to `size`; RTX VSR is image-only, its masks go bilinear."""
    if method == "lanczos":
        return _lanczos(frame, size)
    mode = "bilinear" if method == VSR else method
    return F.interpolate(frame[None, None], size=size, mode=mode)[0, 0]


def _replicate_into(dst, frame, top, left):
    """Write `frame` into `dst` at (top, left) and fill the rest of `dst` with its edge pixels
    repeated outwards (corners take the corner pixel). Works on (H, W) and (H, W, C)."""
    height, width = frame.shape[0], frame.shape[1]
    dst[top:top + height, left:left + width] = frame
    dst[:top, left:left + width] = frame[:1]
    dst[top + height:, left:left + width] = frame[-1:]
    dst[:, :left] = dst[:, left:left + 1]
    dst[:, left + width:] = dst[:, left + width - 1:left + width]


def _gaussian_blur(image, sigma):
    """Separable Gaussian of (1, C, H, W), zero padded, radius int(3 sigma)."""
    radius = max(1, int(3.0 * float(sigma)))
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel = torch.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    kernel = kernel / kernel.sum()
    channels = image.shape[1]
    image = F.conv2d(image, kernel.view(1, 1, 1, size).repeat(channels, 1, 1, 1), padding=(0, radius), groups=channels)
    return F.conv2d(image, kernel.view(1, 1, size, 1).repeat(channels, 1, 1, 1), padding=(radius, 0), groups=channels)


def _pillarbox_background(frame, out_height, out_width):
    """The frame scaled to cover out_height x out_width, centre-cropped, blurred, 20% desaturated
    and dimmed to 35%; (out_height, out_width, C)."""
    height, width, channels = frame.shape
    scale = max(out_width / float(width), out_height / float(height))
    cover_width, cover_height = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    bg = F.interpolate(frame.movedim(-1, 0).unsqueeze(0), size=(cover_height, cover_width), mode="bilinear")
    y0, x0 = max(0, (cover_height - out_height) // 2), max(0, (cover_width - out_width) // 2)
    bg = bg[:, :, y0:min(cover_height, y0 + out_height), x0:min(cover_width, x0 + out_width)]
    short_height, short_width = out_height - bg.shape[2], out_width - bg.shape[3]
    if short_height or short_width:
        bg = F.pad(bg, (short_width // 2, short_width - short_width // 2, short_height // 2, short_height - short_height // 2),
                   mode="replicate")
    bg = _gaussian_blur(bg, max(1.0, 0.006 * float(min(out_height, out_width))))
    if channels >= 3:
        luma = 0.2126 * bg[:, 0:1] + 0.7152 * bg[:, 1:2] + 0.0722 * bg[:, 2:3]
        desaturate = 0.20
        bg[:, 0:3] = bg[:, 0:3] * (1.0 - desaturate) + luma * desaturate
    return torch.clamp(bg * 0.35, 0.0, 1.0)[0].movedim(0, -1)


def _pad_image_into(dst, frame, pad, keep_proportion, fill):
    """Write the resampled frame into dst (the output frame) with the padding of `keep_proportion`."""
    top, _, left, _ = pad
    height, width = frame.shape[0], frame.shape[1]
    if keep_proportion == "pad_edge_pixel":
        _replicate_into(dst, frame, top, left)
        return
    if keep_proportion == "pillarbox_blur":
        dst.copy_(_pillarbox_background(frame, dst.shape[0], dst.shape[1]))
    elif keep_proportion == "pad_edge":
        # Band order matters: the left / right means overwrite the corners of the top / bottom bands.
        dst[:top] = frame[0].mean(dim=0)
        dst[top + height:] = frame[-1].mean(dim=0)
        dst[:, :left] = frame[:, 0].mean(dim=0)
        dst[:, left + width:] = frame[:, -1].mean(dim=0)
    else:
        dst[...] = fill
    dst[top:top + height, left:left + width] = frame


def _pad_mask_into(dst, frame, pad, keep_proportion):
    top, _, left, _ = pad
    if keep_proportion == "pillarbox_blur":
        dst.fill_(1.0)
        dst[top:top + frame.shape[0], left:left + frame.shape[1]] = frame
    else:
        _replicate_into(dst, frame, top, left)


def parse_pad_color(text):
    """`pad_color` -> 0-255 channel values: "r, g, b[, a]" (all values in 0-1 are scaled by 255),
    "#rrggbb[aa]" or "rrggbb[aa]", a colour name, or one grey value (0-1 or 0-255)."""
    from PIL import ImageColor

    text = str(text).strip()
    hint = "use 'r, g, b' (0-255 or 0-1), '#rrggbb', a colour name or one grey value"
    try:
        if "," in text:
            values = [float(v.strip()) for v in text.split(",")]
            channels = [int(v * 255) for v in values] if all(0 <= v <= 1 for v in values) else [int(v) for v in values]
        elif text.startswith("#") or (text.lstrip("#").isalnum() and not text.lstrip("#").replace(".", "", 1).isdigit()):
            digits = text.lstrip("#")
            if len(digits) in (6, 8) and all(c in "0123456789abcdefABCDEF" for c in digits):
                channels = [int(digits[i:i + 2], 16) for i in range(0, len(digits), 2)]
            else:
                channels = list(ImageColor.getrgb(text))
        else:
            value = float(text)
            channels = [int(value * 255) if 0 <= value <= 1 else int(value)] * 3
    except ValueError as e:
        raise ValueError(f"Image Resize: pad_color {text!r} is not a colour; {hint}") from e
    return [min(max(c, 0), 255) for c in channels]


def _pad_fill(pad_color, channels, dtype, device):
    color = parse_pad_color(pad_color)
    if len(color) != channels:
        raise ValueError(f"Image Resize: pad_color {pad_color!r} has {len(color)} values but the image has {channels} "
                         f"channels; give {channels} values")
    return torch.tensor([c / 255.0 for c in color], dtype=dtype, device=device)


def _super_resolution(size):
    """An RTX Video Super Resolution context set up for `size` (height, width)."""
    try:
        import nvvfx
    except ImportError as e:
        raise ImportError("Image Resize: nvidia_rtx_vsr needs the nvidia-vfx package and an NVIDIA RTX GPU; install "
                          "nvidia-vfx or pick another upscale_method") from e

    class _Context:
        def __enter__(self):
            self.ctx = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
            sr = self.ctx.__enter__()
            sr.output_height, sr.output_width = size
            sr.load()
            return sr

        def __exit__(self, *exc):
            return self.ctx.__exit__(*exc)

    return _Context()


def _is_placeholder(mask, height, width):
    """ComfyUI's stand-in for "no mask": a zero 64x64 mask on an image of another size."""
    return tuple(mask.shape[-2:]) == (64, 64) and (height, width) != (64, 64) and not bool(mask.any())


def resize_image(image, mask, width, height, upscale_method, keep_proportion, pad_color, crop_position, divisible_by, device):
    """-> (image, width, height, mask) as the module doc describes. `image` is (B, H, W, C), `mask`
    any MASK or None, `device` the torch device the frames are processed on; the outputs are on
    the CPU."""
    batch, src_height, src_width, channels = image.shape
    if upscale_method == "lanczos" and device.type != "cpu":
        raise ValueError("Image Resize: lanczos runs on the CPU only; set device to cpu or pick another upscale_method")
    if mask is not None:
        mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        if _is_placeholder(mask, src_height, src_width):
            mask = None
    p = plan(src_height, src_width, width, height, keep_proportion, crop_position, divisible_by, upscale_method)
    out_height, out_width = p.output
    on_cpu = device.type == "cpu"
    unchanged = p.crop is None and p.pad is None and p.size == (src_height, src_width) and upscale_method != VSR

    if unchanged:
        out_image = image.cpu()
    else:
        out_image = torch.empty((batch, out_height, out_width, channels), dtype=image.dtype)
        fill = _pad_fill(pad_color, channels, image.dtype, device) if p.pad and keep_proportion == "pad" else None
        with _super_resolution(p.size) if upscale_method == VSR else nullcontext() as sr:
            for i in range(batch):
                frame = _crop(image[i:i + 1].to(device), p.crop, True)
                if frame.shape[1:3] != p.size or upscale_method == VSR:
                    frame = _resample_image(frame, p.size, upscale_method, sr)
                frame = frame[0]
                if p.pad is None:
                    out_image[i].copy_(frame)
                    continue
                dst = out_image[i] if on_cpu else torch.empty(out_image.shape[1:], dtype=image.dtype, device=device)
                _pad_image_into(dst, frame, p.pad, keep_proportion, fill)
                if not on_cpu:
                    out_image[i].copy_(dst)

    if mask is None:
        if p.pad is None:
            return out_image, out_width, out_height, torch.zeros((1, 64, 64), dtype=torch.float32)
        top, _, left, _ = p.pad
        out_mask = torch.ones((batch, out_height, out_width), dtype=image.dtype)
        out_mask[:, top:top + p.size[0], left:left + p.size[1]] = 0.0
        return out_image, out_width, out_height, out_mask

    fitted = tuple(mask.shape[-2:]) == (src_height, src_width)
    if unchanged and fitted:
        return out_image, out_width, out_height, mask.cpu()
    mask_dtype = image.dtype if keep_proportion == "pillarbox_blur" and p.pad else mask.dtype
    out_mask = torch.empty((mask.shape[0], out_height, out_width), dtype=mask_dtype)
    for i in range(mask.shape[0]):
        frame = mask[i].to(device)
        if not fitted:
            frame = F.interpolate(frame[None, None], size=(src_height, src_width), mode="bilinear")[0, 0]
        frame = _crop(frame, p.crop, False)
        if frame.shape != p.size or upscale_method == VSR:
            frame = _resample_mask(frame, p.size, upscale_method)
        if p.pad is None:
            out_mask[i].copy_(frame)
            continue
        dst = out_mask[i] if on_cpu else torch.empty(out_mask.shape[1:], dtype=mask_dtype, device=device)
        _pad_mask_into(dst, frame, p.pad, keep_proportion)
        if not on_cpu:
            out_mask[i].copy_(dst)
    return out_image, out_width, out_height, out_mask

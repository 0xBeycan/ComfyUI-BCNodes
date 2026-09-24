"""SeedVR2's frame-shape rules: the shortest-edge resize, the pad to a multiple of
PAD_MULTIPLE, the 4n+1 frame count and the resize of a colour reference frame.
torchvision is imported inside the functions."""

import torch

PAD_MULTIPLE = 16


def side_resize(frames, resolution, max_resolution):
    """Shortest-edge resize on a (B, C, H, W) tensor, followed by the clamp."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF

    resized = TVF.resize(frames, resolution, InterpolationMode.BICUBIC, antialias=True)
    if max_resolution > 0:
        h, w = resized.shape[-2:]
        if max(h, w) > max_resolution:
            scale = max_resolution / max(h, w)
            resized = TVF.resize(resized, (round(h * scale), round(w * scale)), InterpolationMode.BICUBIC, antialias=True)
    return torch.clamp(resized, 0.0, 1.0)


def divisible_pad(frames, multiple=PAD_MULTIPLE):
    """Zeros on the bottom / right up to a multiple."""
    h, w = frames.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return frames
    return torch.nn.functional.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def frames_to_4n1(t):
    """Frames to append so a batch of t frames is 4n+1 (Pre-Process SeedVR2 Input `cut_videos`)."""
    if t == 1:
        return 0
    if t <= 4:
        return 4 - t + 1
    return (4 - (t - 1) % 4) % 4


def resize_reference(frame_hwc, height, width):
    """nodes_seedvr.py _resize_reference for one frame: bicubic, antialiased except on MPS."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF

    chw = frame_hwc.to(torch.float32).permute(2, 0, 1)[None]
    resized = TVF.resize(chw, size=(height, width), interpolation=InterpolationMode.BICUBIC,
                         antialias=not chw.device.type == "mps")
    return resized[0].permute(1, 2, 0)

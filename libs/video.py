"""Side-by-side video for BC_VideoComparer: fit a frame into the shared (h, w), the common
geometry of the clips, and the H.264 MP4 (+ optional AAC) writer. PyAV is imported inside
write_side_by_side.
"""

import math
from fractions import Fraction

import torch
import torch.nn.functional as F


def fit_frame(frame, w, h):
    """One frame (H, W, C) as (h, w, 3): the larger clip only loses the odd
    row / column an even target cuts off; a smaller clip is scaled to fit and
    letterboxed on black."""
    frame = frame[:, :, :3]
    fh, fw = frame.shape[0], frame.shape[1]
    if fh >= h and fw >= w:
        return frame[:h, :w]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * scale)), max(1, round(fh * scale))
    scaled = F.interpolate(frame.movedim(-1, 0)[None], size=(nh, nw), mode="bicubic", align_corners=False, antialias=True)
    scaled = scaled[0].movedim(0, -1).clamp(0, 1)
    out = torch.zeros((h, w, 3), dtype=frame.dtype)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y : y + nh, x : x + nw] = scaled
    return out


def side_by_side_geometry(sides, fps):
    """Frame count (the shorter clip), even output height and width of one side, and the
    stream rate for (tag, IMAGE) pairs."""
    frames = min(images.shape[0] for _, images in sides)
    # 4:2:0 wants even sizes; an even half width also lands the A|B seam on a
    # chroma block boundary.
    h = max(images.shape[1] for _, images in sides)
    w = max(images.shape[2] for _, images in sides)
    h -= h % 2
    w -= w % 2
    rate = Fraction(round(fps * 1000), 1000)
    return frames, h, w, rate


def write_side_by_side(path, sides, frames, h, w, rate, fps, audio):
    """Writes the clips side by side, one frame at a time, as an H.264 MP4 at path; the
    optional AUDIO is muxed in as AAC."""
    import av

    with av.open(path, "w", format="mp4") as out:
        video = out.add_stream("libx264", rate=rate)
        video.width = w * len(sides)
        video.height = h
        video.pix_fmt = "yuv420p"
        # A keyframe every second keeps the seek bar responsive.
        video.options = {"crf": "18", "g": str(max(1, round(fps)))}
        # Every stream must exist before the first packet writes the header.
        if audio is not None:
            sample_rate = int(audio["sample_rate"])
            wave = audio["waveform"][0, :, : math.ceil(sample_rate / rate * frames)]
            layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(wave.shape[0], "stereo")
            track = out.add_stream("aac", rate=sample_rate, layout=layout)

        for i in range(frames):
            img = torch.cat([fit_frame(images[i], w, h) for _, images in sides], dim=1)
            arr = (img * 255).clamp(0, 255).byte().cpu().numpy()
            out.mux(video.encode(av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")))
        out.mux(video.encode(None))

        if audio is not None:
            chunk = av.AudioFrame.from_ndarray(wave.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
            chunk.sample_rate = sample_rate
            chunk.pts = 0
            out.mux(track.encode(chunk))
            out.mux(track.encode(None))

"""BC_VideoComparer (Video Comparer)

Two videos as IMAGE frame batches (what Load Video → Get Video Components
gives, or any frames you generated), compared with a draggable divider
(web/js/comparer.js). Both batches are written into ONE H.264 MP4 side by
side — A on the left half, B on the right — so the browser decodes a single
stream and the two halves cannot drift apart; the widget draws each half in
its own place. Clips are cut to the shorter one, the smaller frame is
letterboxed into the larger one, and an optional AUDIO track is muxed in.
Written with PyAV, which ComfyUI itself depends on, so nothing extra to
install and no dependency on any video pack.
"""

import math
import os
import uuid
from fractions import Fraction

import torch
import torch.nn.functional as F


def _fit(frame, w, h):
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


def _encode(sides, fps, audio):
    """Writes the side-by-side clip to temp/<name>.mp4, one frame at a time,
    and returns the /view info plus what the widget needs to split it."""
    import av
    import folder_paths

    frames = min(images.shape[0] for _, images in sides)
    # 4:2:0 wants even sizes; an even half width also lands the A|B seam on a
    # chroma block boundary.
    h = max(images.shape[1] for _, images in sides)
    w = max(images.shape[2] for _, images in sides)
    h -= h % 2
    w -= w % 2
    rate = Fraction(round(fps * 1000), 1000)

    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    filename = f"bcnodes.compare.{uuid.uuid4().hex[:8]}.mp4"
    with av.open(os.path.join(temp, filename), "w", format="mp4") as out:
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
            img = torch.cat([_fit(images[i], w, h) for _, images in sides], dim=1)
            arr = (img * 255).clamp(0, 255).byte().cpu().numpy()
            out.mux(video.encode(av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")))
        out.mux(video.encode(None))

        if audio is not None:
            chunk = av.AudioFrame.from_ndarray(wave.float().cpu().contiguous().numpy(), format="fltp", layout=layout)
            chunk.sample_rate = sample_rate
            chunk.pts = 0
            out.mux(track.encode(chunk))
            out.mux(track.encode(None))

    return {
        "filename": filename,
        "subfolder": "",
        "type": "temp",
        "format": "video",
        "sides": [tag for tag, _ in sides],
        "frames": {tag: int(images.shape[0]) for tag, images in sides},
        "fps": float(fps),
        "audio": audio is not None,
    }


class VideoComparer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
            },
            "optional": {
                "video_a": ("IMAGE",),
                "video_b": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ["BCNodes", "video comparer", "compare videos"]

    def compare(self, fps=24.0, video_a=None, video_b=None, audio=None):
        fps = float(fps) if fps else 24.0
        sides = [(tag, images) for tag, images in (("A", video_a), ("B", video_b)) if images is not None and images.shape[0] > 0]
        if audio is not None and audio["waveform"].shape[-1] == 0:
            audio = None
        return {"ui": {"bc_video": [_encode(sides, fps, audio)] if sides else []}}


NODE_CLASS_MAPPINGS = {
    "BC_VideoComparer": VideoComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_VideoComparer": "Video Comparer",
}

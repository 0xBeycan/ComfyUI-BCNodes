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

import os
import uuid

from ..libs.video import side_by_side_geometry, write_side_by_side


def _encode(sides, fps, audio):
    """Writes the side-by-side clip to temp/<name>.mp4, one frame at a time,
    and returns the /view info plus what the widget needs to split it."""
    import av  # unused here: a missing PyAV must fail before the geometry and the temp file
    import folder_paths

    frames, h, w, rate = side_by_side_geometry(sides, fps)

    temp = folder_paths.get_temp_directory()
    os.makedirs(temp, exist_ok=True)
    filename = f"bcnodes.compare.{uuid.uuid4().hex[:8]}.mp4"
    write_side_by_side(os.path.join(temp, filename), sides, frames, h, w, rate, fps, audio)

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

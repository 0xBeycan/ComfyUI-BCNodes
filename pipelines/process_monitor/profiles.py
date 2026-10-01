"""Emulate's per-node-type cost profiles: what a node creates (its outputs, exact from sizes) and
what it holds besides them while it runs (its transient), each derived from that node's code.

A profile takes a Ctx and returns ({output slot: estimate}, transient bytes or None, note). An
estimate is a dict: "type", "shape", "bytes", and the dims later profiles read ("n", "h", "w" for
frames; "lt", "lh", "lw" for a SeedVR2 latent). "shared": True marks an output that is one of the
node's inputs passed on: it adds no bytes. A transient of None is "not counted"; the note says
which part of the node's memory the profile does not size (a model's own working set, mostly).
A profile raises NotCounted when an input it needs has no estimate.
"""

import math

from ...libs.geometry import aspect_ratio, round_up_to_multiple, target_size
from ...libs.resize import plan as resize_plan

F32, F16 = 4, 2
FACE_SIZE = 512  # Face Crop's square
SEEDVR2_LATENT_CHANNELS = 16
SEEDVR2_PAD = 16
# a guard's timeline image: 1200 wide, 128 + 190 per panel high with one flag row (+24 per more flag)
TIMELINE_W, TIMELINE_BASE, TIMELINE_PANEL = 1200, 128, 190
MASK_GUARD_WINDOW = 5  # frames of booleans the mask guard keeps (the frame and 2 either side)
RENDER_CHUNK = 16  # frames the colored masks are cut at a time


class NotCounted(Exception):
    pass


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


class Ctx:
    """One node as a profile sees it: its widget values, its upstream estimates, the scenario."""

    def __init__(self, node, upstream, scenario, env):
        self.inputs = node.get("inputs", {})
        self.upstream = upstream
        self.scenario = scenario
        self.env = env
        self.types = env.return_types(node["class_type"])

    def widget(self, name, default=None):
        value = self.inputs.get(name, default)
        return default if is_link(value) else value

    def linked(self, name):
        return is_link(self.inputs.get(name))

    def tensor(self, name, required=True):
        """The upstream estimate of input `name` as a fresh dict (a passed-on output is no longer
        shared once a node builds on it), or None when not required and absent."""
        value = self.inputs.get(name)
        est = self.upstream.get(value[0], {}).get(value[1]) if is_link(value) else None
        if est is None:
            if required:
                raise NotCounted(f"input {name!r} has no estimate")
            return None
        return {k: v for k, v in est.items() if k != "shared"}

    def slot(self, kind):
        return self.types.index(kind)


# -- tensor formulas ------------------------------------------------------------------------------

def image_bytes(n, h, w, dtype_bytes=F32, channels=3):
    """IMAGE: N x H x W x 3 (float32 4 bytes, float16 2)."""
    return n * h * w * channels * dtype_bytes


def mask_bytes(n, h, w, dtype_bytes=F32):
    """MASK: N x H x W."""
    return n * h * w * dtype_bytes


def wan_latent_frames(n):
    return 1 + (n - 1) // 4


def wan_latent_bytes(n, h, w, dtype_bytes=F32):
    """Wan video latent of N pixel frames: (1 + (N - 1) / 4) x 16 x H/8 x W/8."""
    return wan_latent_frames(n) * 16 * (h // 8) * (w // 8) * dtype_bytes


def image(n, h, w, dtype_bytes=F32, channels=3):
    return {"type": "IMAGE", "shape": [n, h, w, channels], "n": n, "h": h, "w": w, "dtype_bytes": dtype_bytes,
            "bytes": image_bytes(n, h, w, dtype_bytes, channels)}


def mask(n, h, w, dtype_bytes=F32):
    return {"type": "MASK", "shape": [n, h, w], "n": n, "h": h, "w": w, "dtype_bytes": dtype_bytes,
            "bytes": mask_bytes(n, h, w, dtype_bytes)}


def wan_latent(n, h, w):
    return {"type": "LATENT", "shape": [1, 16, wan_latent_frames(n), h // 8, w // 8], "n": n, "h": h, "w": w,
            "bytes": wan_latent_bytes(n, h, w)}


def shared(est):
    """An input passed on as an output: the same tensor, no new bytes."""
    return dict(est, shared=True)


def timeline(panels):
    """A guard's timeline image at its smallest (one flag row)."""
    return image(1, TIMELINE_BASE + TIMELINE_PANEL * panels, TIMELINE_W)


# -- loaders --------------------------------------------------------------------------------------

def _frames_after_selection(meta, force_rate, cap, skip, nth):
    frames = meta["frames"] * force_rate / meta["fps"] if force_rate else meta["frames"]
    frames = int(frames) - skip
    frames //= max(nth, 1)
    return min(frames, cap) if cap else frames


def _whole_clip_loader(c):
    meta = c.env.video_meta(c.widget("video"))
    if c.scenario:
        n, w, h = c.scenario["frames"], c.scenario["w"], c.scenario["h"]
    elif meta is None:
        raise NotCounted("the video file cannot be read for its size")
    else:
        n = _frames_after_selection(meta, c.widget("force_rate", 0), c.widget("frame_load_cap", 0),
                                    c.widget("skip_first_frames", 0), c.widget("select_every_nth", 1))
        w, h, cw, ch = meta["width"], meta["height"], c.widget("custom_width", 0), c.widget("custom_height", 0)
        if cw or ch:  # the custom size is rounded to a multiple of 8
            w, h = (cw or w * ch / h), (ch or h * cw / w)
            w, h = int(w / 8 + 0.5) * 8, int(h / 8 + 0.5) * 8
    img = image(n, h, w)
    # np.fromiter without a count grows its buffer by ~1.5x: up to half the clip again before the
    # final shrink (no copy where realloc can remap, as glibc does for large blocks).
    return {c.slot("IMAGE"): img}, img["bytes"] // 2, "np.fromiter growth, upper bound 0.5 x output"


def _bcv_load_video(c):
    options = c.env.input_types("BCVLoadVideo")["required"]["resolution"][1]
    step = options.get("bcv_frames", {}).get(c.widget("model"), 4)
    if c.scenario:
        n, w, h = c.scenario["frames"], c.scenario["w"], c.scenario["h"]
    else:
        meta = c.env.video_meta(c.widget("video"))
        if meta is None:
            raise NotCounted("the video file cannot be read for its size")
        w, h = options["bcv_sizes"][c.widget("model")][c.widget("resolution")]
        orientation = c.widget("orientation", "auto")
        if orientation == "landscape" or (orientation == "auto" and meta["height"] <= meta["width"]):
            w, h = h, w
        fps = float(c.widget("force_fps") or meta["fps"])
        count = str(c.widget("frame_count") or "").strip()
        n = int(count) if count else int(meta["frames"] * fps / meta["fps"]) - int(c.widget("start_frame", 1)) + 1
        n = (n - 1) // step * step + 1
    img = image(n, h, w)
    info = {"type": "BCV_VIDEO_INFO", "shape": [], "n": n, "h": h, "w": w, "bytes": 0}
    return {c.slot("IMAGE"): img, c.slot("BCV_VIDEO_INFO"): info}, 0, "one frame at a time into a preallocated output"


def _load_image(c):
    size = c.env.image_size(c.widget("image"))
    if size is None:
        raise NotCounted("the image file cannot be read for its size")
    w, h = size
    return {0: image(1, h, w), 1: mask(1, h, w)}, 0, "one image"


def _bcv_load_reference(c):
    info = c.tensor("video_info")
    return {0: image(1, info["h"], info["w"]), 1: mask(1, info["h"], info["w"])}, 0, "one image at the video's size"


# -- resizers -------------------------------------------------------------------------------------

def _resized(c):
    img = c.tensor("image")
    h, w = resize_plan(img["h"], img["w"], c.widget("width", 0), c.widget("height", 0), c.widget("keep_proportion", "stretch"),
                       c.widget("crop_position", "center"), c.widget("divisible_by", 0),
                       c.widget("upscale_method", "bilinear")).output
    out = {0: image(img["n"], h, w)}
    m = c.tensor("mask", required=False)
    if m is not None:
        out[3] = mask(m["n"], h, w)
    return out


def _resize_listed_then_cat(c):
    out = _resized(c)
    # 64-frame sub-batches listed, then torch.cat: the list is a second copy of the output
    return out, sum(t["bytes"] for t in out.values()), "sub-batch list + torch.cat: 1 x output"


def _resize_preallocated(c):
    return _resized(c), 0, "one frame at a time into a preallocated output"


def _image_scale(c):
    img = c.tensor("image")
    w, h = c.widget("width", 0), c.widget("height", 0)
    w, h = w or round(img["w"] * h / img["h"]), h or round(img["h"] * w / img["w"])
    return {0: image(img["n"], h, w)}, None, "transient not profiled"


def _conform_video(c):
    img = c.tensor("images")
    short, portrait = min(img["h"], img["w"]), img["h"] > img["w"]
    target = min(((480, 854), (720, 1280), (1080, 1920)), key=lambda t: abs(math.log(t[0] / short)))
    w, h = target if portrait else target[::-1]
    if (h, w) == (img["h"], img["w"]):
        return {0: shared(img)}, 0, "already at its size: passed on untouched"
    return {0: image(img["n"], h, w, img.get("dtype_bytes", F32))}, 0, "one frame at a time into a preallocated output"


def _scale_by_aspect_ratio(c):
    img, m = c.tensor("image", required=False), c.tensor("mask", required=False)
    src = img or m
    if src is None:
        raise NotCounted("neither image nor mask has an estimate")
    ratio = aspect_ratio(c.widget("aspect_ratio", "original"), src["w"], src["h"], c.widget("proportional_width", 1),
                         c.widget("proportional_height", 1))
    w, h = target_size(src["w"], src["h"], ratio, c.widget("scale_to_side", "None"), c.widget("scale_to_length", 1024))
    multiple = c.widget("round_to_multiple", "None")
    if multiple != "None":
        w, h = round_up_to_multiple(w, int(multiple)), round_up_to_multiple(h, int(multiple))
    # without an image the image output is None; without a mask the mask is zeros, one per frame
    out = {1: mask(m["n"] if m else img["n"], h, w)}
    if img:
        out[0] = image(img["n"], h, w)
    # per-frame PIL results listed, then torch.stack: the lists are a second copy of each output
    listed = (out[0]["bytes"] if img else 0) + (out[1]["bytes"] if m else 0)
    return out, listed, "frame list + torch.stack: 1 x output"


# -- masks ----------------------------------------------------------------------------------------

def _same_mask(note, transient_factor=0, name="mask"):
    """A mask node whose output is its input mask's size; `name` is its mask input."""
    def profile(c):
        m = c.tensor(name)
        return {0: m}, m["bytes"] * transient_factor, note
    return profile


def _repeat_mask(widget):
    def profile(c):
        m = c.tensor("mask")
        # repeat, or one cat over n references to the same mask: no copy besides the output
        return {0: mask(m["n"] * int(c.widget(widget, 1)), m["h"], m["w"])}, 0, "output only"
    return profile


def _draw_mask_cloned(c):
    img, m = c.tensor("image"), c.tensor("mask")
    # clones of image and mask, a per-frame list, then torch.stack into the output
    return {0: img}, img["bytes"] * 2 + m["bytes"], "clones of image and mask + frame list"


def _draw_mask_preallocated(c):
    return {0: c.tensor("image")}, 0, "one frame at a time into a preallocated output"


# -- Wan and core sampling ------------------------------------------------------------------------

def _wan_to_video(c):
    n, h, w = int(c.widget("length", 81)), int(c.widget("height", 480)), int(c.widget("width", 832))
    return {c.slot("LATENT"): wan_latent(n, h, w)}, None, "conditioning outputs not counted"


def _sampler(c):
    latent = c.tensor("latent_image")
    return {i: dict(latent) for i, t in enumerate(c.types) if t == "LATENT"}, None, "activations (VRAM) not counted"


def _vae_decode(c):
    latent = c.tensor("samples")
    return {0: image(latent["n"], latent["h"], latent["w"])}, None, "decode working set not counted"


def _long_video(held):
    """The long-video samplers' chunk loop: the output preallocated at total_frames, each chunk's
    decoded frames until they are copied in, and, when a chunk reads past a driving video's end,
    that video's window gathered for the chunk (counted for every held video: an upper bound). The
    anchor the next chunk is seeded with is a view of the output."""
    def profile(c):
        pose = c.tensor("pose_video")
        total = int(c.widget("total_frames", 0)) or pose["n"]
        chunk = max(5, (int(c.widget("frames_per_chunk", 81)) - 1) // 4 * 4 + 1)
        h, w = int(c.widget("height")) // 8 * 8, int(c.widget("width")) // 8 * 8
        out = image(total, h, w)
        # one chunk: the output is a view of its decoded frames, so only the frames past total add
        decoded = image_bytes(chunk - total if total <= chunk else chunk, h, w)
        windows = sum(v["bytes"] // max(v["n"], 1) * chunk for v in (c.tensor(name, required=False) for name in held) if v)
        return ({0: out}, decoded + windows,
                "decoded chunk + driving-video windows (upper bound); the core node's conditioning and the VAE / "
                "sampler working sets not counted")
    return profile


# -- the preprocess (pose, SAM 3.1 Multiplex, face, SCAIL-2) --------------------------------------

def _pose(c):
    img = c.tensor("images")
    return {0: image(img["n"], img["h"], img["w"])}, 0, "drawn frame by frame into a preallocated output"


def _sam_track(c):
    img = c.tensor("images")
    return {0: mask(img["n"], img["h"], img["w"])}, 0, "frames read as a view; SAM 3.1's own working set not counted"


def _face_crop(c):
    img = c.tensor("images")
    return ({0: image(img["n"], FACE_SIZE, FACE_SIZE, img.get("dtype_bytes", F32))}, 0,
            "512 x 512 crops into a preallocated output")


def _wan_animate_preprocess(c):
    img = c.tensor("images")
    n, h, w = img["n"], img["h"], img["w"]
    return ({0: image(n, h, w), 1: image(n, FACE_SIZE, FACE_SIZE, img.get("dtype_bytes", F32)), 2: mask(n, h, w)}, 0,
            "pose, mask and face crops into preallocated outputs; SAM 3.1's own working set not counted")


def _reference_size(c):
    """(n, h, w) of the reference mask SCAIL-2 renders: the connected reference_mask's, else the
    reference image's (SAM 3.1 tracks it at its size)."""
    ref = c.tensor("reference_mask", required=False) or c.tensor("reference_image")
    return ref["n"], ref["h"], ref["w"]


def _scail2_preprocess(c):
    img = c.tensor("images")
    n, h, w = img["n"], img["h"], img["w"]
    rn, rh, rw = _reference_size(c)
    black = bool(c.widget("black_background", False))
    out = {0: image(n, h, w, img.get("dtype_bytes", F32)) if black else shared(img), 1: image(n, h, w),
           2: image(rn, rh, rw), 3: mask(n, h, w),
           4: shared(c.tensor("reference_mask")) if c.linked("reference_mask") else mask(rn, rh, rw)}
    # in the pose modes the pose images are drawn and dropped: only pose_data is read
    pose_images = image_bytes(n, h, w) if c.widget("mode", "prompt") != "prompt" else 0
    return out, pose_images, "the pose modes' discarded pose images; SAM 3.1's own working set not counted"


def _scail2_colored_mask(c):
    driving = c.tensor("driving_mask")
    ref = c.tensor("reference_mask", required=False)
    n, h, w = (ref["n"], ref["h"], ref["w"]) if ref else (1, driving["h"], driving["w"])
    chunk = mask_bytes(min(RENDER_CHUNK, driving["n"]), driving["h"], driving["w"], 1)  # the boolean cut
    return {0: image(driving["n"], driving["h"], driving["w"]), 1: image(n, h, w)}, chunk, "16 frames at a time"


def _pose_guard(c):
    return {3: timeline(2)}, 0, "pose_data passed on; the timeline at one flag row"


def _mask_guard(panels):
    def profile(c):
        m = c.tensor("mask")
        window = mask_bytes(MASK_GUARD_WINDOW, m["h"], m["w"], 1)
        slot = c.slot("IMAGE")
        return {0: shared(m), slot: timeline(panels)}, window, "the mask passed on; a 5-frame boolean window"
    return profile


def _scail2_guard(c):
    driving, ref = c.tensor("pose_video_mask"), c.tensor("reference_image_mask")
    person = mask_bytes(driving["n"], driving["h"], driving["w"], 1)  # the driving person as booleans, whole clip
    return ({0: shared(driving), 1: shared(ref), 4: timeline(2)}, person,
            "the masks passed on; the driving person as a whole-clip boolean array")


def _per_frame_output(c):
    """An output node that writes frame by frame; an IMAGE it returns is its input passed on."""
    images = next((c.tensor(name) for name in ("images", "image") if c.linked(name)), None)
    out = {0: shared(images)} if images is not None and c.types[:1] == ["IMAGE"] else {}
    return out, 0, "writes frame by frame"


# -- SeedVR2 (upscale) ----------------------------------------------------------------------------

def _shortest_edge(h, w, resolution, max_resolution):
    """The size SeedVR2 Resize's resize gives: the short edge to `resolution` (the long one floored),
    then fitted inside max_resolution."""
    if h <= w:
        h, w = resolution, int(resolution * w / h)
    else:
        h, w = int(resolution * h / w), resolution
    if max_resolution > 0 and max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        h, w = round(h * scale), round(w * scale)
    return h, w


def _seedvr2_resize(c):
    img = c.tensor("image")
    n, h, w = img["n"], img["h"], img["w"]
    resolution = int(round(min(h, w) * c.widget("upscale_factor", 2.0)))
    down = c.widget("downscale_factor", 0.5)
    if down != 1.0:
        h_d, w_d = round(h * down), round(w * down)
        # lanczos per frame through PIL, listed, then torch.stack: the list and the stack together
        listed = 2 * image_bytes(n, h_d, w_d)
    else:
        h_d, w_d, listed = h, w, 0
    rh, rw = _shortest_edge(h_d, w_d, resolution, c.widget("max_resolution", 4096))
    extra = 0 if n == 1 else (4 - n + 1 if n <= 4 else (4 - (n - 1) % 4) % 4)
    padded = image(n + extra, round_up_to_multiple(rh, SEEDVR2_PAD), round_up_to_multiple(rw, SEEDVR2_PAD), F16)
    reference = image(n, rh // 2 * 2, rw // 2 * 2, F16)
    return {0: padded, 1: reference}, listed, "float16 outputs; the downscale's frame list + stack"


def _seedvr2_encode(c):
    pixels = c.tensor("pixels")
    n, h, w = pixels["n"], pixels["h"], pixels["w"]
    lt, lh, lw = (n + 3) // 4, (h + 7) // 8, (w + 7) // 8
    elements = SEEDVR2_LATENT_CHANNELS * lt * lh * lw
    tile = int(c.widget("tile_size", 1024))
    single = h <= tile and w <= tile
    # one tile: the slices listed and concatenated in the VAE dtype (2 bytes); tiles: a float32 sum
    # and its cast to the VAE dtype
    transient = elements * (4 if single else 6)
    latent = {"type": "LATENT", "shape": [1, SEEDVR2_LATENT_CHANNELS, lt, lh, lw], "lt": lt, "lh": lh, "lw": lw,
              "bytes": elements * F32}
    return {0: latent}, transient, "the encoder's working set (device) not counted"


def _seedvr2_decode(c):
    latent = c.tensor("samples")
    if "lt" not in latent:
        raise NotCounted("samples is not a SeedVR2 latent estimate")
    frames, h, w = max(1, latent["lt"] * 4 - 3), latent["lh"] * 8, latent["lw"] * 8
    tile_latent = max(1, int(c.widget("tile_size", 1024)) // 8)
    tiled = latent["lh"] > tile_latent or latent["lw"] > tile_latent
    out = image(frames, h // 2 * 2, w // 2 * 2, F16)
    # tiles are summed into a float16 buffer of the whole decoded clip before the output
    return {0: out}, image_bytes(frames, h, w, F16) if tiled else 0, "decoded frames streamed into a float16 output"


def _seedvr2_postprocess(c):
    images, ref = c.tensor("images"), c.tensor("original_resized_images")
    t, h, w = min(images["n"], ref["n"]), min(images["h"], ref["h"]), min(images["w"], ref["w"])
    return {0: image(t, h // 2 * 2, w // 2 * 2, F16)}, 0, "one frame at a time into a float16 output"


# -- BCNodes image nodes --------------------------------------------------------------------------

def _birefnet(c):
    img = c.tensor("image")
    n, h, w = img["n"], img["h"], img["w"]
    alpha = c.widget("background", "Alpha") == "Alpha"
    one = mask_bytes(n, h, w)
    # the per-frame mattes listed before torch.stack; the raw matte kept while an option makes a new one
    options = (c.widget("sensitivity", 1.0) < 1.0 or c.widget("mask_blur", 0) > 0 or c.widget("mask_offset", 0) != 0
               or bool(c.widget("invert_output", False)))
    transient = one + (one if options else 0)
    transient += image_bytes(n, h, w) if c.widget("refine_foreground", False) else 0
    transient += 0 if alpha else 3 * image_bytes(n, h, w)  # the "over" blend's full-size temporaries
    out = {0: image(n, h, w, channels=4 if alpha else 3), 1: mask(n, h, w), 2: image(n, h, w)}
    return out, transient, "frame list + option and blend temporaries; the network's working set not counted"


def _depth_anything(c):
    img = c.tensor("image")
    depth = mask_bytes(img["n"], img["h"], img["w"])
    # the frame list, its stack and normalize's temporaries reach 5 x the depth before the 3-channel output
    return {0: image(img["n"], img["h"], img["w"])}, 2 * depth, "frame list, stack and normalize temporaries"


def _postfx_apply(c):
    img = c.tensor("image")
    if c.widget("theme") == "none" and not c.linked("look"):
        return {0: shared(img)}, 0, "no look: the input passed on"
    out = image(img["n"], img["h"], img["w"])
    # the processed frames listed, their clipped copies listed, then np.stack
    return {0: out}, 2 * out["bytes"], "frame list + clipped copies before np.stack"


def _skin_texture(c):
    img = c.tensor("image")
    return {0: image(img["n"], img["h"], img["w"]), 1: mask(img["n"], img["h"], img["w"])}, None, \
        "SAM 3 detections and the texture's working set not counted"


def _any_switch(c):
    names = sorted((n for n in c.inputs if n.startswith("any_") and c.linked(n)), key=lambda n: int(n[4:] or 0))
    for name in names:
        est = c.tensor(name, required=False)
        if est is not None:
            return {0: shared(est)}, 0, f"passes on {name}"
    raise NotCounted("no connected input has an estimate")


def _select_switch(c):
    selected = c.widget("selected")
    est = c.tensor(selected, required=False) if isinstance(selected, str) else None
    if est is None:
        raise NotCounted(f"the selected input {selected!r} has no estimate")
    return {0: shared(est)}, 0, f"passes on {selected}"


def _join_lists(c):
    return {}, 0, "a list of references to its inputs, no copy (downstream sizes are not followed through lists)"


PROFILES = {
    # loaders and resizers
    "VHS_LoadVideo": _whole_clip_loader, "VHS_LoadVideoPath": _whole_clip_loader, "BCVLoadVideo": _bcv_load_video,
    "LoadImage": _load_image, "BCVLoadReferenceImage": _bcv_load_reference,
    "ImageResizeKJv2": _resize_listed_then_cat, "BC_ImageResize": _resize_preallocated, "ImageScale": _image_scale,
    "BCVConformVideo": _conform_video, "BC_ImageScaleByAspectRatio": _scale_by_aspect_ratio,
    # masks
    "BC_MaskGrow": _same_mask("preallocated output"), "BC_MaskFillHoles": _same_mask("preallocated output", name="masks"),
    "BC_BlockifyMask": _same_mask("preallocated output", name="masks"),
    "BlockifyMask": _same_mask("zeros_like work buffer + clamp copy: 1 x mask", 1, name="masks"),
    "BC_RepeatMaskBatch": _repeat_mask("amount"), "VHS_DuplicateMasks": _repeat_mask("multiply_by"),
    "DrawMaskOnImage": _draw_mask_cloned, "BC_DrawMaskOnImage": _draw_mask_preallocated,
    # Wan and core sampling
    "WanAnimateToVideo": _wan_to_video, "WanImageToVideo": _wan_to_video,
    "KSampler": _sampler, "KSamplerAdvanced": _sampler, "VAEDecode": _vae_decode, "VAEDecodeTiled": _vae_decode,
    "BCVWanAnimateLongVideoSampler": _long_video(("pose_video", "face_video", "background_video")),
    "BCVWanAnimate2LongVideoSampler": _long_video(("pose_video", "face_video", "background_video")),
    "BCVSCAIL2LongVideoSampler": _long_video(("pose_video", "pose_video_mask")),
    # the preprocess
    "BCVPoseDetection": _pose, "BCVSapiens2Pose": _pose, "BCVSAM3VideoTrack": _sam_track, "BCVFaceCrop": _face_crop,
    "BCVWanAnimatePreprocess": _wan_animate_preprocess, "BCVSCAIL2Preprocess": _scail2_preprocess,
    "BCVSCAIL2ColoredMask": _scail2_colored_mask,
    "BCVPoseGuard": _pose_guard, "BCVMaskGuard": _mask_guard(3), "BCVWanAnimatePreprocessGuard": _mask_guard(4),
    "BCVSCAIL2PreprocessGuard": _scail2_guard,
    # outputs
    "VHS_VideoCombine": _per_frame_output, "BCVSaveVideo": _per_frame_output, "BCVVideoComparer": _per_frame_output,
    "SaveImage": _per_frame_output, "PreviewImage": _per_frame_output,
    # SeedVR2
    "BC_SeedVR2Resize": _seedvr2_resize, "BC_SeedVR2VAEEncode": _seedvr2_encode, "BC_SeedVR2VAEDecode": _seedvr2_decode,
    "BC_SeedVR2PostProcess": _seedvr2_postprocess,
    # BCNodes image nodes and switches
    "BC_BiRefNetRemoveBackground": _birefnet, "BC_DepthAnythingV2": _depth_anything, "BC_PostFxApply": _postfx_apply,
    "BC_SkinTexture": _skin_texture, "BC_AnySwitch": _any_switch, "BC_SelectSwitch": _select_switch,
    "BC_JoinImageLists": _join_lists,
}

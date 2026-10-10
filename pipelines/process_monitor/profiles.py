"""Emulate's per-node-type cost profiles: what a node creates (its outputs, exact from sizes) and
what it holds besides them while it runs (its transient), each derived from that node's code.

A profile takes a Ctx and returns ({output slot: estimate}, transient bytes or None, note). An
estimate is a dict: "type", "shape", "bytes", and the dims later profiles read ("n", "h", "w" for
frames; "lt", "lh", "lw" for a SeedVR2 latent). "shared": True marks an output that is one of the
node's inputs passed on: it adds no bytes. A transient of None is "not counted"; the note says
which part of the node's memory the profile does not size (a model's own working set, mostly).
A profile raises NotCounted when an input it needs has no estimate, or when the prompt does not say
what the node holds (Qwen LM: its model is a catalog name, not a file).
"""

import math

from ...libs.geometry import aspect_ratio, round_up_to_multiple, short_side_size, target_size
from ...libs.resize import plan as resize_plan

F32, F16 = 4, 2
FACE_SIZE = 512  # Face Crop's square
SEEDVR2_LATENT_CHANNELS = 16
SEEDVR2_PAD = 16
SEEDVR2_CHUNK = 4  # the frames the SeedVR2 flows handle at a time (pipelines/seedvr2 FRAMES_PER_CHUNK)
# a guard's timeline image: 1200 wide, 128 + 190 per panel high with one flag row (+24 per more flag)
TIMELINE_W, TIMELINE_BASE, TIMELINE_PANEL = 1200, 128, 190
MASK_GUARD_WINDOW = 5  # frames of booleans the mask guard keeps (the frame and 2 either side)
RENDER_CHUNK = 16  # frames the colored masks are cut at a time
SAM3_SIDE = 1008  # core SAM3_Detect scales its input to 1008 x 1008
FRAMING_CHUNK = 4  # frames SeedVR2 Framing Downscale hands SAM 3 at a time (pipelines/seedvr2 FRAMES_PER_CHUNK)
# BCVLoadVideo's model rules: its frame counts are step * n + 1, and with resolution "source" each side
# is cut down to a multiple of the grid. Its sizes come from its own node definition (bcv_sizes).
LOAD_VIDEO_MODELS = {"Wan": {"step": 4, "grid": 16}, "SCAIL": {"step": 4, "grid": 32}, "None": {"step": 1, "grid": 1}}
# BCVLoadVideo's precision widget: the dtype its frames are stored in. A workflow saved before the
# widget gets its default, fp16. The clips BCVideoNodes makes from a half clip keep its dtype.
LOAD_VIDEO_PRECISIONS = {"fp32": F32, "fp16": F16}
MIN_CHUNK, MIN_LAST_CHUNK = 5, 29  # the long-video samplers' shortest chunk, and last_chunk min29's last one


class NotCounted(Exception):
    pass


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


class Ctx:
    """One node as a profile sees it: its widget values, its upstream estimates, the scenario, and
    which of its output slots some node of the prompt links (`consumed`)."""

    def __init__(self, node, upstream, scenario, env, consumed):
        self.inputs = node.get("inputs", {})
        self.upstream = upstream
        self.scenario = scenario
        self.env = env
        self.types = env.return_types(node["class_type"])
        self.consumed = consumed

    def widget(self, name, default=None):
        value = self.inputs.get(name, default)
        return default if is_link(value) else value

    def linked(self, name):
        return is_link(self.inputs.get(name))

    def output_linked(self, slot):
        return slot in self.consumed

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


def dtype_of(est):
    """The bytes per value of an IMAGE or MASK estimate (float32 unless it says otherwise)."""
    return est.get("dtype_bytes", F32)


def widened(est):
    """A half-precision IMAGE or MASK estimate's float32 copy, in bytes; 0 for a float32 one."""
    return est["bytes"] * F32 // F16 if dtype_of(est) == F16 else 0


def given_size(c, what):
    """(width, height) from a node's optional width and height inputs, None when neither is given.
    Raises NotCounted when they come from links (`what`'s size is known at run time only) or only
    one is given (the node stops with an error)."""
    if c.linked("width") or c.linked("height"):
        raise NotCounted(f"width / height come from links: {what} size is known at run time only")
    width, height = c.widget("width"), c.widget("height")
    if (width is None) != (height is None):
        raise NotCounted("only one of width / height is set: the node stops with an error")
    return None if width is None else (width, height)


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


def _source_size(meta, portrait, grid):
    """BCVLoadVideo's resolution "source": the video's own size; in the other orientation its centred
    crop to that orientation's aspect, the short side kept (1920x1080 -> 608x1080); then each side
    cut down to the grid."""
    w, h = meta["width"], meta["height"]
    if portrait != (h > w):
        aspect, target = w / h, h / w
        w, h = (round(h * target), h) if aspect > target else (w, round(w / target))
    return w // grid * grid, h // grid * grid


def _bcv_load_video(c):
    dtype = LOAD_VIDEO_PRECISIONS[c.widget("precision", "fp16")]
    if c.scenario:
        n, w, h = c.scenario["frames"], c.scenario["w"], c.scenario["h"]
    else:
        model = LOAD_VIDEO_MODELS[c.widget("model")]
        meta = c.env.video_meta(c.widget("video"))
        if meta is None:
            raise NotCounted("the video file cannot be read for its size")
        orientation = c.widget("orientation", "auto")
        portrait = meta["height"] > meta["width"] if orientation == "auto" else orientation == "portrait"
        options = c.env.input_types("BCVLoadVideo")["required"]["resolution"][1]
        size = options["bcv_sizes"][c.widget("model")][c.widget("resolution")]
        if size is None:
            w, h = _source_size(meta, portrait, model["grid"])
        else:
            w, h = size if portrait else size[::-1]
        fps = float(c.widget("force_fps") or meta["fps"])
        count = str(c.widget("frame_count") or "").strip()
        n = int(count) if count else int(meta["frames"] * fps / meta["fps"]) - int(c.widget("start_frame", 1)) + 1
        n = (n - 1) // model["step"] * model["step"] + 1
    img = image(n, h, w, dtype)
    out = {c.slot("BCV_VIDEO_INFO"): {"type": "BCV_VIDEO_INFO", "shape": [], "n": n, "h": h, "w": w, "bytes": 0}}
    note = "one frame at a time into a preallocated output; the audio waveform not counted"
    if not c.output_linked(c.slot("IMAGE")):
        # video_info counts the decoded frames: the clip is loaded all the same and dropped at return
        return out, img["bytes"], "images not linked: loaded, then dropped at return; " + note
    out[c.slot("IMAGE")] = img
    return out, 0, note


def _bcv_get_video_info(c):
    return {}, 0, "numbers, and Load Video's audio output itself (the same object: no new bytes)"


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

def _resized(c, keep_dtype=False):
    """The resize's outputs; `keep_dtype`: each in its input's dtype (BC_ImageResize), else float32."""
    img = c.tensor("image")
    plan = resize_plan(img["h"], img["w"], c.widget("width", 0), c.widget("height", 0), c.widget("keep_proportion", "stretch"),
                       c.widget("crop_position", "center"), c.widget("divisible_by", 0), c.widget("upscale_method", "bilinear"))
    h, w = plan.output
    out = {0: image(img["n"], h, w, dtype_of(img) if keep_dtype else F32)}
    m = c.tensor("mask", required=False)
    if m is not None:
        # BC_ImageResize: a pillarbox_blur pad gives the mask the image's dtype
        source = img if c.widget("keep_proportion", "stretch") == "pillarbox_blur" and plan.pad else m
        out[3] = mask(m["n"], h, w, dtype_of(source) if keep_dtype else F32)
    return out


def _resize_listed_then_cat(c):
    out = _resized(c)
    # 64-frame sub-batches listed, then torch.cat: the list is a second copy of the output
    return out, sum(t["bytes"] for t in out.values()), "sub-batch list + torch.cat: 1 x output"


def _resize_preallocated(c):
    out, img = _resized(c, keep_dtype=True), c.tensor("image")
    # a half frame is read as float32 and resampled / padded in a float32 frame before its output gets it
    work = image_bytes(1, img["h"], img["w"]) + image_bytes(1, out[0]["h"], out[0]["w"]) if dtype_of(img) == F16 else 0
    return out, work, "one frame at a time into a preallocated output of the input's dtype"


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
    # without an image the image output is None; without a mask the mask is zeros, one per frame, in the
    # image's dtype; each output in its input's dtype (fitted through 8-bit PIL)
    out = {1: mask(m["n"] if m else img["n"], h, w, dtype_of(m or img))}
    if img:
        out[0] = image(img["n"], h, w, dtype_of(img))
    return out, 0, "one frame at a time (8-bit PIL) into preallocated outputs"


# -- masks ----------------------------------------------------------------------------------------

def _same_mask(note, transient_factor=0, name="mask"):
    """A mask node whose output is its input mask's size; `name` is its mask input."""
    def profile(c):
        m = c.tensor(name)
        return {0: m}, m["bytes"] * transient_factor, note
    return profile


def _frame_mask(name="mask", half_frames=1):
    """A BCNodes mask node: a mask of its input mask's size and dtype (`name`; float32 unless half),
    frame by frame into a preallocated output; a half mask's node holds `half_frames` float32 frames
    (its frame read as float32, and what the node makes of it before the output gets it)."""
    def profile(c):
        m = c.tensor(name)
        frames = half_frames if dtype_of(m) == F16 else 0
        return ({0: mask(m["n"], m["h"], m["w"], dtype_of(m))}, frames * mask_bytes(1, m["h"], m["w"]),
                "one frame at a time into a preallocated output of the mask's dtype")
    return profile


def _repeat_mask(widget):
    def profile(c):
        m = c.tensor("mask")
        # repeat, or one cat over n references to the same mask, in the mask's dtype: no copy besides the output
        return {0: mask(m["n"] * int(c.widget(widget, 1)), m["h"], m["w"], dtype_of(m))}, 0, "output only"
    return profile


def _draw_mask_cloned(c):
    img, m = c.tensor("image"), c.tensor("mask")
    # clones of image and mask, a per-frame list, then torch.stack into the output
    return {0: img}, img["bytes"] * 2 + m["bytes"], "clones of image and mask + frame list"


def _draw_mask_preallocated(c):
    img, m = c.tensor("image"), c.tensor("mask")
    channels = img["shape"][-1]
    # in the image's dtype; per frame in float32: a half image's frame read as float32 (libs/image.float_frame)
    # and blended into a float32 frame copied into the output, a half mask's frame read as float32
    work = 2 * image_bytes(1, img["h"], img["w"], channels=channels) if dtype_of(img) == F16 else 0
    work += mask_bytes(1, m["h"], m["w"]) if dtype_of(m) == F16 else 0
    return ({0: image(img["n"], img["h"], img["w"], dtype_of(img), channels=channels)}, work,
            "one frame at a time into a preallocated output of the image's dtype")


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


def snap_up(frames):
    """The smallest 4k+1 >= frames, never below MIN_CHUNK (a Wan chunk length)."""
    return MIN_CHUNK if frames < MIN_CHUNK else -(-(frames - 1) // 4) * 4 + 1


def snap_down(frames):
    """The largest 4k+1 <= frames, never below MIN_CHUNK."""
    return MIN_CHUNK if frames < MIN_CHUNK else (frames - 1) // 4 * 4 + 1


def kept_frames(motion):
    """The frames a chained chunk keeps of the one before and trims back off (the overlap), for a
    continuation of `motion` frames: the latent frames they fill, back in pixel frames."""
    return 0 if motion <= 0 else ((motion - 1) // 4 + 1) * 4 - 3


def chunk_lengths(total, chunk, overlap, last_chunk):
    """The long-video samplers' chunk plan: the chunk lengths that cover `total` frames, every
    chained chunk adding its length less `overlap`. last_chunk "fit": the last chunk fitted to the
    frames still needed (snapped up to 4k+1); "full": every chunk `chunk` long; "min29": "fit", the
    last chunk never under 29 frames."""
    if last_chunk not in ("fit", "full", "min29"):
        raise ValueError(f"last_chunk {last_chunk!r}")
    if overlap >= chunk:
        raise ValueError(f"frames_per_chunk ({chunk} on the 4k+1 grid) must exceed the overlap ({overlap})")
    lengths, produced = [], 0
    while produced < total:
        length = chunk if last_chunk == "full" else min(chunk, snap_up(total - produced + (overlap if lengths else 0)))
        if last_chunk == "min29":
            length = min(chunk, max(length, MIN_LAST_CHUNK))
        lengths.append(length)
        produced += length - (overlap if len(lengths) > 1 else 0)
    return lengths


def chunk_windows(lengths, overlap, total, seeked=()):
    """Per chunk of the plan `lengths`: (its length, the frames produced before it, the frames it
    decodes, start, stop), start..stop the window of the driving videos it reads when it reads one.
    The first chunk reads from frame 0; the second from the frame its seed (the whole first chunk)
    starts at, through its own frames: the loop does not know yet how far core moves the offset
    back; later chunks from `overlap` frames before their own. A start is moved back while a
    seeked video (`seeked`: their frame counts) would keep one frame of the window. The last chunk
    decodes only the frames total_frames needs (snapped up to 4k+1)."""
    produced = 0
    for k, length in enumerate(lengths):
        trim = overlap if k else 0
        decoded = min(length, snap_up(trim + total - produced))
        start = max(0, produced - overlap) if k > 1 else 0
        stop = start + length if k > 1 else produced + length
        while any(frames - start == 1 for frames in seeked):
            start -= 1
        yield length, produced, decoded, start, stop
        produced += decoded - trim


def _long_video(held, seeked=(), painted=None, overlap=None, last_chunk="fit", whole=()):
    """A long-video sampler's chunk loop. `held`: the driving videos it extends past their end
    (a gather of the frames past it); `seeked`: the other videos it cuts to a chunk's window;
    `painted`: the (video, mask) it paints black in replacement mode, both connected; `overlap`: the
    widget of the frames a chained chunk keeps of the one before (None: one, WanAnimate2ToVideo's);
    `whole`: the other IMAGE / MASK inputs the core node gets whole.

    The output: total_frames at the generation size, in the pose video's dtype when that is half,
    else float32. Per chunk, besides it: its decoded frames (float32); and on a chunk that reads
    past a held video's end, or on every chunk when a video is half precision or one is painted,
    each video's window (chunk_windows): a held one gathered past its end, the painted one painted
    into a new window (its mask gathered past its end while it is), a half one requantized into a
    float32 window; the last held video's half window stays referenced through the chunk and into
    the next. With a half output the seed is kept apart as decoded: chunk 1's decoded frames
    through chunk 2, then a new chunk of them, made while the one before still lives. A half
    input the core node gets whole is requantized once for the run."""
    def profile(c):
        pose = c.tensor("pose_video")
        total = int(c.widget("total_frames", 0)) or pose["n"]
        chunk = snap_down(int(c.widget("frames_per_chunk", 81)))
        keep = kept_frames(int(c.widget(overlap, 5))) if overlap else 1
        lengths = chunk_lengths(total, chunk, keep, c.widget("last_chunk", last_chunk))
        h, w = int(c.widget("height")) // 8 * 8, int(c.widget("width")) // 8 * 8
        frame = image_bytes(1, h, w)
        videos = {name: pose if name == "pose_video" else c.tensor(name) for name in held
                  if name == "pose_video" or c.linked(name)}
        seeks = {name: c.tensor(name) for name in seeked if c.linked(name)}
        cut = {name: v for name, v in seeks.items() if v["n"] > 1}  # a single frame is handed over whole
        once = sum(widened(v) for name, v in seeks.items() if name not in cut)
        unknown = []
        for name in ("reference_image", *whole):
            v = c.tensor(name, required=False)
            once += widened(v) if v else 0
            unknown += [name] if c.linked(name) and v is None else []
        paint = painted if painted and all(c.linked(name) for name in painted) else None
        half = any(dtype_of(v) == F16 for v in (*videos.values(), *cut.values()))
        half_out = dtype_of(pose) == F16
        reach = max(total, lengths[0] + sum(length - keep for length in lengths[1:]))
        short = [v["n"] for v in videos.values() if v["n"] < reach]
        peak = local = 0  # local: the last held video's half window, referenced until the next is cut
        decoded_before = 0
        for k, (length, produced, decoded, start, stop) in enumerate(
                chunk_windows(lengths, keep, total, [v["n"] for v in cut.values()])):
            seed = 0 if not half_out or k == 0 else (decoded_before if k == 1 else chunk) * frame
            moments, inputs = [], 0
            if half or paint or any(produced + length > frames for frames in short):
                for name, v in videos.items():
                    per = v["bytes"] // v["n"]
                    reached = min(stop, max(reach, v["n"]))
                    window = (reached - start) * per if reached > v["n"] else 0  # a view inside the video
                    moments.append(inputs + local + window)
                    if paint and name == paint[0]:
                        m = seeks[paint[1]]
                        # a single frame (already requantized whole) is repeated over the video
                        source, per_mask = (m["n"], m["bytes"] // m["n"]) if m["n"] > 1 else (v["n"], widened(m) or m["bytes"])
                        gathered = (reached - start) * per_mask if reached > source else 0
                        moments.append(inputs + window + gathered + (reached - start) * per)
                        window = (reached - start) * per
                    wide = (reached - start) * per * F32 // F16 if dtype_of(v) == F16 else 0
                    moments.append(inputs + window + wide)
                    inputs += wide or window
                    local = window if wide else 0
                for v in cut.values():
                    if dtype_of(v) == F16:
                        inputs += max(0, min(stop, v["n"]) - start) * (v["bytes"] // v["n"]) * F32 // F16
            # one chunk at float32: the output is a view of its decoded frames, so only those past total add
            own = decoded * frame if half_out or len(lengths) > 1 else (decoded - total) * frame
            steady = inputs + local + own + seed
            # a half output's next seed, made while the decoded chunk and the seed before it live
            reseed = decoded * frame + seed + chunk * frame + local if half_out and k > 0 else 0
            peak = max(peak, steady, reseed, *(moment + seed for moment in moments))
            decoded_before = decoded
        out = image(total, h, w, F16 if half_out else F32)
        note = (f"chunk plan {' + '.join(map(str, lengths))}: decoded chunk, driving-video windows and seed; the core "
                "node's conditioning and the VAE / sampler working sets not counted")
        if float(c.widget("color_anchor_strength", 0.0)) > 0:
            note += ", nor the colour anchor's"
        if unknown:
            note += f"; {', '.join(unknown)} has no estimate: its float32 copy, when half, not counted"
        if not c.output_linked(0):
            # the frames are generated all the same (frame_count counts them) and dropped at return
            return {}, peak + once + out["bytes"], "images not linked: dropped at return; " + note
        return {0: out}, peak + once, note
    return profile


# -- the preprocess (pose, SAM 3.1 Multiplex, face, SCAIL-2) --------------------------------------

def half_frames(img):
    """What Pose Detection's frame loops hold besides their outputs on a half-precision clip: the
    next frame is requantized on a worker thread into one of two reused float32 frames. 0 for a
    float32 clip, read as a view."""
    return 2 * image_bytes(1, img["h"], img["w"]) if dtype_of(img) == F16 else 0


def _pose(c):
    """Pose Detection: the pose images in the frames' dtype at the frame size, or at width x height
    when both are given; not drawn when nothing links them."""
    img = c.tensor("images")
    out = {}
    if c.output_linked(0):
        size = given_size(c, "the pose images'")
        w, h = (img["w"], img["h"]) if size is None else size
        out[0] = image(img["n"], h, w, dtype_of(img))
    return (out, half_frames(img), "drawn frame by frame into a preallocated output (not drawn when not linked); "
            f"the pose models' ({c.widget('pose_model', 'ViTPose-H')}) weights and working set not counted")


def _sam_track(c):
    img = c.tensor("images")
    return ({0: mask(img["n"], img["h"], img["w"], dtype_of(img))}, 0,
            "a mask of the frames' dtype; frames read as a view; SAM 3.1's own working set not counted")


def _face_crop(c):
    img = c.tensor("images")
    if not c.output_linked(0):
        return {}, 0, "face_images not linked: the face boxes only, nothing cut"
    return {0: image(img["n"], FACE_SIZE, FACE_SIZE)}, 0, "512 x 512 float32 crops (of a half clip too) into a preallocated output"


def _wan_animate_preprocess(c):
    """WanAnimate Preprocess: Pose Detection, SAM 3.1 and Face Crop chained, each heavy output made
    only when linked, in the frames' dtype but the float32 face crops; the raw mask is made for the
    final mask too, and the final mask for bg_images. An unlinked raw mask is freed once the final
    mask is made of it; an unlinked final mask lives until bg_images is painted with it."""
    img = c.tensor("images")
    n, h, w, dtype = img["n"], img["h"], img["w"], dtype_of(img)
    made = {0: image(n, h, w, dtype), 1: image(n, FACE_SIZE, FACE_SIZE), 2: mask(n, h, w, dtype), 7: mask(n, h, w, dtype),
            8: image(n, h, w, dtype)}
    out = {slot: est for slot, est in made.items() if c.output_linked(slot)}
    one = mask_bytes(n, h, w, dtype)  # the raw mask, and the final mask
    finals = 7 in out or 8 in out
    unlinked = one * (finals and 2 not in out) + one * (8 in out and 7 not in out)
    return (out, max(half_frames(img), unlinked), "pose, face crops, mask, final mask and background frames (each only "
            "when linked) into preallocated outputs; SAM 3.1's own working set not counted")


def render_cut(m):
    """The colored masks' cut of RENDER_CHUNK frames of the MASK estimate `m`: booleans, a half mask's
    frames copied to float32 first."""
    return min(RENDER_CHUNK, m["n"]) * m["h"] * m["w"] * (1 + (F32 if dtype_of(m) == F16 else 0))


def _scail2_preprocess(c):
    """SCAIL-2 Preprocess: Pose Detection in the pose modes (its pose images not drawn), SAM 3.1 on
    the driving video and, without a connected reference_mask, on the reference image, then
    SCAIL-2 Colored Mask; the masks and colored masks in their frames' dtype. pose_video_mask, the
    driving video on black and the mask only when linked; an unlinked mask lives until the end."""
    img = c.tensor("images")
    n, h, w, dtype = img["n"], img["h"], img["w"], dtype_of(img)
    if c.linked("reference_mask"):
        ref = c.tensor("reference_mask")
        out = {4: shared(ref)}
    else:
        reference = c.tensor("reference_image")
        ref = mask(reference["n"], reference["h"], reference["w"], dtype_of(reference))
        out = {4: ref}
    driving = mask(n, h, w, dtype)
    out[2] = image(ref["n"], ref["h"], ref["w"], dtype_of(ref))
    cuts = [render_cut(ref)]
    if c.output_linked(1):
        out[1] = image(n, h, w, dtype)
        cuts.append(render_cut(driving))
    if c.output_linked(0):
        black = bool(c.widget("black_background", False))
        out[0] = image(n, h, w, dtype) if black else shared(img)
        cuts.append(mask_bytes(min(RENDER_CHUNK, n), h, w, 1) if black else 0)  # the boolean cut, 16 frames
    if c.output_linked(3):
        out[3] = driving
    pose = half_frames(img) if c.widget("mode") != "prompt" else 0
    dropped = 0 if 3 in out else driving["bytes"]
    return out, max(pose, dropped + max(cuts)), "cut 16 frames at a time; SAM 3.1's own working set not counted"


def _scail2_colored_mask(c):
    driving = c.tensor("driving_mask")
    ref = c.tensor("reference_mask") if c.linked("reference_mask") else None
    # without a reference mask the reference is one float32 frame of zeros at the driving mask's size
    reference = ref or mask(1, driving["h"], driving["w"])
    out = {1: image(reference["n"], reference["h"], reference["w"], dtype_of(reference))}
    cuts = [render_cut(reference) + (0 if ref else reference["bytes"])]
    if c.output_linked(0):
        out[0] = image(driving["n"], driving["h"], driving["w"], dtype_of(driving))
        cuts.append(render_cut(driving))
    return out, max(cuts), "in the masks' dtype, cut 16 frames at a time; pose_video_mask not rendered when not linked"


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


def _seedvr2_resized(c):
    """SeedVR2 Resize's two outputs for the node's `image` and resize widgets: (padded, reference)."""
    img = c.tensor("image")
    n, h, w = img["n"], img["h"], img["w"]
    resolution = int(round(min(h, w) * c.widget("upscale_factor", 2.0)))
    down = c.widget("downscale_factor", 0.5)
    h_d, w_d = (round(h * down), round(w * down)) if down != 1.0 else (h, w)
    rh, rw = _shortest_edge(h_d, w_d, resolution, c.widget("max_resolution", 4096))
    extra = 0 if n == 1 else (4 - n + 1 if n <= 4 else (4 - (n - 1) % 4) % 4)
    padded = image(n + extra, round_up_to_multiple(rh, SEEDVR2_PAD), round_up_to_multiple(rw, SEEDVR2_PAD), F16)
    reference = image(n, rh // 2 * 2, rw // 2 * 2, F16)
    return padded, reference


def _seedvr2_resize(c):
    padded, reference = _seedvr2_resized(c)
    return {0: padded, 1: reference}, 0, "float16 outputs; downscaled and resized four frames at a time"


def _seedvr2_encoded(c, pixels):
    """SeedVR2 VAE Encode's latent of the `pixels` estimate under the node's tile_size, and its transient."""
    n, h, w = pixels["n"], pixels["h"], pixels["w"]
    lt, lh, lw = (n + 3) // 4, (h + 7) // 8, (w + 7) // 8
    elements = SEEDVR2_LATENT_CHANNELS * lt * lh * lw
    tile = int(c.widget("tile_size", 1024))
    # tile_size 0 (auto) is decided by the card the node runs on: counted as tiled, the larger of the two
    single = 0 < tile and h <= tile and w <= tile
    # the latent slices go straight into the float32 output; tiles: the output is the float32 sum,
    # rounded through the VAE dtype (2 bytes) in place, a chunk of latent frames at a time
    transient = 0 if single else SEEDVR2_LATENT_CHANNELS * min(lt, SEEDVR2_CHUNK) * lh * lw * 2
    latent = {"type": "LATENT", "shape": [1, SEEDVR2_LATENT_CHANNELS, lt, lh, lw], "lt": lt, "lh": lh, "lw": lw,
              "bytes": elements * F32}
    return latent, transient


def _seedvr2_encode(c):
    latent, transient = _seedvr2_encoded(c, c.tensor("pixels"))
    return {0: latent}, transient, "the encoder's working set (device) not counted"


def _seedvr2_decoded(c):
    """SeedVR2 VAE Decode's output (frames, height, width) for the node's `samples`."""
    latent = c.tensor("samples")
    if "lt" not in latent:
        raise NotCounted("samples is not a SeedVR2 latent estimate")
    return max(1, latent["lt"] * 4 - 3), latent["lh"] * 8 // 2 * 2, latent["lw"] * 8 // 2 * 2


def _seedvr2_decode(c):
    return {0: image(*_seedvr2_decoded(c), F16)}, 0, "decoded frames streamed into a float16 output, tiles summed in it"


def _seedvr2_preprocess_compact(c):
    padded, reference = _seedvr2_resized(c)
    latent, transient = _seedvr2_encoded(c, padded)
    plan = {"type": "SEEDVR2_PLAN", "shape": [], "bytes": 0, "n": reference["n"], "h": reference["h"], "w": reference["w"]}
    return ({0: latent, 1: plan}, padded["bytes"] + transient,
            "the padded float16 clip lives while it is encoded; the encoder's working set (device) not counted")


def _seedvr2_postprocess_compact(c):
    frames, h, w = _seedvr2_decoded(c)
    plan = c.tensor("plan")
    out = image(min(frames, plan["n"]), min(h, plan["h"]), min(w, plan["w"]), F16)
    return {0: out}, 0, "decoded into its float16 output cut to the plan, colour-corrected there; references four frames at a time"


def _seedvr2_postprocess(c):
    images, ref = c.tensor("images"), c.tensor("original_resized_images")
    t, h, w = min(images["n"], ref["n"]), min(images["h"], ref["h"]), min(images["w"], ref["w"])
    return {0: image(t, h // 2 * 2, w // 2 * 2, F16)}, 0, "one frame at a time into a float16 output"


def _seedvr2_framing_downscale(c):
    img = c.tensor("image")
    k = min(img["n"], FRAMING_CHUNK)
    # per group of k frames, in RAM: a half group read as float32, which core's SAM3_Detect scales
    # to 1008 x 1008, and their union masks as a list of float32 frames stacked into one (both held at its return)
    transient = image_bytes(k, SAM3_SIDE, SAM3_SIDE) + 2 * mask_bytes(k, img["h"], img["w"])
    transient += image_bytes(k, img["h"], img["w"]) if dtype_of(img) == F16 else 0
    return {}, transient, "two numbers; SAM 3's working set on its device not counted"


# -- BCNodes image nodes --------------------------------------------------------------------------

def _birefnet(c):
    img = c.tensor("image")
    n, h, w = img["n"], img["h"], img["w"]
    dtype = dtype_of(img)
    alpha = c.widget("background", "Alpha") == "Alpha"
    one, frame = mask_bytes(1, h, w), image_bytes(1, h, w)
    # one frame at a time in float32: its raw matte, a new one when an option changes it, a half frame read
    # as float32, the refined colours, the "over" blend's background term, and the blend itself when it
    # cannot be made in a float32 output
    options = (c.widget("sensitivity", 1.0) < 1.0 or c.widget("mask_blur", 0) > 0 or c.widget("mask_offset", 0) != 0
               or bool(c.widget("invert_output", False)))
    half = dtype == F16
    transient = one + (one if options else 0) + (frame if half else 0)
    transient += frame if c.widget("refine_foreground", False) else 0
    transient += 0 if alpha else frame + (frame if half else 0)
    out = {0: image(n, h, w, dtype, channels=4 if alpha else 3), 1: mask(n, h, w, dtype), 2: image(n, h, w, dtype)}
    return out, transient, "one frame at a time into outputs of the image's dtype; the network's working set not counted"


def _depth_anything(c):
    img = c.tensor("image")
    size = given_size(c, "the depth map's")
    if size is None:
        height, width = short_side_size(img["h"], img["w"], c.widget("resolution", 518))
    else:
        width, height = size
    # a half frame read as float32
    return {0: image(img["n"], height, width, dtype_of(img))}, image_bytes(1, img["h"], img["w"]) if dtype_of(img) == F16 else 0, \
        "one frame at a time into a preallocated output of the image's dtype; the network's working set not counted"


def _postfx_apply(c):
    img = c.tensor("image")
    if c.widget("theme") == "none" and not c.linked("look"):
        return {0: shared(img)}, 0, "no look: the input passed on"
    out = image(img["n"], img["h"], img["w"], dtype_of(img))
    # per frame: the float32 frame (a half one read as float32) and the processed one, clipped into
    # the output; postfx's own working set not counted
    frames = 1 + (1 if dtype_of(img) == F16 else 0)
    return {0: out}, frames * image_bytes(1, img["h"], img["w"]), \
        "one frame at a time, clipped into a preallocated output of the image's dtype; postfx's working set not counted"


def _skin_texture(c):
    img = c.tensor("image")
    return {0: image(img["n"], img["h"], img["w"], dtype_of(img)), 1: mask(img["n"], img["h"], img["w"], dtype_of(img))}, None, \
        "one frame at a time into outputs of the image's dtype; SAM 3 and the texture's working set not counted"


def _frequency_merge(c):
    base, detail = c.tensor("base"), c.tensor("detail")
    n, h, w = base["n"], base["h"], base["w"]
    if (detail["n"], detail["h"], detail["w"]) != (n, h, w):
        raise NotCounted("base and detail differ in size or image count: the node stops with an error")
    if c.linked("split_sigma"):
        raise NotCounted("split_sigma comes from a link: the blur's padding is known at run time only")
    r = math.ceil(3 * c.widget("split_sigma", 3.0))
    if r >= min(h, w):
        raise NotCounted("split_sigma blurs over more than the image holds: the node stops with an error")
    half = base.get("dtype_bytes", F32) == F16 and detail.get("dtype_bytes", F32) == F16
    out = {0: image(n, h, w, F16 if half else F32)}
    if c.widget("device", "cpu") == "gpu":
        return out, 0, ("one image at a time into a preallocated output; the float32 working buffers on ComfyUI's device "
                         "not counted")
    one = image_bytes(1, h, w)
    # per image in float32 (libs/frequency.py, gauss_reflect): detail's high-pass held while base is
    # blurred, the blur's row-padded copy, row sum and column-padded copy; an input that is not float32
    # copied to float32 first
    work = 2 * one + image_bytes(1, h, w + 2 * r) + image_bytes(1, h + 2 * r, w)
    work += sum(one for t in (base, detail) if t.get("dtype_bytes", F32) != F32)
    return out, work, "one image at a time into a preallocated output, float32 working buffers"


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


# -- LM nodes -------------------------------------------------------------------------------------

def _qwen_lm(c):
    # Qwen LM loads its own model: the model widget holds a catalog name, not a file name, so no weights
    # are read for it; its outputs are two strings.
    raise NotCounted("the model widget names a catalog entry, not a file: the model file's weights, the LoRA, the "
                     "KV cache core reserves for prompt + max_new_tokens and the images resized for the model are "
                     "not counted")


def _lm_config(c):
    return {}, 0, "a dict of the edited sampling fields, at most eight plain values; no tensor"


# -- PreFlight ------------------------------------------------------------------------------------

def _preflight_observe(c):
    # PreFlight Observe runs a fixed catalog model through the LM runtime (pipelines/preflight/observe.py): no
    # widget names its file, so no weights are read for it; its outputs are two strings.
    raise NotCounted("runs Qwen3.5-9B INT8 ConvRot, which no widget names as a file: the model file's weights, the "
                     "KV cache core reserves for prompt + 300 tokens, the sampled frames (a copy when fewer than "
                     "the batch) and the images resized for the model are not counted")


def _preflight_report(c):
    # the image input is returned as it is; the other outputs are strings
    if not c.linked("image"):
        return {}, 0, "text outputs only: no image to pass on"
    return {c.slot("IMAGE"): shared(c.tensor("image"))}, 0, "passes the image input on as it is; the report is text"


PROFILES = {
    # loaders and resizers
    "VHS_LoadVideo": _whole_clip_loader, "VHS_LoadVideoPath": _whole_clip_loader, "BCVLoadVideo": _bcv_load_video,
    "BCVGetVideoInfo": _bcv_get_video_info,
    "LoadImage": _load_image, "BCVLoadReferenceImage": _bcv_load_reference,
    "ImageResizeKJv2": _resize_listed_then_cat, "BC_ImageResize": _resize_preallocated, "ImageScale": _image_scale,
    "BCVConformVideo": _conform_video, "BC_ImageScaleByAspectRatio": _scale_by_aspect_ratio,
    # masks
    "BC_MaskGrow": _frame_mask(half_frames=2), "BC_MaskFillHoles": _frame_mask("masks"),
    "BC_BlockifyMask": _same_mask("one frame at a time into a preallocated output of the mask's dtype", name="masks"),
    "BlockifyMask": _same_mask("zeros_like work buffer + clamp copy: 1 x mask", 1, name="masks"),
    "BC_RepeatMaskBatch": _repeat_mask("amount"), "VHS_DuplicateMasks": _repeat_mask("multiply_by"),
    "DrawMaskOnImage": _draw_mask_cloned, "BC_DrawMaskOnImage": _draw_mask_preallocated,
    # Wan and core sampling
    "WanAnimateToVideo": _wan_to_video, "WanImageToVideo": _wan_to_video,
    "KSampler": _sampler, "KSamplerAdvanced": _sampler, "VAEDecode": _vae_decode, "VAEDecodeTiled": _vae_decode,
    "BCVWanAnimateLongVideoSampler": _long_video(("pose_video", "face_video", "background_video"), ("character_mask",),
                                                 ("background_video", "character_mask"), "continue_motion_max_frames"),
    "BCVWanAnimate2LongVideoSampler": _long_video(("pose_video",)),  # its node takes no face or background video
    "BCVSCAIL2LongVideoSampler": _long_video(("pose_video", "pose_video_mask"), overlap="previous_frame_count", last_chunk="full",
                                             whole=("reference_image_mask",)),
    # the preprocess
    "BCVPoseDetection": _pose, "BCVSAM3VideoTrack": _sam_track, "BCVFaceCrop": _face_crop,
    "BCVWanAnimatePreprocess": _wan_animate_preprocess, "BCVSCAIL2Preprocess": _scail2_preprocess,
    "BCVSCAIL2ColoredMask": _scail2_colored_mask,
    "BCVPoseGuard": _pose_guard, "BCVMaskGuard": _mask_guard(3), "BCVWanAnimatePreprocessGuard": _mask_guard(4),
    "BCVSCAIL2PreprocessGuard": _scail2_guard,
    # outputs
    "VHS_VideoCombine": _per_frame_output, "BCVSaveVideo": _per_frame_output, "BCVVideoComparer": _per_frame_output,
    "SaveImage": _per_frame_output, "PreviewImage": _per_frame_output,
    # SeedVR2
    "BC_SeedVR2Resize": _seedvr2_resize, "BC_SeedVR2VAEEncode": _seedvr2_encode, "BC_SeedVR2VAEDecode": _seedvr2_decode,
    "BC_SeedVR2PostProcess": _seedvr2_postprocess, "BC_SeedVR2PreprocessCompact": _seedvr2_preprocess_compact,
    "BC_SeedVR2PostProcessCompact": _seedvr2_postprocess_compact,
    "BC_SeedVR2FramingDownscale": _seedvr2_framing_downscale,
    # BCNodes image nodes and switches
    "BC_BiRefNetRemoveBackground": _birefnet, "BC_DepthAnythingV2": _depth_anything, "BC_PostFxApply": _postfx_apply,
    "BC_SkinTexture": _skin_texture, "BC_AnySwitch": _any_switch, "BC_SelectSwitch": _select_switch,
    "BC_JoinImageLists": _join_lists,
    "BC_FrequencyMerge": _frequency_merge,
    # LM nodes
    "BC_QwenLM": _qwen_lm, "BC_LMConfig": _lm_config,
    # PreFlight (Outcome and Calibrate output text only: no profile needed)
    "BC_PreFlightObserve": _preflight_observe, "BC_PreFlightReport": _preflight_report,
}

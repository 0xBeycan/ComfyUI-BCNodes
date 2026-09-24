"""Social Media Export engine: geometry, encoding and the report line.

No ComfyUI or torch imports, so it unit-tests on its own; PIL is its only
third-party dependency and is imported inside the functions that use it.

Responsibilities:
  * load platform specs from JSON (fresh on every call, no module caching)
  * plan the minimum crop needed to reach a platform's aspect-ratio band
  * apply the crop budget + overflow strategy (error / pad / crop)
  * render the blurred-letterbox pad fallback
  * encode to a high-fidelity, metadata-stripped, sRGB-tagged file
  * step quality down to satisfy a byte cap
  * format a readable per-derivative report line
"""

from __future__ import annotations

import functools
import io
import json
from typing import Dict, Optional, Tuple, TypedDict


@functools.lru_cache(maxsize=None)
def _srgb_icc() -> Optional[bytes]:
    """sRGB ICC profile bytes (~588 B), embedded via icc_profile= on save.
    Every standard Pillow build ships ImageCms with littleCMS; a stripped build
    without it degrades to "no profile" rather than crashing."""
    try:
        from PIL import ImageCms
        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:  # pragma: no cover - only on a Pillow built without littleCMS
        return None


# Byte-cap quality stepping.
_QUALITY_STEP = 5
_QUALITY_FLOOR = 70


# --------------------------------------------------------------------------- #
# Contracts (annotation only: both stay plain dicts)
# --------------------------------------------------------------------------- #

class PlatformSpec(TypedDict, total=False):
    """One platform of social_specs.json, keys in the order of its "_schema". ar_min, ar_max,
    max_w and max_h are indexed; min_w, format (default "jpg") and max_bytes are read only
    when present. The '_'-prefixed notes beside them (_source, _verified, _note) are not read."""
    ar_min: float
    ar_max: float
    max_w: int
    max_h: int
    min_w: int
    format: str
    max_bytes: int


class RenderMeta(TypedDict):
    """render's metadata dict, keys in the order both of its producers build them. A dict,
    not a dataclass: format_report_line and the tests index it."""
    platform: str
    input: Tuple[int, int]
    crop: Tuple[int, int]
    out: Tuple[int, int]
    crop_pct: float
    strategy: str  # 'scale' | 'crop' | 'crop!' | 'pad'
    over: bool


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class UnknownPlatformError(ValueError):
    """Raised when a requested platform is not present in the specs file."""


class OverflowCropError(ValueError):
    """Raised when a crop would discard more than max_crop_ratio and the
    overflow strategy is 'error'."""


# --------------------------------------------------------------------------- #
# Spec loading
# --------------------------------------------------------------------------- #

def load_specs(path: str) -> Dict[str, dict]:
    """Read and parse the specs JSON fresh from disk. No caching, so a user can
    edit ``social_specs.json`` and have it picked up on the next execution."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def platform_names(specs: Dict[str, dict]) -> list:
    """Return the list of selectable platform keys, excluding '_'-prefixed
    metadata/comment keys."""
    return [k for k in specs.keys() if not k.startswith("_")]


def get_spec(specs: Dict[str, dict], platform: str) -> PlatformSpec:
    """Fetch a platform spec, raising UnknownPlatformError (listing valid keys)
    if it is missing or is a metadata key."""
    if platform.startswith("_") or platform not in specs:
        valid = ", ".join(sorted(platform_names(specs)))
        raise UnknownPlatformError(
            f"Unknown platform '{platform}'. Valid platforms: {valid}"
        )
    return specs[platform]


# --------------------------------------------------------------------------- #
# The planner  (ported verbatim from the validated reference; semantics kept)
# --------------------------------------------------------------------------- #

def plan(
    w: int,
    h: int,
    spec: PlatformSpec,
    anchor: float = 0.4,
    allow_upscale: bool = False,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """Return ((ox, oy, cw, ch), (out_w, out_h)).

    * If the input aspect ratio already falls inside [ar_min, ar_max] -> zero
      crop (cw==w, ch==h).
    * Otherwise crop the minimum amount on a single axis to reach the nearest
      band edge.
    * ``anchor`` positions the crop window vertically: 0=top, 0.5=center,
      1=bottom (default 0.4, slightly top-weighted for portrait subjects).
    * Output size scales the crop into the pixel envelope; downscale-only unless
      ``allow_upscale`` (in which case ``min_w`` acts as an upscale floor).
    """
    ar = w / h
    lo, hi = spec["ar_min"], spec["ar_max"]
    if ar < lo - 1e-6:        # too tall  -> trim height
        cw, ch = w, round(w / lo)
    elif ar > hi + 1e-6:      # too wide  -> trim width
        cw, ch = round(h * hi), h
    else:                     # already legal -> zero crop
        cw, ch = w, h
    ox = round((w - cw) * 0.5)
    oy = round((h - ch) * anchor)      # anchor 0=top, 0.5=center, 1=bottom
    s = min(spec["max_w"] / cw, spec["max_h"] / ch)
    if not allow_upscale:
        s = min(s, 1.0)
    elif "min_w" in spec:
        s = max(s, spec["min_w"] / cw)
    return (ox, oy, cw, ch), (max(1, round(cw * s)), max(1, round(ch * s)))


def crop_fraction(w: int, h: int, cw: int, ch: int) -> float:
    """Fraction of the original pixels discarded by cropping (w*h) -> (cw*ch)."""
    if w <= 0 or h <= 0:
        return 0.0
    return 1.0 - (cw * ch) / float(w * h)


# --------------------------------------------------------------------------- #
# Pad fallback  (blurred cover-scale letterbox — never flat black bars)
# --------------------------------------------------------------------------- #

def pad_box(w: int, h: int, spec: PlatformSpec, allow_upscale: bool = False) -> Tuple[int, int]:
    """Compute the output canvas for the pad strategy: the largest box of the
    nearest band-edge aspect ratio that fits inside the pixel envelope. When
    upscaling is disallowed the box is shrunk (preserving AR) so it never
    exceeds the source dimensions."""
    lo, hi = spec["ar_min"], spec["ar_max"]
    ar = w / h
    target = ar
    if ar < lo:
        target = lo
    elif ar > hi:
        target = hi
    max_w, max_h = spec["max_w"], spec["max_h"]
    box_w = min(max_w, max_h * target)
    box_h = box_w / target
    if not allow_upscale:
        shrink = min(1.0, w / box_w, h / box_h)
        box_w *= shrink
        box_h *= shrink
    return max(1, round(box_w)), max(1, round(box_h))


def pad_composite(img: Image.Image, box_w: int, box_h: int, allow_upscale: bool = False) -> Image.Image:
    """Fit the whole image (no crop) centered inside a box_w x box_h canvas whose
    background is a Gaussian-blurred, cover-scaled copy of the same image."""
    from PIL import Image, ImageFilter

    img = flatten_to_rgb(img)

    # Background: cover-scale to fill the box, center-crop, blur.
    cover = max(box_w / img.width, box_h / img.height)
    bg = img.resize(
        (max(1, round(img.width * cover)), max(1, round(img.height * cover))),
        Image.LANCZOS,
    )
    left = (bg.width - box_w) // 2
    top = (bg.height - box_h) // 2
    bg = bg.crop((left, top, left + box_w, top + box_h))
    radius = max(1.0, 0.04 * box_w)  # ~4% of target width
    bg = bg.filter(ImageFilter.GaussianBlur(radius))

    # Foreground: contain-scale (downscale-only unless allowed), centered.
    fit = min(box_w / img.width, box_h / img.height)
    if not allow_upscale:
        fit = min(fit, 1.0)
    fw = max(1, round(img.width * fit))
    fh = max(1, round(img.height * fit))
    fg = img.resize((fw, fh), Image.LANCZOS)
    bg.paste(fg, ((box_w - fw) // 2, (box_h - fh) // 2))
    return bg


# --------------------------------------------------------------------------- #
# Top-level render: plan -> budget check -> crop/pad/scale
# --------------------------------------------------------------------------- #

def render(
    img: Image.Image,
    spec: PlatformSpec,
    *,
    anchor: float = 0.4,
    allow_upscale: bool = False,
    max_crop_ratio: float = 0.15,
    overflow_strategy: str = "error",
    platform: str = "",
) -> Tuple[Image.Image, RenderMeta]:
    """Produce the platform derivative image and a metadata dict.

    meta keys: platform, input (w,h), crop (w,h), out (w,h), crop_pct (float),
    strategy (str: 'scale' | 'crop' | 'crop!' | 'pad'), over (bool).
    """
    from PIL import Image

    if overflow_strategy not in ("error", "pad", "crop"):
        raise ValueError(
            f"overflow_strategy must be 'error', 'pad' or 'crop', got {overflow_strategy!r}"
        )

    w, h = img.size
    (ox, oy, cw, ch), (ow, oh) = plan(w, h, spec, anchor, allow_upscale)
    frac = crop_fraction(w, h, cw, ch)
    over = frac > max_crop_ratio + 1e-9

    if over:
        if overflow_strategy == "error":
            raise OverflowCropError(_overflow_message(platform, spec, w, h, cw, ch, frac, max_crop_ratio))
        if overflow_strategy == "pad":
            bw, bh = pad_box(w, h, spec, allow_upscale)
            result = pad_composite(img, bw, bh, allow_upscale)
            meta = {
                "platform": platform,
                "input": (w, h),
                "crop": (w, h),          # nothing cropped
                "out": result.size,
                "crop_pct": 0.0,
                "strategy": "pad",
                "over": True,
            }
            return result, meta
        # overflow_strategy == "crop": explicit opt-in, proceed with the crop.
        strategy = "crop!"
    else:
        strategy = "crop" if (cw < w or ch < h) else "scale"

    result = img
    if cw < w or ch < h:
        result = img.crop((ox, oy, ox + cw, oy + ch))
    if (ow, oh) != result.size:
        result = result.resize((ow, oh), Image.LANCZOS)

    meta = {
        "platform": platform,
        "input": (w, h),
        "crop": (cw, ch),
        "out": (ow, oh),
        "crop_pct": frac * 100.0,
        "strategy": strategy,
        "over": over,
    }
    return result, meta


def _overflow_message(platform, spec, w, h, cw, ch, frac, max_crop_ratio) -> str:
    return (
        f"SocialMediaExport: exporting a {w}x{h} master to '{platform}' would crop "
        f"{frac * 100:.1f}% of its pixels (down to {cw}x{ch}) to reach the platform's "
        f"aspect-ratio band {spec['ar_min']}..{spec['ar_max']} (w/h) — that exceeds "
        f"max_crop_ratio={max_crop_ratio:.0%}. Choose overflow_strategy='pad' to letterbox "
        f"the full frame onto a blurred background (no crop), or 'crop' to discard those "
        f"pixels anyway, or feed a master closer to the target aspect ratio."
    )


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #

def flatten_to_rgb(img: Image.Image) -> Image.Image:
    """Flatten to RGB, compositing any alpha onto white. Derivatives published to
    third parties are always opaque RGB."""
    from PIL import Image

    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return img.convert("RGB")


def encode(
    img: Image.Image,
    spec: PlatformSpec,
    *,
    quality: int = 92,
    chroma_444: bool = True,
    progressive: bool = True,
) -> Tuple[bytes, int, str, bool]:
    """Encode ``img`` per the platform spec.

    Returns (data, final_quality, ext, exceeded) where ``ext`` is the file
    extension ('jpg'/'webp'/'png') and ``exceeded`` is True only if a byte cap
    was set and could not be met even at the quality floor.

    All workflow/prompt metadata is stripped: we build the save from pixels only
    and never pass exif/pnginfo. The sRGB ICC profile is embedded. For JPEG,
    subsampling=0 (4:4:4) is forced when chroma_444, because Pillow writes 4:2:0
    at every quality level otherwise.
    """
    fmt = str(spec.get("format", "jpg")).lower()
    max_bytes = spec.get("max_bytes")
    rgb = flatten_to_rgb(img)

    if fmt == "png":
        # Lossless: quality/subsampling are irrelevant; no quality loop.
        data = _encode_once(rgb, "png", 100, chroma_444, progressive)
        exceeded = bool(max_bytes) and len(data) > max_bytes
        return data, 100, "png", exceeded

    ext = "webp" if fmt == "webp" else "jpg"
    q = int(quality)
    while True:
        data = _encode_once(rgb, ext, q, chroma_444, progressive)
        if max_bytes is None or len(data) <= max_bytes or q <= _QUALITY_FLOOR:
            exceeded = bool(max_bytes) and len(data) > max_bytes
            return data, q, ext, exceeded
        q = max(_QUALITY_FLOOR, q - _QUALITY_STEP)


def _encode_once(rgb: Image.Image, ext: str, q: int, chroma_444: bool, progressive: bool) -> bytes:
    from PIL import ImageFile

    icc = _srgb_icc()
    buf = io.BytesIO()
    if ext == "jpg":
        # optimize/progressive JPEG must buffer a full scan; with the default
        # MAXBLOCK libjpeg raises "Suspension not allowed here" on large or
        # incompressible frames (fails to a real file too, not just BytesIO).
        # Raising MAXBLOCK to cover this frame is Pillow's documented remedy.
        ImageFile.MAXBLOCK = max(ImageFile.MAXBLOCK, 3 * rgb.width * rgb.height)
        params = dict(
            format="JPEG",
            quality=int(q),
            optimize=True,
            progressive=bool(progressive),
        )
        if chroma_444:
            params["subsampling"] = 0     # mandatory for true 4:4:4
        if icc:
            params["icc_profile"] = icc
        rgb.save(buf, **params)
    elif ext == "webp":
        params = dict(format="WEBP", quality=int(q), method=6)
        if icc:
            params["icc_profile"] = icc
        rgb.save(buf, **params)
    elif ext == "png":
        params = dict(format="PNG", optimize=True)
        if icc:
            params["icc_profile"] = icc
        rgb.save(buf, **params)
    else:  # pragma: no cover - guarded by encode()
        raise ValueError(f"Unsupported format: {ext}")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Report formatting
# --------------------------------------------------------------------------- #

def human_bytes(n: int) -> str:
    """Compact human-readable byte size."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.2f}MB"


def _dims(w: int, h: int) -> str:
    return f"{w}x{h}"


REPORT_HEADER = (
    f"{'#':>3}  {'platform':<16} "
    f"{'input':>11} -> {'crop':>11} -> {'output':>11}  "
    f"{'crop%':>6}  {'mode':<5} {'qual':<4} {'fmt':<4} {'size':>9}  notes"
)


def format_report_line(
    index: int,
    meta: dict,
    quality: int,
    size_bytes: int,
    ext: str,
    exceeded: bool,
    max_bytes: Optional[int],
) -> str:
    """One aligned line per (image, platform). This is the debugging surface:
    input dims -> crop dims -> output dims, crop %, strategy, final quality, size.
    """
    notes = []
    if meta["strategy"] == "pad":
        notes.append("letterboxed (blurred bg), no crop")
    elif meta["strategy"] == "crop!":
        notes.append("OVER-BUDGET crop (forced)")
    if exceeded and max_bytes:
        notes.append(f">max_bytes ({human_bytes(max_bytes)}) even at q{quality}")
    return (
        f"{index:>3}  {meta['platform']:<16} "
        f"{_dims(*meta['input']):>11} -> {_dims(*meta['crop']):>11} -> "
        f"{_dims(*meta['out']):>11}  "
        f"{meta['crop_pct']:>5.1f}%  {meta['strategy']:<5} "
        f"q{quality:<3} {ext:<4} {human_bytes(size_bytes):>9}  "
        f"{'; '.join(notes)}"
    ).rstrip()

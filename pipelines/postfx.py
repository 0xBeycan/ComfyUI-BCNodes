"""The postfx adapter behind the PostFx nodes: the `postfx` package guard, the catalogs that
fill their dropdowns, the pack's luts/ folder, the look builders, the apply flow, the contact
sheet and the IMAGE / MASK <-> postfx numpy bridge. `postfx` (and cv2, which it brings) is
imported inside the functions that use it.
"""

import copy
import os
import tempfile
import uuid

import numpy as np

# Users drop their own .cube files here (listed by PostFx LUT). Kept out of
# git by .gitignore; see luts/README.md.
LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")


def postfx_package():
    try:
        import postfx
    except ImportError as exc:
        raise ImportError("the PostFx nodes need the `postfx` package: pip install 'postfx>=1.1.0'") from exc
    return postfx


# --- Catalogs (populate dropdowns) ----------------------------------------

def theme_names():
    """'category/stem' labels for every built-in theme, ordered signature ->
    luts -> experimental."""
    return [f"{cat}/{stem}" if cat else stem for stem, _desc, cat in postfx_package().list_themes()]


def theme_stem(name):
    """Dropdown label -> theme stem accepted by load_theme. A bare stem (older
    saved workflows) passes through unchanged."""
    return name.rsplit("/", 1)[-1]


def default_theme(names):
    """portra_400 when present, else the first entry."""
    for name in names:
        if theme_stem(name).endswith("portra_400"):
            return name
    return names[0]


def condition_names():
    """All condition names, ordered by increasing texture intensity."""
    return [name for name, _desc in postfx_package().list_conditions()]


def lut_files():
    """`.cube` file names sitting in luts/, sorted."""
    if not os.path.isdir(LUTS_DIR):
        return []
    return sorted(f for f in os.listdir(LUTS_DIR) if f.lower().endswith(".cube"))


def neutral_look():
    """A full, valid look with every op at its no-op default."""
    return postfx_package().resolve_theme({})


def merge_look(base, overrides):
    """Deep-merge `overrides` (partial op blocks) onto a base look dict."""
    from postfx.theme import _deep_merge
    return _deep_merge(base, overrides)


def _base_look(look):
    """A deep copy of the incoming look, or a neutral look when none is connected."""
    return copy.deepcopy(look) if look is not None else neutral_look()


# --- Tensor <-> numpy bridge ----------------------------------------------

def image_to_np_list(image):
    arr = image.detach().cpu().numpy()
    return [np.ascontiguousarray(arr[i], dtype=np.float32) for i in range(arr.shape[0])]


def np_list_to_image(frames):
    import torch
    stacked = np.stack([np.clip(f, 0.0, 1.0).astype(np.float32) for f in frames], axis=0)
    return torch.from_numpy(stacked)


def mask_to_np_list(mask, count, hw):
    """MASK (B, H, W) or (H, W) -> `count` float32 (H, W) arrays resized to
    `hw` and clamped. A single mask broadcasts to every frame; a shorter batch
    reuses its last mask."""
    import cv2
    m = mask.detach().cpu().numpy().astype(np.float32)
    if m.ndim == 2:
        m = m[None]
    h, w = hw
    out = []
    for i in range(count):
        mm = m[i] if i < m.shape[0] else m[-1]
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(np.clip(mm, 0.0, 1.0))
    return out


# --- Looks and flows ------------------------------------------------------

def apply_look(image, theme, condition, strength, seed, batch_seed, look, mask):
    """IMAGE batch -> (IMAGE,) with the look (or the built-in theme) applied per frame,
    blended through the mask when one is set."""
    postfx = postfx_package()
    look_cfg = look if look is not None else postfx.resolve_theme(theme_stem(theme))
    cond_cfg = postfx.get_condition(condition)

    frames = image_to_np_list(image)
    h, w = frames[0].shape[:2]
    # LoadImage emits an all-zero placeholder mask for images without an
    # alpha channel; blending against it would be a silent no-op.
    if mask is not None and not bool(mask.any()):
        mask = None
    masks = mask_to_np_list(mask, len(frames), (h, w)) if mask is not None else None

    out_frames = []
    for i, src in enumerate(frames):
        frame_seed = seed + (i if batch_seed == "increment" else 0)
        out = postfx.process(src, look_cfg, cond_cfg, float(strength), int(frame_seed))
        if masks is not None:
            m = masks[i][..., None]
            out = src * (1.0 - m) + out * m
        out_frames.append(out)

    return (np_list_to_image(out_frames),)


def custom_look(temp, tint, exposure, contrast, vibrance, saturation,
                grain, grain_size, vignette, halation, clarity, look):
    """-> (look,): the base look with only the controls moved off neutral overridden."""
    base = _base_look(look)

    ov = {}
    if temp != 0.0 or tint != 0.0:
        ov["white_balance"] = {"temp": float(temp), "tint": float(tint)}
    if exposure != 0.0:
        ov["exposure"] = {"stops": float(exposure)}
    if contrast != 0.0:
        ov["tone_curve"] = {"strength": float(contrast)}
    if vibrance != 0.0 or saturation != 1.0:
        ov["vibrance"] = {"vibrance": float(vibrance), "saturation": float(saturation)}
    if grain > 0.0:
        ov["grain"] = {"luma": float(grain), "size": float(grain_size)}
    if vignette > 0.0:
        ov["vignette"] = {"strength": float(vignette)}
    if halation > 0.0:
        ov["halation"] = {"strength": float(halation)}
    if clarity != 0.0:
        ov["clarity"] = {"amount": float(clarity)}

    merged = merge_look(base, ov)
    base_name = base.get("name", "look")
    merged["name"] = base_name if not ov else f"{base_name}+custom"
    return (merged,)


def lut_look(lut, intensity, lut_path, look):
    """-> (look,): the base look with a .cube LUT from lut_path or luts/ attached."""
    base = _base_look(look)

    path = lut_path.strip()
    if not path and lut != "none":
        path = os.path.join(LUTS_DIR, lut)

    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LUT file not found: {path!r}")
        base = merge_look(base, {"lut": {"file": path, "intensity": float(intensity)}})
        base["name"] = f"{base.get('name', 'look')}+lut"
    return (base,)


def signature_sheet(image, category, condition, strength, columns):
    """-> (IMAGE,): the postfx contact sheet of a theme category on the first frame."""
    postfx = postfx_package()
    from postfx.sheet import build_contact_sheet

    src = image_to_np_list(image)[0]  # first frame is the sample
    tmp = os.path.join(tempfile.gettempdir(), f"postfx_sheet_{uuid.uuid4().hex}.jpg")
    try:
        build_contact_sheet(src, tmp, category=category, condition=condition,
                            strength=float(strength), cols=int(columns))
        rgb, _alpha = postfx.imgio.load_image(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return (np_list_to_image([rgb]),)

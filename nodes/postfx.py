"""PostFx nodes — film-emulation looks from the `postfx` package.

    BC_PostFxApply           (PostFx Apply)
    BC_PostFxTheme           (PostFx Theme)
    BC_PostFxCustomLook      (PostFx Custom Look)
    BC_PostFxLut             (PostFx LUT)
    BC_PostFxSignatureSheet  (PostFx Signature Sheet)

A look is a full theme dict (output of postfx.load_theme / resolve_theme) and
travels between nodes on the POSTFX_LOOK socket. ComfyUI IMAGE (B, H, W, 3)
float32 RGB and postfx (H, W, 3) float32 RGB line up exactly, so the bridge is
a per-frame copy.

`postfx` (and cv2, which it brings) is imported inside the functions that use
it: the package is a pip dependency, and a missing install must not take the
rest of the pack down with it.
"""

import copy
import os
import tempfile
import uuid

import numpy as np

POSTFX_LOOK = "POSTFX_LOOK"

# Users drop their own .cube files here (listed by PostFx LUT). Kept out of
# git by .gitignore; see luts/README.md.
LUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "luts")

_CATEGORY = "BCNodes/postfx"


def _postfx():
    try:
        import postfx
    except ImportError as exc:
        raise ImportError("the PostFx nodes need the `postfx` package: pip install 'postfx>=1.1.0'") from exc
    return postfx


# --- Catalogs (populate dropdowns) ----------------------------------------

def theme_names():
    """'category/stem' labels for every built-in theme, ordered signature ->
    luts -> experimental."""
    return [f"{cat}/{stem}" if cat else stem for stem, _desc, cat in _postfx().list_themes()]


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
    return [name for name, _desc in _postfx().list_conditions()]


def lut_files():
    """`.cube` file names sitting in luts/, sorted."""
    if not os.path.isdir(LUTS_DIR):
        return []
    return sorted(f for f in os.listdir(LUTS_DIR) if f.lower().endswith(".cube"))


def neutral_look():
    """A full, valid look with every op at its no-op default."""
    return _postfx().resolve_theme({})


def merge_look(base, overrides):
    """Deep-merge `overrides` (partial op blocks) onto a base look dict."""
    from postfx.theme import _deep_merge
    return _deep_merge(base, overrides)


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


# --- Nodes ----------------------------------------------------------------

class PostFxApply:
    """Apply a look + condition + strength to an IMAGE batch, with optional
    look override and mask-limited blend."""

    CATEGORY = _CATEGORY
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    SEARCH_ALIASES = ["BCNodes", "postfx", "film emulation", "color grade", "look"]
    DESCRIPTION = ("Apply a postfx film-emulation look to an image batch. Pick a "
                   "built-in theme + shooting condition, or feed a look from a "
                   "PostFx Theme / Custom Look / LUT node (a connected look "
                   "overrides the theme dropdown). An optional mask limits the "
                   "effect to the masked region.")

    @classmethod
    def INPUT_TYPES(cls):
        themes = ["none"] + theme_names()
        return {
            "required": {
                "image": ("IMAGE",),
                "theme": (themes, {"default": default_theme(themes),
                                   "tooltip": "Built-in look. 'none' passes the image through "
                                              "untouched. Ignored when a 'look' input is connected."}),
                "condition": (condition_names(), {"default": "neutral",
                                                  "tooltip": "Shooting condition: scales grain/chroma/halation only."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                                       "tooltip": "0 = original, 1 = full look, >1 = over."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff,
                                 "tooltip": "Grain seed (deterministic)."}),
                "batch_seed": (["fixed", "increment"], {"default": "fixed",
                               "tooltip": "increment = a different grain per frame across the batch."}),
            },
            "optional": {
                "look": (POSTFX_LOOK, {"tooltip": "A look from a PostFx Theme / Custom Look / LUT node. "
                                                  "Overrides 'theme'."}),
                "mask": ("MASK", {"tooltip": "Apply the effect only where the mask is white; blend with "
                                             "original. An all-black mask (e.g. LoadImage on an image "
                                             "without alpha) is ignored and the effect applies everywhere."}),
            },
        }

    def apply(self, image, theme, condition, strength, seed, batch_seed, look=None, mask=None):
        if look is None and theme == "none":
            return (image,)
        postfx = _postfx()
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


class PostFxTheme:
    """Emit a built-in theme as a POSTFX_LOOK so a named look can start a
    chain (Theme -> LUT / Custom Look -> Apply)."""

    CATEGORY = _CATEGORY
    FUNCTION = "load"
    RETURN_TYPES = (POSTFX_LOOK,)
    RETURN_NAMES = ("look",)
    SEARCH_ALIASES = ["BCNodes", "postfx", "theme", "look"]
    DESCRIPTION = ("Load a built-in postfx theme as a look you can pipe into "
                   "PostFx LUT / Custom Look / Apply. Use this only to build a "
                   "chain from a named theme; otherwise pick the theme on Apply.")

    @classmethod
    def INPUT_TYPES(cls):
        themes = theme_names()
        return {"required": {"theme": (themes, {"default": default_theme(themes)})}}

    def load(self, theme):
        return (_postfx().load_theme(theme_stem(theme)),)


class PostFxCustomLook:
    """Build a look from the most-used knobs, or override those knobs on top
    of an incoming look. A control at its neutral value is left untouched, so
    base look + one moved slider changes only that op."""

    CATEGORY = _CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = (POSTFX_LOOK,)
    RETURN_NAMES = ("look",)
    SEARCH_ALIASES = ["BCNodes", "postfx", "custom look", "color grade"]
    DESCRIPTION = ("Build a custom look from common controls (white balance, "
                   "exposure, contrast, vibrance/saturation, grain, vignette, "
                   "halation, clarity). With a 'look' input, only the controls "
                   "you move off neutral override it; the rest pass through.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temp": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "White balance: >0 warm, <0 cool."}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                   "tooltip": ">0 magenta, <0 green."}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05,
                                       "tooltip": "Stops of light."}),
                "contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "S-curve contrast strength."}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Lifts low-sat pixels, protects skin."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01,
                                         "tooltip": "Uniform saturation (1 = unchanged)."}),
                "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.1, "step": 0.002,
                                    "tooltip": "Luma grain amount."}),
                "grain_size": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 4.0, "step": 0.1,
                                         "tooltip": "Grain size (px @ 1024 edge)."}),
                "vignette": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Edge darkening amount."}),
                "halation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                       "tooltip": "Highlight bloom strength."}),
                "clarity": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Local contrast; <0 softens."}),
            },
            "optional": {
                "look": (POSTFX_LOOK, {"tooltip": "Base look to override on top of (e.g. a theme)."}),
            },
        }

    def build(self, temp, tint, exposure, contrast, vibrance, saturation,
              grain, grain_size, vignette, halation, clarity, look=None):
        base = copy.deepcopy(look) if look is not None else neutral_look()

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


class PostFxLut:
    """Attach a 3D .cube LUT to a look. Standalone by default; connect a look
    to ride the LUT on top of a theme."""

    CATEGORY = _CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = (POSTFX_LOOK,)
    RETURN_NAMES = ("look",)
    SEARCH_ALIASES = ["BCNodes", "postfx", "lut", "cube"]
    DESCRIPTION = ("Apply a 3D .cube LUT as a look. Select a file from the pack's "
                   "luts/ folder or give an absolute lut_path. Standalone by "
                   "default; connect a look to layer the LUT on top of it. The "
                   "LUT applies mid-pipeline, so grain/vignette still finish on top.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lut": (["none"] + lut_files(), {"default": "none",
                                                 "tooltip": "A .cube file from the pack's luts/ folder."}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Blend the LUT with its input (0 = off)."}),
            },
            "optional": {
                "lut_path": ("STRING", {"default": "",
                                        "tooltip": "Absolute path to a .cube (overrides the dropdown)."}),
                "look": (POSTFX_LOOK, {"tooltip": "Base look to layer the LUT on top of."}),
            },
        }

    def build(self, lut, intensity, lut_path="", look=None):
        base = copy.deepcopy(look) if look is not None else neutral_look()

        path = lut_path.strip()
        if not path and lut != "none":
            path = os.path.join(LUTS_DIR, lut)

        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"LUT file not found: {path!r}")
            base = merge_look(base, {"lut": {"file": path, "intensity": float(intensity)}})
            base["name"] = f"{base.get('name', 'look')}+lut"
        return (base,)


class PostFxSignatureSheet:
    """Render a labeled grid of every theme in a category applied to the
    first image (postfx `sheet`)."""

    CATEGORY = _CATEGORY
    FUNCTION = "build"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    SEARCH_ALIASES = ["BCNodes", "postfx", "contact sheet", "theme preview"]
    DESCRIPTION = ("Render a labeled contact sheet of every theme in a category "
                   "applied to the first image, as a single preview image. "
                   "'signature' is the market-standard set.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "category": (["signature", "experimental", "all"], {"default": "signature"}),
                "condition": (condition_names(), {"default": "neutral"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05}),
                "columns": ("INT", {"default": 5, "min": 1, "max": 10}),
            },
        }

    def build(self, image, category, condition, strength, columns):
        postfx = _postfx()
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


NODE_CLASS_MAPPINGS = {
    "BC_PostFxApply": PostFxApply,
    "BC_PostFxTheme": PostFxTheme,
    "BC_PostFxCustomLook": PostFxCustomLook,
    "BC_PostFxLut": PostFxLut,
    "BC_PostFxSignatureSheet": PostFxSignatureSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PostFxApply": "PostFx Apply",
    "BC_PostFxTheme": "PostFx Theme",
    "BC_PostFxCustomLook": "PostFx Custom Look",
    "BC_PostFxLut": "PostFx LUT",
    "BC_PostFxSignatureSheet": "PostFx Signature Sheet",
}

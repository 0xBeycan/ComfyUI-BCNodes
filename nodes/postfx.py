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

pipelines/postfx.py imports `postfx` (and cv2, which it brings) inside the
functions that use it: the package is a pip dependency, and a missing install
must not take the rest of the pack down with it.
"""

from ..pipelines.postfx import (
    apply_look, condition_names, custom_look, default_theme, lut_files, lut_look, postfx_package,
    signature_sheet, theme_names, theme_stem,
)

POSTFX_LOOK = "POSTFX_LOOK"

_CATEGORY = "BCNodes/postfx"


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
        return apply_look(image, theme, condition, strength, seed, batch_seed, look, mask)


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
        return (postfx_package().load_theme(theme_stem(theme)),)


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
        return custom_look(temp, tint, exposure, contrast, vibrance, saturation,
                           grain, grain_size, vignette, halation, clarity, look)


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
        return lut_look(lut, intensity, lut_path, look)


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
        return signature_sheet(image, category, condition, strength, columns)


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

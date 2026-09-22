"""Social Media Export — platform-ready derivatives of a master image.

    BC_SocialMediaExport (Social Media Export)

Takes a full-quality master image and emits one derivative per selected
platform so each platform receives an image that already matches its
aspect-ratio and size rules — sparing it from re-cropping / re-compressing.
Runs beside (never touches) the normal master save node.

This module is the ComfyUI adapter: it converts torch IMAGE tensors to PIL,
writes files and builds the report. All geometry and encoding live in
social_export_core (no ComfyUI or torch imports, unit-tested standalone);
the platform table is nodes/social_specs.json, re-read on every execution.

PIL and folder_paths are imported inside the methods; the torch IMAGE tensor
is handled by duck typing (.detach().cpu().numpy()), so torch is never
imported here.
"""

import os

import numpy as np

from . import social_export_core as core

_SPECS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "social_specs.json")

# Platforms whose checkbox starts ON when the node is first dropped in.
_DEFAULT_ON = {"instagram_feed", "instagram_story"}

# Fixed, no-brainer encoding choices (previously exposed as widgets). These are
# always the best-quality option, so there is no reason to make the user pick:
#   * 4:4:4 chroma  -> no colour subsampling artefacts
#   * progressive   -> best-practice for web delivery
_CHROMA_444 = True
_PROGRESSIVE = True
# Vertical crop bias when a crop is needed: 0=top, 0.5=center, 1=bottom.
# 0.4 is slightly top-weighted, which keeps faces/subjects in portrait crops.
_ANCHOR = 0.4


def _pretty(name: str) -> str:
    """'instagram_story' -> 'Instagram Story' for human-facing labels."""
    return name.replace("_", " ").title()


def _platform_tooltip(name: str, spec: dict) -> str:
    """Short, plain-language description of what this platform's export targets."""
    lo, hi = spec.get("ar_min"), spec.get("ar_max")
    mw, mh = spec.get("max_w"), spec.get("max_h")
    fmt = str(spec.get("format", "jpg")).upper()
    if lo == hi:
        ar = f"fixed {lo:g} aspect (w/h)"
    else:
        ar = f"{lo:g}–{hi:g} aspect (w/h)"
    return f"{_pretty(name)} — {ar}, up to {mw}×{mh}px, {fmt}."


def _tensor_to_pil(image):
    """ComfyUI IMAGE tensor [H,W,C] float 0..1 -> opaque RGB PIL image."""
    from PIL import Image

    arr = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    channels = arr.shape[2]
    if channels == 1:
        return Image.fromarray(arr[:, :, 0], "L").convert("RGB")
    if channels == 4:
        # Flatten alpha onto white; core also guards this, but keep it explicit.
        rgba = Image.fromarray(arr, "RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return Image.fromarray(arr[:, :, :3], "RGB")


class SocialMediaExport:
    """Emit platform-ready image derivatives, one file per selected platform."""

    @classmethod
    def INPUT_TYPES(cls):
        # Load the specs so the platform checkboxes are always in sync with
        # social_specs.json (add a platform there and a checkbox appears here).
        try:
            specs = core.load_specs(_SPECS_PATH)
            names = core.platform_names(specs)
        except Exception:
            specs, names = {}, []

        required = {"images": ("IMAGE",)}

        # --- one checkbox per platform (auto-generated) ----------------------
        for name in names:
            required[name] = ("BOOLEAN", {
                "default": name in _DEFAULT_ON,
                "label_on": "export",
                "label_off": "skip",
                "tooltip": _platform_tooltip(name, specs[name]),
            })

        # --- the single aspect-ratio decision, in plain language -------------
        required["resize_mode"] = (["crop", "pad"], {
            "default": "crop",
            "tooltip": (
                "How to reach each platform's shape when your image doesn't fit:\n"
                "• crop — trim the edges to fill the frame (no bars; may cut a little off)\n"
                "• pad  — keep the whole image, filling the rest with a blurred copy "
                "(nothing is cut; adds soft bars).\n"
                "Either way the platform gets an image it won't re-crop."
            ),
        })
        required["quality"] = ("INT", {
            "default": 92, "min": 1, "max": 100,
            "tooltip": (
                "JPEG/WebP quality. 92 is visually lossless; lower it only if you need "
                "smaller files. (If a platform has a byte cap, quality auto-steps down.)"
            ),
        })
        required["allow_upscale"] = ("BOOLEAN", {
            "default": False,
            "tooltip": (
                "Enlarge small images up to the platform's recommended size. "
                "Off = never upscale (safest quality)."
            ),
        })
        required["filename_prefix"] = ("STRING", {
            "default": "social/export",
            "tooltip": (
                "Output name prefix. One file per selected platform is written under "
                "ComfyUI/output (e.g. social/export_instagram_story_00001_.jpg)."
            ),
        })
        return {"required": required}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "social media export", "instagram", "tiktok", "export"]
    DESCRIPTION = (
        "Make one social-media-ready copy of your image for each platform you tick, "
        "so the platform serves your file as-is instead of re-cropping and "
        "re-compressing it.\n"
        "\n"
        "Tick the platforms you want (each checkbox shows that platform's shape and "
        "size). For every ticked platform the node reshapes the image to that "
        "platform's aspect ratio and saves a clean, high-quality JPEG/WebP under "
        "ComfyUI/output. Your original image passes straight through the 'images' "
        "output untouched, and a text 'report' lists exactly what was written.\n"
        "\n"
        "Settings:\n"
        "• resize_mode — crop (trim edges to fill, no bars) or pad (keep everything, "
        "blurred bars). This is the only aspect-ratio choice you need to make.\n"
        "• quality — encode quality, 92 is visually lossless.\n"
        "• allow_upscale — enlarge small images to the platform's recommended size.\n"
        "• filename_prefix — where/what to name the saved files.\n"
        "\n"
        "Colour (4:4:4) and progressive encoding are always on — they are simply the "
        "best-quality choice, so they aren't shown as options."
    )

    def export(self, images, resize_mode, quality, allow_upscale, filename_prefix,
               **platform_flags):
        import folder_paths

        # Fresh spec load every execution so edits to social_specs.json apply
        # without restarting ComfyUI.
        specs = core.load_specs(_SPECS_PATH)
        names = core.platform_names(specs)

        requested = [n for n in names if bool(platform_flags.get(n))]
        if not requested:
            raise ValueError(
                "SocialMediaExport: no platform selected. Turn on at least one "
                "platform checkbox (for example instagram_feed)."
            )

        # Map the single, user-facing resize_mode onto the core engine's two
        # controls. 'crop' = always crop-to-fill with no ceiling; 'pad' = never
        # crop, letterbox onto a blurred background instead. Neither can raise —
        # the old 'error' path is gone, so the node never blocks the graph.
        if resize_mode == "pad":
            overflow_strategy, max_crop_ratio = "pad", 0.0
        else:
            overflow_strategy, max_crop_ratio = "crop", 1.0

        output_dir = folder_paths.get_output_directory()
        lines = [core.REPORT_HEADER]

        for idx, image in enumerate(images):
            pil = _tensor_to_pil(image)
            for platform in requested:
                spec = specs[platform]
                result, meta = core.render(
                    pil, spec,
                    anchor=_ANCHOR,
                    allow_upscale=bool(allow_upscale),
                    max_crop_ratio=max_crop_ratio,
                    overflow_strategy=overflow_strategy,
                    platform=platform,
                )
                data, final_q, ext, exceeded = core.encode(
                    result, spec,
                    quality=int(quality),
                    chroma_444=_CHROMA_444,
                    progressive=_PROGRESSIVE,
                )
                self._write(
                    data, ext, output_dir, filename_prefix, platform,
                    result.width, result.height,
                )
                lines.append(core.format_report_line(
                    idx, meta, final_q, len(data), ext, exceeded, spec.get("max_bytes"),
                ))

        report = "\n".join(lines)
        # Pass the untouched input through; expose the report string.
        # No "ui" payload: files are on disk, the node shows no preview.
        return {"result": (images, report)}

    @staticmethod
    def _write(data, ext, output_dir, filename_prefix, platform, width, height):
        """Write bytes under the output directory, letting ComfyUI resolve the
        subfolder + auto-incrementing counter (creating directories as needed)."""
        import folder_paths

        prefix = f"{filename_prefix}_{platform}"
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            prefix, output_dir, width, height
        )
        os.makedirs(full_output_folder, exist_ok=True)
        fname = f"{filename}_{counter:05}_.{ext}"
        with open(os.path.join(full_output_folder, fname), "wb") as fh:
            fh.write(data)
        return subfolder, fname


NODE_CLASS_MAPPINGS = {"BC_SocialMediaExport": SocialMediaExport}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_SocialMediaExport": "Social Media Export"}

"""Image file writing for Save Image: the format list (the AVIF and JXL Pillow plugins join it
when installed), the prompt / workflow metadata (PNG text chunks for PNG; EXIF `Make` /
`ImageDescription` for WebP, AVIF, JXL, JPEG and JPEG 2000) and write_image, which turns the
quality into each format's encoder options. PIL is imported only inside png_info; a plugin
module is imported only when a file of its type is written.
"""

import importlib.util
import json
from pathlib import Path

BASE_EXTENSIONS = [".webp", ".png", ".jpg", ".jpeg", ".j2k", ".jp2", ".gif", ".tiff", ".bmp"]
DEFAULT_QUALITY = 90
# jpeg / webp / avif / jxl / tiff only; PNG is left out because Pillow then
# compresses at level 9 regardless of compress_level.
OPTIMIZE = True
PLUGINS = {".avif": "pillow_avif", ".jxl": "pillow_jxl"}


def output_extensions():
    exts = list(BASE_EXTENSIONS)
    for ext, module in PLUGINS.items():
        if importlib.util.find_spec(module) is not None:
            exts.insert(0, ext)
    return exts


def png_info(prompt, extra_pnginfo):
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            info.add_text(key, json.dumps(value))
    return info


def exif_bytes(img, prompt, extra_pnginfo):
    """Prompt in `Make` (0x010f), workflow in `ImageDescription` (0x010e):
    two IFD0 tags close together, which is what ComfyUI's pnginfo reader
    parses back on load for WebP."""
    exif = img.getexif()
    exif[0x010F] = "Prompt: " + json.dumps(prompt if prompt is not None else {})
    exif[0x010E] = "Workflow: " + json.dumps((extra_pnginfo or {}).get("workflow", {}))
    return exif.tobytes()


def write_image(path, img, prompt, save_metadata, extra_pnginfo, quality):
    ext = Path(path).suffix
    if ext in PLUGINS:
        importlib.import_module(PLUGINS[ext])  # registers the Pillow codec
    if quality == 0:
        quality = DEFAULT_QUALITY
    kwargs = {}
    if ext in (".avif", ".webp", ".jxl"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        if quality == 100:
            kwargs["lossless"] = True
        else:
            kwargs["quality"] = quality
        kwargs["optimize"] = OPTIMIZE
    elif ext in (".j2k", ".jp2", ".jpc", ".jpf", ".jpx", ".j2c"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        if quality < 100:
            kwargs["irreversible"] = True
        else:
            kwargs["quality"] = quality
    elif ext in (".jpg", ".jpeg"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        kwargs["subsampling"] = 0
        kwargs["quality"] = quality
        kwargs["optimize"] = OPTIMIZE
    elif ext == ".tiff":
        kwargs["optimize"] = OPTIMIZE
    elif ext in (".png", ".gif"):
        if save_metadata:
            kwargs["pnginfo"] = png_info(prompt, extra_pnginfo)
        # quality 0-90 -> compress_level 0-9; above 90 buys nothing but time
        kwargs["compress_level"] = round(min(quality, 90) / 10)
    img.save(path, **kwargs)

"""BiRefNet background removal.

    BC_BiRefNetRemoveBackground (BiRefNet Remove Background)

The architecture lives in ../birefnet and the weights are loaded
straight from safetensors into a plain nn.Module — no transformers, no
trust_remote_code, no timm. Weights come from the model authors' Hugging Face
repos through the pack's own downloader. Everything beyond torch (safetensors,
folder_paths, comfy.model_management) is imported on the first run, never at
import time.

The matte can be post-processed before it is applied: sensitivity, blur,
grow / shrink, invert, an edge refinement of the foreground colours, and a
solid background colour instead of alpha. All of it is plain torch on the
(B, 1, H, W) matte, so the whole batch goes through at once.
"""

import math
import os

import torch
import torch.nn.functional as F

# `res` is the square side the image is fed at; `backbone` is the Swin variant
# each checkpoint was trained with (the lite checkpoints use swin_v1_t). The
# file is saved as models/background_removal/<name>.safetensors — ComfyUI's own
# folder for its core BiRefNet nodes, so the same weights serve both.
MODEL_CONFIG = {
    "BiRefNet-general":      {"repo": "ZhengPeng7/BiRefNet",              "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 1024},
    "BiRefNet_512x512":      {"repo": "ZhengPeng7/BiRefNet_512x512",      "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 512},
    "BiRefNet-HR":           {"repo": "ZhengPeng7/BiRefNet_HR",           "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 2048},
    "BiRefNet-portrait":     {"repo": "ZhengPeng7/BiRefNet-portrait",     "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 1024},
    "BiRefNet-matting":      {"repo": "ZhengPeng7/BiRefNet-matting",      "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 1024},
    "BiRefNet-HR-matting":   {"repo": "ZhengPeng7/BiRefNet_HR-matting",   "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 2048},
    "BiRefNet_lite":         {"repo": "ZhengPeng7/BiRefNet_lite",         "file": "model.safetensors",            "backbone": "swin_v1_t", "res": 1024},
    "BiRefNet_lite-2K":      {"repo": "ZhengPeng7/BiRefNet_lite-2K",      "file": "model.safetensors",            "backbone": "swin_v1_t", "res": 2048},
    "BiRefNet_dynamic":      {"repo": "ZhengPeng7/BiRefNet_dynamic",      "file": "model.safetensors",            "backbone": "swin_v1_l", "res": 1024},
    "BiRefNet_lite-matting": {"repo": "ZhengPeng7/BiRefNet_lite-matting", "file": "model.safetensors",            "backbone": "swin_v1_t", "res": 1024},
    "Lucida":                {"repo": "egeorcun/lucida",                  "file": "lucida-m35-comfy.safetensors", "backbone": "swin_v1_l", "res": 1024},
}

FOLDER_KEY = "background_removal"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _model_dir():
    import folder_paths

    path = os.path.join(folder_paths.models_dir, FOLDER_KEY)
    # ComfyUI registers this key itself (2026-05, nodes_bg_removal); on older
    # builds the call adds it, on current ones it is a no-op.
    folder_paths.add_model_folder_path(FOLDER_KEY, path)
    return path


def _weights_path(name):
    import folder_paths

    from .downloader import download_file

    primary = _model_dir()
    filename = f"{name}.safetensors"
    found = folder_paths.get_full_path(FOLDER_KEY, filename)
    if found is not None:
        return found

    cfg = MODEL_CONFIG[name]
    url = f"https://huggingface.co/{cfg['repo']}/resolve/main/{cfg['file']}"
    path = os.path.join(primary, filename)
    state = {"step": -1}

    def on_progress(downloaded, total):
        step = int(10 * downloaded / total) if total else 0
        if step != state["step"]:
            state["step"] = step
            print(f"[BCNodes] {filename} {step * 10}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)")

    print(f"[BCNodes] downloading {url} -> {path}")
    download_file(url, path, on_progress)
    return path


class _Loaded:
    name = None
    model = None
    device = None
    dtype = None


def _load(name):
    if _Loaded.name == name and _Loaded.model is not None:
        return _Loaded.model, _Loaded.device, _Loaded.dtype

    import comfy.model_management as mm
    from safetensors.torch import load_file

    from ..birefnet import BiRefNet

    if _Loaded.model is not None:
        _Loaded.model = None
        _Loaded.name = None
        mm.soft_empty_cache()

    cfg = MODEL_CONFIG[name]
    path = _weights_path(name)
    model = BiRefNet(cfg["backbone"])
    model.load_state_dict(load_file(path), strict=True)
    model.eval()

    device = mm.get_torch_device()
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype)

    _Loaded.name, _Loaded.model, _Loaded.device, _Loaded.dtype = name, model, device, dtype
    return model, device, dtype


def _blur(mask, radius):
    """Gaussian blur of a (B, 1, H, W) matte; `radius` is the sigma in pixels."""
    sigma = float(radius)
    size = int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(size, dtype=mask.dtype, device=mask.device) - size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    pad = size // 2
    out = F.pad(mask, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(out, kernel.view(1, 1, 1, size))
    return F.conv2d(out, kernel.view(1, 1, size, 1))


def _offset(mask, steps):
    """Grow (steps > 0) or shrink (steps < 0) a matte by one pixel per step."""
    for _ in range(abs(steps)):
        mask = F.max_pool2d(mask, 3, 1, 1) if steps > 0 else -F.max_pool2d(-mask, 3, 1, 1)
    return mask


def _refine_foreground(rgb, mask):
    """Edge refinement: the matte is hardened at 0.45, its transition band is
    blended with a 3x3-blurred copy and slightly darkened, and the colours are
    scaled by that refined matte. `rgb` is (B, H, W, 3), `mask` (B, 1, H, W)."""
    binary = (mask > 0.45).to(mask.dtype)
    k = torch.tensor([0.25, 0.5, 0.25], dtype=mask.dtype, device=mask.device)
    edge = F.pad(binary, (1, 1, 1, 1), mode="replicate")
    edge = F.conv2d(edge, k.view(1, 1, 1, 3))
    edge = F.conv2d(edge, k.view(1, 1, 3, 1))
    refined = torch.where((mask > 0.05) & (mask < 0.95), 0.85 * mask + 0.15 * edge, binary)
    refined = torch.where((mask > 0.2) & (mask < 0.8), refined * 0.98, refined)
    return rgb * refined.permute(0, 2, 3, 1)


def _parse_color(value):
    """'#rrggbb' or '#rrggbbaa' (also the 3-digit short form) -> (r, g, b, a) in 0..1."""
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) not in (6, 8) or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise ValueError(f"background_color must be #rrggbb or #rrggbbaa, got {value!r}")
    channels = [int(text[i:i + 2], 16) / 255.0 for i in range(0, len(text), 2)]
    if len(channels) == 3:
        channels.append(1.0)
    return channels


def finish(rgb, mask, sensitivity=1.0, mask_blur=0, mask_offset=0, invert_output=False,
           refine_foreground=False, background="Alpha", background_color="#222222"):
    """Apply the options to a raw matte and build the three outputs.
    `rgb` is (B, H, W, 3) in 0..1, `mask` (B, H, W) in 0..1, both on the CPU."""
    m = mask.unsqueeze(1).float()
    if sensitivity < 1.0:
        m = (m * (1 + (1 - sensitivity))).clamp_(0, 1)
    if mask_blur > 0:
        m = _blur(m, mask_blur)
    if mask_offset != 0:
        m = _offset(m, mask_offset)
    if invert_output:
        m = 1 - m
    color = rgb.float()
    if refine_foreground:
        color = _refine_foreground(color, m)
    alpha = m.permute(0, 2, 3, 1)  # (B, H, W, 1)
    if background == "Alpha":
        image = torch.cat((color, alpha), dim=-1)
    else:
        r, g, b, a = _parse_color(background_color)
        bg = torch.tensor([r, g, b], dtype=color.dtype).view(1, 1, 1, 3)
        # Straight-alpha "over" onto a background that may itself be
        # translucent, then the alpha is dropped.
        weight = a * (1 - alpha)
        total = alpha + weight
        image = torch.where(total > 0, (color * alpha + bg * weight) / total.clamp(min=1e-6), torch.zeros_like(color))
    mask_out = m[:, 0]
    mask_image = mask_out.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()
    return image, mask_out, mask_image


class BiRefNetRemoveBackground:
    """IMAGE in -> image (RGBA with alpha = matte, or RGB over a solid colour),
    MASK and the mask as an RGB image. The model is fetched from Hugging Face
    on first use and kept loaded between runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (list(MODEL_CONFIG.keys()), {"default": "BiRefNet-general"}),
            },
            "optional": {
                "sensitivity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Below 1 the matte is amplified, so faint areas count as foreground."}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                                      "tooltip": "Gaussian blur of the matte edges, in pixels."}),
                "mask_offset": ("INT", {"default": 0, "min": -20, "max": 20, "step": 1,
                                        "tooltip": "Grow (positive) or shrink (negative) the matte, one pixel per step."}),
                "invert_output": ("BOOLEAN", {"default": False, "tooltip": "Keep the background instead of the subject."}),
                "refine_foreground": ("BOOLEAN", {"default": False,
                                                  "tooltip": "Harden the matte edge and scale the colours by it, for cleaner cut-outs on transparent output."}),
                "background": (["Alpha", "Color"], {"default": "Alpha",
                                                    "tooltip": "Alpha: RGBA output with the matte as alpha. Color: RGB output over background_color."}),
                "background_color": ("COLOR", {"default": "#222222", "tooltip": "Used when background is Color."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "remove_background"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ['BCNodes', 'birefnet', 'remove background', 'rmbg', 'matting']

    def remove_background(self, image, model, sensitivity=1.0, mask_blur=0, mask_offset=0, invert_output=False,
                          refine_foreground=False, background="Alpha", background_color="#222222"):
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 4)), torch.zeros((0, 64, 64)), torch.zeros((0, 64, 64, 3)))
        net, device, dtype = _load(model)
        res = MODEL_CONFIG[model]["res"]
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

        rgb = image[..., :3]
        masks = []
        with torch.no_grad():
            for img in rgb:
                x = img.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
                h, w = x.shape[-2:]
                x = F.interpolate(x, size=(res, res), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
                x = ((x - mean) / std).to(dtype)
                pred = net(x).float().sigmoid()
                pred = F.interpolate(pred, size=(h, w), mode="bicubic", align_corners=False, antialias=True).clamp_(0, 1)
                masks.append(pred[0, 0].cpu())

        mask = torch.stack(masks, dim=0)
        return finish(rgb.cpu(), mask, sensitivity, mask_blur, mask_offset, invert_output,
                      refine_foreground, background, background_color)


NODE_CLASS_MAPPINGS = {
    "BC_BiRefNetRemoveBackground": BiRefNetRemoveBackground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_BiRefNetRemoveBackground": "BiRefNet Remove Background",
}

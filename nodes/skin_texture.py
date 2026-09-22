"""Skin Texture — micro-texture on skin, masked by SAM 3.

    BC_SkinTexture (Skin Texture)

Rendered skin comes out as a smooth gradient; grain on top of it reads as
noise over plastic. This node puts a surface under the grain: inside a skin
mask it (a) boosts the image's own high-frequency detail and (b) multiplies
in a synthetic pore field — band-passed noise at pore scale with sparse
darker pits — in linear light, so the texture follows the shading instead
of sitting on it like a sticker. Highlights and deep shadows get less, and
the whole effect fades when the face is small in the frame (pore-scale
detail has nowhere to live there).

The mask comes from SAM 3 (the multiplex checkpoint under models/checkpoints,
downloaded on first use when missing): "skin" minus "eyes / eyebrows / lips
/ teeth". A connected `mask` replaces the skin detection; `exclude_mask` is
subtracted either way. Nothing here moves geometry — moles and marks live
in the low band the node never touches. Everything beyond torch (SAM 3,
folder_paths, comfy.*) is imported on the first run.
"""

import math
import os

import torch
import torch.nn.functional as F

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
DEFAULT_SAM3_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
SKIN_PROMPT = "skin:6"
EXCLUDE_PROMPT = "eyes:4, eyebrows:4, lips:2, teeth:2"
FACE_PROMPT = "face:2"

# Pore field geometry, in px at a 1024 px long edge (scaled with the image).
PORE_SIGMA = 0.9          # fine octave
PORE_SIGMA_COARSE = 2.2   # coarse octave
DETAIL_SIGMA = 2.0        # high-pass radius for the self-detail boost
MAX_AMPLITUDE = 0.06      # luminance modulation at texture = 1.0 (unit-std field)
FACE_FULL_FRACTION = 0.30  # face height / frame height at which texture is at full strength


# --- SAM 3 -------------------------------------------------------------------

def _sam3_choices():
    import folder_paths

    names = list(folder_paths.get_filename_list("checkpoints"))
    if DEFAULT_SAM3 not in names:
        names.insert(0, DEFAULT_SAM3)
    return names


def _sam3_path(name):
    import folder_paths

    found = folder_paths.get_full_path("checkpoints", name)
    if found is not None:
        return found
    if name != DEFAULT_SAM3:
        raise FileNotFoundError(f"SAM 3 checkpoint {name} is not in models/checkpoints")

    from .downloader import download_file

    path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
    state = {"step": -1}

    def on_progress(downloaded, total):
        step = int(10 * downloaded / total) if total else 0
        if step != state["step"]:
            state["step"] = step
            print(f"[BCNodes] {name} {step * 10}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)")

    print(f"[BCNodes] downloading {DEFAULT_SAM3_URL} -> {path}")
    download_file(DEFAULT_SAM3_URL, path, on_progress)
    return path


class _Sam3:
    name = None
    model = None
    clip = None


def _load_sam3(name):
    if _Sam3.name == name and _Sam3.model is not None:
        return _Sam3.model, _Sam3.clip
    import comfy.sd

    _Sam3.name, _Sam3.model, _Sam3.clip = None, None, None
    loaded = comfy.sd.load_checkpoint_guess_config(_sam3_path(name), output_vae=False, output_clip=True)
    _Sam3.name, _Sam3.model, _Sam3.clip = name, loaded[0], loaded[1]
    return _Sam3.model, _Sam3.clip


def _detect(model, clip, image, text, threshold):
    """SAM 3 text detection through the core node: -> (B, H, W) union mask, per-frame boxes."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    out = SAM3_Detect.execute(model, image, conditioning=cond, threshold=threshold, refine_iterations=2, individual_masks=False)
    masks, boxes = out.args
    return masks.float().cpu(), boxes


# --- texture -----------------------------------------------------------------

def _gauss(x, sigma):
    """Separable Gaussian blur of (B, C, H, W), reflect-padded; sigma in px."""
    if sigma <= 0.2:
        return x
    radius = int(math.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


def _srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=0) ** (1 / 2.4) - 0.055)


def _smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _unit(x):
    return x / (x.flatten(1).std(dim=1).view(-1, 1, 1, 1) + 1e-8)


def pore_field(shape, scale, seed, device):
    """Unit-std pore field (B, 1, H, W): two band-passed noise octaves plus
    sparse darker pits. `scale` is px per reference px (long edge / 1024)."""
    b, _, h, w = shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn((b, 1, h, w), generator=gen).to(device)
    fine = _unit(_gauss(noise, PORE_SIGMA * scale * 0.55) - _gauss(noise, PORE_SIGMA * scale * 1.6))
    coarse = _unit(_gauss(noise, PORE_SIGMA_COARSE * scale * 0.55) - _gauss(noise, PORE_SIGMA_COARSE * scale * 1.6))
    field = _unit(0.65 * fine + 0.35 * coarse)
    pits = -F.relu(coarse - 1.25) * 1.5
    return _unit(field + pits)


def face_gate(face_mask, height):
    """1.0 when the tallest face box spans FACE_FULL_FRACTION of the frame,
    fading linearly to 0 as the face gets smaller; 1.0 when there is no face
    (a body-only frame keeps its texture)."""
    if face_mask is None or not bool(face_mask.any()):
        return 1.0
    rows = face_mask.flatten(0, 0).any(dim=-1).any(dim=0) if face_mask.ndim == 3 else face_mask.any(dim=-1)
    ys = torch.where(rows)[0]
    face_h = float(ys.max() - ys.min() + 1) / float(height)
    return float(min(1.0, face_h / FACE_FULL_FRACTION))


def apply_texture(image, mask, texture=0.5, detail=0.6, pore_scale=1.0, seed=0, gate=1.0):
    """image (B, H, W, 3) sRGB, mask (B, H, W) 0..1 -> textured image, same shape."""
    b, h, w, _ = image.shape
    device = image.device
    scale = max(h, w) / 1024.0 * float(pore_scale)
    x = image[..., :3].permute(0, 3, 1, 2).float()
    lin = _srgb_to_linear(x.clamp(0, 1))
    lum = (lin * torch.tensor([0.2126, 0.7152, 0.0722], device=device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    m = mask.to(device).float().unsqueeze(1).clamp(0, 1)

    # where the texture lands: the mask, less in highlights, gone in the black
    weight = m * _smoothstep(0.01, 0.08, lum) * (1.0 - 0.7 * _smoothstep(0.55, 0.95, lum)) * float(gate)

    # (a) the image's own detail, boosted (a ratio, so colour is untouched)
    hf = lum - _gauss(lum, DETAIL_SIGMA * scale)
    gain = (1.0 + float(detail) * hf / (lum + 0.02)).clamp(0.5, 1.5)
    lin = lin * (1.0 + (gain - 1.0) * weight)

    # (b) the synthetic pore field, multiplicative in linear light
    if texture > 0:
        field = pore_field(lum.shape, scale, seed, device)
        lin = lin * (1.0 + MAX_AMPLITUDE * float(texture) * field * weight)

    out = _linear_to_srgb(lin.clamp(0, 1)).clamp(0, 1)
    return out.permute(0, 2, 3, 1).to(image.dtype)


# --- node --------------------------------------------------------------------

class SkinTexture:
    """Micro-texture inside a SAM 3 skin mask (or a connected mask)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam3_model": (_sam3_choices(), {"default": DEFAULT_SAM3,
                                                "tooltip": "SAM 3 checkpoint under models/checkpoints; the default is downloaded when missing. "
                                                           "Unused when a mask is connected."}),
                "texture": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Synthetic pore field strength. 0 = off; 1 = ±6% luminance modulation."}),
                "detail": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 2.0, "step": 0.05,
                                     "tooltip": "Gain on the image's own fine detail inside the mask (0 = off)."}),
                "pore_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 4.0, "step": 0.05,
                                         "tooltip": "Pore size multiplier. 1 = ~1 px pores at a 1024 px long edge, scaled with the image."}),
                "feather": ("INT", {"default": 8, "min": 0, "max": 64,
                                    "tooltip": "Mask edge softness in px (at a 1024 px long edge)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "SAM 3 detection threshold."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Where to texture. Connected, it replaces the SAM 3 skin detection."}),
                "exclude_mask": ("MASK", {"tooltip": "Subtracted from the skin mask (added to the eyes / lips exclusion)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "run"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "skin texture", "pores", "micro texture", "skin detail", "sam3"]
    DESCRIPTION = ("Adds micro-texture to skin: boosts the image's own fine detail and multiplies in a synthetic pore "
                   "field, in linear light, only inside a SAM 3 skin mask (eyes, eyebrows, lips and teeth excluded). "
                   "Put it before upscaling and before grain. Connect a mask to skip the detection.")

    def run(self, image, sam3_model, texture, detail, pore_scale, feather, seed, threshold, mask=None, exclude_mask=None):
        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        b, h, w, _ = image.shape
        gate = 1.0

        if mask is not None:
            skin = mask.float().reshape(-1, mask.shape[-2], mask.shape[-1])
            if skin.shape[-2:] != (h, w):
                skin = F.interpolate(skin.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
            skin = skin[:b] if skin.shape[0] >= b else skin.expand(b, -1, -1)
        else:
            model, clip = _load_sam3(sam3_model)
            skin, _ = _detect(model, clip, image, SKIN_PROMPT, threshold)
            drop, _ = _detect(model, clip, image, EXCLUDE_PROMPT, threshold)
            face, _ = _detect(model, clip, image, FACE_PROMPT, threshold)
            skin = (skin - drop).clamp(0, 1)
            gate = min(face_gate(face[i], h) for i in range(b))

        if exclude_mask is not None:
            ex = exclude_mask.float().reshape(-1, exclude_mask.shape[-2], exclude_mask.shape[-1])
            if ex.shape[-2:] != (h, w):
                ex = F.interpolate(ex.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
            ex = ex[:b] if ex.shape[0] >= b else ex.expand(b, -1, -1)
            skin = (skin - ex.to(skin.device)).clamp(0, 1)

        skin = skin.to(image.device)
        if feather > 0:
            skin = _gauss(skin.unsqueeze(1), feather * max(h, w) / 1024.0).squeeze(1).clamp(0, 1)

        out = apply_texture(image, skin, texture=texture, detail=detail, pore_scale=pore_scale, seed=seed, gate=gate)
        return (out, skin.cpu())


NODE_CLASS_MAPPINGS = {
    "BC_SkinTexture": SkinTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_SkinTexture": "Skin Texture",
}

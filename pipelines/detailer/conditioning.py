"""The conditioning and the model a crop is sampled with: every ControlNet in them cut to the crop, then
core's inpaint conditioning of the crop. comfy.controlnet and node_helpers are imported on first use.

Every ControlNet core has keeps its control image whole and lays it onto the latent at sampling time
(comfy.utils.common_upscale to the latent's pixel size, then the VAE encode where the ControlNet needs one), so
a copy holding the image cut to the crop is all a crop needs: core scales the cut to the crop's sampling size
as it always does. Two places hold them:

- The conditioning (core's Apply ControlNet and the nodes built on it): a chain of ControlBase objects
  (comfy/controlnet.py: ControlNet, QwenFunControlNet, ControlLora, ControlNetSD35, T2IAdapter), each with its
  hint in cond_hint_original and any mask in extra_concat_orig, both laid on through common_upscale's
  "center" cut. Each link is copied through its own copy(), which keeps its settings (crop_conditioning).
- The model (core's model-patch ControlNets, PATCHES): a patch object in the model's patch lists, its replace
  patches or its wrappers, holding its images and masks and what it computed from them at one latent size.
  Each such patch is copied with its images cut and that state dropped, and every object that reaches it
  (SATELLITES, a bound method of it) is pointed at the copy, on a clone of the model (crop_model).

A ControlNet of another kind cannot be cut: the Detailer raises an error that names its class instead of
sampling a crop with the whole image's hint.
"""

import copy

# How core lays a field onto the latent: (channels last, (..., H, W, C), else (..., H, W); common_upscale's
# "center" cut, else scaled as it is).
HWC_CENTRE, HW_CENTRE, HWC_STRETCH, HW_STRETCH = (True, True), (False, True), (True, False), (False, False)

# Core's model-patch ControlNets, by class name: (each image or mask field and how core lays it onto the
# latent, the fields holding what core computed from them -> the value that makes the copy compute it again).
PATCHES = {
    # comfy_extras/nodes_model_patch.py. Z-Image's Fun ControlNet (ZImageFunControlnet); mask: core's (B, 1, 1, H, W)
    # of 1 - mask
    "ZImageControlPatch": ({"image": HWC_CENTRE, "inpaint_image": HWC_CENTRE, "mask": HW_CENTRE},
                           {"encoded_image": None, "temp_data": None}),
    # Qwen-Image's DiffSynth ControlNets (QwenImageDiffsynthControlnet); mask as Z-Image's. Its to() moves
    # encoded_image with no None check, so the size it was encoded at is dropped instead: no latent matches it
    "DiffSynthCnetPatch": ({"image": HWC_CENTRE, "mask": HW_STRETCH}, {"encoded_image_size": None}),
    # Qwen-Image 2.1's Fun ControlNet (ZImageFunControlnet): a wrapper and block patches; mask: the node's
    "QwenImage21FunControlPatch": ({"image": HWC_STRETCH, "inpaint_image": HWC_STRETCH, "mask": HW_STRETCH},
                                   {"control": None, "stream": None, "pristine": None}),
    # Wan's Uni3C ControlNet (WanUni3CControlnetApply)
    "WanUni3CCnetPatch": ({"render_video": HWC_CENTRE}, {"prepared_render": None, "temp_data": None}),
    # comfy/ldm/anima/lllite.py. Anima's LLLite (AnimaLLLiteApply): recomputed on every call
    "AnimaLLLitePatch": ({"image": HWC_CENTRE, "mask": HW_CENTRE}, {}),
    # comfy_extras/nodes_minimax_h3.py. MiniMax H3's Fun ControlNet (MiniMaxH3FunControlNetApply): a wrapper and
    # block patches; the videos (T, C, H, W)
    "MiniMaxH3FunControlPatch": ({"control_video": HW_CENTRE, "mask": HW_CENTRE, "source_video": HW_CENTRE},
                                 {"control_latent": None, "control_latent_shape": None, "control_stream": None,
                                  "pristine_stream": None}),
}

# Core's objects that reach a PATCHES kind, by class name: the fields that do.
SATELLITES = {
    "QwenImage21FunControlBlockPatch": ("control_patch", "previous"),
    "MiniMaxH3FunControlBlockPatch": ("control_patch", "previous"),
    "AnimaLLLiteAttentionPatch": ("patch",),
    "AnimaLLLiteMLPPatch": ("patch",),
}

# Core's model patches that hold the whole image in a form no cut reaches, by class name: why.
UNCUT = {
    # comfy/ldm/supir/supir_patch.py (SUPIRApply)
    "SUPIRPatch": "it holds the latent of the whole image, encoded when the patch is applied",
}


def _cut(tensor, crop, image_width, image_height, centre=True, channels_last=False):
    """The part of a control image (..., h, w) ((..., h, w, c) with channels_last) under `crop` of the image.
    Core lays the whole control image onto the image through common_upscale: with `centre`, its centre cut to
    the image's aspect, then scaled; without, scaled as it is. The crop's edges are mapped through the same."""
    h, w = tensor.shape[-3:-1] if channels_last else tensor.shape[-2:]
    x = y = 0
    if centre:
        hint_aspect, image_aspect = w / h, image_width / image_height
        if hint_aspect > image_aspect:
            x = round((w - w * (image_aspect / hint_aspect)) / 2)
        elif hint_aspect < image_aspect:
            y = round((h - h * (hint_aspect / image_aspect)) / 2)
    sx, sy = (w - 2 * x) / image_width, (h - 2 * y) / image_height
    left, top = x + round(crop.x0 * sx), y + round(crop.y0 * sy)
    right, bottom = max(left + 1, x + round(crop.x1 * sx)), max(top + 1, y + round(crop.y1 * sy))
    return tensor[..., top:bottom, left:right, :] if channels_last else tensor[..., top:bottom, left:right]


def _crop_control(control, crop, image_width, image_height, done):
    """A copy of the ControlNet chain `control` (each link through its own copy()) with every hint and
    spatial extra concat cut to `crop`; a 1x1 extra concat (core's concat_mask placeholder) is kept.
    `done` maps each link already cut to its copy. A link that is no ControlBase is an error."""
    if control is None:
        return None
    from comfy.controlnet import ControlBase

    if not isinstance(control, ControlBase):
        raise ValueError(f"Detailer: the conditioning holds a ControlNet of class {type(control).__name__}, which the Detailer "
                         "cannot cut to its crops (it cuts core's ControlNets, comfy.controlnet.ControlBase). Apply the "
                         "ControlNet with a core ControlNet node, or connect a conditioning without it.")
    if control not in done:
        c = control.copy()
        c.cond_hint_original = _cut(control.cond_hint_original, crop, image_width, image_height)
        c.extra_concat_orig = [e if e.shape[-2:] == (1, 1) else _cut(e, crop, image_width, image_height)
                               for e in control.extra_concat_orig]
        c.set_previous_controlnet(_crop_control(control.previous_controlnet, crop, image_width, image_height, done))
        done[control] = c
    return done[control]


def crop_conditioning(conditioning, crop, image_width, image_height, done):
    """`conditioning` with every ControlNet chain in it cut to `crop` of an image_width x image_height
    image; the input is left as it is. `done` ({}, then handed to the negative too) keeps one copy of a
    chain the positive and the negative share, as core's Apply ControlNet makes it."""
    out = []
    for t in conditioning:
        extra = t[1]
        if extra.get("control") is not None:
            extra = dict(extra, control=_crop_control(extra["control"], crop, image_width, image_height, done))
        out.append([t[0], extra])
    return out


def _crop_patch(patch, crop, image_width, image_height):
    """A copy of the model-patch ControlNet `patch` (a PATCHES kind) with each image and mask (None when not
    given) cut to `crop` and what core computed from them dropped; every other field (strength, sigma range,
    VAE, ControlNet model) as core set it. A copy: core shares one patch object between a model and every
    clone of it, and the patch keeps what it computes in itself. Dropped: core keeps one result per latent
    size, which another crop sampled at the same size would reuse, so the copy computes it from the cut at
    the crop's sampling size on its first call."""
    fields, computed = PATCHES[type(patch).__name__]
    c = copy.copy(patch)
    for name, (channels_last, centre) in fields.items():
        t = getattr(patch, name)
        setattr(c, name, None if t is None else _cut(t, crop, image_width, image_height, centre, channels_last))
    for name, value in computed.items():
        setattr(c, name, value)
    return c


def _slots(model):
    """Every (container, key) where `model` keeps a patch: transformer_options' patch lists and replace
    patches, and the model's wrappers."""
    to = model.model_options["transformer_options"]
    containers = [*to.get("patches", {}).values(), *to.get("patches_replace", {}).values()]
    containers += [entries for by_key in model.wrappers.values() for entries in by_key.values()]
    return [(c, key) for c in containers for key in (list(c) if isinstance(c, dict) else range(len(c)))]


def crop_model(model, crop, image_width, image_height):
    """`model` itself when no model-patch ControlNet patches it; otherwise a clone of it (Differential
    Diffusion and every other patch kept) with each one replaced by its copy cut to `crop` (_crop_patch), the
    same copy wherever the patch is reached: in every list that holds it, behind a bound method of it, and
    in a copy of each SATELLITES object that reaches it. The input model and its patches are left as they
    are. A patch of an UNCUT kind is an error.

    Patches are found by their class name: core loads each comfy_extras module under its file path
    (nodes.py load_custom_node), so an import of comfy_extras.nodes_model_patch would make a second class
    no patch is an instance of."""
    done = {}  # id(object) -> its copy (by id: another pack's patch may be unhashable)

    def swap(entry):
        owner = getattr(entry, "__self__", None)  # core registers some patches as their bound methods
        target = entry if owner is None else owner
        name = type(target).__name__
        if name in UNCUT:
            raise ValueError(f"Detailer: the model has core's {name}, which the Detailer cannot use: {UNCUT[name]}, and "
                             "each crop is sampled at a size of its own. Connect a model without it.")
        if name not in PATCHES and name not in SATELLITES:
            return tuple(swap(e) for e in entry) if isinstance(entry, tuple) else entry  # (block class, patch) pairs
        if id(target) not in done:
            if name in PATCHES:
                done[id(target)] = _crop_patch(target, crop, image_width, image_height)
            else:
                c = done[id(target)] = copy.copy(target)
                for field in SATELLITES[name]:
                    setattr(c, field, swap(getattr(target, field)))
        c = done[id(target)]
        return c if owner is None else getattr(c, entry.__name__)

    for container, key in _slots(model):
        swap(container[key])
    if not done:
        return model
    clone = model.clone()
    for container, key in _slots(clone):
        container[key] = swap(container[key])
    return clone


def inpaint(positive, negative, vae, pixels, mask):
    """Core's InpaintModelConditioning (nodes.py, which this pack may not import) for `pixels` (1, H, W, 3),
    H and W multiples of 8 (so without its centre cut), and `mask` (1, 1, H, W): (positive, negative, the
    latent of pixels). The concat latent holds the pixels with those under the rounded mask grey;
    concat_mask is the mask itself. Only a model with inpaint channels reads either
    (comfy/model_base.py concat_cond)."""
    import node_helpers

    masked = (pixels - 0.5) * (1.0 - mask.round())[:, 0, :, :, None] + 0.5
    values = {"concat_latent_image": vae.encode(masked), "concat_mask": mask}
    return (node_helpers.conditioning_set_values(positive, values), node_helpers.conditioning_set_values(negative, values),
            vae.encode(pixels))

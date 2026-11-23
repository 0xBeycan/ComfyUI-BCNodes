"""The conditioning a crop is sampled with: every ControlNet in it cut to the crop, then core's inpaint
conditioning of the crop. node_helpers is imported on first use.

A ControlNet applied with core's Apply ControlNet (Qwen-Image's Fun ControlNet Union among them,
comfy/controlnet.py QwenFunControlNet) sits in the conditioning as a chain of ControlBase objects, each
holding its hint image in cond_hint_original. Core scales that hint onto the latent at sampling time
(ControlNet.get_control: comfy.utils.common_upscale(..., "center"), then the VAE encode when the
ControlNet needs one), so a hint cut to the crop is all a crop needs: core does the rest.
"""


def _cut(tensor, crop, image_width, image_height):
    """The part of a hint (or an extra concat) (..., h, w) under `crop` of the image. Core lays the whole
    hint onto the image through common_upscale's "center" cut (the hint's centre cut to the image's
    aspect, then scaled), so the crop's edges are mapped through that cut."""
    h, w = tensor.shape[-2:]
    hint_aspect, image_aspect = w / h, image_width / image_height
    x = y = 0
    if hint_aspect > image_aspect:
        x = round((w - w * (image_aspect / hint_aspect)) / 2)
    elif hint_aspect < image_aspect:
        y = round((h - h * (hint_aspect / image_aspect)) / 2)
    sx, sy = (w - 2 * x) / image_width, (h - 2 * y) / image_height
    left, top = x + round(crop.x0 * sx), y + round(crop.y0 * sy)
    right, bottom = max(left + 1, x + round(crop.x1 * sx)), max(top + 1, y + round(crop.y1 * sy))
    return tensor[..., top:bottom, left:right]


def _crop_control(control, crop, image_width, image_height, done):
    """A copy of the ControlNet chain `control` (each link through its own copy()) with every hint and
    spatial extra concat cut to `crop`; a 1x1 extra concat (core's concat_mask placeholder) is kept.
    `done` maps each link already cut to its copy."""
    if control is None:
        return None
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

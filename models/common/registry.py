"""Model registry: a family key maps model names to their entries, in registration order.

A model package registers its members when it is imported; models/__init__.py imports every
such package, so any import of this module has filled the registry before a lookup. Node combo
lists come from names(), name lookups from get().
"""

MATTING = "matting"
# A depth entry is a zero-argument loader: it loads its model (kept between runs) and returns
# predict(frame, resolution, size) -> the relative inverse depth (near = larger) of one (H, W, C)
# frame, run at short side `resolution` and resampled to size = (height, width), float32 on the
# compute device.
DEPTH = "depth"
# An LM family entry is a models.common.lm.family.LMFamily: one family of chat models (its folder, built-in
# catalog, prompt templates, image preparation and output split), served by one LM node.
LM_FAMILY = "lm_family"
# An LM backend entry is an object of the models.common.lm.backend.LMBackend protocol: what loads a model
# file, applies a LoRA, generates and unloads. A catalog model names its backend.
LM_BACKEND = "lm_backend"
# A VAE entry is load(path) -> (vae, note): the file loaded as core's comfy.sd.VAE, with what the
# model's special files need done to it; note says what was done ("" for the model's plain VAE).
# A file that is not a VAE of the entry's model raises ValueError naming it.
VAE = "vae"
# A latent upscale entry is upscale(samples, path) -> samples: a latent tensor of the entry's model, larger on
# height and width by the weights file at path, in the input's dtype on ComfyUI's intermediate device. A file
# whose weights do not fit the entry's model raises ValueError naming it.
LATENT_UPSCALE = "latent_upscale"
# A DyPE entry is a model's DyPE adapter (models.common.dype.Adapter): which ModelPatchers it takes (matches), their
# DyPE values (defaults), the latent's px per cell, the diffusion model's rotary embedder
# attribute, which ControlNet models carry their own (own_embedder), and the embedder of one call.
# models/common/dype.py picks the first entry that takes the model; its name is DyPE Config's model.
DYPE = "dype"

_FAMILIES = {}


def register(family: str, name: str, model: object) -> None:
    members = _FAMILIES.setdefault(family, {})
    if name in members:
        raise ValueError(f"{family} model {name!r} is already registered")
    members[name] = model


def names(family: str) -> list[str]:
    """A new list of the family's names, in registration order; KeyError for an unknown family."""
    return list(_FAMILIES[family])


def get(family: str, name: str) -> object:
    """The entry registered under name; KeyError(name) for an unknown name."""
    return _FAMILIES[family][name]

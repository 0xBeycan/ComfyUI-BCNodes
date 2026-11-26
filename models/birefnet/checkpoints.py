"""The BiRefNet checkpoints, one typed row each, registered under the matting family."""

from dataclasses import dataclass

from ..common.registry import MATTING, register


@dataclass(frozen=True)
class BiRefNetCheckpoint:
    name: str
    backbone: str
    res: int


# `res` is the square side the image is fed at; `backbone` is the Swin variant
# each checkpoint was trained with (the lite checkpoints use swin_v1_t). The
# file is beycanai/BiRefNet's <name>.safetensors (an unchanged copy of the
# authors' checkpoint; the sources are in that repository's README), saved as
# models/background_removal/<name>.safetensors — ComfyUI's own folder for its
# core BiRefNet nodes, so the same weights serve both.
CHECKPOINTS = (
    BiRefNetCheckpoint(name="BiRefNet-general",      backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_512x512",      backbone="swin_v1_l", res=512),
    BiRefNetCheckpoint(name="BiRefNet-HR",           backbone="swin_v1_l", res=2048),
    BiRefNetCheckpoint(name="BiRefNet-portrait",     backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet-matting",      backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet-HR-matting",   backbone="swin_v1_l", res=2048),
    BiRefNetCheckpoint(name="BiRefNet_lite",         backbone="swin_v1_t", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_lite-2K",      backbone="swin_v1_t", res=2048),
    BiRefNetCheckpoint(name="BiRefNet_dynamic",      backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_lite-matting", backbone="swin_v1_t", res=1024),
    BiRefNetCheckpoint(name="Lucida",                backbone="swin_v1_l", res=1024),
)

for _checkpoint in CHECKPOINTS:
    register(MATTING, _checkpoint.name, _checkpoint)

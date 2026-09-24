"""The BiRefNet checkpoints, one typed row each, registered under the matting family."""

from dataclasses import dataclass

from ..common.registry import MATTING, register


@dataclass(frozen=True)
class BiRefNetCheckpoint:
    name: str
    repo: str
    file: str
    backbone: str
    res: int


# `res` is the square side the image is fed at; `backbone` is the Swin variant
# each checkpoint was trained with (the lite checkpoints use swin_v1_t). The
# file is saved as models/background_removal/<name>.safetensors — ComfyUI's own
# folder for its core BiRefNet nodes, so the same weights serve both.
CHECKPOINTS = (
    BiRefNetCheckpoint(name="BiRefNet-general",      repo="ZhengPeng7/BiRefNet",              file="model.safetensors",            backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_512x512",      repo="ZhengPeng7/BiRefNet_512x512",      file="model.safetensors",            backbone="swin_v1_l", res=512),
    BiRefNetCheckpoint(name="BiRefNet-HR",           repo="ZhengPeng7/BiRefNet_HR",           file="model.safetensors",            backbone="swin_v1_l", res=2048),
    BiRefNetCheckpoint(name="BiRefNet-portrait",     repo="ZhengPeng7/BiRefNet-portrait",     file="model.safetensors",            backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet-matting",      repo="ZhengPeng7/BiRefNet-matting",      file="model.safetensors",            backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet-HR-matting",   repo="ZhengPeng7/BiRefNet_HR-matting",   file="model.safetensors",            backbone="swin_v1_l", res=2048),
    BiRefNetCheckpoint(name="BiRefNet_lite",         repo="ZhengPeng7/BiRefNet_lite",         file="model.safetensors",            backbone="swin_v1_t", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_lite-2K",      repo="ZhengPeng7/BiRefNet_lite-2K",      file="model.safetensors",            backbone="swin_v1_t", res=2048),
    BiRefNetCheckpoint(name="BiRefNet_dynamic",      repo="ZhengPeng7/BiRefNet_dynamic",      file="model.safetensors",            backbone="swin_v1_l", res=1024),
    BiRefNetCheckpoint(name="BiRefNet_lite-matting", repo="ZhengPeng7/BiRefNet_lite-matting", file="model.safetensors",            backbone="swin_v1_t", res=1024),
    BiRefNetCheckpoint(name="Lucida",                repo="egeorcun/lucida",                  file="lucida-m35-comfy.safetensors", backbone="swin_v1_l", res=1024),
)

for _checkpoint in CHECKPOINTS:
    register(MATTING, _checkpoint.name, _checkpoint)

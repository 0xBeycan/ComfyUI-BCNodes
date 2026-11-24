"""Qwen-Image (core's comfy.supported_models.QwenImage). Registers its DyPE adapter in the dype family."""

from ..common.registry import DYPE, register
from .adapter import ADAPTER

register(DYPE, ADAPTER.name, ADAPTER)

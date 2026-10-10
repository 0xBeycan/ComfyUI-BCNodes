"""The LM family contract: what one family of chat models (one LM node) gives the pipeline. A family
package builds an LMFamily and registers it under LM_FAMILY in models/common/registry.py."""

from dataclasses import dataclass
from typing import Callable

from .catalog import CatalogModel


@dataclass(frozen=True)
class PromptParts:
    system: str     # "" = no system message
    user: str
    assistant: str  # the prefill: the answer starts with this text
    n_images: int   # frames of the image batch, one image block each
    thinking: bool


@dataclass(frozen=True)
class LMFamily:
    key: str              # its registry name, e.g. 'qwen_lm'
    folder: str           # the folder_paths key and the folder under models/, e.g. 'Qwen-LM'
    lora_subfolder: str   # where its LoRAs live inside that folder
    catalog_path: str     # absolute path of the built-in models.yaml
    default_backend: str  # the backend of a catalog entry that names none
    templates: tuple      # the template ids its catalog entries may use
    # (model, parts) -> the raw chat prompt; raises ValueError on parts the model cannot take
    build_prompt: Callable[[CatalogModel, PromptParts], str]
    # IMAGE batch (N, H, W, C) or None -> the float32 (N, h, w, 3) batch the model is given, or None
    prepare_images: Callable[[object], object]
    # (model, decoded new text, parts) -> (text, thinking)
    split_output: Callable[[CatalogModel, str, PromptParts], tuple]

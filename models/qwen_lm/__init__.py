"""The Qwen LM family over the LM runtime (models/common/lm): the built-in catalog (models.yaml), the
official chat templates and output split (chat.py) and the official image sizing (images.py), registered
as qwen_lm in the LM family registry. Model files live in models/Qwen-LM/, LoRAs in models/Qwen-LM/loras/.
"""

import os

from ..common.lm.family import LMFamily
from ..common.registry import LM_FAMILY, register
from .chat import TEMPLATES, build_prompt, split_output
from .images import prepare_images

FAMILY = LMFamily(
    key="qwen_lm",
    folder="Qwen-LM",
    lora_subfolder="loras",
    catalog_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml"),
    default_backend="core",
    templates=TEMPLATES,
    build_prompt=build_prompt,
    prepare_images=prepare_images,
    split_output=split_output,
)

register(LM_FAMILY, FAMILY.key, FAMILY)

"""The LM backend contract: what runs a chat model file. One implementation today (core_backend.py,
ComfyUI core's text generation); a runner outside core is a second implementation of LMBackend,
registered under its own name in models/common/registry.py (LM_BACKEND)."""

from dataclasses import dataclass
from typing import Callable, Protocol

# The values of GenerateParams.mtp, as the config node offers them: auto (the backend picks the draft
# depth), off, or a fixed draft depth.
MTP_CHOICES = ("auto", "off", "2", "3", "4", "5")


@dataclass(frozen=True)
class GenerateParams:
    do_sample: bool
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    presence_penalty: float
    seed: int
    max_new_tokens: int
    mtp: str  # one of MTP_CHOICES


@dataclass(frozen=True)
class LoraSpec:
    key: tuple  # cache identity: (path, mtime, size, strength)
    strength: float
    # (key_map, weight_shapes) -> the core-ready LoRA tensors; raises when the LoRA does not fit the model.
    # key_map: LoRA module name -> the model's state-dict key; weight_shapes: state-dict key -> the
    # weight's logical shape. Called only when the LoRA is not cached already.
    build: Callable[[dict, dict], dict]


class LMBackend(Protocol):
    name: str

    def load(self, path: str, lora: "LoraSpec | None") -> object:
        """A handle for generate: the model file at `path`, with `lora` applied. One slot: the base
        model is kept by path, the LoRA'd model by lora.key; anything else held is dropped first."""

    def generate(self, handle, prompt: str, images, params: GenerateParams) -> str:
        """The decoded new text (special tokens stripped) for the raw chat `prompt`, `images` (a float32
        (N, H, W, 3) batch, one frame per image block of the prompt, or None)."""

    def unload(self) -> dict:
        """Drops every model held: {name: bytes of its weights}, {} when nothing was held."""

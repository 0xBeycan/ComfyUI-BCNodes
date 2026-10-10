"""The LM runtime the LM families share: the catalog (catalog.py), the family contract (family.py), the
backend contract (backend.py) and its ComfyUI core implementation (core_backend.py), registered here as the
"core" LM backend."""

from ..registry import LM_BACKEND, register
from .core_backend import CoreBackend

register(LM_BACKEND, CoreBackend.name, CoreBackend())

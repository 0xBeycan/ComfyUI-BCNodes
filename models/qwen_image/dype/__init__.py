"""DyPE's rotary embedder for Qwen-Image (Apache-2.0, see LICENSE and NOTICE in this directory). Plain torch;
core's fp64 check is imported on the first forward pass."""

from .embedder import DyPERope

__all__ = ["DyPERope"]

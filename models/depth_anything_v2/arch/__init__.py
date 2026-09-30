"""Vendored Depth Anything V2 architecture, ViT-S only (Apache-2.0, see LICENSE in this
directory).

Importing this package only touches torch; no xFormers, cv2 or torchvision.
"""

from .dpt import DepthAnythingV2

__all__ = ["DepthAnythingV2"]

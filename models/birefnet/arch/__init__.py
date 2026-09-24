"""Vendored BiRefNet architecture (MIT, see LICENSE in this directory).

Importing this package only touches torch; torchvision (deform_conv2d) is
imported on the first forward pass.
"""

from .model import BiRefNet
from .swin_v1 import BACKBONES

__all__ = ["BiRefNet", "BACKBONES"]

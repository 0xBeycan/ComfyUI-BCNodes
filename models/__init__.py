"""One package per model plus common/; a model package imports libs/ and models/common/ only,
never pipelines/, nodes/ or another model package.

This file is the registration hub: it imports every model package that registers a family
member in common/registry.py, so the registry is filled before any lookup.
"""

from . import birefnet  # noqa: F401  (registers the matting checkpoints)

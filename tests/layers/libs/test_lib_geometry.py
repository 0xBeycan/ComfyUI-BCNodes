"""libs/geometry.short_side_size against comfyui_controlnet_aux's output size (resize_image_with_pad:
k = resolution / min(H, W); H, W = int(np.round(H * k)), int(np.round(W * k))), written out here,
with hand-worked values including the halves numpy rounds to even."""

import numpy as np
import pytest


def _cnaux_size(h, w, resolution):
    k = float(resolution) / float(min(h, w))
    return int(np.round(float(h) * k)), int(np.round(float(w) * k))


@pytest.mark.parametrize("h, w, resolution, size", [
    (480, 640, 518, (518, 691)),     # 640 * 518 / 480 = 690.67
    (1080, 1920, 518, (518, 921)),   # 1920 * 518 / 1080 = 920.89
    (1920, 1080, 518, (921, 518)),
    (512, 512, 1036, (1036, 1036)),  # upscaled
    (2, 5, 1, (1, 2)),               # 2.5 rounds half to even
    (2, 7, 1, (1, 4)),               # 3.5 rounds half to even
])
def test_short_side_size(h, w, resolution, size, bcnodes):
    got = bcnodes["libs.geometry"].short_side_size(h, w, resolution)
    assert got == size == _cnaux_size(h, w, resolution)

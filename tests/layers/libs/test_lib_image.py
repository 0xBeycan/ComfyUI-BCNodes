"""Golden: fit_image (letterbox / crop / fill) on fixed PIL images, called directly.

Seeded RGB images wider and taller than square, a square L image and an RGBA image, each fitted
to targets narrower, wider, equal and larger, with two resampling filters and a background
colour; an unknown fit name falls through to the plain resize. One digest per image and fit.
"""

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
}

GOLDEN = {
    'rgb_wide/letterbox': '3b82fa766e6761c40db72db348989004',
    'rgb_wide/crop': 'a8bf7cfa414d1ea06a5e94e86968272a',
    'rgb_wide/fill': '9e7dd308595d74c4c18e44f14120cb6d',
    'rgb_wide/stretch': '9e7dd308595d74c4c18e44f14120cb6d',
    'rgb_tall/letterbox': 'a12c3dc45bfe48237206e91150ba5a20',
    'rgb_tall/crop': 'aedbad034d135916479c1cec7223272e',
    'rgb_tall/fill': 'c699ad8197f2cfb488952dff176c7885',
    'rgb_tall/stretch': 'c699ad8197f2cfb488952dff176c7885',
    'l_square/letterbox': '0dcdb28d54424a4305ac4a3842cf306e',
    'l_square/crop': '8e375542425d57b9f1a14811110466c8',
    'l_square/fill': 'eed82f0da34983c94a27ca114b7e925d',
    'l_square/stretch': 'eed82f0da34983c94a27ca114b7e925d',
    'rgba_wide/letterbox': 'c0a9a0cd1ce4d3e5adc607e2d0eaf696',
    'rgba_wide/crop': 'c3cd86742599828a2f14fff16e19605c',
    'rgba_wide/fill': '4d8c951cfb446165310ab0a393300093',
    'rgba_wide/stretch': '4d8c951cfb446165310ab0a393300093',
}

WHERE = Where({
    "fit_image": "libs.image:fit_image",
})


def _pil(shape, seed):
    """uint8 (H, W) -> L, (H, W, 3) -> RGB, (H, W, 4) -> RGBA."""
    from PIL import Image

    return Image.fromarray((torch.rand(shape, generator=torch.Generator().manual_seed(seed)) * 255).to(torch.uint8).numpy())


IMAGES = {
    "rgb_wide": lambda: _pil((24, 40, 3), 13),
    "rgb_tall": lambda: _pil((40, 24, 3), 13),
    "l_square": lambda: _pil((30, 30), 13),
    "rgba_wide": lambda: _pil((10, 20, 4), 13),
}
BACKGROUND = {"rgb_wide": "#123456", "rgb_tall": "#ff0000", "l_square": "black", "rgba_wide": (10, 20, 30, 40)}
TARGETS = [(32, 32), (48, 16), (16, 48), (40, 24), (80, 48), (7, 5)]
FITS = ["letterbox", "crop", "fill", "stretch"]


@pytest.mark.parametrize("fit", FITS)
@pytest.mark.parametrize("name", list(IMAGES))
def test_fit_image(name, fit, bcnodes):
    from PIL import Image

    check_env(ENV, "torch", "numpy", "Pillow")
    image = IMAGES[name]()
    assert image.mode == name.split("_")[0].upper()
    out = {}
    for w, h in TARGETS:
        for sampler in (Image.Resampling.LANCZOS, Image.Resampling.NEAREST):
            out[f"{w}x{h}/{int(sampler)}"] = WHERE["fit_image"](image, w, h, fit, sampler, BACKGROUND[name])
    check(GOLDEN, f"{name}/{fit}", digest(out))
    assert digest(image) == digest(IMAGES[name]())  # the input is not modified

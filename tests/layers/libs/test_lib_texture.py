"""Golden: the skin-texture engine, called directly.

pore_field((2, 1, 64, 80), 1.0, 3, cpu) and three other shapes / scales / seeds; apply_texture
for gate 1.0 and 0.5, at pore_scale 1 (at 80 px its blur radii fall under the 0.2 px cut-off)
and at pore_scale 12.8 (scale 1.0, every blur active), with texture 0, detail 0, fp16 and a
batch of 2; the reflect-padded Gaussian on a ramp (sigma at and above the cut-off, 3 channels,
float64); the sRGB <-> linear pair around both thresholds, float32 and float64. Seeded.
"""

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'pore_field/plan_2x1x64x80_scale1_seed3': '94bebecb6fbad8169400c784767b6ae2',
    'pore_field/scale_0.078_seed0': 'd1ce16f185d596a67b273287b58b95e8',
    'pore_field/scale_0.3_seed0': '191d9f30e9cc333b1829451840f532a8',
    'pore_field/scale_2.5_seed7': 'e69fbf4d1b78fc54eeedaa9bc03fafb5',
    'apply_texture/gate_1.0': 'e39b60a328af94419e2b0df47990cadf',
    'apply_texture/gate_0.5': '2bef5bbd97300983d53f2104f072f5fb',
    'apply_texture/scale1_gate_1.0': 'f312631868b956945d1fe5df479db01b',
    'apply_texture/scale1_gate_0.5': '983ed5030e59a4d5712c0c065ae8e632',
    'apply_texture/scale1_texture_0': '42291f23cea7ad71f50e2ded8abaab6e',
    'apply_texture/scale1_detail_0': 'b7a8c772f598a3a2b5257abe2537eca5',
    'apply_texture/defaults': 'e39b60a328af94419e2b0df47990cadf',
    'apply_texture/fp16': '5a2780a0879357c3e40722ac2ed07765',
    'apply_texture/batch_2': 'c4ce636305b278dea6bc82447bc722b2',
    'gauss/sigma_0.2': ('827b123bcbcb2d0a98f5a0ac1ddb1ecc', True),
    'gauss/sigma_0.21': ('12a8b16e71b4ec6065c6366d76e97eeb', False),
    'gauss/sigma_0.9': ('a43bfe51f4f432bf82c3049817a8b821', False),
    'gauss/sigma_2.0': ('2ff533b6fa7ee095c6a2ca178761a89a', False),
    'gauss/sigma_3.7': ('90aec034e3efcd16230c178a61bd8db8', False),
    'gauss/three_channels_1.5': ('4b98af22e6980384004cfe7f5f00c062', False),
    'gauss/float64_1.0': ('a1320f895c01715c266b1da0d334f8d3', False),
    'srgb/torch.float32': '9e09701b39cf28c8878b4ef314ced149',
    'srgb/torch.float64': 'f0488729bfc451022ba3330d103bb466',
}

WHERE = Where({
    "pore_field": "libs.texture:pore_field",
    "apply_texture": "libs.texture:apply_texture",
    "gauss": "libs.filters:gauss_reflect",
    "srgb_to_linear": "libs.color:srgb_to_linear",
    "linear_to_srgb": "libs.color:linear_to_srgb",
})

CPU = torch.device("cpu")


def _rand(shape, seed, dtype=torch.float32):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


PORE = {
    "plan_2x1x64x80_scale1_seed3": ((2, 1, 64, 80), 1.0, 3),
    "scale_0.078_seed0": ((1, 1, 64, 80), 80 / 1024.0, 0),
    "scale_0.3_seed0": ((1, 1, 32, 48), 0.3, 0),
    "scale_2.5_seed7": ((1, 1, 48, 64), 2.5, 7),
}


@pytest.mark.parametrize("name", list(PORE))
def test_pore_field(name, bcnodes):
    check_env(ENV, "torch")
    shape, scale, seed = PORE[name]
    check(GOLDEN, f"pore_field/{name}", digest(WHERE["pore_field"](shape, scale, seed, CPU)))


def _image(b=1, dtype=torch.float32):
    return (_rand((b, 64, 80, 3), 5) * 0.5 + 0.25).to(dtype)


def _mask(b=1):
    return _rand((b, 64, 80), 6)


APPLY = {
    "gate_1.0": lambda: (_image(), _mask(), dict(texture=0.5, detail=0.6, pore_scale=1.0, seed=0, gate=1.0)),
    "gate_0.5": lambda: (_image(), _mask(), dict(texture=0.5, detail=0.6, pore_scale=1.0, seed=0, gate=0.5)),
    "scale1_gate_1.0": lambda: (_image(), _mask(), dict(texture=0.5, detail=0.6, pore_scale=12.8, seed=0, gate=1.0)),
    "scale1_gate_0.5": lambda: (_image(), _mask(), dict(texture=0.5, detail=0.6, pore_scale=12.8, seed=0, gate=0.5)),
    "scale1_texture_0": lambda: (_image(), _mask(), dict(texture=0.0, detail=0.6, pore_scale=12.8, seed=0, gate=1.0)),
    "scale1_detail_0": lambda: (_image(), _mask(), dict(texture=0.5, detail=0.0, pore_scale=12.8, seed=0, gate=1.0)),
    "defaults": lambda: (_image(), _mask(), dict()),
    "fp16": lambda: (_image(dtype=torch.float16), _mask(), dict(texture=0.35, detail=0.45, pore_scale=12.8, seed=3, gate=0.75)),
    "batch_2": lambda: (_image(b=2), _mask(b=2), dict(texture=1.0, detail=2.0, pore_scale=6.4, seed=9, gate=1.0)),
}


@pytest.mark.parametrize("name", list(APPLY))
def test_apply_texture(name, bcnodes):
    check_env(ENV, "torch")
    image, mask, kwargs = APPLY[name]()
    check(GOLDEN, f"apply_texture/{name}", digest(WHERE["apply_texture"](image, mask, **kwargs)))


def _ramp(c=1, dtype=torch.float32):
    ramp = torch.linspace(0, 1, 16 * 24, dtype=dtype).view(1, 1, 16, 24) ** 2
    return torch.cat([ramp * (k + 1) / c for k in range(c)], dim=1)


GAUSS = {
    "sigma_0.2": (lambda: _ramp(), 0.2),
    "sigma_0.21": (lambda: _ramp(), 0.21),
    "sigma_0.9": (lambda: _ramp(), 0.9),
    "sigma_2.0": (lambda: _ramp(), 2.0),
    "sigma_3.7": (lambda: _ramp(), 3.7),
    "three_channels_1.5": (lambda: _ramp(c=3), 1.5),
    "float64_1.0": (lambda: _ramp(dtype=torch.float64), 1.0),
}


@pytest.mark.parametrize("name", list(GAUSS))
def test_gauss(name, bcnodes):
    check_env(ENV, "torch")
    x, sigma = GAUSS[name]
    x = x()
    out = WHERE["gauss"](x, sigma)
    check(GOLDEN, f"gauss/{name}", (digest(out), out is x))


def _levels(dtype):
    points = torch.linspace(-0.1, 1.1, 241, dtype=dtype)
    return torch.cat([points, torch.tensor([0.04045, 0.0031308, 0.0, -0.0, 1.0], dtype=dtype)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_srgb(dtype, bcnodes):
    check_env(ENV, "torch")
    x = _levels(dtype)
    to_linear = WHERE["srgb_to_linear"](x)
    check(GOLDEN, f"srgb/{dtype}", digest({
        "srgb_to_linear": to_linear,
        "linear_to_srgb": WHERE["linear_to_srgb"](x),
        "round_trip": WHERE["linear_to_srgb"](to_linear),
    }))

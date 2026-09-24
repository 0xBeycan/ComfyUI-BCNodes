"""F3: Image Quality Gate clips to [0, 1] before the uint8 cast.

`(image[0].cpu().numpy() * 255).astype(np.uint8)` wraps values outside [0, 1] (1.2 -> 50,
-0.1 -> 231), and such IMAGEs reach the node in ordinary graphs: ComfyUI's bicubic upscale
overshoots. F3 wraps the product in np.clip(..., 0, 255).

- Property (red on F3's parent): analyze(x) equals analyze(x.clamp(0, 1)) on all 8 outputs
  (badge digest, verdict, report, repr and type of each score) for three out-of-range inputs,
  each x 4 shot types x 2 blur modes.
- UNCHANGED (recorded with BCNODES_GOLDEN_RECORD=1 on F3's parent, asserted since): the 8
  outputs for six in-range inputs (the near-edge one overshoots by less than 1/255, so the clip
  must leave it identical), each x 4 x 2.

PIL.ImageFont.truetype raises OSError here for a font file, so the badge uses PIL's default font
on every platform (the DejaVu paths are Linux-only); the built-in font that load_default() hands
to truetype in memory still loads.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
    'cv2': '5.0.0',
}

UNCHANGED = {
    'noise/custom/full image': 'cf96125a7c4f6d7dc2e19e2381c2f092',
    'noise/custom/center-weighted': '492d1eb19d7498255badd2bd71c93f9d',
    'noise/close-up/full image': 'abadce45c1d4d225217f4a4aa6d8bbf8',
    'noise/close-up/center-weighted': '7ddbe634b84408390a6a8085facc7373',
    'noise/medium/full image': 'c0cabbcbb90f12d5c47932d7a4598faf',
    'noise/medium/center-weighted': '5a6c4d8ddbf8eedd8cc72d57690e9ab9',
    'noise/wide / full-body/full image': '45b26e588800a70de502e7aaf1aef5ec',
    'noise/wide / full-body/center-weighted': 'd93d23ac28dad92da99f6dc2fc19f37c',
    'flat_0.5/custom/full image': '4d8069dd46d3f608b21b2f738ccbac9c',
    'flat_0.5/custom/center-weighted': '8f04e2b3b32f1aed9430cc63eee96729',
    'flat_0.5/close-up/full image': 'dbe5eced832fab93b6d5c2840eb4adcc',
    'flat_0.5/close-up/center-weighted': '61633ad5a5107b0cb43807a2f83891d6',
    'flat_0.5/medium/full image': '3129b83edb4c1f7c301bbb8c88f20957',
    'flat_0.5/medium/center-weighted': '9c2b138b22edc3caeff7dea73156d978',
    'flat_0.5/wide / full-body/full image': '9ef9a8b7fb97c3070427b3d76ba3b5a1',
    'flat_0.5/wide / full-body/center-weighted': '99ef7c54dc4e28597cd1a675c66dd85d',
    'horizontal_ramp/custom/full image': 'dc38ee161bce64bf92b016c9feea41a7',
    'horizontal_ramp/custom/center-weighted': 'ce3e3348c2c6d7506463189cef2cd228',
    'horizontal_ramp/close-up/full image': '076ce498b7af63243e72529c8e00ba25',
    'horizontal_ramp/close-up/center-weighted': 'e2ff3831e52aa6238a94490021b41a6a',
    'horizontal_ramp/medium/full image': '882acf1594706e16f4ebb297d088ad05',
    'horizontal_ramp/medium/center-weighted': '4de12bc8940047e18de7cb9d3f8439d3',
    'horizontal_ramp/wide / full-body/full image': '8e9b56edff60fabe06b68cedab517644',
    'horizontal_ramp/wide / full-body/center-weighted': 'caa2032c56b3952e0c107f48619e5645',
    '8x8/custom/full image': '73aff0081d5c71d182c46dbf0f8930fc',
    '8x8/custom/center-weighted': 'be65a8178958300e01fbf6a8a6053e08',
    '8x8/close-up/full image': '4eda606d3bbdf4b919438833c69d707f',
    '8x8/close-up/center-weighted': '1252276afe5624679949e306ef43f106',
    '8x8/medium/full image': '0c2cd71d2f1ab9a51038ec2fce5e03ab',
    '8x8/medium/center-weighted': 'ca11176932230019c5ac63636c0bf925',
    '8x8/wide / full-body/full image': '62a92220dd3b8449354c4b675d5db6cb',
    '8x8/wide / full-body/center-weighted': 'dc0b112beda3154b98a5b86e846d8209',
    'batch_2/custom/full image': '0a7233368cc7119405c2692d4f913b4c',
    'batch_2/custom/center-weighted': '6a349fe5d8d82f9bfb9070ea469fde99',
    'batch_2/close-up/full image': 'a3ff4dc3e2d4518ab2479e0dddb8efb1',
    'batch_2/close-up/center-weighted': 'ad723eda15d21d03e07434b7d7d1f7c2',
    'batch_2/medium/full image': 'eccee08419b93576419cbbb5e6b75814',
    'batch_2/medium/center-weighted': 'e46b8245830bbad30e4baa410d68831d',
    'batch_2/wide / full-body/full image': 'e55511cfedf367237ec120828c5e7374',
    'batch_2/wide / full-body/center-weighted': '55dc7a50bdad878ffea40a849a2b242b',
    'near_edge/custom/full image': '042b87d557000edd9e5f45a2ddb83f4f',
    'near_edge/custom/center-weighted': '1b420f130c6add8be4fa45c57052b609',
    'near_edge/close-up/full image': '716944b56ef46256f48560128e1f094f',
    'near_edge/close-up/center-weighted': '88dde0cb9b97ef1140fdd383aa440ae3',
    'near_edge/medium/full image': '72ee21765e2b70b5985c38159d4a53e7',
    'near_edge/medium/center-weighted': '237757b8e4b2147b5a960045ce7d72a5',
    'near_edge/wide / full-body/full image': '75f6f81ac4b89350f52abb751620b0c9',
    'near_edge/wide / full-body/center-weighted': '0e17de561e67d38baabb15d9856f7748',
}

SHOT_TYPES = ["custom", "close-up", "medium", "wide / full-body"]
BLUR_MODES = ["full image", "center-weighted"]
THRESHOLDS = {"blur_threshold": 0.40, "blur_var_threshold": 30.0, "sharpness_threshold": 15.0,
              "noise_threshold": 25.0, "clipping_threshold": 0.02, "entropy_threshold": 5.0}  # the widget defaults


def _rand(shape, seed):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


def _half(value):
    x = torch.full((1, 32, 48, 3), 0.5)
    x[:, :, 24:, :] = value
    return x


def _bicubic_x2():
    """A binary image upscaled x2 as comfy.utils.common_upscale does for "bicubic" (channels to
    dim 1, F.interpolate, back): overshoots below 0 and above 1."""
    binary = (_rand((1, 32, 48, 3), 7) > 0.5).float()
    return F.interpolate(binary.movedim(-1, 1), size=(64, 96), mode="bicubic").movedim(1, -1)


OUT_OF_RANGE = {
    "half_1.2": lambda: _half(1.2),
    "half_-0.1": lambda: _half(-0.1),
    "bicubic_x2": _bicubic_x2,
}

UNCHANGED_INPUTS = {
    "noise": lambda: _rand((1, 64, 64, 3), 0),
    "flat_0.5": lambda: torch.full((1, 64, 64, 3), 0.5),
    "horizontal_ramp": lambda: torch.linspace(0, 1, 64).view(1, 1, 64, 1).expand(1, 48, 64, 3).contiguous(),
    "8x8": lambda: _rand((1, 8, 8, 3), 1),
    "batch_2": lambda: _rand((2, 32, 48, 3), 2),  # frame 0 only is analysed
    "near_edge": lambda: _rand((1, 32, 48, 3), 3) * (1 + 0.007) - 0.0035,
}


@pytest.fixture
def analyze(bcnodes, monkeypatch):
    from PIL import ImageFont

    real_truetype = ImageFont.truetype

    def truetype(font=None, *args, **kwargs):
        if isinstance(font, (str, os.PathLike)):
            raise OSError(f"no font file in this test: {font}")
        return real_truetype(font, *args, **kwargs)  # load_default() loads its built-in font through truetype

    monkeypatch.setattr(ImageFont, "truetype", truetype, raising=True)
    gate = bcnodes["image_quality_gate"].ImageQualityGate()

    def run(image, shot_type, blur_mode):
        badge, verdict, report, *scores = gate.analyze(image, shot_type, blur_mode, **THRESHOLDS)
        return digest(badge), verdict, report, [(repr(s), type(s).__name__) for s in scores]

    return run


def _combos(inputs):
    return [(name, shot, mode) for name in inputs for shot in SHOT_TYPES for mode in BLUR_MODES]


@pytest.mark.parametrize("name,shot_type,blur_mode", _combos(OUT_OF_RANGE))
def test_out_of_range_input_is_read_clipped(name, shot_type, blur_mode, analyze):
    x = OUT_OF_RANGE[name]()
    assert x.min() < 0 or x.max() > 1
    assert analyze(x, shot_type, blur_mode) == analyze(x.clamp(0, 1), shot_type, blur_mode)


@pytest.mark.parametrize("name,shot_type,blur_mode", _combos(UNCHANGED_INPUTS))
def test_unchanged(name, shot_type, blur_mode, analyze):
    check_env(ENV, "torch", "numpy", "Pillow", "cv2")
    check(UNCHANGED, f"{name}/{shot_type}/{blur_mode}", digest(analyze(UNCHANGED_INPUTS[name](), shot_type, blur_mode)))

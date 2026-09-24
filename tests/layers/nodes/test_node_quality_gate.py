"""Golden: Image Quality Gate end to end, through ImageQualityGate().analyze.

All 8 outputs (badge digest, verdict, report text, repr and type of each of the 5 scores) for
seven inputs x 4 shot types x 2 blur modes, on the widget-default thresholds:
  - Generator(0) noise, flat 0.5, a horizontal ramp, a graded-noise ramp (blocks on both sides
    of the Laplacian-variance threshold, so centre weighting matters; the only PASS input),
    out-of-range 1.2 / -0.1 bands on the noise (recorded on FIXED_BASE, so read clipped: F3),
    an 8x8 image (smaller than every block), a 2-frame batch (frame 0 only is analysed).
PIL.ImageFont.truetype raises OSError for a font file in that grid, so the badge uses PIL's
default font (the DejaVu paths of nodes/image_quality_gate.py are Linux-only).

The Linux font branch: truetype is replaced by a recorder that returns ImageFont.load_default(size)
for a font file, so the three truetype calls (path, size) and the badge drawn with them are
pinned, for one input per verdict (PASS, SO-SO, FAIL).

analyze stays on the node, so no WHERE.
"""

import os

import pytest
import torch

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
    'cv2': '5.0.0',
}

GOLDEN = {
    'noise/custom/full image': '904cf4413dac79ffc1380b5b17583b2b',
    'noise/custom/center-weighted': 'a299c003d873953dfbeae25537cb0215',
    'noise/close-up/full image': '48b07eab2d3d17f6d4701151630a0f4f',
    'noise/close-up/center-weighted': '645732f9f4ab92a04df0eb0c5bff5ed6',
    'noise/medium/full image': '9cc8c886ed72d030d135d0c2740918e2',
    'noise/medium/center-weighted': '0ea9bc0ddd6c39e6a0d0f00920d447ce',
    'noise/wide / full-body/full image': '3f344b9800dba3b8e0a9e1cb079e03be',
    'noise/wide / full-body/center-weighted': '40bc06d1ef511c482cc179ffef3d18fc',
    'flat_0.5/custom/full image': '4d8069dd46d3f608b21b2f738ccbac9c',
    'flat_0.5/custom/center-weighted': '8f04e2b3b32f1aed9430cc63eee96729',
    'flat_0.5/close-up/full image': 'dbe5eced832fab93b6d5c2840eb4adcc',
    'flat_0.5/close-up/center-weighted': '61633ad5a5107b0cb43807a2f83891d6',
    'flat_0.5/medium/full image': '3129b83edb4c1f7c301bbb8c88f20957',
    'flat_0.5/medium/center-weighted': '9c2b138b22edc3caeff7dea73156d978',
    'flat_0.5/wide / full-body/full image': '9ef9a8b7fb97c3070427b3d76ba3b5a1',
    'flat_0.5/wide / full-body/center-weighted': '99ef7c54dc4e28597cd1a675c66dd85d',
    'horizontal_ramp/custom/full image': 'a42c77ffe0fa5683122734cac4365743',
    'horizontal_ramp/custom/center-weighted': 'f147a9d4c3484cb80bb2935a9e116f33',
    'horizontal_ramp/close-up/full image': '60a19762e8ec83c2403959568625fb5d',
    'horizontal_ramp/close-up/center-weighted': '673ec315b47f766d7b109aef5e2689dc',
    'horizontal_ramp/medium/full image': '8b6e56c28db8213b46f6b2373b462c71',
    'horizontal_ramp/medium/center-weighted': '87f0111a32042136558cb8be28e4e1d8',
    'horizontal_ramp/wide / full-body/full image': '95c3c15f51bc07ba08a986a94280e86b',
    'horizontal_ramp/wide / full-body/center-weighted': '8d177bbe9833014015f79863c8b540b2',
    'graded_noise/custom/full image': '9c551d4e3649e7002896487b149cd697',
    'graded_noise/custom/center-weighted': 'f6283d5e74c4c795ea694a6e10330c3f',
    'graded_noise/close-up/full image': 'dbc9079ccff493e52dfd3f57fb5ea199',
    'graded_noise/close-up/center-weighted': 'afba4de627d47ac545a5f574f193db4a',
    'graded_noise/medium/full image': '7fd5efc8ebe9e1f18433f0bef7d2b53a',
    'graded_noise/medium/center-weighted': 'ee5e7b3fc7c67efc7a62f38cdad5cccb',
    'graded_noise/wide / full-body/full image': 'ef5d874f6ae01d167f895ffdcf0a5495',
    'graded_noise/wide / full-body/center-weighted': '5bc0398bd2fb34d7abcea0b851cbf95c',
    'out_of_range/custom/full image': '7ec7068055ba3a8a35d1c2c9c06f0e1e',
    'out_of_range/custom/center-weighted': '61e591ce952bdff1609c8fa466eb6888',
    'out_of_range/close-up/full image': '820dc9613c656166bf2b6382bd143750',
    'out_of_range/close-up/center-weighted': '3b75d86442632bbf3df47ea41878edeb',
    'out_of_range/medium/full image': 'bfc03b2e09a21196731a3403e5440e44',
    'out_of_range/medium/center-weighted': '12c35552495bad92e788b455004c617f',
    'out_of_range/wide / full-body/full image': '54acdf893f7f48c09213d6fe0d3934f4',
    'out_of_range/wide / full-body/center-weighted': '54db0ab9005adf2824962a28d56ba2c6',
    '8x8/custom/full image': '73aff0081d5c71d182c46dbf0f8930fc',
    '8x8/custom/center-weighted': 'be65a8178958300e01fbf6a8a6053e08',
    '8x8/close-up/full image': '4eda606d3bbdf4b919438833c69d707f',
    '8x8/close-up/center-weighted': '1252276afe5624679949e306ef43f106',
    '8x8/medium/full image': '0c2cd71d2f1ab9a51038ec2fce5e03ab',
    '8x8/medium/center-weighted': 'ca11176932230019c5ac63636c0bf925',
    '8x8/wide / full-body/full image': '62a92220dd3b8449354c4b675d5db6cb',
    '8x8/wide / full-body/center-weighted': 'dc0b112beda3154b98a5b86e846d8209',
    'batch_2/custom/full image': '99c3398faad882b9dc6b6f52c8e2ba2c',
    'batch_2/custom/center-weighted': 'fd76cf933b5140ff262d81cc9fa952fd',
    'batch_2/close-up/full image': 'dcbd9be37400f735d791a3fcb7359dc5',
    'batch_2/close-up/center-weighted': 'a15f2145b88d966f23597b07af680924',
    'batch_2/medium/full image': '730f05d517c7a80a41925ba6f0d4df7b',
    'batch_2/medium/center-weighted': 'f2380d765c8babcfe12304caa3eafe70',
    'batch_2/wide / full-body/full image': '951c9005fb372e88390678e3ae927eca',
    'batch_2/wide / full-body/center-weighted': 'd43d6aefa679c4ddee172ed7ce0168ef',
    'font_branch/PASS': '521da55824545de889ab3aa8221f8d1f',
    'font_branch/SO-SO': '254523493a0e32c86f786a432e29673a',
    'font_branch/FAIL': '8682b12f1dc96bcead0031634b043d29',
}

SHOT_TYPES = ["custom", "close-up", "medium", "wide / full-body"]
BLUR_MODES = ["full image", "center-weighted"]
THRESHOLDS = {"blur_threshold": 0.40, "blur_var_threshold": 30.0, "sharpness_threshold": 15.0,
              "noise_threshold": 25.0, "clipping_threshold": 0.02, "entropy_threshold": 5.0}  # the widget defaults


def _rand(shape, seed):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


def _ramp():
    return torch.linspace(0, 1, 96).view(1, 1, 96, 1).expand(1, 64, 96, 3).contiguous()


def _graded_noise():
    amp = torch.linspace(0, 0.12, 96).view(1, 1, 96, 1)
    return (_ramp() * 0.6 + 0.2 + (_rand((1, 64, 96, 3), 5) - 0.5) * amp).clamp(0, 1)


def _out_of_range():
    x = _rand((1, 64, 96, 3), 0)
    x[:, :, :32, :] = 1.2
    x[:, :, 64:, :] = -0.1
    return x


INPUTS = {
    "noise": lambda: _rand((1, 64, 96, 3), 0),
    "flat_0.5": lambda: torch.full((1, 64, 96, 3), 0.5),
    "horizontal_ramp": _ramp,
    "graded_noise": _graded_noise,
    "out_of_range": _out_of_range,
    "8x8": lambda: _rand((1, 8, 8, 3), 1),
    "batch_2": lambda: _rand((2, 64, 96, 3), 2),
}

# one input per verdict for the font branch: (input, shot type, blur mode, expected verdict int)
FONT_BRANCH = {
    "PASS": ("graded_noise", "custom", "full image", 2),
    "SO-SO": ("noise", "wide / full-body", "full image", 1),
    "FAIL": ("flat_0.5", "custom", "full image", 0),
}


def _outputs(result):
    badge, verdict, report, *scores = result
    return digest(badge), verdict, report, [(repr(s), type(s).__name__) for s in scores]


@pytest.fixture
def gate(bcnodes):
    return bcnodes["image_quality_gate"].ImageQualityGate()


@pytest.mark.parametrize("blur_mode", BLUR_MODES)
@pytest.mark.parametrize("shot_type", SHOT_TYPES)
@pytest.mark.parametrize("name", list(INPUTS))
def test_analyze(name, shot_type, blur_mode, gate, monkeypatch):
    check_env(ENV, "torch", "numpy", "Pillow", "cv2")
    from PIL import ImageFont

    real_truetype = ImageFont.truetype

    def truetype(font=None, *args, **kwargs):
        if isinstance(font, (str, os.PathLike)):
            raise OSError(f"no font file in this test: {font}")
        return real_truetype(font, *args, **kwargs)  # load_default() loads its built-in font through truetype

    monkeypatch.setattr(ImageFont, "truetype", truetype, raising=True)
    result = gate.analyze(INPUTS[name](), shot_type, blur_mode, **THRESHOLDS)
    check(GOLDEN, f"{name}/{shot_type}/{blur_mode}", digest(_outputs(result)))


@pytest.mark.parametrize("verdict", list(FONT_BRANCH))
def test_font_branch(verdict, gate, monkeypatch):
    check_env(ENV, "torch", "numpy", "Pillow", "cv2")
    from PIL import ImageFont

    real_truetype = ImageFont.truetype
    calls = []

    def truetype(font=None, size=10, *args, **kwargs):
        if isinstance(font, (str, os.PathLike)):
            calls.append((font, size))
            return ImageFont.load_default(size)
        return real_truetype(font, size, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", truetype, raising=True)
    name, shot_type, blur_mode, expected = FONT_BRANCH[verdict]
    result = gate.analyze(INPUTS[name](), shot_type, blur_mode, **THRESHOLDS)
    assert result[1] == expected
    check(GOLDEN, f"font_branch/{verdict}", digest({"calls": calls, "outputs": _outputs(result)}))

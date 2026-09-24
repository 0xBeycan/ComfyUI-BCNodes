"""Golden: the BiRefNet matte option chain `finish` (plan 8.2 row 14).

pipelines/matting.py:finish applies sensitivity, blur,
grow/shrink, invert, the foreground refinement and the background to a raw matte and builds
the three outputs. Each case pins the digests of (image, mask, mask_image): shape, dtype,
requires_grad and bytes.

The matte is a seeded soft blob with exact 0 and 1 regions, so the Color "#00000000" case
reaches the total == 0 branch. Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move
edits only WHERE.
"""

import pytest
import torch
import torch.nn.functional as F

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'defaults': '550d40c28b5a7d70d0fc708f1a9b2a12',
    'sensitivity_0.3': 'f4937363e27afb6b2df1b38bd5a3865b',
    'blur_1': '21c52d58dc49cf7fb3bf7ad1be7851bc',
    'blur_3': '14e2dbf12765b8eeef18c754fd1f6c03',
    'offset_+3': 'e94f938231444b247806bb13e564f96e',
    'offset_-3': '8aef00c1e2be1582ff997aedd13bb947',
    'invert': '7ea360fce2b0e1c01a12eba0f4be40b1',
    'refine': '3acb7c0a87936dcbe60aeeed97662b73',
    'color_#222222': 'e2a5b497dba441c77ba9993c927a6baf',
    'color_#11223380': '5bfc985a0dc46ba4df4593105269feae',
    'color_#00000000': '956bdf06c508fcad35fc4b4f0c8ffbbe',
    'all_combined': '9fb90cfe4824312d618e92d63cd62e46',
}

WHERE = Where({
    "finish": "pipelines.matting:finish",
})

CASES = {
    "defaults": {},
    "sensitivity_0.3": {"sensitivity": 0.3},
    "blur_1": {"mask_blur": 1},
    "blur_3": {"mask_blur": 3},
    "offset_+3": {"mask_offset": 3},
    "offset_-3": {"mask_offset": -3},
    "invert": {"invert_output": True},
    "refine": {"refine_foreground": True},
    "color_#222222": {"background": "Color", "background_color": "#222222"},
    "color_#11223380": {"background": "Color", "background_color": "#11223380"},
    "color_#00000000": {"background": "Color", "background_color": "#00000000"},
    "all_combined": {"sensitivity": 0.5, "mask_blur": 2, "mask_offset": 2, "invert_output": True,
                     "refine_foreground": True, "background": "Color", "background_color": "#11223380"},
}


def _inputs():
    g = torch.Generator().manual_seed(14)
    coarse = torch.rand((1, 1, 4, 4), generator=g)
    blob = F.interpolate(coarse, size=(24, 20), mode="bilinear", align_corners=False)[:, 0]
    matte = ((blob - 0.3) * 2.5).clamp(0, 1)  # (1, 24, 20) with exact 0s and 1s
    rgb = torch.rand((1, 24, 20, 3), generator=g)
    return rgb, matte


def test_matte_has_hard_regions():
    _, matte = _inputs()
    assert (matte == 0).any() and (matte == 1).any() and ((matte > 0) & (matte < 1)).any()


@pytest.mark.parametrize("name", list(CASES))
def test_finish(name, bcnodes):
    check_env(ENV, "torch")
    rgb, matte = _inputs()
    check(GOLDEN, name, digest(WHERE["finish"](rgb, matte, **CASES[name])))

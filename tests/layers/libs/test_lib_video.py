"""Golden: Video Comparer's frame fit (crop when larger, scale + letterbox otherwise), called
directly: crop, equal size, 4 channels, upscale, the two mixed branches (taller but narrower,
wider but shorter), a 1x1 frame, a result 1 px wide, float64, and a binary frame whose bicubic
overshoot is clamped. Seeded frames.
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
    'crop': 'd7853ed5b1185c688ba5cc826ed05655',
    'equal': '223b1f07bdbb3e889b7dee5caa6994e2',
    'crop_rgba': '4d17f008e1c5df4fa0d23df07e766f22',
    'upscale': 'ffff8717b81f3edc618a735f29ddbc57',
    'taller_narrower': 'a2bf03f67fb62ebd6370253d04b5a185',
    'wider_shorter': 'bb127c402ebbdb21a95038433b709728',
    'one_pixel': '60110b3b70cc7ccdc5af3b56dc9f5da2',
    'one_pixel_wide_result': '65f75177a3b85fb76508d3c2ebc04442',
    'float64': 'df357f136ad99b4214f3be300f19b572',
    'overshoot_clamped': '0dc99a1b31f625f6ebf8279831310656',
    'rgba_upscale': '2c98d8e0962b467c03d4f61c5fa58713',
}

WHERE = Where({
    "fit": "libs.video:fit_frame",
})


def _frame(h, w, c=3, seed=0, dtype=torch.float32):
    return torch.rand((h, w, c), generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _binary(h, w):
    return (_frame(h, w, seed=5) > 0.5).float()


CASES = {
    "crop": lambda: (_frame(33, 65), 64, 32),
    "equal": lambda: (_frame(32, 64, seed=1), 64, 32),
    "crop_rgba": lambda: (_frame(33, 65, c=4, seed=2), 64, 32),
    "upscale": lambda: (_frame(20, 30, seed=3), 64, 32),
    "taller_narrower": lambda: (_frame(40, 30, seed=4), 64, 32),
    "wider_shorter": lambda: (_frame(20, 80, seed=6), 64, 32),
    "one_pixel": lambda: (_frame(1, 1, seed=7), 64, 32),
    "one_pixel_wide_result": lambda: (_frame(100, 2, seed=8), 64, 32),
    "float64": lambda: (_frame(20, 30, seed=3, dtype=torch.float64), 64, 32),
    "overshoot_clamped": lambda: (_binary(16, 24), 64, 48),
    "rgba_upscale": lambda: (_frame(20, 30, c=4, seed=9), 48, 40),
}


@pytest.mark.parametrize("name", list(CASES))
def test_fit(name, bcnodes):
    check_env(ENV, "torch")
    frame, w, h = CASES[name]()
    out = WHERE["fit"](frame, w, h)
    check(GOLDEN, name, digest(out))

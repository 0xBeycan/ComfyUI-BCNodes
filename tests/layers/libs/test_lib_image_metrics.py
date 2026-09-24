"""Golden: the five Image Quality Gate metrics (target libs/image_metrics.py).

Each metric runs on a uint8 grey image and its result is recorded as repr, so the exact float
and its type (Python float for blur, np.float64 for the others) are pinned. Inputs:
  - "uniform": RandomState(0) uint8 (64, 96), full range;
  - "graded": RandomState(0) noise whose amplitude grows from 0 to 6 grey levels left to right on
    mid grey, so some blocks fall under the Laplacian-variance threshold and some do not.
blur_detection runs for block 16/24/32 x centre-weighted False/True x var threshold 30/80, and
once with its defaults (block only); clipping_analysis with its default threshold.

The metrics are plain functions of libs/image_metrics.py, reached through WHERE.
"""

import numpy as np
import pytest

from _golden import Where, check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'numpy': '2.5.3',
    'cv2': '5.0.0',
}

GOLDEN = {
    'uniform/sharpness_hybrid': 'np.float64(53770.71847959337)',
    'uniform/noise_estimation': 'np.float64(55.780988693237305)',
    'uniform/clipping_analysis': 'np.float64(0.047688802083333336)',
    'uniform/entropy_analysis': 'np.float64(7.968535562932153)',
    'graded/sharpness_hybrid': 'np.float64(119.10271334499782)',
    'graded/noise_estimation': 'np.float64(2.1114063262939453)',
    'graded/clipping_analysis': 'np.float64(0.0)',
    'graded/entropy_analysis': 'np.float64(3.663327810840612)',
    'uniform/blur_detection/24/defaults': '0.0',
    'graded/blur_detection/24/defaults': '0.25',
    'uniform/blur_detection/16/False/30.0': '0.0',
    'uniform/blur_detection/16/False/80.0': '0.0',
    'uniform/blur_detection/16/True/30.0': '0.0',
    'uniform/blur_detection/16/True/80.0': '0.0',
    'uniform/blur_detection/24/False/30.0': '0.0',
    'uniform/blur_detection/24/False/80.0': '0.0',
    'uniform/blur_detection/24/True/30.0': '0.0',
    'uniform/blur_detection/24/True/80.0': '0.0',
    'uniform/blur_detection/32/False/30.0': '0.0',
    'uniform/blur_detection/32/False/80.0': '0.0',
    'uniform/blur_detection/32/True/30.0': '0.0',
    'uniform/blur_detection/32/True/80.0': '0.0',
    'graded/blur_detection/16/False/30.0': '0.16666666666666666',
    'graded/blur_detection/16/False/80.0': '0.3333333333333333',
    'graded/blur_detection/16/True/30.0': '0.04545454545454545',
    'graded/blur_detection/16/True/80.0': '0.19696969696969696',
    'graded/blur_detection/24/False/30.0': '0.25',
    'graded/blur_detection/24/False/80.0': '0.25',
    'graded/blur_detection/24/True/30.0': '0.06521739130434782',
    'graded/blur_detection/24/True/80.0': '0.06521739130434782',
    'graded/blur_detection/32/False/30.0': '0.3333333333333333',
    'graded/blur_detection/32/False/80.0': '0.3333333333333333',
    'graded/blur_detection/32/True/30.0': '0.08333333333333334',
    'graded/blur_detection/32/True/80.0': '0.08333333333333334',
}

WHERE = Where({
    "blur_detection": "libs.image_metrics:blur_detection",
    "sharpness_hybrid": "libs.image_metrics:sharpness_hybrid",
    "noise_estimation": "libs.image_metrics:noise_estimation",
    "clipping_analysis": "libs.image_metrics:clipping_analysis",
    "entropy_analysis": "libs.image_metrics:entropy_analysis",
})


def _uniform():
    return np.random.RandomState(0).randint(0, 256, size=(64, 96), dtype=np.uint8)


def _graded():
    amp = np.linspace(0.0, 6.0, 96)[None, :]
    noise = np.random.RandomState(0).standard_normal((64, 96))
    return np.clip(np.round(128.0 + noise * amp), 0, 255).astype(np.uint8)


IMAGES = {"uniform": _uniform, "graded": _graded}
METRICS = ["sharpness_hybrid", "noise_estimation", "clipping_analysis", "entropy_analysis"]


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("image", list(IMAGES))
def test_metric(image, metric, bcnodes):
    check_env(ENV, "numpy", "cv2")
    check(GOLDEN, f"{image}/{metric}", repr(WHERE[metric](IMAGES[image]())))


@pytest.mark.parametrize("image", list(IMAGES))
def test_blur_detection_defaults(image, bcnodes):
    check_env(ENV, "numpy", "cv2")
    check(GOLDEN, f"{image}/blur_detection/24/defaults", repr(WHERE["blur_detection"](IMAGES[image](), 24)))


@pytest.mark.parametrize("var_threshold", [30.0, 80.0])
@pytest.mark.parametrize("center_weighted", [False, True])
@pytest.mark.parametrize("block", [16, 24, 32])
@pytest.mark.parametrize("image", list(IMAGES))
def test_blur_detection(image, block, center_weighted, var_threshold, bcnodes):
    check_env(ENV, "numpy", "cv2")
    value = WHERE["blur_detection"](IMAGES[image](), block, center_weighted, var_threshold)
    check(GOLDEN, f"{image}/blur_detection/{block}/{center_weighted}/{var_threshold}", repr(value))

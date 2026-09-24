"""Golden: Skin Texture's face gate, called directly, on 2-D (H, W) and 3-D (B, H, W) face masks:
no face, a face spanning a fraction of the frame, at and past FACE_FULL_FRACTION, two blobs
(the span counts, not the area), rows of several frames combined, another frame height, and
float / bool / int masks. The exact float and its type are pinned.
"""

import pytest
import torch

from _golden import Where, check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'none': ('1.0', 'float'),
    'zeros_2d': ('1.0', 'float'),
    'zeros_3d': ('1.0', 'float'),
    'tenth_2d': ('0.33333333333333337', 'float'),
    'tenth_3d': ('0.33333333333333337', 'float'),
    'one_row': ('0.03333333333333333', 'float'),
    'exactly_full': ('1.0', 'float'),
    'just_past_full': ('1.0', 'float'),
    'whole_frame': ('1.0', 'float'),
    'two_blobs_span': ('0.4', 'float'),
    'frames_combined': ('0.4', 'float'),
    'frames_combined_past_full': ('1.0', 'float'),
    'other_height': ('0.33333333333333337', 'float'),
    'soft_values': ('0.4', 'float'),
    'bool': ('0.4', 'float'),
    'int': ('0.4', 'float'),
    'negative': ('0.4', 'float'),
    'tenth_of_90': ('0.33333333333333337', 'float'),
}

WHERE = Where({
    "face_gate": "pipelines.skin_texture:face_gate",
})


def _mask(shape, *spans, value=1.0, dtype=torch.float32):
    """Zeros of `shape` with rows [top, bottom) set to `value` in columns 3:7; a span may carry a
    frame index as its third item (3-D masks)."""
    m = torch.zeros(shape, dtype=dtype)
    for span in spans:
        top, bottom = span[0], span[1]
        if len(shape) == 3:
            m[span[2], top:bottom, 3:7] = value
        else:
            m[top:bottom, 3:7] = value
    return m


CASES = {
    "none": (lambda: None, 100),
    "zeros_2d": (lambda: torch.zeros((100, 100)), 100),
    "zeros_3d": (lambda: torch.zeros((1, 100, 100)), 100),
    "tenth_2d": (lambda: _mask((100, 10), (40, 50)), 100),
    "tenth_3d": (lambda: _mask((1, 100, 10), (40, 50, 0)), 100),
    "one_row": (lambda: _mask((100, 10), (7, 8)), 100),
    "exactly_full": (lambda: _mask((10, 10), (0, 3)), 10),
    "just_past_full": (lambda: _mask((100, 10), (0, 31)), 100),
    "whole_frame": (lambda: torch.ones((64, 80)), 64),
    "two_blobs_span": (lambda: _mask((100, 10), (40, 45), (50, 52)), 100),
    "frames_combined": (lambda: _mask((2, 100, 10), (10, 15, 0), (20, 22, 1)), 100),
    "frames_combined_past_full": (lambda: _mask((2, 100, 10), (10, 15, 0), (60, 70, 1)), 100),
    "other_height": (lambda: _mask((50, 10), (10, 20)), 100),
    "soft_values": (lambda: _mask((100, 10), (40, 52), value=0.2), 100),
    "bool": (lambda: _mask((100, 10), (40, 52), dtype=torch.bool), 100),
    "int": (lambda: _mask((1, 100, 10), (40, 52, 0), dtype=torch.int64), 100),
    "negative": (lambda: _mask((100, 10), (40, 52), value=-1.0), 100),
    "tenth_of_90": (lambda: _mask((90, 10), (0, 9)), 90),
}


@pytest.mark.parametrize("name", list(CASES))
def test_face_gate(name, bcnodes):
    check_env(ENV, "torch")
    mask, height = CASES[name]
    value = WHERE["face_gate"](mask(), height)
    check(GOLDEN, name, (repr(value), type(value).__name__))

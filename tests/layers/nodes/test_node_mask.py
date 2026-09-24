"""Golden: Mask Fill Holes and MaskGrow, exact output digests through the node methods.

MaskGrow: the 20 (grow, blur, invert) combos of tests/test_nodes.py on the centred 16x16
square, and a seeded soft mask (its values survive neither 8-bit round trip unchanged, so both
show). Mask Fill Holes: a ring with a hole, a ring with a gap in its wall, a U open to the image
edge, rings drawn at 0.003 and 0.004 (8-bit 0 and 1). Both nodes: (H, W), (B, 1, H, W) and fp16
input, and the blank (1, 64, 64) for None, an empty batch, a 1-D tensor and a non-tensor.
"""

import pytest
import torch

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'scipy': '1.18.1',
    'Pillow': '12.3.0',
}

GOLDEN = {
    'MaskGrow/square/-6/0/False': 'e33bf03c5f3b841fd375729251114a33',
    'MaskGrow/square/-6/0/True': '2204ecfe502c73c3ec0791fc56a44c3b',
    'MaskGrow/square/-6/6/False': '1d19fefcea661cd55653885441dc262d',
    'MaskGrow/square/-6/6/True': '49c2537dc8030cad7608b2f429d5cf26',
    'MaskGrow/square/-4/0/False': '8351b72af9a5f9840374ee8426e14600',
    'MaskGrow/square/-4/0/True': 'b6c581a7bf261f702a4dd2c65d06e4cb',
    'MaskGrow/square/-4/6/False': '33d0823951b972c4787f39d76514c411',
    'MaskGrow/square/-4/6/True': 'c3b007f2dd2bf93d698219610ec4498e',
    'MaskGrow/square/0/0/False': '408ea7911b6bf98928968221accea5b1',
    'MaskGrow/square/0/0/True': '88e82c2091e8d8c282e107fea47ed5f7',
    'MaskGrow/square/0/6/False': '622fb3a45c2f3942e610be3831a4cc8a',
    'MaskGrow/square/0/6/True': '28157dc8264ab751dd8491137e3f2d21',
    'MaskGrow/square/4/0/False': 'cc147a4ae7c994936443645779d858b4',
    'MaskGrow/square/4/0/True': 'de7ac51a5b4833a1804d6faa7530ea8e',
    'MaskGrow/square/4/6/False': '385ea3052c0dd4a57edb18f98a8f08b2',
    'MaskGrow/square/4/6/True': '612606e95dc093681d2cf06de5eef2a1',
    'MaskGrow/square/10/0/False': 'aea255a226793b7a09c13776d251e461',
    'MaskGrow/square/10/0/True': 'b85591dc6fffad597dc6fcd4580288ab',
    'MaskGrow/square/10/6/False': '4757d86fb462672e32a56369f7898e54',
    'MaskGrow/square/10/6/True': 'b85591dc6fffad597dc6fcd4580288ab',
    'MaskGrow/soft/4/4/False': '6d9857d51691295d710fd2f7709167f3',
    'MaskGrow/soft/0/0/False': '56a517b98d9a0051b519489c5eb1a246',
    'MaskGrow/soft/0/0/True': '5e2e6e44ee848e878cc2fa8704e0d8e0',
    'MaskGrow/soft/-2/0/True': '552bee3126eb65aa837d6418631a1395',
    'MaskGrow/soft/3/1/False': '39e3cd4666f082bbe8a4283469a803cc',
    'MaskGrow/soft/0/2/True': 'b11c6e781ff4cd6d5448eaab5fe2ada6',
    'MaskGrow/soft/-1/3/False': 'e0d4370647c87520250577dc866d51a5',
    'MaskFillHoles/ring': '8b8b37a33d968ccb59e86aae95a38c01',
    'MaskFillHoles/ring_with_gap': 'da339652bd8aecf05cd1f009ce7b2015',
    'MaskFillHoles/u_open_to_edge': 'd9d14a1cdc557a8d6ee1a47e71b89b9e',
    'MaskFillHoles/ring_0.004': '8b8b37a33d968ccb59e86aae95a38c01',
    'MaskFillHoles/ring_0.003': 'd7e64a1c1d73cedbc3e7f9b75fdb9302',
    'MaskFillHoles/soft': '97d0d4bff35a8c23af99943e90a3bfeb',
    'MaskFillHoles/two_frames': '257e3155bd7a47a1a8014f7b2ecb6595',
    'shapes/2d': 'd420e311d3a89fe8a181310128580cd5',
    'shapes/b1hw': '0cd3fb381805c339d6d484f7ad41d780',
    'shapes/fp16': '47915c77d18edf329a3be19ad2e9804f',
    'shapes/ring_2d': '7d139a5e1d981cc482225ba44147db99',
    'blank/None': 'f121501fbe85766df21a2520a71f3480',
    'blank/empty_batch': 'f121501fbe85766df21a2520a71f3480',
    'blank/empty_width': 'f121501fbe85766df21a2520a71f3480',
    'blank/1d': 'f121501fbe85766df21a2520a71f3480',
    'blank/list': 'f121501fbe85766df21a2520a71f3480',
    'blank/nothing_wired': 'f121501fbe85766df21a2520a71f3480',
}


def _square():
    square = torch.zeros(1, 32, 32)
    square[:, 8:24, 8:24] = 1.0
    return square


def _soft(shape):
    return torch.rand(shape, generator=torch.Generator().manual_seed(21))


def _ring(value=1.0, gap=False):
    ring = torch.zeros(1, 20, 24)
    ring[:, 4:14, 5:17] = value
    ring[:, 6:12, 7:15] = 0.0
    if gap:
        ring[:, 8:10, 5:7] = 0.0
    return ring


def _u_open_to_edge():
    u = torch.zeros(1, 16, 16)
    u[:, 0:10, 4:12] = 1.0
    u[:, 0:8, 6:10] = 0.0
    return u


GROW_COMBOS = [(g, b, i) for g in (-6, -4, 0, 4, 10) for b in (0, 6) for i in (False, True)]


@pytest.mark.parametrize("grow,blur,invert", GROW_COMBOS)
def test_mask_grow_square(grow, blur, invert, bcnodes):
    check_env(ENV, "torch", "numpy", "scipy", "Pillow")
    out = bcnodes["mask"].MaskGrow().mask_grow(invert_mask=invert, grow=grow, blur=blur, mask=_square())
    check(GOLDEN, f"MaskGrow/square/{grow}/{blur}/{invert}", digest(out))


SOFT_COMBOS = [(4, 4, False), (0, 0, False), (0, 0, True), (-2, 0, True), (3, 1, False), (0, 2, True), (-1, 3, False)]


@pytest.mark.parametrize("grow,blur,invert", SOFT_COMBOS)
def test_mask_grow_soft(grow, blur, invert, bcnodes):
    out = bcnodes["mask"].MaskGrow().mask_grow(invert, grow, blur, mask=_soft((1, 24, 32)))
    check(GOLDEN, f"MaskGrow/soft/{grow}/{blur}/{invert}", digest(out))


FILL = {
    "ring": lambda: _ring(),
    "ring_with_gap": lambda: _ring(gap=True),
    "u_open_to_edge": _u_open_to_edge,
    "ring_0.004": lambda: _ring(0.004),
    "ring_0.003": lambda: _ring(0.003),
    "soft": lambda: _soft((1, 24, 32)),
    "two_frames": lambda: torch.cat([_ring(), _ring(gap=True)]),
}


@pytest.mark.parametrize("name", list(FILL))
def test_mask_fill_holes(name, bcnodes):
    check_env(ENV, "torch", "numpy", "scipy", "Pillow")
    check(GOLDEN, f"MaskFillHoles/{name}", digest(bcnodes["mask"].MaskFillHoles().fill_region(FILL[name]())))


SHAPES = {
    "2d": lambda: _soft((24, 32)),
    "b1hw": lambda: _soft((2, 1, 24, 32)),
    "fp16": lambda: _soft((1, 24, 32)).half(),
    "ring_2d": lambda: _ring()[0],
}


@pytest.mark.parametrize("name", list(SHAPES))
def test_input_shapes(name, bcnodes):
    node = bcnodes["mask"]
    check(GOLDEN, f"shapes/{name}", digest({
        "MaskGrow": node.MaskGrow().mask_grow(False, 2, 1, mask=SHAPES[name]()),
        "MaskFillHoles": node.MaskFillHoles().fill_region(SHAPES[name]()),
    }))


BLANK = {
    "None": lambda: None,
    "empty_batch": lambda: torch.zeros((0, 8, 8)),
    "empty_width": lambda: torch.zeros((1, 8, 0)),
    "1d": lambda: torch.ones(8),
    "list": lambda: [[1.0, 0.0], [0.0, 1.0]],
}


@pytest.mark.parametrize("name", list(BLANK))
def test_blank(name, bcnodes):
    node = bcnodes["mask"]
    check(GOLDEN, f"blank/{name}", digest({
        "MaskGrow": node.MaskGrow().mask_grow(True, 4, 4, mask=BLANK[name]()),
        "MaskFillHoles": node.MaskFillHoles().fill_region(BLANK[name]()),
    }))


def test_nothing_wired(bcnodes):
    node = bcnodes["mask"]
    check(GOLDEN, "blank/nothing_wired", digest({
        "MaskGrow": node.MaskGrow().mask_grow(False, 4, 4),
        "MaskFillHoles": node.MaskFillHoles().fill_region(),
    }))

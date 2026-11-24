"""DyPE's frequency tables (libs/dype/rope.py), at Z-Image's sizes (an image axis of 48 dimensions, theta 256, a
RoPE length of 512):

- the plain table is core's omega (float64, linspace exponents); the NTK table at factor 1 is the plain one;
- YaRN's correction range: the frequency indices turning 1.25 / 0.75 and 16 / 2 times over 512 positions, each
  ratio raised to k_t (DyPE), clamped to [0, dim / 2];
- the ramp weight: 1 up to low, 0 from high, linear between, and a range of one index still splits;
- Vision YaRN: at scale 1 the plain table; at a scale s the fastest frequency stays plain and the slowest are the
  plain ones over s (interpolated), the blend between the two ranges;
- YaRN's attention scale 1 + 0.1 ln(s) / sqrt(s), 1 when the stretch is not over the base length.
"""

import math

import pytest
import torch

DIM, THETA, LENGTH = 48, 256.0, 512


@pytest.fixture
def rope(bcnodes):
    return bcnodes["libs.dype.rope"]


def test_the_plain_table_is_cores_omega(rope):
    core = 1.0 / (THETA ** torch.linspace(0, (DIM - 2) / DIM, steps=DIM // 2, dtype=torch.float64))
    assert rope.base(DIM, THETA, "cpu").dtype == torch.float64
    assert torch.equal(rope.base(DIM, THETA, "cpu"), core)
    assert torch.equal(rope.ntk(DIM, THETA, 1.0, "cpu"), core)
    assert torch.equal(rope.ntk(DIM, THETA, 3.0, "cpu"), 1.0 / ((THETA * 3.0) ** torch.linspace(0, (DIM - 2) / DIM, DIM // 2,
                                                                                                 dtype=torch.float64)))


def test_the_correction_range_follows_yarns_formula_and_k_t(rope):
    def index(rotations):
        return DIM * math.log(LENGTH / (rotations * 2 * math.pi)) / (2 * math.log(THETA))

    assert rope.correction_range(1.25, 0.75, DIM, THETA, LENGTH) == (math.floor(index(1.25)), math.ceil(index(0.75))) == (18, 21)
    assert rope.correction_range(16.0, 2.0, DIM, THETA, LENGTH) == (7, 17)
    # k_t 2 at the start of a run: both ramps widen
    assert rope.correction_range(1.25 ** 2, 0.75 ** 2, DIM, THETA, LENGTH) == (17, 22)
    assert rope.correction_range(16.0 ** 2, 2.0 ** 2, DIM, THETA, LENGTH) == (0, 14)
    # clamped: a ratio turning more often than any frequency starts at 0, one slower ends at dim / 2
    assert rope.correction_range(1e6, 1e-6, DIM, THETA, LENGTH) == (0, DIM // 2)


def test_the_ramp_weight(rope):
    w = rope.below(2, 6, 8, "cpu")
    assert w.tolist() == [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0]
    w = rope.below(3, 3, 6, "cpu")  # low == high: split at that index
    assert w[:4].tolist() == [1.0, 1.0, 1.0, 1.0] and w[4:].tolist() == [0.0, 0.0]


def test_vision_yarn(rope):
    plain = rope.base(DIM, THETA, "cpu")
    assert torch.allclose(rope.vision_yarn(DIM, THETA, LENGTH, 1.0, 1.0, 2.0, "cpu"), plain, rtol=1e-15, atol=0)
    table = rope.vision_yarn(DIM, THETA, LENGTH, 3.0, 3.0, 2.0, "cpu")
    ratio = (table / plain).tolist()
    assert ratio[0] == 1.0 and ratio[-2:] == [1 / 3, 1 / 3]  # the fastest plain, the slowest interpolated
    assert all(a >= b for a, b in zip(ratio, ratio[1:]))  # from plain to interpolated, never back
    # a scale below 1 is taken as 1
    assert torch.equal(rope.vision_yarn(DIM, THETA, LENGTH, 0.5, 0.5, 2.0, "cpu"), rope.vision_yarn(DIM, THETA, LENGTH, 1.0, 1.0, 2.0, "cpu"))


def test_yarn_and_its_attention_scale(rope):
    table, scale = rope.yarn(DIM, THETA, LENGTH, LENGTH * 3, 1.0, "cpu")
    assert scale == pytest.approx(1 + 0.1 * math.log(3) / math.sqrt(3))
    plain = rope.base(DIM, THETA, "cpu")
    assert table[0] == plain[0] and table[-1] == pytest.approx(plain[-1] / 3)
    assert rope.yarn(DIM, THETA, LENGTH, LENGTH, 1.0, "cpu")[1] == 1.0
    assert rope.yarn(DIM, THETA, LENGTH, LENGTH // 2, 1.0, "cpu")[1] == 1.0

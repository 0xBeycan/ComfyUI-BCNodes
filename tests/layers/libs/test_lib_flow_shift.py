"""libs/flow_shift.py against diffusers' own schedules (tests/_diffusers_sigmas.py): the pipeline's token
count, mu, and the whole schedule (diffusers' timestep grid, the flow shift at e^mu, the terminal stretch)
for every reference case, step counts that do not divide 1000 and token counts past max_seq_len included;
the stretch on a schedule's tail (denoise below 1) equal to the stretch of the whole.

The flow shift here is written out (S t / (1 + (S - 1) t), core's time_snr_shift); the node applies
core's own function, checked against ModelSamplingAuraFlow in tests/test_runtime.py (qwen_image_shift).
"""

import math

import numpy as np
import pytest
import torch

from _diffusers_sigmas import ATOL, DIFFUSERS, OFFICIAL


@pytest.fixture
def fs(bcnodes):
    return bcnodes["libs.flow_shift"]


def flow_shift(shift, t):
    return shift * t / (1 + (shift - 1) * t)


def test_image_seq_len(fs):
    for width, height, _, seq, _, _ in DIFFUSERS.values():
        # the pipeline's latent sides: 2 * (side // 16)
        assert fs.image_seq_len(2 * (height // 16), 2 * (width // 16)) == seq
    assert fs.image_seq_len(166, 166) == 83 * 83
    assert fs.image_seq_len(207, 207) == 104 * 104  # odd sides padded, as the model pads them
    assert fs.image_seq_len(1, 3) == 2


def test_mu(fs):
    for name, (_, _, _, seq, mu, _) in DIFFUSERS.items():
        if name.startswith("lightning"):
            got = fs.mu(seq, 256, 8192, math.log(3), math.log(3))
        else:
            got = fs.mu(seq, OFFICIAL["base_seq_len"], OFFICIAL["max_seq_len"], OFFICIAL["base_shift"], OFFICIAL["max_shift"])
        assert got == mu, name
    # past max_seq_len the line goes on (10609 tokens: 1660 x 1660)
    assert DIFFUSERS["1660x1660_20_terminal"][4] > OFFICIAL["max_shift"]


@pytest.mark.parametrize("name", list(DIFFUSERS))
def test_schedule_matches_diffusers(fs, name):
    _, _, steps, seq, _, expected = DIFFUSERS[name]
    if name.startswith("lightning"):
        m, terminal = math.log(3), None
    else:
        m = fs.mu(seq, OFFICIAL["base_seq_len"], OFFICIAL["max_seq_len"], OFFICIAL["base_shift"], OFFICIAL["max_shift"])
        terminal = None if name.endswith("_no_terminal") else OFFICIAL["terminal"]
    grid = np.linspace(1.0, 1 / steps, steps).astype(np.float32)  # the pipeline's grid
    t = torch.cat([torch.from_numpy(grid), torch.zeros(1)])
    sigmas = flow_shift(math.exp(m), t)
    if terminal is not None:
        sigmas = fs.stretch_to_terminal(sigmas, terminal)
    expected = torch.tensor(expected)
    assert sigmas.shape == expected.shape
    assert torch.allclose(sigmas, expected, rtol=0, atol=ATOL), (sigmas - expected).abs().max().item()
    if terminal is not None:
        assert math.isclose(sigmas[-2].item(), terminal, abs_tol=ATOL) and sigmas[-1].item() == 0.0


def test_stretch_keeps_the_input_and_the_final_zero(fs):
    sigmas = torch.tensor([1.0, 0.8, 0.5, 0.25, 0.0])
    before = sigmas.clone()
    out = fs.stretch_to_terminal(sigmas, 0.02)
    assert torch.equal(sigmas, before) and out is not sigmas
    assert out[0].item() == 1.0 and out[-1].item() == 0.0 and math.isclose(out[-2].item(), 0.02, abs_tol=ATOL)
    # 1 - sigma scaled by one factor
    ratio = (1 - out[:-1]) / (1 - sigmas[:-1])
    assert torch.allclose(ratio[1:], torch.full((3,), 0.98 / 0.75))


def test_stretch_of_a_tail_is_the_tail_of_the_stretch(fs):
    """KSampler at denoise below 1 samples the tail of a longer schedule; it ends at the same last sigma,
    so stretching the tail gives the stretched schedule's tail."""
    full = flow_shift(2.3, torch.cat([torch.linspace(1.0, 1 / 20, 20), torch.zeros(1)]))
    whole = fs.stretch_to_terminal(full, 0.02)
    for keep in (2, 5, 11):
        assert torch.allclose(fs.stretch_to_terminal(full[-keep:], 0.02), whole[-keep:], rtol=0, atol=ATOL)

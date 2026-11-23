"""libs/detail_boost.py: every mode's lowered sigma at one point, hard's closed forms (flow and other
models), eta 0 leaving every mode at the call's sigma, the cap for a flow model and the formulas with no
real answer; where a call sits in the run (schedule points, calls between them, half-precision sigmas,
outside the run); and the step range, the run's first call and the weight blend.
"""

import math

import pytest
import torch


@pytest.fixture
def db(bcnodes):
    return bcnodes["libs.detail_boost"]


# A flow model's call at sigma 0.4 after the schedule point 0.5, eta 0.5.
AT_04_AFTER_05 = {
    "hard": 0.3660254037844387,
    "soft": 0.22516254443257336,
    "softer": 0.3997451649725273,
    "soft-linear": 0.3660254037844387,
    "sinusoidal": 0.37288061334554695,
    "eps": 0.3982607066742351,
    "exp": 0.373545369999913,
    "lorentzian": 0.37125643804997577,
    "hard_var": 0.4,  # 0.4 > (sqrt(1 + 4 * 0.5) - 1) / 2: not a late step
    "er4": 0.3970748645020685,
    "vpsde": 0.39687500000000003,
}


def test_every_mode_has_a_value(db):
    assert set(AT_04_AFTER_05) == set(db.MODES)
    for mode, expected in AT_04_AFTER_05.items():
        assert db.lowered_sigma(mode, 0.4, 0.5, 0.5, True, 1.0) == pytest.approx(expected, rel=1e-12), mode


def test_hard_flow_scales_noise_to_signal(db):
    for sigma in (0.95, 0.6, 0.2, 0.05):
        for eta in (0.1, 0.5, 0.9):
            low = db.lowered_sigma("hard", sigma, min(1.0, sigma + 0.1), eta, True, 1.0)
            assert low / (1 - low) == pytest.approx(sigma / (1 - sigma) * math.sqrt(1 - eta ** 2), rel=1e-12)


def test_hard_other_models_scale_sigma(db):
    assert db.lowered_sigma("hard", 3.0, 6.0, 0.5, False, 14.6146) == pytest.approx(2.598076211353316, rel=1e-12)


def test_hard_var_late_step(db):
    assert db.lowered_sigma("hard_var", 0.2, 0.3, 0.5, True, 1.0) == pytest.approx(0.13276734344217328, rel=1e-12)


def test_eta_zero_leaves_every_mode(db):
    for mode in db.MODES:
        assert db.lowered_sigma(mode, 0.4, 0.5, 0.0, True, 1.0) == pytest.approx(0.4, rel=1e-12), mode


def test_flow_cap(db):
    capped = db.lowered_sigma("hard", 0.6, 0.7, 1.0, True, 1.0)
    assert capped == pytest.approx(0.0207720425506497, rel=1e-12)
    assert db.lowered_sigma("hard", 0.6, 0.7, 2.0, True, 1.0) == capped
    # soft-linear's ratio is above 1 on a falling schedule: capped to eta * sigma, hard's own
    assert db.lowered_sigma("soft-linear", 0.6, 0.7, 0.5, True, 1.0) == pytest.approx(
        db.lowered_sigma("hard", 0.6, 0.7, 0.5, True, 1.0), rel=1e-12)


def test_no_real_answer_keeps_sigma(db):
    assert db.lowered_sigma("softer", 0.2, 0.6, 3.0, True, 1.0) == 0.2  # sqrt(1 - 9 * 0.32 / 0.36)
    assert db.lowered_sigma("hard", 3.0, 6.0, 1.5, False, 14.6146) == 3.0  # up over sigma, no cap off flow


def test_unknown_mode(db):
    with pytest.raises(ValueError, match="unknown mode 'nope'"):
        db.lowered_sigma("nope", 0.4, 0.5, 0.5, True, 1.0)


SCHEDULE = [1.0, 0.8, 0.5, 0.2, 0.0]


def test_locate(db):
    assert db.locate(1.0, SCHEDULE) == (0, None)
    assert db.locate(0.9, SCHEDULE) == (0, 1.0)
    assert db.locate(0.8, SCHEDULE) == (1, 1.0)
    assert db.locate(0.65, SCHEDULE) == (1, 0.8)
    assert db.locate(0.2, SCHEDULE) == (3, 0.5)
    assert db.locate(0.1, SCHEDULE) == (3, 0.2)
    # a half-precision latent's sigma is the schedule point it rounds
    half = torch.tensor(SCHEDULE[1], dtype=torch.float16).item()
    assert half != SCHEDULE[1] and db.locate(half, SCHEDULE) == (1, 1.0)
    assert db.locate(1.2, SCHEDULE) is None
    assert db.locate(0.0, SCHEDULE) is None
    # a run that stops before 0 (KSamplerAdvanced, leftover noise): its last point is not a step
    assert db.locate(0.5, [1.0, 0.8, 0.5]) is None


def settings(db, start=3, end=10, weight=1.0):
    return db.Settings(weight=weight, mode="hard", eta=0.5, start_step=start, end_step=end)


def test_step_range(db):
    schedule = [1.0 - i / 20 for i in range(21)]
    told = [db.model_sigma(settings(db), s, schedule, True, 1.0) for s in schedule[:-1]]
    for i, (s, t) in enumerate(zip(schedule, told)):
        if 3 <= i < 10:
            assert t == pytest.approx(db.lowered_sigma("hard", s, schedule[i - 1], 0.5, True, 1.0)) and t < s, i
        else:
            assert t == s, i
    to_end = [db.model_sigma(settings(db, 3, -1), s, schedule, True, 1.0) for s in schedule[:-1]]
    assert all(t == s for t, s in zip(to_end[:3], schedule)) and all(t < s for t, s in zip(to_end[3:], schedule[3:]))


def test_first_call_of_the_run(db):
    # point 0 has nothing before it; a call between points 0 and 1 has point 0
    assert db.model_sigma(settings(db, 0, -1), 1.0, SCHEDULE, True, 1.0) == 1.0
    assert db.model_sigma(settings(db, 0, -1), 0.9, SCHEDULE, True, 1.0) < 0.9


def test_weight_blend(db):
    low = db.lowered_sigma("hard", 0.5, 0.8, 0.5, True, 1.0)
    for weight in (2.0, 0.5, -1.0):
        assert db.model_sigma(settings(db, 0, -1, weight), 0.5, SCHEDULE, True, 1.0) == pytest.approx(0.5 + weight * (low - 0.5))
    assert db.model_sigma(settings(db, 0, -1, -1.0), 0.5, SCHEDULE, True, 1.0) > 0.5

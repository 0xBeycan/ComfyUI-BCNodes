"""Detail Boost: the noise level a model call is told, lowered inside a step range.

A sampler calls the model at its schedule's sigmas (some samplers also between them) and steps with
the prediction it gets back. Detail Boost tells the model a lower sigma than the sampler's in the steps
[start_step, end_step) of the run and leaves the sampler's own sigmas alone: the model takes the latent
for less noisy than it is and removes less noise, so fine structure the step would have smoothed away
stays (positive weight), or more (negative weight).

The lowered sigma is the sigma left after taking noise of size `up` out of the call's sigma, `up` being
the mode's ratio times a base sigma (the call's own, unless the mode says otherwise). For a flow model
(ComfyUI's CONST: x = (1 - sigma) x0 + sigma noise) what is left is renormalised by the signal, so the
noise-to-signal ratio sigma / (1 - sigma) shrinks by sqrt(1 - (up / sigma)^2); for any other model the
sigma itself does. `weight` blends: the model is told sigma + weight * (lowered - sigma).

Modes, with s the call's sigma and p the schedule sigma before it, both divided by the model's
sigma_max (1 for a flow model); the ratio applies to s unless a base is given:
    hard         eta
    soft         eta * (1 + s / p)
    softer       1 - sqrt(1 - eta^2 (p^2 - s^2) / p^2)
    soft-linear  1 - eta (s - p)
    sinusoidal   eta sin(pi s)^2
    eps          eta (s / p) sqrt(p^2 - s^2)
    exp          sqrt(1 - (s / p)^(2 eta))
    lorentzian   eta, base sigma_max * s / sqrt(s^2 + 1)
    hard_var     eta, base sqrt(sigma_prev - sigma), where s <= (sqrt(1 + 4p) - 1) / 2; else 0
    er4          sqrt(sigma^2 - sigma_prev^2 (n(s) / n(p))^2) / sigma, n(v) = v (e^(v^eta) + 10)
    vpsde        no ratio: the lowered sigma is sigma - eta / 4 * sigma_prev (1 - sigma_prev) (sigma_prev - sigma)
For a flow model an `up` at or above the call's sigma is capped to eta * sigma (0.9999 * sigma from eta 1
on); a formula with no real answer (a negative square root, an `up` over the sigma of another model)
leaves the call's sigma as it is.

Only the standard library here; the node reads ComfyUI's objects and hands in plain floats.
"""

import dataclasses
import math
from typing import Optional, Sequence

MODES = ("hard", "soft", "softer", "soft-linear", "sinusoidal", "eps", "exp", "lorentzian", "hard_var", "er4", "vpsde")
# A call sigma within this relative distance of a schedule sigma is that schedule point: a half-precision
# latent rounds the sampler's sigma to 11 bits (4.9e-4 relative).
MATCH_TOLERANCE = 1e-3


@dataclasses.dataclass(frozen=True)
class Settings:
    weight: float
    mode: str
    eta: float
    start_step: int
    end_step: int  # exclusive; -1: to the end of the run


def _sqrt(v: float) -> float:
    return math.sqrt(v) if v >= 0 else math.nan


def lowered_sigma(mode: str, sigma: float, prev: float, eta: float, flow: bool, sigma_max: float) -> float:
    """The lowered sigma for a call at `sigma` whose schedule point before it is `prev` (> sigma);
    `sigma` itself where the mode's formula has no real answer."""
    if mode == "vpsde":
        low = sigma - eta / 4 * prev * (1 - prev) * (prev - sigma)
        return low if math.isfinite(low) else sigma
    s, p = sigma / sigma_max, prev / sigma_max
    base = sigma
    if mode == "hard":
        ratio = eta
    elif mode == "soft":
        ratio = eta * (1 + s / p)
    elif mode == "softer":
        ratio = 1 - _sqrt(1 - eta ** 2 * (p ** 2 - s ** 2) / p ** 2)
    elif mode == "soft-linear":
        ratio = 1 - eta * (s - p)
    elif mode == "sinusoidal":
        ratio = eta * math.sin(math.pi * s) ** 2
    elif mode == "eps":
        ratio = eta * (s / p) * _sqrt(p ** 2 - s ** 2)
    elif mode == "exp":
        ratio = _sqrt(1 - (s / p) ** (2 * eta))
    elif mode == "lorentzian":
        ratio = eta
        base = sigma_max * s / math.sqrt(s ** 2 + 1)
    elif mode == "hard_var":
        if s > (math.sqrt(1 + 4 * p) - 1) / 2:
            return sigma
        ratio = eta
        base = math.sqrt(abs(prev - sigma) + 1e-10)
    elif mode == "er4":
        def n(v):
            return v * (math.exp(v ** eta) + 10)
        ratio = _sqrt(sigma ** 2 - prev ** 2 * (n(s) / n(p)) ** 2) / sigma
    else:
        raise ValueError(f"Detail Boost: unknown mode {mode!r}; use one of {', '.join(MODES)}.")
    up = base * ratio
    if math.isnan(up):
        return sigma
    if flow:
        if up >= sigma:
            up = sigma * (0.9999 if eta >= 1 else eta)
        rest = math.sqrt(sigma ** 2 - up ** 2)
        return rest / (1 - sigma + rest)
    low = _sqrt(sigma ** 2 - up ** 2)
    return sigma if math.isnan(low) else low


def locate(sigma: float, schedule: Sequence[float]) -> Optional[tuple[int, Optional[float]]]:
    """-> (step, the schedule sigma before `sigma`) for a call at `sigma` in a run over `schedule`
    (falling; its last entry is where the run ends), None outside it. A call at point i is step i with
    point i - 1 before it (None at point 0); a call between points i and i + 1 is step i with point i
    before it."""
    def at(k):
        return abs(sigma - schedule[k]) <= MATCH_TOLERANCE * schedule[k]

    for i in range(len(schedule) - 1):
        if at(i):
            return i, (schedule[i - 1] if i else None)
        if schedule[i + 1] < sigma < schedule[i] and not at(i + 1):
            return i, schedule[i]
    return None


def model_sigma(settings: Settings, sigma: float, schedule: Sequence[float], flow: bool, sigma_max: float) -> float:
    """The sigma to tell the model for a call at `sigma` in a run over `schedule`: `sigma` itself
    outside [start_step, end_step), at the run's first point (no point before it) and outside the run."""
    found = locate(sigma, schedule)
    if found is None:
        return sigma
    step, prev = found
    if prev is None or step < settings.start_step or (settings.end_step != -1 and step >= settings.end_step):
        return sigma
    low = lowered_sigma(settings.mode, sigma, prev, settings.eta, flow, sigma_max)
    return sigma + settings.weight * (low - sigma)

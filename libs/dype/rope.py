# DyPE — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
#
# Derived from ComfyUI-DyPE (https://github.com/wildminder/ComfyUI-DyPE, src/rope.py at 2664058), which
# implements DyPE (Dynamic Position Extrapolation; Issachar, Yariv et al., https://github.com/guyyariv/DyPE)
# over YaRN's NTK-by-parts interpolation. Changed (NOTICE in this directory lists every change):
# - the frequencies are float64 on the caller's device, as ComfyUI core's rope (comfy/ldm/flux/math.py)
#   computes them, in place of the original's caller-chosen dtype (bfloat16 on CUDA); the exponents are
#   core's linspace(0, (dim - 2) / dim);
# - the functions return the frequency table only; the caller takes the angles, their cos / sin and the
#   attention scale (the original returned cos / sin repeat-interleaved, which its Z-Image embedder
#   de-interleaved again);
# - DyPE's modulation exponent k_t = dype_scale * t ** dype_exponent is computed by the caller and passed in.

import math

import torch

METHODS = ("vision_yarn", "yarn", "ntk", "pi", "base")  # the default first
BETA = (1.25, 0.75)    # YaRN's ramp from the linear (PI) to the NTK frequencies, in rotations over the base length
GAMMA = (16.0, 2.0)    # its ramp from there back to the base frequencies


def exponents(dim, device):
    """The rotary exponents 2i / dim of the dim // 2 frequencies, float64, as core's rope makes them."""
    return torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)


def base(dim, theta, device):
    """The plain rotary frequencies theta ** -(2i / dim), float64 (core's omega)."""
    return 1.0 / (theta ** exponents(dim, device))


def ntk(dim, theta, factor, device):
    """The NTK-aware frequencies: the base frequencies of theta * factor."""
    return 1.0 / ((theta * factor) ** exponents(dim, device))


def correction_range(low_rotations, high_rotations, dim, theta, base_len):
    """(low, high): the frequency indices YaRN ramps between, those that turn `low_rotations` and
    `high_rotations` times over `base_len` positions, within [0, dim // 2]."""
    def index(rotations):
        return dim * math.log(base_len / (rotations * 2 * math.pi)) / (2 * math.log(theta))

    low = max(math.floor(index(low_rotations)), 0)
    high = min(math.ceil(index(high_rotations)), dim - 1)
    return max(0, low), min(dim // 2, high)


def below(low, high, n, device):
    """The weight of the lower-index frequencies in YaRN's ramp: 1 up to `low`, 0 from `high`, linear between."""
    if low == high:
        high += 0.001
    return 1 - ((torch.arange(n, dtype=torch.float64, device=device) - low) / (high - low)).clamp(0, 1)


def _parts(linear, ntk_freqs, plain, dim, theta, base_len, k_t, device):
    """YaRN's NTK-by-parts table: the NTK frequencies below the beta ramp, the linear ones above it, then the
    plain frequencies below the gamma ramp. k_t (DyPE) raises each ramp bound to that power; None leaves them."""
    beta, gamma = BETA, GAMMA
    if k_t is not None:
        beta = tuple(b ** k_t for b in beta)
        gamma = tuple(g ** k_t for g in gamma)
    weight = below(*correction_range(*beta, dim, theta, base_len), dim // 2, device)
    freqs = linear * (1 - weight) + ntk_freqs * weight
    weight = below(*correction_range(*gamma, dim, theta, base_len), dim // 2, device)
    return freqs * (1 - weight) + plain * weight


def vision_yarn(dim, theta, base_len, linear_scale, ntk_scale, k_t, device):
    """DyPE's Vision YaRN table: the linear frequencies at `linear_scale` and the NTK ones at `ntk_scale`
    (each at least 1), blended over `base_len` positions with the ramps raised to k_t."""
    linear_scale, ntk_scale = max(linear_scale, 1.0), max(ntk_scale, 1.0)
    plain = base(dim, theta, device)
    return _parts(plain / linear_scale, ntk(dim, theta, ntk_scale ** (dim / (dim - 2)), device), plain, dim, theta,
                  base_len, k_t, device)


def yarn(dim, theta, base_len, max_len, k_t, device):
    """(table, attention scale) of YaRN stretching `base_len` positions to `max_len`, the ramps raised to k_t:
    the scale is 1 + 0.1 ln(s) / sqrt(s) for the stretch s = max_len / base_len (1 when s is 1 or less)."""
    scale = max(max_len / base_len, 1.0)
    plain = base(dim, theta, device)
    table = _parts(plain / scale, ntk(dim, theta, scale ** (dim / (dim - 2)), device), plain, dim, theta, base_len, k_t, device)
    return table, (1.0 + 0.1 * math.log(scale) / math.sqrt(scale) if scale > 1.0 else 1.0)

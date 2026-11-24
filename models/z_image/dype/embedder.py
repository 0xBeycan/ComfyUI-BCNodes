# DyPE — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
#
# Derived from ComfyUI-DyPE (https://github.com/wildminder/ComfyUI-DyPE at 2664058): src/models/zimage.py
# (PosEmbedZImage: the per-axis tables, the attention scale, the output layout) and src/base.py (the
# dynamic PI branch, _calc_pi_components). Changed (NOTICE in this directory lists every change):
# - every table is float64 on a device that has it, the output float32, as core's rope (the original:
#   bfloat16 on CUDA);
# - `pi` scales the positions (DY-PI, the original's base-class branch); the original's Z-Image embedder sent
#   `pi` to its NTK branch;
# - the coordinate resize (_resize_rope_grid) is left out: core's Z-Image image positions step by 1 on both
#   image axes, where it returns them unchanged (Z-Image Omni's SigLIP tokens step by about 8: the original
#   would draw those toward steps of 1 as the run goes on, this port leaves them where core puts them);
# - the scale, the timestep and the settings are fixed per call (a new embedder for each model call, made by
#   the APPLY_MODEL wrapper in models/common/dype.py) in place of state set on a shared embedder.

import math

import torch

from ....libs.dype import rope


class DyPERope(torch.nn.Module):
    """Core's EmbedND (comfy/ldm/flux/layers.py) of Z-Image, with DyPE on the two image axes: called with the
    position ids (B, N, 3) of one token group (caption or image: the time / sequence axis, then rows and
    columns), returns the rotations (B, 1, N, sum(axes_dim) / 2, 2, 2) in core's layout.

    theta, axes_dim: the model's (core's embedder's); base_len: the RoPE length of an image axis the model
    holds (its axes_lens), YaRN's original length; scale: the image's longer side over base_resolution (more
    than 1); t: the call's sigma over sigma_max, in [0, 1]; start_sigma: below it the attention scale decays;
    dype_scale, dype_exponent: DyPE's lambda_s, lambda_t."""

    def __init__(self, theta, axes_dim, base_len, method, scale, t, start_sigma, dype_scale, dype_exponent):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.base_len = base_len
        self.method = method
        self.scale = scale
        self.t = t
        self.dype = method != "base"  # base: the static NTK table, no modulation over the run
        self.k_t = dype_scale * t ** dype_exponent
        # the attention scale Z-Image's Vision YaRN and YaRN apply: 1 + 0.05 ln(scale), decaying from start_sigma down
        t_norm = 1.0 if t > start_sigma else t / start_sigma
        self.mscale = 1.0 + 0.05 * math.log(scale) * t_norm ** dype_exponent

    def table(self, dim, device):
        """(frequencies, attention scale) of an image axis of `dim` dimensions; frequencies None for the
        plain ones (pi: the positions are scaled instead)."""
        if self.method == "vision_yarn":
            return rope.vision_yarn(dim, self.theta, self.base_len, self.scale, self.scale, self.k_t, device), self.mscale
        if self.method == "yarn":
            freqs, yarn_scale = rope.yarn(dim, self.theta, self.base_len, int(self.base_len * self.scale), self.k_t, device)
            return freqs, yarn_scale * self.mscale
        if self.method == "pi":
            return None, 1.0
        factor = self.scale ** (dim / (dim - 2))  # ntk, and base without the modulation
        if self.dype:
            factor = factor ** self.k_t
        return rope.ntk(dim, self.theta, max(1.0, factor), device), 1.0

    def axis(self, pos, i, device):
        """Core's rope of axis i: (..., N, dim / 2, 2, 2) float64 on `device`."""
        dim = self.axes_dim[i]
        pos = pos.to(dtype=torch.float32, device=device)
        freqs, mscale = (None, 1.0) if i == 0 else self.table(dim, device)
        if freqs is None:
            freqs = rope.base(dim, self.theta, device)
            if i > 0 and self.method == "pi":
                pos = pos / max(1.0, self.scale ** self.k_t)  # DY-PI: the positions interpolated, less as t falls
        angles = torch.einsum("...n,d->...nd", pos, freqs)
        cos, sin = torch.cos(angles), torch.sin(angles)
        if mscale != 1.0:
            cos, sin = cos * mscale, sin * mscale
        out = torch.stack([cos, -sin, sin, cos], dim=-1)
        return out.reshape(*out.shape[:-1], 2, 2)

    def forward(self, ids):
        import comfy.model_management

        device = ids.device if comfy.model_management.supports_fp64(ids.device) else torch.device("cpu")
        emb = torch.cat([self.axis(ids[..., i], i, device).to(dtype=torch.float32, device=ids.device)
                         for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)

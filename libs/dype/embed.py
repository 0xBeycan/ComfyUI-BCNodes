# DyPE — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
#
# Derived from ComfyUI-DyPE (https://github.com/wildminder/ComfyUI-DyPE at 2664058): src/base.py
# (DyPEBasePosEmbed: the NTK branch, the dynamic PI branch) and the rotation layout of src/models/zimage.py and
# src/models/qwen.py. Changed (NOTICE in this directory lists every change): every table float64 on a device that
# has it, the output float32, as core's rope; the scale, timestep and settings fixed per sampling step.

import torch

from . import rope


class DyPEEmbedND(torch.nn.Module):
    """Core's EmbedND (comfy/ldm/flux/layers.py) with DyPE on the two image axes: called with the position ids
    (B, N, 3) (the time / sequence axis, then rows and columns), returns the rotations (B, 1, N, sum(axes_dim) / 2,
    2, 2) in core's layout, float32 on the ids' device; the tables float64 on a device that has it (the CPU
    otherwise), as core's rope. Built for one sampling step.

    A model's embedder subclasses it with its vision_yarn and yarn tables (yarn_table); ntk, pi and base are the
    same for every model (the original's base class). theta, axes_dim: the model's (its embedder's); scale: the
    image's longer side over base_resolution (more than 1); t: the step's sigma over sigma_max, in [0, 1];
    dype_scale, dype_exponent: DyPE's lambda_s, lambda_t."""

    def __init__(self, theta, axes_dim, method, scale, t, dype_scale, dype_exponent):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.scale = scale
        self.t = t
        self.dype = method != "base"  # base: the static NTK table, no modulation over the run
        self.k_t = dype_scale * t ** dype_exponent

    def yarn_table(self, i, dim, device):
        """(frequencies, attention scale) of image axis i (1 or 2) for vision_yarn and yarn: the model's own."""
        raise NotImplementedError

    def table(self, i, dim, device):
        """(frequencies, attention scale) of image axis i of `dim` dimensions; frequencies None for the plain ones
        (pi: the positions are scaled instead)."""
        if self.method in ("vision_yarn", "yarn"):
            return self.yarn_table(i, dim, device)
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
        freqs, mscale = (None, 1.0) if i == 0 else self.table(i, dim, device)
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

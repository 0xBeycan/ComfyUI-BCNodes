# DyPE — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
#
# Derived from ComfyUI-DyPE (https://github.com/wildminder/ComfyUI-DyPE at 2664058): src/models/zimage.py
# (PosEmbedZImage: the per-axis tables, the attention scale). The layout, the time axis and the ntk / pi / base
# tables every model shares are libs/dype/embed.py. Changed (NOTICE in this directory lists every change):
# - every table is float64 on a device that has it, the output float32, as core's rope (the original:
#   bfloat16 on CUDA);
# - `pi` scales the positions (DY-PI, the original's base-class branch); the original's Z-Image embedder sent
#   `pi` to its NTK branch;
# - the coordinate resize (_resize_rope_grid) is left out: core's Z-Image image positions step by 1 on both
#   image axes, where it returns them unchanged (Z-Image Omni's SigLIP tokens step by about 8: the original
#   would draw those toward steps of 1 as the run goes on, this port leaves them where core puts them);
# - the scale, the timestep and the settings are fixed per call size and step (an embedder for each, made by the
#   wrapper in models/common/dype.py) in place of state set on a shared embedder.

from ....libs.dype import rope
from ....libs.dype.embed import DyPEEmbedND


class DyPERope(DyPEEmbedND):
    """Core's EmbedND of Z-Image with DyPE on the two image axes (libs/dype/embed.py), called with the position
    ids of one token group (caption or image).

    base_len: the RoPE length of an image axis the model holds (its axes_lens), YaRN's original length; scale:
    the image's longer side over base_resolution (more than 1), the same on both image axes; start_sigma: below
    it the attention scale decays."""

    def __init__(self, theta, axes_dim, base_len, method, scale, t, start_sigma, dype_scale, dype_exponent):
        super().__init__(theta, axes_dim, method, scale, t, dype_scale, dype_exponent)
        self.base_len = base_len
        # the attention scale Z-Image's Vision YaRN and YaRN apply: 1 + 0.05 ln(scale), decaying from start_sigma down
        self.mscale = rope.attention_scale(0.05, scale, t, start_sigma, dype_exponent)

    def yarn_table(self, i, dim, device):
        if self.method == "vision_yarn":
            return rope.vision_yarn(dim, self.theta, self.base_len, self.scale, self.scale, self.k_t, device), self.mscale
        freqs, yarn_scale = rope.yarn(dim, self.theta, self.base_len, int(self.base_len * self.scale), self.k_t, device)
        return freqs, yarn_scale * self.mscale

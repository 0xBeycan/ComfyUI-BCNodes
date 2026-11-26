# DyPE — Licensed under the Apache License, Version 2.0 (LICENSE in this directory).
#
# Derived from ComfyUI-DyPE (https://github.com/wildminder/ComfyUI-DyPE at 2664058): src/models/qwen.py
# (PosEmbedQwen) and the branches of src/base.py it runs (DyPEBasePosEmbed: _calc_vision_yarn_components,
# _calc_yarn_components with and without yarn_alt_scaling, _get_mscale). The layout, the time axis and the
# ntk / pi / base tables every model shares are libs/dype/embed.py. Changed (NOTICE in this directory lists every
# change):
# - the scale of each image axis is the latent's side over base_resolution, from the latent of the call; the original
#   read it from the span of every position id it is given, which on core's Qwen-Image holds the caption tokens
#   too (placed after the image's positions on all three axes), so its scale grew with the prompt's length;
# - every table is float64 on a device that has it, the output float32, as core's rope (the original:
#   bfloat16 on CUDA);
# - yarn takes its table at the positions core gives, where the original looked it up from 0 at the lowest
#   position (clamped to its length): the same rotation between any two tokens of an axis, RoPE being relative;
# - the scale, the timestep and the settings are fixed per call size and step (an embedder for each, made by the
#   wrapper in models/common/dype.py) in place of state set on a shared embedder.

import math

from ....libs.dype import rope
from ....libs.dype.embed import DyPEEmbedND


class DyPERope(DyPEEmbedND):
    """Core's EmbedND of Qwen-Image with DyPE on the two image axes (libs/dype/embed.py), called with the position
    ids of the caption and image tokens together; every token takes the same tables, as the original.

    base_len: the image axis length of base_resolution in tokens, YaRN's original length; scales: (rows, columns),
    each image axis's side over base_resolution (the larger, more than 1, is the scale); start_sigma: below it the
    Vision YaRN attention scale decays; yarn_alt_scaling: yarn stretches each image axis to its own length (an axis
    not over base_len keeps the plain table) in place of both to the longer one."""

    def __init__(self, theta, axes_dim, base_len, method, scales, t, start_sigma, dype_scale, dype_exponent,
                 yarn_alt_scaling):
        super().__init__(theta, axes_dim, method, max(scales), t, dype_scale, dype_exponent)
        self.base_len = base_len
        self.scales = tuple(max(1.0, s) for s in scales)
        self.yarn_alt_scaling = yarn_alt_scaling
        # the attention scale of the original's Vision YaRN: 1 + 0.1 ln(scale), decaying from start_sigma down
        self.mscale = rope.attention_scale(0.1, self.scale, t, start_sigma, dype_exponent)

    def yarn_table(self, i, dim, device):
        axis_scale = self.scales[i - 1]
        if self.method == "vision_yarn":  # each axis's own linear scale, the NTK one at the image's scale
            return rope.vision_yarn(dim, self.theta, self.base_len, axis_scale, self.scale, self.k_t, device), self.mscale
        if not self.yarn_alt_scaling:  # both axes stretched to the longer one, YaRN's own attention scale
            return rope.yarn(dim, self.theta, self.base_len, self.base_len * self.scale, self.k_t, device)
        if axis_scale <= 1.0:
            return None, 1.0
        freqs, _ = rope.yarn(dim, self.theta, self.base_len, self.base_len * axis_scale, self.k_t, device)
        return freqs, 1.0 + 0.1 * math.log(axis_scale)  # the original's "aggressive" attention scale

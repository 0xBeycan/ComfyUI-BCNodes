"""Flow-matching shift: the resolution-dependent shift of diffusers' FlowMatchEulerDiscreteScheduler, as
the official Qwen-Image pipeline drives it, and the scheduler's stretch to a terminal sigma.

The pipeline (QwenImagePipeline.__call__) counts the image's tokens (the latent in 2x2 patches) and
puts mu on the line through (base_seq_len, base_shift) and (max_seq_len, max_shift), not clamped:
above max_seq_len it goes on rising (calculate_shift). The scheduler's exponential time shift with mu,
e^mu / (e^mu + 1/t - 1), is the flow shift S t / (1 + (S - 1) t) with S = e^mu: ComfyUI's
time_snr_shift, the formula of ModelSamplingAuraFlow.

The stretch (stretch_shift_to_terminal) moves 1 - sigma by one factor so the schedule's last sigma
lands on the terminal value; the sigma 0 the scheduler appends after it is not stretched.
"""

import math

import torch

PATCH = 2  # the latent's patch side (the pipeline's _pack_latents, the model's patch size)


def image_seq_len(height: int, width: int) -> int:
    """Tokens of a latent `height` x `width`: 2x2 patches, an odd side padded as the model pads it.
    The pipeline's own latents always have even sides."""
    return math.ceil(height / PATCH) * math.ceil(width / PATCH)


def mu(seq_len: int, base_seq_len: int, max_seq_len: int, base_shift: float, max_shift: float) -> float:
    """The pipeline's calculate_shift: base_shift at base_seq_len tokens, max_shift at max_seq_len, the
    same line beyond both."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return seq_len * m + b


def stretch_to_terminal(sigmas: torch.Tensor, terminal: float) -> torch.Tensor:
    """A new schedule: `sigmas` (ending in 0, its last nonzero sigma below 1) with 1 - sigma scaled so
    the last nonzero sigma becomes `terminal`; the final 0 stays."""
    one_minus = 1 - sigmas[:-1]
    out = sigmas.clone()
    out[:-1] = 1 - one_minus / (one_minus[-1] / (1 - terminal))
    return out

"""Flow-matching shift: the resolution-dependent shift of diffusers' FlowMatchEulerDiscreteScheduler, as
the official Qwen-Image pipeline drives it, and the scheduler's stretch to a terminal sigma.

The pipeline (QwenImagePipeline.__call__) counts the image's tokens (the latent in 2x2 patches) and
puts mu on the line through (base_seq_len, base_shift) and (max_seq_len, max_shift), not clamped:
above max_seq_len it goes on rising (calculate_shift). The scheduler's exponential time shift with mu,
e^mu / (e^mu + 1/t - 1), is the flow shift S t / (1 + (S - 1) t) with S = e^mu: ComfyUI's
time_snr_shift, the formula of ModelSamplingAuraFlow.

The stretch (stretch_shift_to_terminal) moves 1 - sigma by one factor so the schedule's last sigma
lands on the terminal value; the sigma 0 the scheduler appends after it is not stretched.

The shift nodes (Qwen Image Shift, Dual Sampling Shift) share the model_sampling ModelSamplingAuraFlow puts on
a model (flow_sampling) and, in their OUTER_SAMPLE wrappers, the refine rule (refine_pass), the check that the
run's model_sampling is still the node's (check_model_sampling) and the run with model_sampling held at another
shift (run_at_shift).
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


def flow_sampling(model, shift: float):
    """The model_sampling ModelSamplingAuraFlow makes (core's ModelSamplingSD3.patch, flow, multiplier
    1.0) at `shift`."""
    from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow

    class ModelSamplingAdvanced(ModelSamplingDiscreteFlow, CONST):
        pass

    original = model.get_model_object("model_sampling")
    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(shift=shift, multiplier=1.0)
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    return sampling


def refine_pass(noise: torch.Tensor, sigmas: torch.Tensor, full_noise: float) -> bool:
    """A refine pass: a sampler run on new noise whose schedule starts below `full_noise` (denoise below 1,
    or a later start step). Neither the denoise nor add_noise reaches an OUTER_SAMPLE wrapper; core's empty
    noise (KSamplerAdvanced's add_noise off, SamplerCustomAdvanced's DisableNoise) is all zeros and adds
    nothing to the latent, so nonzero noise is exactly "new noise added", and the later part of a split run
    is no refine pass."""
    return float(sigmas[0]) < full_noise and bool(noise.any())


def check_model_sampling(executor, sampling, prefix: str) -> None:
    """Stops an OUTER_SAMPLE wrapper's run when the run's model_sampling is no longer `sampling`, the object
    its node put on the model: a node after it replaced it."""
    if executor.class_obj.model_patcher.get_model_object("model_sampling") is not sampling:
        raise ValueError(f"{prefix} a node after it replaced the model's model_sampling (ModelSamplingAuraFlow or "
                         "another shift node), so its shift would be applied twice. Keep one of the two.")


def run_at_shift(sampling, shift: float, after: float, multiplier: float, executor, args, kwargs):
    """Runs an OUTER_SAMPLE executor with `sampling` (the run's model_sampling) at `shift` (percent_to_sigma:
    ControlNet and conditioning step ranges follow it), then sets it to `after`, also when the run fails."""
    sampling.set_parameters(shift=shift, multiplier=multiplier)
    try:
        return executor(*args, **kwargs)
    finally:
        sampling.set_parameters(shift=after, multiplier=multiplier)

"""BC_QwenImageShift (Qwen Image Shift)

MODEL -> a clone whose flow shift follows the official Qwen-Image-2512 scheduler, in place of
ModelSamplingAuraFlow (libs/flow_shift.py has the formulas). The clone's model_sampling is the object
ModelSamplingAuraFlow puts there (libs/flow_shift.flow_sampling: core's ModelSamplingDiscreteFlow + CONST,
multiplier 1.0, the model's noise_scale kept).

fixed: model_sampling at `shift`, so every scheduler gives what ModelSamplingAuraFlow gives at that shift.
dynamic: the shift depends on the latent being sampled, and a sampler builds its sigmas before any patch
sees the latent (KSampler.__init__ asks the scheduler; nothing runs between it and the sampling). The
first point that sees both is the OUTER_SAMPLE wrapper (CFGGuider.sample, ComfyUI's patcher wrappers), so
model_sampling stays at shift 1 outside a run, the scheduler gives its plain timestep grid t, and the
wrapper turns it into time_snr_shift(e^mu x multiplier, t), mu from the run's latent size; for the run
model_sampling is at that shift too (percent_to_sigma: ControlNet and conditioning ranges), then back at
1. For simple, beta and ddim_uniform (schedulers that read model_sampling's table) the result is exactly
the scheduler at that shift; normal and sgm_uniform take their grid's end from the shift-1 table;
karras, exponential, kl_optimal and linear_quadratic give their shift-1 schedule shifted.
shift_terminal: the wrapper stretches the run's schedule so its last nonzero sigma lands on `terminal`.

A refine pass (dynamic mode): a run whose schedule starts below sigma 1 (denoise below 1, or a later start step)
on new noise. Neither the denoise nor add_noise reaches the wrapper; core's empty noise (KSamplerAdvanced's
add_noise off, SamplerCustomAdvanced's DisableNoise) is all zeros, and zero noise adds nothing to the latent, so
nonzero noise is exactly "new noise added". The later part of a split run (no noise added) is no refine pass, so
one generation keeps one shift. A refine pass above refine_threshold tokens takes mu at
refine_threshold + refine_rate x (tokens - refine_threshold) tokens: mu is linear in the tokens, so that is
mu(threshold) + refine_rate x (mu(tokens) - mu(threshold)), and rate 1 gives the latent's own mu, bit for bit.

The wrapper runs before prepare_sampling, process_conds and sample_sigmas, so everything after it sees
the new schedule. Core chains wrappers: the model's other wrappers stay; a Qwen Image Shift already on
the model is replaced (its model_sampling would be overwritten anyway).
"""

import logging
import math

WRAPPER_KEY = "bcnodes_qwen_image_shift"
PREFIX = "[BCNodes] Qwen Image Shift:"
REFINE_THRESHOLD = 9600  # tokens: a 1024 x 1536 image upscaled 1.25x
REFINE_RATE = 0.17       # 2x of 1024 x 1536 (24576 tokens): shift 3.0


class Shift:
    """The OUTER_SAMPLE wrapper: (executor, noise, latent_image, sampler, sigmas, denoise_mask, callback,
    disable_pbar, seed, latent_shapes=...), as CFGGuider.outer_sample; `dynamic` holds (multiplier,
    base_seq_len, max_seq_len, base_shift, max_shift, refine_threshold, refine_rate) or None (fixed),
    `terminal` the terminal sigma or None (off)."""

    def __init__(self, sampling, dynamic, terminal):
        self.sampling = sampling
        self.dynamic = dynamic
        self.terminal = terminal

    def __call__(self, executor, *args, **kwargs):
        from comfy.model_sampling import time_snr_shift

        from ..libs.flow_shift import (check_model_sampling, image_seq_len, mu, refine_pass, run_at_shift,
                                       stretch_to_terminal)

        check_model_sampling(executor, self.sampling, PREFIX)
        args = list(args)
        sigmas = args[3]
        shift = None
        if self.dynamic is not None:
            multiplier, base_seq_len, max_seq_len, base_shift, max_shift, threshold, rate = self.dynamic
            height, width = kwargs["latent_shapes"][0][-2:]
            tokens = image_seq_len(height, width)
            refine = refine_pass(args[0], sigmas, 1.0)  # model_sampling at shift 1: full noise is sigma 1
            seq = threshold + rate * (tokens - threshold) if refine and tokens > threshold else tokens
            m = mu(seq, base_seq_len, max_seq_len, base_shift, max_shift)
            shift = math.exp(m) * multiplier
            if not refine:
                state = f"refine: no, tokens {tokens}"
            elif tokens > threshold:
                state = f"refine: yes, tokens {tokens}, threshold {threshold}, rate {rate:g}"
            else:
                state = f"refine: yes, tokens {tokens} <= threshold {threshold}"
            logging.info("%s latent %dx%d, %s, mu %.4f, shift %.4f", PREFIX, width, height, state, m, shift)
            sigmas = time_snr_shift(shift, sigmas)
        if self.terminal is not None:
            if float(sigmas[-1]) != 0.0:
                raise ValueError(f"{PREFIX} shift_terminal needs a run that ends at sigma 0; this one stops at "
                                 f"{float(sigmas[-1]):.4f} (KSamplerAdvanced returning with leftover noise, or the first "
                                 "part of split sigmas). Turn shift_terminal off for it.")
            if float(sigmas[-2]) >= 1.0:
                raise ValueError(f"{PREFIX} shift_terminal has nothing to stretch in a run whose only sigma is 1 (one "
                                 "step from full noise). Use more steps or turn shift_terminal off.")
            sigmas = stretch_to_terminal(sigmas, self.terminal)
        args[3] = sigmas
        if shift is None:
            return executor(*args, **kwargs)
        return run_at_shift(self.sampling, shift, 1.0, 1.0, executor, args, kwargs)


class QwenImageShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Qwen-Image model. The input model is not changed; the output is a "
                                    "patched clone. Use it in place of ModelSamplingAuraFlow."}),
                "mode": (["dynamic", "fixed"], {"default": "dynamic", "tooltip":
                         "dynamic: the official scheduler's shift, from the size of the latent each sampler run gets "
                         "(the same model gives a 1.25x upscaled second pass a larger shift); exact for the simple, "
                         "beta and ddim_uniform schedulers, see the docs for the others. fixed: `shift` for every "
                         "run, the same sigmas as ModelSamplingAuraFlow at that shift with every scheduler."}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip":
                          "The shift in fixed mode (not used in dynamic mode). Qwen-Image-Lightning was distilled "
                          "with shift 3: fixed, 3, shift_terminal off."}),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01, "tooltip":
                               "Dynamic mode: the shift is e^mu x multiplier. 1: the official shift. Not used in "
                               "fixed mode."}),
                "shift_terminal": ("BOOLEAN", {"default": True, "tooltip":
                                   "Stretch each run's schedule so its last sigma before 0 is `terminal`, as the "
                                   "official scheduler does (on). Off for Lightning. Needs a run that ends at sigma 0."}),
                "terminal": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.999, "step": 0.001, "tooltip":
                             "The last sigma before 0 when shift_terminal is on (official: 0.02)."}),
                "base_seq_len": ("INT", {"default": 256, "min": 1, "max": 1000000, "tooltip":
                                 "Dynamic mode: the token count (latent in 2x2 patches) that gets base_shift "
                                 "(official: 256)."}),
                "max_seq_len": ("INT", {"default": 8192, "min": 1, "max": 1000000, "tooltip":
                                "Dynamic mode: the token count that gets max_shift (official: 8192); larger images "
                                "continue on the same line."}),
                "base_shift": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip":
                               "Dynamic mode: mu at base_seq_len tokens (official: 0.5)."}),
                "max_shift": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip":
                              "Dynamic mode: mu at max_seq_len tokens (official: 0.9)."}),
            },
            # Added after the node's release, optional so an API-format prompt saved without them still validates.
            "optional": {
                "refine_threshold": ("INT", {"default": REFINE_THRESHOLD, "min": 0, "max": 1000000, "tooltip":
                                     "Dynamic mode: a refine pass (denoise below 1 on new noise: a hires or "
                                     "upscaled second pass) on a latent of more tokens than this gets a lower "
                                     "shift (refine_rate). At or below it, and in a generation from noise, the "
                                     "latent's own shift. 9600: 1024 x 1536 upscaled 1.25x."}),
                "refine_rate": ("FLOAT", {"default": REFINE_RATE, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip":
                                "Dynamic mode, a refine pass above refine_threshold tokens: mu moves this part of "
                                "the way from refine_threshold's mu to the latent's own. 1: the latent's own "
                                "shift; 0: the threshold's shift; below 0: lower than the threshold's. 0.17: a 2x "
                                "pass of 1024 x 1536 gets shift 3.0 (its own: 5.62)."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "qwen image shift", "shift", "model sampling", "aura flow", "sigma"]
    DESCRIPTION = ("The flow shift of the official Qwen-Image-2512 scheduler, in place of ModelSamplingAuraFlow, for "
                   "the stock KSampler and schedulers. dynamic: the shift follows each sampler run's latent size "
                   "(e^mu, mu on the line base_shift @ base_seq_len -> max_shift @ max_seq_len tokens), so one model "
                   "gives each pass its own shift; a refine pass above refine_threshold tokens moves only "
                   "refine_rate of the way from the threshold's mu to its own. fixed: `shift` for every run, as "
                   "ModelSamplingAuraFlow. shift_terminal stretches each run's schedule to end at `terminal` before 0.")

    def patch(self, model, mode, shift, multiplier, shift_terminal, terminal, base_seq_len, max_seq_len,
              base_shift, max_shift, refine_threshold=REFINE_THRESHOLD, refine_rate=REFINE_RATE):
        # as the pack's other MODEL nodes: nothing to patch
        if model is None:
            return (model,)
        dynamic = None
        if mode == "dynamic":
            if max_seq_len == base_seq_len:
                raise ValueError(f"{PREFIX} max_seq_len ({max_seq_len}) must differ from base_seq_len "
                                 f"({base_seq_len}): the two points of the shift line.")
            dynamic = (multiplier, base_seq_len, max_seq_len, base_shift, max_shift, refine_threshold, refine_rate)
        from comfy.patcher_extension import WrappersMP

        from ..libs.flow_shift import flow_sampling

        patched = model.clone()
        patched.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
        sampling = flow_sampling(model, 1.0 if dynamic else shift)
        patched.add_object_patch("model_sampling", sampling)
        if dynamic or shift_terminal:
            patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
                                         Shift(sampling, dynamic, terminal if shift_terminal else None))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "BC_QwenImageShift": QwenImageShift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_QwenImageShift": "Qwen Image Shift",
}

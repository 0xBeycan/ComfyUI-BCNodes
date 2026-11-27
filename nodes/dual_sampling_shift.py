"""BC_DualSamplingShift (Dual Sampling Shift)

MODEL -> a clone with two flow shifts, in place of ModelSamplingAuraFlow / ModelSamplingSD3: shift_1 for a
sampler run from full noise, shift_2 for a refine pass (a run on new noise that starts below full noise:
denoise below 1, a later start step).

The clone's model_sampling is a new object of the model's own model_sampling class (core's
ModelSamplingDiscreteFlow with its prediction mixin) at shift_1, with the model's multiplier and noise_scale,
built as ModelSamplingSD3.patch builds it: a run from full noise gets what ModelSamplingAuraFlow (multiplier 1)
or ModelSamplingSD3 (multiplier 1000) gives at shift_1, with every scheduler, and the wrapper hands it on
unchanged. A sampler builds its sigmas before any patch sees the run, so a refine pass's sigmas are shift_1's
too. The OUTER_SAMPLE wrapper (CFGGuider.sample, ComfyUI's patcher wrappers) maps them to shift_2: t from the
inverse of the shift_1 flow shift, then time_snr_shift(shift_2, t); for the run model_sampling is at shift_2
(percent_to_sigma: ControlNet and conditioning ranges), then back at shift_1, also when the run fails. For the
schedulers that read model_sampling's table (simple, beta, ddim_uniform) that is the scheduler at shift_2, to
float32 rounding (the map goes through t).

The refine rule is Qwen Image Shift's (libs/flow_shift.refine_pass): the later part of a split run (zero noise)
stays on shift_1. Full noise: model_sampling's top sigma at shift_1 is 1 only to float32 rounding, and the
schedulers that compute their first sigma from it (normal, sgm_uniform, karras) land slightly on either side,
so a run starts below full noise when its first sigma is below the midpoint of model_sampling's two highest
sigmas (about half a timestep below the top). ddim_uniform never starts at the top sigma, so its runs on new
noise are refine passes.

Flow models with one shift only: any other model_sampling (ModelSamplingFlux, which Qwen-Image and Flux load
with; EPS and v-prediction models; an audio-video model's ModelSamplingAV, whose audio has a second shift) is
an error, and so is a Qwen Image Shift wrapper on the input model (both nodes set the shift). A node after it
that replaces model_sampling stops the run. Core chains wrappers: the model's other wrappers stay; a Dual
Sampling Shift already on the model is replaced.
"""

WRAPPER_KEY = "bcnodes_dual_sampling_shift"
QWEN_IMAGE_SHIFT_KEY = "bcnodes_qwen_image_shift"  # nodes/qwen_image_shift.py WRAPPER_KEY
PREFIX = "[BCNodes] Dual Sampling Shift:"


class Refine:
    """The OUTER_SAMPLE wrapper: (executor, noise, latent_image, sampler, sigmas, denoise_mask, callback,
    disable_pbar, seed, latent_shapes=...), as CFGGuider.outer_sample. `full_noise`: a run whose first sigma is
    below it starts below full noise."""

    def __init__(self, sampling, shift_1, shift_2, multiplier, full_noise):
        self.sampling = sampling
        self.shift_1 = shift_1
        self.shift_2 = shift_2
        self.multiplier = multiplier
        self.full_noise = full_noise

    def __call__(self, executor, *args, **kwargs):
        from comfy.model_sampling import time_snr_shift

        from ..libs.flow_shift import check_model_sampling, refine_pass, run_at_shift

        check_model_sampling(executor, self.sampling, PREFIX)
        sigmas = args[3]
        if not refine_pass(args[0], sigmas, self.full_noise):
            return executor(*args, **kwargs)
        # t of each sigma: the inverse of sigma = shift_1 t / (1 + (shift_1 - 1) t)
        t = sigmas / (self.shift_1 - (self.shift_1 - 1) * sigmas)
        args = list(args)
        args[3] = time_snr_shift(self.shift_2, t)
        return run_at_shift(self.sampling, self.shift_2, self.shift_1, self.multiplier, executor, args, kwargs)


class DualSamplingShift:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A flow model (Z-Image, Wan, SD3, or any model after "
                                    "ModelSamplingAuraFlow). The input model is not changed; the output is a patched "
                                    "clone. Use it in place of ModelSamplingAuraFlow / ModelSamplingSD3."}),
                "shift_1": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip":
                            "The shift of a sampler run from full noise (denoise 1): the same sigmas as "
                            "ModelSamplingAuraFlow / ModelSamplingSD3 at this shift with every scheduler but "
                            "ddim_uniform (see shift_2). The later part of a split run (add_noise off) keeps it "
                            "too."}),
                "shift_2": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip":
                            "The shift of a refine pass: a run on new noise that starts below full noise (denoise "
                            "below 1, a later start step): a hires or upscaled second pass, Dual Sampler's second "
                            "pass, a detailer's crop. Its sigmas are the scheduler's at this shift (exact for "
                            "simple, beta and ddim_uniform). ddim_uniform never starts at full noise: its runs on "
                            "new noise always take this shift."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "dual sampling shift", "shift", "model sampling", "aura flow", "sigma", "refine"]
    DESCRIPTION = ("Two flow shifts on one model, in place of ModelSamplingAuraFlow / ModelSamplingSD3: shift_1 for a "
                   "sampler run from full noise (the same sigmas as ModelSamplingAuraFlow at that shift), shift_2 for "
                   "a refine pass (a run on new noise that starts below full noise: denoise below 1, a hires or "
                   "upscaled second pass, Dual Sampler's second pass, a detailer's crop). The later part of a split "
                   "run (add_noise off) keeps shift_1. Flow models with core's ModelSamplingDiscreteFlow only; not "
                   "together with Qwen Image Shift.")

    def patch(self, model, shift_1, shift_2):
        # as the pack's other MODEL nodes: nothing to patch
        if model is None:
            return (model,)
        from comfy.model_sampling import ModelSamplingAV, ModelSamplingDiscreteFlow
        from comfy.patcher_extension import WrappersMP

        original = model.get_model_object("model_sampling")
        if not isinstance(original, ModelSamplingDiscreteFlow) or isinstance(original, ModelSamplingAV):
            names = " + ".join(base.__name__ for base in type(original).__bases__)
            raise ValueError(f"{PREFIX} needs a flow model with one shift, whose model_sampling is core's "
                             "ModelSamplingDiscreteFlow (Z-Image, Wan, SD3, or any model after ModelSamplingAuraFlow "
                             f"/ ModelSamplingSD3); this model's is {names}. Qwen-Image and Flux load with "
                             "ModelSamplingFlux: put ModelSamplingAuraFlow before this node (its shift is replaced "
                             "by shift_1). Any other model (EPS or v-prediction, an audio-video model's "
                             "ModelSamplingAV with a second shift for its audio): remove the node.")
        if model.get_wrappers(WrappersMP.OUTER_SAMPLE, QWEN_IMAGE_SHIFT_KEY):
            raise ValueError(f"{PREFIX} the model has a Qwen Image Shift: both nodes set the model's shift. Keep one "
                             "of the two: Qwen Image Shift (the shift from each run's latent size) or Dual Sampling "
                             "Shift (shift_1 from full noise, shift_2 for refine passes).")
        patched = model.clone()
        patched.remove_wrappers_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY)
        sampling = type(original)(model.model.model_config)
        sampling.set_parameters(shift=shift_1, multiplier=original.multiplier)
        sampling.set_noise_scale(original.noise_scale)
        patched.add_object_patch("model_sampling", sampling)
        full_noise = (float(sampling.sigmas[-1]) + float(sampling.sigmas[-2])) / 2
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
                                     Refine(sampling, shift_1, shift_2, original.multiplier, full_noise))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "BC_DualSamplingShift": DualSamplingShift,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DualSamplingShift": "Dual Sampling Shift",
}

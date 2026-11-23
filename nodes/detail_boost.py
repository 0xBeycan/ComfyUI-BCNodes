"""BC_DetailBoost (Detail Boost)

MODEL -> a clone of it that is told a lower noise level than the sampler's in a step range, so it removes
less noise there and keeps more fine detail (libs/detail_boost.py has the formulas). The clone carries a
PREDICT_NOISE wrapper (ComfyUI's patcher wrappers): every model call of a sampling run passes through it
with the sigma the sampler calls at and the run's schedule (transformer_options["sample_sigmas"]), and
the wrapper hands the call on with the lowered sigma. Below it everything sees that sigma: the cond
batches, a model_function_wrapper the model already has, apply_model (input scaling and the denoised
output), the CFG functions. The sampler steps with its own sigmas. Core chains wrappers, so the model's
other patches stay; a weight or eta of 0 leaves the clone unpatched.
"""


class Boost:
    """The PREDICT_NOISE wrapper: (executor, x, timestep, model_options, seed), as CFGGuider.predict_noise."""

    def __init__(self, settings):
        self.settings = settings

    def __call__(self, executor, x, timestep, model_options, seed):
        from comfy.model_sampling import CONST

        from ..libs.detail_boost import model_sigma

        sampling = executor.class_obj.inner_model.model_sampling  # as patched for this run (shift nodes included)
        flow = isinstance(sampling, CONST)
        sigma = float(timestep.max())
        schedule = model_options["transformer_options"]["sample_sigmas"].tolist()
        told = model_sigma(self.settings, sigma, schedule, flow, float(sampling.sigma_max))
        if told != sigma:
            timestep = timestep * (told / sigma)
        return executor(x, timestep, model_options, seed)


class DetailBoost:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs.detail_boost import MODES

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to boost. The input model is not changed; the output is a patched clone."}),
                "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip":
                           "How far the sigma the model is told moves toward the lowered one: sigma + weight * (lowered - "
                           "sigma). 1: the mode's full effect; positive values add fine detail, negative values soften. "
                           "0: the model is not patched (the output is bit-identical to the input model's)."}),
                "mode": (list(MODES), {"default": "hard", "tooltip":
                         "How much noise the model is told is gone, with s the call's sigma and p the schedule sigma "
                         "before it. hard: eta at every step (flow models: noise/signal x sqrt(1 - eta^2)). soft: "
                         "eta * (1 + s/p), about twice hard. softer, eps, exp: grow with the step's size (p - s). "
                         "soft-linear: above 1 on a falling schedule, so capped: hard's result below eta 1. sinusoidal: "
                         "eta * sin(pi s)^2, strongest at s 0.5 (middle steps). lorentzian: eta of s / sqrt(s^2 + 1), "
                         "weaker at high sigma. hard_var: eta of sqrt(p - s), only in the late steps where s <= "
                         "(sqrt(1 + 4p) - 1) / 2. er4: eta is an exponent of its noise scaler. vpsde: a flow-model "
                         "formula, s - eta/4 * p(1 - p)(p - s)."}),
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip":
                        "The mode's strength (no noise is added). 0: the model is not patched. hard on a flow model: "
                        "0.5 tells the model noise/signal x 0.866; from 1 on the cut is capped (x 0.014)."}),
                "start_step": ("INT", {"default": 3, "min": 0, "max": 10000, "tooltip":
                               "First step boosted, counted from 0 in each sampler run (with denoise < 1, from the "
                               "run's first step). The run's very first model call is never changed: it has no "
                               "schedule sigma before it."}),
                "end_step": ("INT", {"default": 10, "min": -1, "max": 10000, "tooltip":
                             "The step the boost stops at (not boosted itself); -1: to the end of the run."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "detail boost", "detail", "micro texture", "sigma"]
    DESCRIPTION = ("Tells the model a lower noise level than the sampler's in the steps [start_step, end_step) of each "
                   "sampling run, so it removes less noise there and keeps more fine detail (skin pores, hair, stone); "
                   "the sampler keeps its own sigmas and adds no noise. Works with KSampler and KSamplerAdvanced (no "
                   "custom sampler); the model's other patches stay. weight 0 or eta 0: the model is not patched.")

    def patch(self, model, weight, mode, eta, start_step, end_step):
        # as the pack's other MODEL nodes: nothing to patch
        if model is None:
            return (model,)
        if end_step != -1 and end_step <= start_step:
            raise ValueError(f"Detail Boost: end_step ({end_step}) must be greater than start_step ({start_step}), "
                             "or -1 to boost to the end of the run.")
        patched = model.clone()
        if weight == 0 or eta == 0:
            return (patched,)
        from comfy.patcher_extension import WrappersMP

        from ..libs.detail_boost import Settings

        settings = Settings(weight=weight, mode=mode, eta=eta, start_step=start_step, end_step=end_step)
        patched.add_wrapper_with_key(WrappersMP.PREDICT_NOISE, "bcnodes_detail_boost", Boost(settings))
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "BC_DetailBoost": DetailBoost,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_DetailBoost": "Detail Boost",
}

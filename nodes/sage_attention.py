"""BC_PatchSageAttention (Patch Sage Attention)

MODEL -> a clone of it whose attention runs one SageAttention kernel, instead of ComfyUI's global
--use-sage-attention: every other model and workflow keeps ComfyUI's attention. The clone carries a
libs/sage_attention.SageOverride as its transformer_options' optimized_attention_override (ComfyUI
0.3.60+), so the patch lives and dies with the clone; nothing global is touched. Two patcher
callbacks frame each sampling run: ON_PRE_RUN logs the kernel, the GPU and the package version,
ON_CLEANUP (run in the sampler's finally) logs how many attention calls ran the kernel and how many
went to the original attention, by reason.
"""

import logging


class PatchSageAttention:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs.sage_attention import MODES, QUANTS

        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model whose attention runs SageAttention. The input model is not changed; "
                                               "the output is a patched clone."}),
                "mode": (list(MODES), {"default": "auto", "tooltip":
                         "disabled: the clone, unpatched (for A/B in the same workflow). auto: the most accurate kernel, "
                         "INT8 QK with fp16 PV (CUDA), on Ampere, Ada, Hopper and Blackwell; never triton, fp8 PV or "
                         "sageattn3. The other modes name one kernel; one this GPU or the installed package cannot run "
                         "leaves the model on ComfyUI's attention, with the reason in the console."}),
                "qk_quant": (list(QUANTS), {"default": "auto", "tooltip":
                             "How Q and K are quantized to INT8, for auto, fp16_cuda, fp8_cuda and fp8_cuda++ (the Triton "
                             "kernel and sageattn3 ignore it). per_thread: one scale per thread's tokens, the most accurate; "
                             "slower on short sequences (~4k tokens). per_warp: one scale per warp's tokens. auto: per_thread "
                             "when its test call passes on this GPU, else per_warp. A per_thread call on tensors past 2^31 "
                             "elements of offset (long video, fused QKV) runs per_warp: per_thread's indexing wraps there."}),
                "nan_guard": ("BOOLEAN", {"default": True, "tooltip":
                              "Check every kernel output for NaN / inf and run that call again with the original attention "
                              "when it has any. Costs one GPU sync per attention call."}),
                "log_level": (["once", "verbose"], {"default": "once", "tooltip":
                              "once: each reason a call goes to the original attention is logged once per run (the run "
                              "summary counts them all). verbose: every such call is logged."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "BCNodes/model"
    SEARCH_ALIASES = ["BCNodes", "sage attention", "sageattention", "sageattn", "attention patch"]
    DESCRIPTION = ("Runs this model's attention with a SageAttention kernel, for this model only: no global "
                   "--use-sage-attention, every other model keeps ComfyUI's attention. qk_quant picks how Q and K are "
                   "quantized (per_thread or per_warp). A call the kernel cannot take (a mask, fp32, a head size over "
                   "128, a non-CUDA tensor, core's low_precision_attention=False, offsets past the kernel's indexing), a "
                   "kernel error and, with nan_guard, a NaN / inf output run the original attention instead. The console "
                   "names the kernel at each run's start and counts the calls per path at its end.")

    def patch(self, model, mode, qk_quant, nan_guard, log_level):
        # as the pack's other MODEL nodes: nothing to patch
        if model is None:
            return (model,)
        patched = model.clone()
        if mode == "disabled":
            return (patched,)
        import comfy.ldm.modules.attention as attention

        if not hasattr(attention, "wrap_attn"):
            raise RuntimeError("Patch Sage Attention needs ComfyUI 0.3.60 or newer (the per-model attention override); "
                               "update ComfyUI.")
        from comfy.patcher_extension import CallbacksMP

        from ..libs.sage_attention import PREFIX, SageOverride, resolve

        active = resolve(mode, qk_quant, patched.load_device)
        if isinstance(active, str):
            logging.warning("%s %s; the model keeps ComfyUI's attention.", PREFIX, active)
            return (patched,)
        options = patched.model_options["transformer_options"]
        override = SageOverride(active, options.get("optimized_attention_override"), nan_guard, log_level == "verbose")
        options["optimized_attention_override"] = override
        patched.add_callback(CallbacksMP.ON_PRE_RUN, lambda _patcher: override.start_run())
        patched.add_callback(CallbacksMP.ON_CLEANUP, lambda _patcher: override.end_run())
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "BC_PatchSageAttention": PatchSageAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PatchSageAttention": "Patch Sage Attention",
}

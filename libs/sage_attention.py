"""SageAttention for one model, as ComfyUI's attention override (Patch Sage Attention).

ComfyUI's attention entry points (`wrap_attn` in comfy/ldm/modules/attention.py: Qwen-Image, Wan,
LTX, Flux and the rest) call `transformer_options["optimized_attention_override"](func, q, k, v,
heads, mask, attn_precision, skip_reshape, skip_output_reshape, **kwargs)` when a model carries one;
`func` is the attention the call would have run. `SageOverride` runs one SageAttention kernel and
gives every call it cannot take, every call whose kernel raises and (nan_guard on) every call whose
output is not finite back to the original attention: the override that was on the model before
it, else `func`. Layouts follow ComfyUI's own attention: `skip_reshape` hands (B, H, N, D) in (HND),
otherwise (B, N, H*D) (NHD); `skip_output_reshape` asks for (B, H, N, D) back, otherwise (B, N, H*D).

Kernel facts come from the official code, thu-ml/SageAttention 2.2.0 (`sageattention`) and its
`sageattn3` 1.0.0: the compute capabilities each kernel is built for (setup.py), the arguments of
each function and the official dispatch's choices (`sageattn`: per_warp quantization on sm_120 /
sm_121). Only torch and the standard library at module level; `sageattention` is imported by
`resolve`.
"""

import dataclasses
import importlib
import logging
from typing import Callable, Optional

import torch

PREFIX = "[BCNodes] Patch Sage Attention:"
AUTO = "sageattn_qk_int8_pv_fp16_cuda"


@dataclasses.dataclass(frozen=True)
class Kernel:
    """One SageAttention function the node offers, as the official code defines it."""
    mode: str                    # the node's combo value
    package: str                 # distribution and import name
    function: str                # the function in `package`
    capabilities: tuple          # compute capabilities the official build compiles it for
    kwargs: tuple = ()           # fixed keyword arguments, (name, value) pairs
    per_arch_quant: bool = False  # takes qk_quant_gran: per_warp on sm_12x (the official dispatch), else per_thread
    hnd_only: bool = False       # takes (B, H, N, D) only, no tensor_layout argument
    max_head_dim: int = 128      # larger head dims raise in the kernel
    takes_scale: bool = True     # takes sm_scale; sageattn3 fixes the scale at 1 / sqrt(head_dim)

    def arguments(self, capability):
        kwargs = dict(self.kwargs)
        if self.per_arch_quant:
            kwargs["qk_quant_gran"] = "per_warp" if capability[0] == 12 else "per_thread"
        return kwargs


# setup.py's SUPPORTED_ARCHS; the _qattn_sm80 module (the fp16 CUDA kernel) is built for each of them.
_SM80_BUILD = ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1))
# The _qattn_sm89 module (the fp8 CUDA kernels) is built for 8.9 and newer.
_SM89_BUILD = ((8, 9), (9, 0), (10, 0), (12, 0), (12, 1))
# The official dispatch: "triton kernel is currently not usable on sm120" (and sm121).
_TRITON = ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0))
# sageattn3's setup.py: sm_100a, sm_120a, sm_121a.
_SM100_BUILD = ((10, 0), (12, 0), (12, 1))

KERNELS = {k.mode: k for k in (
    Kernel("sageattn_qk_int8_pv_fp16_cuda", "sageattention", "sageattn_qk_int8_pv_fp16_cuda", _SM80_BUILD,
           kwargs=(("pv_accum_dtype", "fp32"), ("smooth_k", True)), per_arch_quant=True),
    Kernel("sageattn_qk_int8_pv_fp16_triton", "sageattention", "sageattn_qk_int8_pv_fp16_triton", _TRITON,
           kwargs=(("smooth_k", True),)),
    # SageAttention2: fp8 PV accumulated in fp32 into an fp32 buffer
    Kernel("sageattn_qk_int8_pv_fp8_cuda", "sageattention", "sageattn_qk_int8_pv_fp8_cuda", _SM89_BUILD,
           kwargs=(("pv_accum_dtype", "fp32+fp32"), ("smooth_k", True)), per_arch_quant=True),
    # SageAttention2++: fp16 accumulation into the fp32 buffer, what the official sageattn picks on sm_89 / sm_12x
    Kernel("sageattn_qk_int8_pv_fp8_cuda++", "sageattention", "sageattn_qk_int8_pv_fp8_cuda", _SM89_BUILD,
           kwargs=(("pv_accum_dtype", "fp32+fp16"), ("smooth_k", True)), per_arch_quant=True),
    Kernel("sageattn3 (experimental)", "sageattn3", "sageattn3_blackwell", _SM100_BUILD,
           kwargs=(("per_block_mean", True),), hnd_only=True, max_head_dim=255, takes_scale=False),
)}
MODES = ("disabled", "auto") + tuple(KERNELS)


@dataclasses.dataclass(frozen=True)
class Active:
    """The kernel resolved for one device: the function, its arguments, what it was checked on."""
    kernel: Kernel
    label: str                   # the mode as the logs name it ("auto (sageattn_qk_int8_pv_fp16_cuda)")
    call: Callable
    kwargs: dict
    device_name: str
    capability: tuple
    version: str
    device_type: str = "cuda"

    def describe(self):
        args = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return (f"{self.label} [{args}] on {self.device_name} (sm_{self.capability[0]}{self.capability[1]}), "
                f"{self.kernel.package} {self.version}")


def _installed(package):
    """The installed version of `package`, None when it is not installed."""
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sm(capabilities):
    return ", ".join(f"sm_{major}{minor}" for major, minor in capabilities)


def resolve(mode, device):
    """-> Active, or the reason (a str) the model keeps ComfyUI's attention. `mode` is a kernel
    mode or "auto" (the fp16-PV CUDA kernel, the most accurate one; never triton, fp8 or
    sageattn3). Checks the package, its version, the GPU's compute capability, then runs one small
    test call on the device."""
    kernel = KERNELS[AUTO if mode == "auto" else mode]
    label = f"auto ({kernel.mode})" if mode == "auto" else mode
    version = _installed(kernel.package)
    if version is None:
        return f"{label}: {kernel.package} is not installed"
    if kernel.package == "sageattention" and int(version.split(".", 1)[0]) < 2:
        return (f"{label}: sageattention {version} is a 1.x release (the plain pip package, known to give black "
                "images); install a SageAttention 2.2.x wheel built for this torch and CUDA")
    try:
        call = getattr(importlib.import_module(kernel.package), kernel.function)
    except Exception as e:  # a compiled extension fails to import in many ways (missing libs, ABI)
        return f"{label}: {kernel.package} {version} has no usable {kernel.function} ({type(e).__name__}: {e})"
    if device.type != "cuda" or not torch.cuda.is_available():
        return f"{label}: the model runs on {device}, not on a CUDA GPU"
    capability = tuple(torch.cuda.get_device_capability(device))
    name = torch.cuda.get_device_name(device)
    if capability not in kernel.capabilities:
        return f"{label}: does not run on {name} (sm_{capability[0]}{capability[1]}); it runs on {_sm(kernel.capabilities)}"
    active = Active(kernel, label, call, kernel.arguments(capability), name, capability, version)
    reason = probe(active, device)
    return active if reason is None else f"{label}: {reason}"


def probe(active, device):
    """One small call (1, 2, 256, 128) bf16 HND on the device: None when it runs and gives a
    finite output of the right shape, else the reason. A wheel without kernels for this GPU fails
    here, not in the middle of a run."""
    gen = torch.Generator(device=device).manual_seed(0)
    q, k, v = (torch.randn(1, 2, 256, 128, device=device, dtype=torch.bfloat16, generator=gen) for _ in range(3))
    try:
        out = kernel_call(active, q, k, v, "HND", None)
    except Exception as e:  # whatever the kernel raises, the model keeps ComfyUI's attention
        return f"its test call on {active.device_name} failed ({type(e).__name__}: {e})"
    if tuple(out.shape) != tuple(q.shape):
        return f"its test call on {active.device_name} returned shape {tuple(out.shape)} for {tuple(q.shape)}"
    if not bool(torch.isfinite(out).all()):
        return f"its test call on {active.device_name} returned non-finite values"
    return None


def kernel_call(active, q, k, v, layout, scale):
    """The kernel on q, k, v already in `layout` ("HND" or "NHD")."""
    if active.kernel.hnd_only:
        return active.call(q, k, v, is_causal=False, **active.kwargs)
    return active.call(q, k, v, tensor_layout=layout, is_causal=False, sm_scale=scale, **active.kwargs)


def decline(active, q, k, v, heads, mask, attn_precision, skip_reshape, kwargs):
    """Why this call goes to the original attention, None when the kernel takes it."""
    if kwargs.get("low_precision_attention", True) is False:
        return "low_precision_attention=False"  # core's own flag where sage is known to fail
    if mask is not None:
        return "mask"  # the CUDA kernels take no mask and drop one silently
    if attn_precision == torch.float32:
        return "attn_precision float32"
    if kwargs.get("enable_gqa", False):
        return "enable_gqa"
    if q.device.type != active.device_type:
        return f"not on {active.device_type.upper()}"
    if not q.dtype == k.dtype == v.dtype:
        return "mixed dtypes"
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {str(q.dtype).removeprefix('torch.')}"
    head_dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if head_dim > active.kernel.max_head_dim:
        return f"head_dim {head_dim}"
    if not active.kernel.takes_scale and kwargs.get("scale") is not None:
        return "custom scale"
    return None


def attend(active, q, k, v, heads, skip_reshape, skip_output_reshape, scale):
    """The kernel with ComfyUI's attention layouts in and out."""
    if skip_reshape:  # (B, H, N, D)
        b, _, _, dim_head = q.shape
        layout = "HND"
    else:  # (B, N, H*D)
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.reshape(b, -1, heads, dim_head) for t in (q, k, v))
        layout = "NHD"
    if active.kernel.hnd_only and layout == "NHD":
        q, k, v = (t.transpose(1, 2).contiguous() for t in (q, k, v))
        layout = "HND"
    out = kernel_call(active, q, k, v, layout, scale)
    if layout == "HND":
        return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out.transpose(1, 2) if skip_output_reshape else out.reshape(b, -1, heads * dim_head)


class RunStats:
    """One sampling run's attention calls: how many ran the kernel, how many went to the original
    attention and why. Each reason is logged the first time in a run, or on every call (verbose)."""

    def __init__(self, verbose):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.sage = 0
        self.fallbacks = {}

    def fallback(self, reason, detail, q):
        count = self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        if self.verbose or count == 1:
            problem = reason.startswith("kernel error") or reason == "non-finite output"
            line = f"{PREFIX} original attention for q {tuple(q.shape)} {str(q.dtype).removeprefix('torch.')}: {reason}"
            if detail:
                line += f" ({detail})"
            if not self.verbose:
                line += "; logged once per run, counted in the run summary"
            (logging.warning if problem else logging.info)(line)

    def summary(self, kernel_label):
        total = self.sage + sum(self.fallbacks.values())
        reasons = ", ".join(f"{reason}: {n}" for reason, n in sorted(self.fallbacks.items(), key=lambda item: -item[1]))
        return (f"{PREFIX} run summary: {total} attention calls, {self.sage} ran {kernel_label}, "
                f"{total - self.sage} ran the original attention" + (f" ({reasons})" if reasons else ""))


class SageOverride:
    """ComfyUI's optimized_attention_override for one patched model. `previous` is the override
    the model carried before (None: ComfyUI's own `func`); it gets every call this one declines."""

    def __init__(self, active: Active, previous: Optional[Callable], nan_guard: bool, verbose: bool):
        self.active = active
        self.previous = previous
        self.nan_guard = nan_guard
        self.stats = RunStats(verbose)

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        detail = ""
        reason = decline(self.active, q, k, v, heads, mask, attn_precision, skip_reshape, kwargs)
        if reason is None:
            try:
                out = attend(self.active, q, k, v, heads, skip_reshape, skip_output_reshape, kwargs.get("scale"))
            except Exception as e:  # whatever the kernel raises, this call runs the original attention
                reason, detail = f"kernel error {type(e).__name__}", str(e)
            else:
                if not self.nan_guard or bool(torch.isfinite(out).all()):
                    self.stats.sage += 1
                    return out
                reason = "non-finite output"
                del out
        self.stats.fallback(reason, detail, q)
        options = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                       skip_output_reshape=skip_output_reshape, **kwargs)
        if self.previous is not None:
            return self.previous(func, q, k, v, heads, **options)
        return func(q, k, v, heads, **options)

    def start_run(self):
        """A sampling run starts (the patcher's ON_PRE_RUN): the kernel line, fresh counts."""
        self.stats.reset()
        logging.info("%s %s, nan_guard %s", PREFIX, self.active.describe(), "on" if self.nan_guard else "off")

    def end_run(self):
        """A sampling run ended (the patcher's ON_CLEANUP, also after an error): the summary, then
        fresh counts. A warning when no call ran the kernel."""
        line = self.stats.summary(self.active.label)
        (logging.warning if self.stats.sage == 0 else logging.info)(line)
        self.stats.reset()

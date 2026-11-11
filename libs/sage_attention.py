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
each function, and the integer width each kernel indexes elements in (the CUDA kernels uint32, the
Triton ones int32). Q·K quantization (`qk_quant_gran`) is the node's `qk_quant`: per_thread (the
finest, quantized by a Triton kernel) or per_warp (quantized by a CUDA kernel); the official dispatch
takes per_warp on sm_120 / sm_121 only because Triton did not run there when it was written (PR
#109). Only torch and the standard library at module level; `sageattention` is imported by
`resolve`.
"""

import dataclasses
import importlib
import logging
from typing import Callable, Optional

import torch

PREFIX = "[BCNodes] Patch Sage Attention:"
AUTO = "sageattn_qk_int8_pv_fp16_cuda"
# The node's qk_quant: auto takes per_thread when its test call passes, else per_warp.
QUANTS = ("auto", "per_thread", "per_warp")
# per_thread's Triton quantization indexes in int32: element offsets from 2**31 on wrap and corrupt
# the output silently (thu-ml/SageAttention#386, measured on an RTX 5090: cosine 0.94 at 110k tokens).
PER_THREAD_BITS = 31
# The probe's floor against torch SDPA: under every kernel's accuracy (INT8 Q·K >= 0.9992 measured on
# an RTX 5090; SageAttention3's FP4 0.9955 in its paper), far over a broken kernel's.
PROBE_MIN_COSINE = 0.98


@dataclasses.dataclass(frozen=True)
class Kernel:
    """One SageAttention function the node offers, as the official code defines it."""
    mode: str                    # the node's combo value
    package: str                 # distribution and import name
    function: str                # the function in `package`
    capabilities: tuple          # compute capabilities the official build compiles it for
    kwargs: tuple = ()           # fixed keyword arguments, (name, value) pairs
    takes_quant: bool = False    # takes qk_quant_gran (per_thread / per_warp)
    index_bits: Optional[int] = None  # elements indexed in uint32 (32) or int32 (31); None: not read from its code
    hnd_only: bool = False       # takes (B, H, N, D) only, no tensor_layout argument
    max_head_dim: int = 128      # larger head dims raise in the kernel
    takes_scale: bool = True     # takes sm_scale; sageattn3 fixes the scale at 1 / sqrt(head_dim)

    def arguments(self, quant):
        kwargs = dict(self.kwargs)
        if self.takes_quant:
            kwargs["qk_quant_gran"] = quant
        return kwargs


# setup.py's SUPPORTED_ARCHS; the _qattn_sm80 module (the fp16 CUDA kernel) is built for each of them.
_SM80_BUILD = ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0), (12, 1))
# The _qattn_sm89 module (the fp8 CUDA kernels) is built for 8.9 and newer.
_SM89_BUILD = ((8, 9), (9, 0), (10, 0), (12, 0), (12, 1))
# The official dispatch: "triton kernel is currently not usable on sm120" (and sm121).
_TRITON = ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0))
# sageattn3's setup.py: sm_100a, sm_120a, sm_121a.
_SM100_BUILD = ((10, 0), (12, 0), (12, 1))

# Index widths: csrc/qattn and csrc/fused take uint32 strides and offsets; sageattention/triton
# computes offsets from int32 program ids and aranges.
KERNELS = {k.mode: k for k in (
    Kernel("sageattn_qk_int8_pv_fp16_cuda", "sageattention", "sageattn_qk_int8_pv_fp16_cuda", _SM80_BUILD,
           kwargs=(("pv_accum_dtype", "fp32"), ("smooth_k", True)), takes_quant=True, index_bits=32),
    Kernel("sageattn_qk_int8_pv_fp16_triton", "sageattention", "sageattn_qk_int8_pv_fp16_triton", _TRITON,
           kwargs=(("smooth_k", True),), index_bits=31),
    # SageAttention2: fp8 PV accumulated in fp32 into an fp32 buffer
    Kernel("sageattn_qk_int8_pv_fp8_cuda", "sageattention", "sageattn_qk_int8_pv_fp8_cuda", _SM89_BUILD,
           kwargs=(("pv_accum_dtype", "fp32+fp32"), ("smooth_k", True)), takes_quant=True, index_bits=32),
    # SageAttention2++: fp16 accumulation into the fp32 buffer, what the official sageattn picks on sm_89 / sm_12x
    Kernel("sageattn_qk_int8_pv_fp8_cuda++", "sageattention", "sageattn_qk_int8_pv_fp8_cuda", _SM89_BUILD,
           kwargs=(("pv_accum_dtype", "fp32+fp16"), ("smooth_k", True)), takes_quant=True, index_bits=32),
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
    note: str = ""               # why per_thread was refused when auto took per_warp

    def describe(self):
        args = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return (f"{self.label} [{args}] on {self.device_name} (sm_{self.capability[0]}{self.capability[1]}), "
                f"{self.kernel.package} {self.version}" + (f" ({self.note})" if self.note else ""))


def _installed(package):
    """The installed version of `package`, None when it is not installed."""
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sm(capabilities):
    return ", ".join(f"sm_{major}{minor}" for major, minor in capabilities)


def resolve(mode, quant, device):
    """-> Active, or the reason (a str) the model keeps ComfyUI's attention. `mode` is a kernel
    mode or "auto" (the fp16-PV CUDA kernel, the most accurate one; never triton, fp8 or
    sageattn3); `quant` one of QUANTS, for the kernels that take qk_quant_gran (auto: per_thread,
    then per_warp). Checks the package, its version, the GPU's compute capability, then runs one
    small test call on the device per quantization tried."""
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
    if not kernel.takes_quant:
        quants = (None,)
    else:
        quants = ("per_thread", "per_warp") if quant == "auto" else (quant,)
    refused = []
    for gran in quants:
        active = Active(kernel, label, call, kernel.arguments(gran), name, capability, version, note="; ".join(refused))
        reason = probe(active, device)
        if reason is None:
            return active
        refused.append(f"{gran} refused: {reason}" if gran else reason)
    return f"{label}: " + "; ".join(refused)


def probe(active, device):
    """One small call (1, 2, 256, 128) bf16 HND on the device: None when it runs and gives a
    finite output of the right shape within PROBE_MIN_COSINE of torch SDPA, else the reason. A
    wheel without kernels for this GPU, or a kernel that computes garbage on it, fails here, not in
    the middle of a run."""
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
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float())
    cosine = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.flatten(), dim=0).item()
    if cosine < PROBE_MIN_COSINE:
        return f"its test call on {active.device_name} is off from torch SDPA (cosine {cosine:.4f}, under {PROBE_MIN_COSINE})"
    return None


def kernel_call(active, q, k, v, layout, scale):
    """The kernel on q, k, v already in `layout` ("HND" or "NHD")."""
    if active.kernel.hnd_only:
        return active.call(q, k, v, is_causal=False, **active.kwargs)
    return active.call(q, k, v, tensor_layout=layout, is_causal=False, sm_scale=scale, **active.kwargs)


def max_offset(*tensors):
    """The largest element offset a kernel computes from these tensors' data pointers: the last
    element's, sum((size - 1) * stride). Views into a fused QKV buffer reach it at a third of the
    tokens contiguous tensors need."""
    return max(sum((n - 1) * s for n, s in zip(t.shape, t.stride())) for t in tensors)


def decline(active, q, k, v, heads, mask, attn_precision, skip_reshape, kwargs, offset):
    """Why this call goes to the original attention, None when the kernel takes it. `offset`:
    max_offset(q, k, v)."""
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
    bits = active.kernel.index_bits
    if bits is not None and offset >= 2**bits:
        return f"offsets past 2**{bits}"  # the kernel's own indexing wraps there
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
    """One sampling run's attention calls: how many ran the kernel (and how many of those per_warp
    in place of per_thread), how many went to the original attention and why. Each reason is logged
    the first time in a run, or on every call (verbose)."""

    def __init__(self, verbose):
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.sage = 0
        self.per_warp = 0
        self.fallbacks = {}

    def _log(self, line, count, level):
        if self.verbose or count == 1:
            level(line if self.verbose else line + "; logged once per run, counted in the run summary")

    def ran(self, q, switched):
        self.sage += 1
        if switched:
            self.per_warp += 1
            self._log(f"{PREFIX} per_warp for q {_tensor(q)}: offsets past 2**{PER_THREAD_BITS}, where per_thread's "
                      "Triton quantization wraps (thu-ml/SageAttention#386)", self.per_warp, logging.info)

    def fallback(self, reason, detail, q):
        count = self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        problem = reason.startswith("kernel error") or reason == "non-finite output"
        line = f"{PREFIX} original attention for q {_tensor(q)}: {reason}" + (f" ({detail})" if detail else "")
        self._log(line, count, logging.warning if problem else logging.info)

    def summary(self, kernel_label):
        total = self.sage + sum(self.fallbacks.values())
        reasons = ", ".join(f"{reason}: {n}" for reason, n in sorted(self.fallbacks.items(), key=lambda item: -item[1]))
        switched = f" ({self.per_warp} of them per_warp: offsets past 2**{PER_THREAD_BITS})" if self.per_warp else ""
        return (f"{PREFIX} run summary: {total} attention calls, {self.sage} ran {kernel_label}{switched}, "
                f"{total - self.sage} ran the original attention" + (f" ({reasons})" if reasons else ""))


def _tensor(q):
    return f"{tuple(q.shape)} {str(q.dtype).removeprefix('torch.')}"


class SageOverride:
    """ComfyUI's optimized_attention_override for one patched model. `previous` is the override
    the model carried before (None: ComfyUI's own `func`); it gets every call this one declines."""

    def __init__(self, active: Active, previous: Optional[Callable], nan_guard: bool, verbose: bool):
        self.active = active
        self.previous = previous
        self.nan_guard = nan_guard
        self.stats = RunStats(verbose)
        # per_thread's stand-in for a call whose offsets its Triton quantization cannot index
        self.per_warp = (dataclasses.replace(active, kwargs={**active.kwargs, "qk_quant_gran": "per_warp"})
                         if active.kwargs.get("qk_quant_gran") == "per_thread" else None)

    def __call__(self, func, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        detail = ""
        offset = max_offset(q, k, v)
        reason = decline(self.active, q, k, v, heads, mask, attn_precision, skip_reshape, kwargs, offset)
        if reason is None:
            switched = self.per_warp is not None and offset >= 2**PER_THREAD_BITS
            active = self.per_warp if switched else self.active
            try:
                out = attend(active, q, k, v, heads, skip_reshape, skip_output_reshape, kwargs.get("scale"))
            except Exception as e:  # whatever the kernel raises, this call runs the original attention
                reason, detail = f"kernel error {type(e).__name__}", str(e)
            else:
                if not self.nan_guard or bool(torch.isfinite(out).all()):
                    self.stats.ran(q, switched)
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

"""libs/sage_attention.py: the override's layouts against a reference with ComfyUI's attention
contract, every reason a call goes to the original attention, the kernel error and NaN paths, the
chain to an override the model already had, the once / verbose logs, the per-run summary, and
`resolve` / `probe` up to the GPU (the package, its version, the compute capability, the test call).

The kernel is a stand-in: SDPA on CPU with SageAttention's signature (tensor_layout, sm_scale), in an
Active whose device type is "cpu", so the kernel path runs here; the real kernels are measured on a
GPU by tests/compare_sage_attention.py.
"""

import logging
import sys
import types

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def sa(bcnodes):
    return bcnodes["libs.sage_attention"]


def sdpa_kernel(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, **kwargs):
    """SageAttention's call signature over torch SDPA."""
    if tensor_layout == "NHD":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=sm_scale).to(q.dtype)
    return out.transpose(1, 2) if tensor_layout == "NHD" else out


def hnd_kernel(q, k, v, is_causal=False, **kwargs):
    """sageattn3's call signature: (B, H, N, D) only, no layout, no scale."""
    assert "tensor_layout" not in kwargs and "sm_scale" not in kwargs
    return sdpa_kernel(q, k, v)


def reference(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    """ComfyUI's attention contract (its attention_pytorch), in float32: HND in with skip_reshape,
    else (B, N, H*D); HND out with skip_output_reshape, else (B, N, H*D)."""
    if skip_reshape:
        b, _, _, d = q.shape
    else:
        b, _, inner = q.shape
        d = inner // heads
        q, k, v = (t.reshape(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=kwargs.get("scale")).to(q.dtype)
    return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * d)


def active(sa, mode=None, call=sdpa_kernel, device_type="cpu"):
    kernel = sa.KERNELS[mode or sa.AUTO]
    return sa.Active(kernel, "stand-in", call, kernel.arguments("per_thread"), "CPU", (8, 6), "2.2.0", device_type=device_type)


def qkv(skip_reshape, heads=4, d=64, n=96, kv_n=96, b=2, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(0)
    if skip_reshape:
        shapes = ((b, heads, n, d), (b, heads, kv_n, d), (b, heads, kv_n, d))
    else:
        shapes = ((b, n, heads * d), (b, kv_n, heads * d), (b, kv_n, heads * d))
    return tuple(torch.randn(s, generator=g).to(dtype) for s in shapes)


@pytest.mark.parametrize("skip_reshape", [False, True])
@pytest.mark.parametrize("skip_output_reshape", [False, True])
@pytest.mark.parametrize("kv_n", [96, 40])  # self-attention, and cross-attention with a shorter context
def test_layouts_match_comfy_attention(sa, skip_reshape, skip_output_reshape, kv_n):
    q, k, v = qkv(skip_reshape, kv_n=kv_n)
    override = sa.SageOverride(active(sa), None, True, False)
    out = override(reference, q, k, v, 4, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                   transformer_options={})
    want = reference(q, k, v, 4, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape)
    assert out.shape == want.shape and out.dtype == want.dtype
    torch.testing.assert_close(out, want, atol=2e-2, rtol=2e-2)
    assert (override.stats.sage, override.stats.fallbacks) == (1, {})


def test_scale_reaches_the_kernel(sa):
    q, k, v = qkv(True)
    out = sa.SageOverride(active(sa), None, True, False)(reference, q, k, v, 4, skip_reshape=True, scale=0.5)
    torch.testing.assert_close(out, reference(q, k, v, 4, skip_reshape=True, scale=0.5), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("skip_reshape", [False, True])
def test_an_hnd_only_kernel_gets_hnd(sa, skip_reshape):
    q, k, v = qkv(skip_reshape)
    override = sa.SageOverride(active(sa, "sageattn3 (experimental)", hnd_kernel), None, True, False)
    out = override(reference, q, k, v, 4, skip_reshape=skip_reshape)
    torch.testing.assert_close(out, reference(q, k, v, 4, skip_reshape=skip_reshape), atol=2e-2, rtol=2e-2)
    assert override.stats.sage == 1


def declined_cases():
    """(name, mode, device_type, the call's q k v, keyword arguments, the reason)."""
    hnd = qkv(True)
    nhd = qkv(False)
    return [
        ("low precision off", None, "cpu", nhd, {"low_precision_attention": False}, "low_precision_attention=False"),
        ("mask", None, "cpu", nhd, {"mask": torch.zeros(2, 1, 96)}, "mask"),
        ("fp32 attention precision", None, "cpu", nhd, {"attn_precision": torch.float32}, "attn_precision float32"),
        ("gqa", None, "cpu", nhd, {"enable_gqa": True}, "enable_gqa"),
        ("not on the kernel's device", None, "cuda", nhd, {}, "not on CUDA"),
        ("float32", None, "cpu", tuple(t.float() for t in nhd), {}, "dtype float32"),
        ("mixed dtypes", None, "cpu", (nhd[0], nhd[1].half(), nhd[2]), {}, "mixed dtypes"),
        ("head_dim over 128", None, "cpu", qkv(True, heads=2, d=256), {"skip_reshape": True}, "head_dim 256"),
        ("custom scale, sageattn3", "sageattn3 (experimental)", "cpu", hnd, {"skip_reshape": True, "scale": 0.1}, "custom scale"),
    ]


@pytest.mark.parametrize("case", declined_cases(), ids=lambda c: c[0])
def test_each_declined_call_runs_the_original_with_its_arguments(sa, case):
    _, mode, device_type, (q, k, v), kwargs, reason = case
    kernel_calls, seen = [], []

    def kernel(*args, **kw):
        kernel_calls.append(1)
        return sdpa_kernel(*args, **kw)

    def func(*args, **kw):
        seen.append((args, kw))
        return "original"

    override = sa.SageOverride(active(sa, mode, kernel, device_type), None, True, False)
    assert override(func, q, k, v, 2 if reason == "head_dim 256" else 4, transformer_options={}, **kwargs) == "original"
    assert kernel_calls == [] and override.stats.fallbacks == {reason: 1} and override.stats.sage == 0
    args, kw = seen[0]
    assert args[0] is q and args[1] is k and args[2] is v and kw["transformer_options"] == {}
    for name, value in kwargs.items():
        assert kw[name] is value


def test_a_kernel_error_runs_the_original_and_is_a_warning(sa, caplog):
    def broken(*args, **kw):
        raise RuntimeError("no kernel image is available for execution on the device")

    q, k, v = qkv(False)
    override = sa.SageOverride(active(sa, call=broken), None, True, False)
    with caplog.at_level(logging.INFO):
        out = override(reference, q, k, v, 4)
    assert torch.equal(out, reference(q, k, v, 4))
    assert override.stats.fallbacks == {"kernel error RuntimeError": 1}
    record = caplog.records[-1]
    assert record.levelno == logging.WARNING and "no kernel image" in record.getMessage()


def test_nan_guard_recomputes_a_non_finite_output(sa):
    def poisoned(*args, **kw):
        out = sdpa_kernel(*args, **kw)
        out.view(-1)[0] = float("nan")
        return out

    q, k, v = qkv(False)
    guarded = sa.SageOverride(active(sa, call=poisoned), None, True, False)
    out = guarded(reference, q, k, v, 4)
    assert torch.isfinite(out).all() and torch.equal(out, reference(q, k, v, 4))
    assert (guarded.stats.sage, guarded.stats.fallbacks) == (0, {"non-finite output": 1})

    unguarded = sa.SageOverride(active(sa, call=poisoned), None, False, False)
    assert torch.isnan(unguarded(reference, q, k, v, 4)).any() and unguarded.stats.sage == 1


def test_a_declined_call_goes_to_the_override_the_model_had(sa):
    calls = []

    def previous(func, q, k, v, heads, **kw):
        calls.append(kw["mask"])
        return func(q, k, v, heads, **kw)

    q, k, v = qkv(False)
    mask = torch.zeros(2, 1, 96)
    override = sa.SageOverride(active(sa), previous, True, False)
    assert override(lambda *a, **kw: "core", q, k, v, 4, mask=mask) == "core"
    assert calls == [mask]
    override(reference, q, k, v, 4)  # a call the kernel takes never reaches `previous`
    assert len(calls) == 1 and override.stats.sage == 1


@pytest.mark.parametrize("verbose, lines", [(False, 1), (True, 3)])
def test_once_logs_a_reason_once_per_run_verbose_every_call(sa, caplog, verbose, lines):
    q, k, v = qkv(False)
    override = sa.SageOverride(active(sa), None, True, verbose)
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            override(reference, q, k, v, 4, mask=torch.zeros(2, 1, 96))
        assert len([r for r in caplog.records if ": mask" in r.getMessage()]) == lines
        override.end_run()
        caplog.clear()
        override(reference, q, k, v, 4, mask=torch.zeros(2, 1, 96))  # a new run logs the reason again
        assert len([r for r in caplog.records if ": mask" in r.getMessage()]) == 1


def test_run_summary_counts_each_path_and_reason(sa, caplog):
    """The run's end says how many calls ran the kernel and how many the original attention, by
    reason; a run where every call fell back (as a mask on every Qwen call would) is a warning."""
    q, k, v = qkv(False)
    override = sa.SageOverride(active(sa), None, True, False)
    with caplog.at_level(logging.INFO):
        override.start_run()
        assert caplog.records[-1].getMessage() == (
            "[BCNodes] Patch Sage Attention: stand-in [pv_accum_dtype=fp32, smooth_k=True, qk_quant_gran=per_thread] "
            "on CPU (sm_86), sageattention 2.2.0, nan_guard on")
        for _ in range(2):
            override(reference, q, k, v, 4)
        for _ in range(3):
            override(reference, q, k, v, 4, mask=torch.zeros(2, 1, 96))
        override(reference, q, k, v, 4, low_precision_attention=False)
        override.end_run()
        summary = caplog.records[-1]
        assert summary.levelno == logging.INFO and summary.getMessage() == (
            "[BCNodes] Patch Sage Attention: run summary: 6 attention calls, 2 ran stand-in, 4 ran the original "
            "attention (mask: 3, low_precision_attention=False: 1)")

        override.start_run()
        for _ in range(5):
            override(reference, q, k, v, 4, mask=torch.zeros(2, 1, 96))
        override.end_run()
        summary = caplog.records[-1]
        assert summary.levelno == logging.WARNING and summary.getMessage() == (
            "[BCNodes] Patch Sage Attention: run summary: 5 attention calls, 0 ran stand-in, 5 ran the original "
            "attention (mask: 5)")
    assert (override.stats.sage, override.stats.fallbacks) == (0, {})


def test_modes_are_the_static_combo(sa):
    assert sa.MODES == ("disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton",
                        "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3 (experimental)")


def test_kernel_arguments_follow_the_official_code(sa):
    k = sa.KERNELS
    assert k["sageattn_qk_int8_pv_fp16_cuda"].arguments("per_thread") == {"pv_accum_dtype": "fp32", "smooth_k": True, "qk_quant_gran": "per_thread"}
    assert k["sageattn_qk_int8_pv_fp16_cuda"].arguments("per_warp")["qk_quant_gran"] == "per_warp"
    assert k["sageattn_qk_int8_pv_fp8_cuda"].arguments("per_thread")["pv_accum_dtype"] == "fp32+fp32"
    assert k["sageattn_qk_int8_pv_fp8_cuda++"].arguments("per_warp") == {"pv_accum_dtype": "fp32+fp16", "smooth_k": True, "qk_quant_gran": "per_warp"}
    assert k["sageattn_qk_int8_pv_fp16_triton"].arguments(None) == {"smooth_k": True}
    assert k["sageattn3 (experimental)"].arguments(None) == {"per_block_mean": True}
    assert (12, 0) not in k["sageattn_qk_int8_pv_fp16_triton"].capabilities
    assert (8, 6) not in k["sageattn_qk_int8_pv_fp8_cuda"].capabilities
    # the integer width each kernel indexes in: CUDA uint32, Triton int32, sageattn3 not read
    assert {mode: kernel.index_bits for mode, kernel in k.items()} == {
        "sageattn_qk_int8_pv_fp16_cuda": 32, "sageattn_qk_int8_pv_fp16_triton": 31, "sageattn_qk_int8_pv_fp8_cuda": 32,
        "sageattn_qk_int8_pv_fp8_cuda++": 32, "sageattn3 (experimental)": None}
    assert sa.QUANTS == ("auto", "per_thread", "per_warp")


def test_max_offset_is_the_last_elements_offset(sa):
    fused = torch.zeros(2, 10, 3 * 4 * 8)  # (B, N, 3 * H * D): q, k, v as views, as a fused QKV projection gives
    q, k, v = fused.chunk(3, dim=-1)
    assert sa.max_offset(q, k, v) == 1 * 10 * 96 + 9 * 96 + 31  # batch, token and channel strides of the buffer
    contiguous = torch.zeros(2, 10, 32)
    assert sa.max_offset(contiguous) == contiguous.numel() - 1
    assert sa.max_offset(contiguous, q) == sa.max_offset(q)


def recording(calls):
    def kernel(*args, **kw):
        calls.append(kw.get("qk_quant_gran"))
        return sdpa_kernel(*args, **kw)
    return kernel


@pytest.mark.parametrize("offset, ran, summary", [
    (2**31 - 1, "per_thread", "1 ran stand-in, 0 ran"),
    (2**31, "per_warp", "1 ran stand-in (1 of them per_warp: offsets past 2**31), 0 ran"),
    (2**32 - 1, "per_warp", "1 ran stand-in (1 of them per_warp: offsets past 2**31), 0 ran"),
])
def test_per_thread_past_int32_offsets_runs_per_warp(sa, monkeypatch, caplog, offset, ran, summary):
    """thu-ml/SageAttention#386: per_thread's Triton quantization indexes in int32 and corrupts the
    output silently from 2**31 on; such a call runs per_warp (CUDA, uint32), counted and logged."""
    monkeypatch.setattr(sa, "max_offset", lambda *tensors: offset)
    calls = []
    q, k, v = qkv(False)
    override = sa.SageOverride(active(sa, call=recording(calls)), None, True, False)
    with caplog.at_level(logging.INFO):
        out = override(reference, q, k, v, 4)
        assert summary in override.stats.summary("stand-in")
        override(reference, q, k, v, 4)  # the second call is counted, not logged again
    torch.testing.assert_close(out, reference(q, k, v, 4), atol=2e-2, rtol=2e-2)
    assert calls == [ran, ran] and override.active.kwargs["qk_quant_gran"] == "per_thread"
    lines = [r.getMessage() for r in caplog.records if "per_warp for q" in r.getMessage()]
    assert lines == ([] if ran == "per_thread" else [
        "[BCNodes] Patch Sage Attention: per_warp for q (2, 96, 256) bfloat16: offsets past 2**31, where per_thread's "
        "Triton quantization wraps (thu-ml/SageAttention#386); logged once per run, counted in the run summary"])


@pytest.mark.parametrize("mode, quant, offset, reason", [
    (None, "per_thread", 2**32, "offsets past 2**32"),  # CUDA kernels index in uint32
    (None, "per_warp", 2**32, "offsets past 2**32"),
    ("sageattn_qk_int8_pv_fp16_triton", None, 2**31, "offsets past 2**31"),  # Triton in int32
])
def test_offsets_past_the_kernels_indexing_run_the_original(sa, monkeypatch, mode, quant, offset, reason):
    monkeypatch.setattr(sa, "max_offset", lambda *tensors: offset)
    calls = []
    kernel = sa.KERNELS[mode or sa.AUTO]
    a = sa.Active(kernel, "stand-in", recording(calls), kernel.arguments(quant), "CPU", (8, 6), "2.2.0", device_type="cpu")
    q, k, v = qkv(False)
    override = sa.SageOverride(a, None, True, False)
    assert override(lambda *args, **kw: "original", q, k, v, 4) == "original"
    assert calls == [] and override.stats.fallbacks == {reason: 1}


def test_per_warp_and_sageattn3_are_not_switched(sa, monkeypatch):
    monkeypatch.setattr(sa, "max_offset", lambda *tensors: 2**31)
    calls = []
    kernel = sa.KERNELS[sa.AUTO]
    per_warp = sa.Active(kernel, "stand-in", recording(calls), kernel.arguments("per_warp"), "CPU", (8, 6), "2.2.0", device_type="cpu")
    q, k, v = qkv(False)
    override = sa.SageOverride(per_warp, None, True, False)
    override(reference, q, k, v, 4)
    assert calls == ["per_warp"] and override.stats.per_warp == 0 and override.per_warp is None
    monkeypatch.setattr(sa, "max_offset", lambda *tensors: 2**40)  # sageattn3's indexing is not claimed
    hnd = sa.SageOverride(active(sa, "sageattn3 (experimental)", hnd_kernel), None, True, False)
    hnd(reference, *qkv(True), 4, skip_reshape=True)
    assert hnd.stats.sage == 1 and hnd.stats.fallbacks == {}


@pytest.fixture
def fake_gpu(sa, monkeypatch):
    """An importable `sageattention` 2.2.0 and a CUDA device of the capability set in `gpu`."""
    module = types.ModuleType("sageattention")
    for name in ("sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda"):
        setattr(module, name, sdpa_kernel)
    monkeypatch.setitem(sys.modules, "sageattention", module)
    monkeypatch.setattr(sa, "_installed", lambda package: "2.2.0" if package == "sageattention" else None)
    gpu = {"capability": (12, 0), "name": "NVIDIA RTX PRO 4500 Blackwell"}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: gpu["capability"])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: gpu["name"])
    monkeypatch.setattr(sa, "probe", lambda active, device: None)
    return gpu


def test_resolve_auto_is_the_fp16_cuda_kernel_with_per_thread(sa, fake_gpu):
    got = sa.resolve("auto", "auto", torch.device("cuda", 0))
    assert got.kernel.mode == "sageattn_qk_int8_pv_fp16_cuda" and got.label == "auto (sageattn_qk_int8_pv_fp16_cuda)"
    assert got.kwargs == {"pv_accum_dtype": "fp32", "smooth_k": True, "qk_quant_gran": "per_thread"}
    assert got.describe() == ("auto (sageattn_qk_int8_pv_fp16_cuda) [pv_accum_dtype=fp32, smooth_k=True, qk_quant_gran=per_thread] "
                              "on NVIDIA RTX PRO 4500 Blackwell (sm_120), sageattention 2.2.0")


def test_resolve_quant_order(sa, fake_gpu, monkeypatch):
    """auto: per_thread, then per_warp when per_thread's test call fails (the reason in the kernel
    line); a quantization named in qk_quant is the only one tried; a kernel without qk_quant_gran
    is probed once, without it."""
    probed = []

    def probe(active, device):
        gran = active.kwargs.get("qk_quant_gran")
        probed.append(gran)
        return "its test call on GPU failed (RuntimeError: boom)" if gran == "per_thread" else None

    monkeypatch.setattr(sa, "probe", probe)
    cuda = torch.device("cuda", 0)
    got = sa.resolve("auto", "auto", cuda)
    assert probed == ["per_thread", "per_warp"] and got.kwargs["qk_quant_gran"] == "per_warp"
    assert got.describe().endswith("sageattention 2.2.0 (per_thread refused: its test call on GPU failed (RuntimeError: boom))")
    probed.clear()
    assert sa.resolve("sageattn_qk_int8_pv_fp8_cuda", "per_warp", cuda).note == "" and probed == ["per_warp"]
    probed.clear()
    assert sa.resolve("auto", "per_thread", cuda) == (
        "auto (sageattn_qk_int8_pv_fp16_cuda): per_thread refused: its test call on GPU failed (RuntimeError: boom)")
    assert probed == ["per_thread"]
    fake_gpu.update(capability=(8, 6), name="NVIDIA GeForce RTX 3090")
    probed.clear()
    got = sa.resolve("sageattn_qk_int8_pv_fp16_triton", "per_thread", cuda)
    assert probed == [None] and got.kwargs == {"smooth_k": True}


def test_resolve_refuses_what_cannot_run(sa, fake_gpu, monkeypatch):
    cuda = torch.device("cuda", 0)
    assert sa.resolve("sageattn_qk_int8_pv_fp16_triton", "auto", cuda) == (
        "sageattn_qk_int8_pv_fp16_triton: does not run on NVIDIA RTX PRO 4500 Blackwell (sm_120); "
        "it runs on sm_80, sm_86, sm_89, sm_90, sm_100")
    fake_gpu.update(capability=(7, 5), name="Tesla T4")
    assert sa.resolve("auto", "auto", cuda).startswith("auto (sageattn_qk_int8_pv_fp16_cuda): does not run on Tesla T4 (sm_75)")
    assert sa.resolve("auto", "auto", torch.device("cpu")) == "auto (sageattn_qk_int8_pv_fp16_cuda): the model runs on cpu, not on a CUDA GPU"
    assert sa.resolve("sageattn3 (experimental)", "auto", cuda) == "sageattn3 (experimental): sageattn3 is not installed"
    monkeypatch.setattr(sa, "_installed", lambda package: "1.0.6")
    assert "1.x release" in sa.resolve("auto", "auto", cuda)
    monkeypatch.setattr(sa, "_installed", lambda package: None)
    assert sa.resolve("auto", "auto", cuda) == "auto (sageattn_qk_int8_pv_fp16_cuda): sageattention is not installed"


def test_resolve_names_a_kernel_the_package_lacks(sa, fake_gpu, monkeypatch):
    monkeypatch.delattr(sys.modules["sageattention"], "sageattn_qk_int8_pv_fp8_cuda")
    got = sa.resolve("sageattn_qk_int8_pv_fp8_cuda", "auto", torch.device("cuda", 0))
    assert got.startswith("sageattn_qk_int8_pv_fp8_cuda: sageattention 2.2.0 has no usable sageattn_qk_int8_pv_fp8_cuda (AttributeError")


def test_resolve_refuses_a_failed_test_call(sa, fake_gpu, monkeypatch):
    monkeypatch.setattr(sa, "probe", lambda active, device: "its test call on GPU failed (RuntimeError: boom)")
    assert sa.resolve("auto", "auto", torch.device("cuda", 0)) == (
        "auto (sageattn_qk_int8_pv_fp16_cuda): per_thread refused: its test call on GPU failed (RuntimeError: boom); "
        "per_warp refused: its test call on GPU failed (RuntimeError: boom)")
    fake_gpu.update(capability=(8, 6), name="NVIDIA GeForce RTX 3090")
    assert sa.resolve("sageattn_qk_int8_pv_fp16_triton", "auto", torch.device("cuda", 0)) == (
        "sageattn_qk_int8_pv_fp16_triton: its test call on GPU failed (RuntimeError: boom)")


def test_probe_runs_one_call_and_checks_it(sa):
    cpu = torch.device("cpu")
    assert sa.probe(active(sa), cpu) is None

    def raises(*args, **kw):
        raise RuntimeError("boom")

    def nan(*args, **kw):
        return torch.full_like(args[0], float("nan"))

    def wrong_shape(*args, **kw):
        return args[0][:, :1]

    def garbage(q, k, v, **kw):  # runs, finite, right shape, wrong values
        return v

    assert sa.probe(active(sa, call=raises), cpu) == "its test call on CPU failed (RuntimeError: boom)"
    assert sa.probe(active(sa, call=nan), cpu) == "its test call on CPU returned non-finite values"
    assert sa.probe(active(sa, call=wrong_shape), cpu) == "its test call on CPU returned shape (1, 1, 256, 128) for (1, 2, 256, 128)"
    got = sa.probe(active(sa, call=garbage), cpu)
    assert got.startswith("its test call on CPU is off from torch SDPA (cosine 0.") and got.endswith(", under 0.98)")

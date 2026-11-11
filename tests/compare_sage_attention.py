"""Patch Sage Attention's kernels against torch SDPA on a CUDA GPU, numerically and in time.

    python tests/compare_sage_attention.py                # from the pack's directory, one command
    python tests/compare_sage_attention.py | tee sage.txt # keep the report

Needs torch with CUDA and SageAttention 2.x (sageattn3 too, for its mode); ComfyUI is not needed.
Without a CUDA GPU it prints SKIP and exits 0. Not part of the test suite.

1. Which modes the node accepts on this GPU: each mode through the node's own `resolve` with
   qk_quant auto (the package, its version, the compute capability, the test calls), with the
   arguments it would use.
2. Accuracy and speed. Random q / k / v at Qwen-Image shapes (24 heads x 128, 4k / 8k / 16k tokens,
   bf16, (B, H, N, D) as Qwen-Image calls attention) and Wan 2.x 14B video shapes (40 heads x 128,
   81 frames at 480p / 720p: 32760 / 75600 tokens, (B, N, H*D) as Wan calls it). Two input sets:
   plain randn, and randn with a per-channel offset on K (the case smooth_k exists for). The
   reference is torch SDPA in float32; every row is compared with it (cosine similarity, max and
   mean absolute error, non-finite values) and timed (ms per call, speed against SDPA in bf16,
   ComfyUI's default). Rows: SDPA bf16 itself, then every kernel the node accepts here with the
   node's arguments (marked "node"), with smooth_k flipped, and with the other qk_quant_gran.
   Random inputs give flatter attention than a real model; an image A/B is the final check.
3. The override's fallbacks on the auto kernel: a masked call and a kernel output with an injected
   NaN both run the original attention (exit 1 if not); the cost of nan_guard per call.
4. Fused QKV views on the auto kernel: q / k / v as non-contiguous views into one fused buffer
   (sequence stride 3 x 56 x 128), under and over the length where the Triton per_thread quant's
   int32 offsets pass 2**31 (thu-ml/SageAttention#386); per_thread and per_warp, each also on
   contiguous copies of the same values, and the node's own path (SageOverride, whose guard runs
   per_warp past 2**31). Reported only, never the exit code. Runs last: an illegal memory access
   ends the CUDA context.
"""

import dataclasses
import logging
import sys
import time

import torch
import torch.nn.functional as F

from _harness import bind_package

QWEN = [("Qwen-Image 4k", 1, 24, 4096), ("Qwen-Image 8k", 1, 24, 8192), ("Qwen-Image 16k", 1, 24, 16384)]
WAN = [("Wan 14B 480p 81f", 1, 40, 32760), ("Wan 14B 720p 81f", 1, 40, 75600)]
HEAD_DIM = 128
K_OFFSET = 4.0  # std of the per-channel offset added to K in the second input set
ITERS = 3       # timed calls per row, after one warm-up call
FUSED_HEADS = 56                   # the fused-QKV case: #386's geometry, 56 heads x 128
FUSED_LENGTHS = (90_000, 110_000)  # under / over 2**31 // (3 x 56 x 128) = 99864 tokens


def comfy_sdpa(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    """torch SDPA with ComfyUI's attention contract (its attention_pytorch): (B, H, N, D) in with
    skip_reshape, else (B, N, H*D); (B, H, N, D) out with skip_output_reshape, else (B, N, H*D)."""
    if skip_reshape:
        b, _, _, d = q.shape
    else:
        b, _, inner = q.shape
        d = inner // heads
        q, k, v = (t.view(b, -1, heads, d).transpose(1, 2) for t in (q, k, v))
    if mask is not None and mask.ndim == 3:
        mask = mask.unsqueeze(1)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=kwargs.get("scale"))
    return out if skip_output_reshape else out.transpose(1, 2).reshape(b, -1, heads * d)


def reference(q, k, v, heads, hnd):
    """SDPA in float32 (memory-efficient kernel: the math kernel would need N x N per head)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        return comfy_sdpa(q.float(), k.float(), v.float(), heads, skip_reshape=hnd)


def make_inputs(b, h, n, hnd, offset, device):
    gen = torch.Generator(device=device).manual_seed(0)
    shape = (b, h, n, HEAD_DIM) if hnd else (b, n, h * HEAD_DIM)
    q, k, v = (torch.randn(shape, device=device, generator=gen) for _ in range(3))
    if offset:
        bias = torch.randn(HEAD_DIM, device=device, generator=gen) * K_OFFSET
        k = k + bias if hnd else (k.unflatten(-1, (h, HEAD_DIM)) + bias).flatten(-2)
    return tuple(t.to(torch.bfloat16) for t in (q, k, v))


def metrics(out, ref):
    out = out.float()
    nonfinite = int((~torch.isfinite(out)).sum())
    err = (out - ref).abs()
    cos = F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item()
    return cos, err.max().item(), err.mean().item(), nonfinite


def time_ms(fn):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def table(headers, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    lines = [headers, ["-" * w for w in widths]] + list(rows)
    return "\n".join("  ".join(str(x).ljust(w) for x, w in zip(row, widths)).rstrip() for row in lines)


def short(mode):
    return mode.replace("sageattn_qk_int8_pv_", "").replace(" (experimental)", "")


def args_label(kwargs):
    parts = []
    if "qk_quant_gran" in kwargs:
        parts.append(kwargs["qk_quant_gran"])
    if "pv_accum_dtype" in kwargs:
        parts.append("pv " + kwargs["pv_accum_dtype"])
    if "smooth_k" in kwargs:
        parts.append("smooth_k " + ("on" if kwargs["smooth_k"] else "off"))
    if "per_block_mean" in kwargs:
        parts.append("per_block_mean " + ("on" if kwargs["per_block_mean"] else "off"))
    return ", ".join(parts)


def variants(actives):
    """(label, Active, is the node's own configuration) per row: the node's arguments, smooth_k
    flipped, the other qk_quant_gran; one row per distinct kernel + arguments."""
    seen, rows = set(), []
    for active in actives:
        options = [(active.kwargs, True)]
        if "smooth_k" in active.kwargs:
            options.append(({**active.kwargs, "smooth_k": not active.kwargs["smooth_k"]}, False))
        if active.kernel.takes_quant:
            other = "per_thread" if active.kwargs["qk_quant_gran"] == "per_warp" else "per_warp"
            options.append(({**active.kwargs, "qk_quant_gran": other}, False))
        for kwargs, node in options:
            key = (active.kernel.mode, tuple(sorted(kwargs.items())))
            if key in seen:
                continue
            seen.add(key)
            rows.append((f"{short(active.kernel.mode)} [{args_label(kwargs)}]" + (" node" if node else ""),
                         dataclasses.replace(active, kwargs=kwargs), node))
    return rows


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA GPU; this script measures the SageAttention kernels on one.")
        return 0
    logging.disable(logging.WARNING)  # the override's own console lines; this report prints what they say
    sa = bind_package("bcnodes_compare")["libs.sage_attention"]
    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    print(f"GPU: {torch.cuda.get_device_name(device)} (sm_{major}{minor}), {torch.cuda.get_device_properties(device).total_memory / 2**30:.1f} GiB")
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, sageattention {sa._installed('sageattention')}, "
          f"sageattn3 {sa._installed('sageattn3')}")

    print("\n== 1. Modes on this GPU (the node's resolve: package, version, compute capability, test call)\n")
    resolved = {mode: sa.resolve(mode, "auto", device) for mode in ("auto",) + tuple(sa.KERNELS)}
    print(table(["mode", "", "what the node runs, or why it refuses"],
                [[mode, "OK" if not isinstance(r, str) else "--", r.describe() if not isinstance(r, str) else r]
                 for mode, r in resolved.items()]))
    actives = [r for r in resolved.values() if not isinstance(r, str)]
    auto = resolved["auto"]
    rows = variants(actives)
    if not rows:
        print("\nNo kernel runs on this GPU; nothing to measure.")
        return 1

    print(f"\n== 2. Accuracy against SDPA float32, time per call ({ITERS} calls after a warm-up)")
    print("   'node' = the arguments the node uses for that mode; auto = "
          + (f"{short(auto.kernel.mode)} [{args_label(auto.kwargs)}]" if not isinstance(auto, str) else "refused here"))
    worst = {}
    for (name, b, h, n), hnd in [(s, True) for s in QWEN] + [(s, False) for s in WAN]:
        for offset in (False, True):
            q, k, v = make_inputs(b, h, n, hnd, offset, device)
            ref = reference(q, k, v, h, hnd)
            sdpa_ms = time_ms(lambda: comfy_sdpa(q, k, v, h, skip_reshape=hnd))
            out_rows = [["sdpa bf16 (ComfyUI default)", *fmt(metrics(comfy_sdpa(q, k, v, h, skip_reshape=hnd), ref)),
                         f"{sdpa_ms:.2f}", "1.00x", ""]]
            for label, active, _ in rows:
                try:
                    out = sa.attend(active, q, k, v, h, hnd, False, None)
                    m = metrics(out, ref)
                    del out
                    ms = time_ms(lambda: sa.attend(active, q, k, v, h, hnd, False, None))
                except Exception as e:  # a kernel that fails on this GPU is a row, not the end of the report
                    out_rows.append([label, "-", "-", "-", "-", "-", "-", f"error: {type(e).__name__}: {str(e)[:100]}"])
                    worst.setdefault(label, []).append(None)
                    continue
                out_rows.append([label, *fmt(m), f"{ms:.2f}", f"{sdpa_ms / ms:.2f}x", ""])
                worst.setdefault(label, []).append((m, sdpa_ms / ms))
            layout = "(B, H, N, D)" if hnd else "(B, N, H*D)"
            inputs = f"randn, K + per-channel offset (std {K_OFFSET:g})" if offset else "randn"
            print(f"\n-- {name}: {h} heads x {HEAD_DIM}, {n} tokens, bf16 {layout}; inputs: {inputs}\n")
            print(table(["kernel [arguments]", "cosine", "max abs", "mean abs", "non-finite", "ms/call", "vs sdpa", "note"], out_rows))
            del q, k, v, ref
            torch.cuda.empty_cache()

    print("\n== Worst case per row over every shape and input set\n")
    summary = []
    for label, results in worst.items():
        ok = [r for r in results if r is not None]
        errors = len(results) - len(ok)
        if not ok:
            summary.append([label, "-", "-", "-", "-", f"{errors} errors"])
            continue
        summary.append([label, f"{min(m[0] for m, _ in ok):.6f}", f"{max(m[1] for m, _ in ok):.4f}",
                        f"{max(m[3] for m, _ in ok)}", f"{min(s for _, s in ok):.2f}x - {max(s for _, s in ok):.2f}x",
                        f"{errors} errors" if errors else ""])
    print(table(["kernel [arguments]", "min cosine", "max abs", "non-finite", "speed vs sdpa", "errors"], summary))

    failures = fallback_checks(sa, auto, device)
    print(f"\n{'FAILED: ' + ', '.join(failures) if failures else 'All fallback checks passed.'}")
    fused_qkv_check(sa, auto, device)
    return 1 if failures else 0


def fmt(m):
    cos, max_abs, mean_abs, nonfinite = m
    return [f"{cos:.6f}", f"{max_abs:.4f}", f"{mean_abs:.6f}", str(nonfinite)]


def fallback_checks(sa, auto, device):
    """-> the names of the failed checks."""
    print("\n== 3. The override's fallbacks (auto kernel, Qwen-Image 4k)\n")
    if isinstance(auto, str):
        print(f"SKIP: auto refused here: {auto}")
        return []
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    _, b, h, n = QWEN[0]
    q, k, v = make_inputs(b, h, n, True, False, device)
    mask = torch.zeros(b, 1, n, dtype=q.dtype, device=device)
    mask[..., -256:] = -torch.finfo(q.dtype).max  # as Qwen-Image masks padded text tokens
    override = sa.SageOverride(auto, None, True, False)
    out = override(comfy_sdpa, q, k, v, h, mask, skip_reshape=True)
    check("a masked call runs the original attention", override.stats.fallbacks == {"mask": 1}
          and torch.equal(out, comfy_sdpa(q, k, v, h, mask, skip_reshape=True)), f"{override.stats.fallbacks}")

    def poisoned(*args, **kwargs):
        out = auto.call(*args, **kwargs)
        out[(0,) * out.ndim] = float("nan")
        return out

    guarded = sa.SageOverride(dataclasses.replace(auto, call=poisoned), None, True, False)
    out = guarded(comfy_sdpa, q, k, v, h, skip_reshape=True)
    check("a NaN in the kernel output runs the original attention (nan_guard on)",
          guarded.stats.fallbacks == {"non-finite output": 1} and bool(torch.isfinite(out).all())
          and torch.equal(out, comfy_sdpa(q, k, v, h, skip_reshape=True)), f"{guarded.stats.fallbacks}")
    unguarded = sa.SageOverride(dataclasses.replace(auto, call=poisoned), None, False, False)
    check("the same NaN comes through with nan_guard off (the injection works)",
          bool(torch.isnan(unguarded(comfy_sdpa, q, k, v, h, skip_reshape=True)).any()))
    plain = sa.SageOverride(auto, None, True, False)
    out = plain(comfy_sdpa, q, k, v, h, skip_reshape=True)
    check("an unmasked call runs the kernel", plain.stats.sage == 1 and plain.stats.fallbacks == {}, f"{plain.stats.fallbacks}")
    print("\nThe run summary the console shows for the NaN case: " + guarded.stats.summary(auto.label))
    del q, k, v, mask, out

    print("\n-- nan_guard cost (auto through the override, plain randn)\n")
    rows = []
    for (name, b, h, n), hnd in [(QWEN[0], True), (QWEN[-1], True), (WAN[-1], False)]:
        q, k, v = make_inputs(b, h, n, hnd, False, device)
        on, off = (sa.SageOverride(auto, None, guard, False) for guard in (True, False))
        ms_on = time_ms(lambda: on(comfy_sdpa, q, k, v, h, skip_reshape=hnd))
        ms_off = time_ms(lambda: off(comfy_sdpa, q, k, v, h, skip_reshape=hnd))
        rows.append([name, f"{ms_off:.2f}", f"{ms_on:.2f}", f"{ms_on - ms_off:+.2f}"])
        del q, k, v
        torch.cuda.empty_cache()
    print(table(["shape", "ms nan_guard off", "ms nan_guard on", "difference"], rows))
    return failures


def fused_qkv_check(sa, auto, device):
    """q / k / v as views into one fused QKV buffer, as a model with a fused QKV projection hands
    them over: the Triton per_thread quant computes offsets in int32 (thu-ml/SageAttention#386)."""
    stride = 3 * FUSED_HEADS * HEAD_DIM
    print(f"\n== 4. Fused QKV views, auto kernel (sequence stride {stride}; int32 offsets pass 2**31 "
          f"from {2**31 // stride + 1} tokens)")
    if isinstance(auto, str) or not auto.kernel.takes_quant:
        print(f"\nSKIP: auto takes no qk_quant_gran here: {auto if isinstance(auto, str) else auto.label}")
        return
    for n in FUSED_LENGTHS:
        gen = torch.Generator(device=device).manual_seed(0)
        fused = torch.randn(1, n, stride, device=device, generator=gen, dtype=torch.bfloat16)
        views = fused.chunk(3, dim=-1)
        ref = reference(*views, FUSED_HEADS, False)
        copies = tuple(t.contiguous() for t in views)
        rows = []
        for gran in ("per_thread", "per_warp"):
            active = dataclasses.replace(auto, kwargs={**auto.kwargs, "qk_quant_gran": gran})
            for name, qkv in (("views", views), ("contiguous", copies)):
                try:
                    out = sa.attend(active, *qkv, FUSED_HEADS, False, False, None)
                    rows.append([gran, name, *fmt(metrics(out, ref)), ""])
                    del out
                except Exception as e:  # an illegal access is a row; the CUDA context is gone after it
                    rows.append([gran, name, "-", "-", "-", "-", f"error: {type(e).__name__}: {str(e)[:100]}"])
        node = sa.SageOverride(auto, None, True, False)
        try:
            out = node(comfy_sdpa, *views, FUSED_HEADS)
            ran = (f"per_warp ({node.stats.per_warp} switched)" if node.stats.per_warp
                   else auto.kwargs["qk_quant_gran"] if node.stats.sage else f"original attention {node.stats.fallbacks}")
            rows.append(["node", "views", *fmt(metrics(out, ref)), f"ran {ran}"])
            del out
        except Exception as e:
            rows.append(["node", "views", "-", "-", "-", "-", f"error: {type(e).__name__}: {str(e)[:100]}"])
        print(f"\n-- {FUSED_HEADS} heads x {HEAD_DIM}, {n} tokens, bf16 (B, N, H*D), randn\n")
        print(table(["qk_quant_gran", "q / k / v", "cosine", "max abs", "mean abs", "non-finite", "note"], rows), flush=True)
        del fused, views, copies, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    started = time.monotonic()
    code = main()
    print(f"\n({time.monotonic() - started:.0f} s)")
    sys.exit(code)

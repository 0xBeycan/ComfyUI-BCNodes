# Model

## `BC_PatchSageAttention` — Patch Sage Attention

`model` (`MODEL`) + `mode` + `qk_quant` + `nan_guard` + `log_level` → `MODEL`: a clone of the model whose attention runs one SageAttention kernel. The input model is never changed.

### One model, one workflow

ComfyUI's `--use-sage-attention` flag switches every model in every workflow to SageAttention, and on Ada and Blackwell cards (RTX 4090, RTX 5090, RTX PRO 4500 / 6000 Blackwell) the `sageattn` it calls picks the fp8-PV kernel. This node patches only the model that goes through it, through ComfyUI's per-model attention override (`transformer_options["optimized_attention_override"]`, ComfyUI 0.3.60 or newer; an older ComfyUI stops with an error saying so). The models beside it, the text encoder, the VAE and every other workflow keep ComfyUI's attention. Nothing global is patched: the patch is on the clone and goes away with it. Start ComfyUI without `--use-sage-attention`.

Every model that calls ComfyUI's attention entry points takes the patch: Qwen-Image (Edit included), Wan, LTX, Flux and the others. Put the node between the model loader (and its LoRA loaders) and the sampler.

### Modes

| `mode` | Kernel | GPUs (compute capability) |
| --- | --- | --- |
| `disabled` | none: the clone, unpatched, for an A/B in the same workflow | any |
| `auto` (default) | `sageattn_qk_int8_pv_fp16_cuda`: INT8 Q·K, fp16 P·V accumulated in fp32 | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120, sm_121: Ampere, Ada, Hopper, Blackwell (RTX 3090, 4090, 5090, RTX PRO Blackwell, A100, H100, B200) |
| `sageattn_qk_int8_pv_fp16_cuda` | the same kernel as `auto` | the same |
| `sageattn_qk_int8_pv_fp16_triton` | INT8 Q·K, fp16 P·V, in Triton | sm_80 to sm_100; not sm_120 / sm_121, where the official code calls the Triton kernel unusable |
| `sageattn_qk_int8_pv_fp8_cuda` | INT8 Q·K, fp8 P·V accumulated in fp32 into an fp32 buffer (SageAttention2) | sm_89 and newer |
| `sageattn_qk_int8_pv_fp8_cuda++` | INT8 Q·K, fp8 P·V accumulated in fp16 into the fp32 buffer (SageAttention2++: what the global flag runs on Ada and Blackwell) | sm_89 and newer |
| `sageattn3 (experimental)` | SageAttention3, FP4, from the separate `sageattn3` package | sm_100, sm_120, sm_121 |

`auto` is conservative: it takes the most accurate kernel and nothing else, and never falls through to Triton, fp8 P·V or SageAttention3. Those trade accuracy for speed and are reported to give black images or lose detail and identity with Qwen-Image and Wan, where the output matters more than the time saved. A GPU with no compute capability in `auto`'s row (Turing and older) keeps ComfyUI's attention.

The kernels get the official arguments (thu-ml/SageAttention 2.2.0): `smooth_k` on (K's mean over the tokens is taken out before quantization, which softmax does not see; ComfyUI's own sage path turns it off), and `qk_quant_gran` from `qk_quant` (below). `sageattn3` runs with its default `per_block_mean`.

Before patching, the node checks the package and its version (sageattention 1.x is refused: the plain `pip install sageattention`, 1.0.6, is known to give black images), that the function exists, the GPU's compute capability, and runs one small test call on the GPU, whose output must be finite and within cosine 0.98 of torch SDPA (every kernel is far above it: INT8 Q·K 0.9992 or more on an RTX 5090, SageAttention3's FP4 0.9955 in its paper; a broken kernel is far below). If any of it fails, the model keeps ComfyUI's attention, the console says why, and the workflow runs on.

### `qk_quant`

How Q and K are quantized to INT8 before Q·K, for the kernels that take it: `auto`, `sageattn_qk_int8_pv_fp16_cuda`, `sageattn_qk_int8_pv_fp8_cuda` and `sageattn_qk_int8_pv_fp8_cuda++`. The Triton kernel and SageAttention3 ignore it.

| `qk_quant` | Quantization |
| --- | --- |
| `auto` (default) | `per_thread` when its test call passes on this GPU, else `per_warp`; the console's kernel line shows which, and why `per_thread` was refused |
| `per_thread` | one scale for the tokens each GPU thread multiplies (8 groups per 32 query tokens, 4 groups per 64 key tokens), quantized by a Triton kernel |
| `per_warp` | one scale per warp's query tokens and one per 64-key block, quantized by a CUDA kernel |

A named one that fails its test call leaves the model on ComfyUI's attention; the node does not switch to the other on its own.

Every `per_thread` group lies inside one `per_warp` group, so its scale is never larger and its rounding error never has a wider bound; the SageAttention authors call per-thread "the most accurate" (thu-ml/SageAttention#94). Measured on an RTX 5090 (torch 2.11 with CUDA 13.0, Triton 3.6, sageattention 2.2.0) against torch SDPA in fp32, `sageattn_qk_int8_pv_fp16_cuda` at Qwen-Image and Wan shapes: cosine 0.999934 against 0.999919 and about 10% lower mean error with `per_thread`, at every shape. Per call, `per_thread` was up to ~20% slower at 4k tokens and within ~4% from 8k tokens on. The official `sageattn` picks `per_warp` on sm_120 / sm_121 because the Triton kernel did not run there when that line was written (February 2025; Triton 3.3, April 2025, added Blackwell); it runs there now.

### Long sequences

The kernels compute each element's position as a 32-bit integer counted from the tensor's start: the CUDA kernels in uint32 (up to 2^32), the Triton ones, `per_thread`'s quantization included, in int32 (up to 2^31). Past that the position wraps and the output is wrong without a NaN, so `nan_guard` does not see it (thu-ml/SageAttention#386). The largest position is the last element's, about tokens × the token stride, and q, k, v cut from one fused QKV projection (Flux and others) have three times the stride of separate ones: 56 heads × 128, fused, pass 2^31 at 99,865 tokens. The node reads it from each call's shapes and strides, with no GPU work:

- `per_thread` at 2^31 or more: that call runs `per_warp`; the run summary counts these calls, and the first one in a run is logged.
- At the kernel's own limit (2^32 for the CUDA kernels, 2^31 for the Triton kernel): the call runs the original attention, reason `offsets past 2**32` (or `2**31`).

SageAttention3's indexing was not read, so no limit is applied to it. On the RTX 5090, fused views at 110k tokens gave cosine 0.941 with `per_thread` and 0.99992 with `per_warp`.

### Calls that run the original attention

With a kernel active, a call runs the original attention (the override the model had before this node, if any, else ComfyUI's attention for that call) when the console's reason is:

| Reason | Why |
| --- | --- |
| `low_precision_attention=False` | ComfyUI marks the calls where SageAttention is known to fail (Wan's image cross-attention, LTX's guide masks, audio models) |
| `mask` | the CUDA kernels take no mask and would drop one silently |
| `attn_precision float32` | the model asks for fp32 attention (SD 2.x, SVD, HunyuanDiT) |
| `enable_gqa` | grouped-query attention |
| `not on CUDA` | the tensors are not on a CUDA GPU |
| `dtype …`, `mixed dtypes` | the kernels take fp16 or bf16 |
| `head_dim …` | a head size over 128 (SageAttention3: 256 and over) |
| `custom scale` | SageAttention3 only: its softmax scale is fixed |
| `offsets past 2**32`, `offsets past 2**31` | the tensors reach past the kernel's 32-bit indexing (see Long sequences) |
| `kernel error …` | the kernel raised; a warning with its message |
| `non-finite output` | `nan_guard` on and the output holds a NaN or inf: the call runs again with the original attention; a warning |

Qwen-Image's text encoder drops its mask when no token of the prompt is padding, so a prompt encoded on its own brings none and its calls run the kernel; a mask comes in when texts of different lengths are padded into one batch. The run summary counts the calls it hits.

### `nan_guard`

On (default): every kernel output is checked with `torch.isfinite`, and a call with a NaN or inf runs again with the original attention. The check reads one value back from the GPU, so it costs one GPU sync per attention call; `tests/compare_sage_attention.py` measures that cost on your card. The fp16-PV kernels turn V into fp16 even for a bf16 model, so a value beyond ±65504 becomes inf: the guard catches that call. Off: no sync, and a NaN goes into the result.

### Console

- At the start of each sampling run, the kernel, its arguments, the GPU, its compute capability, the package version and `nan_guard`:
  `[BCNodes] Patch Sage Attention: auto (sageattn_qk_int8_pv_fp16_cuda) [pv_accum_dtype=fp32, smooth_k=True, qk_quant_gran=per_thread] on NVIDIA GeForce RTX 5090 (sm_120), sageattention 2.2.0, nan_guard on`
- A call that runs the original attention, or a `per_thread` call that runs `per_warp` (Long sequences): with `log_level` `once`, one line per reason per run; with `verbose`, one line per call.
- At the end of each run (also when the run fails), every call counted:
  `[BCNodes] Patch Sage Attention: run summary: 1240 attention calls, 1200 ran auto (sageattn_qk_int8_pv_fp16_cuda) (30 of them per_warp: offsets past 2**31), 40 ran the original attention (low_precision_attention=False: 40)`.
  A run in which no call ran the kernel is a warning, so a reason that hits every call (a mask on every Qwen-Image call, a dtype) stands out.

### Install

SageAttention 2.2.x built for the torch and CUDA ComfyUI runs on: `pip install sageattention==2.2.0 --no-build-isolation` (it compiles against the installed torch, so the CUDA toolkit has to be there), a prebuilt wheel made for exactly that torch, CUDA and Python, or a build from [the source](https://github.com/thu-ml/SageAttention) (`python setup.py install`). Blackwell needs CUDA 12.8 or newer. Not the plain `pip install sageattention`: that is 1.0.6, which the node refuses. The `sageattn3 (experimental)` mode needs the `sageattn3` package too, built from the same repository's `sageattention3_blackwell/` (Python 3.13+, torch 2.8+, CUDA 12.8+). It is not in this pack's `requirements.txt`.

### A/B test

Two branches from the same model loader: one Patch Sage Attention on `disabled`, one on `auto`, each feeding its own sampler with the same seed, sampler, scheduler, steps, prompt and image inputs. Compare the two results side by side (Image Comparer), and read the run summary of the `auto` branch: it says how many attention calls ran the kernel. One branch works too: switch `mode` between two queues of the same workflow.

To measure the kernels themselves on a GPU (accuracy against torch SDPA at Qwen-Image and Wan shapes, `smooth_k` on and off, `per_thread` against `per_warp`, time per call, the fallbacks, the cost of `nan_guard`, and fused QKV views past 2^31), run `python tests/compare_sage_attention.py` from the pack's directory; it needs no ComfyUI.

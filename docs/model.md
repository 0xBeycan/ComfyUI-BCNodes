# Model

## `BC_PatchSageAttention` — Patch Sage Attention

`model` (`MODEL`) + `mode` + `nan_guard` + `log_level` → `MODEL`: a clone of the model whose attention runs one SageAttention kernel. The input model is never changed.

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

The kernels get the official arguments (thu-ml/SageAttention 2.2.0): `smooth_k` on (K's mean over the tokens is taken out before quantization, which softmax does not see; ComfyUI's own sage path turns it off), and `qk_quant_gran` `per_thread`, except `per_warp` on sm_120 / sm_121, as the official dispatch does there (`per_thread` quantizes through a Triton kernel). `sageattn3` runs with its default `per_block_mean`.

Before patching, the node checks the package and its version (sageattention 1.x is refused: the plain `pip install sageattention`, 1.0.6, is known to give black images), that the function exists, the GPU's compute capability, and runs one small test call on the GPU. If any of it fails, the model keeps ComfyUI's attention, the console says why, and the workflow runs on.

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
| `kernel error …` | the kernel raised; a warning with its message |
| `non-finite output` | `nan_guard` on and the output holds a NaN or inf: the call runs again with the original attention; a warning |

Qwen-Image's text encoder drops its mask when no token of the prompt is padding, so a prompt encoded on its own brings none and its calls run the kernel; a mask comes in when texts of different lengths are padded into one batch. The run summary counts the calls it hits.

### `nan_guard`

On (default): every kernel output is checked with `torch.isfinite`, and a call with a NaN or inf runs again with the original attention. The check reads one value back from the GPU, so it costs one GPU sync per attention call; `tests/compare_sage_attention.py` measures that cost on your card. The fp16-PV kernels turn V into fp16 even for a bf16 model, so a value beyond ±65504 becomes inf: the guard catches that call. Off: no sync, and a NaN goes into the result.

### Console

- At the start of each sampling run, the kernel, its arguments, the GPU, its compute capability, the package version and `nan_guard`:
  `[BCNodes] Patch Sage Attention: auto (sageattn_qk_int8_pv_fp16_cuda) [pv_accum_dtype=fp32, smooth_k=True, qk_quant_gran=per_warp] on NVIDIA GeForce RTX 5090 (sm_120), sageattention 2.2.0, nan_guard on`
- A call that runs the original attention: with `log_level` `once`, one line per reason per run; with `verbose`, one line per call.
- At the end of each run (also when the run fails), every call counted:
  `[BCNodes] Patch Sage Attention: run summary: 1240 attention calls, 1200 ran auto (sageattn_qk_int8_pv_fp16_cuda), 40 ran the original attention (low_precision_attention=False: 40)`.
  A run in which no call ran the kernel is a warning, so a reason that hits every call (a mask on every Qwen-Image call, a dtype) stands out.

### Install

SageAttention 2.2.x built for the torch and CUDA ComfyUI runs on: `pip install sageattention==2.2.0 --no-build-isolation` (it compiles against the installed torch, so the CUDA toolkit has to be there), a prebuilt wheel made for exactly that torch, CUDA and Python, or a build from [the source](https://github.com/thu-ml/SageAttention) (`python setup.py install`). Blackwell needs CUDA 12.8 or newer. Not the plain `pip install sageattention`: that is 1.0.6, which the node refuses. The `sageattn3 (experimental)` mode needs the `sageattn3` package too, built from the same repository's `sageattention3_blackwell/` (Python 3.13+, torch 2.8+, CUDA 12.8+). It is not in this pack's `requirements.txt`.

### A/B test

Two branches from the same model loader: one Patch Sage Attention on `disabled`, one on `auto`, each feeding its own sampler with the same seed, sampler, scheduler, steps, prompt and image inputs. Compare the two results side by side (Image Comparer), and read the run summary of the `auto` branch: it says how many attention calls ran the kernel. One branch works too: switch `mode` between two queues of the same workflow.

To measure the kernels themselves on a GPU (accuracy against torch SDPA at Qwen-Image and Wan shapes, `smooth_k` on and off, time per call, the fallbacks and the cost of `nan_guard`), run `python tests/compare_sage_attention.py` from the pack's directory; it needs no ComfyUI.

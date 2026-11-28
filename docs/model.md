# Model

The model nodes patch a model between its loader (and its LoRA loaders) and the sampler. Each of them but DyPE Config returns a patched clone and leaves the input model unchanged. [Patch Sage Attention](#bc_patchsageattention--patch-sage-attention) runs this model's attention, and no other's, on one SageAttention kernel. [Detail Boost](#bc_detailboost--detail-boost) tells the model a lower noise level than the sampler's in a range of steps, so it keeps more fine detail. [Qwen Image Shift](#bc_qwenimageshift--qwen-image-shift) gives a Qwen-Image model the official Qwen-Image-2512 flow shift. [Dual Sampling Shift](#bc_dualsamplingshift--dual-sampling-shift) gives a flow model one shift for a run from full noise and another for a refine pass. [DyPE Patcher](#bc_dypepatcher--dype-patcher) puts DyPE on the rotary positions of a Z-Image or Qwen-Image model, against the scene repeating itself above the training resolution; [DyPE Config](#bc_dypeconfig--dype-config) sets its values for DyPE Patcher and Dual Sampler.

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
| `sageattn_qk_int8_pv_fp16_triton` | INT8 Q·K, fp16 P·V, in Triton | sm_80, sm_86, sm_89, sm_90, sm_100; not sm_120 / sm_121, where the official code calls the Triton kernel unusable |
| `sageattn_qk_int8_pv_fp8_cuda` | INT8 Q·K, fp8 P·V accumulated in fp32 into an fp32 buffer (SageAttention2) | sm_89, sm_90, sm_100, sm_120, sm_121 |
| `sageattn_qk_int8_pv_fp8_cuda++` | INT8 Q·K, fp8 P·V accumulated in fp16 into the fp32 buffer (SageAttention2++: what the global flag runs on Ada and Blackwell) | sm_89, sm_90, sm_100, sm_120, sm_121 |
| `sageattn3 (experimental)` | SageAttention3, FP4, from the separate `sageattn3` package | sm_100, sm_120, sm_121 |

`auto` is conservative: it takes the most accurate kernel and nothing else, and never falls through to Triton, fp8 P·V or SageAttention3. Those trade accuracy for speed and are reported to give black images or lose detail and identity with Qwen-Image and Wan, where the output matters more than the time saved. Each kernel runs only on the compute capabilities of its row, taken from the official code: on any other GPU (Turing and older, or a capability not listed) the model keeps ComfyUI's attention and the console says which capabilities the kernel runs on.

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

SageAttention 2.2.x built for the torch and CUDA ComfyUI runs on: a prebuilt wheel made for exactly that torch, CUDA and Python, or a build from [the source](https://github.com/thu-ml/SageAttention) (`python setup.py install`; it compiles against the installed torch, so the CUDA toolkit has to be there). Blackwell needs CUDA 12.8 or newer. Not the plain `pip install sageattention`: that is a 1.x release (1.0.6), which the node refuses. The `sageattn3 (experimental)` mode needs the `sageattn3` package too, built from the same repository's `sageattention3_blackwell/` (Python 3.13+, torch 2.8+, CUDA 12.8+). Neither package is in this pack's `requirements.txt`.

The `per_thread` quantization (which `qk_quant` `auto` tries first) and the `sageattn_qk_int8_pv_fp16_triton` kernel are Triton kernels, so both need Triton in ComfyUI's Python. The pack imports neither Triton nor SageAttention when it loads: the node imports `sageattention` when it runs, before patching. An import that fails is reported as `sageattention <version> has no usable <function> (<the error>)`, and the model keeps ComfyUI's attention. With `qk_quant` `auto`, a `per_thread` test call that fails moves to `per_warp` (the `qk_quant` table).

### A/B test

Two branches from the same model loader: one Patch Sage Attention on `disabled`, one on `auto`, each feeding its own sampler with the same seed, sampler, scheduler, steps, prompt and image inputs. Compare the two results side by side (Image Comparer), and read the run summary of the `auto` branch: it says how many attention calls ran the kernel. One branch works too: switch `mode` between two queues of the same workflow.

To measure the kernels themselves on a GPU (accuracy against torch SDPA at Qwen-Image and Wan shapes, `smooth_k` on and off, `per_thread` against `per_warp`, time per call, the fallbacks, the cost of `nan_guard`, and fused QKV views past 2^31), run `python tests/compare_sage_attention.py` from the pack's directory; it needs no ComfyUI.

## `BC_DetailBoost` — Detail Boost

`model` (`MODEL`) + `weight` + `mode` + `eta` + `start_step` + `end_step` → `MODEL`: a clone of the model that is told a lower noise level than the sampler's in a range of steps. The input model is never changed.

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `weight` | 1.0 | −10 to 10, step 0.01 | the blend toward the lowered sigma (`weight` and `eta` below); 0: the model is not patched |
| `mode` | `hard` | the 11 modes below | how much noise the model is told is gone |
| `eta` | 0.5 | 0 to 10, step 0.01 | the mode's strength; 0: the model is not patched |
| `start_step` | 3 | 0 to 10000 | the first step boosted, counted from 0 in each sampler run |
| `end_step` | 10 | −1 to 10000 | the step the boost stops at, not boosted itself; −1: to the end of the run |

An `end_step` at or below `start_step` (other than −1) stops the node with an error.

### What it does

A sampler calls the model at each step's sigma and steps with the prediction it gets back. In the steps `start_step` to `end_step` the node tells the model a lower sigma: the model takes the latent for less noisy than it is and removes less noise, so the fine structure a step would smooth away (skin pores, fine hair, stone grain) stays. The sampler's own sigmas, its noise and its steps are unchanged, and nothing is added to the latent. That is the difference from raising `s_noise` on an SDE sampler, which adds noise the model then has to remove again.

Put the node between the model loader (and its LoRA loaders) and the sampler: KSampler or KSamplerAdvanced, no custom sampler. In a two-pass workflow give it only to the pass that should get the detail (for example the second pass at denoise 0.45); each pass is a run of its own.

### Steps

Steps are counted from 0 in each sampler run; with denoise below 1 they are the run's own steps (20 steps at denoise 0.45: steps 0 to 19). `end_step` is not boosted itself; -1 boosts to the end of the run. The run's very first model call keeps its sigma: it has no schedule sigma before it. Samplers that call the model between two schedule points (dpmpp_sde, heun, the 2s samplers) get every call of a boosted step boosted; euler and the 2m samplers call the model once per step.

### `weight` and `eta`

`eta` is the mode's strength (no noise is added); `weight` blends: the model is told sigma + weight × (lowered sigma − sigma). 1 is the mode's full effect; negative values move the other way (the model is told more noise: softer, smoother). `weight` 0 or `eta` 0 leaves the model unpatched: the output is bit-identical to the model without the node.

`hard` on a flow model (Qwen-Image, Flux, Wan) multiplies the noise-to-signal ratio sigma / (1 − sigma) the model is told by √(1 − eta²). At `eta` 0.5 and `weight` 1:

| Sampler's sigma | Sigma the model is told |
| --- | --- |
| 0.95 | 0.943 |
| 0.60 | 0.565 |
| 0.20 | 0.178 |
| 0.05 | 0.044 |

From `eta` 1 on, `hard` is capped: the ratio is multiplied by 0.014, and the model is told almost no noise.

### Modes

Each mode takes noise out of the call's sigma: an amount of ratio × base, the base being the call's sigma unless the table says otherwise. On a flow model the noise that is left is renormalised by the signal (x = (1 − sigma) x0 + sigma noise); on any other model the sigma itself is what is left. `sigma` is the call's sigma and `sigma_prev` the schedule sigma before it, as the sampler has them; `s` and `p` are the two divided by the model's sigma_max (1 on a flow model, 14.6 on SD 1.x and SDXL). `hard_var`'s base and `vpsde` take `sigma` and `sigma_prev` as they are.

| `mode` | Ratio | Where it acts |
| --- | --- | --- |
| `hard` (default) | eta | every boosted step alike |
| `soft` | eta (1 + s / p) | about twice `hard` |
| `softer` | 1 − √(1 − eta² (p² − s²) / p²) | grows with the step's size |
| `soft-linear` | 1 − eta (s − p) | above 1 on a falling schedule: on a flow model capped, `hard`'s result below `eta` 1; on another model no effect |
| `sinusoidal` | eta sin²(π s) | strongest at s 0.5, the middle steps; nothing near 0 and 1 |
| `eps` | eta (s / p) √(p² − s²) | grows with the step's size |
| `exp` | √(1 − (s / p)^(2 eta)) | grows with the step's size and `eta` |
| `lorentzian` | eta, of the base sigma_max · s / √(s² + 1) | weaker at high sigma |
| `hard_var` | eta, of the base √(sigma_prev − sigma) | only the late steps, where s ≤ (√(1 + 4p) − 1) / 2 |
| `er4` | √(s² − p² (n(s) / n(p))²) / s, with n(v) = v (e^(v^eta) + 10) | `eta` is an exponent |
| `vpsde` | none: the model is told sigma − eta / 4 · sigma_prev (1 − sigma_prev)(sigma_prev − sigma) | a flow-model formula, a small cut there; on another model a `sigma_prev` over 1 raises the sigma |

On a flow model an amount at or over the call's sigma is capped to eta × sigma (0.9999 × sigma from `eta` 1 on). A formula with no real answer (a negative square root, an amount over the sigma of another model) leaves the call's sigma as it is.

### Other patches

The node puts a `PREDICT_NOISE` wrapper on the clone (ComfyUI's patcher wrappers). ComfyUI chains wrappers, so the model's other wrappers and a `model_function_wrapper` it already has stay, and run inside it with the sigma the model is told; two Detail Boost nodes in a row both apply. With an inpaint noise mask, the masked area is re-noised at the sampler's own sigma (that happens before the wrapper).

### A/B test

`weight` 0 is the model without the node: queue the same seed, sampler and steps with `weight` 0 and with your value, and compare the two images side by side (Image Comparer).

## `BC_QwenImageShift` — Qwen Image Shift

`model` (`MODEL`) + `mode` + `shift` + `multiplier` + `shift_terminal` + `terminal` + `base_seq_len` + `max_seq_len` + `base_shift` + `max_shift` + `refine_threshold` + `refine_rate` → `MODEL`: a clone with the flow shift of the official Qwen-Image-2512 scheduler, for the stock KSampler and schedulers. Use it in place of ModelSamplingAuraFlow. The input model is never changed.

| Widget | Default | Range | Used by |
| --- | --- | --- | --- |
| `mode` | `dynamic` | `dynamic`, `fixed` | every run (`mode` below) |
| `shift` | 3.0 | 0 to 100, step 0.01 | `fixed` mode |
| `multiplier` | 1.0 | 0.01 to 100, step 0.01 | `dynamic` mode: the shift is e^mu × `multiplier` |
| `shift_terminal` | on | | every run (`shift_terminal` below) |
| `terminal` | 0.02 | 0.001 to 0.999, step 0.001 | `shift_terminal` on: the last sigma before 0 |
| `base_seq_len` | 256 | 1 to 1000000 | `dynamic` mode: the tokens that get `base_shift` |
| `max_seq_len` | 8192 | 1 to 1000000 | `dynamic` mode: the tokens that get `max_shift` |
| `base_shift` | 0.5 | 0 to 100, step 0.01 | `dynamic` mode: mu at `base_seq_len` |
| `max_shift` | 0.9 | 0 to 100, step 0.01 | `dynamic` mode: mu at `max_seq_len` |
| `refine_threshold` | 10816 | 0 to 1000000 | `dynamic` mode, refine passes (below) |
| `refine_rate` | 0.15 | −1 to 2, step 0.01 | `dynamic` mode, refine passes (below) |

`refine_threshold` and `refine_rate` are optional inputs, so a prompt saved before they were added runs with their defaults. In `dynamic` mode a `max_seq_len` equal to `base_seq_len` stops the node with an error: the two points of the shift line must differ.

### What it does

The official pipeline (diffusers' QwenImagePipeline with FlowMatchEulerDiscreteScheduler, the config in `Qwen/Qwen-Image-2512/scheduler`) shifts its schedule by the image size: it counts the latent's tokens (2×2 patches: (width / 16) × (height / 16)), puts mu on the line through (`base_seq_len`, `base_shift`) and (`max_seq_len`, `max_shift`), and shifts with e^mu, the same formula ModelSamplingAuraFlow uses with its `shift`. The line is not clamped: past 8192 tokens the shift keeps rising. Then it stretches the schedule so its last sigma before 0 is `terminal` (0.02).

| Image | Tokens | Shift | 1.25x latent (w × h) | Tokens | Own shift | As a refine pass (defaults) |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 × 1024 | 4096 | 2.001 | 160 × 160 | 6400 | 2.247 | 2.247 |
| 1328 × 1328 | 6889 | 2.303 | 208 × 208 | 10816 | 2.807 | 2.807 |
| 1664 × 928 | 6032 | 2.206 | 260 × 145 | 9490 | 2.626 | 2.626 |
| 1472 × 1104 | 6348 | 2.241 | 230 × 172 | 9890 | 2.679 | 2.679 |

### `mode`

- `dynamic` (default): each sampler run takes its shift from the latent it gets, e^mu × `multiplier`. One patched model gives each pass its own shift: in a two-pass workflow the 1.25x upscaled second pass gets a larger one (a refine pass above `refine_threshold` tokens: see below). The console prints each run's latent size, refine state, tokens, mu and shift.
- `fixed`: `shift` for every run. With `shift_terminal` off the model is exactly ModelSamplingAuraFlow at that shift (every scheduler gives the same sigmas). Qwen-Image-Lightning was distilled with shift 3 and no terminal: `fixed`, 3, `shift_terminal` off.

`multiplier`, `base_seq_len`, `max_seq_len`, `base_shift`, `max_shift`, `refine_threshold` and `refine_rate` are used in `dynamic` mode only; the four line values' defaults are the official config (256, 8192, 0.5, 0.9).

### Refine passes (`refine_threshold`, `refine_rate`)

A refine pass is a sampler run whose schedule starts below sigma 1 (denoise below 1, or a later start step) on new noise: a hires or latent-upscaled second pass, Dual Sampler's second pass, a detailer's crop. At the latent's own shift a large one rewrites the structure: a 2x latent upscale of 1024 × 1536 (24576 tokens) at denoise 0.45 gets 5.62. So a refine pass of more than `refine_threshold` tokens (default 10816: 1328 × 1328, the largest official size, upscaled 1.25x to 1664 × 1664; every official size upscaled 1.25x is at or below it, every one upscaled 1.5x above it) takes mu `refine_rate` of the way from the threshold's mu to its own:

mu = mu(`refine_threshold`) + `refine_rate` × (mu(tokens) − mu(`refine_threshold`)), shift = e^mu × `multiplier`

mu is linear in the tokens, so the node computes it as the mu of `refine_threshold` + `refine_rate` × (tokens − `refine_threshold`) tokens, one point on the same line: `refine_rate` 1 is the latent's own shift and 0 the threshold's, bit for bit. `shift_terminal` stretches the result as in any run.

| 1024 × 1536 upscaled | Tokens | Own shift | `refine_rate` 0.15 (default) | 0 | −1 |
| --- | --- | --- | --- | --- | --- |
| 1.25x | 9600 | 2.640 | 2.640 (below the threshold: own) | 2.640 | 2.640 |
| 1.5x | 13824 | 3.267 | 2.872 | 2.807 | 2.412 |
| 2x | 24576 | 5.617 | 3.115 | 2.807 | 1.403 |

Unchanged: every run from full noise (a schedule starting at sigma 1), a refine pass at or below `refine_threshold` tokens, and `fixed` mode. The later part of a split run is no refine pass: KSamplerAdvanced with `add_noise` off (or SamplerCustomAdvanced with DisableNoise) gives the run all-zero noise, which adds nothing to the latent, so one generation split in two keeps one shift. A later part with `add_noise` on adds new noise and is a refine pass. Neither the denoise nor `add_noise` reaches the shift; the node reads both from the run (its first sigma, its noise).

The console line of a refine pass above the threshold: `latent 256x384, refine: yes, tokens 24576, threshold 10816, rate 0.15, mu 1.1363, shift 3.1152`.

### Schedulers in `dynamic` mode

A sampler builds its sigmas before any model patch sees the latent, so the node keeps the model's shift at 1 outside a run (the scheduler gives its plain timestep grid) and shifts that grid when the run starts (an `OUTER_SAMPLE` wrapper, ComfyUI's patcher wrappers). During the run the model's `model_sampling` has the run's shift (ControlNet and conditioning step ranges follow it), and 1 again after.

| Scheduler | Result in `dynamic` mode |
| --- | --- |
| `simple`, `beta`, `ddim_uniform` | exactly the scheduler at the run's shift (ModelSamplingAuraFlow at that shift gives the same); `simple` is the official schedule when the step count divides 1000 (8, 20, 25, 40, 50) |
| `normal`, `sgm_uniform` | the grid's end comes from shift 1 (0.001 instead of about 0.002): the last steps differ slightly from ModelSamplingAuraFlow at that shift |
| `karras`, `exponential`, `kl_optimal` | the scheduler's shift-1 curve, shifted: not the curve ModelSamplingAuraFlow's shift would give these schedulers |
| `linear_quadratic` | shifted here; with ModelSamplingAuraFlow it ignores the shift |

In `fixed` mode every scheduler is exactly ModelSamplingAuraFlow's.

### `shift_terminal`

On (default, as the official config): each run's schedule is stretched so its last sigma before 0 is `terminal`. At denoise below 1 the run is the tail of a longer schedule with the same last sigma, so the stretch is the same. A run that does not end at sigma 0 (KSamplerAdvanced returning with leftover noise, the first part of split sigmas) or a one-step run from full noise has nothing to stretch to: the node stops with an error; turn `shift_terminal` off for it. Off for Lightning.

### Other patches

The node replaces the model's `model_sampling`, as ModelSamplingAuraFlow does, and puts an `OUTER_SAMPLE` wrapper on the clone in `dynamic` mode or with `shift_terminal` on (fixed mode without it needs none). ComfyUI chains wrappers, so the model's other wrappers stay; a second Qwen Image Shift replaces the first one's. A node after it that replaces `model_sampling` again (ModelSamplingAuraFlow, ModelSamplingSD3) would have the shift applied twice: the run stops with an error saying to keep one of the two.

## `BC_DualSamplingShift` — Dual Sampling Shift

`model` (`MODEL`) + `shift_1` + `shift_2` → `MODEL`: a clone of a flow model with two shifts, `shift_1` for a sampler run from full noise and `shift_2` for a refine pass. Use it in place of ModelSamplingAuraFlow or ModelSamplingSD3. The input model is never changed.

| Widget | Default | Range | Used by |
| --- | --- | --- | --- |
| `shift_1` | 3.0 | 0 to 100, step 0.01 | a run from full noise (denoise 1), and the later part of a split run |
| `shift_2` | 3.0 | 0 to 100, step 0.01 | a refine pass: a run on new noise that starts below full noise |

### What it does

The clone's `model_sampling` is a new object at `shift_1`. For a ModelSamplingDiscreteFlow model it is of the model's own `model_sampling` class (its prediction type kept), with the model's multiplier and noise scale, built as ModelSamplingSD3 builds it. For a ModelSamplingFlux model (Qwen-Image, Flux) it is the object ModelSamplingAuraFlow puts on it (see Models below). A run from full noise is handed to the sampler unchanged: with every scheduler its sigmas are exactly ModelSamplingAuraFlow's at `shift_1` (multiplier 1: Z-Image, Qwen-Image, Flux) or ModelSamplingSD3's (multiplier 1000: Wan, SD3).

A sampler builds its sigmas before any model patch sees the run, so a refine pass gets `shift_1`'s sigmas too. An `OUTER_SAMPLE` wrapper (ComfyUI's patcher wrappers) maps them to `shift_2` when the run starts. It takes t of each sigma from `shift_1` (the inverse of sigma = shift_1 · t / (1 + (shift_1 − 1) · t)), then applies the same formula at `shift_2`. During the run `model_sampling` is at `shift_2`, so ControlNet and conditioning step ranges follow it. After the run it is back at `shift_1`, also when the run fails.

Static use: with `shift_2` equal to `shift_1` the wrapper hands every run on as it is, so every run, a refine pass too, gets exactly ModelSamplingAuraFlow's (or ModelSamplingSD3's) sigmas at that shift.

### Refine passes

A refine pass is a run on new noise whose schedule starts below full noise: denoise below 1 or a later start step. Examples are a hires or latent-upscaled second pass, Dual Sampler's second pass and a detailer's crop. Qwen Image Shift uses the same rule.

The later part of a split run is no refine pass. KSamplerAdvanced with `add_noise` off (or SamplerCustomAdvanced with DisableNoise) gives the run all-zero noise, so one generation split in two keeps `shift_1`. A later part with `add_noise` on adds new noise and is a refine pass. Neither the denoise nor `add_noise` reaches the node; it reads both from the run (its first sigma, its noise).

Full noise: `model_sampling`'s top sigma at `shift_1` is 1 only to float32 rounding (0.99999988 at 1.18, for example). `normal`, `sgm_uniform` and `karras` compute their first sigma from it and land slightly on either side. So a run starts below full noise when its first sigma is below the midpoint of `model_sampling`'s two highest sigmas, about half a timestep below the top.

### Schedulers

| Scheduler | Run from full noise | Refine pass |
| --- | --- | --- |
| `simple`, `beta` | exactly ModelSamplingAuraFlow / ModelSamplingSD3 at `shift_1` | the scheduler at `shift_2`, to float32 rounding (the map goes through t) |
| `ddim_uniform` | never starts at full noise (its first timestep is below the top: 0.952 at 20 steps), so its runs on new noise are refine passes and take `shift_2` | the scheduler at `shift_2`, to float32 rounding |
| `normal`, `sgm_uniform` | exactly ModelSamplingAuraFlow / ModelSamplingSD3 at `shift_1` | `shift_1`'s grid, mapped: its end comes from `shift_1`, so the steps differ from the scheduler at `shift_2`, most near the end |
| `karras`, `exponential`, `kl_optimal` | exactly ModelSamplingAuraFlow / ModelSamplingSD3 at `shift_1` | the scheduler's `shift_1` curve, mapped: not the curve these schedulers give at `shift_2` |
| `linear_quadratic` | exactly ModelSamplingAuraFlow / ModelSamplingSD3 at `shift_1`, which ignore the shift for it | mapped here; with ModelSamplingAuraFlow it ignores the shift |

### Models

The node takes flow models with one shift. A model whose `model_sampling` is core's ModelSamplingDiscreteFlow (Z-Image, Wan, SD3, or any model after ModelSamplingAuraFlow / ModelSamplingSD3) keeps its class, prediction type and multiplier.

A model whose `model_sampling` is core's ModelSamplingFlux (Qwen-Image and Flux from the loader, or any model after core's ModelSamplingFlux node) gets what ModelSamplingAuraFlow puts on it: ModelSamplingDiscreteFlow with flow prediction, multiplier 1 and the noise scale ModelSamplingAuraFlow gives it. ModelSamplingFlux's timestep is the sigma, and so is a multiplier-1 one's: the model gets the same timestep for a sigma as from the loader. No ModelSamplingAuraFlow is needed before the node. The model's own shift is replaced by `shift_1`; ModelSamplingFlux's shift is an exponent, so Qwen-Image's loader default of 1.15 is a flow shift of e^1.15 ≈ 3.16.

Any other model stops with an error: EPS or v-prediction models, and audio-video models (ModelSamplingAV, whose audio has a second shift).

### Other patches

Qwen Image Shift and this node both set the model's shift. A Qwen Image Shift before it with a wrapper (dynamic mode, or `shift_terminal` on) is an error saying to keep one of the two. One in fixed mode with `shift_terminal` off carries no wrapper; its `model_sampling` is replaced, as ModelSamplingAuraFlow's is.

A node after it that replaces `model_sampling` again (Qwen Image Shift, ModelSamplingAuraFlow, ModelSamplingSD3) stops the run with an error saying to keep one of the two. ComfyUI chains wrappers, so the model's other wrappers stay; a second Dual Sampling Shift replaces the first one's.

## `BC_DyPEPatcher` — DyPE Patcher

`model` (`MODEL`) + `mode` + `dype_start_sigma` + optional `dype_config` (`BC_DYPE_CONFIG`) → `MODEL`: a clone with DyPE (Dynamic Position Extrapolation) on the model's rotary position embedding. The input model is never changed. Z-Image and Qwen-Image (2512).

### What it does

Sampled above its training resolution (a 2x hires pass of Z-Image Turbo, say 1024 × 1536 → 2048 × 3072), a model extrapolates its rotary positions past the lengths it was trained on, and the scene starts repeating itself, mostly in the lower part of the image. DyPE changes the rotary frequencies of the two image axes at each sampling step, with a strength that follows the sigma: strong at high sigma, where the layout forms, fading out toward 0, where the detail forms. The node recognises the model and takes its own method and values; the image size is read from the latent of each step, so one patched model serves any size.

| Value | Z-Image | Qwen-Image | From DyPE Config |
| --- | --- | --- | --- |
| `method` | `vision_yarn` | `vision_yarn` | any of `vision_yarn`, `yarn`, `ntk`, `pi`, `base` |
| `base_resolution` | 1024 px | 1328 px | yes |
| `dype_scale` (λs) | 2.0 | 2.0 | yes |
| `dype_exponent` (λt) | from the scale: 2.0 at 4x `base_resolution` and above (from 4096 px), 1.0 from 2x (from 2048 px), 0.5 below | from the scale, the same tiers: 2.0 from 5312 px, 1.0 from 2656 px, 0.5 below | yes (then fixed) |
| `auto_threshold` | 1920 px | 1920 px | yes |
| `yarn_alt_scaling` | — (Z-Image's `yarn` has none) | off | yes |

The scale is the latent's longer side in px over `base_resolution` (2048 × 3072 on Z-Image: 3.0). Each model has its own `auto_threshold`; the samplers hold none. The model's shift is never changed: DyPE's own shift override (the original node replaces `model_sampling` with a resolution-dependent one, which keeps a hires pass from cleaning the latent upscale's residue) is not ported, so ModelSamplingAuraFlow's shift, Qwen Image Shift's, or the model's own, stays.

The models are ComfyUI core's configs `ZImage` and `QwenImage`. Core's `QwenImage` config also holds the Qwen-Image Edit and Layered checkpoints (it cannot tell 2509 Edit from 2512), so DyPE takes them too; they are not tested. Qwen-Image 2.1, Krea-2, Lumina 2 and other models are refused with an error naming the models DyPE takes (set the mode to `disable` for them).

### `mode`

- `auto` (default): DyPE only for a latent whose longer side is over the model's `auto_threshold` (1920 px for both). At or below it the model runs exactly as unpatched (bit-identical output). Z-Image Turbo at 1024 × 1536: 1.25x (1280 × 1920) stays clean without DyPE, 1.5x (1536 × 2304) starts to repeat.
- `enable`: DyPE for every latent larger than `base_resolution`.
- `disable`: the model is returned as it is.

### `dype_start_sigma`

Below this sigma DyPE's attention scale decays. Set it to the first sigma of the sampler run the model goes into: 1.0 for a full run from noise; for a second pass at a denoise below 1, the first value of that run's own schedule (it depends on the steps, the scheduler, the denoise and the shift: 9 steps, `beta`, denoise 0.45 at shift 3 start at 0.6944). Dual Sampler takes it from its second pass's first model call: the sigma as the run gets it, after a shift the model applies per run (Qwen Image Shift's dynamic shift and its `shift_terminal` stretch), which a schedule computed before the run does not include. A value outside [0.001, 1] is clamped.

### How it is applied

The clone carries a `CALC_COND_BATCH` wrapper (ComfyUI's patcher wrappers) around each sampling step: core's `calc_cond_batch` runs the step's ControlNets (`get_control`) and then the model, so both see the same positions. When DyPE is on for the step's latent, the wrapper sets a step embedder for that step only, on the diffusion model (and on each multi-GPU clone's) and on every ControlNet model of the step's conditioning that carries an embedder of its own, and puts each model's own embedder back when the step returns (or fails). Nothing stays on a model between steps: another clone of the same model (a first pass without DyPE, the next run) always runs with the model's own embedder.

Each call of the step is sized from its own input, as the original sizes each forward from the positions it is given: a model call from its latent (an `APPLY_MODEL` wrapper the clone also carries), a ControlNet from the input its control model is given (a forward hook, removed with the step). So a Conditioning (Set Area) crop runs, model and ControlNet, at the crop's size: a 1024 px crop of a 2048 px Z-Image latent computes with the model's own embedder (not larger than `base_resolution`), the full latent with DyPE's. The step embedder computes with DyPE's embedder for that size and the step's sigma, or with the model's own where DyPE leaves the size alone. A second DyPE Patcher (or Dual Sampler's DyPE) replaces the first's; the model's other wrappers stay. The console says once per size whether DyPE is on, with the scale and values.

ControlNets:

| ControlNet | Its positions |
| --- | --- |
| Qwen-Image Fun ControlNet (core's `QwenFunControlNet`, Apply ControlNet) | the diffusion model's embedder: DyPE's in the step |
| Qwen-Image InstantX ControlNet (core's `QwenImageControlNetModel`, Apply ControlNet) | its own embedder, replaced with the step embedder |
| DiffSynth model patches (Qwen-Image's block-wise ControlNet, Z-Image's Fun ControlNet) | run inside the model: DyPE's |

The rotary tables are computed in float64 and given in float32, as ComfyUI core computes each model's own. The original node computes them in bfloat16 on CUDA, so its output is not bit-identical to this one's.

### Methods

| `method` | Z-Image's image axes | Qwen-Image's image axes |
| --- | --- | --- |
| `vision_yarn` | YaRN's per-band blend (NTK-aware for the fast frequencies, interpolated for the slow) at the scale, its ramps raised to k_t = λs · t^λt, times the attention scale 1 + 0.05 ln(scale) decaying below `dype_start_sigma` | the same blend with each axis's own interpolation scale (its side over `base_resolution`, at least 1) and the image's NTK scale, times 1 + 0.1 ln(scale) decaying below `dype_start_sigma` |
| `yarn` | classic YaRN at the stretched length, the same modulation and attention scale | classic YaRN, both axes stretched to the longer side, YaRN's own attention scale 1 + 0.1 ln(s) / √s; with `yarn_alt_scaling` each axis stretched to its own length (attention scale 1 + 0.1 ln(s)), an axis not over `base_resolution` left plain |
| `ntk` | NTK-aware scaling of theta by scale^(d / (d − 2)), raised to k_t: strongest at high sigma | the same |
| `pi` | position interpolation: the positions divided by scale^k_t (DY-PI) | the same |
| `base` | static NTK scaling, no change over the run | the same |

The time / caption axis keeps core's rotary table. On Qwen-Image the caption tokens, which core places after the image on all three axes, take the same tables as the image tokens, as in the original. The ported code (`libs/dype/`, `models/z_image/dype/`, `models/qwen_image/dype/`) comes from wildminder's ComfyUI-DyPE, Apache-2.0; each folder's NOTICE lists the changes. Among them: `pi` on Z-Image is DY-PI, where the original falls back to `ntk`; on Qwen-Image the scale comes from the latent's size, where the original reads it from the span of every position id, caption tokens included (at 1328 × 1328 px with 100 caption tokens it extrapolates at 2.19x) and through a cache that gives the columns the rows' span (a landscape image is scaled by its height).

## `BC_DyPEConfig` — DyPE Config

`model` + `method` + `base_resolution` + `dype_scale` + `dype_exponent` + `auto_threshold` + `yarn_alt_scaling` + `edited` (hidden) → `dype_config` (`BC_DYPE_CONFIG`), for DyPE Patcher's and Dual Sampler's optional `dype_config`. Only the fields you change on the node, or feed by a link, are applied; every other field keeps the automatic value of the model selected in `model` (the table in DyPE Patcher).

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `model` | `Z-Image` | `Z-Image`, `Qwen-Image` | the model kind the values are for; always applied |
| `method` | `vision_yarn` | `vision_yarn`, `yarn`, `ntk`, `pi`, `base` | the rotary method ([Methods](#methods)) |
| `base_resolution` | 1024 | 256 to 4096, step 16 | the model's training resolution, px |
| `dype_scale` | 2.0 | 0 to 8, step 0.1 | DyPE's magnitude (λs) |
| `dype_exponent` | 1.0 | 0 to 1000, step 0.1 | how fast DyPE's effect decays over the run (λt); changed, it replaces the per-scale choice with that value at every size |
| `auto_threshold` | 1920 | 0 to 16384, step 16 | `auto` mode: DyPE for a latent whose longer side is over this, px |
| `yarn_alt_scaling` | off | | Qwen-Image's `yarn` only: each image axis stretched to its own length |
| `edited` | empty | hidden | the fields changed on this node, comma-separated |

**What is applied.** The `dype_config` output holds `model` and the fields named in `edited`, then every other field whose input is fed by a link. The node reads the links from the prompt it runs in (ComfyUI's hidden `PROMPT` and `UNIQUE_ID` inputs), so a linked field is always applied, whatever `edited` says; a call without a prompt (code calling the node directly, a node made by another node's expansion) applies `edited` alone. The frontend keeps `edited`: a change you make to a field's value adds it (a value the frontend sets itself does not). The node's menu item *Reset to automatic values* clears `edited`. Through the API, set `edited` yourself, e.g. `"base_resolution, dype_exponent"`: names are trimmed, empty ones skipped, a repeated one counted once, and an unknown name is an error that lists the fields. A DyPE Config whose `model` is not the patched model's kind stops the run with an error naming both. The shift is not a DyPE value and cannot be set here.

**What is shown.** The fields not in `edited` show the selected model's automatic values: when you select another `model` and on *Reset to automatic values*, never on load (a saved workflow's values come back as saved). `dype_exponent` has no single automatic value, since it follows each latent's scale (2.0, 1.0 or 0.5), so un-changed it shows its widget default, 1.0, while the run uses the value for the latent's scale. On the classic canvas the un-changed fields are drawn greyed and still take input; the Vue node renderer has no greyed look that still takes input, so there they look like changed fields.

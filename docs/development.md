# Development

```
ComfyUI-BCNodes/
  __init__.py              assembles the mappings, nothing else
  nodes/
    logic.py               BC_LogicBoolean, BC_IsMaskEmpty
    mask.py                BC_MaskFillHoles, BC_MaskGrow, BC_DrawMaskOnImage, BC_BlockifyMask, BC_RepeatMaskBatch
    image_scale.py         BC_ImageScaleByAspectRatio, BC_ImageResize
    lists.py               BC_JoinImageLists
    math_expression.py     BC_MathExpression
    prompt_list.py         BC_PromptList
    any_switch.py          BC_AnySwitch
    select_switch.py       BC_SelectSwitch
    seed.py                BC_Seed
    show_text.py           BC_ShowText
    image_comparer.py      BC_ImageComparer
    power_lora_loader.py   BC_PowerLoraLoader
    lora_key_fix.py        BC_LoraLoaderKeyFix
    everywhere.py          BC_AnythingEverywhere, BC_FastGroupsBypasser (no-ops)
    seedvr2.py             BC_SeedVR2FramingDownscale, BC_SeedVR2Resize, BC_SeedVR2VAEEncode, BC_SeedVR2ChunkSize,
                           BC_SeedVR2VAEDecode, BC_SeedVR2PostProcess, BC_SeedVR2PreprocessCompact, BC_SeedVR2PostProcessCompact
    common.py              wildcard type + flexible optional inputs + slot order + the device widget + unused outputs
    birefnet.py            BC_BiRefNetRemoveBackground
    depth_anything.py      BC_DepthAnythingV2
    downloader.py          BC_AutoModelDownloader + its HTTP routes
    postfx.py              BC_PostFxApply, BC_PostFxTheme, BC_PostFxCustomLook, BC_PostFxLut, BC_PostFxSignatureSheet
    caption_audit.py       BC_CaptionAudit
    social_media_export.py BC_SocialMediaExport (the ComfyUI adapter)
    social_specs.json      the platform table it reads on every run
    image_quality_gate.py  BC_ImageQualityGate
    save_image.py          BC_SaveImage
    save_image_with_caption.py  BC_SaveImageWithCaption
    skin_texture.py        BC_SkinTexture
    frequency_merge.py     BC_FrequencyMerge
    process_monitor.py     Process Monitor: its HTTP routes and live event, the full clear's baseline (no nodes)
  pipelines/               flows that combine models and libs; no ComfyUI node classes
    matting.py             the BiRefNet matte, then the matte options
    model_download.py      downloader lines resolved to files under ComfyUI/models, token gate
    postfx.py              the postfx adapter: dropdown catalogs, looks, apply, contact sheet
    skin_texture.py        SAM 3 prompts, face gate, mask assembly, texture call
    quality_gate.py        shot profiles, the three-tier checks, verdict, report, badge
    social_export.py       Social Media Export geometry / encoding engine, no ComfyUI or torch imports
    save_image.py          Save Image name grammar, job JSON, save loop; Save Image With Caption's folder,
                           caption extension check, save loop
    depth_anything.py      Depth Anything: output size (short side or cover + crop), per-frame normalisation
    lora.py                a LoRA file through core's conversion, the key fix and core's loader (both LoRA loaders)
    caption_audit/
      audit.py             audit plumbing: package guard, allowed roots, run_audit, reports
      card.py              the fixed-size card
    seedvr2/
      resize.py            BC_SeedVR2Resize flow
      encode.py            streaming tiled VAE encode
      decode.py            streaming tiled VAE decode
      postprocess.py       per-frame colour correction
      compact.py           the compact pair: Resize + Encode, Decode + PostProcess, the SEEDVR2_PLAN between them
      framing.py           Framing Downscale: the tallest SAM 3 face box -> close-up / medium / far -> the factor
      chunk_size.py        Chunk Size: the DiT's VRAM law against the card -> frames per chunk (4n+1)
      progress.py          progress bar + timed log lines of the slice loops
    process_monitor/       the Process Monitor
      monitor.py           sampler thread, runs, per-node records, threshold snapshot
      hook.py              the hook into ComfyUI's executor: detection, install, the per-node wrapper
      blackbox.py          run logs, rotation, the Last run and Crash reports
      emulate.py           the estimate before a run: graph, weights, fit check, calibration
      clear.py             the full clear: its steps, each measured against the startup baseline
      profiles.py          per-node-type cost profiles for Emulate
      settings.py          the monitor's settings file
  models/                  one package per model; __init__.py imports those that register
    common/
      registry.py          model families (matting, depth): register / names / get
      download.py          weight download with console progress
    birefnet/
      checkpoints.py       the 11 checkpoints, registered as matting models
      weights.py           weights folder (ComfyUI/models/background_removal) and download
      loader.py            one model loaded at a time; unload() for the full clear
      inference.py         resize, normalise, run, matte back to size
      arch/                BiRefNet + Swin v1 architecture (see LICENSE in the folder)
    depth_anything_v2/     Depth Anything V2 Small, registered as v2-small in the depth family
      weights.py           weights folder (ComfyUI/models/depthanything) and download
      loader.py            the model, loaded once; unload() for the full clear
      inference.py         resize as the authors, run, resample to the size asked for
      arch/                DPT head + DINOv2 ViT-S architecture (see LICENSE in the folder)
    depth_anything_3/      Depth Anything 3 over ComfyUI core, registered as v3-small, v3-base, v3-mono-large,
                           v3-metric-large in the depth family (no code vendored)
      weights.py           Comfy-Org/Depth-Anything-3 files into ComfyUI/models/geometry_estimation
      loader.py            one model at a time, core's loader; unload() for the full clear
      inference.py         core's preprocess and forward, sky, inverse depth, percentile clip
    seedvr2/               adapters over ComfyUI's SeedVR2 VAE
      vae.py               VAE check, the tiling a tile_size runs (0: auto, least work that fits) + VRAM room
      dit.py               the DiT's VRAM law per chunk (measured), SeedVR2 Chunk Size's margin
      tiling.py            tile plan per axis (rows, columns), a typed tile's overlap rules, blend weights
      frames.py            shortest-edge resize, pad, 4n+1 frame count
    sam3/                  adapter over ComfyUI's SAM 3
      checkpoint.py        default checkpoint, combo list, path (downloads the default)
      loader.py            one checkpoint held at a time; unload() for the full clear
      detect.py            text detection through ComfyUI's SAM3_Detect
  libs/                    model-independent helpers
    image.py               IMAGE frame <-> PIL, fit into a target size
    geometry.py            integer size arithmetic for resizing; a ControlNet preprocessor's output size
    filters.py             the two separable Gaussians (reflect, replicate)
    frequency.py           Frequency Merge: one image's Gaussian low-pass + another's high-pass
    mask.py                fill holes, grow / blur, draw a colour through a mask, blockify, offset, refine foreground, fit a mask batch
    resize.py              Image Resize: size plan, crop / resample / pad per frame
    color.py               hex colour parser, sRGB <-> linear
    texture.py             the skin-texture engine
    image_metrics.py       blur / sharpness / noise / clipping / entropy
    math_expression.py     the whitelisted expression evaluator
    download.py            HTTP download with resume, allowed hosts, token store
    files.py               the next image counter from the files in a folder
    image_write.py         Save Image formats, metadata, writer
    memory_sources.py      RAM (cgroup v2 / v1, process RSS) and VRAM (CUDA, MPS, NVML) readers; what the RAM is
                           made of (anon / file, RssAnon / RssFile), glibc's free blocks and malloc_trim
    tensor_census.py       tensor bytes, each byte counted once by address range, file-backed memory told apart; the live tensor census
    safetensors_info.py    weights from a safetensors header
    lora_keys.py           LoRA keys core's loader leaves out (.diff_m, PEFT keys without the prefix), renamed to the
                           names it maps; the keys still unmapped
  docs/                    the node documentation, one page per menu category; README.md links them
  luts/                    drop .cube LUTs here for PostFx LUT (gitignored)
  web/js/
    auto_bypass.js         the BC_AutoBypass virtual node
    join_image_lists.js    unlimited slots for Join Image Lists
    any_switch.js          unlimited slots + type following for Any Switch
    select_switch.js       named option slots, rows and the selected combo for Select Switch
    wildcard_type.js       socket type following, shared by the two switches
    math_expression.js     result overlay for Math Expression
    seed.js                Seed buttons + prompt rewrite of -1
    show_text.js           Show Text boxes
    comparer.js            image comparer widget
    power_lora_loader.js   LoRA rows
    fast_groups_bypasser.js  group toggles
    anything_everywhere.js prompt-time input filling
    auto_model_downloader.js  node UI, first-open dialog, progress
    align.js               toolbox align / distribute buttons
    save_image.js          no output preview under Save Image
    save_image_with_caption.js  the frontend's text replacements in its filename_prefix
    process_monitor.js     Process Monitor: top bar, modal, the on / off setting
    bcnodes_api.js         JSON calls to the pack's routes (downloader, Process Monitor)
  locales/en/main.json     tooltips for the Align buttons
  tests/
    test_import_time.py    import gate
    test_nodes.py          every node with None / empty input, plus plain spot checks
    test_runtime.py        headless ComfyUI: real validation + execution
    test_layers.py         layer rule + module-level import rule, checked statically
    layers/                pytest unit tests per layer (nodes/, pipelines/, models/, libs/)
    conftest.py _harness.py  ComfyUI stubs, package binding
    parity_seedvr2_video.py  BC_SeedVR2VAEEncode / VAEDecode / PostProcess vs ComfyUI's own nodes, numerically
```

Tests:

```
python tests/test_import_time.py                  # import budget and heavy-module ban
python tests/test_nodes.py                        # None / empty input never raises unexpectedly
COMFYUI_DIR=../ComfyUI python tests/test_runtime.py   # nodes through ComfyUI's validate_prompt + PromptExecutor
python -m pytest tests -q                         # layer rule, unit tests
python tests/parity_seedvr2_video.py --comfy ../ComfyUI --vae ../ComfyUI/models/vae/seedvr2_ema_vae_fp16.safetensors   # SeedVR2 VAE Encode / Decode / PostProcess against ComfyUI's nodes (GPU for the VAE)
```

The tests need `postfx` and `caption-audit` importable: `pip install -r requirements.txt`, or `PYTHONPATH=/path/to/postfx:/path/to/caption-audit` for local checkouts.

The runtime test needs a ComfyUI checkout with its requirements installed in the same Python; it starts no server. It covers the things that only the real executor can prove: canvas-only slots (`In3`, `any_03`) reaching the node, wildcard sockets validating in both directions, list outputs fanning out, Select Switch running only the selected lazy branch, that a genuine type mismatch is still rejected, and Lora Loader (Key Fix) against core's own LoRA loader on a tiny real Wan model (the keys core leaves out as they are, and loads once renamed).

Every module in `nodes/`, `pipelines/`, `models/` and `libs/` imports only `torch`, `numpy` and the standard library at module level; `scipy`, `PIL`, `cv2`, `safetensors`, `torchvision`, `folder_paths`, `comfy.*` and the pip packages `postfx` / `caption_audit` are imported inside the functions that use them (the downloader also touches `server` / `aiohttp`, which ComfyUI has loaded already), so the pack adds nothing to ComfyUI's startup. A node module also imports its pipeline, model and lib modules inside the methods that use them (a widget list such as the model names inside `INPUT_TYPES`), so the package import reads and compiles little more than `nodes/`; the Process Monitor's own modules load at import, where it registers and starts. `python tests/test_import_time.py` checks both.

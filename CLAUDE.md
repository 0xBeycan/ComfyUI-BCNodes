# ComfyUI-BCNodes

A ComfyUI custom-node pack (`BC_*` utility nodes plus a `web/js` frontend). The node list is in
`README.md`; what each node does is in `docs/`, one page per menu category.

## Architecture

Four layers, one-way dependencies:

```
nodes/  ->  pipelines/  ->  models/  ->  libs/
```

| layer | holds | never holds |
|---|---|---|
| `nodes/` | The ComfyUI surface: `INPUT_TYPES`, tooltips, `DESCRIPTION`, categories. Converting ComfyUI types (IMAGE/MASK tensors, LATENT dicts, hidden `PROMPT`/`EXTRA_PNGINFO`, `ui` payloads, temp/output folders) to library types and back, then one pipeline call. One file per domain. The downloader's HTTP routes. | algorithms |
| `pipelines/` | Flows that combine models and libs. Config and contract types (dataclasses, TypedDicts). | model architectures, ComfyUI node classes |
| `models/` | One package per model: architecture, weights, cache, model-specific pre/post-processing, adapters over ComfyUI core models. `models/common/` holds the registry and the weight download. | pipeline logic |
| `libs/` | Model-independent helpers. | imports of `models/`, `pipelines/` or `nodes/`; ComfyUI node classes |

The root `__init__.py` only registers: it imports the node modules and merges their mappings (their
order is the menu order), then registers the link stamp of the unused heavy outputs
(`register_link_stamp`, which also hooks `stamps_last` to the server's startup; below).
`WEB_DIRECTORY = "./web"`.
It also imports `nodes/process_monitor.py`, which registers no nodes: it registers the Process
Monitor's HTTP routes and, when the monitor's saved setting is on, starts it (see Process Monitor).

```
nodes/common.py               AnyType, FlexibleOptionalInputType, slot_index, compute_device (the device widgets),
                              the unused-heavy-outputs helper (LinkStamp, register_link_stamp, stamps_last,
                              heavy_wanted, wants, drop_unwanted, drop_unlinked_heavy)
nodes/<domain>.py             one per domain (24); social_specs.json is the user-editable platform table
pipelines/matting.py          finish() option chain; remove_background() -> models.birefnet.inference.matte
pipelines/model_download.py   downloader entries -> resolved items, token gate, "seen" marker
pipelines/postfx.py           postfx adapter: catalogs, LUTS_DIR, looks, apply, contact sheet
pipelines/skin_texture.py     SAM 3 prompts, face gate, mask assembly, texture call
pipelines/quality_gate.py     shot profiles, QualityCheck, verdict, report, badge
pipelines/social_export.py    Social Media Export engine: plan, crop/pad, encode, report line
pipelines/save_image.py       name grammar, PROMPT walking, job JSON, save loop; Save Image With Caption's folder,
                              caption extension check, save loop
pipelines/depth_anything.py   Depth Anything: output size (short side, or cover + centre crop), per-frame
                              normalisation, one depth-family predict per frame
pipelines/caption_audit/      audit.py (args, dataset roots, run, reports), card.py (the card)
pipelines/seedvr2/            resize, encode, decode, postprocess flows; progress; shared constants
pipelines/process_monitor/    monitor (sampler thread, runs, per-node records, snapshot), hook (the executor hook),
                              blackbox (run logs, reports), emulate + profiles (estimate, per-node-type costs), settings
models/common/                registry.py (families matting and depth; a depth entry is a loader returning
                              `predict`), download.py (fetch_with_progress)
models/birefnet/              checkpoints (registered under MATTING), weights, loader, inference, arch/ (vendored, MIT)
models/depth_anything_v2/     Depth Anything V2 Small: weights, loader, inference, arch/ (vendored, Apache-2.0); registered
                              as v2-small in the depth family
models/depth_anything_3/      Depth Anything 3 over core (comfy.ldm.depth_anything_3, nothing vendored): weights
                              (Comfy-Org/Depth-Anything-3 into models/geometry_estimation), loader, inference; registered
                              as v3-small, v3-base, v3-mono-large, v3-metric-large in the depth family
models/seedvr2/               VAE adapter, tiling, frame-shape rules (no registry)
models/sam3/                  checkpoint, loader, detect (over ComfyUI core SAM 3)
libs/image.py                 tensor_to_pil_u8, pil_to_tensor_hwc, fit_image
libs/mask.py filters.py       mask ops; the two Gaussians (reflect / replicate), kept apart on purpose
libs/color.py texture.py image_metrics.py
libs/geometry.py              integer size arithmetic; short_side_size (a ControlNet preprocessor's output size)
libs/resize.py                Image Resize: size plan, crop / resample / pad per frame
libs/math_expression.py       whitelisted AST evaluator with injected resolvers
libs/download.py              HTTP download with resume, host list, token store
libs/files.py image_write.py  output counters; image formats, metadata, write_image
libs/memory_sources.py        RAM (cgroup v2 / v1, process RSS) and VRAM (CUDA, MPS, NVML) readers
libs/tensor_census.py         tensor bytes per storage, the live tensor census
libs/safetensors_info.py      weights from a safetensors header, no load
```

## Process Monitor

Not a node. Its placement follows the layers:
- `nodes/process_monitor.py` is the ComfyUI surface, like the downloader's routes: the
  `/bcnodes/monitor/*` routes, the `bcnodes.monitor` live event, and the adapters that hand the
  pipeline ComfyUI's prompt queue (`ServerProbe`), folders and node classes (`ComfyEnv`; the node
  classes through `execution.nodes`, since an absolute `import nodes` is banned here). It registers
  only when `PromptServer.instance` exists; `server`, `aiohttp`, `folder_paths`, PIL and PyAV are
  imported inside functions.
- `pipelines/process_monitor/` holds the flows: the sampler thread and runs (`monitor.py`), the
  hook (`hook.py`), the run logs and reports (`blackbox.py`), the estimate (`emulate.py`) and its
  per-node-type cost profiles (`profiles.py`). Nothing there imports ComfyUI at module level.
- `libs/` holds what any flow could use: the RAM / VRAM readers, the tensor census and the
  safetensors header reader.

Rules that keep it cheap and safe:
- Off means no thread, no hook and no file write. Run boundaries come from the prompt queue
  (every ComfyUI version); the per-node layers need the hook on `execution.execute`, detected by
  feature (ComfyUI 0.17.0+). The wrapper always awaits the original and returns its result
  untouched; an error in the monitor turns the measurement off with a message.
- Counters only, never tensor copies: cgroup files, allocator statistics, shape x dtype. The one
  scan over all objects is the threshold snapshot, once per run, after the execution thread's own
  tensors are written.
- Emulate profiles are derived from the node's code. A part that cannot be derived is
  "not counted" with a note, never guessed. A profile keyed by another pack's class name is data;
  its comments describe what the node does to memory, not the other pack's code.
- Run logs, settings and calibration measurements live in `user/BCNodes/process_monitor/`, never in
  a workflow. A workflow id names a file only when it is a plain token.

## The layer rule

Allowed import edges (source -> target):
- root `__init__.py` -> `nodes.*` only.
- `nodes.X` -> `nodes.common`, `pipelines.*`, `models.*`, `libs.*`. `nodes.common` imports nothing from the pack.
- `pipelines.U` -> the same `U` (module or package), `models.*`, `libs.*`. Never another pipeline.
- `models.P` -> the same `P`, `models.common`, `libs.*`. `models.common` -> `models.common`, `libs.*`. `models/__init__.py` imports each model package (the registration hub).
- `libs.X` -> `libs.*` only.

Every import inside the pack is relative (`from ..libs.image import fit_image`). An absolute import
of `nodes`, `pipelines`, `models`, `libs`, `birefnet` or `tests` is an error: under ComfyUI those
names reach ComfyUI's own modules or nothing. Two permanent exceptions: `nodes/image_comparer.py` lazily
imports `PreviewImage` from ComfyUI's `nodes` module, and `nodes/downloader.py` imports
`server`/`aiohttp` at module level inside a guarded `try`. `tests/test_layers.py` enforces all of
this statically; its `TRANSITIONAL` list stays empty (fix a violation, never allowlist it).

## Import-time rule

At module level only the standard library, `torch`, `numpy` and relative pack modules. Everything
else (`scipy`, `PIL`, `cv2`, `safetensors`, `torchvision`, `folder_paths`, `comfy.*`,
`comfy_extras.*`, `av`, `postfx`, `caption_audit`, `yaml`) is imported inside the function that
uses it. The only exception is the downloader's guarded `server`/`aiohttp`. Gates:
`tests/test_import_time.py` (package import under 0.1 s with torch and numpy preloaded, no module
of its HEAVY list loaded, every layer module cold-imported with `PYTHONSAFEPATH=1`) and the static
check in `tests/test_layers.py`.

A lazy import that looks unused can be an order lock: `from PIL import Image` in
`nodes/save_image.py` `save_images` makes a missing Pillow fail before any other work. Keep it
where it is.

## How to add a node

- Key `BC_<Name>`, a display name, and `CATEGORY` one of the 9 groups: `BCNodes/analysis`,
  `image`, `loaders`, `logic`, `mask`, `postfx`, `seedvr2`, `text`, `workflow`.
- A literal `INPUT_TYPES` in the node class. The node file holds the surface only; the work goes
  to a pipeline (or straight to libs/models when there is no flow).
- Import the module in the root `__init__.py` and add it to the registration loop (order = menu
  order). Add it to `NODE_MODULES` and its keys to `expected` in `tests/test_import_time.py`, and
  to the eager tuple of `tests/_harness.py` `load_package`.
- Add unit tests for its behaviour under `tests/layers/<layer>/`.
- Frontend code goes under `web/js/`, loaded by path.
- A whole-batch IMAGE or MASK output the node makes, among two or more outputs (or on an output
  node), is a heavy output: list it in `HEAVY_OUTPUTS` (names from `RETURN_NAMES`), add
  `"hidden": dict(LINK_INPUTS)` to `INPUT_TYPES` (`prompt_graph`, `unique_id`; not `prompt`, which a
  widget may be called), read `wanted = heavy_wanted(type(self), prompt_graph, unique_id)` and return
  through `drop_unwanted(type(self), outputs, wanted)`. When the output is a step of its own that no
  other output reads, pass `wants(wanted, name)` down so the step does not run; when the other
  outputs need it, it is only dropped at return. A pass-through (the input tensor itself) or a
  one-image output is not heavy, and neither is the output of a single-output node that is not an
  output node: it runs only when that output is linked. Test it in
  `tests/layers/nodes/test_node_unused_outputs.py` (its `HEAVY` table, the dropped output empty with
  the full one's dtype and trailing shape, the linked ones equal, the skipped step not run); the
  mechanism through ComfyUI's executor is in `tests/test_runtime.py` (`unused_outputs`).

## How to add a model

- A family exists only when there is a real choice between models. Its key lives in
  `models/common/registry.py`.
- A `models/<name>/` package: architecture, weights (download through
  `models.common.download.fetch_with_progress`), a single-slot cache as in
  `models/birefnet/loader.py`, model-specific pre/post-processing.
- Register each member with `register(FAMILY, name, entry)` in the package, and add one import
  line to `models/__init__.py` so the registry is filled before any lookup. A node combo is
  `registry.names(FAMILY)`, a lookup `registry.get(FAMILY, name)`.
- Matting has one implementation: `pipelines/matting.py` calls `models/birefnet/inference.matte`
  directly, and `models/birefnet/loader.load` evicts the cached model before an unknown name
  raises `KeyError`. A second matting implementation needs the owner's decision on dispatch.
- Vendored code keeps its LICENSE in its own subdirectory (as `models/birefnet/arch/`).

## Coding style

- Before writing new code, search the pack for code that already does the same work (grep for
  the operation, not only the name). If it exists, call it. If the same code would end up in
  two places, move it into one function and call that from both. Never write a second copy.
- Small functions with one job; explicit data contracts (a dataclass inside, a TypedDict for a
  dict that goes over the wire or into tests); errors that say what to do; no silent defaults.
- A new function exists only when identical code already lives in 2+ places (reduce it to one)
  or future nodes of this pack are likely (~70-80%) to reuse it. Otherwise the code stays inline.
  SeedVR2-only code stays in the SeedVR2 packages.
- "Identical code" means code with work of its own, written the same at sites that are not
  alternative arms of one branch. Bare calls of an existing function, equal literal values, and
  conditions whose action is the caller's `return`/`continue` do not count.
- Two near-copies are merged only after they are proven identical; a difference is a question
  for the owner. One-caller helpers live next to their caller. Converters (float <-> uint8,
  PIL <-> tensor) that differ in any detail are never unified without the owner's word.

## Optimization principles

- Speed and RAM are equal priorities. No RAM saving that makes generation slower.
- Bit-exact output is not required, but the output never drifts from the origin. A departure
  from the origin is a widget setting, never hidden behaviour.
- A port of a third-party node does the same job in our style: preallocated outputs, no
  list -> stack/cat, no clones of read-only inputs, no leaks. Read the original first and never
  reproduce its bugs. Take the logic only: no import of, dependency on or reference to the
  original.
- Precision is decided per tensor, by measurement, never globally.
- Unused heavy outputs are not kept. ComfyUI's cache key holds a node's inputs and ancestors only,
  so an on_prompt handler (`LinkStamp` in `nodes/common.py`, registered by the root `__init__`)
  writes the linked heavy outputs of each heavy node into its inputs as `bc_linked_heavy`; the node
  returns an unlinked one as a 0-frame tensor (or never computes it). No stamp (no server, a direct
  executor call): every output full. A link the stamp missed: full, with a warning. At server
  startup (aiohttp `on_startup`, after every custom node has loaded) `stamps_last` moves every
  `bc_link_stamp` handler to the end of the list; a handler added later still turns the saving off
  for that prompt, with a console line (no toast).
- A node never resizes itself to its content; previews and widgets scale to the node.
- Values that can differ between uses are widgets, not constants.

## Tests and gates

`postfx` and `caption-audit` must be importable: `pip install -r requirements.txt`, or
`PYTHONPATH=/path/to/postfx:/path/to/caption-audit` for local checkouts. From the repo root:

```
python tests/test_import_time.py                      # import budget, no heavy module, every module cold
python tests/test_nodes.py                            # node smoke and spot checks (None / empty inputs)
COMFYUI_DIR=../ComfyUI python tests/test_runtime.py   # through ComfyUI's validate_prompt + PromptExecutor; a SKIP is red
python -m pytest tests -q                             # layer rule, unit tests
```

- A case lives in `tests/layers/<layer>/` of the code it calls directly; a case entered through a
  node method lives in `tests/layers/nodes/`. Basenames carry the layer prefix (`test_node_`,
  `test_pipe_`, `test_model_`, `test_lib_`). Never a `tests/nodes/` or `tests/models/`: pytest
  would make it an importable top-level `nodes` / `models`.
- `tests/parity_seedvr2_video.py` compares the SeedVR2 nodes with ComfyUI's own and needs a GPU
  for the VAE; it is not part of the suite.

## Compatibility locks

Saved workflows must load and run unchanged. Never change: node keys, display names, categories,
`INPUT_TYPES` names, types, order, defaults and ranges, `RETURN_TYPES`/`RETURN_NAMES`,
`OUTPUT_NODE`, `web/js` paths, the `/bcnodes/downloader/*` routes and the `bcnodes.downloader`
event name, the location of `nodes/social_specs.json` and of `luts/`. Also locked: the stamp key
`bc_linked_heavy` (part of every heavy node's cache key) and the `bc_link_stamp` marker on the
stamping handler, which ComfyUI-BCVideoNodes' handler reads to leave ours out of "another pack" and
`stamps_last` reads to move it last (and ours reads on its).

## Closed decisions

The owner's project memory for this pack (the "ComfyUI-BCNodes pack" entry) holds them; do not
reopen them:
- `BC_` keys and the 9 categories; Logic Boolean takes a FLOAT; mask nodes return zeros on
  empty input.
- BiRefNet: vendored architecture; the authors' weights under `models/background_removal/`; an
  unknown name evicts the cached model before the `KeyError`.
- Downloader: tokens are set in the UI only and never echoed; `Authorization` never follows a
  redirect to another host; a stored token is sent over https only; host restriction;
  final-path confinement.
- Image Quality Gate clips to [0, 1] before the uint8 cast. No SeedVR2 registry.
- The lazy-import rule, Fast Groups Bypasser behaviour, comparer UX, MathExpression
  `IS_CHANGED` and ShowText serialisation; the SeedVR2 5090 retest is parked.

## Git

The owner's global `CLAUDE.md` (user-level Claude Code instructions) governs git: no commits on
his branch, work on `local-*` worker branches, never push. Several sessions edit this repo:
claim files before editing.

# ComfyUI-BCNodes

A ComfyUI custom-node pack (`BC_*` utility nodes plus a `web/js` frontend). The node list and
what each node does are in `README.md`.

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

The root `__init__.py` only registers: it imports the node modules and merges their mappings, and
their order is the menu order. `WEB_DIRECTORY = "./web"`.

```
nodes/common.py               AnyType, FlexibleOptionalInputType, slot_index
nodes/<domain>.py             one per domain (22); social_specs.json is the user-editable platform table
pipelines/matting.py          finish() option chain; remove_background() -> models.birefnet.inference.matte
pipelines/model_download.py   downloader entries -> resolved items, token gate, "seen" marker
pipelines/postfx.py           postfx adapter: catalogs, LUTS_DIR, looks, apply, contact sheet
pipelines/skin_texture.py     SAM 3 prompts, face gate, mask assembly, texture call
pipelines/quality_gate.py     shot profiles, QualityCheck, verdict, report, badge
pipelines/social_export.py    Social Media Export engine: plan, crop/pad, encode, report line
pipelines/save_image.py       name grammar, PROMPT walking, job JSON, save loop
pipelines/caption_audit/      audit.py (args, dataset roots, run, reports), card.py (the card)
pipelines/seedvr2/            resize, encode, decode, postprocess flows; progress; shared constants
models/common/                registry.py (families), download.py (fetch_with_progress)
models/birefnet/              checkpoints (registered under MATTING), weights, loader, inference, arch/ (vendored, MIT)
models/seedvr2/               VAE adapter, tiling, frame-shape rules (no registry)
models/sam3/                  checkpoint, loader, detect (over ComfyUI core SAM 3)
libs/image.py                 tensor_to_pil_u8, pil_to_tensor_hwc, fit_image
libs/mask.py filters.py       mask ops; the two Gaussians (reflect / replicate), kept apart on purpose
libs/color.py geometry.py texture.py image_metrics.py
libs/video.py                 side-by-side geometry and PyAV writer
libs/math_expression.py       whitelisted AST evaluator with injected resolvers
libs/download.py              HTTP download with resume, host list, token store
libs/files.py image_write.py  output counters; image formats, metadata, write_image
```

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

A lazy import that looks unused can be an order lock: `import av` in `nodes/video_comparer.py`
`_encode` and `from PIL import Image` in `nodes/save_image.py` `save_images` make a missing PyAV
or Pillow fail before any other work. Keep them where they are.

## How to add a node

- Key `BC_<Name>`, a display name, and `CATEGORY` one of the 9 groups: `BCNodes/analysis`,
  `image`, `loaders`, `logic`, `mask`, `postfx`, `seedvr2`, `text`, `workflow`.
- A literal `INPUT_TYPES` in the node class. The node file holds the surface only; the work goes
  to a pipeline (or straight to libs/models when there is no flow).
- Import the module in the root `__init__.py` and add it to the registration loop (order = menu
  order). Add it to `NODE_MODULES` and its keys to `expected` in `tests/test_import_time.py`, and
  to the eager tuple of `tests/_harness.py` `load_package`.
- Add its case to the surface golden `tests/layers/nodes/test_node_surface.py` and goldens for its
  outputs.
- Frontend code goes under `web/js/`, loaded by path.

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

## Tests and gates

`postfx` and `caption-audit` must be importable: `pip install -r requirements.txt`, or
`PYTHONPATH=/path/to/postfx:/path/to/caption-audit` for local checkouts. From the repo root:

```
python tests/test_import_time.py                      # import budget, no heavy module, every module cold
python tests/test_nodes.py                            # node smoke and spot checks (None / empty inputs)
COMFYUI_DIR=../ComfyUI python tests/test_runtime.py   # through ComfyUI's validate_prompt + PromptExecutor; a SKIP is red
python -m pytest tests -q                             # layer rule, per-layer goldens, unit tests
```

- Goldens are behaviour fingerprints. Each file pins `ENV` (platform, package versions, the
  postfx / caption_audit sources) and fails, never skips, when it differs.
- A `GOLDEN` value changes only with the owner's word, for an intended behaviour change. It is
  then re-recorded with the change applied (`BCNODES_GOLDEN_RECORD=1` prints the tables) and `ENV`
  re-pinned. The cases that differ from a recording at the change's parent commit are shown to
  the owner; only the intended ones may differ.
- Moving code edits only `WHERE` entries; `ENV`, `GOLDEN`, inputs and assertions stay.
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
event name, the location of `nodes/social_specs.json` and of `luts/`.

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

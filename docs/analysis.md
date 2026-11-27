# Analysis

## `BC_CaptionAudit` — Caption Audit

Audits a LoRA caption set before a training run, with the [`caption-audit`](https://github.com/0xBeycan/caption-audit) package. A token that appears in nearly every caption stops being a describable attribute: the model cannot tell it apart from the trigger word, so it bakes the concept into the identity — it can no longer be prompted in or out. That is a caption distribution problem, and nothing downstream of the dataset fixes it.

The node needs `caption-audit>=1.3`, in the pack's `requirements.txt`. Without it the node stays in the node list and every run draws the error card naming the package (`pip install 'caption-audit>=1.3'`), with `critical = 1`.

A caption set is a **folder**: images plus `.txt` sidecars sharing each image's basename. The node takes the path, runs the audit in-process and draws the report card as a preview inside the node.

The node runs again when a widget changes or when the caption files change: it fingerprints the `.txt` files in `directory` (path, modification time and size; subdirectories too with `recursive`), so captions edited outside ComfyUI are audited again on the next queue and an untouched set stays cached. A path outside the allowed roots stays cached on its error card.

| Input | Default | Meaning |
| --- | --- | --- |
| `directory` | *(empty = ComfyUI's cwd)* | The caption set to audit. Confined to the ComfyUI tree and the roots named in `BC_CAPTION_ROOTS` — see below. |
| `trigger` | *(empty)* | The trigger word this LoRA owns. Empty = inferred and marked **[INFERRED]** on the card. |
| `class_words` | — | Comma-separated, e.g. `woman, car`. Shown as `EXPECTED`, never flagged; coverage is measured per word. |
| `fuse` | — | Comma-separated attributes you *want* welded to the trigger, e.g. `red scarf`. Shown as `INTENDED`, excluded from `critical`. |
| `critical_threshold` / `warn_threshold` / `info_threshold` | `0.85` / `0.60` / `0.35` | Document-frequency cut-offs for fused / strong bias / not reported. |
| `ngram_max` | `3` | Longest phrase analysed; phrases never cross a comma. |
| `no_stopwords` | `false` | On shows function and relational words too. |
| `recursive` | `false` | Descend into subdirectories. |
| `table_rows` | `12` | Rows in the card's term table — the only input that changes the card's size, so the node never resizes between runs. |
| `images_dir` *(optional)* | — | Where the images live when captions are kept apart. With no images anywhere the audit runs in caption-only mode. |

`directory` and `images_dir` are widget values, and a widget value comes out of the workflow JSON — a graph you downloaded could otherwise point the node at any folder on the machine and read the caption text back out through `report_text` / `report_json`. So both paths are resolved with `realpath` and must land inside the ComfyUI tree (the ComfyUI root, `input/`, `output/`, `user/`). Datasets normally live somewhere else, and the way to allow one is the **`BC_CAPTION_ROOTS`** environment variable — `:`-separated roots on Linux and macOS, `;`-separated on Windows — read from the environment ComfyUI starts in, never from the graph:

```bash
BC_CAPTION_ROOTS=/path/to/datasets python main.py
```

A path outside those roots is refused on the error card with the roots listed, the same as a missing folder.

Outputs: `report_image` (the card), `report_text` (the complete terminal report), `report_json` (identical to `caption-audit --format json`), `critical` and `warning` counts. `critical` exists to stop a workflow: wire it into a compare node ahead of the training branch. If the audit cannot run (missing folder, no `.txt` files, thresholds out of order, no `caption-audit` package) the node renders an error card and returns `critical = 1` instead of raising — a set that could not be checked has not been cleared.

The card ends with the question the tool refuses to answer: every flagged term has two possible causes with opposite fixes — caption hygiene (delete the word where it is not the point) or dataset composition (collect contrast data) — and telling them apart means looking at the images, which neither the node nor the package does.

## `BC_ImageQualityGate` — Image Quality Gate

Single-node quality control for AI-generated images, built for filtering LoRA training sets. Five metrics — blur (block-wise Laplacian variance, optionally **center-weighted** so background bokeh does not inflate the score), sharpness (Laplacian + Tenengrad), noise (Gaussian difference), highlight / shadow clipping and Shannon entropy — each scored against a threshold, folded into a three-tier verdict:

| Verdict | `verdict` | Meaning |
| --- | --- | --- |
| PASS | `2` | every metric within its threshold |
| SO-SO | `1` | no hard failure, but a metric sits in the margin zone (within 1.4× of its threshold) |
| FAIL | `0` | a metric is past the margin zone |

Only the first image of a batch is analysed: the verdict, the badge and the scores are that image's (an image list runs the node once per image).

The node needs OpenCV: `opencv-python` from the pack's `requirements.txt`, or the `opencv-python-headless` that `postfx` installs. Without it the node stops with Python's import error when run.

| Widget | Default | Range | Meaning |
| --- | --- | --- | --- |
| `shot_type` | `custom` | `custom`, `close-up`, `medium`, `wide / full-body` | Preset that scales the thresholds and sets the blur block size (table below). |
| `blur_mode` | `full image` | `full image`, `center-weighted` | `center-weighted` weighs each block by its distance from the centre, as a share of the half diagonal: `3` under 0.3, `1` under 0.6, `0.3` beyond, so background bokeh does not inflate the score. |
| `blur_threshold` | `0.40` | `0..1` | Highest share of blurry blocks. |
| `blur_var_threshold` | `30` | `5..500` | Laplacian variance below which a block counts as blurry: 20–50 for AI images (distilled models have inherently lower variance), 80–150 for photographs. Not scaled by the preset. |
| `sharpness_threshold` | `15` | `0..500` | Lowest sharpness. |
| `noise_threshold` | `25` | `0..100` | Highest noise. |
| `clipping_threshold` | `0.02` | `0..1` | Highest share of clipped pixels (grey level `≤ 5` or `≥ 250`). |
| `entropy_threshold` | `5.0` | `0..8` | Lowest entropy, in bits. |

`shot_type` multiplies each threshold and picks the blur block size:

| `shot_type` | blur | sharpness | noise | clipping | entropy | block |
| --- | --- | --- | --- | --- | --- | --- |
| `custom` | 1 | 1 | 1 | 1 | 1 | 24 px |
| `close-up` | 0.6 | 1.5 | 0.8 | 0.8 | 1.1 | 16 px |
| `medium` | 1 | 1 | 1 | 1 | 1 | 24 px |
| `wide / full-body` | 1.6 | 0.6 | 1.3 | 1.2 | 0.85 | 32 px |

A close-up is judged stricter, a wide shot looser; `medium` is the same as `custom`, which uses the widget values as they are.

Outputs: a colour-coded `badge` image with the per-metric breakdown, the integer `verdict` for a Switch node, a text `report` and the five raw scores (`blur_score`, `sharpness_score`, `noise_score`, `clipping_score`, `entropy_score`).

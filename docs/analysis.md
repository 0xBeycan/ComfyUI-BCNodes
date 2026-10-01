# Analysis

## `BC_CaptionAudit` — Caption Audit

Audits a LoRA caption set before a training run, with the [`caption-audit`](https://github.com/0xBeycan/caption-audit) package. A token that appears in nearly every caption stops being a describable attribute: the model cannot tell it apart from the trigger word, so it bakes the concept into the identity — it can no longer be prompted in or out. That is a caption distribution problem, and nothing downstream of the dataset fixes it.

A caption set is a **folder**: images plus `.txt` sidecars sharing each image's basename. The node takes the path, runs the audit in-process and draws the report card as a preview inside the node.

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

Outputs: `report_image` (the card), `report_text` (the complete terminal report), `report_json` (identical to `caption-audit --format json`), `critical` and `warning` counts. `critical` exists to stop a workflow: wire it into a compare node ahead of the training branch. If the audit cannot run (missing folder, no `.txt` files, thresholds out of order) the node renders an error card and returns `critical = 1` instead of raising — a set that could not be checked has not been cleared.

The card ends with the question the tool refuses to answer: every flagged term has two possible causes with opposite fixes — caption hygiene (delete the word where it is not the point) or dataset composition (collect contrast data) — and telling them apart means looking at the images, which neither the node nor the package does.

## `BC_ImageQualityGate` — Image Quality Gate

Single-node quality control for AI-generated images, built for filtering LoRA training sets. Five metrics — blur (block-wise Laplacian variance, optionally **center-weighted** so background bokeh does not inflate the score), sharpness (Laplacian + Tenengrad), noise (Gaussian difference), highlight / shadow clipping and Shannon entropy — each scored against a threshold, folded into a three-tier verdict:

| Verdict | `verdict` | Meaning |
| --- | --- | --- |
| PASS | `2` | every metric within its threshold |
| SO-SO | `1` | no hard failure, but a metric sits in the margin zone (within 1.4× of its threshold) |
| FAIL | `0` | a metric is past the margin zone |

`shot_type` presets (`close-up`, `medium`, `wide / full-body`) scale the sliders — a close-up is judged stricter on blur and sharpness, a wide shot looser — and `custom` uses the raw values. `blur_var_threshold` is the per-block Laplacian variance that counts as blurry: 20–50 for AI images (distilled models have inherently lower variance), 80–150 for photographs. Outputs: a colour-coded `badge` image with the per-metric breakdown, the integer `verdict` for a Switch node, a text `report` and the five raw scores.

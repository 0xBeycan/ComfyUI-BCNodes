# PreFlight

How Instagram, TikTok and X would likely treat a generated image or video, before you publish it. PreFlight flags two risks per platform:

- **Hard limit**: removal (violates community standards).
- **Soft limit**: reach suppression (Instagram's recommendation exclusion, TikTok's For-You-Feed ineligibility).

A local chat model, Qwen3.5-9B, acts as a *pure observation sensor*: it describes what is visually present and does **not** judge. Its observations go into a **deterministic rules engine** that maps them to per-platform verdict ranges. A feedback loop records what the platform actually did, so the rules can be checked against reality over time.

> **An assistive signal, not an approval gate.** It never says "approved" or "rejected". Verdicts are ranges (best … worst); the report names the unknown behind each range, so you can collapse it yourself. The final judgment is yours.

## Why four nodes

| Node | What it does |
| --- | --- |
| [PreFlight Observe](#bc_preflightobserve--preflight-observe) | Runs the model and returns a structured **observations JSON** (garment, exposure, framing, pose, in-image text, …). |
| [PreFlight Report](#bc_preflightreport--preflight-report) | Applies the rules engine: **per-platform verdicts**, a summary to read, and the prediction logged to the feedback store. |
| [PreFlight Outcome](#bc_preflightoutcome--preflight-outcome) | Records what a platform actually did to a published post, days later. |
| [PreFlight Calibrate](#bc_preflightcalibrate--preflight-calibrate) | Turns the store into the calibration tables: how often the predictions held, per platform and per rule. |

Observation and judgment are split on purpose: when a verdict looks wrong you need to know whether **perception** failed (the model misread the garment) or the **rules** did. Keeping the observations as a separate record also lets old content be judged again by updated rules **without running the model again**: every observation carries the schema and prompt version that produced it.

A typical graph: Load Image → **PreFlight Observe** → **PreFlight Report** (`image` passed through to a preview, `summary` into [Show Text](text.md#bc_showtext--show-text)). Days after posting: a **PreFlight Outcome** per platform. From time to time: **PreFlight Calibrate** → Show Text.

## `BC_PreFlightObserve` — PreFlight Observe

`image`, `max_frames`, `keep_model_loaded` → `observations_json` (`STRING`), `raw_response` (`STRING`).

| Input | Default | Range | Meaning |
| --- | --- | --- | --- |
| `image` | | `IMAGE` | A single image or a video's frame batch. |
| `max_frames` | `6` | `1`–`16` | For a video: how many frames are sampled evenly from the batch. All of them go to the model in **one** call, so it can tell motion from the sequence. |
| `keep_model_loaded` | on | | On: the model stays loaded for the next run. Off: it is unloaded when the run ends, after an error too. |

| Output | Meaning |
| --- | --- |
| `observations_json` | The observation, or `{"error": ...}` when the run failed. |
| `raw_response` | The model's answer as it came, both answers when it was retried, for debugging. |

**The model.** Observe has no model choice: it always runs **Qwen3.5-9B, INT8 ConvRot**, downloaded on first use. It goes through the same runtime as [Qwen LM](lm.md#bc_qwenlm--qwen-lm): the file `Qwen3.5-9B_int8_convrot.safetensors` is found in `models/Qwen-LM` or under `models/text_encoders`, or downloaded into `models/Qwen-LM` on the first run ([Models and files](lm.md#models-and-files)), and ComfyUI core runs it. One LM model is held at a time, shared with Qwen LM: a Qwen LM run with another model file replaces it, and the other way round. Its peak VRAM is about **10.5 GB** (measured on an RTX 3090: one image or a 4-frame batch).

**The run.** The sampled frames go first, then the observation prompt as the user text, in the model's official chat template ([Prompt and thinking](lm.md#prompt-and-thinking)): no system turn, thinking off, no prefill. Decoding is pure greedy (the most likely token at each step, no penalties), seed 42, at most 300 new tokens, so the same image and settings give the same observation run to run, a requirement for calibration. The model is asked for a strict JSON object; the first `{…}` of its answer is read, whatever prose or code fence surrounds it. An answer with no usable object gets **exactly one** retry, with `Your previous response was not valid JSON. Return ONLY the JSON object, nothing else.` added to the prompt after a blank line; a second failure is `{"error": ...}`. Values outside the schema are replaced by cautious defaults (a missing `exposure` reads as `moderate`, a missing `pose` as `mildly_suggestive`, `confidence` as `low`), never by the permissive end; the high-stakes flags (minor, nudity, see-through) are only set by a positive observation.

Every valid observation carries a `meta` stamp: `schema_version`, `prompt_version`, `model_name` (`Qwen3.5-9B`), `precision` (`INT8 ConvRot`) and `frames_analyzed`.

**Errors never break the graph.** Anything that goes wrong (the model file, the download, the GPU, an empty batch) comes back as `{"error": "<type>: <message>"}` with the traceback in the console, and Report then gives `UNKNOWN` for that run. Cancelling the queue still stops the run.

## `BC_PreFlightReport` — PreFlight Report

`observations_json`, `caption`, `log_prediction`; optional `image` → `report_json`, `summary`, `record_id` (`STRING`), `image` (`IMAGE`).

| Input | Default | Meaning |
| --- | --- | --- |
| `observations_json` | input only | Observe's `observations_json`. |
| `caption` | empty, multiline | The caption to be published (optional). |
| `log_prediction` | on | Append this prediction to the [feedback store](#the-feedback-loop). |
| `image` | optional | Passed through unchanged, so the node can sit inline. |

| Output | Meaning |
| --- | --- |
| `report_json` | The full report: the observation, the caption flags, the rules that fired in order, the verdicts, the range drivers, the engine version, the record id. |
| `summary` | The verdicts to read, one line per platform, then the range drivers and `record: <id>`. |
| `record_id` | The prediction's id in the store; empty when logging is off or the store cannot be written (the verdicts still come out). |
| `image` | The `image` input. |

The caption **and** the text the model read inside the image (`visible_text`) are scanned together, so an adult-platform watermark burned into a frame flags exactly like a link in the caption. Motion rules apply only to a video (more than one frame analysed).

### Reading a verdict

Every verdict is a **range**, `best → worst`:

- **OK** ✅: no expected problem.
- **RISK** ⚠️: **reach demotion** (recommendation exclusion, general down-ranking). The post stays up but reaches fewer people.
- **BLOCK** ❌: **removal** (Instagram, TikTok) or **TikTok For-You-Feed ineligibility**: a hard limit.
- **UNKNOWN** ❓: the sensor failed or its observation is from an unknown schema; judge for yourself.

`OK → RISK` means "probably fine, at worst demoted"; `RISK → BLOCK` means "demoted, and possibly removed depending on a named factor". `best` is the favourable case (clean account, neutral context), `worst` the cautious ceiling. When `best` and `worst` differ on Instagram or TikTok, a **range driver** names what would collapse the range (the setting, the sensor's confidence, …). **BLOCK at `worst` is reserved for genuine policy triggers**: nudity, see-through, significant exposure, adult solicitation or links, explicit sexual motion, or a *sexualized* depiction of an apparent minor. Soft signals (framing, pose, mild exposure, suggestive text) top out at **RISK**: they demote, they do not remove. On X the verdict marks whether an adult label is recommended.

> **Apparent minors.** The block fires only when an apparent minor is shown in a **sexualized** context (exposure, a swim or intimate garment, a suggestive pose or framing, sexual motion or text). A clothed, neutral subject is ordinary content, and a youthful-looking adult is not flagged as a minor.

## `BC_PreFlightOutcome` — PreFlight Outcome

`record`, `platform`, `result`, `record_id_override` → `status` (`STRING`). An output node.

| Input | Default | Meaning |
| --- | --- | --- |
| `record` | the newest | The prediction to attach the outcome to: the store's 50 newest, newest first. |
| `platform` | `instagram` | `instagram` / `tiktok` / `x`. |
| `result` | `clean` | `clean` / `demoted` / `removed`. |
| `record_id_override` | empty | When set, this record id is used instead of the `record` selection. |

`status` says what was logged, e.g. `logged: a3f9c2d1 instagram=demoted`, or why nothing was (no record selected, an unknown platform); errors come back in the string, never raised. The node has no image input: outcomes attach to records, not files.

**The record list follows the store.** Each prediction is listed by a label built from the record itself, so no file-name convention is needed to find it later:

```
a3f9c2d1 · 07-24 14:02 · IG:RISK TT:RISK X:RISK · bikini/beach_pool · "summer drop 🌞 lim…"
```

(the id, the time logged in UTC, the worst verdict per platform, garment/setting, the start of the caption). The list is current without a page reload or Refresh: when Report logs a prediction, the server sends the new list to every open browser and each Outcome node shows it at once; a node gets the current list when it is added or a workflow is loaded (`web/js/preflight.js`, in both the classic canvas and the Vue node renderer). Only the list changes: a saved workflow's `record` value is never rewritten on load, even when the list no longer holds it, and the node still runs it (its id is the label's first token). A node you add starts on the newest record. For a prediction older than the 50 listed, paste its id (shown in the Report summary) into `record_id_override`, which bypasses the list.

Log **each platform separately**: queue once per platform. Every queue writes, even with the same inputs.

**Pick one fixed personal heuristic for `demoted` and apply it consistently.** `demoted` = clear reach suppression, an account-status flag or FYF ineligibility. Decide *your* threshold (e.g. "40% or more below my median reach") and stick to it: inconsistent labels poison the calibration. `clean` = normal performance; `removed` = taken down.

## `BC_PreFlightCalibrate` — PreFlight Calibrate

No inputs → `report` (`STRING`). Connect it to [Show Text](text.md#bc_showtext--show-text). It reads the store and changes nothing:

1. **Per platform**: how often the real outcome fell inside the predicted `[best, worst]` range (consistent), **above** it (under-predicted, the dangerous kind) or **below** it (over-predicted: reach lost on content that was fine). `UNKNOWN` verdicts are skipped.
2. **Per rule**: for each rule ID that fired, how many posts it fired on and the clean / demoted / removed counts of their outcomes per platform. This is the table that answers *which rule is miscalibrated*: e.g. `base.bikini` fired 30 times and 28 came back `clean` on Instagram → the rule is too harsh there.

It runs again only when the store has changed (a prediction or an outcome logged since its last run); otherwise the queue returns the cached tables.

## The feedback loop

The rules are **hypotheses** about platform behaviour, and this loop is how they get corrected. Nothing here is machine learning and nothing tunes itself: predictions and real outcomes accumulate in a log, Calibrate turns the log into per-rule statistics, and **a person** edits the rules (`pipelines/preflight/rules.py`) and bumps `ENGINE_VERSION`. Entering outcomes is optional: a prediction without one costs nothing and is skipped.

**What is logged.** With `log_prediction` on, each Report run appends **one prediction line** to the store, an append-only JSONL file in ComfyUI's user directory:

```
ComfyUI/user/BCNodes/preflight/feedback.jsonl
```

holding the observation, the first 80 characters of the caption, the caption flags, the IDs of the rules that fired, the verdicts and the engine and schema versions, keyed by an 8-character record id (shown in the summary as `record: a3f9c2d1` and on the `record_id` output). Outcome appends one outcome line per queue; the two are joined by id when the store is read, and for one record and platform the last outcome logged wins. Nothing is ever rewritten in place, and a corrupt line is skipped. No other place is read: a store kept anywhere else (such as an older `output/preflight/feedback.jsonl`) is not picked up; move its file here to keep its records.

**Changing the rules.** Every prediction records the engine version that produced it, so after a rule edit the old and the new predictions can be told apart in the calibration tables (`engine_versions seen`).

## Determinism and re-evaluating old content

Decoding is greedy and the observations are stored apart from the verdicts with their `schema_version` / `prompt_version` stamp, so old observations can be judged again by new rules without the model (`rules.judge` over a stored observation). An observation whose schema version the current engine does not know **fails closed** to `UNKNOWN` instead of being misjudged.

## Limitations

Read this before trusting a verdict:

- **It is a prediction**, from a visual observation and a **static rule set**, not a guarantee.
- **Platform classifiers are black boxes.** The thresholds in the rules are informed hypotheses, not the platforms' actual policies, which change without notice.
- **Enforcement depends on signals this node cannot see**: account history, bio, follower and link patterns, posting cadence, cluster signals, region. Two identical images on two accounts can be treated differently.
- **The rules need calibration against your own outcomes.** Out of the box they are a starting point; the feedback loop exists because they will be wrong until tuned.
- **The sensor can misread.** A low-confidence observation widens the verdict ranges on purpose.
- **Process Monitor.** Emulate lists Observe as not counted: no widget names its model file, so the weights, the KV cache and the frames resized for the model are not in the estimate. Report is counted (its image output is the input passed on); Outcome and Calibrate output text only.

PreFlight reports. A person decides.

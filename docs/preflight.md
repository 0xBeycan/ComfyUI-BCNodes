# PreFlight

How Instagram, TikTok and X would likely treat a generated image or video, before you publish it. PreFlight flags two risks per platform:

- **Hard limit** (`BLOCK`): removal (violates community standards).
- **Soft limit** (`RISK`): reach suppression (Instagram's recommendation exclusion, TikTok's For-You-Feed ineligibility). The post stays up.

A local chat model, picked on an [LM Loader](lm.md#bc_lmloader--lm-loader) (its default: Qwen3.5-9B, INT8 ConvRot), acts as a *pure observation sensor*: it describes what is visually present and does **not** judge. Its observations go into a **deterministic rules engine** that maps them to per-platform verdict ranges. A feedback loop records what the platform actually did, so the rules can be checked against reality over time.

> **An assistive signal, not an approval gate.** It never says "approved" or "rejected". Verdicts are ranges (best … worst); the report names the unknown behind each range, so you can collapse it yourself. The final judgment is yours.

## Why four nodes

| Node | What it does |
| --- | --- |
| [PreFlight Observe](#bc_preflightobserve--preflight-observe) | Runs the model and returns a structured **observations JSON** (garment, exposure, framing, pose, in-image text, …; [the schema](#the-observation-schema)). |
| [PreFlight Report](#bc_preflightreport--preflight-report) | Applies the rules engine: **per-platform verdicts**, a summary to read, and the prediction logged to the feedback store. |
| [PreFlight Outcome](#bc_preflightoutcome--preflight-outcome) | Records what a platform actually did to a published post, days later. |
| [PreFlight Calibrate](#bc_preflightcalibrate--preflight-calibrate) | Turns the store into the calibration tables: how often the predictions held, per platform and per rule. |

Observation and judgment are split on purpose: when a verdict looks wrong you need to know whether **perception** failed (the model misread the garment) or the **rules** did. Keeping the observations as a separate record also lets old content be judged again by updated rules **without running the model again**: every observation carries the schema and prompt version that produced it.

A typical graph: Load Image and [LM Loader](lm.md#bc_lmloader--lm-loader) → **PreFlight Observe** → **PreFlight Report** (`image` passed through to a preview, `summary` into [Show Text](text.md#bc_showtext--show-text)). Days after posting: a **PreFlight Outcome** per platform. From time to time: **PreFlight Calibrate** → Show Text.

## `BC_PreFlightObserve` — PreFlight Observe

`image`, `lm`, `max_frames` → `observations_json` (`STRING`), `raw_response` (`STRING`).

| Input | Default | Range | Meaning |
| --- | --- | --- | --- |
| `image` | | `IMAGE` | A single image or a video's frame batch. |
| `lm` | | `BC_LM`, required | The model, from an [LM Loader](lm.md#bc_lmloader--lm-loader): its model, precision, LoRA at its strength, and `keep_model_loaded` (on: the model stays loaded after the node, in the one LM model slot it shares with Qwen LM; off: unloaded once after the observation, its retry included, after an error too). |
| `max_frames` | `6` | `1`–`16` | For a video: how many frames are sampled evenly from the batch. All of them go to the model in **one** call, so it can tell motion from the sequence. |

| Output | Meaning |
| --- | --- |
| `observations_json` | The observation, or `{"error": ...}` when the run failed. |
| `raw_response` | The model's answer as it came, both answers when it was retried, for debugging. |

**The model.** Observe is a wrapper: it lists no model and runs the one its LM Loader picks, the LoRA included. LM Loader's default is **Qwen3.5-9B, INT8 ConvRot**. It goes through the same runtime as [Qwen LM](lm.md#bc_qwenlm--qwen-lm): the model file is found in `models/Qwen-LM` or under `models/text_encoders`, or downloaded into `models/Qwen-LM` on the first run ([Models and files](lm.md#models-and-files)), and ComfyUI core runs it. One LM model is held at a time, shared with the other LM nodes: a run with another model file replaces it. Qwen3.5-9B INT8 ConvRot peaks at about **10.5 GB** of VRAM here (measured on an RTX 3090: one image or a 4-frame batch).

**The run.** Whatever the model, the run is the same. The sampled frames go first, then the observation prompt as the user text, in the model's official chat template ([Prompt and thinking](lm.md#prompt-and-thinking)): no system turn, thinking off, no prefill. Decoding is pure greedy (the most likely token at each step, no penalties), seed 42, at most 300 new tokens, so the same image and settings give the same observation run to run, a requirement for calibration. The model is asked for a strict JSON object; the first `{…}` of its answer is read, whatever prose or code fence surrounds it. An answer with no usable object gets **exactly one** retry, with `Your previous response was not valid JSON. Return ONLY the JSON object, nothing else.` added to the prompt after a blank line; a second failure is `{"error": ...}`. A missing value, or one outside the [schema](#the-observation-schema), is replaced by its cautious default, never by the permissive end; the high-stakes flags (minor, nudity, see-through) are only set by a positive observation.

Every valid observation carries a `meta` stamp: `schema_version`, `prompt_version`, the LM Loader's `model_name` (e.g. `Qwen3.5-9B`), `precision` (e.g. `INT8 ConvRot`), `lora` (`None` without one) and `lora_strength`, and `frames_analyzed`. Another model or LoRA observes differently, so the stamp tells which one a logged prediction came from.

**Errors never break the graph.** Anything that goes wrong (the model file, the download, the GPU, an empty batch) comes back as `{"error": "<type>: <message>"}` with the traceback in the console, and Report then gives `UNKNOWN` for that run. Cancelling the queue still stops the run.

### The observation schema

Schema version `1`: the keys the model is asked for (`pipelines/preflight/prompts.py`). For a video, the prompt asks for the appearance keys of the most revealing frame.

| Key | Values | Default |
| --- | --- | --- |
| `subject_appears_under_18` | `true` / `false`: concrete visual signs of a minor; a youthful-looking adult is `false` | `false` |
| `garment` | the most revealing garment: `bikini`, `lingerie`, `underwear_only`, `miniskirt`, `croptop`, `shorts`, `dress`, `fitness_wear`, `swimsuit_onepiece`, `regular`, `other` | `croptop` |
| `setting` | `beach_pool`, `bedroom`, `indoor_other`, `outdoor`, `gym`, `studio` | `indoor_other` |
| `framing` | `face_centric`, `full_body`, `chest_focus`, `butt_focus`, `lowangle_body` | `full_body` |
| `pose` | `neutral`, `mildly_suggestive`, `suggestive` | `mildly_suggestive` |
| `see_through_or_wet` | `true` / `false` | `false` |
| `exposure` | `none`; `mild` (bare arms, legs, midriff, a low neckline); `moderate` (the side or underside of the breast, part of the buttocks); `significant` (underwear only, nipple covers only, implied nudity) | `moderate` |
| `nudity_or_sexual_act` | `true` / `false` | `false` |
| `motion_flags` | `none`, `twerk_grind_striptease`, `intimate_kissing`; the rules read it only for a video | `none` |
| `visible_text` | the text readable in the image (watermarks, overlays, usernames, URLs), verbatim | `""` |
| `confidence` | `high`, `medium`, `low` | `low` |

The default replaces a missing value or one outside the list. A boolean also reads from `"true"` / `"yes"` / `1` and their opposites; a `visible_text` that is not a string reads as `""`. Observe adds `meta`; the model does not.

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
- **RISK** ⚠️: **reach demotion** (recommendation exclusion, general down-ranking, **TikTok For-You-Feed ineligibility**). The post stays up but reaches fewer people.
- **BLOCK** ❌: **removal** (violates community standards).
- **UNKNOWN** ❓: the sensor failed or its observation is from an unknown schema; judge for yourself.

`OK → RISK` means "probably fine, at worst demoted"; `RISK → BLOCK` means "demoted, and possibly removed". `best` is the favourable case (clean account, neutral context), `worst` the cautious ceiling. When `best` and `worst` differ on Instagram or TikTok, the summary carries a **range driver** that says why: the sensor's low confidence, or otherwise soft demotion risk (the low end assumes a clean account and neutral context, the high end an unlucky classifier pass).

**What gives BLOCK.** At `high` or `medium` confidence, only these: nudity, see-through or wet fabric and twerk, grind or striptease motion in a video (Instagram, TikTok); sexual solicitation (Instagram); an adult-platform link in the caption or the image (Instagram, TikTok); a sexualized depiction of an apparent minor (all three). Everything else tops out at **RISK**: minimal coverage, significant exposure and a bikini (on TikTok, For-You-Feed ineligibility; for a bikini, indoors), moderate exposure, revealing casualwear with a pose other than `neutral`, and the soft signals (framing, pose, intimate kissing, age-coded or sexual text), which never raise a verdict above RISK by themselves. Each rule's trigger and effect is in [The rules](#the-rules).

**Low confidence widens the range.** With `confidence` `low` (also the default for a missing value) every verdict that is not hard moves one level down at `best` and one level up at `worst`: `OK → OK` becomes `OK → RISK`, `OK → RISK` and `RISK → RISK` become `OK → BLOCK`, `BLOCK → BLOCK` becomes `RISK → BLOCK`. So at low confidence a soft signal, and an X verdict, can reach BLOCK at `worst`. The hard verdicts are never widened: solicitation (Instagram), the adult-platform link (Instagram, TikTok) and the apparent-minor override.

**On X** the verdict is about labelling. `RISK` means the post may need X's adult-content label; the X line of the summary reads `adult label recommended` when `report_json` sets `label_required` (nudity, significant exposure, see-through, lingerie or underwear only, a suggestive pose, twerk, grind or striptease motion), and its reasons in `report_json` add that the post is not for a profile photo, header, banner or live thumbnail. X reaches BLOCK only through the apparent-minor override (removal and report risk) or the low-confidence widening.

> **Apparent minors.** The override fires only when an apparent minor is shown with a suggestive signal: `moderate` or `significant` exposure, see-through or nudity, a pose other than `neutral`, a bikini, one-piece swimsuit, lingerie or underwear only, a sexualized framing (`chest_focus`, `butt_focus`, `lowangle_body`), any motion flag in a video, or any [text flag](#the-rules). Every platform is then `BLOCK → BLOCK`, hard, and no later rule runs. A clothed, neutral subject is ordinary content (`mild` exposure alone does not fire it), and a youthful-looking adult is not flagged as a minor.

### The rules

The stages run in a fixed order, and the ID of every rule that fires is appended to `fired_rules` in that order: the IDs Calibrate's per-rule table counts. Apart from the low-confidence widening, a rule never lowers a verdict: each platform keeps the highest `best` and the highest `worst` any rule gave it. `—`: no change. The rules live in `pipelines/preflight/rules.py`.

| Stage | Rule ID | Fires when | Instagram | TikTok | X |
| --- | --- | --- | --- | --- | --- |
| 0 | `guard.fail_closed` | the observation is an error, not a JSON object, or its `meta.schema_version` is not one the engine knows; no later stage runs | UNKNOWN | UNKNOWN | UNKNOWN |
| 1 | `override.minor` | `subject_appears_under_18` with a suggestive signal ([Apparent minors](#reading-a-verdict)); hard, no later stage runs | BLOCK | BLOCK | BLOCK |
| 2 | `base.nudity` | `nudity_or_sexual_act` | BLOCK | BLOCK | RISK |
| 2 | `base.minimal` | `garment` `lingerie` or `underwear_only`, or `exposure` `significant` | RISK | RISK (FYF ineligible) | RISK |
| 2 | `base.bikini` | `garment` `bikini` | RISK | RISK (FYF ineligible when `setting` is `bedroom` or `indoor_other`) | OK → RISK |
| 2 | `base.see_through` | `see_through_or_wet` | BLOCK | BLOCK | RISK |
| 2 | `base.exposure_mod` | `exposure` `moderate` | OK → RISK | OK → RISK | — |
| 2 | `base.exposure_mild` | `garment` `miniskirt`, `croptop`, `shorts` or `fitness_wear` with a `pose` other than `neutral` | OK → RISK | OK → RISK | — |
| 2 | `base.default` | no other base rule fired | — | — | — |
| 3 | `mod.framing` | `framing` `chest_focus`, `butt_focus` or `lowangle_body`, with `moderate` or `significant` exposure, a `pose` other than `neutral`, a bikini, lingerie, underwear only or see-through | worst at least RISK | worst at least RISK | — |
| 3 | `mod.pose_suggestive` | `pose` `suggestive` | best and worst at least RISK | worst at least RISK | — |
| 3 | `mod.pose_mild` | `pose` `mildly_suggestive` | — | worst at least RISK | — |
| 3 | `mod.motion_twerk` | a video with `motion_flags` `twerk_grind_striptease` | BLOCK | BLOCK | RISK |
| 3 | `mod.motion_kiss` | a video with `motion_flags` `intimate_kissing` | — | worst at least RISK | — |
| 3 | `mod.text_age` | text flag `age_marker` or `flagged_emoji` | worst at least RISK | worst at least RISK | — |
| 3 | `mod.sexual_text` | text flag `sexual_text` | worst at least RISK | worst at least RISK | — |
| 4 | `hard.solicitation` | text flag `adult_platform_mention`, `price_or_subscription_cta` or `adult_link`, with `exposure` other than `none`, a `pose` other than `neutral`, a bikini, lingerie or underwear only | BLOCK, hard | — | — |
| 4 | `hard.adult_link` | text flag `adult_link` | BLOCK, hard | BLOCK, hard | — |
| 5 | `mod.confidence_low` | `confidence` `low` | widened unless hard | widened unless hard | widened |

`base.*` rules all fire when they match, in the order listed. Stage 6 sets X's `label_required` (no rule ID). `OK → RISK` in a cell is the range that rule gives (`best → worst`).

**Text flags** come from the caption and `visible_text` together, by exact pattern match, and are listed in `report_json`'s `caption_flags`: `adult_platform_mention` (OnlyFans, Fanvue, Fansly), `price_or_subscription_cta` ("DM for" more, content, pics, vids, access or fun; "pay to", "subscribe", "sub for", "PPV", "exclusive content"; "DM for collabs" does not match), `age_marker` ("18+", "NSFW", "spicy" content, pics, vids or link), `adult_link` (fanvue.com, onlyfans.com, fansly.com), `sexual_text` (overt sexual words such as "sex", "nudes", "porn"; "sexy" does not match), `flagged_emoji` (the peach, eggplant, sweat droplets, tongue and hot face emoji).

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

**Pick one fixed personal heuristic for `demoted` and apply it consistently.** Calibrate compares each result with one verdict level: `clean` (normal performance) with OK, `demoted` (clear reach suppression, an account-status flag or TikTok For-You-Feed ineligibility) with RISK, `removed` (taken down) with BLOCK. Decide *your* threshold for `demoted` (e.g. "40% or more below my median reach") and stick to it: inconsistent labels poison the calibration.

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

holding the observation, the first 80 characters of the caption, the caption flags, the IDs of the rules that fired, the verdicts and the engine and schema versions, keyed by an 8-character record id (shown in the summary as `record: a3f9c2d1` and on the `record_id` output). Outcome appends one outcome line per queue; the two are joined by id when the store is read, and for one record and platform the last outcome logged wins. Nothing is ever rewritten in place, and a corrupt line is skipped.

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
- **Process Monitor.** Emulate lists Observe as not counted: its LM Loader names the model by catalog name, not by file, so the weights, the LoRA, the KV cache and the frames resized for the model are not in the estimate. Report is counted (its image output is the input passed on); Outcome and Calibrate output text only.

PreFlight reports. A person decides.

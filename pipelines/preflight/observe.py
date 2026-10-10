"""PreFlight Observe's flow: the observation sensor run over an image batch, as one strict JSON observation.

observe(image, max_frames, generator), in this order:
  1. up to `max_frames` frames sampled evenly from the batch (sample_frames: index selection on the
     tensor; the batch itself when every frame is kept);
  2. generator(frames) -> generate(extra_instruction) -> the model's raw text. The node builds it over the
     LM runtime: MODEL at PRECISION, the frames then user_text(extra_instruction) as the user turn, no
     system turn, thinking off, no prefill, greedy (GREEDY), SEED, MAX_NEW_TOKENS;
  3. the first JSON object of the answer (extract_json); an answer without one, or an object with no
     observation key, gets exactly one retry with RETRY_INSTRUCTION appended to the user text after a
     blank line; a second failure is {"error": ...} with no meta, which rules.judge fails closed on;
  4. prompts.validate_observations coerces the object to the schema, and the meta stamp records the schema
     and prompt versions, the model, its precision and the frames analysed.
Fail closed: an error comes back as {"error": "<type>: <message>"} with its traceback on the console, never
raised into the graph. ComfyUI's interrupt is a BaseException (comfy.model_management.InterruptProcessingException)
and passes, so cancelling the queue still stops the run.
"""

import json
import logging

import numpy as np
import torch

from . import prompts

MODEL = "Qwen3.5-9B"          # the Qwen LM catalog's name (models/qwen_lm/models.yaml)
PRECISION = "INT8 ConvRot"    # that model's precision
SEED = 42
MAX_NEW_TOKENS = 300
# Pure greedy, the same tokens run to run: core takes the most likely token at each step with do_sample off and
# applies no penalty then; the penalties are given their off values too, so the run stays plain greedy whatever
# the model's defaults carry (Qwen3.5's presence_penalty is 1.5). The fields of an LM Config's edited set.
GREEDY = {"do_sample": False, "repetition_penalty": 1.0, "presence_penalty": 0.0}
RETRY_INSTRUCTION = ("Your previous response was not valid JSON. Return ONLY "
                     "the JSON object, nothing else.")


def sample_frames(image, max_frames):
    """Up to `max_frames` frames of the IMAGE batch `image` (N, H, W, C), evenly spaced: the rounded positions
    of linspace(0, N - 1, count), each frame once. The batch itself when every frame is kept, else a new
    tensor holding only the picked frames. ValueError on an empty batch."""
    batch = int(image.shape[0])
    if batch == 0:
        raise ValueError("image holds no frames: connect an image or a frame batch")
    count = max(1, min(int(max_frames), batch))
    picks = np.unique(np.linspace(0, batch - 1, count).round().astype(int))
    if len(picks) == batch:
        return image
    return image.index_select(0, torch.from_numpy(picks).to(image.device))


def user_text(extra_instruction):
    """The user turn: the observation prompt, then `extra_instruction` (the retry's) after a blank line."""
    prompt = prompts.build_prompt()
    return prompt + "\n\n" + extra_instruction if extra_instruction else prompt


def extract_json(raw):
    """Return the first balanced ``{...}`` object parsed from ``raw``, or None.

    Scans for the first '{' and its matching '}', ignoring braces inside JSON
    strings — this transparently skips markdown fences and any prose the model
    wraps around the object.
    """
    if not isinstance(raw, str):
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        c = raw[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except ValueError:
                    return None
    return None


def meta(frames_analyzed):
    """The meta block stamped into every valid observation."""
    return {
        "schema_version": prompts.SCHEMA_VERSION,
        "prompt_version": prompts.PROMPT_VERSION,
        "model_name": MODEL,
        "precision": PRECISION,
        "frames_analyzed": int(frames_analyzed),
    }


def observe_from_generate(generate, frames_analyzed):
    """Drive generation -> parse -> validate, with exactly one retry.

    ``generate(extra_instruction)`` returns raw model text. On the first
    unparseable / non-observation response we retry once with the corrective
    instruction; a second failure returns ``{"error": ...}`` (no meta — the
    error key alone makes rules.judge fail closed). Returns ``(observations,
    raw_response)``.
    """
    raw = generate("")
    parsed = extract_json(raw)

    if not prompts.has_observation_keys(parsed):
        raw2 = generate(RETRY_INSTRUCTION)
        raw = raw + "\n\n--- retry ---\n\n" + raw2
        parsed = extract_json(raw2)
        if not prompts.has_observation_keys(parsed):
            return {"error": "model did not return valid observations JSON after "
                             "one retry"}, raw

    observations = prompts.validate_observations(parsed)
    observations["meta"] = meta(frames_analyzed)
    return observations, raw


def observe(image, max_frames, generator):
    """(observations_json, raw_response) of one sensor run (module docstring); `generator(frames)` returns the
    run's generate callable."""
    raw = ""
    try:
        frames = sample_frames(image, max_frames)
        observations, raw = observe_from_generate(generator(frames), len(frames))
    except Exception as exc:  # noqa: BLE001  fail closed: Report gives UNKNOWN for this run
        logging.exception("[BCNodes] PreFlight Observe: the observation failed; Report gives UNKNOWN for this run")
        observations = {"error": "%s: %s" % (type(exc).__name__, exc)}
    return json.dumps(observations, ensure_ascii=False), raw

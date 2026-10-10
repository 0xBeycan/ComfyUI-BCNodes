"""PreFlight Observe's flow (pipelines/preflight/observe.py) with a stand-in generate, no ComfyUI, no model:

  - extract_json: the first balanced object, braces inside strings ignored, fences and prose skipped;
  - observe_from_generate: one call on a valid answer, exactly one retry (the corrective instruction) on an
    unusable one, {"error": ...} without meta after the second, enums coerced to the cautious defaults; the
    meta stamp (schema and prompt versions, model, precision, frames analysed);
  - sample_frames: evenly spaced picks (rounded linspace, each frame once) by index selection on the tensor,
    the batch itself when every frame is kept, dtype and device kept, an empty batch refused;
  - user_text: the prompt verbatim, the retry's instruction after a blank line;
  - observe: the generator gets the sampled frames; any error comes back as {"error": "<type>: <message>"}
    with the traceback logged, never raised; ComfyUI's interrupt (a BaseException) passes;
  - the decoding contract: Qwen3.5-9B INT8 ConvRot, seed 42, 300 tokens, greedy with the penalties off.
"""

import json
import logging

import pytest
import torch

VALID_JSON = json.dumps({
    "subject_appears_under_18": False, "garment": "bikini", "setting": "beach_pool",
    "framing": "full_body", "pose": "neutral", "see_through_or_wet": False,
    "exposure": "mild", "nudity_or_sexual_act": False, "motion_flags": "none",
    "visible_text": "", "confidence": "high",
})


@pytest.fixture
def ob(bcnodes):
    return bcnodes["pipelines.preflight.observe"]


@pytest.fixture
def prompts(bcnodes):
    return bcnodes["pipelines.preflight.prompts"]


# --- extract_json -------------------------------------------------------------


@pytest.mark.parametrize("raw,ok", [
    ('{"a": 1}', True),
    ('```json\n{"a": 1}\n```', True),                    # fenced
    ('Sure! Here you go:\n{"a": 1}\nHope that helps', True),  # prose around it
    ('{"a": {"b": 2}, "c": 3}', True),                   # nested
    ('{"text": "has } brace in string"}', True),         # brace inside string
    ('no json here', False),
    ('', False),
    (None, False),
])
def test_extract_json(ob, raw, ok):
    result = ob.extract_json(raw)
    assert (result is not None) == ok


def test_extract_json_brace_in_string_value(ob):
    r = ob.extract_json('{"visible_text": "50% off {sale}"}')
    assert r == {"visible_text": "50% off {sale}"}


# --- observe_from_generate: retry + meta ----------------------------------------


def test_observe_valid_first_try_single_call(ob, prompts):
    calls = []

    def gen(extra):
        calls.append(extra)
        return VALID_JSON

    obs, raw = ob.observe_from_generate(gen, 6)
    assert calls == [""]                                 # exactly one call, no retry
    assert "error" not in obs and raw == VALID_JSON
    assert obs["garment"] == "bikini"
    assert obs["meta"] == {"schema_version": prompts.SCHEMA_VERSION, "prompt_version": prompts.PROMPT_VERSION,
                           "model_name": "Qwen3.5-9B", "precision": "INT8 ConvRot", "frames_analyzed": 6}


def test_observe_retries_once_then_succeeds(ob):
    outs = iter(["not json at all", "```json\n" + VALID_JSON + "\n```"])
    calls = []

    def gen(extra):
        calls.append(extra)
        return next(outs)

    obs, raw = ob.observe_from_generate(gen, 1)
    assert len(calls) == 2 and calls[1] == ob.RETRY_INSTRUCTION
    assert obs["garment"] == "bikini"
    assert raw == "not json at all\n\n--- retry ---\n\n```json\n" + VALID_JSON + "\n```"  # both kept for debugging


def test_observe_retries_on_an_object_without_observation_keys(ob):
    outs = iter(['{"foo": 1}', VALID_JSON])
    obs, _ = ob.observe_from_generate(lambda extra: next(outs), 1)
    assert obs["garment"] == "bikini"


def test_observe_two_failures_returns_error_no_meta(ob):
    calls = []

    def gen(extra):
        calls.append(extra)
        return "still not json"

    obs, raw = ob.observe_from_generate(gen, 1)
    assert len(calls) == 2                               # exactly one retry, then stop
    assert "error" in obs and "meta" not in obs          # error obs -> rules fail closed


def test_observe_coerces_unknown_enum(ob):
    bad = json.dumps({**json.loads(VALID_JSON), "exposure": "extreme"})
    obs, _ = ob.observe_from_generate(lambda e: bad, 1)
    assert obs["exposure"] == "moderate"                 # coerced to cautious default


# --- sample_frames ----------------------------------------------------------------


def numbered(batch):
    """A batch whose frame i holds the value i."""
    return torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1).expand(batch, 2, 2, 3).contiguous()


@pytest.mark.parametrize("batch,max_frames,picks", [
    (10, 6, [0, 2, 4, 5, 7, 9]),        # linspace 0, 1.8, 3.6, 5.4, 7.2, 9
    (10, 3, [0, 4, 9]),                 # 4.5 rounds half to even
    (5, 2, [0, 4]),
    (7, 4, [0, 2, 4, 6]),
    (20, 16, [0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19]),
    (3, 1, [0]),
])
def test_sample_frames_picks(ob, batch, max_frames, picks):
    assert ob.sample_frames(numbered(batch), max_frames)[:, 0, 0, 0].int().tolist() == picks


def test_sample_frames_spread(ob):
    """min(max_frames, batch) frames, each once, in order, the first and the last frame among them."""
    for batch in range(1, 40):
        for max_frames in range(1, 17):
            picked = ob.sample_frames(numbered(batch), max_frames)[:, 0, 0, 0].int().tolist()
            count = min(max_frames, batch)
            assert len(picked) == count and picked == sorted(set(picked)) and picked[0] == 0, (batch, max_frames)
            assert count == 1 or picked[-1] == batch - 1, (batch, max_frames)


def test_sample_frames_keeps_the_batch_when_every_frame_is_kept(ob):
    image = torch.rand(4, 8, 8, 3)
    assert ob.sample_frames(image, 6) is image           # fewer frames than max_frames: no copy
    assert ob.sample_frames(image, 4) is image
    single = torch.rand(1, 8, 8, 3)
    assert ob.sample_frames(single, 1) is single


def test_sample_frames_selects_without_touching_the_input(ob):
    image = torch.rand(10, 4, 4, 3, dtype=torch.float16)
    before = image.clone()
    out = ob.sample_frames(image, 3)
    assert out.dtype == torch.float16 and out.shape == (3, 4, 4, 3) and out.device == image.device
    assert torch.equal(out, image[[0, 4, 9]])            # linspace(0, 9, 3) = 0, 4.5 -> 4 (half to even), 9
    assert out.data_ptr() != image.data_ptr() and torch.equal(image, before)
    assert ob.sample_frames(image, 1)[:, 0, 0, 0].tolist() == image[:1, 0, 0, 0].tolist()


def test_sample_frames_refuses_an_empty_batch(ob):
    with pytest.raises(ValueError, match="no frames"):
        ob.sample_frames(torch.zeros(0, 8, 8, 3), 6)


# --- user_text and the decoding contract --------------------------------------------


def test_user_text(ob, prompts):
    assert ob.user_text("") == prompts.build_prompt() == prompts.OBSERVATION_PROMPT
    assert ob.user_text(ob.RETRY_INSTRUCTION) == prompts.OBSERVATION_PROMPT + "\n\n" + ob.RETRY_INSTRUCTION


def test_decoding_contract(ob, bcnodes):
    assert (ob.MODEL, ob.PRECISION, ob.SEED, ob.MAX_NEW_TOKENS) == ("Qwen3.5-9B", "INT8 ConvRot", 42, 300)
    assert ob.GREEDY == {"do_sample": False, "repetition_penalty": 1.0, "presence_penalty": 0.0}
    # an LM Config value: its keys are LMConfigValues fields, in widget order, typed as the node casts them
    fields = bcnodes["pipelines.lm"].LMConfigValues.__annotations__
    assert list(ob.GREEDY) == [name for name in fields if name in ob.GREEDY]
    assert all(isinstance(value, fields[name]) for name, value in ob.GREEDY.items())
    assert ob.RETRY_INSTRUCTION == "Your previous response was not valid JSON. Return ONLY the JSON object, nothing else."


# --- observe: frames, fail closed -----------------------------------------------------


def test_observe_hands_the_sampled_frames_to_the_generator(ob):
    seen = []

    def generator(frames):
        seen.append(frames)
        return lambda extra: VALID_JSON

    image = torch.rand(10, 8, 8, 3)
    obs_json, raw = ob.observe(image, 6, generator)
    obs = json.loads(obs_json)
    assert len(seen) == 1 and seen[0].shape == (6, 8, 8, 3)  # one generator, all frames in one call
    assert obs["garment"] == "bikini" and obs["meta"]["frames_analyzed"] == 6 and raw == VALID_JSON


def test_observe_never_raises_on_model_error(ob, caplog):
    def generator(frames):
        def boom(extra):
            raise RuntimeError("CUDA out of memory")
        return boom

    with caplog.at_level(logging.ERROR):
        obs_json, raw = ob.observe(torch.rand(1, 8, 8, 3), 6, generator)
    assert json.loads(obs_json) == {"error": "RuntimeError: CUDA out of memory"} and raw == ""
    assert "PreFlight Observe" in caplog.text and "Traceback" in caplog.text and "CUDA out of memory" in caplog.text


def test_observe_errors_before_the_model_fail_closed_too(ob):
    def generator(frames):
        raise FileNotFoundError("no model file")

    assert json.loads(ob.observe(torch.rand(2, 8, 8, 3), 6, generator)[0]) == {"error": "FileNotFoundError: no model file"}
    obs_json, raw = ob.observe(torch.zeros(0, 8, 8, 3), 6, generator)  # an empty batch
    assert json.loads(obs_json)["error"].startswith("ValueError: image holds no frames") and raw == ""


def test_observe_lets_the_interrupt_through(ob):
    class InterruptProcessingException(BaseException):  # as comfy.model_management's
        pass

    def generator(frames):
        def stop(extra):
            raise InterruptProcessingException()
        return stop

    with pytest.raises(InterruptProcessingException):
        ob.observe(torch.rand(1, 8, 8, 3), 6, generator)

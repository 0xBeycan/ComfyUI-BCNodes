"""Captioner's flow (pipelines/captioner/caption.py) with a recording start / generate pair (no LM runtime):

  - the user text per style: the built-in prompts read from prompts/<style>.txt at call time (no leading blank
    line, no header or GENDER line, nothing about other captions; the tags prompt names no placeholder word and
    shows a left-out field by example), "" for none; the instruction replaces the
    style's text with any style, as it is; an empty or whitespace instruction counts as not connected; an
    unknown style is an error naming the styles;
  - the batch split into frames in order, each a (1, H, W, C) view of the input (same storage, same dtype),
    start called once after the input checks (never for an unknown style, an empty batch or a tensor that is
    not a batch, which are errors), then one generate per frame with the same text;
  - the frame the model sees: each frame through libs/image.thumbnail_frame at MAX_SIDE 1024 (its own cases are
    tests/layers/libs/test_lib_image.py), so a large frame reaches generate capped; the frames returned are the
    originals;
  - the caption: the model's answer as written, its surrounding whitespace stripped, for every style.
"""

import os

import pytest
import torch


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.captioner.caption"]


class Generate:
    """generate(frame, user): records (frame, user), answers the next of `answers` (the last one again when
    they run out). start() (the flow's start callable) counts its calls in `starts` and returns it."""

    def __init__(self, answers=("woman, standing",)):
        self.calls, self.answers, self.starts = [], list(answers), 0

    def start(self):
        assert not self.calls, "start comes before the first generate"
        self.starts += 1
        return self

    def __call__(self, frame, user):
        self.calls.append((frame, user))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def prompt_file(flow, style):
    with open(os.path.join(flow.PROMPTS_DIR, f"{style}.txt"), encoding="utf-8") as f:
        return f.read()


# --- the user text -------------------------------------------------------------------------------------------


def test_the_styles_in_widget_order(flow):
    assert flow.STYLES == ("none", "tags", "prose")


@pytest.mark.parametrize("style", ["tags", "prose"])
def test_a_style_is_its_built_in_prompt(flow, style):
    text = flow.user_text(style, None)
    assert text == prompt_file(flow, style).strip()
    assert text.startswith("You are an image captioning tool for LoRA character training.\nThe first token of every "
                           "caption")
    assert '"woman" or "man"' in text and text.endswith(".")


def test_the_prompts_open_with_the_text_and_leave_out_the_header_and_other_captions(flow):
    for style in ("tags", "prose"):
        raw = prompt_file(flow, style)
        assert raw.startswith("You are"), style  # the leading blank lines are gone
        lowered = raw.lower()
        for gone in ("header", "gender", "previous caption", "later"):
            assert gone not in lowered, (style, gone)
    tags, prose = prompt_file(flow, "tags"), prompt_file(flow, "prose")
    assert "<class word>, shot type, pose/action" in tags and "\nEXAMPLES:\n" in tags
    assert "Image: a man, tight face crop, downward gaze, plain backdrop." in tags
    # the tags prompt never names the word it must not write: a left-out field is phrased positively and shown
    for word in ("none", "n/a", "placeholder", "no accessories", "no jewelry"):
        assert word not in tags.lower(), word
    assert "Every field after the class word is optional: write a field only when the image shows it." in tags
    assert "--- Example 7 (a field with nothing to tag) ---" in tags
    assert "VARY THE STRUCTURE" not in prose and "CONSISTENT WORDING" not in prose
    assert '— "woman" or "man" — for the person in the image, then' in prose


def test_the_prompt_is_read_at_call_time(flow, monkeypatch, tmp_path):
    (tmp_path / "tags.txt").write_text("\n\nmy own tags prompt\n", encoding="utf-8")
    monkeypatch.setattr(flow, "PROMPTS_DIR", str(tmp_path))
    assert flow.user_text("tags", None) == "my own tags prompt"


def test_none_is_the_frame_alone(flow):
    assert flow.user_text("none", None) == ""


@pytest.mark.parametrize("style", ["none", "tags", "prose"])
def test_the_instruction_replaces_the_style_as_it_is(flow, style):
    assert flow.user_text(style, "  Describe the outfit.\n") == "  Describe the outfit.\n"


@pytest.mark.parametrize("instruction", [None, "", "   ", "\n\t "])
def test_an_empty_instruction_is_not_connected(flow, instruction):
    assert flow.user_text("prose", instruction) == flow.user_text("prose", None)
    assert flow.user_text("none", instruction) == ""


def test_an_unknown_style_is_an_error(flow):
    with pytest.raises(ValueError, match="style 'poem' is not one of none, tags, prose"):
        flow.user_text("poem", "an instruction")


# --- frames --------------------------------------------------------------------------------------------------


def test_one_generate_per_frame_in_order_with_the_original_frames(flow):
    image = torch.rand(3, 32, 48, 3, dtype=torch.float16)
    for i in range(3):
        image[i, 0, 0, 0] = i / 4  # each frame told apart
    generate = Generate(["c0", "c1", "c2"])
    frames, captions = flow.caption(image, "prose", None, generate.start)
    assert captions == ["c0", "c1", "c2"]
    assert len(frames) == 3 and len(generate.calls) == 3 and generate.starts == 1
    for i, frame in enumerate(frames):
        assert frame.shape == (1, 32, 48, 3) and frame.dtype == torch.float16
        assert frame.data_ptr() == image[i].data_ptr() and frame.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
        assert generate.calls[i][0] is not None and float(generate.calls[i][0][0, 0, 0, 0]) == i / 4
    assert {user for _, user in generate.calls} == {flow.user_text("prose", None)}


def test_a_single_image_is_one_frame(flow):
    image = torch.rand(1, 16, 16, 4)
    frames, captions = flow.caption(image, "none", None, Generate(["only"]).start)
    assert captions == ["only"] and frames[0].data_ptr() == image.data_ptr() and frames[0].shape == (1, 16, 16, 4)


def test_an_empty_batch_is_an_error_saying_what_to_connect(flow):
    generate = Generate()
    with pytest.raises(ValueError, match="image holds no frames: connect an image or a batch with at least one frame"):
        flow.caption(torch.zeros(0, 16, 16, 3), "prose", None, generate.start)
    assert generate.calls == [] and generate.starts == 0


def test_a_tensor_that_is_not_a_batch_is_an_error(flow):
    generate = Generate()
    with pytest.raises(ValueError, match=r"image must be an IMAGE batch .* got shape \(16, 16, 3\)"):
        flow.caption(torch.zeros(16, 16, 3), "prose", None, generate.start)
    assert generate.starts == 0


def test_an_unknown_style_starts_nothing(flow):
    generate = Generate()
    with pytest.raises(ValueError, match="style 'poem'"):
        flow.caption(torch.zeros(1, 8, 8, 3), "poem", None, generate.start)
    assert generate.starts == 0


# --- the frame the model sees --------------------------------------------------------------------------------


def test_the_model_gets_the_capped_frame_the_output_keeps_the_original(flow, bcnodes):
    image = torch.rand(2, 1200, 1600, 3)
    generate = Generate()
    frames, _ = flow.caption(image, "tags", None, generate.start)
    assert flow.MAX_SIDE == 1024
    assert [tuple(f.shape) for f, _ in generate.calls] == [(1, 768, 1024, 3)] * 2
    thumbnail_frame = bcnodes["libs.image"].thumbnail_frame
    assert all(torch.equal(f, thumbnail_frame(image[i:i + 1], 1024)) for i, (f, _) in enumerate(generate.calls))
    assert [tuple(f.shape) for f in frames] == [(1, 1200, 1600, 3)] * 2
    assert all(f.data_ptr() == image[i].data_ptr() for i, f in enumerate(frames))


# --- the caption ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("style, instruction", [("tags", None), ("prose", None), ("none", None),
                                                ("tags", "List the tags."), ("none", "Describe it.")])
def test_the_caption_is_the_answer_as_written(flow, style, instruction):
    """Every style, with or without an instruction: the model's answer, its surrounding whitespace stripped
    and nothing else (a slot the tags prompt asks to leave out is the prompt's job, not a filter's)."""
    generate = Generate(["  woman, standing, none, beach \n"])
    _, captions = flow.caption(torch.rand(1, 8, 8, 3), style, instruction, generate.start)
    assert captions == ["woman, standing, none, beach"]

"""Prompter's flow (pipelines/prompter/write.py) with a recording start / generate pair (no LM runtime):

  - the instructions values in widget order; the instruction text per path and mode: the built-in prompt read
    from prompts/<path>-<include | exclude>.txt at call time, its surrounding whitespace stripped; "" for none;
    an instruction override as it is, with any instructions value; an empty or whitespace override counts as
    not connected; an unknown instructions value is an error naming the values;
  - every row of the table: text only, image only, text + image, with include / exclude identity and with none
    (the user text, and the image the model gets or None); the same rows with an override; the brief as typed,
    the instruction and the brief joined by one blank line, no blank line when the instruction text is empty;
  - the errors, raised before start: no brief and no image (a whitespace brief is none), an image batch of
    more or fewer than one frame (also when a brief means the image would not be used); start called once;
  - the image the model sees: within MAX_SIDE 1024 the input itself, above it libs/image.thumbnail_frame's;
  - the prompt: the model's answer, its surrounding whitespace stripped;
  - the prompt files: the four exist; no GENDER header, HEADER_ERROR, quality tail or 4K quality is left; each
    ends on its final line; every example opens with the shot type and camera height and ends on a content
    sentence; the brief prompts hold an example about a man; the exclude prompts name the class word after the
    framing ("of one woman" / "of one man").
"""

import os
import re

import pytest
import torch

BRIEF = "kadın, kafede kahve içiyor"


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.prompter.write"]


class Generate:
    """generate(frame, user): records (frame, user), answers `answer`. start() counts its calls in `starts`."""

    def __init__(self, answer="  A half-body shot at eye level of one woman.\n"):
        self.calls, self.answer, self.starts = [], answer, 0

    def start(self):
        assert not self.calls, "start comes before the generate"
        self.starts += 1
        return self

    def __call__(self, frame, user):
        self.calls.append((frame, user))
        return self.answer


def prompt_file(flow, name):
    with open(os.path.join(flow.PROMPTS_DIR, name), encoding="utf-8") as f:
        return f.read()


def run(flow, text, image, instructions, instruction=None):
    generate = Generate()
    out = flow.write(text, image, instructions, instruction, generate.start)
    assert generate.starts == 1 and len(generate.calls) == 1
    return out, generate.calls[0]


# --- the instruction text ------------------------------------------------------------------------------------


def test_the_instructions_in_widget_order(flow):
    assert flow.INSTRUCTIONS == ("none", "include identity", "exclude identity")


@pytest.mark.parametrize("instructions, path, name", [
    ("include identity", "brief", "brief-include.txt"), ("exclude identity", "brief", "brief-exclude.txt"),
    ("include identity", "image", "image-include.txt"), ("exclude identity", "image", "image-exclude.txt")])
def test_a_mode_is_its_built_in_prompt(flow, instructions, path, name):
    assert flow.instruction_text(instructions, None, path) == prompt_file(flow, name).strip()


def test_the_prompt_is_read_at_call_time(flow, monkeypatch, tmp_path):
    (tmp_path / "image-exclude.txt").write_text("\n my own image prompt \n\n", encoding="utf-8")
    monkeypatch.setattr(flow, "PROMPTS_DIR", str(tmp_path))
    assert flow.instruction_text("exclude identity", None, "image") == "my own image prompt"


@pytest.mark.parametrize("path", ["brief", "image"])
def test_none_has_no_instruction_text(flow, path):
    assert flow.instruction_text("none", None, path) == ""


@pytest.mark.parametrize("instructions", ["none", "include identity", "exclude identity"])
@pytest.mark.parametrize("path", ["brief", "image"])
def test_the_override_replaces_the_mode_as_it_is(flow, instructions, path):
    assert flow.instruction_text(instructions, "  Write a haiku.\n", path) == "  Write a haiku.\n"


@pytest.mark.parametrize("instruction", [None, "", "   ", "\n\t "])
def test_an_empty_override_is_not_connected(flow, instruction):
    assert flow.instruction_text("include identity", instruction, "brief") == flow.instruction_text(
        "include identity", None, "brief")
    assert flow.instruction_text("none", instruction, "image") == ""


def test_an_unknown_instructions_value_is_an_error(flow):
    with pytest.raises(ValueError, match="instructions 'poem' is not one of none, include identity, exclude identity"):
        flow.instruction_text("poem", "an override", "brief")
    generate = Generate()
    with pytest.raises(ValueError, match="instructions 'poem'"):
        flow.write(BRIEF, None, "poem", None, generate.start)
    assert generate.starts == 0


# --- the table -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("instructions, name", [("include identity", "brief-include.txt"),
                                                ("exclude identity", "brief-exclude.txt")])
def test_text_only_is_the_brief_prompt_then_the_brief(flow, instructions, name):
    _, (frame, user) = run(flow, BRIEF, None, instructions)
    assert frame is None and user == prompt_file(flow, name).strip() + "\n\n" + BRIEF


@pytest.mark.parametrize("instructions, name", [("include identity", "image-include.txt"),
                                                ("exclude identity", "image-exclude.txt")])
def test_image_only_is_the_image_then_the_image_prompt(flow, instructions, name):
    image = torch.rand(1, 32, 48, 3)
    _, (frame, user) = run(flow, "", image, instructions)
    assert frame is image and user == prompt_file(flow, name).strip()


@pytest.mark.parametrize("instructions, name", [("include identity", "brief-include.txt"),
                                                ("exclude identity", "brief-exclude.txt")])
def test_text_and_image_is_the_brief_prompt_and_the_image_is_not_used(flow, instructions, name):
    _, (frame, user) = run(flow, BRIEF, torch.rand(1, 32, 48, 3), instructions)
    assert frame is None and user == prompt_file(flow, name).strip() + "\n\n" + BRIEF


def test_none_text_only_is_the_text_alone(flow):
    _, (frame, user) = run(flow, BRIEF, None, "none")
    assert frame is None and user == BRIEF


def test_none_image_only_is_the_image_alone(flow):
    image = torch.rand(1, 32, 48, 3)
    _, (frame, user) = run(flow, "", image, "none")
    assert frame is image and user == ""


def test_none_text_and_image_is_the_image_then_the_text(flow):
    image = torch.rand(1, 32, 48, 3)
    _, (frame, user) = run(flow, BRIEF, image, "none")
    assert frame is image and user == BRIEF


def test_the_brief_goes_in_as_typed(flow):
    brief = "  two lines\nof brief  "
    assert run(flow, brief, None, "none")[1][1] == brief
    assert run(flow, brief, None, "include identity")[1][1].endswith("\n\n" + brief)


@pytest.mark.parametrize("instructions", ["none", "include identity", "exclude identity"])
def test_the_override_with_text_only(flow, instructions):
    _, (frame, user) = run(flow, BRIEF, None, instructions, "Rewrite this.")
    assert frame is None and user == "Rewrite this.\n\n" + BRIEF


@pytest.mark.parametrize("instructions", ["none", "include identity", "exclude identity"])
def test_the_override_with_image_only(flow, instructions):
    image = torch.rand(1, 32, 48, 3)
    _, (frame, user) = run(flow, "  \n", image, instructions, "Describe it.")
    assert frame is image and user == "Describe it."


@pytest.mark.parametrize("instructions, uses_image", [("none", True), ("include identity", False),
                                                      ("exclude identity", False)])
def test_the_override_with_text_and_image(flow, instructions, uses_image):
    image = torch.rand(1, 32, 48, 3)
    _, (frame, user) = run(flow, BRIEF, image, instructions, "Rewrite this.")
    assert (frame is image) is uses_image and (frame is None) is not uses_image
    assert user == "Rewrite this.\n\n" + BRIEF


# --- errors --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
@pytest.mark.parametrize("instructions", ["none", "include identity", "exclude identity"])
def test_no_brief_and_no_image_is_an_error_before_start(flow, text, instructions):
    generate = Generate()
    with pytest.raises(ValueError, match="Prompter has nothing to work from: type a brief in text, or connect an "
                                         "image"):
        flow.write(text, None, instructions, "an override", generate.start)
    assert generate.starts == 0


@pytest.mark.parametrize("frames", [0, 2, 5])
@pytest.mark.parametrize("text", ["", BRIEF])
def test_an_image_of_other_than_one_frame_is_an_error_before_start(flow, frames, text):
    generate = Generate()
    with pytest.raises(ValueError, match=f"Prompter takes one image, got a batch of {frames} frames: connect a single "
                                         "image"):
        flow.write(text, torch.rand(frames, 16, 16, 3), "include identity", None, generate.start)
    assert generate.starts == 0


# --- the image the model sees and the answer -----------------------------------------------------------------


def test_a_large_image_is_capped_at_1024(flow, bcnodes):
    image = torch.rand(1, 1500, 2000, 4)
    _, (frame, _) = run(flow, "", image, "exclude identity")
    assert flow.MAX_SIDE == 1024 and frame.shape == (1, 768, 1024, 3)
    assert torch.equal(frame, bcnodes["libs.image"].thumbnail_frame(image, 1024))


@pytest.mark.parametrize("h, w", [(1024, 1024), (600, 800)])
def test_an_image_within_the_cap_is_the_input(flow, h, w):
    image = torch.rand(1, h, w, 3, dtype=torch.float16)
    assert run(flow, "", image, "none")[1][0] is image


def test_the_prompt_is_the_answer_stripped(flow):
    assert run(flow, BRIEF, None, "include identity")[0] == "A half-body shot at eye level of one woman."


def test_the_decoding_and_budget(flow):
    assert flow.DECODING == {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                             "presence_penalty": 0.0, "repetition_penalty": 1.0}
    assert flow.MAX_NEW_TOKENS == 600


# --- the prompt files ----------------------------------------------------------------------------------------

FILES = ["brief-include.txt", "brief-exclude.txt", "image-include.txt", "image-exclude.txt"]
SHOTS = ("extreme close-up", "close-up", "chest-up shot", "half-body shot", "three-quarter shot", "full-body shot")


def examples(text):
    return re.findall(r"^Output: (.*)$", text, re.M)


def test_the_four_prompt_files_exist_and_nothing_else(flow):
    assert sorted(os.listdir(flow.PROMPTS_DIR)) == sorted(FILES)


@pytest.mark.parametrize("name", FILES)
def test_no_header_and_no_quality_tail_is_left(flow, name):
    text = prompt_file(flow, name)
    for gone in ("GENDER", "HEADER_ERROR", "header", "4K quality", "QUALITY TAIL", "quality tail", "photorealistic",
                 "ultra-detailed", "style label", "composition clause"):
        assert gone not in text, (name, gone)
    assert not text.startswith("\n") and text.startswith("You are a prompt-writing tool")


@pytest.mark.parametrize("name, last", [
    ("brief-include.txt", "Write only the single-line prompt for the user's input."),
    ("brief-exclude.txt", "Write only the single-line prompt for the user's input."),
    ("image-include.txt", "Write only the single-line prompt for the reference photograph."),
    ("image-exclude.txt", "Write only the single-line prompt for the reference photograph.")])
def test_each_file_ends_on_its_final_line(flow, name, last):
    assert prompt_file(flow, name).strip().splitlines()[-1] == last


@pytest.mark.parametrize("name", FILES)
def test_every_example_opens_with_the_framing_and_ends_on_content(flow, name):
    outputs = examples(prompt_file(flow, name))
    assert len(outputs) >= 3
    for output in outputs:
        opening = re.match(r"An? (.+?) (at eye level|from a [a-z ]+ angle) of ", output)
        assert opening and opening.group(1) in SHOTS, output[:80]
        rest = output[opening.end():]
        assert not any(shot in rest for shot in SHOTS) and " eye level" not in rest, output  # the framing once
        assert output.endswith(".") and 150 <= len(output.split()) <= 200, (name, len(output.split()))


@pytest.mark.parametrize("name", ["brief-include.txt", "brief-exclude.txt"])
def test_the_brief_prompts_have_an_example_about_a_man(flow, name):
    text = prompt_file(flow, name)
    pairs = re.findall(r'^User input: "(.*)"\nOutput: (.*)$', text, re.M)
    assert any("adam" in brief and re.search(r"\bof (one|a) man\b", output) for brief, output in pairs), name


@pytest.mark.parametrize("name", ["image-include.txt", "image-exclude.txt"])
def test_the_image_prompts_have_an_example_of_a_man(flow, name):
    text = prompt_file(flow, name)
    pairs = re.findall(r"^Reference: (.*)\nOutput: (.*)$", text, re.M)
    assert any(re.search(r"\bman\b", reference) and re.search(r"\bof (one|a [A-Z][a-z]+) man\b", output)
               for reference, output in pairs), name


@pytest.mark.parametrize("name", ["brief-exclude.txt", "image-exclude.txt"])
def test_the_exclude_prompts_name_the_class_word_after_the_framing(flow, name):
    text = prompt_file(flow, name)
    assert all(re.search(r" of one (woman|man) [a-z]+ing ", output) for output in examples(text))
    assert 'The class word is "woman" or "man"' in text
    assert "the very first token" not in text and "no article" not in text

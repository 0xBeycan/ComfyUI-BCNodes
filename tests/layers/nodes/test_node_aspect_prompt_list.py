"""Aspect Prompt List, entered through its node method.

Prompt lines are grouped under [WxH] headers; each prompt carries its header's size, so the
prompt, body_text, width and height lists always have the same length. Blank lines are skipped;
a prompt above the first header, a zero side or a text with no prompts is a ValueError.
"""

import pytest


@pytest.fixture
def make_list(bcnodes):
    return bcnodes["prompt_list"].AspectPromptList().make_list


def test_each_prompt_carries_its_section_size(make_list):
    text = "[1280x1280]\nsquare a\nsquare b\n[1024x1536]\nportrait a\n[1536x1024]\nlandscape a"
    prompts, body, widths, heights = make_list(text)
    assert prompts == ["square a", "square b", "portrait a", "landscape a"]
    assert widths == [1280, 1280, 1024, 1536]
    assert heights == [1280, 1280, 1536, 1024]
    assert len(body) == len(prompts)


def test_prepend_and_append_wrap_prompt_not_body(make_list):
    prompts, body, _, _ = make_list("[8x8]\nline", prepend_text="<", append_text=">")
    assert prompts == ["<line>"]
    assert body == ["line"]


def test_blank_lines_and_surrounding_spaces_are_dropped(make_list):
    prompts, _, widths, _ = make_list("\n  [ 16 X 8 ]  \n\n  a  \n\n\nb\n")
    assert prompts == ["a", "b"]
    assert widths == [16, 16]


def test_header_without_prompts_adds_nothing(make_list):
    prompts, _, widths, heights = make_list("[8x8]\n[16x24]\na")
    assert (prompts, widths, heights) == (["a"], [16], [24])


def test_bracketed_prompt_line_that_is_not_a_size_stays_a_prompt(make_list):
    prompts, _, _, _ = make_list("[8x8]\n[soft light] portrait")
    assert prompts == ["[soft light] portrait"]


def test_prompt_before_first_header_is_an_error(make_list):
    with pytest.raises(ValueError, match=r"line 1 is a prompt before any size header"):
        make_list("orphan\n[8x8]\na")


def test_zero_side_is_an_error(make_list):
    with pytest.raises(ValueError, match=r"line 1 '\[0x8\]' has a zero size"):
        make_list("[0x8]\na")


@pytest.mark.parametrize("text", ["", None, "\n\n", "[8x8]\n"])
def test_no_prompt_lines_is_an_error(make_list, text):
    with pytest.raises(ValueError, match=r"no prompt lines"):
        make_list(text)

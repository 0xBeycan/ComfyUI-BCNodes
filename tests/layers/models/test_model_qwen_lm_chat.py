"""models/qwen_lm/chat.py: the raw chat prompts against the official chat templates' renderings, written
out below (transformers' render_jinja_template on Qwen/Qwen3.5-9B, Qwen/Qwen3.8-27B and
Qwen/Qwen3-VL-8B-Instruct chat_template, add_generation_prompt / continue_final_message, enable_thinking
true or false; the build spec's verified strings among them):

  - qwen3.5: system turn only with a system text, stripped texts, images before the text, an empty closed
    think block (thinking off) followed by the stripped prefill, or an open one (thinking on); the user
    content is stripped with its images, so after an image the text keeps its leading whitespace;
  - qwen3.8: the same, plus the xhigh reasoning-effort instruction in the system turn with thinking on
    (the turn created when there is no system text);
  - qwen3-vl: no think block, texts verbatim;
  - a blank system (whitespace only) is no system turn for every template, as a blank user is an empty
    user; every catalog model renders its template's prompt;
  - the checks: an empty user with no image, thinking on a model without it, thinking with a prefill, a
    negative image count, a <|image_pad|> typed in system, user or assistant (each one stands for an image
    frame); a text holding only parts of it renders as before;
  - split_output: reasoning / answer at the first </think>, the cut-off thinking case (one warning), the
    prefill joined to its continuation (which keeps its leading space or newline), qwen3-vl's verbatim
    prefill.
"""

import logging

import pytest

S = "You are a helpful assistant."
U = "Describe this."
X = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
     "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.")
IMG1 = "<|vision_start|><|image_pad|><|vision_end|>"
IMG2 = IMG1 * 2


@pytest.fixture(scope="module")
def chat(bcnodes):
    return bcnodes["models.qwen_lm.chat"]


@pytest.fixture(scope="module")
def models(bcnodes):
    family = bcnodes["models.qwen_lm"].FAMILY
    catalog = bcnodes["models.common.lm.catalog"].load_catalog(
        family.catalog_path, [], family.templates, default_backend=family.default_backend)
    return {"qwen3.5": catalog["Qwen3.5-9B"], "qwen3.8": catalog["Qwen3.8-27B"], "qwen3-vl": catalog["Qwen3-VL-8B Instruct"]}


@pytest.fixture(scope="module")
def parts(bcnodes):
    PromptParts = bcnodes["models.common.lm.family"].PromptParts

    def make(system=S, user=U, assistant="", n_images=0, thinking=False):
        return PromptParts(system=system, user=user, assistant=assistant, n_images=n_images, thinking=thinking)
    return make


OFFICIAL = [
    # qwen3.5: the spec's verified strings
    ("qwen3.5", {}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.5", {"thinking": True}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.5", {"n_images": 2}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG2}{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.5", {"assistant": "ABC"}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nABC"),
    # qwen3.5: the other official renderings
    ("qwen3.5", {"system": ""}, f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.5", {"system": "", "thinking": True}, f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.5", {"n_images": 2, "thinking": True}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG2}{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.5", {"assistant": "  ABC  "}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nABC"),
    ("qwen3.5", {"assistant": "<think>\nI should"}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\nI should"),
    ("qwen3.5", {"system": "  S \n", "user": "\n U  "}, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.5", {"system": "  S \n", "user": "\n U  ", "thinking": True}, "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    # qwen3.5 / 3.8 trim the user content with its images: after an image the text's leading whitespace stays
    ("qwen3.5", {"user": "\n\nDescribe the scene.", "n_images": 1},
     f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG1}\n\nDescribe the scene.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.5", {"system": "  S \n", "user": "\n U  ", "n_images": 2},
     f"<|im_start|>system\nS<|im_end|>\n<|im_start|>user\n{IMG2}\n U<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    # qwen3.8
    ("qwen3.8", {"thinking": True}, f"<|im_start|>system\n{X}\n\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.8", {"system": "", "thinking": True}, f"<|im_start|>system\n{X}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.8", {}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.8", {"system": ""}, f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.8", {"n_images": 2, "thinking": True}, f"<|im_start|>system\n{X}\n\n{S}<|im_end|>\n<|im_start|>user\n{IMG2}{U}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.8", {"n_images": 2}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG2}{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3.8", {"assistant": "ABC"}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nABC"),
    ("qwen3.8", {"system": "  S \n", "user": "\n U  ", "thinking": True}, f"<|im_start|>system\n{X}\n\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.8", {"system": "  S \n", "user": "\n U  ", "n_images": 2, "thinking": True},
     f"<|im_start|>system\n{X}\n\nS<|im_end|>\n<|im_start|>user\n{IMG2}\n U<|im_end|>\n<|im_start|>assistant\n<think>\n"),
    ("qwen3.8", {"user": "  lead and trail  ", "n_images": 1, "assistant": "ABC"},
     f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG1}  lead and trail<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nABC"),
    ("qwen3.8", {"system": "", "user": "", "n_images": 1, "thinking": True},
     f"<|im_start|>system\n{X}<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<think>\n"),
    # qwen3-vl
    ("qwen3-vl", {}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n"),
    ("qwen3-vl", {"system": ""}, f"<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n"),
    ("qwen3-vl", {"n_images": 2}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG2}{U}<|im_end|>\n<|im_start|>assistant\n"),
    ("qwen3-vl", {"assistant": "ABC"}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\nABC"),
    ("qwen3-vl", {"assistant": "  ABC  "}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\n  ABC  "),
    ("qwen3-vl", {"assistant": "plan</think>\n\nABC"}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{U}<|im_end|>\n<|im_start|>assistant\nplan</think>\n\nABC"),
    ("qwen3-vl", {"system": "  S \n", "user": "\n U  "}, "<|im_start|>system\n  S \n<|im_end|>\n<|im_start|>user\n\n U  <|im_end|>\n<|im_start|>assistant\n"),
    ("qwen3-vl", {"user": "\n U  ", "n_images": 1}, f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\n{IMG1}\n U  <|im_end|>\n<|im_start|>assistant\n"),
]


@pytest.mark.parametrize("template, given, expected", OFFICIAL)
def test_prompt_is_the_official_rendering(chat, models, parts, template, given, expected):
    assert chat.build_prompt(models[template], parts(**given)) == expected


def test_a_blank_system_is_no_system_turn(chat, models, parts):
    # blank = empty for every template (qwen3-vl's texts go in verbatim, but a blank system is no system
    # text), as a blank user is an empty user
    for template, thinking in (("qwen3.5", False), ("qwen3.5", True), ("qwen3.8", False), ("qwen3.8", True), ("qwen3-vl", False)):
        blank = chat.build_prompt(models[template], parts(system=" \n ", thinking=thinking))
        assert blank == chat.build_prompt(models[template], parts(system="", thinking=thinking))
    assert chat.build_prompt(models["qwen3.8"], parts(system="\t\n", thinking=True)).startswith(f"<|im_start|>system\n{X}<|im_end|>\n")


def test_every_catalog_model_takes_its_template(chat, bcnodes, parts):
    family = bcnodes["models.qwen_lm"].FAMILY
    catalog = bcnodes["models.common.lm.catalog"].load_catalog(
        family.catalog_path, [], family.templates, default_backend=family.default_backend)
    by_template = {}
    for model in catalog.values():
        by_template.setdefault(model.template, set()).add(chat.build_prompt(model, parts(n_images=1)))
    assert {t: len(p) for t, p in by_template.items()} == {"qwen3.5": 1, "qwen3-vl": 1, "qwen3.8": 1}


def test_qwen3_vl_image_alone(chat, models, parts):
    assert chat.build_prompt(models["qwen3-vl"], parts(system="", user="", n_images=1)) == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n")


def test_a_negative_image_count_is_an_error(chat, models, parts):
    with pytest.raises(ValueError, match="n_images is -1"):
        chat.build_prompt(models["qwen3.5"], parts(n_images=-1))


def test_an_image_alone_needs_no_user_text(chat, models, parts):
    assert chat.build_prompt(models["qwen3.5"], parts(system="", user="  ", n_images=1)) == (
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


@pytest.mark.parametrize("template", ["qwen3.5", "qwen3.8", "qwen3-vl"])
@pytest.mark.parametrize("user", ["", "  \n"])
def test_empty_user_without_image_is_an_error(chat, models, parts, template, user):
    with pytest.raises(ValueError, match="user is empty"):
        chat.build_prompt(models[template], parts(user=user))


def test_thinking_on_a_model_without_it_is_an_error(chat, models, parts):
    with pytest.raises(ValueError, match="has no thinking mode"):
        chat.build_prompt(models["qwen3-vl"], parts(thinking=True))


def test_thinking_on_a_qwen35_entry_without_thinking_is_an_error(chat, models, parts):
    import dataclasses
    model = dataclasses.replace(models["qwen3.5"], thinking=False)
    with pytest.raises(ValueError, match="Qwen3.5-9B has no thinking mode: turn thinking off"):
        chat.build_prompt(model, parts(thinking=True))


def test_qwen3_vl_template_never_thinks_even_if_an_entry_says_so(chat, models, parts):
    import dataclasses
    model = dataclasses.replace(models["qwen3-vl"], thinking=True)
    with pytest.raises(ValueError, match="has no thinking mode"):
        chat.build_prompt(model, parts(thinking=True))


@pytest.mark.parametrize("template", ["qwen3.5", "qwen3.8"])
def test_thinking_with_a_prefill_is_an_error(chat, models, parts, template):
    with pytest.raises(ValueError, match="cannot be used with thinking on"):
        chat.build_prompt(models[template], parts(assistant="ABC", thinking=True))
    assert chat.build_prompt(models[template], parts(assistant=" \n", thinking=True)).endswith("<think>\n")


PAD_REFUSED = r"<\|image_pad\|> in {}: it stands for an image .*; remove <\|image_pad\|> from the system, user and assistant text"


@pytest.mark.parametrize("template", ["qwen3.5", "qwen3.8", "qwen3-vl"])
@pytest.mark.parametrize("field", ["system", "user", "assistant"])
@pytest.mark.parametrize("n_images", [0, 1])
def test_a_typed_image_pad_is_an_error(chat, models, parts, template, field, n_images):
    # core keeps a typed <|image_pad|> as the image token: one with no image, or one taking another
    # frame's place; refused here, before any model file is looked for
    with pytest.raises(ValueError, match=PAD_REFUSED.format(field)):
        chat.build_prompt(models[template], parts(**{field: "Compare with <|image_pad|> here."}, n_images=n_images))


def test_every_field_holding_a_typed_image_pad_is_named(chat, models, parts):
    with pytest.raises(ValueError, match=PAD_REFUSED.format("system and assistant")):
        chat.build_prompt(models["qwen3.5"], parts(system="<|image_pad|>", assistant=IMG2))


@pytest.mark.parametrize("template, given, expected", [
    ("qwen3.5", {"user": "image_pad <image_pad> <|image|> |image_pad|"},
     f"<|im_start|>system\n{S}<|im_end|>\n<|im_start|>user\nimage_pad <image_pad> <|image|> |image_pad|<|im_end|>\n"
     "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("qwen3-vl", {"system": "<|vision_start|><|vision_end|>", "n_images": 1},
     "<|im_start|>system\n<|vision_start|><|vision_end|><|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
     f"{U}<|im_end|>\n<|im_start|>assistant\n"),
])
def test_text_without_a_whole_image_pad_renders_as_before(chat, models, parts, template, given, expected):
    assert chat.IMAGE_BLOCK == "<|vision_start|><|image_pad|><|vision_end|>"
    assert chat.build_prompt(models[template], parts(**given)) == expected


def test_unknown_template_is_an_error(chat, models, parts):
    import dataclasses
    with pytest.raises(ValueError, match="not one of"):
        chat.build_prompt(dataclasses.replace(models["qwen3.5"], template="qwen9"), parts())


@pytest.mark.parametrize("raw, expected", [
    ("I look at it.\n</think>\n\nA red bicycle.\n", ("A red bicycle.", "I look at it.")),
    ("\n</think>\n\nShort.", ("Short.", "")),
    ("a</think>b</think>c", ("b</think>c", "a")),
])
def test_split_with_thinking(chat, models, parts, raw, expected):
    assert chat.split_output(models["qwen3.5"], raw, parts(thinking=True)) == expected


@pytest.mark.parametrize("raw, thinking", [(" still weighing the options\n", "still weighing the options"), ("", "")])
def test_split_cut_off_while_thinking(chat, models, parts, caplog, raw, thinking):
    with caplog.at_level(logging.WARNING):
        assert chat.split_output(models["qwen3.8"], raw, parts(thinking=True)) == ("", thinking)
    assert [r.getMessage() for r in caplog.records] == [
        "[BCNodes] Qwen LM: Qwen3.8-27B stopped while thinking (no </think>; max_new_tokens reached?): text is empty "
        "and thinking holds all of the output"]


@pytest.mark.parametrize("template, assistant, raw, expected", [
    ("qwen3.5", "", "\n\nA red bicycle.\n", "A red bicycle."),
    ("qwen3.5", "  The image shows", " a red bicycle.\n", "The image shows a red bicycle."),
    ("qwen3.5", '{"name": "', 'bicycle"}\n', '{"name": "bicycle"}'),
    ("qwen3.8", "ABC", "DEF  ", "ABCDEF"),
    ("qwen3.5", "```json", '\n{"a": 1}\n```\n', '```json\n{"a": 1}\n```'),
    ("qwen3.5", "", "", ""),
    ("qwen3-vl", "", "  A red bicycle.  \n", "A red bicycle."),
    ("qwen3-vl", "  The image shows", " a red bicycle.\n", "  The image shows a red bicycle."),
])
def test_split_without_thinking(chat, models, parts, template, assistant, raw, expected):
    assert chat.split_output(models[template], raw, parts(assistant=assistant)) == (expected, "")


def test_split_without_thinking_keeps_think_tags_it_did_not_open(chat, models, parts):
    # no cleaning: a think block the model writes on its own stays in the text
    assert chat.split_output(models["qwen3.5"], "<think>x</think>\n\nY", parts()) == ("<think>x</think>\n\nY", "")

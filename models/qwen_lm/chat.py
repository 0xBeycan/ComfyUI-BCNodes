"""Qwen chat prompts and outputs as the official chat templates render them (core's own Qwen templates are
not used): qwen3.5 (Qwen3.5-4B / 9B and their derivatives), qwen3.8 (Qwen3.8-27B) and qwen3-vl (Qwen3-VL-8B
Instruct and its derivatives).

- qwen3.5 / qwen3.8: system and prefill are stripped; the images come before the user text, and the user
  content (images and text together, as the template trims it) is stripped, so after an image the text
  keeps its leading whitespace; the assistant turn opens with `<think>\\n` (thinking on) or an empty,
  closed think block followed by the prefill (thinking off). qwen3.8 with thinking on also writes its
  default reasoning effort (xhigh) into the system turn, creating the turn when there is no system text.
- qwen3-vl: no think block, no stripping: the texts go in verbatim.

A blank system text (empty or whitespace only) means no system turn. A <|image_pad|> typed in a text is
refused: the prompt holds exactly one per image frame.
"""

import logging

QWEN35, QWEN38, QWEN3VL = "qwen3.5", "qwen3.8", "qwen3-vl"
TEMPLATES = (QWEN35, QWEN38, QWEN3VL)
IMAGE_PAD = "<|image_pad|>"  # the tokenizer's image placeholder: one per frame, written only by IMAGE_BLOCK
IMAGE_BLOCK = f"<|vision_start|>{IMAGE_PAD}<|vision_end|>"
# The system instruction Qwen3.8's template adds with thinking on: its default reasoning_effort, xhigh.
REASONING_XHIGH = ("Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                   "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                   "clarity in the final answer.")
THINK_OPEN = "<think>\n"
THINK_EMPTY = "<think>\n\n</think>\n\n"
THINK_END = "</think>"


def build_prompt(model, parts):
    """The raw chat prompt of `parts` for the catalog `model`, ending where the model's answer starts."""
    _check(model, parts)
    images = IMAGE_BLOCK * parts.n_images
    if model.template == QWEN3VL:
        head = f"<|im_start|>system\n{parts.system}<|im_end|>\n" if parts.system.strip() else ""
        return f"{head}<|im_start|>user\n{images}{parts.user}<|im_end|>\n<|im_start|>assistant\n{parts.assistant}"
    system = parts.system.strip()
    if parts.thinking and model.template == QWEN38:
        system = f"{REASONING_XHIGH}\n\n{system}" if system else REASONING_XHIGH
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    answer = THINK_OPEN if parts.thinking else THINK_EMPTY + parts.assistant.strip()
    user = (images + parts.user).strip()  # the template trims the whole content: after an image the text's start stays
    return f"{head}<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{answer}"


def _check(model, parts):
    if model.template not in TEMPLATES:
        raise ValueError(f"{model.name}: template {model.template!r} is not one of {', '.join(TEMPLATES)}")
    if parts.n_images < 0:
        raise ValueError(f"n_images is {parts.n_images}; it counts the image frames")
    typed = [field for field in ("system", "user", "assistant") if IMAGE_PAD in getattr(parts, field)]
    if typed:
        raise ValueError(f"{IMAGE_PAD} in {' and '.join(typed)}: it stands for an image (the prompt gets one per frame "
                         f"of the image input); remove {IMAGE_PAD} from the system, user and assistant text")
    if not parts.user.strip() and parts.n_images == 0:
        raise ValueError("user is empty: write the prompt in user, or connect an image")
    if parts.thinking and (not model.thinking or model.template == QWEN3VL):
        raise ValueError(f"{model.name} has no thinking mode: turn thinking off")
    if parts.thinking and parts.assistant.strip():
        raise ValueError("assistant (a prefill) closes the think block, so it cannot be used with thinking on: "
                         "clear assistant or turn thinking off")


def split_output(model, raw, parts):
    """(text, thinking) of `raw`, the decoded new text. Thinking on: the prompt ended inside the think block,
    so the reasoning is everything before the first </think> and the answer everything after it; with no
    </think> (max_new_tokens reached while thinking) the text is empty and thinking holds all of it, with a
    console warning. Thinking off: the text is the prefill (as the prompt holds it) and its continuation.
    Only surrounding whitespace is stripped; after a prefill the continuation keeps its leading space, which
    belongs between the prefill's last word and the next."""
    if parts.thinking:
        reasoning, closed, answer = raw.partition(THINK_END)
        if not closed:
            logging.warning("[BCNodes] Qwen LM: %s stopped while thinking (no %s; max_new_tokens reached?): "
                            "text is empty and thinking holds all of the output", model.name, THINK_END)
            return "", raw.strip()
        return answer.strip(), reasoning.strip()
    prefill = parts.assistant if model.template == QWEN3VL else parts.assistant.strip()
    return prefill + (raw.rstrip() if prefill else raw.strip()), ""

"""Prompter's flow: one text-to-image prompt, one LM generate.

write(text, image, instructions, instruction, start), in this order:
  1. the inputs: `instructions` one of INSTRUCTIONS; `image`, when connected, one frame (an IMAGE batch of
     exactly one); the brief (`text` holding more than whitespace) or the image, at least one of them;
  2. the path: brief when there is a brief (with include / exclude identity the image is then not used), image
     when there is only an image;
  3. the instruction text (instruction_text): `instruction` when it holds more than whitespace (any
     instructions value), else the built-in prompt of the path and identity mode (prompts/<path>-<include or
     exclude>.txt, read at call time, its surrounding whitespace stripped), else "" for none;
  4. the user turn: the image first when it is used, then the instruction text and the brief joined by a blank
     line (an empty one left out with its blank line):

       input          include / exclude identity                none
       text only      instruction + "\\n\\n" + text               text
       image only     the image, then the instruction           the image
       text + image   instruction + "\\n\\n" + text (no image)    the image, then the text

     (with an instruction override and none, the override comes before the text as in the other columns);
  5. the image the model sees: its long side capped at MAX_SIDE (libs/image.thumbnail_frame);
  6. start() -> generate, once: the node prepares the model of its LM Loader (the model file found, the LoRA
     read and checked, the sampling: thinking off, DECODING or the connected LM Config's value, MAX_NEW_TOKENS,
     its seed) and returns generate(frame or None, user) -> the model's answer: no system turn, no prefill;
  7. the prompt: the answer, its surrounding whitespace stripped.
"""

import os

from ...libs.image import thumbnail_frame

# The instructions widget's values. A new value is appended at the end: a saved workflow keeps its value.
INSTRUCTIONS = ("none", "include identity", "exclude identity")
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
# The built-in prompt file of each identity mode, per path: prompts/<path>-<mode>.txt.
MODES = {"include identity": "include", "exclude identity": "exclude"}
# The long side the model is given at most, as Captioner gives it (PIL's thumbnail).
MAX_SIDE = 1024
# A prompt is one paragraph of 150 to 200 words (about 250 to 300 tokens); the budget only stops a runaway
# answer, and core reserves the KV cache for prompt + this many tokens.
MAX_NEW_TOKENS = 600
# The decoding the prompter LoRA was trained and is served with: Qwen3.5's thinking-off sampling (temperature 0.7,
# top_p 0.8, top_k 20, min_p 0) with presence_penalty 0 instead of Qwen's 1.5. The fields of an LM Config's edited
# set, given whatever the model's own defaults are; used when no LM Config is connected (a connected one replaces
# it whole: the model's thinking-off defaults with that node's edited and linked fields).
DECODING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0}


def has_text(value):
    """Whether a text input holds more than whitespace (a socket that is not connected gives None)."""
    return value is not None and bool(value.strip())


def instruction_text(instructions, instruction, path):
    """The instruction text of the `path` ("brief" or "image"): `instruction` as it is when it holds more than
    whitespace, else the built-in prompt of `instructions` for that path, "" for none."""
    if instructions not in INSTRUCTIONS:
        raise ValueError(f"instructions {instructions!r} is not one of {', '.join(INSTRUCTIONS)}")
    if has_text(instruction):
        return instruction
    if instructions == "none":
        return ""
    with open(os.path.join(PROMPTS_DIR, f"{path}-{MODES[instructions]}.txt"), encoding="utf-8") as f:
        return f.read().strip()


def user_turn(text, image, instructions, instruction):
    """(the user text, the image the model is given or None) for the inputs (module docstring, steps 1 - 5).
    Raises ValueError with what to connect for an image that is not one frame, or for no brief and no image."""
    if image is not None and image.shape[0] != 1:
        raise ValueError(f"Prompter takes one image, got a batch of {image.shape[0]} frames: connect a single image "
                         "(pick one frame of a batch with an image-from-batch node)")
    brief = has_text(text)
    if not brief and image is None:
        raise ValueError("Prompter has nothing to work from: type a brief in text, or connect an image")
    instruction = instruction_text(instructions, instruction, "brief" if brief else "image")
    user = "\n\n".join(part for part in (instruction, text if brief else "") if part)
    use_image = image is not None and (not brief or instructions == "none")
    return user, thumbnail_frame(image, MAX_SIDE) if use_image else None


def write(text, image, instructions, instruction, start):
    """The prompt for the inputs (module docstring); `start()`, called once after the input checks, returns
    generate(frame or None, user) -> the model's answer."""
    user, frame = user_turn(text, image, instructions, instruction)
    generate = start()
    return generate(frame, user).strip()

"""Captioner's flow: one training caption per image frame, each frame its own LM generate.

caption(image, style, instruction, start), in this order:
  1. the user text (user_text): `instruction` when it holds more than whitespace (any style), else the
     style's built-in prompt (prompts/<style>.txt, read at call time, its surrounding whitespace stripped),
     else "" for style none: the model gets the frame alone;
  2. the batch split into frames, in order, each a (1, H, W, C) view of `image` (no copy, no dtype change);
     a batch without frames raises with what to connect;
  3. start() -> generate, once: the node prepares the model of its LM Loader (the model file found, the
     LoRA read and checked, the sampling: thinking off, DECODING, MAX_NEW_TOKENS, its seed for every frame)
     and returns generate(frame, user) -> the model's answer, which runs that prepared model: no system
     turn, no prefill, the model kept loaded from one frame to the next;
  4. per frame: the frame the model sees (model_frame), then generate(that frame, the user text) -> the
     caption as the model wrote it (its surrounding whitespace stripped).
Returns (frames, captions): element i of each belongs to frame i of the batch; the frames are the input's
own, not the copies the model saw.
"""

import os

from ...libs.image import pil_to_tensor_hwc, tensor_to_pil_u8

# The style widget's values. A new style is appended at the end: a saved workflow keeps its value.
STYLES = ("none", "tags", "prose")
PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
# The long side the model is given at most: the cap the image-to-caption training and serving code this node
# was built for applies (PIL's thumbnail), so a LoRA trained that way sees images as it was trained on them.
MAX_SIDE = 1024
# A caption is one line of tags or two or three sentences (well under 200 tokens); the budget only stops a
# runaway answer, and core reserves the KV cache for prompt + this many tokens.
MAX_NEW_TOKENS = 600
# The decoding the image-to-caption serving code this node was built for uses: Qwen3.5's thinking-off sampling
# (temperature 0.7, top_p 0.8, top_k 20, min_p 0) with presence_penalty 0 instead of Qwen's 1.5. A presence
# penalty is subtracted from every token already generated, and a caption reuses tokens by design (the comma
# between every pair of tags, the pronoun of the class word), so 1.5 pushes it off its format. The fields of an
# LM Config's edited set; given as fixed values, whatever the model's own defaults are.
DECODING = {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "repetition_penalty": 1.0}


def user_text(style, instruction):
    """The user turn's text: `instruction` as it is when it holds more than whitespace (a socket that is not
    connected gives None), else the built-in prompt of `style`, "" for none (the frame alone)."""
    if style not in STYLES:
        raise ValueError(f"style {style!r} is not one of {', '.join(STYLES)}")
    if instruction is not None and instruction.strip():
        return instruction
    if style == "none":
        return ""
    with open(os.path.join(PROMPTS_DIR, f"{style}.txt"), encoding="utf-8") as f:
        return f.read().strip()


def model_frame(frame):
    """The frame (1, H, W, C) as the model is given it: a frame whose long side is within MAX_SIDE as it is;
    a larger one as PIL's Image.thumbnail((MAX_SIDE, MAX_SIDE)) leaves it (long side MAX_SIDE, aspect kept,
    never enlarged) on the frame as 8-bit RGB, back as a float32 (1, h, w, 3) frame. The LM family's own image
    sizing then runs on it as on any image."""
    if max(frame.shape[1], frame.shape[2]) <= MAX_SIDE:
        return frame
    pil = tensor_to_pil_u8(frame[0, :, :, :3])
    pil.thumbnail((MAX_SIDE, MAX_SIDE))
    return pil_to_tensor_hwc(pil).unsqueeze(0)


def caption(image, style, instruction, start):
    """(frames, captions) of the IMAGE batch `image` (module docstring); `start()`, called once after the input
    checks, returns generate(frame, user) -> the model's answer for one frame."""
    text = user_text(style, instruction)
    if image.ndim != 4:
        raise ValueError(f"image must be an IMAGE batch (frames, height, width, channels), got shape "
                         f"{tuple(image.shape)}")
    if image.shape[0] == 0:
        raise ValueError("image holds no frames: connect an image or a batch with at least one frame (an empty "
                         "caption list would stop the nodes it feeds with an error of their own)")
    frames = [image[i:i + 1] for i in range(image.shape[0])]
    generate = start()
    return frames, [generate(model_frame(frame), text).strip() for frame in frames]

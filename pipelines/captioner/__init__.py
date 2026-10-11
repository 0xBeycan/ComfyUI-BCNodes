"""Captioner: one training caption per image frame, each frame its own LM generate.

    caption    the flow: the batch split into frames, the user text (a built-in style prompt or the
               instruction), the frame the model sees (long side capped at 1024 as PIL's thumbnail does
               it), the model started once, one generate per frame; the fixed decoding and token budget
    prompts/   the built-in style prompts (tags.txt, prose.txt), read at call time

Nothing here imports ComfyUI or pipelines/lm: the node layer (nodes/lm.py) hands in start, which prepares the
model of its LM Loader once and returns the generate callable.
"""

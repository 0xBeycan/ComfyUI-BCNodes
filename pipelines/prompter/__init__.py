"""Prompter: one text-to-image prompt from a short brief, a reference image, or both, with one LM generate.

    write      the flow: the path (brief or image) from what is connected, the user text (a built-in prompt for
               include / exclude identity, the instruction, or none), the image the model sees (one frame, long
               side capped at 1024 as PIL's thumbnail does it, libs/image.thumbnail_frame), the model started
               once and run once; the decoding used without an LM Config, and the token budget
    prompts/   the built-in prompts, one per path and identity mode (brief-include.txt, brief-exclude.txt,
               image-include.txt, image-exclude.txt), read at call time

Nothing here imports ComfyUI or pipelines/lm: the node layer (nodes/lm.py) hands in start, which prepares the
model of its LM Loader once and returns the generate callable.
"""

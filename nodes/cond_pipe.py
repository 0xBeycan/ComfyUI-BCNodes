"""BC_CondPipePack (Cond Pipe Pack) and BC_CondPipeUnpack (Cond Pipe Unpack)

Four conditionings on one link (BC_COND_PIPE): positive and negative, and cn_positive and
cn_negative, the pair with a ControlNet applied. A two-pass workflow samples the first pass with
the ControlNet pair and the second with the plain pair; the pipe takes all four into a subgraph on
one link. The pipe holds references: nothing is copied.

An input left unlinked on Pack is an empty slot. Unpack returns an empty slot as ComfyUI's
ExecutionBlocker with a message: an unlinked output does nothing, and the node a linked one feeds
stops with the message instead of receiving None. Unpack does not raise for it itself: the link
state is not part of its cache key, so a cached Unpack does not run again when one more of its
outputs is linked; the blocker is part of the cached output.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CondPipe:
    """The BC_COND_PIPE value, Cond Pipe Pack's output. Each field is the CONDITIONING linked to Pack's
    input of that name, or None when that input was not linked."""
    positive: object
    negative: object
    cn_positive: object
    cn_negative: object


class CondPipePack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "The positive conditioning without a ControlNet."}),
                "negative": ("CONDITIONING", {"tooltip": "The negative conditioning without a ControlNet."}),
                "cn_positive": ("CONDITIONING", {"tooltip": "The positive conditioning with a ControlNet applied."}),
                "cn_negative": ("CONDITIONING", {"tooltip": "The negative conditioning with a ControlNet applied."}),
            },
        }

    RETURN_TYPES = ("BC_COND_PIPE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_TOOLTIPS = ("The four conditionings on one link, for Cond Pipe Unpack.",)
    FUNCTION = "pack"
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ["BCNodes", "cond pipe", "conditioning pipe", "pipe", "controlnet"]
    DESCRIPTION = ("Puts four conditionings on one link: positive and negative, and cn_positive and cn_negative with a "
                   "ControlNet applied. Cond Pipe Unpack takes them apart again. Every input is optional; an output of "
                   "Unpack whose input was left unlinked here stops the node it feeds with a message.")

    def pack(self, positive=None, negative=None, cn_positive=None, cn_negative=None):
        return (CondPipe(positive=positive, negative=negative, cn_positive=cn_positive, cn_negative=cn_negative),)


class CondPipeUnpack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("BC_COND_PIPE", {"tooltip": "The pipe of a Cond Pipe Pack."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative", "cn_positive", "cn_negative")
    OUTPUT_TOOLTIPS = ("The positive conditioning without a ControlNet.",
                       "The negative conditioning without a ControlNet.",
                       "The positive conditioning with a ControlNet applied.",
                       "The negative conditioning with a ControlNet applied.")
    FUNCTION = "unpack"
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ["BCNodes", "cond pipe", "conditioning pipe", "pipe", "controlnet"]
    DESCRIPTION = ("Takes a Cond Pipe Pack's pipe apart into its four conditionings. An output whose input was left "
                   "unlinked on Pack stops the node it feeds with a message; left unlinked here, it does nothing.")

    def unpack(self, pipe):
        from comfy_execution.graph import ExecutionBlocker

        values = (pipe.positive, pipe.negative, pipe.cn_positive, pipe.cn_negative)
        return tuple(value if value is not None else
                     ExecutionBlocker(f"Cond Pipe Unpack: the pipe holds no {name}. Link {name} on Cond Pipe Pack, "
                                      f"or unlink Cond Pipe Unpack's {name} output.")
                     for name, value in zip(self.RETURN_NAMES, values))


NODE_CLASS_MAPPINGS = {
    "BC_CondPipePack": CondPipePack,
    "BC_CondPipeUnpack": CondPipeUnpack,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_CondPipePack": "Cond Pipe Pack",
    "BC_CondPipeUnpack": "Cond Pipe Unpack",
}

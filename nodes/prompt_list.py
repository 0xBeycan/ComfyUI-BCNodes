"""BC_PromptList (Prompt List), BC_AspectPromptList (Aspect Prompt List)

BC_PromptList: one prompt per line of `multiline_text`, optionally wrapped in
`prepend_text` and `append_text`, windowed by `start_index` / `max_rows`. The
two list outputs (OUTPUT_IS_LIST) drive one execution per line downstream; the
third output is a plain string.

BC_AspectPromptList: the same prompt lines grouped under `[WxH]` header lines;
every prompt carries the size of the header above it, as width / height lists
of the same length as the prompt list.
"""

import re

HELP = "One prompt per line. prompt = prepend_text + line + append_text; body_text = the bare line. start_index and max_rows pick a window of lines."


class PromptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": ("STRING", {"multiline": False, "default": ""}),
                "multiline_text": ("STRING", {"multiline": True, "default": ""}),
                "append_text": ("STRING", {"multiline": False, "default": ""}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "body_text", "show_help")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "make_list"
    CATEGORY = "BCNodes/text"
    SEARCH_ALIASES = ['BCNodes', 'prompt list', 'prompts', 'multiline list']

    def make_list(self, multiline_text="", prepend_text="", append_text="", start_index=0, max_rows=1000):
        lines = (multiline_text or "").split("\n")
        start = max(0, min(int(start_index), len(lines) - 1))
        end = min(start + max(1, int(max_rows)), len(lines))
        body = lines[start:end]
        prompts = [f"{prepend_text or ''}{line}{append_text or ''}" for line in body]
        return (prompts, body, HELP)


ASPECT_HELP = "Prompts grouped under [WxH] header lines, one prompt per line; blank lines are skipped. Each prompt gets the width and height of the header above it."

_SIZE_HEADER = re.compile(r"^\[\s*(\d+)\s*[xX]\s*(\d+)\s*\]$")


class AspectPromptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prepend_text": ("STRING", {"multiline": False, "default": ""}),
                "multiline_text": ("STRING", {"multiline": True, "default": "", "tooltip": ASPECT_HELP}),
                "append_text": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "body_text", "width", "height")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "make_list"
    CATEGORY = "BCNodes/text"
    DESCRIPTION = ASPECT_HELP
    SEARCH_ALIASES = ['BCNodes', 'aspect prompt list', 'resolution prompt list', 'prompts', 'width height list']

    def make_list(self, multiline_text="", prepend_text="", append_text=""):
        body, widths, heights = [], [], []
        size = None
        for number, raw in enumerate((multiline_text or "").split("\n"), start=1):
            line = raw.strip()
            if not line:
                continue
            header = _SIZE_HEADER.match(line)
            if header:
                size = (int(header.group(1)), int(header.group(2)))
                if size[0] < 1 or size[1] < 1:
                    raise ValueError(f"Aspect Prompt List: line {number} '{line}' has a zero size; write it as [WxH] with both sides above 0.")
                continue
            if size is None:
                raise ValueError(f"Aspect Prompt List: line {number} is a prompt before any size header; put a [WxH] line, e.g. [1024x1536], above it.")
            body.append(line)
            widths.append(size[0])
            heights.append(size[1])
        if not body:
            raise ValueError("Aspect Prompt List: no prompt lines; write a [WxH] header line and one prompt per line under it.")
        prompts = [f"{prepend_text or ''}{line}{append_text or ''}" for line in body]
        return (prompts, body, widths, heights)


NODE_CLASS_MAPPINGS = {
    "BC_PromptList": PromptList,
    "BC_AspectPromptList": AspectPromptList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PromptList": "Prompt List",
    "BC_AspectPromptList": "Aspect Prompt List",
}

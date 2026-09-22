"""BC_PromptList (Prompt List)

One prompt per line of `multiline_text`, optionally wrapped in `prepend_text`
and `append_text`, windowed by `start_index` / `max_rows`. The two list
outputs (OUTPUT_IS_LIST) drive one execution per line downstream; the third
output is a plain string.
"""

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


NODE_CLASS_MAPPINGS = {
    "BC_PromptList": PromptList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PromptList": "Prompt List",
}

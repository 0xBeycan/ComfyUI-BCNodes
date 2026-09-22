"""BC_ShowText (Show Text)

Shows the incoming text on the node (web/js/show_text.js) and passes it on.
Works on lists: every element gets its own box. The text is also written
into the workflow copy that goes into image metadata, so a saved image shows
what was displayed.
"""


class ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "show"
    CATEGORY = "BCNodes/text"
    SEARCH_ALIASES = ['BCNodes', 'show text', 'display text', 'preview text']

    def show(self, text=None, unique_id=None, extra_pnginfo=None):
        if text is None:
            text = []
        elif not isinstance(text, list):
            text = [text]
        text = ["" if t is None else str(t) for t in text]

        # INPUT_IS_LIST wraps the hidden inputs in lists as well.
        node_id = unique_id[0] if isinstance(unique_id, list) and unique_id else unique_id
        info = extra_pnginfo[0] if isinstance(extra_pnginfo, list) and extra_pnginfo else extra_pnginfo
        workflow = (info or {}).get("workflow") if isinstance(info, dict) else None
        if node_id is not None and workflow:
            wanted = str(node_id).split(":")[-1]
            graphs = [workflow] + list((workflow.get("definitions") or {}).get("subgraphs") or [])
            for g in graphs:
                for node in g.get("nodes") or []:
                    if str(node.get("id")) == wanted:
                        node["widgets_values"] = list(text)
        return {"ui": {"text": text}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "BC_ShowText": ShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ShowText": "Show Text",
}

"""BC_MathExpression (Math Expression)

Evaluates a small arithmetic language over the optional inputs a, b and c
without eval(): the expression is parsed with `ast` and only a fixed set of
node types and functions is walked; anything else is a ValueError.

    a * 2 + 1            numbers from INT / FLOAT inputs
    a.width / a.height   size of an IMAGE or LATENT input
    Loader.width         a widget of another node, by node title or type
    iif(a > b, a, b)     comparisons yield 1 / 0
"""

from ..libs.math_expression import evaluate, is_volatile
from .common import ANY


def _size_of(value, name, attr):
    """width / height of an IMAGE (B, H, W, C) or LATENT ({"samples": (B, C, H/8, W/8)})."""
    if isinstance(value, dict) and "samples" in value:
        shape = value["samples"].shape
        return (shape[3] if attr == "width" else shape[2]) * 8
    if hasattr(value, "shape") and len(value.shape) >= 3:
        shape = value.shape
        return shape[2] if attr == "width" else shape[1]
    raise ValueError(f"{name}.{attr}: {name} is not an IMAGE or LATENT")


def _widget_value(extra_pnginfo, prompt, node_name, widget_name):
    """Value of another node's widget, addressed as Title.widget or Type.widget."""
    workflow = (extra_pnginfo or {}).get("workflow") or {}
    node_id = None
    for node in workflow.get("nodes", []):
        names = {node.get("type"), node.get("title"), (node.get("properties") or {}).get("Node name for S&R")}
        if node_name in names:
            node_id = node.get("id")
            break
    if node_id is None:
        raise ValueError(f"node not found: {node_name}")
    inputs = ((prompt or {}).get(str(node_id)) or {}).get("inputs") or {}
    if widget_name not in inputs:
        raise ValueError(f"widget not found: {node_name}.{widget_name}")
    value = inputs[widget_name]
    if isinstance(value, list):
        raise ValueError(f"{node_name}.{widget_name} is linked, not a widget; wire it into a, b or c instead")
    return value


class MathExpression:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "expression": ("STRING", {"multiline": True, "dynamicPrompts": False, "default": ""}),
            },
            "optional": {
                "a": (ANY,),
                "b": (ANY,),
                "c": (ANY,),
            },
            "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("INT", "FLOAT")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/logic"
    SEARCH_ALIASES = ['BCNodes', 'math expression', 'math', 'expression', 'calculate']

    @classmethod
    def IS_CHANGED(cls, expression="", **kwargs):
        # The cache key already holds the expression (a widget) and a / b / c
        # (links to their producers), so any constant would do here; the
        # expression is returned for readability. NaN marks the node as
        # changed on every run, which also re-runs everything below it, so it
        # is reserved for expressions whose result can differ with the same
        # inputs.
        return float("nan") if is_volatile(expression) else expression

    def run(self, expression, prompt=None, extra_pnginfo=None, a=None, b=None, c=None):
        result = evaluate(expression, {"a": a, "b": b, "c": c}, _size_of, lambda owner, attr: _widget_value(extra_pnginfo, prompt, owner, attr))
        return {"ui": {"value": [result]}, "result": (int(result), float(result))}


NODE_CLASS_MAPPINGS = {
    "BC_MathExpression": MathExpression,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_MathExpression": "Math Expression",
}

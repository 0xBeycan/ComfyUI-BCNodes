"""BC_AnythingEverywhere (Anything Everywhere) and BC_FastGroupsBypasser
(Fast Groups Bypasser).

Both decide things on the canvas; the Python side only exists so the nodes
are part of the pack and appear in the prompt. Neither has outputs, so the
executor never runs them.

Anything Everywhere: whatever is wired into the node is offered to every
unconnected input of the same type when the prompt is built
(web/js/anything_everywhere.js). One node takes any number of sources: the
canvas adds a slot per link (anything, anything11, anything12, …), so the
optional inputs accept any name. The node properties `title_regex` /
`input_regex` (right click → Properties Panel) narrow the targets by node
title and input name.

Fast Groups Bypasser: one toggle per group in the graph, switching the
group's nodes between active and bypass (web/js/fast_groups_bypasser.js).
"""

from .common import ANY, FlexibleOptionalInputType


class AnythingEverywhere:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY, {"anything": (ANY,)}),
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ['BCNodes', 'anything everywhere', 'use everywhere', 'broadcast']

    def noop(self, **kwargs):
        return ()


class FastGroupsBypasser:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ['BCNodes', 'fast groups bypasser', 'group bypass', 'bypass groups']

    def noop(self, **kwargs):
        return ()


NODE_CLASS_MAPPINGS = {
    "BC_AnythingEverywhere": AnythingEverywhere,
    "BC_FastGroupsBypasser": FastGroupsBypasser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_AnythingEverywhere": "Anything Everywhere",
    "BC_FastGroupsBypasser": "Fast Groups Bypasser",
}

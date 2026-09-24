"""BC_AnySwitch (Any Switch)

Any number of wildcard inputs (any_01, any_02, ... — the slots are added by
web/js/any_switch.js); the first one that is connected and not None comes
out. Nothing connected -> None.
"""

import re

from .common import ANY, FlexibleOptionalInputType, slot_index

_SLOT = re.compile(r"^any_(\d+)$")


class AnySwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(ANY),
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("*",)
    FUNCTION = "switch"
    CATEGORY = "BCNodes/logic"
    SEARCH_ALIASES = ['BCNodes', 'any switch', 'switch', 'fallback']

    def switch(self, **kwargs):
        for name in sorted(kwargs, key=lambda name: slot_index(_SLOT, name)):
            if _SLOT.match(name) and kwargs[name] is not None:
                return (kwargs[name],)
        return (None,)


NODE_CLASS_MAPPINGS = {
    "BC_AnySwitch": AnySwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_AnySwitch": "Any Switch",
}

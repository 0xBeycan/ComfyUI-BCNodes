"""Logic nodes.

    BC_LogicBoolean  (Logic Boolean)
    BC_IsMaskEmpty   (Is Mask Empty)
"""

import torch


class LogicBoolean:
    """A 0-1 float rounded to a boolean, plus the same decision as NUMBER / INT
    and the raw float. NUMBER is a plain numeric socket type: it links to any
    input declared as NUMBER or *."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boolean": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "NUMBER", "INT", "FLOAT")
    RETURN_NAMES = ("BOOLEAN", "NUMBER", "INT", "FLOAT")
    FUNCTION = "return_boolean"
    CATEGORY = "BCNodes/logic"
    SEARCH_ALIASES = ['BCNodes', 'logic boolean', 'bool', 'float to boolean']

    def return_boolean(self, boolean=1.0):
        rounded = int(round(boolean))
        return (bool(rounded), rounded, rounded, boolean)


class IsMaskEmpty:
    """True when there is no mask or every pixel is 0."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",)}}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "is_empty"
    CATEGORY = "BCNodes/mask"
    SEARCH_ALIASES = ['BCNodes', 'is mask empty', 'mask empty', 'empty mask']

    def is_empty(self, mask):
        if mask is None or bool(torch.all(mask == 0)):
            return (True,)
        return (False,)


NODE_CLASS_MAPPINGS = {
    "BC_LogicBoolean": LogicBoolean,
    "BC_IsMaskEmpty": IsMaskEmpty,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_LogicBoolean": "Logic Boolean",
    "BC_IsMaskEmpty": "Is Mask Empty",
}

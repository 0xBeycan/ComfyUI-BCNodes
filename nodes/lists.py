"""List nodes.

    BC_JoinImageLists (Join Image Lists)
"""

import re

from .common import FlexibleOptionalInputType, slot_index

_SLOT = re.compile(r"^In(\d+)$")


class JoinImageLists:
    """Concatenate any number of image lists into one, plus each list's size.

    In1 and In2 are required; web/js/join_image_lists.js adds In3, In4, ...
    as slots get connected and they arrive here through **kwargs (the
    optional mapping answers for any name).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "In1": ("IMAGE",),
                "In2": ("IMAGE",),
            },
            "optional": FlexibleOptionalInputType("IMAGE"),
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("Joined", "Sizes")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "join_lists"
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ['BCNodes', 'join image lists', 'concatenate images', 'image list']

    def join_lists(self, **kwargs):
        sizes = []
        joined = []
        for name in sorted(kwargs, key=lambda name: slot_index(_SLOT, name)):
            images = kwargs[name]
            if images is None:
                continue
            sizes.append(len(images))
            joined.extend(images)
        return (joined, sizes)


NODE_CLASS_MAPPINGS = {
    "BC_JoinImageLists": JoinImageLists,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_JoinImageLists": "Join Image Lists",
}

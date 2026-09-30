"""BC_SelectSwitch (Select Switch)

One wildcard input per named option (the slots and the `selected` list are
made by web/js/select_switch.js); the input of the option named by `selected`
comes out. The option inputs are lazy: only the selected branch is evaluated,
the others never run. The selected option having no input connected is an
error.

`selected` is a COMBO whose values live on the canvas, so ComfyUI's built-in
"value not in list" check is replaced by VALIDATE_INPUTS (taking `selected`
turns that check off for it, taking `input_types` turns the link type check
off so a STRING or COMBO output can drive it).
"""

from .common import ANY, FlexibleOptionalInputType

# The frontend refuses the option names `selected` (the widget) and `self`
# (the bound methods below are called with the inputs as keyword arguments).
_DRIVER_TYPES = ("STRING", "COMBO", "*")


def _require_connected(selected, options):
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"Select Switch: `selected` must be an option name, got {selected!r}. "
                         "Add options with + Add option and pick one in `selected`.")
    if selected not in options:
        raise ValueError(f"Select Switch: the selected option '{selected}' has no input connected. "
                         f"Connect a node to the '{selected}' input, or select a connected option.")


class SelectSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected": ([], {"tooltip": "The option whose input is passed to the output. "
                                             "Only that input's branch runs."}),
            },
            "optional": FlexibleOptionalInputType(ANY, socket_options={"lazy": True}),
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("*",)
    FUNCTION = "select"
    CATEGORY = "BCNodes/logic"
    SEARCH_ALIASES = ['BCNodes', 'select switch', 'switch', 'select', 'choose']
    DESCRIPTION = ("Passes the input of the selected option to the output. Options are added and named on the "
                   "node; only the selected option's branch is executed.")

    @classmethod
    def VALIDATE_INPUTS(cls, selected, input_types):
        driver = input_types.get("selected")
        if driver is not None and not isinstance(driver, list) and driver not in _DRIVER_TYPES:
            return f"`selected` is driven by a {driver} output; connect a STRING or COMBO output instead."
        if selected is None:  # driven by a link: checked when the node runs
            return True
        if not isinstance(selected, str) or not selected:
            return "No option is selected. Add options with + Add option and pick one in `selected`."
        return True

    def check_lazy_status(self, selected, **options):
        _require_connected(selected, options)
        return [selected] if options[selected] is None else []

    def select(self, selected, **options):
        _require_connected(selected, options)
        return (options[selected],)


NODE_CLASS_MAPPINGS = {
    "BC_SelectSwitch": SelectSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_SelectSwitch": "Select Switch",
}

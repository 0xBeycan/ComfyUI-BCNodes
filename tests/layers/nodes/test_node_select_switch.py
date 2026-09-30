"""Select Switch, entered through its node methods and ComfyUI's lookups.

check_lazy_status asks for the selected input only, and only while it is unevaluated; an
unconnected selection is a ValueError naming the option; VALIDATE_INPUTS accepts any selected
name (the list lives on the canvas), refuses an empty one and a non-string driver; every dynamic
option slot is declared lazy through the lookup ComfyUI's execution graph uses.
"""

import pytest


@pytest.fixture
def switch(bcnodes):
    return bcnodes["select_switch"].SelectSwitch


def test_only_the_selected_branch_is_requested(switch):
    s = switch()
    assert s.check_lazy_status("option_b", option_a=None, option_b=None, option_c=None) == ["option_b"]


def test_nothing_requested_once_the_selected_input_is_evaluated(switch):
    s = switch()
    assert s.check_lazy_status("option_b", option_a=None, option_b="b") == []


def test_output_is_the_selected_input(switch):
    assert switch().select("option_b", option_a="a", option_b="b") == ("b",)


def test_selected_input_that_evaluated_to_none_passes_none(switch):
    assert switch().select("option_a", option_a=None, option_b="b") == (None,)


@pytest.mark.parametrize("method", ["check_lazy_status", "select"])
def test_unconnected_selection_is_an_error(switch, method):
    with pytest.raises(ValueError, match=r"selected option 'option_c' has no input connected.*'option_c' input"):
        getattr(switch(), method)("option_c", option_a=None, option_b=None)


@pytest.mark.parametrize("selected", ["", None, 3])
def test_selection_that_is_not_a_name_is_an_error(switch, selected):
    with pytest.raises(ValueError, match="must be an option name"):
        switch().select(selected, option_a="a")


def test_validation_accepts_a_name_not_in_the_declared_list(switch):
    assert switch.INPUT_TYPES()["required"]["selected"][0] == []
    assert switch.VALIDATE_INPUTS("option_a", {}) is True


def test_validation_refuses_an_empty_selection(switch):
    assert "No option is selected" in switch.VALIDATE_INPUTS("", {})


@pytest.mark.parametrize("driver", ["STRING", "COMBO", "*", ["option_a", "option_b"]])
def test_validation_accepts_a_string_or_combo_driver(switch, driver):
    assert switch.VALIDATE_INPUTS(None, {"selected": driver}) is True


def test_validation_refuses_another_driver_type(switch):
    assert "driven by a IMAGE output" in switch.VALIDATE_INPUTS(None, {"selected": "IMAGE"})


def test_validation_takes_selected_and_input_types(switch):
    # ComfyUI skips its combo check for the arguments VALIDATE_INPUTS names, and its link type
    # check when `input_types` is one of them.
    import inspect
    assert inspect.getfullargspec(switch.VALIDATE_INPUTS).args == ["cls", "selected", "input_types"]


@pytest.mark.parametrize("name", ["option_a", "any name"])
def test_dynamic_option_slots_are_lazy(switch, name):
    optional = switch.INPUT_TYPES()["optional"]
    assert name in optional
    socket_type, options = optional[name]
    assert socket_type == "*" and options == {"lazy": True}


def test_flexible_optional_without_options_is_unchanged(bcnodes):
    common = bcnodes["nodes.common"]
    assert common.FlexibleOptionalInputType(common.ANY)["x"] == (common.ANY,)

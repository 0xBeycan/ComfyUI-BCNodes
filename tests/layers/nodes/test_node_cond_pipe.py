"""Cond Pipe Pack / Unpack, entered through their node methods.

Each linked conditioning comes out of Unpack as the same object it went into Pack (nothing is
copied); an input left unlinked on Pack comes out as ComfyUI's ExecutionBlocker, whose message names
the slot and what to do; a pipe with every slot filled holds no blocker.
"""

import sys
import types

import pytest

SLOTS = ("positive", "negative", "cn_positive", "cn_negative")


class ExecutionBlocker:
    """Stands in for comfy_execution.graph.ExecutionBlocker: the message it was made with."""

    def __init__(self, message):
        self.message = message


@pytest.fixture
def cond_pipe(bcnodes, monkeypatch):
    graph = types.ModuleType("comfy_execution.graph")
    graph.ExecutionBlocker = ExecutionBlocker
    monkeypatch.setitem(sys.modules, "comfy_execution", types.ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph)
    return bcnodes["cond_pipe"]


def conditionings():
    """Four distinct CONDITIONING-shaped values, keyed by slot."""
    import torch

    return {name: [[torch.full((1, 2, 4), float(i)), {"slot": name}]] for i, name in enumerate(SLOTS)}


def round_trip(module, **linked):
    (pipe,) = module.CondPipePack().pack(**linked)
    return module.CondPipeUnpack().unpack(pipe)


def test_unpack_order_is_return_names(cond_pipe):
    assert cond_pipe.CondPipeUnpack.RETURN_NAMES == SLOTS
    assert tuple(cond_pipe.CondPipePack.INPUT_TYPES()["optional"]) == SLOTS


def test_every_slot_comes_out_as_the_object_that_went_in(cond_pipe):
    conds = conditionings()
    out = round_trip(cond_pipe, **conds)
    assert len(out) == 4
    for name, value in zip(SLOTS, out):
        assert value is conds[name]


@pytest.mark.parametrize("empty", SLOTS)
def test_an_unlinked_input_comes_out_as_a_blocker_naming_it(cond_pipe, empty):
    conds = conditionings()
    del conds[empty]
    out = round_trip(cond_pipe, **conds)
    for name, value in zip(SLOTS, out):
        if name == empty:
            assert isinstance(value, ExecutionBlocker)
            assert value.message == (f"Cond Pipe Unpack: the pipe holds no {empty}. Link {empty} on Cond Pipe Pack, "
                                     f"or unlink Cond Pipe Unpack's {empty} output.")
        else:
            assert value is conds[name]


def test_pack_with_no_inputs_gives_four_blockers(cond_pipe):
    out = round_trip(cond_pipe)
    assert [type(value) for value in out] == [ExecutionBlocker] * 4
    assert [value.message.split(".")[0] for value in out] == [f"Cond Pipe Unpack: the pipe holds no {name}" for name in SLOTS]

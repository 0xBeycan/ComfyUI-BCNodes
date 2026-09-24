"""Golden: the small nodes, entered through their node methods.

Logic Boolean, Is Mask Empty, Prompt List, Any Switch and Join Image Lists (slot order and
non-slot names, C32), Seed (the -1 draw and its write-back), Show Text (write-back), Power Lora
Loader (row order, strengths, name matching, calls into comfy), Image Comparer (calls into
ComfyUI's PreviewImage), Anything Everywhere and Fast Groups Bypasser (no-ops).

Seams: Seed's module RNG `_rng` (WHERE) is replaced by random.Random(0) per case; folder_paths'
LoRA listing is scripted; comfy.sd, comfy.utils.load_torch_file and ComfyUI's `nodes` module
(PreviewImage) are recording stubs installed per test.
"""

import copy
import random
import sys
import types

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'LogicBoolean/0': [('False', 'bool'), ('0', 'int'), ('0', 'int'), ('0', 'int')],
    'LogicBoolean/0.0': [('False', 'bool'), ('0', 'int'), ('0', 'int'), ('0.0', 'float')],
    'LogicBoolean/0.49': [('False', 'bool'), ('0', 'int'), ('0', 'int'), ('0.49', 'float')],
    'LogicBoolean/0.5': [('False', 'bool'), ('0', 'int'), ('0', 'int'), ('0.5', 'float')],
    'LogicBoolean/0.51': [('True', 'bool'), ('1', 'int'), ('1', 'int'), ('0.51', 'float')],
    'LogicBoolean/0.7': [('True', 'bool'), ('1', 'int'), ('1', 'int'), ('0.7', 'float')],
    'LogicBoolean/1': [('True', 'bool'), ('1', 'int'), ('1', 'int'), ('1', 'int')],
    'LogicBoolean/1.0': [('True', 'bool'), ('1', 'int'), ('1', 'int'), ('1.0', 'float')],
    'LogicBoolean/default': [('True', 'bool'), ('1', 'int'), ('1', 'int'), ('1.0', 'float')],
    'IsMaskEmpty/None': [('True', 'bool')],
    'IsMaskEmpty/empty': [('True', 'bool')],
    'IsMaskEmpty/zeros': [('True', 'bool')],
    'IsMaskEmpty/-0.0': [('True', 'bool')],
    'IsMaskEmpty/1e-12': [('False', 'bool')],
    'IsMaskEmpty/nan': [('False', 'bool')],
    'IsMaskEmpty/-1': [('False', 'bool')],
    'PromptList/empty': '1f88537ea224297b8578a61057a73deb',
    'PromptList/None': '1f88537ea224297b8578a61057a73deb',
    'PromptList/three': '71ea76ab627d61f5ab0a4b7fb350952b',
    'PromptList/trailing_newline': 'b01ee853cd91fb362354f70d0f3cfe51',
    'PromptList/crlf': '5885203eedf678d0162c98b35fcc69ec',
    'PromptList/blank_lines': 'ad951e84f2cabb6e536ac06c75620c59',
    'PromptList/one': '972fbe3b539da31ee03d741300cc6924',
    'PromptList/defaults': '31fd546e52651665b864cee827ec167c',
    'AnySwitch/nothing': '(None,)',
    'AnySwitch/all_none': '(None,)',
    'AnySwitch/order_by_number': '(1,)',
    'AnySwitch/number_not_text': '(2,)',
    'AnySwitch/zero_is_a_value': '(0,)',
    'AnySwitch/non_slot_skipped': '(4,)',
    'AnySwitch/only_non_slot': '(None,)',
    'AnySwitch/same_index_keeps_order': "('first',)",
    'JoinImageLists/nothing': '([], [])',
    'JoinImageLists/none_inputs': '([], [])',
    'JoinImageLists/empty_lists': '([], [0, 0])',
    'JoinImageLists/order_by_number': "(['a1', 'a2', 'b1'], [2, 1])",
    'JoinImageLists/number_not_text': "(['a1', 'b1', 'b2', 'j1'], [1, 2, 1])",
    'JoinImageLists/gap_and_none': "(['b1', 'c1'], [1, 1])",
    'JoinImageLists/non_slot_names_last': "(['a1', 'b1', 'z1', 'x1', 'x2'], [1, 1, 1, 2])",
    'JoinImageLists/same_index_keeps_order': "(['second', 'first'], [1, 1])",
    'JoinImageLists/tensors': '8e88c682a28fac269b65aa65fcf844ce',
    'Seed/None': '9879bca9a2ad4a0183a97ac2aabb7a0f',
    'Seed/zero': '9879bca9a2ad4a0183a97ac2aabb7a0f',
    'Seed/five': 'c92849ae6cfb76a3207fca262c71b160',
    'Seed/string': '2981811db481e0dbbe9a1252100a78c4',
    'Seed/max': '772b349585b27892ef78872a95c9b2b5',
    'Seed/minus_one_bare': '774697753de6c58841cb63586858db2d',
    'Seed/minus_one_float': '774697753de6c58841cb63586858db2d',
    'Seed/minus_one_uid_none': '9ec19f1d9800e2f171c2b9ced3d6643e',
    'Seed/minus_one_root_and_subgraph': '25db8df200687e45ed39f00883659177',
    'Seed/minus_one_subgraph_only': 'c95c055a3549beeecd18b04537c74dae',
    'Seed/minus_one_named_differs': 'e902960af056e3fe6e455e99622139bb',
    'Seed/minus_one_prompt_not_dict': 'f059627151e8c423ba8e528d7b5ce710',
    'Seed/minus_one_no_workflow_node': '5dcb038c55df9c30611b224eee6b3c00',
    'Seed/draws': [6939962107496397, 729300138103893, 8753695300970082, 6450042198180356],
    'Seed.IS_CHANGED/-1': 'nan',
    'Seed.IS_CHANGED/0': '0',
    'Seed.IS_CHANGED/5': '5',
    "Seed.IS_CHANGED/'5'": "'5'",
    'Seed.IS_CHANGED/default': '0',
    'ShowText/None': '49643ec34b564a92041c529ee3a76873',
    'ShowText/empty_list': '49643ec34b564a92041c529ee3a76873',
    'ShowText/scalar': 'ba23e0639de698177124333e0d545b0d',
    'ShowText/list_with_none_and_number': '08a47dd3ea1e6d9c33ef9dc2999c0b0a',
    'ShowText/write_back_every_match': '28cc1fd7146c0d9cb94d993075828f42',
    'ShowText/subgraph_uid': 'b0b2cd5bca862cfabfc186a91c25aaa1',
    'ShowText/not_wrapped': '4fba4f8b5d5529c835bdf9fc5dd8dfb3',
    'ShowText/uid_none': '2fad9626ceb836ab1b5cc5b00c8f43dd',
    'ShowText/no_workflow': 'eea3cb2ecd354f58d3c003ca087300d7',
    'ShowText/info_not_dict': '2fd1da905add4a0906ff85f644ed7a97',
    'PowerLoraLoader/no_model': '16479e450f796f59c8293bcfce33cf11',
    'PowerLoraLoader/model_no_rows': '1eca98578bd0df07c3807d8ea744be8b',
    'PowerLoraLoader/rows': 'ff90f2046213c85a052d68a1003023d7',
    'ImageComparer/nothing': '9677ad52a033891bea258be06f0e82c2',
    'ImageComparer/empty_batches': '9677ad52a033891bea258be06f0e82c2',
    'ImageComparer/a_only': '82ae40fd0b1007f3e72776b3da383534',
    'ImageComparer/b_only': 'fd0680282a6c94d7012622ab38f61dfb',
    'ImageComparer/both': '3724fa7a89642fcfbca004a5e4bf1db5',
    'ImageComparer/empty_a_and_b': 'f9c1bd0f0ede0f8df90f59b51fb7d160',
}

WHERE = Where({
    "seed_rng": "nodes.seed:_rng",
})


def _typed(values):
    return [(repr(v), type(v).__name__) for v in values]


# --- Logic Boolean / Is Mask Empty ---------------------------------------------

@pytest.mark.parametrize("value", [0, 0.0, 0.49, 0.5, 0.51, 0.7, 1, 1.0])
def test_logic_boolean(value, bcnodes):
    check(GOLDEN, f"LogicBoolean/{value!r}", _typed(bcnodes["logic"].LogicBoolean().return_boolean(value)))


def test_logic_boolean_default(bcnodes):
    check(GOLDEN, "LogicBoolean/default", _typed(bcnodes["logic"].LogicBoolean().return_boolean()))


MASKS = {
    "None": lambda: None,
    "empty": lambda: torch.zeros((0, 8, 8)),
    "zeros": lambda: torch.zeros((1, 8, 8)),
    "-0.0": lambda: torch.full((1, 8, 8), -0.0),
    "1e-12": lambda: torch.zeros((1, 8, 8)).index_fill_(2, torch.tensor([3]), 1e-12),
    "nan": lambda: torch.zeros((1, 8, 8)).index_fill_(2, torch.tensor([3]), float("nan")),
    "-1": lambda: torch.zeros((2, 8, 8)).index_fill_(0, torch.tensor([1]), -1.0),
}


@pytest.mark.parametrize("name", list(MASKS))
def test_is_mask_empty(name, bcnodes):
    check_env(ENV, "torch")
    check(GOLDEN, f"IsMaskEmpty/{name}", _typed(bcnodes["logic"].IsMaskEmpty().is_empty(MASKS[name]())))


# --- Prompt List ----------------------------------------------------------------

TEXTS = {
    "empty": "",
    "None": None,
    "three": "a\nb\nc",
    "trailing_newline": "a\nb\nc\n",
    "crlf": "a\r\nb\r\nc",
    "blank_lines": "\n\nx\n\n",
    "one": "<trigger>, person",
}


@pytest.mark.parametrize("name", list(TEXTS))
def test_prompt_list(name, bcnodes):
    pl = bcnodes["prompt_list"].PromptList()
    grid = {}
    for start in (0, 1, 2, 99):
        for rows in (1, 2, 1000):
            for pre, post in (("", ""), ("pre ", " post"), (None, None), ("term_a, ", None)):
                grid[repr((start, rows, pre, post))] = pl.make_list(
                    TEXTS[name], prepend_text=pre, append_text=post, start_index=start, max_rows=rows)
    check(GOLDEN, f"PromptList/{name}", digest(grid))


def test_prompt_list_defaults(bcnodes):
    check(GOLDEN, "PromptList/defaults", digest(bcnodes["prompt_list"].PromptList().make_list("a\nb")))


# --- Any Switch / Join Image Lists (C32) ------------------------------------------

SWITCH = {
    "nothing": {},
    "all_none": {"any_01": None, "any_02": None},
    "order_by_number": {"any_02": 2, "any_01": 1},
    "number_not_text": {"any_10": 10, "any_2": 2},
    "zero_is_a_value": {"any_01": None, "any_02": 0, "any_03": 5},
    "non_slot_skipped": {"foo": 5, "any_03": None, "any_04": 4},
    "only_non_slot": {"foo": 5, "Any_01": 6, "any_": 7},
    "same_index_keeps_order": {"any_1": "first", "any_01": "second"},
}


@pytest.mark.parametrize("name", list(SWITCH))
def test_any_switch(name, bcnodes):
    check(GOLDEN, f"AnySwitch/{name}", repr(bcnodes["any_switch"].AnySwitch().switch(**SWITCH[name])))


JOIN = {
    "nothing": {},
    "none_inputs": {"In1": None, "In2": None},
    "empty_lists": {"In1": [], "In2": []},
    "order_by_number": {"In2": ["b1"], "In1": ["a1", "a2"]},
    "number_not_text": {"In10": ["j1"], "In2": ["b1", "b2"], "In1": ["a1"]},
    "gap_and_none": {"In3": ["c1"], "In1": None, "In2": ["b1"]},
    "non_slot_names_last": {"zeta": ["z1"], "In2": ["b1"], "alpha": ["x1", "x2"], "In1": ["a1"]},
    "same_index_keeps_order": {"In01": ["second"], "In1": ["first"]},
}


@pytest.mark.parametrize("name", list(JOIN))
def test_join_image_lists(name, bcnodes):
    check(GOLDEN, f"JoinImageLists/{name}", repr(bcnodes["lists"].JoinImageLists().join_lists(**JOIN[name])))


def test_join_image_lists_tensors(bcnodes):
    g = torch.Generator().manual_seed(0)
    a = [torch.rand((1, 4, 6, 3), generator=g), torch.rand((1, 4, 6, 3), generator=g)]
    b = [torch.rand((2, 5, 3, 3), generator=g)]
    joined, sizes = bcnodes["lists"].JoinImageLists().join_lists(In2=b, In1=a)
    assert joined[0] is a[0] and joined[1] is a[1] and joined[2] is b[0]
    check(GOLDEN, "JoinImageLists/tensors", digest((joined, sizes)))


# --- Seed -------------------------------------------------------------------------

def _seed_workflow():
    return {
        "nodes": [
            {"id": 3, "type": "BC_Seed", "widgets_values": [-1, "fixed", -1, 5]},
            {"id": 4, "type": "BC_Seed", "widgets_values": [-1]},
        ],
        "definitions": {"subgraphs": [
            {"nodes": [{"id": 3, "type": "BC_Seed", "widgets_values": [-1], "widgets_values_named": {"seed": -1}}]},
        ]},
    }


def _subgraph_only_workflow():
    return {
        "nodes": [{"id": 4, "type": "BC_Seed", "widgets_values": [-1]}],
        "definitions": {"subgraphs": [
            {"nodes": [{"id": 9, "widgets_values": [-1]}]},
            {"nodes": [{"id": 3, "type": "BC_Seed", "widgets_values": [-1, -1],
                        "widgets_values_named": {"seed": -1, "other": -1}}]},
        ]},
    }


SEED = {
    "None": dict(seed=None),
    "zero": dict(seed=0),
    "five": dict(seed=5),
    "string": dict(seed="7"),
    "max": dict(seed=0xFFFFFFFFFFFFFFFF),
    "minus_one_bare": dict(seed=-1),
    "minus_one_float": dict(seed=-1.0),
    "minus_one_uid_none": dict(seed=-1, prompt={"3": {"inputs": {"seed": -1}}}, extra_pnginfo={"workflow": _seed_workflow()}),
    "minus_one_root_and_subgraph": dict(seed=-1, prompt={"7:3": {"inputs": {"seed": -1, "other": 1}}, "3": {"inputs": {"seed": -1}}},
                                        extra_pnginfo={"workflow": _seed_workflow()}, unique_id="7:3"),
    "minus_one_subgraph_only": dict(seed=-1, prompt={"7:3": {"inputs": {"seed": -1}}},
                                    extra_pnginfo={"workflow": _subgraph_only_workflow()}, unique_id="7:3"),
    "minus_one_named_differs": dict(seed=-1, prompt={"3": {"inputs": {}}},
                                    extra_pnginfo={"workflow": {"nodes": [{"id": 3, "widgets_values": [0, -1], "widgets_values_named": {"seed": 0}}]}},
                                    unique_id=3),
    "minus_one_prompt_not_dict": dict(seed=-1, prompt={"3": {"inputs": [-1]}}, extra_pnginfo={"workflow": None}, unique_id="3"),
    "minus_one_no_workflow_node": dict(seed=-1, prompt={}, extra_pnginfo={"workflow": _seed_workflow()}, unique_id="8"),
}


@pytest.mark.parametrize("name", list(SEED))
def test_seed(name, bcnodes, monkeypatch, capsys):
    WHERE.patch(monkeypatch, "seed_rng", random.Random(0))
    kwargs = copy.deepcopy(SEED[name])
    result = bcnodes["seed"].Seed().main(**kwargs)
    check(GOLDEN, f"Seed/{name}", digest({"result": _typed(result), "prompt": kwargs.get("prompt"),
                                          "extra_pnginfo": kwargs.get("extra_pnginfo"), "stdout": capsys.readouterr().out}))


def test_seed_draws_follow_the_module_rng(bcnodes, monkeypatch):
    WHERE.patch(monkeypatch, "seed_rng", random.Random(0))
    node = bcnodes["seed"].Seed()
    check(GOLDEN, "Seed/draws", [node.main(-1)[0] for _ in range(4)])


@pytest.mark.parametrize("seed", [-1, 0, 5, "5"])
def test_seed_is_changed(seed, bcnodes):
    check(GOLDEN, f"Seed.IS_CHANGED/{seed!r}", repr(bcnodes["seed"].Seed.IS_CHANGED(seed=seed)))


def test_seed_is_changed_default(bcnodes):
    check(GOLDEN, "Seed.IS_CHANGED/default", repr(bcnodes["seed"].Seed.IS_CHANGED()))


# --- Show Text --------------------------------------------------------------------

def _show_workflow():
    return {
        "nodes": [{"id": 5, "type": "BC_ShowText", "widgets_values": ["old"]}, {"id": 6, "widgets_values": ["keep"]}],
        "definitions": {"subgraphs": [
            {"nodes": [{"id": 5, "widgets_values": []}]},
            {"nodes": [{"id": "5", "widgets_values": ["old", "older"]}, {"id": 50}, {"id": 5, "widgets_values": ["dup"]}]},
        ]},
    }


SHOW = {
    "None": dict(text=None),
    "empty_list": dict(text=[]),
    "scalar": dict(text="solo"),
    "list_with_none_and_number": dict(text=["a", None, 3]),
    "write_back_every_match": dict(text=["a", None, 3], unique_id=["5"], extra_pnginfo=[{"workflow": _show_workflow()}]),
    "subgraph_uid": dict(text=["x"], unique_id=["9:5"], extra_pnginfo=[{"workflow": _show_workflow()}]),
    "not_wrapped": dict(text="y", unique_id="5", extra_pnginfo={"workflow": _show_workflow()}),
    "uid_none": dict(text=["z"], unique_id=[None], extra_pnginfo=[{"workflow": _show_workflow()}]),
    "no_workflow": dict(text=["z"], unique_id=["5"], extra_pnginfo=[{"other": 1}]),
    "info_not_dict": dict(text=["z"], unique_id=["5"], extra_pnginfo=["workflow"]),
}


@pytest.mark.parametrize("name", list(SHOW))
def test_show_text(name, bcnodes):
    kwargs = copy.deepcopy(SHOW[name])
    result = bcnodes["show_text"].ShowText().show(**kwargs)
    check(GOLDEN, f"ShowText/{name}", digest({"result": result, "extra_pnginfo": kwargs.get("extra_pnginfo")}))


# --- Power Lora Loader -----------------------------------------------------------

LORAS = {
    "style_a.safetensors": "/path/to/loras/style_a.safetensors",
    "sub/detail_b.safetensors": "/path/to/loras/sub/detail_b.safetensors",
    "c.pt": "/path/to/loras/c.pt",
}


@pytest.fixture
def comfy_lora(monkeypatch):
    """Scripted LoRA listing and recording comfy.utils.load_torch_file / comfy.sd.load_lora_for_models;
    returns the call log."""
    calls = []
    fp = sys.modules["folder_paths"]

    def get_full_path(folder, name):
        calls.append(("get_full_path", folder, name))
        return LORAS.get(name) if folder == "loras" else None

    def get_filename_list(folder):
        calls.append(("get_filename_list", folder))
        return list(LORAS) if folder == "loras" else []

    def load_torch_file(path, safe_load=False):
        calls.append(("load_torch_file", path, safe_load))
        return {"lora": path}

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
        calls.append(("load_lora_for_models", model, clip, lora, _typed([strength_model, strength_clip])))
        return f"{model}+{lora['lora'].rsplit('/', 1)[-1]}@{strength_model}", "clip_out"

    monkeypatch.setattr(fp, "get_full_path", get_full_path, raising=True)
    monkeypatch.setattr(fp, "get_filename_list", get_filename_list, raising=True)
    sd = types.ModuleType("comfy.sd")
    sd.load_lora_for_models = load_lora_for_models
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    monkeypatch.setattr(sys.modules["comfy"], "sd", sd, raising=False)
    monkeypatch.setattr(sys.modules["comfy.utils"], "load_torch_file", load_torch_file, raising=False)
    return calls


LORA_ROWS = {
    "no_model": dict(lora_1={"on": True, "lora": "style_a.safetensors", "strength": 1.0}),
    "model_no_rows": dict(model="MODEL"),
    "rows": dict(
        model="MODEL",
        lora_3={"on": True, "lora": "style_a", "strength": 1},
        lora_1={"on": True, "lora": "style_a.safetensors", "strength": "0.5"},
        lora_2={"on": False, "lora": "c.pt", "strength": 1.0},
        lora_4={"on": True, "lora": "c.pt", "strength": 0},
        lora_5={"on": True, "lora": "None", "strength": 1.0},
        lora_6={"on": True, "lora": "detail_b.safetensors", "strength": -0.25},
        lora_7={"on": True, "lora": "c"},
        lora_8={"on": True, "lora": "missing.safetensors", "strength": 1.0},
        lora_9=None,
        lora_10={"on": True, "lora": "", "strength": 1.0},
        lora_11={"on": True, "lora": "b.safetensors", "strength": 1.0},  # a suffix, but not after a "/"
        lora_12={"on": True, "lora": "sub/detail_b", "strength": 0.75},
        foo={"on": True, "lora": "c.pt", "strength": 1.0},
        lora_x={"on": True, "lora": "c.pt", "strength": 1.0},
    ),
}


@pytest.mark.parametrize("name", list(LORA_ROWS))
def test_power_lora_loader(name, bcnodes, comfy_lora, capsys):
    result = bcnodes["power_lora_loader"].PowerLoraLoader().load_loras(**copy.deepcopy(LORA_ROWS[name]))
    check(GOLDEN, f"PowerLoraLoader/{name}", digest({"result": result, "calls": comfy_lora, "stdout": capsys.readouterr().out}))


# --- Image Comparer ----------------------------------------------------------------

@pytest.fixture
def preview_image(monkeypatch):
    """ComfyUI's `nodes` module with a recording PreviewImage; returns the call log."""
    calls = []

    class PreviewImage:
        def __init__(self):
            calls.append(("PreviewImage",))

        def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
            calls.append(("save_images", digest(images), filename_prefix, prompt, extra_pnginfo))
            return {"ui": {"images": [{"filename": f"{filename_prefix}{i:05}.png", "subfolder": "", "type": "temp"}
                                      for i in range(len(images))]}}

    comfy_nodes = types.ModuleType("nodes")
    comfy_nodes.PreviewImage = PreviewImage
    monkeypatch.setitem(sys.modules, "nodes", comfy_nodes)
    return calls


def _images(n, seed):
    return torch.rand((n, 4, 6, 3), generator=torch.Generator().manual_seed(seed))


COMPARE = {
    "nothing": lambda: dict(),
    "empty_batches": lambda: dict(image_a=torch.zeros((0, 8, 8, 3)), image_b=torch.zeros((0, 8, 8, 3))),
    "a_only": lambda: dict(image_a=_images(2, 1)),
    "b_only": lambda: dict(image_b=_images(1, 2), prompt={"1": {}}, extra_pnginfo={"workflow": {}}),
    "both": lambda: dict(image_a=_images(2, 1), image_b=_images(3, 2), prompt={"1": {"inputs": {}}}, extra_pnginfo={"workflow": {"nodes": []}}),
    "empty_a_and_b": lambda: dict(image_a=torch.zeros((0, 8, 8, 3)), image_b=_images(1, 3)),
}


@pytest.mark.parametrize("name", list(COMPARE))
def test_image_comparer(name, bcnodes, preview_image):
    check_env(ENV, "torch")
    result = bcnodes["image_comparer"].ImageComparer().compare(**COMPARE[name]())
    check(GOLDEN, f"ImageComparer/{name}", digest({"result": result, "calls": preview_image}))


# --- Anything Everywhere / Fast Groups Bypasser ---------------------------------------

def test_everywhere_noops(bcnodes):
    everywhere = bcnodes["everywhere"]
    assert everywhere.AnythingEverywhere().noop() == ()
    assert everywhere.AnythingEverywhere().noop(anything=1, anything11=None, other="x") == ()
    assert everywhere.FastGroupsBypasser().noop() == ()
    assert everywhere.FastGroupsBypasser().noop(x=1) == ()

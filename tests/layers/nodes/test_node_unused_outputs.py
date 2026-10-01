"""Unused heavy outputs return empty (nodes/common.py), node by node and the handler alone:

- which outputs are heavy: a whole-batch IMAGE / MASK the node makes. Pass-throughs (Social Media
  Export's images, Join Image Lists) and one-image outputs (badges, cards, sheets) are not, and a
  node with one output (not an output node) runs only when that output is linked, so it declares
  none;
- the on_prompt handler's stamp, a stale stamp overwritten, the feature off (stamp removed, a
  console line) when another pack's handler runs after it, still on when the handler after it is
  a stamping one (ComfyUI-BCVideoNodes'), the pack named from its module;
- the startup hook: the stamping handlers moved to the end of the list (both groups in their own
  order), idempotent, at once when the server has started already;
- the node side: no stamp -> every output full; a link the stamp missed -> full and a warning;
- per node: each unlinked heavy output is a new 0-frame tensor of the full output's dtype and
  trailing shape, every other output equals the full run's, and where the output is a step of its
  own the step does not run (Image Resize's frames, Image Scale By Aspect Ratio's fits, BiRefNet's
  composite and mask image, SeedVR2 Resize's two resizes, Skin Texture's texture).

The full run is the same node call without a stamp. The runtime behaviour through ComfyUI's
executor is in tests/test_runtime.py.
"""

import asyncio
import logging
import sys
import types

import pytest
import torch

NODE = "7"
HEAVY = {
    "BC_ImageResize": ("IMAGE", "mask"),
    "BC_ImageScaleByAspectRatio": ("image", "mask"),
    "BC_BiRefNetRemoveBackground": ("IMAGE", "MASK", "MASK_IMAGE"),
    "BC_SeedVR2Resize": ("image", "reference"),
    "BC_SkinTexture": ("image", "skin_mask"),
}


@pytest.fixture
def common(bcnodes):
    return bcnodes["nodes.common"]


@pytest.fixture
def mappings(bcnodes):
    found = {}
    for name, module in bcnodes.items():
        found.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    return found


def graph(common, cls, linked=(), stamp=True, links=None):
    """The PROMPT graph node NODE of `cls` gets: its stamp lists `linked` (no stamp with `stamp`
    False), and a consumer links each output name of `links` (default: `linked`)."""
    inputs = {common.STAMP: ",".join(sorted(linked))} if stamp else {}
    prompt = {NODE: {"class_type": cls.__name__, "inputs": inputs}}
    for i, name in enumerate(linked if links is None else links):
        prompt[f"c{i}"] = {"class_type": "Consumer", "inputs": {"x": [NODE, cls.RETURN_NAMES.index(name)]}}
    return prompt


def hidden(common, cls, linked=(), **kwargs):
    """The node function's hidden inputs (LINK_INPUTS) for `graph(...)`."""
    values = {"PROMPT": graph(common, cls, linked, **kwargs), "UNIQUE_ID": NODE}
    return {name: values[kind] for name, kind in common.LINK_INPUTS.items()}


def emptied(out, full):
    """`out` is the drop of `full`: a 0-frame tensor of its dtype and trailing shape, owning its
    (empty) storage."""
    return (isinstance(out, torch.Tensor) and out.shape == (0, *full.shape[1:]) and out.dtype == full.dtype
            and out._base is None)


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    return a == b


def check_outputs(cls, out, full, unlinked):
    """Every heavy output of `unlinked` dropped, every other output what the full run returned."""
    assert len(out) == len(full) == len(cls.RETURN_NAMES)
    for name, a, b in zip(cls.RETURN_NAMES, out, full):
        assert emptied(a, b) if name in unlinked else same(a, b), name


def spy(monkeypatch, module, name):
    """Records the arguments of each call of `module.name` and calls through."""
    real = getattr(module, name)
    seen = []

    def recorded(*args, **kwargs):
        seen.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, recorded)
    return seen


# --- which outputs are heavy ---------------------------------------------------------------------

def test_the_heavy_outputs_and_their_hidden_inputs(common, mappings):
    assert set(HEAVY) <= set(mappings)
    for key, cls in mappings.items():
        heavy = getattr(cls, "HEAVY_OUTPUTS", ())
        assert heavy == HEAVY.get(key, ()), key
        if not heavy:
            continue
        assert all(cls.RETURN_TYPES[cls.RETURN_NAMES.index(name)] in ("IMAGE", "MASK") for name in heavy), key
        # a node that runs only when one of its outputs is linked needs a second output to drop one
        assert len(cls.RETURN_NAMES) > 1 or getattr(cls, "OUTPUT_NODE", False), key
        assert cls.INPUT_TYPES()["hidden"] == common.LINK_INPUTS, key


def test_the_pass_through_outputs_are_the_inputs(bcnodes):
    images = [torch.rand(1, 8, 8, 3), torch.rand(2, 8, 8, 3)]
    joined, _ = bcnodes["lists"].JoinImageLists().join_lists(In1=[images[0]], In2=[images[1]])
    assert joined[0] is images[0] and joined[1] is images[1]


# --- the on_prompt handler -------------------------------------------------------------------------

class FakeServer:
    """What the handler reads of ComfyUI's PromptServer: the handler list, and its aiohttp app."""

    def __init__(self):
        from aiohttp import web

        self.on_prompt_handlers = []
        self.app = web.Application()

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def start(self):
        """What aiohttp's AppRunner.setup does to the app when ComfyUI starts its server."""
        self.app.on_startup.freeze()
        asyncio.run(self.app.startup())

    def trigger(self, prompt, client_id="client"):
        json_data = {"prompt": prompt, "client_id": client_id}
        for handler in self.on_prompt_handlers:
            json_data = handler(json_data)
        return json_data["prompt"]


def a_prompt():
    """Image Resize "1" with its IMAGE linked (mask not), BiRefNet "2" with MASK and MASK_IMAGE
    linked, MaskGrow "3" (no heavy output)."""
    return {
        "1": {"class_type": "BC_ImageResize", "inputs": {"width": 32}},
        "2": {"class_type": "BC_BiRefNetRemoveBackground", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "BC_MaskGrow", "inputs": {"mask": ["2", 1]}},
        "4": {"class_type": "Other", "inputs": {"a": ["2", 2], "b": ["1", 1], "c": ["3", 0]}},
    }


@pytest.fixture
def server(common, mappings):
    server = FakeServer()
    server.add_on_prompt_handler(common.LinkStamp(mappings, server))
    return server


def stamps(common, prompt):
    return {node_id: node["inputs"].get(common.STAMP) for node_id, node in prompt.items()}


def test_the_handler_stamps_the_linked_heavy_outputs(common, server, caplog):
    caplog.set_level(logging.WARNING)
    assert stamps(common, server.trigger(a_prompt())) == {"1": "IMAGE", "2": "MASK,MASK_IMAGE", "3": None, "4": None}
    assert caplog.text == ""


def test_the_handler_overwrites_a_stale_stamp(common, server):
    prompt = a_prompt()
    prompt["1"]["inputs"][common.STAMP] = "IMAGE,mask"
    assert stamps(common, server.trigger(prompt))["1"] == "IMAGE"


def test_the_handler_leaves_other_payloads_alone(server):
    handler = server.on_prompt_handlers[0]
    for json_data in ({}, {"prompt": None}, {"prompt": {"1": "not a node"}}, None):
        assert handler(json_data) == json_data


def another_pack_handler(json_data):
    return json_data


def test_another_packs_handler_after_ours_turns_it_off_and_says_so(common, server, caplog):
    server.add_on_prompt_handler(another_pack_handler)
    prompt = a_prompt()
    prompt["1"]["inputs"][common.STAMP] = ""  # a stamp left from an earlier run goes too
    caplog.set_level(logging.WARNING)
    assert stamps(common, server.trigger(prompt, client_id="abc")) == {"1": None, "2": None, "3": None, "4": None}
    assert [r.getMessage() for r in caplog.records] == [
        "BCNodes: RAM saving of unused outputs is off for this run: test_node_unused_outputs changes the prompt after it."]


class BCVideoNodesStamp:
    """ComfyUI-BCVideoNodes' handler, as marked there."""
    bc_link_stamp = True

    def __call__(self, json_data):
        return json_data


def test_a_handler_before_ours_and_a_stamping_handler_after_it_keep_it_on(common, server, caplog):
    server.on_prompt_handlers.insert(0, another_pack_handler)
    server.add_on_prompt_handler(BCVideoNodesStamp())
    caplog.set_level(logging.WARNING)
    assert stamps(common, server.trigger(a_prompt()))["1"] == "IMAGE"
    assert caplog.text == ""


def test_the_pack_is_named_by_its_folder(server, monkeypatch, caplog):
    # ComfyUI imports a custom node folder as a module named by its path, dots written as _x_
    name = "/comfy/custom_nodes/some_x_pack"
    module = types.ModuleType(name)
    module.__file__ = "/comfy/custom_nodes/some.pack/__init__.py"
    monkeypatch.setitem(sys.modules, name, module)
    handler = types.FunctionType(another_pack_handler.__code__, {}, "handler")
    handler.__module__ = name + ".hooks"
    server.add_on_prompt_handler(handler)
    caplog.set_level(logging.WARNING)
    server.trigger(a_prompt())
    assert "some.pack changes the prompt" in caplog.records[0].getMessage()
    del module.__file__
    server.trigger(a_prompt())
    assert "some.pack changes the prompt" in caplog.records[1].getMessage()


def test_registration_reads_the_server_from_sys_modules(common, mappings, monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "server", raising=False)
    caplog.set_level(logging.INFO)
    assert common.register_link_stamp(mappings) is None
    assert "no ComfyUI server" in caplog.text
    fake = FakeServer()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake)))
    handler = common.register_link_stamp(mappings)
    assert fake.on_prompt_handlers == [handler] and handler.bc_link_stamp is True


def other_pack_handler(json_data):
    return json_data


def test_at_startup_the_stamps_move_after_every_other_handler(common, mappings, monkeypatch, caplog):
    # packs load in folder order: one before ours, ours, BCVideoNodes', two after; then the server starts
    fake = FakeServer()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake)))
    fake.add_on_prompt_handler(another_pack_handler)
    ours = common.register_link_stamp(mappings)
    theirs = BCVideoNodesStamp()
    fake.add_on_prompt_handler(theirs)
    fake.add_on_prompt_handler(other_pack_handler)
    fake.add_on_prompt_handler(another_pack_handler)
    caplog.set_level(logging.INFO)
    fake.start()
    assert fake.on_prompt_handlers == [another_pack_handler, other_pack_handler, another_pack_handler, ours, theirs]
    assert "the link stamps now run after the on_prompt handlers of test_node_unused_outputs" in caplog.text
    caplog.clear()
    assert stamps(common, fake.trigger(a_prompt()))["1"] == "IMAGE"  # on, no warning
    assert "RAM saving" not in caplog.text
    common.stamps_last(fake)  # BCVideoNodes' own startup hook does the same: nothing changes
    assert fake.on_prompt_handlers == [another_pack_handler, other_pack_handler, another_pack_handler, ours, theirs]
    assert caplog.text == ""


def test_a_stamp_registered_after_startup_moves_at_once(common, mappings, monkeypatch):
    fake = FakeServer()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake)))
    theirs = BCVideoNodesStamp()
    fake.add_on_prompt_handler(theirs)
    fake.add_on_prompt_handler(another_pack_handler)
    fake.start()
    ours = common.register_link_stamp(mappings)
    assert fake.on_prompt_handlers == [another_pack_handler, theirs, ours]


# --- the node side ---------------------------------------------------------------------------------

def test_no_stamp_wants_everything_and_drops_nothing(common, mappings):
    cls = mappings["BC_ImageResize"]
    assert common.heavy_wanted(cls, graph(common, cls, stamp=False), NODE) is None
    assert common.heavy_wanted(cls, None, None) is None
    outputs = (torch.rand(3, 4, 5, 3), 5, 4, torch.rand(3, 4, 5))
    out = common.drop_unlinked_heavy(cls, outputs, None, None)
    assert all(a is b for a, b in zip(out, outputs))


def test_the_stamp_says_what_is_wanted_and_a_missed_link_is_kept(common, mappings, caplog):
    cls = mappings["BC_ImageResize"]
    assert common.heavy_wanted(cls, graph(common, cls, ["mask"]), NODE) == {"mask"}
    caplog.set_level(logging.WARNING)
    assert common.heavy_wanted(cls, graph(common, cls, links=["IMAGE"]), NODE) == {"IMAGE"}
    assert "the prompt links IMAGE, which its link stamp does not list" in caplog.text


def test_a_dropped_output_is_a_new_empty_tensor_of_its_kind(common, mappings):
    cls = mappings["BC_BiRefNetRemoveBackground"]
    full = (torch.rand(3, 4, 5, 4), torch.rand(3, 4, 5).half(), torch.rand(3, 4, 5, 3))
    out = common.drop_unwanted(cls, full, {"MASK"})
    check_outputs(cls, out, full, {"IMAGE", "MASK_IMAGE"})
    assert out[1] is full[1]


# --- each node ---------------------------------------------------------------------------------------

RESIZES = [
    dict(width=24, height=20, upscale_method="bilinear", keep_proportion="stretch", pad_color="0, 0, 0", crop_position="center",
         divisible_by=2),
    dict(width=40, height=40, upscale_method="lanczos", keep_proportion="pad", pad_color="0, 0, 0", crop_position="top",
         divisible_by=8),
    dict(width=32, height=24, upscale_method="area", keep_proportion="pillarbox_blur", pad_color="0, 0, 0",
         crop_position="center", divisible_by=0),
    dict(width=32, height=24, upscale_method="bilinear", keep_proportion="stretch", pad_color="0, 0, 0",
         crop_position="center", divisible_by=0),  # the input's size: passed through
]


@pytest.mark.parametrize("widgets", RESIZES)
@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("linked", [(), ("IMAGE",), ("mask",)])
def test_image_resize_computes_only_linked_outputs(bcnodes, common, monkeypatch, widgets, with_mask, linked):
    cls, rs = bcnodes["image_scale"].ImageResize, bcnodes["libs.resize"]
    image = torch.rand(3, 24, 32, 3)
    mask = torch.rand(3, 12, 16) if with_mask else None  # another size: scaled to the image first
    full = cls().resize(image, mask=mask, **widgets)
    image_frames = spy(monkeypatch, rs, "_resample_image")
    mask_frames = spy(monkeypatch, rs, "_resample_mask")
    out = cls().resize(image, mask=mask, **widgets, **hidden(common, cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    if "IMAGE" not in linked:
        assert image_frames == []
    if "mask" not in linked:
        assert mask_frames == []


@pytest.mark.parametrize("with_mask", [False, True])
@pytest.mark.parametrize("linked", [(), ("image",), ("mask",)])
def test_image_scale_by_aspect_ratio_computes_only_linked_outputs(bcnodes, common, monkeypatch, with_mask, linked):
    module = bcnodes["image_scale"]
    cls = module.ImageScaleByAspectRatio
    widgets = dict(aspect_ratio="16:9", proportional_width=1, proportional_height=1, fit="letterbox", method="lanczos",
                   round_to_multiple="8", scale_to_side="longest", scale_to_length=40, background_color="#000000")
    image, mask = torch.rand(2, 30, 20, 3), (torch.rand(2, 30, 20) if with_mask else None)
    full = cls().scale(**widgets, image=image, mask=mask)
    fits = spy(monkeypatch, module, "fit_image")
    out = cls().scale(**widgets, image=image, mask=mask, **hidden(common, cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    # one fit per frame of each linked output
    assert len(fits) == 2 * (("image" in linked) + ("mask" in linked and with_mask))


@pytest.fixture
def matted(bcnodes, monkeypatch):
    """BiRefNet's matte replaced by a stand-in: the red channel."""
    matting = bcnodes["pipelines.matting"]
    monkeypatch.setattr(matting, "matte", lambda model, rgb: rgb[..., 0].float().cpu())
    return matting


@pytest.mark.parametrize("background", ["Alpha", "Color"])
@pytest.mark.parametrize("linked", [(), ("IMAGE",), ("MASK",), ("MASK_IMAGE",)])
def test_birefnet_builds_only_linked_outputs(bcnodes, common, matted, monkeypatch, background, linked):
    cls = bcnodes["birefnet"].BiRefNetRemoveBackground
    image = torch.rand(3, 16, 12, 3)
    widgets = dict(mask_blur=2, refine_foreground=True, background=background, background_color="#336699")
    full = cls().remove_background(image, "BiRefNet-general", **widgets)
    refined = spy(monkeypatch, bcnodes["libs.mask"], "refine_foreground")
    out = cls().remove_background(image, "BiRefNet-general", **widgets, **hidden(common, cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    # the composite's edge refinement covers no frame when IMAGE is not linked
    assert [len(args[0]) for args in refined] == [3 if "IMAGE" in linked else 0]


@pytest.mark.parametrize("linked", [(), ("image",), ("reference",)])
def test_seedvr2_resize_runs_only_linked_resizes(bcnodes, common, monkeypatch, linked):
    cls, flow = bcnodes["seedvr2"].SeedVR2Resize, bcnodes["pipelines.seedvr2.resize"]
    image = torch.rand(6, 18, 26, 3)
    widgets = dict(upscale_factor=1.5, downscale_factor=1.0, max_resolution=0, emulate_bf16=False)
    full = cls().resize(image, **widgets)
    resized = spy(monkeypatch, flow, "side_resize")
    out = cls().resize(image, **widgets, **hidden(common, cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    # the six frames resized for the linked output only; an unlinked one's resize runs on no frame
    assert sum(len(args[0]) for args in resized) == 6 * len(linked)
    assert sum(1 for args in resized if len(args[0]) == 0) == 2 - len(linked)


@pytest.mark.parametrize("linked", [(), ("image",), ("skin_mask",)])
def test_skin_texture_applies_the_texture_only_when_linked(bcnodes, common, monkeypatch, linked):
    cls, flow = bcnodes["skin_texture"].SkinTexture, bcnodes["pipelines.skin_texture"]
    image = torch.rand(2, 24, 16, 3)
    mask = torch.zeros(2, 24, 16)
    mask[:, 4:20, 4:12] = 1.0
    widgets = dict(sam3_model="unused", texture=0.5, detail=0.6, pore_scale=1.0, feather=2, seed=1, threshold=0.5, mask=mask)
    full = cls().run(image, **widgets)
    textured = spy(monkeypatch, flow, "apply_texture")
    out = cls().run(image, **widgets, **hidden(common, cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    assert len(textured) == ("image" in linked)

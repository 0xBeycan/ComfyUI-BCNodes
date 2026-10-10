"""Shared bits for node definitions, and the unused-heavy-outputs helper. Imports nothing beyond the
stdlib at module level."""

import functools
import logging
import os
import sys


class AnyType(str):
    """Wildcard socket type. ComfyUI compares socket types with `!=`, so a
    string that is never "not equal" links to, and validates against, any
    type. The frontend treats the plain "*" as its wildcard."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class FlexibleOptionalInputType(dict):
    """An `optional` INPUT_TYPES mapping that claims to hold every key.

    ComfyUI looks an input up with `name in optional` and `optional[name]`;
    answering both for any name lets a node accept inputs that only exist on
    the canvas (slots added by a web extension) under any name, typed as
    `socket_type`, with `socket_options` as its option dict when given (e.g.
    {"lazy": True}; ComfyUI reads "lazy" per input name through the same
    lookup). Keys given in `known` are returned as declared."""

    def __init__(self, socket_type, known=None, socket_options=None):
        super().__init__()
        self.socket_type = socket_type
        self.socket_options = socket_options
        self.known = known or {}
        for k, v in self.known.items():
            self[k] = v

    def __getitem__(self, key):
        if key in self.known:
            return self.known[key]
        if self.socket_options is not None:
            return (self.socket_type, self.socket_options)
        return (self.socket_type,)

    def __contains__(self, key):
        return True


def slot_index(pattern, name):
    m = pattern.match(name)
    return int(m.group(1)) if m else float("inf")


DEVICES = ["cpu", "gpu"]


def compute_device(choice):
    """A `device` widget (DEVICES) as a torch device: "gpu" is ComfyUI's torch device."""
    import torch

    if choice == "gpu":
        import comfy.model_management

        return comfy.model_management.get_torch_device()
    return torch.device("cpu")


def lm_folders(folder):
    """(model folders, text_encoders folders) of an LM family whose folder_paths key, and folder under
    models/, is `folder` (its LMFamily.folder), in folder_paths order: the folders an LM run (pipelines/lm)
    searches. Registers the key as models/<folder> first (folder_paths keeps a folder once); a key with no
    extensions (new, or made by extra_model_paths.yaml) gets .safetensors."""
    import folder_paths

    folder_paths.add_model_folder_path(folder, os.path.join(folder_paths.models_dir, folder))
    extensions = folder_paths.folder_names_and_paths[folder][1]
    if not extensions:
        extensions.add(".safetensors")
    try:
        text_encoders = folder_paths.get_folder_paths("text_encoders")
    except KeyError:
        text_encoders = []
    return tuple(folder_paths.get_folder_paths(folder)), tuple(text_encoders)


# --- unused heavy outputs return empty ---------------------------------------------------------
#
# A whole-clip IMAGE or MASK that no node consumes is not kept in ComfyUI's output cache until the
# prompt ends. A node lists such outputs in HEAVY_OUTPUTS (names from its RETURN_NAMES), takes the
# hidden LINK_INPUTS, reads `heavy_wanted` and returns through `drop_unwanted`: an unlinked heavy
# output leaves as a 0-frame tensor of the same dtype and trailing shape. Where the output is a step
# of its own that the node's other outputs do not depend on, the node does not compute it (`wants`).
#
# The link state must be part of the node's cache key, or connecting the output later would hit the
# cache entry holding the empty tensor. ComfyUI builds that key from the node's own inputs and its
# ancestors only, and IS_CHANGED gets an empty PROMPT. So an on_prompt handler (`LinkStamp`, added by
# `register_link_stamp` from the root __init__) writes the linked heavy outputs into each heavy
# node's inputs as STAMP before the prompt is validated: every input key is part of the cache key,
# and an input the node does not declare is never passed to it.
#
# - No stamp (the handler did not run: a direct executor call, a node made by expansion): every
#   output full.
# - At run time the node also reads the final PROMPT: a link to a heavy output that the stamp does
#   not list returns that output full, with a warning.
# - An on_prompt handler of another pack that runs after ours could add a link the stamp never saw,
#   and a cached empty output would then reach it. So the stamping handlers run last: once every
#   custom node has loaded (the server's aiohttp on_startup), `stamps_last` moves every handler
#   marked `bc_link_stamp` (this pack's and ComfyUI-BCVideoNodes', which does the same) to the end
#   of the list. A handler appended after that still turns the stamp off for its prompts (every
#   output full), with a console line naming its pack. Handlers marked `bc_link_stamp` only write
#   their own stamps and are not counted.

STAMP = "bc_linked_heavy"
# not `prompt`: a hidden input overwrites a widget of the same name
LINK_INPUTS = {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}


def _is_link(value):
    """comfy_execution.graph_utils.is_link, restated so this module imports nothing from ComfyUI."""
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], (int, float))


def _heavy_indices(cls):
    """{output index: name} of the HEAVY_OUTPUTS of `cls`."""
    return {cls.RETURN_NAMES.index(name): name for name in getattr(cls, "HEAVY_OUTPUTS", ())}


def _links(prompt, heavy):
    """{node id: names linked} for `heavy` {node id: {output index: name}}: the heavy outputs some
    node of `prompt` links, whether or not that node runs."""
    linked = {node_id: set() for node_id in heavy}
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        for value in (inputs.values() if isinstance(inputs, dict) else ()):
            if _is_link(value) and value[0] in heavy:
                name = heavy[value[0]].get(int(value[1]))
                if name is not None:
                    linked[value[0]].add(name)
    return linked


def _pack_of(handler):
    """The custom node folder a handler comes from (its module's top-level file), else its module."""
    target = handler
    while isinstance(target, functools.partial):
        target = target.func
    module = getattr(target, "__module__", None) or type(target).__module__
    top = module.split(".")[0]
    file = getattr(sys.modules.get(top), "__file__", None)
    if file:
        path = os.path.dirname(file) if os.path.basename(file) == "__init__.py" else os.path.splitext(file)[0]
        return os.path.basename(path)
    # ComfyUI names a custom node module by its path, dots written as _x_
    return os.path.basename(top).replace("_x_", ".")


class LinkStamp:
    """The on_prompt handler: writes STAMP, the sorted comma-joined linked heavy outputs, into the
    inputs of every node of `classes` that declares HEAVY_OUTPUTS, or removes it from all of them
    when another pack's handler runs after this one. `server` is ComfyUI's PromptServer instance."""

    bc_link_stamp = True

    def __init__(self, classes, server):
        self.classes = classes
        self.server = server

    def __call__(self, json_data):
        prompt = json_data.get("prompt") if isinstance(json_data, dict) else None
        if not isinstance(prompt, dict):
            return json_data
        heavy = {}
        for node_id, node in prompt.items():
            cls = self.classes.get(node.get("class_type")) if isinstance(node, dict) else None
            if cls is not None and getattr(cls, "HEAVY_OUTPUTS", ()) and isinstance(node.get("inputs", {}), dict):
                heavy[node_id] = _heavy_indices(cls)
        if not heavy:
            return json_data
        after = self.packs_after()
        stamps = None if after else {node_id: ",".join(sorted(names)) for node_id, names in _links(prompt, heavy).items()}
        # everything is worked out above; only now is the prompt written
        for node_id in heavy:
            inputs = prompt[node_id].setdefault("inputs", {})
            if stamps is None:
                inputs.pop(STAMP, None)
            else:
                inputs[STAMP] = stamps[node_id]
        if after:
            logging.warning("BCNodes: RAM saving of unused outputs is off for this run: %s %s the prompt after it.",
                            ", ".join(after), "changes" if len(after) == 1 else "change")
        return json_data

    def packs_after(self):
        """The packs whose on_prompt handlers run after this one, the stamping handlers left out."""
        handlers = list(getattr(self.server, "on_prompt_handlers", ()))
        index = next((i for i, handler in enumerate(handlers) if handler is self), None)
        if index is None:
            return []
        return sorted({_pack_of(handler) for handler in handlers[index + 1:] if not getattr(handler, "bc_link_stamp", False)})


def stamps_last(server):
    """Moves every on_prompt handler marked `bc_link_stamp` to the end of the server's list, in
    place, both groups in their own order. Idempotent: each pack that stamps runs it once."""
    handlers = server.on_prompt_handlers
    stamps = [handler for handler in handlers if getattr(handler, "bc_link_stamp", False)]
    others = [handler for handler in handlers if not getattr(handler, "bc_link_stamp", False)]
    if handlers == others + stamps:
        return
    after = handlers[handlers.index(stamps[0]):]
    moved = sorted({_pack_of(handler) for handler in after if not getattr(handler, "bc_link_stamp", False)})
    handlers[:] = others + stamps
    logging.info("BCNodes: the link stamps now run after the on_prompt handlers of %s", ", ".join(moved))


def register_link_stamp(classes):
    """Adds a LinkStamp for `classes` to ComfyUI's on_prompt handlers and returns it; None outside
    ComfyUI. Also hooks `stamps_last` to the server's startup, which aiohttp runs once, after every
    custom node has loaded and before the server takes a request (at once when it has started
    already). The server is read from sys.modules, never imported: importing `server` would load
    aiohttp with the package."""
    server = getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None)
    if server is None:
        logging.info("BCNodes: no ComfyUI server: unused heavy outputs are returned in full")
        return None
    handler = LinkStamp(classes, server)
    server.add_on_prompt_handler(handler)

    async def on_startup(app):
        stamps_last(server)

    if server.app.on_startup.frozen:
        stamps_last(server)
    else:
        server.app.on_startup.append(on_startup)
    return handler


def heavy_wanted(cls, prompt, unique_id, wanted=None):
    """The HEAVY_OUTPUTS of `cls` that some node links, read from the stamp of node `unique_id` in
    the final `prompt`, plus any link the stamp missed (with a warning). None without a stamp:
    every output is wanted. `wanted`, when given, is the answer as it is."""
    if wanted is not None:
        return wanted
    node = prompt.get(unique_id) if isinstance(prompt, dict) else None
    inputs = node.get("inputs") if isinstance(node, dict) else None
    stamp = inputs.get(STAMP) if isinstance(inputs, dict) else None
    if not isinstance(stamp, str):
        return None
    wanted = {name for name in stamp.split(",") if name}
    missed = _links(prompt, {unique_id: _heavy_indices(cls)})[unique_id] - wanted
    if missed:
        logging.warning("BCNodes: %s (node %s): the prompt links %s, which its link stamp does not list; returned in full",
                        cls.__name__, unique_id, ", ".join(sorted(missed)))
        wanted |= missed
    unlinked = [name for name in cls.HEAVY_OUTPUTS if name not in wanted]
    if unlinked:
        logging.info("BCNodes: %s (node %s): %s not linked, returned empty", cls.__name__, unique_id, ", ".join(unlinked))
    return wanted


def wants(wanted, name):
    """Whether heavy output `name` is to be computed (`wanted` from heavy_wanted)."""
    return wanted is None or name in wanted


def drop_unwanted(cls, outputs, wanted):
    """`outputs` as a tuple, each heavy output not in `wanted` a new 0-frame tensor of its dtype,
    device and trailing shape (IMAGE [0, H, W, C], MASK [0, H, W]). A new tensor, never a view, so
    it keeps no storage alive. `wanted` None: unchanged."""
    import torch

    outputs = tuple(outputs)
    if wanted is None:
        return outputs
    outputs = list(outputs)
    for index, name in _heavy_indices(cls).items():
        if name not in wanted and isinstance(outputs[index], torch.Tensor):
            outputs[index] = outputs[index].new_empty((0, *outputs[index].shape[1:]))
    return tuple(outputs)


def drop_unlinked_heavy(cls, outputs, prompt, unique_id):
    """drop_unwanted with what the stamp of node `unique_id` says: for a node that computes every
    output and drops the unlinked ones at return."""
    return drop_unwanted(cls, outputs, heavy_wanted(cls, prompt, unique_id))

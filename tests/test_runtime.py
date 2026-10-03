"""Backend nodes through ComfyUI's real prompt validation and executor, headless.

Needs a ComfyUI checkout whose requirements are installed in the current
Python. It is found through $COMFYUI_DIR or ../ComfyUI next to this pack.
No server is started and no browser is needed; only core CPU nodes are used.

    COMFYUI_DIR=/path/to/ComfyUI python tests/test_runtime.py
"""

import asyncio
import copy
import json
import logging
import math
import os
import sys
import tempfile
import uuid

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.environ.get("COMFYUI_DIR") or os.path.join(os.path.dirname(PACK), "ComfyUI")

if not os.path.isfile(os.path.join(COMFY, "execution.py")):
    print(f"SKIP: ComfyUI not found at {COMFY} (set COMFYUI_DIR)")
    sys.exit(0)

sys.argv = ["comfy", "--cpu", "--disable-metadata"]
os.chdir(COMFY)
sys.path.insert(0, COMFY)

import execution  # noqa: E402
import nodes  # noqa: E402


class Server:
    client_id = None
    last_node_id = None
    sockets_metadata = {}

    def __init__(self):
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))

    def queue_updated(self):
        pass


results = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    results["pass" if cond else "fail"] += 1
    print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail and not cond else ""))


def N(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs}


async def setup():
    import server
    from app.assets.manager import default_asset_manager

    server.PromptServer(asyncio.get_event_loop(), default_asset_manager())
    # Output nodes write files, and the Process Monitor (on by default) its settings and run logs;
    # keep them out of the checkout's output/, temp/ and user/.
    import folder_paths
    scratch = tempfile.mkdtemp(prefix="bcnodes_runtime_")
    folder_paths.set_output_directory(os.path.join(scratch, "output"))
    folder_paths.set_temp_directory(os.path.join(scratch, "temp"))
    folder_paths.set_user_directory(os.path.join(scratch, "user"))
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    assert await nodes.load_custom_node(PACK), "pack failed to load"
    print("loaded:", ", ".join(sorted(k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith("BC_"))))


def executed_nodes(server):
    """Node ids the executor announced as "executing" (it does so when extra_data has a client_id)."""
    return sorted(data["node"] for event, data in server.events if event == "executing" and data.get("node") is not None)


async def run(prompt, label, extra_data=None, server=None):
    """-> (outputs by node id or None, validation tuple). Pass `server` to read its events after."""
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        return None, valid
    ex = execution.PromptExecutor(server or Server(), cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    await ex.execute_async(prompt, prompt_id, extra_data or {}, valid[2])
    if not ex.success:
        errors = [m for m in ex.status_messages if m[0] == "execution_error"]
        print(f"  [{label}] execution failed: {json.dumps(errors, default=str)[:600]}")
        return None, valid
    return ex.history_result["outputs"], valid


async def run_twice(prompt, label, extra_data=None):
    """The same prompt queued twice on one executor, a fresh copy each time as
    the server sends it (IS_CHANGED is memoized on the node dict).
    -> (outputs of run 1 or None, node ids executed on run 2 or None)"""
    server = Server()
    # With a client id the executor announces every node it runs as "executing".
    extra_data = {**(extra_data or {}), "client_id": "test"}
    ex = execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    outputs = None
    for i in range(2):
        prompt_id = str(uuid.uuid4())
        valid = await execution.validate_prompt(prompt_id, prompt, None)
        if not valid[0]:
            print(f"  [{label}] validation failed: {valid[1]}")
            return None, None
        server.events.clear()
        await ex.execute_async(copy.deepcopy(prompt), prompt_id, copy.deepcopy(extra_data), valid[2])
        if not ex.success:
            errors = [m for m in ex.status_messages if m[0] == "execution_error"]
            print(f"  [{label}] run {i + 1} failed: {json.dumps(errors, default=str)[:600]}")
            return None, None
        executed = executed_nodes(server)
        if i == 0:
            outputs = ex.history_result["outputs"]
            if not executed:
                print(f"  [{label}] run 1 executed nothing; the executor is not reporting")
                return None, None
    return outputs, executed


# --- unused heavy outputs return empty (nodes/common.py) ---------------------------------------------
# The pack's on_prompt handler is on the server (the root __init__ registered it when the pack
# loaded); a prompt goes the way POST /prompt runs it: the handlers, validate_prompt, execute. The
# node is Image Resize: IMAGE heavy, width linked to an output node.

UNUSED_A = {"1": N("EmptyImage", width=64, height=48, batch_size=4, color=0),
            "2": N("BC_ImageResize", image=["1", 0], width=32, height=24, upscale_method="bilinear", keep_proportion="stretch",
                   pad_color="0, 0, 0", crop_position="center", divisible_by=2),
            "3": N("BC_MathExpression", expression="a", a=["2", 1])}
# + IMAGE's frame count shown
UNUSED_AB = {**UNUSED_A, "4": N("GetImageSize", image=["2", 0]), "5": N("PreviewAny", source=["4", 2])}
UNUSED_FULL, UNUSED_EMPTY = (4, 24, 32, 3), (0, 24, 32, 3)


def another_pack(json_data):
    """An on_prompt handler of another pack."""
    return json_data


class BCVideoNodesStamp:
    """ComfyUI-BCVideoNodes' stamping handler, marked as it is there."""
    bc_link_stamp = True

    def __call__(self, json_data):
        return json_data


class Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


async def unused_submit(ex, prompt, via_server=True, edit_after=None):
    """-> (the stamp, runs of node 2, the frame count node 5 shows or None, IMAGE's shape in the
    cache or "evicted")."""
    import server

    json_data = {"prompt": copy.deepcopy(prompt), "client_id": "test"}
    if via_server:
        json_data = server.PromptServer.instance.trigger_on_prompt(json_data)
    if edit_after:
        edit_after(json_data["prompt"])
    run = json_data["prompt"]
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, run, None)
    assert valid[0], valid[1]
    ex.server.events.clear()
    await ex.execute_async(run, prompt_id, {"client_id": "test"}, valid[2])
    assert ex.success, [m for m in ex.status_messages if m[0] == "execution_error"]
    entry = ex.caches.outputs.get_local("2")
    shown = ex.history_result["outputs"].get("5", {}).get("text")
    return (run["2"]["inputs"].get(pack_common().STAMP), executed_nodes(ex.server).count("2"),
            int(shown[0]) if shown else None, "evicted" if entry is None else tuple(entry.outputs[0][0].shape))


def pack_common():
    """The pack's nodes/common.py, as ComfyUI loaded it."""
    import importlib

    return importlib.import_module(nodes.NODE_CLASS_MAPPINGS["BC_ImageResize"].__module__.rsplit(".", 1)[0] + ".common")


def unused_executor(kind):
    args = {"lru": 10 if kind == "LRU" else 0, "ram": 0, "ram_inactive": 0}
    if kind == "RAM_PRESSURE":
        import comfy.model_management as mm

        args["ram_inactive"] = min(128.0, mm.total_ram / 1024.0)
    return execution.PromptExecutor(Server(), cache_type=getattr(execution.CacheType, kind), cache_args=args)


async def unused_outputs():
    import server

    instance = server.PromptServer.instance
    check("unused outputs: the root __init__ registered the link stamp",
          [type(h).__name__ for h in instance.on_prompt_handlers] == ["LinkStamp"], f"{instance.on_prompt_handlers}")
    for kind in ("CLASSIC", "LRU", "RAM_PRESSURE"):
        ex = unused_executor(kind)
        p1 = await unused_submit(ex, UNUSED_A)
        p2 = await unused_submit(ex, UNUSED_AB)
        p3 = await unused_submit(ex, UNUSED_AB)
        p4 = await unused_submit(ex, UNUSED_A)
        p5 = await unused_submit(ex, UNUSED_A)
        check(f"unused outputs {kind} P1: IMAGE unlinked -> runs, empty in the cache", p1 == ("", 1, None, UNUSED_EMPTY), f"{p1}")
        check(f"unused outputs {kind} P2: IMAGE linked later -> runs again, the consumer gets 4 frames",
              p2 == ("IMAGE", 1, 4, UNUSED_FULL), f"{p2}")
        check(f"unused outputs {kind} P3: the same again -> a cache hit", p3 == ("IMAGE", 0, 4, UNUSED_FULL), f"{p3}")
        check(f"unused outputs {kind} P4: unlinked again -> empty (run again, or the P1 entry in LRU)",
              p4[3] == UNUSED_EMPTY and p4[1] == (0 if kind == "LRU" else 1), f"{p4}")
        check(f"unused outputs {kind} P5: the same again -> nothing runs", p5[1] == 0 and p5[3] == UNUSED_EMPTY, f"{p5}")

    out = await unused_submit(unused_executor("CLASSIC"), UNUSED_A, via_server=False)
    check("unused outputs: no stamp -> IMAGE full", out == (None, 1, None, UNUSED_FULL), f"{out}")
    stale = copy.deepcopy(UNUSED_A)
    stale["2"]["inputs"][pack_common().STAMP] = "IMAGE"
    out = await unused_submit(unused_executor("CLASSIC"), stale)
    check("unused outputs: a stale stamp is overwritten", out == ("", 1, None, UNUSED_EMPTY), f"{out}")

    lines = Lines()
    logging.getLogger().addHandler(lines)
    try:
        def link_after(prompt):
            prompt.update({"4": N("GetImageSize", image=["2", 0]), "5": N("PreviewAny", source=["4", 2])})

        out = await unused_submit(unused_executor("CLASSIC"), UNUSED_A, edit_after=link_after)
        check("unused outputs: a link the stamp missed -> full, with a warning",
              out == ("", 1, 4, UNUSED_FULL) and any("the prompt links IMAGE, which its link stamp does not list" in line
                                                      for line in lines.lines), f"{out}")

        # A pack loaded after ours adds its handler, then the server starts the way ComfyUI's
        # main.py starts it (aiohttp's AppRunner.setup runs the app's on_startup, no socket).
        from aiohttp import web

        lines.lines.clear()
        ours = instance.on_prompt_handlers[0]
        instance.add_on_prompt_handler(another_pack)
        runner = web.AppRunner(instance.app)
        await runner.setup()
        try:
            check("unused outputs: at startup the link stamp moves after a handler registered after it",
                  instance.on_prompt_handlers == [another_pack, ours], f"{instance.on_prompt_handlers}")
            out = await unused_submit(unused_executor("CLASSIC"), UNUSED_A)
            check("unused outputs: ... and the saving stays on, no warning",
                  out == ("", 1, None, UNUSED_EMPTY) and not any("RAM saving" in line for line in lines.lines), f"{out}")

            # a handler appended after startup still runs after ours: the fallback
            lines.lines.clear()
            instance.add_on_prompt_handler(another_pack)
            out = await unused_submit(unused_executor("CLASSIC"), UNUSED_A)
            instance.on_prompt_handlers.pop()
            message = "RAM saving of unused outputs is off for this run: test_runtime changes the prompt after it."
            check("unused outputs: a handler appended after startup -> off for its prompts, IMAGE full",
                  out == (None, 1, None, UNUSED_FULL), f"{out}")
            check("unused outputs: ... with a console line", any(message in line for line in lines.lines), f"{lines.lines}")
        finally:
            instance.on_prompt_handlers.remove(another_pack)
            await runner.cleanup()

        stamp = BCVideoNodesStamp()
        instance.add_on_prompt_handler(stamp)
        try:
            out = await unused_submit(unused_executor("CLASSIC"), UNUSED_A)
        finally:
            instance.on_prompt_handlers.remove(stamp)
        check("unused outputs: a stamping handler after ours (BCVideoNodes') -> still on", out == ("", 1, None, UNUSED_EMPTY), f"{out}")
    finally:
        logging.getLogger().removeHandler(lines)


async def main():
    await setup()


async def main():
    await setup()
    process_monitor_default()

    # Join Image Lists: In3 exists only on the canvas; list outputs fan out downstream.
    out, _ = await run({
        "1": N("EmptyImage", width=64, height=48, batch_size=2, color=0),
        "2": N("EmptyImage", width=32, height=16, batch_size=1, color=0),
        "3": N("EmptyImage", width=8, height=8, batch_size=3, color=0),
        "4": N("BC_JoinImageLists", In1=["1", 0], In2=["2", 0], In3=["3", 0]),
        "5": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["4", 0]),
        "6": N("BC_MathExpression", expression="b", b=["4", 1]),
    }, "join")
    check("JoinImageLists: canvas-only In3 validates and executes", out is not None)
    check("JoinImageLists: Joined keeps slot order", out and out["5"]["value"] == [64048, 32016, 8008])
    check("JoinImageLists: Sizes", out and out["6"]["value"] == [1, 1, 1])

    # Any Switch: wildcard in and out, chained, feeding a typed INT input.
    out, _ = await run({
        "1": N("BC_MathExpression", expression="7"),
        "2": N("BC_AnySwitch"),
        "3": N("BC_AnySwitch", any_01=["2", 0], any_02=["1", 0]),
        "4": N("EmptyImage", width=8, height=8, batch_size=["3", 0], color=0),
        "5": N("BC_MathExpression", expression="a", a=["3", 0]),
        "6": N("BC_MathExpression", expression="a.width", a=["4", 0]),
    }, "switch")
    check("AnySwitch: wildcard links validate and execute", out is not None)
    check("AnySwitch: first non-None input wins", out and out["5"]["value"] == [7])
    check("AnySwitch: '*' output accepted by an INT input", out and out["6"]["value"] == [8])

    # Select Switch: canvas-only option slots, a selected name no static list holds, and lazy
    # option inputs: only the selected branch (node 3) runs, node 2 never does.
    def select_prompt(selected):
        return {
            "1": N("EmptyImage", width=16, height=8, batch_size=1, color=0),
            "2": N("ImageInvert", image=["1", 0]),
            "3": N("ImageScaleBy", image=["1", 0], upscale_method="nearest-exact", scale_by=2.0),
            "4": N("BC_SelectSwitch", selected=selected, option_a=["2", 0], option_b=["3", 0]),
            "5": N("BC_MathExpression", expression="a.width", a=["4", 0]),
        }
    server = Server()
    out, valid = await run(select_prompt("option_b"), "select", {"client_id": "test"}, server)
    check("SelectSwitch: canvas-only options and selected validate and execute", out is not None, f"{valid[1]}")
    check("SelectSwitch: the selected input comes out", out and out["5"]["value"] == [32])
    # The switch is announced twice: once when it asks for its lazy input, once when it runs.
    check("SelectSwitch: only the selected branch executes", sorted(set(executed_nodes(server))) == ["1", "3", "4", "5"], f"{executed_nodes(server)}")
    server = Server()
    out, _ = await run(select_prompt("option_c"), "select-unconnected", {"client_id": "test"}, server)
    errors = [data.get("exception_message", "") for event, data in server.events if event == "execution_error"]
    check("SelectSwitch: an unconnected selection fails naming the option",
          out is None and any("'option_c' has no input connected" in e for e in errors), f"{errors}")
    check("SelectSwitch: an unconnected selection runs no branch", "2" not in executed_nodes(server) and "3" not in executed_nodes(server), f"{executed_nodes(server)}")
    out, valid = await run(select_prompt(""), "select-empty")
    check("SelectSwitch: an empty selection fails validation", out is None and not valid[0])
    prompt = select_prompt(["6", 0])
    prompt["6"] = N("PrimitiveString", value="option_a")
    out, valid = await run(prompt, "select-driven")
    check("SelectSwitch: `selected` driven by a STRING output", out is not None and out["5"]["value"] == [16], f"{valid[1]}")

    # Math Expression: language coverage.
    out, _ = await run({
        "1": N("BC_MathExpression", expression="2"),
        "2": N("BC_MathExpression", expression="5"),
        "3": N("EmptyLatentImage", width=128, height=64, batch_size=1),
        "4": N("BC_MathExpression", expression="iif(b > a and not (a == 3), max(a, b) ** 2 + floor(7 / 2) + c.width - c.height, -1)", a=["1", 0], b=["2", 0], c=["3", 0]),
        "5": N("BC_MathExpression", expression="round(sqrt(b) * 100) + int(a / 2) + min(1, 2, 3) + abs(-4) + pow(2, 3) + ceil(0.2)", a=["1", 0], b=["2", 0]),
        "6": N("BC_MathExpression", expression=""),
    }, "math")
    check("MathExpression: iif / compare / bool / pow / floor / latent size", out and out["4"]["value"] == [25 + 3 + 128 - 64])
    check("MathExpression: round / sqrt / int / min / abs / pow / ceil", out and out["5"]["value"] == [224 + 1 + 1 + 4 + 8 + 1])
    check("MathExpression: empty expression is 0", out and out["6"]["value"] == [0])

    # Prompt List: two list outputs fan out, the third is a single string.
    out, _ = await run({
        "1": N("BC_PromptList", prepend_text="[", multiline_text="one\ntwo\nthree\nfour", append_text="]", start_index=1, max_rows=2),
        "2": N("PreviewAny", source=["1", 0]),
        "3": N("PreviewAny", source=["1", 1]),
        "4": N("PreviewAny", source=["1", 2]),
    }, "promptlist")
    check("PromptList: prompt list fans out", out and out["2"]["text"] == ["[two]", "[three]"])
    check("PromptList: body_text list", out and out["3"]["text"] == ["two", "three"])
    check("PromptList: show_help is one string", out and len(out["4"]["text"]) == 1)

    # Logic and mask nodes end to end.
    out, _ = await run({
        "1": N("BC_MathExpression", expression="0.7"),
        "2": N("BC_LogicBoolean", boolean=["1", 1]),
        "3": N("SolidMask", value=0.0, width=32, height=32),
        "4": N("BC_IsMaskEmpty", mask=["3", 0]),
        "5": N("BC_MaskFillHoles", masks=["3", 0]),
        "6": N("BC_MaskGrow", invert_mask=True, grow=2, blur=1, mask=["3", 0]),
        "7": N("PreviewAny", source=["2", 0]),
        "8": N("PreviewAny", source=["4", 0]),
        "9": N("BC_MathExpression", expression="a.width", a=["5", 0]),
        "10": N("BC_MathExpression", expression="a.width", a=["6", 0]),
        "11": N("BC_MaskFillHoles"),
        "12": N("BC_MathExpression", expression="a.height", a=["11", 0]),
    }, "logic-mask")
    check("LogicBoolean: linked FLOAT 0.7 -> True", out and out["7"]["text"] == ["True"])
    check("IsMaskEmpty: zero mask -> True", out and out["8"]["text"] == ["True"])
    check("MaskFillHoles / MaskGrow: (B, H, W) out", out and out["9"]["value"] == [32] and out["10"]["value"] == [32])
    check("MaskFillHoles: nothing wired -> 64x64 blank", out and out["12"]["value"] == [64])

    # Image scale: five outputs in the original order, width / height INTs into math nodes, MASK never None.
    scale = dict(aspect_ratio="original", proportional_width=1, proportional_height=1, fit="crop", method="lanczos",
                 round_to_multiple="16", scale_to_side="longest", scale_to_length=64, background_color="#000000")
    out, _ = await run({
        "1": N("EmptyImage", width=96, height=48, batch_size=1, color=0),
        "2": N("BC_ImageScaleByAspectRatio", image=["1", 0], **scale),
        "3": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["2", 0]),
        "4": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["2", 1]),
        "5": N("BC_MathExpression", expression="a * 1000 + b", a=["2", 3], b=["2", 4]),
        "6": N("PreviewAny", source=["2", 2]),
        "7": N("SolidMask", value=0.0, width=64, height=64),
        "8": N("BC_ImageScaleByAspectRatio", image=["1", 0], mask=["7", 0], **scale),
        "9": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["8", 1]),
    }, "image-scale")
    check("ImageScale: IMAGE 96x48 -> 64x32", out and out["3"]["value"] == [64032])
    check("ImageScale: no mask wired -> zero MASK 64x32, not None", out and out["4"]["value"] == [64032])
    check("ImageScale: width / height INT outputs", out and out["5"]["value"] == [64032])
    check("ImageScale: BOX is [orig_width, orig_height]", out and json.loads(out["6"]["text"][0]) == [96, 48])
    check("ImageScale: 64x64 placeholder mask -> zero MASK 64x32", out and out["9"]["value"] == [64032])

    # Seed: -1 from the API becomes a concrete seed, recorded in prompt and workflow metadata.
    workflow = {"nodes": [{"id": 1, "type": "BC_Seed", "widgets_values": [-1]}, {"id": 2, "type": "PreviewAny", "widgets_values": []}]}
    prompt = {"1": N("BC_Seed", seed=-1), "2": N("PreviewAny", source=["1", 0])}
    out, _ = await run(prompt, "seed-random", {"extra_pnginfo": {"workflow": workflow}})
    drawn = int(out["2"]["text"][0]) if out else None
    check("Seed: -1 -> concrete seed downstream", out is not None and drawn != -1 and 0 <= drawn <= 2 ** 53 - 1)
    check("Seed: prompt input rewritten for metadata", out is not None and prompt["1"]["inputs"]["seed"] == drawn)
    check("Seed: workflow widgets_values rewritten for metadata", out is not None and workflow["nodes"][0]["widgets_values"] == [drawn])
    out, _ = await run({"1": N("BC_Seed", seed=123456789), "2": N("PreviewAny", source=["1", 0])}, "seed-fixed")
    check("Seed: fixed seed passes through", out and out["2"]["text"] == ["123456789"])

    # Caching: a fixed seed (0 included) is a cache hit the second time, and so
    # is everything below it; -1 draws a new seed, so it and its consumers run
    # every time.
    out, executed = await run_twice({"1": N("BC_Seed", seed=0), "2": N("PreviewAny", source=["1", 0])}, "seed-cache")
    check("Seed: 0 runs zero nodes on the second queue", out is not None and executed == [], f"{executed}")
    out, executed = await run_twice({"1": N("BC_Seed", seed=-1), "2": N("PreviewAny", source=["1", 0])}, "seed-volatile")
    check("Seed: -1 re-runs itself and its consumer", out is not None and executed == ["1", "2"], f"{executed}")

    # Show Text: list in, one box per element, list out.
    out, _ = await run({
        "1": N("BC_PromptList", prepend_text="", multiline_text="x\ny", append_text="", start_index=0, max_rows=10),
        "2": N("BC_ShowText", text=["1", 0]),
        "3": N("PreviewAny", source=["2", 0]),
    }, "showtext")
    check("ShowText: ui text is the list", out and out["2"]["text"] == ["x", "y"])
    check("ShowText: passes the list on", out and out["3"]["text"] == ["x", "y"])

    # Image Comparer: both sides land in temp as previews.
    out, _ = await run({
        "1": N("EmptyImage", width=16, height=16, batch_size=2, color=0),
        "2": N("EmptyImage", width=16, height=16, batch_size=1, color=255),
        "3": N("BC_ImageComparer", image_a=["1", 0], image_b=["2", 0]),
        "4": N("BC_ImageComparer", image_a=["1", 0]),
    }, "comparer")
    check("ImageComparer: a_images / b_images previews", out and len(out["3"]["a_images"]) == 2 and len(out["3"]["b_images"]) == 1 and out["3"]["a_images"][0]["type"] == "temp")
    check("ImageComparer: one side only", out and len(out["4"]["a_images"]) == 2 and out["4"]["b_images"] == [])

    # Mask ports end to end: Draw Mask On Image paints through a full mask, Blockify keeps a full
    # mask full, Repeat Mask Batch triples the batch.
    out, _ = await run({
        "1": N("EmptyImage", width=32, height=16, batch_size=2, color=0),
        "2": N("SolidMask", value=1.0, width=32, height=16),
        "3": N("BC_DrawMaskOnImage", image=["1", 0], mask=["2", 0], color="255, 0, 0, 0.5", device="cpu"),
        "4": N("ImageToMask", image=["3", 0], channel="red"),
        "5": N("BC_IsMaskEmpty", mask=["4", 0]),
        "6": N("PreviewAny", source=["5", 0]),
        "7": N("BC_BlockifyMask", masks=["2", 0], block_size=8, device="cpu"),
        "8": N("BC_RepeatMaskBatch", mask=["7", 0], amount=3),
        "9": N("MaskToImage", mask=["8", 0]),
        "10": N("GetImageSize", image=["9", 0]),
        "11": N("BC_MathExpression", expression="a * 1000000 + b * 1000 + c", a=["10", 0], b=["10", 1], c=["10", 2]),
    }, "mask-ports")
    check("DrawMaskOnImage: validates, runs, paints the red channel", out is not None and out["6"]["text"] == ["False"])
    check("BlockifyMask -> RepeatMaskBatch: 32x16, 1 mask x 3", out is not None and out["11"]["value"] == [32016003])

    # Image Resize: four outputs; pad writes the padding mask; width / height are INTs.
    out, _ = await run({
        "1": N("EmptyImage", width=96, height=48, batch_size=2, color=0),
        "2": N("BC_ImageResize", image=["1", 0], width=64, height=64, upscale_method="lanczos", keep_proportion="pad",
               pad_color="0, 0, 0", crop_position="center", divisible_by=2),
        "3": N("BC_MathExpression", expression="a * 1000 + b", a=["2", 1], b=["2", 2]),
        "4": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["2", 0]),
        "5": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["2", 3]),
        "6": N("BC_ImageResize", image=["1", 0], width=0, height=24, upscale_method="area", keep_proportion="crop",
               pad_color="0, 0, 0", crop_position="left", divisible_by=8, device="cpu"),
        "7": N("BC_MathExpression", expression="a * 1000 + b", a=["6", 1], b=["6", 2]),
    }, "image-resize")
    check("ImageResize: pad 96x48 -> 64x64, INT outputs", out is not None and out["3"]["value"] == [64064] and out["4"]["value"] == [64064])
    check("ImageResize: pad without a mask -> 64x64 padding mask", out is not None and out["5"]["value"] == [64064])
    check("ImageResize: crop, width 0 keeps the source width -> 96x24", out is not None and out["7"]["value"] == [96024])

    # Anything Everywhere / Fast Groups Bypasser: present in the prompt, never executed, never in the way.
    out, _ = await run({
        "1": N("BC_MathExpression", expression="3"),
        "2": N("BC_AnythingEverywhere", anything=["1", 0]),
        "3": N("BC_FastGroupsBypasser"),
        "4": N("PreviewAny", source=["1", 0]),
    }, "everywhere")
    check("AnythingEverywhere / FastGroupsBypasser: prompt validates and runs", out is not None and out["4"]["text"] == ["3"])

    # Type checking is still on: a real mismatch is rejected.
    out, valid = await run({
        "1": N("EmptyImage", width=8, height=8, batch_size=1, color=0),
        "2": N("BC_LogicBoolean", boolean=["1", 0]),
        "3": N("PreviewAny", source=["2", 0]),
    }, "mismatch")
    check("validation rejects IMAGE -> FLOAT", out is None and not valid[0])

    # Caching: comfy_execution/caching.py builds a node's key from its own
    # inputs and IS_CHANGED value plus those of every ancestor, so one node
    # answering NaN re-runs its whole subtree. Every deterministic node,
    # queued twice with nothing changed, runs zero nodes the second time.
    solid = N("SolidMask", value=0.0, width=8, height=8)
    image = N("EmptyImage", width=16, height=8, batch_size=1, color=0)
    deterministic = {
        "LogicBoolean": {"1": N("BC_LogicBoolean", boolean=0.7), "2": N("PreviewAny", source=["1", 0])},
        "IsMaskEmpty": {"1": solid, "2": N("BC_IsMaskEmpty", mask=["1", 0]), "3": N("PreviewAny", source=["2", 0])},
        "MaskFillHoles": {"1": solid, "2": N("BC_MaskFillHoles", masks=["1", 0]), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "MaskGrow": {"1": solid, "2": N("BC_MaskGrow", invert_mask=False, grow=1, blur=1, mask=["1", 0]), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "ImageScaleByAspectRatio": {"1": image, "2": N("BC_ImageScaleByAspectRatio", image=["1", 0], **scale), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "JoinImageLists": {"1": image, "2": image, "3": N("BC_JoinImageLists", In1=["1", 0], In2=["2", 0]), "4": N("BC_MathExpression", expression="b", b=["3", 1])},
        "MathExpression": {"1": N("BC_MathExpression", expression="2 + 3")},
        "PromptList": {"1": N("BC_PromptList", prepend_text="", multiline_text="x\ny", append_text="", start_index=0, max_rows=10), "2": N("PreviewAny", source=["1", 0])},
        "AnySwitch": {"1": N("BC_MathExpression", expression="7"), "2": N("BC_AnySwitch", any_01=["1", 0]), "3": N("PreviewAny", source=["2", 0])},
        "SelectSwitch": {"1": N("BC_MathExpression", expression="7"), "2": N("BC_SelectSwitch", selected="option_a", option_a=["1", 0]), "3": N("PreviewAny", source=["2", 0])},
        "ShowText": {"1": N("BC_PromptList", prepend_text="", multiline_text="x\ny", append_text="", start_index=0, max_rows=10), "2": N("BC_ShowText", text=["1", 0]), "3": N("PreviewAny", source=["2", 0])},
        "ImageComparer": {"1": image, "2": N("BC_ImageComparer", image_a=["1", 0], image_b=["1", 0])},
        "DrawMaskOnImage": {"1": image, "2": solid, "3": N("BC_DrawMaskOnImage", image=["1", 0], mask=["2", 0], color="0, 0, 0"), "4": N("BC_MathExpression", expression="a.width", a=["3", 0])},
        "BlockifyMask": {"1": solid, "2": N("BC_BlockifyMask", masks=["1", 0], block_size=8), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "RepeatMaskBatch": {"1": solid, "2": N("BC_RepeatMaskBatch", mask=["1", 0], amount=2), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "ImageResize": {"1": image, "2": N("BC_ImageResize", image=["1", 0], width=8, height=8, upscale_method="bilinear", keep_proportion="stretch",
                                           pad_color="0, 0, 0", crop_position="center", divisible_by=2), "3": N("BC_MathExpression", expression="a.width", a=["2", 0])},
        "PowerLoraLoader": {"1": N("BC_PowerLoraLoader"), "2": N("PreviewAny", source=["1", 0])},
        "AnythingEverywhere / FastGroupsBypasser": {"1": N("BC_MathExpression", expression="3"), "2": N("BC_AnythingEverywhere", anything=["1", 0]), "3": N("BC_FastGroupsBypasser"), "4": N("PreviewAny", source=["1", 0])},
    }
    for name, prompt in deterministic.items():
        out, executed = await run_twice(prompt, f"cache-{name}")
        check(f"cache: {name} runs zero nodes on the second queue", out is not None and executed == [], f"{executed}")

    # Math Expression: the expression decides. "b > a" over Get Image Size,
    # through Logic Boolean into a consumer (the shape of a switch feeding a
    # sampler) is cached down to the last node; randomint() re-runs itself and
    # its consumer; a Title.widget reference cannot be fingerprinted (the
    # prompt is not available to IS_CHANGED) and is volatile too.
    out, executed = await run_twice({
        "1": N("EmptyImage", width=48, height=64, batch_size=1, color=0),
        "2": N("GetImageSize", image=["1", 0]),
        "3": N("BC_MathExpression", expression="b > a", a=["2", 0], b=["2", 1]),
        "4": N("BC_LogicBoolean", boolean=["3", 1]),
        "5": N("EmptyLatentImage", width=64, height=64, batch_size=["4", 2]),
        "6": N("BC_MathExpression", expression="a.width", a=["5", 0]),
    }, "cache-math-chain")
    check("MathExpression: 'b > a' evaluates through Logic Boolean", out is not None and out["3"]["value"] == [1] and out["6"]["value"] == [64])
    check("MathExpression: 'b > a' -> LogicBoolean -> consumer, all cached on the second queue", out is not None and executed == [], f"{executed}")
    out, executed = await run_twice({"1": N("BC_MathExpression", expression="randomint(0, 10)"), "2": N("PreviewAny", source=["1", 0])}, "cache-math-random")
    check("MathExpression: randomint() re-runs itself and its consumer", out is not None and executed == ["1", "2"], f"{executed}")
    math_node = nodes.NODE_CLASS_MAPPINGS["BC_MathExpression"]
    check("MathExpression: IS_CHANGED is the expression when deterministic", math_node.IS_CHANGED(expression="b > a", a=None, b=None, c=None, prompt={}, extra_pnginfo=None) == "b > a")
    check("MathExpression: IS_CHANGED is NaN for Title.widget", math.isnan(math_node.IS_CHANGED(expression="Loader.width", prompt={}, extra_pnginfo=None)))
    check("MathExpression: IS_CHANGED is NaN for randomchoice()", math.isnan(math_node.IS_CHANGED(expression="randomchoice(1, 2)")))

    # PostFx: the POSTFX_LOOK socket links and validates through a whole chain
    # (Theme -> LUT -> Custom Look -> Apply); the result is an IMAGE of the
    # input's size; every node is deterministic, so a second queue runs nothing.
    postfx_chain = {
        "1": N("EmptyImage", width=32, height=24, batch_size=2, color=8355711),
        "2": N("BC_PostFxTheme", theme=nodes.NODE_CLASS_MAPPINGS["BC_PostFxTheme"].INPUT_TYPES()["required"]["theme"][1]["default"]),
        "3": N("BC_PostFxLut", lut="none", intensity=1.0, look=["2", 0]),
        "4": N("BC_PostFxCustomLook", temp=0.2, tint=0.0, exposure=0.5, contrast=0.3, vibrance=0.0, saturation=1.0, grain=0.02, grain_size=1.5,
               vignette=0.3, halation=0.0, clarity=0.0, look=["3", 0]),
        "5": N("BC_PostFxApply", image=["1", 0], theme="none", condition="day_outdoor", strength=1.0, seed=3, batch_seed="increment", look=["4", 0]),
        "6": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["5", 0]),
    }
    out, executed = await run_twice(postfx_chain, "postfx-chain")
    check("PostFx: Theme -> LUT -> Custom Look -> Apply validates and runs", out is not None and out["6"]["value"] == [32024])
    check("PostFx: the whole chain is cached on the second queue", out is not None and executed == [], f"{executed}")
    out, _ = await run({
        "1": N("EmptyImage", width=48, height=32, batch_size=1, color=8355711),
        "2": N("BC_PostFxSignatureSheet", image=["1", 0], category="signature", condition="neutral", strength=1.0, columns=5),
        "3": N("BC_MathExpression", expression="a.width", a=["2", 0]),
    }, "postfx-sheet")
    check("PostFx: Signature Sheet renders an image", out is not None and out["3"]["value"][0] > 48)
    out, valid = await run({"1": N("EmptyImage", width=32, height=24, batch_size=1, color=0), "2": N("BC_PostFxApply", image=["1", 0], theme="none", condition="neutral", strength=1.0, seed=0, batch_seed="fixed", look=["1", 0])}, "postfx-badlink")
    check("PostFx: validation rejects IMAGE -> POSTFX_LOOK", out is None and not valid[0])

    # Caption Audit: an output node over a folder; the card lands in temp/ and
    # comes back through "ui"; the folder fingerprint keeps it cached.
    # The node only reads under the ComfyUI tree plus BC_CAPTION_ROOTS, and a
    # temp dataset is outside both; the server reads the variable at audit time.
    dataset = tempfile.mkdtemp(prefix="bcnodes_captions_")
    os.environ["BC_CAPTION_ROOTS"] = dataset
    for i, text in enumerate(["sks1 woman, red scarf, studio", "sks1 woman, red scarf, beach", "sks1 woman, red scarf, street"]):
        with open(os.path.join(dataset, f"{i}.txt"), "w") as fh:
            fh.write(text)
    audit = N("BC_CaptionAudit", directory=dataset, trigger="sks1", class_words="woman", fuse="", critical_threshold=0.85, warn_threshold=0.6,
              info_threshold=0.35, ngram_max=3, no_stopwords=False, recursive=False, table_rows=8)
    out, executed = await run_twice({"1": audit, "2": N("PreviewAny", source=["1", 3])}, "caption-audit")
    check("CaptionAudit: runs, card in temp/, 'red scarf' critical", out is not None and out["1"]["images"][0]["type"] == "temp" and out["2"]["text"] == ["1"])
    check("CaptionAudit: untouched folder is cached on the second queue", out is not None and executed == [], f"{executed}")
    with open(os.path.join(dataset, "0.txt"), "w") as fh:
        fh.write("sks1 woman, studio")
    out, _ = await run({"1": audit, "2": N("PreviewAny", source=["1", 3])}, "caption-audit-edited")
    check("CaptionAudit: editing a caption re-runs and clears the finding", out is not None and out["2"]["text"] == ["0"])

    # Social Media Export: an output node that writes under output/; the input
    # passes through and the report names every file.
    sme_inputs = nodes.NODE_CLASS_MAPPINGS["BC_SocialMediaExport"].INPUT_TYPES()["required"]
    platforms = {name: name == "instagram_feed" for name, spec in sme_inputs.items() if spec[0] == "BOOLEAN" and name != "allow_upscale"}
    out, executed = await run_twice({
        "1": N("EmptyImage", width=160, height=90, batch_size=1, color=8355711),
        "2": N("BC_SocialMediaExport", images=["1", 0], resize_mode="crop", quality=90, allow_upscale=False, filename_prefix="social/export", **platforms),
        "3": N("BC_MathExpression", expression="a.width", a=["2", 0]),
        "4": N("PreviewAny", source=["2", 1]),
    }, "social-export")
    import folder_paths
    written = os.path.join(folder_paths.get_output_directory(), "social", "export_instagram_feed_00001_.jpg")
    check("SocialMediaExport: writes the derivative, passes the input through, reports", out is not None and os.path.isfile(written) and out["3"]["value"] == [160] and "instagram_feed" in out["4"]["text"][0])
    check("SocialMediaExport: cached on the second queue", out is not None and executed == [], f"{executed}")

    # Save Image: names from the prompt's own widgets, files under output/,
    # ui.images only when image_preview is on.
    save_kw = dict(filename_prefix="shot", filename_keys="width, 1.height", foldername_prefix="", foldername_keys="run", delimiter="-",
                   save_job_data="basic, prompt", job_data_per_image=False, job_custom_text="", save_metadata=True, counter_digits=3,
                   counter_position="last", one_counter_per_folder=True, image_preview=True, output_ext=".png", quality=90, named_keys=False)
    out, executed = await run_twice({
        "1": N("EmptyImage", width=40, height=24, batch_size=1, color=8355711),
        "2": N("BC_SaveImage", images=["1", 0], **save_kw),
    }, "save-image")
    saved = os.path.join(folder_paths.get_output_directory(), "run", "shot-40-24-001.png")
    check("SaveImage: writes output/run/shot-40-24-001.png and lists it for the gallery",
          out is not None and os.path.isfile(saved) and out["2"]["images"] == [{"filename": "shot-40-24-001.png", "subfolder": "run", "type": "output"}])
    check("SaveImage: jobs.json written", os.path.isfile(os.path.join(folder_paths.get_output_directory(), "run", "jobs.json")))
    check("SaveImage: cached on the second queue", out is not None and executed == [], f"{executed}")
    out, _ = await run({
        "1": N("EmptyImage", width=40, height=24, batch_size=1, color=8355711),
        "2": N("BC_SaveImage", images=["1", 0], **dict(save_kw, image_preview=False)),
    }, "save-image-no-preview")
    check("SaveImage: image_preview off -> counter continues to 002, nothing listed",
          out is not None and os.path.isfile(saved.replace("001", "002")) and out["2"]["images"] == [])

    # Save Image With Caption: ComfyUI's get_save_image_path fills %width% / %height% and continues
    # the counter from the folder; each image has the caption next to it under its name; Preview as
    # Text shows the `filename` output.
    def captioned(batch, folder="output"):
        return {
            "1": N("EmptyImage", width=40, height=24, batch_size=batch, color=8355711),
            "2": N("PrimitiveString", value="a photo of subject_a"),
            "3": N("BC_SaveImageWithCaption", images=["1", 0], filename_prefix="captioned/shot-%width%x%height%", output_folder=folder,
                   caption_file_extension="caption", caption=["2", 0]),
            "4": N("PreviewAny", source=["3", 0]),
        }

    def listing(folder):
        return sorted(os.listdir(folder)) if os.path.isdir(folder) else []

    def caption_text(folder, name):
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            return f.read()

    folder = os.path.join(folder_paths.get_output_directory(), "captioned")
    out, executed = await run_twice(captioned(2), "save-image-with-caption")
    check("SaveImageWithCaption: output/captioned/shot-40x24_00001_.png + .caption, 00002 likewise, the caption as given",
          out is not None and listing(folder) == ["shot-40x24_00001_.caption", "shot-40x24_00001_.png", "shot-40x24_00002_.caption",
                                                   "shot-40x24_00002_.png"]
          and caption_text(folder, "shot-40x24_00002_.caption") == "a photo of subject_a" and out["4"]["text"] == ["shot-40x24_00002_.png"],
          f"{listing(folder)} {out and out.get('4')}")
    check("SaveImageWithCaption: cached on the second queue", out is not None and executed == [], f"{executed}")
    out, _ = await run(captioned(1), "save-image-with-caption-again")
    check("SaveImageWithCaption: the next prompt continues the counter to 00003",
          out is not None and out["4"]["text"] == ["shot-40x24_00003_.png"] and len(listing(folder)) == 6, f"{listing(folder)}")
    dataset = os.path.join(tempfile.mkdtemp(prefix="bcnodes_dataset_"), "set_a")
    out, _ = await run(captioned(1, dataset), "save-image-with-caption-absolute")
    check("SaveImageWithCaption: an absolute output_folder is used as it is, the prefix folder inside it",
          out is not None and listing(os.path.join(dataset, "captioned")) == ["shot-40x24_00001_.caption", "shot-40x24_00001_.png"],
          f"{listing(dataset)}")

    # Image Quality Gate: a flat image has zero entropy and fails; the verdict
    # int drives a Math Expression the way it would drive a switch.
    out, executed = await run_twice({
        "1": N("EmptyImage", width=96, height=64, batch_size=1, color=8355711),
        "2": N("BC_ImageQualityGate", image=["1", 0], shot_type="medium", blur_mode="center-weighted", blur_threshold=0.4, blur_var_threshold=30.0,
               sharpness_threshold=15.0, noise_threshold=25.0, clipping_threshold=0.02, entropy_threshold=5.0),
        "3": N("BC_MathExpression", expression="a + 10", a=["2", 1]),
        "4": N("BC_MathExpression", expression="a.width", a=["2", 0]),
    }, "quality-gate")
    check("ImageQualityGate: flat image -> FAIL (0), badge 420 wide", out is not None and out["3"]["value"] == [10] and out["4"]["value"] == [420])
    check("ImageQualityGate: cached on the second queue", out is not None and executed == [], f"{executed}")

    # Skin Texture with a connected mask: no SAM 3 needed; the IMAGE and MASK
    # outputs validate downstream and the node is deterministic in its inputs.
    out, executed = await run_twice({
        "1": N("EmptyImage", width=96, height=64, batch_size=1, color=10329501),
        "2": N("SolidMask", value=1.0, width=96, height=64),
        "3": N("BC_SkinTexture", image=["1", 0], sam3_model=nodes.NODE_CLASS_MAPPINGS["BC_SkinTexture"].INPUT_TYPES()["required"]["sam3_model"][1]["default"],
               texture=0.5, detail=0.6, pore_scale=1.0, feather=4, seed=1, threshold=0.5, mask=["2", 0]),
        "4": N("BC_MathExpression", expression="a.width", a=["3", 0]),
        "5": N("BC_IsMaskEmpty", mask=["3", 1]),
        "6": N("PreviewAny", source=["5", 0]),
    }, "skin-texture")
    check("SkinTexture: runs on a connected mask, outputs an image and a non-empty mask", out is not None and out["4"]["value"] == [96] and out["6"]["text"] == ["False"])
    check("SkinTexture: cached on the second queue", out is not None and executed == [], f"{executed}")

    # Frequency Merge: a flat red base and a flat blue detail merge to red (a flat detail has no
    # high frequencies); the output validates downstream and is cached; another size stops the run.
    out, executed = await run_twice({
        "1": N("EmptyImage", width=96, height=64, batch_size=2, color=0xFF0000),
        "2": N("EmptyImage", width=96, height=64, batch_size=2, color=0x0000FF),
        "3": N("BC_FrequencyMerge", base=["1", 0], detail=["2", 0], split_sigma=3.0, detail_strength=1.0),
        "4": N("BC_MathExpression", expression="a.width", a=["3", 0]),
        "5": N("ImageToMask", image=["3", 0], channel="red"),
        "6": N("BC_IsMaskEmpty", mask=["5", 0]),
        "7": N("PreviewAny", source=["6", 0]),
    }, "frequency-merge")
    check("FrequencyMerge: runs, 96 wide, base's red kept", out is not None and out["4"]["value"] == [96] and out["7"]["text"] == ["False"],
          f"{out and (out.get('4'), out.get('7'))}")
    check("FrequencyMerge: cached on the second queue", out is not None and executed == [], f"{executed}")
    out, _ = await run({
        "1": N("EmptyImage", width=96, height=64, batch_size=1, color=0),
        "2": N("EmptyImage", width=64, height=64, batch_size=1, color=0),
        "3": N("BC_FrequencyMerge", base=["1", 0], detail=["2", 0], split_sigma=3.0, detail_strength=1.0),
        "4": N("PreviewImage", images=["3", 0]),
    }, "frequency-merge-sizes")
    check("FrequencyMerge: another size stops the run", out is None)

    # SeedVR2 Framing Downscale needs SAM 3 to run; its downscale_factor validates into SeedVR2 Resize.
    framing = nodes.NODE_CLASS_MAPPINGS["BC_SeedVR2FramingDownscale"].INPUT_TYPES()["required"]
    framing_prompt = {
        "1": N("EmptyImage", width=96, height=64, batch_size=1, color=0),
        "2": N("BC_SeedVR2FramingDownscale", image=["1", 0], **{k: v[1]["default"] for k, v in framing.items() if k != "image"}),
        "3": N("BC_SeedVR2Resize", image=["1", 0], upscale_factor=2.0, downscale_factor=["2", 0], max_resolution=0, emulate_bf16=False),
        "4": N("PreviewImage", images=["3", 0]),
        "5": N("PreviewAny", source=["2", 1]),
    }
    valid = await execution.validate_prompt(str(uuid.uuid4()), framing_prompt, None)
    check("SeedVR2FramingDownscale: downscale_factor -> SeedVR2 Resize and face_fraction -> PreviewAny validate", valid[0], f"{valid[1]}")

    # SeedVR2 Chunk Size: its INT into core's Split SeedVR2 Latent, manual mode, through the DynamicCombo
    # sub-input chunking_mode.frames_per_chunk. The card is a 5090's 31.36 GiB: a 1088x1920 clip gets 41
    # frames per chunk (11 latent frames), so Split clamps a temporal_overlap of 100 to 10 latent frames.
    import comfy.model_management as mm
    real_total = mm.get_total_memory
    mm.get_total_memory = lambda dev=None, torch_total_too=False: int(31.36 * 2 ** 30)
    try:
        out, _ = await run({
            "1": N("EmptyHunyuanLatentVideo", width=1920, height=1088, length=81, batch_size=1),
            "2": N("BC_SeedVR2ChunkSize", latent=["1", 0], safety_margin=0.64),
            "3": N("SeedVR2TemporalChunk", latent=["1", 0], temporal_overlap=100, chunking_mode="manual",
                   **{"chunking_mode.frames_per_chunk": ["2", 0]}),
            "4": N("PreviewAny", source=["2", 0]),
            "5": N("PreviewAny", source=["3", 1]),
        }, "seedvr2-chunk-size")
    finally:
        mm.get_total_memory = real_total
    check("SeedVR2ChunkSize: 41 frames on a 31.36 GiB card, linked into Split SeedVR2 Latent's manual frames_per_chunk",
          out is not None and out["4"]["text"] == ["41"] and out["5"]["text"] == ["10"], f"{out and (out.get('4'), out.get('5'))}")

    # Auto Model Downloader answers NaN on purpose: whether a file exists is
    # decided on disk, not by the inputs. It has no outputs, so nothing below
    # it can be dragged along.
    out, executed = await run_twice({"1": N("BC_AutoModelDownloader", entries="[]")}, "cache-downloader")
    check("AutoModelDownloader: runs every queue by design", out is not None and executed == ["1"], f"{executed}")

    await unused_outputs()
    await process_monitor_hook()
    await process_monitor_clear()

    print(f"\n{results['pass']} passed, {results['fail']} failed")
    sys.exit(1 if results["fail"] else 0)


def pack_module(suffix):
    """A pack module as ComfyUI's loader named it (the pack directory's path is the package name)."""
    return next(m for name, m in sys.modules.items() if name.endswith(suffix))


def process_monitor_default():
    """The pack's own MONITOR is on by default: loaded with no saved setting (the user directory is
    a temp dir here), it started with its sampler thread and its executor hook. It is stopped
    again, so the checks after this one run as before."""
    monitor = pack_module(".nodes.process_monitor").MONITOR
    module, reason = pack_module(".pipelines.process_monitor.hook").find()
    check("ProcessMonitor: on by default: started at load with no saved setting, its hook installed",
          monitor is not None and monitor.settings.enabled and monitor.enabled and module is not None
          and module.execute is monitor.hook.wrapper, f"{monitor and monitor.status()} {reason}")
    if monitor is not None:
        monitor.stop()
        check("ProcessMonitor: ... stopped again, ComfyUI's executor back", not monitor.enabled
              and module is not None and module.execute is monitor.hook.original)


async def process_monitor_hook():
    """Process Monitor: the execution.py hook point found on this ComfyUI and wrapped around the real
    executor for one armed run, then the same prompt queued again and served from the cache. A
    Monitor of its own in a temp dir: the pack's MONITOR was stopped by process_monitor_default."""
    hook = pack_module(".pipelines.process_monitor.hook")
    mon = pack_module(".pipelines.process_monitor.monitor")
    blackbox = pack_module(".pipelines.process_monitor.blackbox")
    module, reason = hook.find()
    check("ProcessMonitor: the execution.py hook point is found", module is not None, reason)
    if module is None:
        return

    class Probe:
        def running(self):
            return None

        def status(self, prompt_id):
            return "success"

        def last_node(self):
            return None

        def push(self, payload):
            pass

    original = module.execute
    m = mon.Monitor(tempfile.mkdtemp(prefix="bcnodes_pm_"), Probe())
    m.ram, m.gpu = mon.memory_sources.ram_source(), None
    m.hook = hook.Hook(module, m)
    m.hook.install()
    prompt = {
        "1": N("EmptyImage", width=64, height=48, batch_size=2, color=0),
        "2": N("ImageInvert", image=["1", 0]),
        "3": N("BC_MathExpression", expression="a.width", a=["2", 0]),
        "4": N("PreviewImage", images=["2", 0]),
    }
    ex = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
    try:
        for _ in range(2):
            m.arm(True)
            prompt_id = str(uuid.uuid4())
            valid = await execution.validate_prompt(prompt_id, prompt, None)
            await ex.execute_async(copy.deepcopy(prompt), prompt_id, {"extra_pnginfo": {"workflow": {"id": "wf-test"}}}, valid[2])
            with m._lock:
                m._end_run("success")
    finally:
        m.hook.uninstall()
    check("ProcessMonitor: the hook is removed again", module.execute is original)
    second, first = [blackbox.read_records(p) for p in blackbox.run_files(m.runs_dir)]
    ends = {r["node"]: r for r in first if r["type"] == "node_end"}
    starts = {r["node"]: r for r in first if r["type"] == "node"}
    image = 2 * 48 * 64 * 3 * 4
    check("ProcessMonitor: armed run measures every node", sorted(ends) == ["1", "2", "3", "4"]
          and all(e["state"] == "executed" and e["seconds"] >= 0 for e in ends.values()), f"{ends}")
    check("ProcessMonitor: outputs by shape and bytes", ends.get("1", {}).get("outputs") == [
        {"shape": [2, 48, 64, 3], "dtype": "float32", "device": "cpu", "bytes": image}], f"{ends.get('1')}")
    check("ProcessMonitor: node start sees its input and the cache", starts.get("2", {}).get("inputs", [{}])[0].get("bytes") == image
          and starts["2"]["cache"] == image, f"{starts.get('2')}")
    rows = blackbox.run_report(second)["nodes"]
    check("ProcessMonitor: the second queue is from cache, not measured",
          sorted(r["node"] for r in rows if r["state"] == "cached") == ["1", "2", "3", "4"] and len(rows) == 4, f"{rows}")
    check("ProcessMonitor: the run keeps its own time and the measurement for Emulate",
          first[-1]["type"] == "end" and first[-1]["monitor"]["hook_s"] > 0 and m.measurement("wf-test") is not None)


def prompt_worker(queue, ex, stop, seen):
    """main.py's prompt worker as far as the full clear meets it: a thread of this name waiting in the
    real PromptQueue.get, and on its free flags ComfyUI's own unload and executor reset."""
    import comfy.model_management

    while not stop.is_set():
        queue.get(timeout=0.05)
        flags = queue.get_flags()
        if flags:
            seen.append(sorted(flags))
        if flags.get("unload_models", flags.get("free_memory", False)):
            comfy.model_management.unload_all_models()
        if flags.get("free_memory", False):
            ex.reset()


async def process_monitor_clear():
    """Process Monitor full clear: POST /bcnodes/monitor/clear on the real PromptServer and prompt
    queue. Refused while a prompt runs (409); an error without ComfyUI's prompt worker (500); with
    it, the report's fields, ComfyUI's own free done by the worker (the executor's caches replaced),
    and the same prompt running again afterwards, from scratch."""
    import threading

    import server
    from aiohttp.test_utils import make_mocked_request

    instance = server.PromptServer.instance
    handlers = {(r.method, r.path): r.handler for r in instance.routes}
    clear = handlers[("POST", "/bcnodes/monitor/clear")]

    async def post():
        response = await clear(make_mocked_request("POST", "/bcnodes/monitor/clear"))
        return response.status, json.loads(response.text)

    queue = instance.prompt_queue
    queue.currently_running[99] = (0, "running-prompt", {}, {}, [])
    try:
        status, body = await post()
    finally:
        del queue.currently_running[99]
    check("ProcessMonitor clear: refused while a prompt runs, saying to wait",
          status == 409 and "wait until the queue is empty" in body.get("error", ""), f"{status} {body}")

    status, body = await post()
    check("ProcessMonitor clear: without ComfyUI's prompt worker -> an error that says so",
          status == 500 and "prompt worker thread was not found" in body.get("error", ""), f"{status} {body}")

    prompt = {"1": N("EmptyImage", width=64, height=48, batch_size=2, color=0), "2": N("ImageInvert", image=["1", 0]),
              "3": N("PreviewImage", images=["2", 0])}
    ex = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})

    async def queue_prompt():
        prompt_id = str(uuid.uuid4())
        valid = await execution.validate_prompt(prompt_id, prompt, None)
        ex.server.events.clear()
        await ex.execute_async(copy.deepcopy(prompt), prompt_id, {"client_id": "test"}, valid[2])
        return ex.success and executed_nodes(ex.server)

    first = await queue_prompt()
    caches = ex.caches
    stop, seen = threading.Event(), []
    worker = threading.Thread(target=prompt_worker, args=(queue, ex, stop, seen), daemon=True)
    worker.start()
    try:
        status, report = await post()
    finally:
        stop.set()
        worker.join(5)
    steps = [s["name"] for s in report.get("steps", [])]
    check("ProcessMonitor clear: the report has the baseline, before, after, each step and what remains",
          status == 200 and set(report) == {"baseline", "before", "after", "steps", "remaining"}
          and steps == ["comfyui_free", "pack_models", "garbage", "torch_caches", "malloc_trim"]
          and all("ram" in s["freed"] and s["found"] for s in report["steps"]), f"{status} {report}")
    check("ProcessMonitor clear: the baseline was read at the server's startup (it ran in unused_outputs)",
          (report.get("baseline") or {}).get("ram") is not None, f"{report.get('baseline')}")
    check("ProcessMonitor clear: ComfyUI's own free ran on its worker: both flags, the executor's caches replaced",
          seen == [["free_memory", "unload_models"]] and ex.caches is not caches, f"{seen}")
    again = await queue_prompt()
    check("ProcessMonitor clear: the next prompt runs normally, every node again (nothing cached)",
          first == ["1", "2", "3"] and again == ["1", "2", "3"], f"{first} {again}")
    pack_hooks = getattr(instance, "bc_full_clear_hooks", None) or []
    detail = next((s["detail"] for s in report.get("steps", []) if s["name"] == "pack_models"), None)
    check("ProcessMonitor clear: the pack's hook is in the server's bc_full_clear_hooks and the report has its row",
          [h.__qualname__ for h in pack_hooks] == ["release_pack_models"] and detail is not None
          and [r["hook"].split()[-1] for r in detail] == ["release_pack_models"] and detail[0]["error"] is None, f"{pack_hooks} {detail}")
    status_handler = handlers[("GET", "/bcnodes/monitor/status")]
    body = json.loads((await status_handler(make_mocked_request("GET", "/bcnodes/monitor/status"))).text)
    check("ProcessMonitor clear: the status route carries the baseline", body.get("baseline") == report.get("baseline"))


if __name__ == "__main__":
    asyncio.run(main())

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
import types
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

    # Aspect Prompt List: width / height lists drive a latent node once per prompt, each at its own size.
    out, _ = await run({
        "1": N("BC_AspectPromptList", prepend_text="[", multiline_text="[64x32]\none\ntwo\n\n[32x48]\nthree", append_text="]"),
        "2": N("PreviewAny", source=["1", 0]),
        "3": N("EmptyLatentImage", width=["1", 2], height=["1", 3], batch_size=1),
        "4": N("BC_MathExpression", expression="a.width * 1000 + a.height", a=["3", 0]),
    }, "aspectpromptlist")
    check("AspectPromptList: prompt list fans out", out and out["2"]["text"] == ["[one]", "[two]", "[three]"])
    check("AspectPromptList: one latent per prompt at its section's size", out and out["4"]["value"] == [64032, 64032, 32048])

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
    # The first input (key image_a) is the B side, the second (image_b) the A side.
    check("ImageComparer: a_images / b_images previews", out and len(out["3"]["b_images"]) == 2 and len(out["3"]["a_images"]) == 1 and out["3"]["b_images"][0]["type"] == "temp")
    check("ImageComparer: one side only", out and len(out["4"]["b_images"]) == 2 and out["4"]["a_images"] == [])

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
        "AspectPromptList": {"1": N("BC_AspectPromptList", prepend_text="", multiline_text="[8x8]\nx\ny", append_text=""), "2": N("PreviewAny", source=["1", 0])},
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

    await cond_pipe()
    await lora_key_fix()
    await vae_loader()
    await latent_upscale()
    await sage_attention()
    await detail_boost()
    await qwen_image_shift()
    await dype_models()
    detailer_zimage_patch()
    detailer_qwen_fun_controlnet()
    lm_core()
    await lm_nodes()
    await claude_nodes()
    await claude_chat_routes()
    await preflight_nodes()
    await unused_outputs()
    await process_monitor_hook()
    await process_monitor_clear()

    print(f"\n{results['pass']} passed, {results['fail']} failed")
    sys.exit(1 if results["fail"] else 0)


async def cond_pipe():
    """Cond Pipe Pack -> Unpack through validate_prompt + PromptExecutor, with two test-only nodes: one makes a
    CONDITIONING, an output node keeps the one it gets. Every slot reaches its keeper as the object its maker
    returned. A slot left unlinked on Pack whose Unpack output is linked stops the keeper it feeds with
    Unpack's message (core's ExecutionBlocker: an execution_error event naming the keeper, which never runs);
    one whose Unpack output is unlinked does nothing."""
    import torch

    slots = ("positive", "negative", "cn_positive", "cn_negative")
    made, kept = {}, {}

    class MakeCond:
        """A test-only node: a CONDITIONING whose tensor holds `tag`."""
        RETURN_TYPES = ("CONDITIONING",)
        FUNCTION = "make"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"tag": ("INT", {"default": 0})}}

        def make(self, tag):
            made[tag] = [[torch.full((1, 2, 4), float(tag)), {}]]
            return (made[tag],)

    class KeepCond:
        """A test-only output node: keeps the CONDITIONING it gets under `name`."""
        RETURN_TYPES = ()
        FUNCTION = "keep"
        OUTPUT_NODE = True
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"conditioning": ("CONDITIONING",), "name": ("STRING", {"default": ""})}}

        def keep(self, conditioning, name):
            kept[name] = conditioning
            return {}

    def prompt(packed, linked):
        """A maker (node i + 1, tag i + 1) linked into Pack per slot in `packed`, a keeper (node i + 21) on
        Unpack's output per slot in `linked`."""
        p = {"10": N("BC_CondPipePack", **{name: [str(i + 1), 0] for i, name in enumerate(slots) if name in packed}),
             "11": N("BC_CondPipeUnpack", pipe=["10", 0])}
        for i, name in enumerate(slots):
            if name in packed:
                p[str(i + 1)] = N("BCTest_MakeCond", tag=i + 1)
            if name in linked:
                p[str(i + 21)] = N("BCTest_KeepCond", conditioning=["11", i], name=name)
        return p

    nodes.NODE_CLASS_MAPPINGS["BCTest_MakeCond"] = MakeCond
    nodes.NODE_CLASS_MAPPINGS["BCTest_KeepCond"] = KeepCond
    try:
        out, valid = await run(prompt(slots, slots), "cond-pipe")
        check("CondPipe: Pack -> Unpack validates and executes", out is not None, f"{valid[1]}")
        check("CondPipe: every slot reaches its keeper as the object its maker returned",
              sorted(kept) == sorted(slots) and all(kept[name] is made[i + 1] for i, name in enumerate(slots)),
              f"{sorted(kept)}")
        kept.clear()
        server = Server()
        await run(prompt(("positive", "negative"), ("positive", "negative", "cn_positive")), "cond-pipe-empty",
                  {"client_id": "test"}, server)
        errors = [(data.get("node_id"), data.get("exception_message")) for event, data in server.events
                  if event == "execution_error"]
        check("CondPipe: a linked empty slot stops the node it feeds with Unpack's message; an unlinked one does nothing",
              errors == [("23", "Execution Blocked: Cond Pipe Unpack: the pipe holds no cn_positive. Link cn_positive "
                                "on Cond Pipe Pack, or unlink Cond Pipe Unpack's cn_positive output.")], f"{errors}")
        check("CondPipe: the keeper of the empty slot never runs, the others do", sorted(kept) == ["negative", "positive"],
              f"{sorted(kept)}")
    finally:
        del nodes.NODE_CLASS_MAPPINGS["BCTest_MakeCond"], nodes.NODE_CLASS_MAPPINGS["BCTest_KeepCond"]


def pack_module(suffix):
    """A pack module as ComfyUI's loader named it (the pack directory's path is the package name)."""
    return next(m for name, m in sys.modules.items() if name.endswith(suffix))


async def lora_key_fix():
    """Lora Loader (Key Fix) against core's own LoRA loader, on a real Wan 2.1 T2V model (2 blocks, dim
    128, random weights) built by core from a synthetic file, with two synthetic LoRA files holding the
    key patterns core leaves out: lightx2v's distill `.diff_m` (two blocks and the head, next to a plain
    LoRA pair core maps) and DiffSynth's PEFT keys without the `diffusion_model.` prefix (SVI 2.0's
    `blocks.N.<module>.lora_A.default.weight`). As they are, core loads none of those keys; renamed,
    every key; through the node each lands on its weight with the strength given.

    Power Lora Loader applies its rows through the same path: a LoRA with nothing to rename gives
    exactly the patches its code before the key fix gave (core's load_lora_for_models on the file as
    loaded, no metadata attached), and the two LoRAs above give the key-fix node's patches."""
    import importlib

    import comfy.lora
    import comfy.lora_convert
    import comfy.ops
    import comfy.sd
    import comfy.utils
    import folder_paths
    import torch
    from comfy.ldm.wan.model import WanModel
    from comfy.weight_adapter import WeightAdapterBase
    from safetensors.torch import save_file

    root = tempfile.mkdtemp(prefix="bcnodes_lora_")
    for folder in ("diffusion_models", "loras"):
        os.makedirs(os.path.join(root, folder))
        folder_paths.add_model_folder_path(folder, os.path.join(root, folder))
    torch.manual_seed(0)
    net = WanModel(model_type="t2v", dim=128, ffn_dim=256, num_heads=1, num_layers=2, operations=comfy.ops.disable_weight_init)
    unet = os.path.join(root, "diffusion_models", "tiny_wan.safetensors")
    save_file({k: torch.randn(v.shape) * 0.02 for k, v in net.state_dict().items()}, unet)
    distill = {f"diffusion_model.blocks.{i}.diff_m": torch.randn(1, 6, 128) for i in range(2)}
    distill["diffusion_model.head.diff_m"] = torch.randn(1, 2, 128)
    distill["diffusion_model.blocks.0.self_attn.q.lora_down.weight"] = torch.randn(4, 128)
    distill["diffusion_model.blocks.0.self_attn.q.lora_up.weight"] = torch.randn(128, 4)
    distill["diffusion_model.blocks.0.self_attn.q.alpha"] = torch.tensor(4.0)
    peft = {}
    for i in range(2):
        peft[f"blocks.{i}.cross_attn.k.lora_A.default.weight"] = torch.randn(4, 128)
        peft[f"blocks.{i}.cross_attn.k.lora_B.default.weight"] = torch.randn(128, 4)
    save_file(distill, os.path.join(root, "loras", "distill.safetensors"))
    save_file(peft, os.path.join(root, "loras", "peft.safetensors"))

    patcher = comfy.sd.load_diffusion_model(unet)
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    node_class = nodes.NODE_CLASS_MAPPINGS["BC_LoraLoaderKeyFix"]
    fix_keys = importlib.import_module(node_class.__module__.rsplit(".", 2)[0] + ".libs.lora_keys").fix_keys

    def not_loaded(lora):
        """The keys core's load_lora reports as not loaded."""
        lines = Lines()
        logging.getLogger().addHandler(lines)
        try:
            comfy.lora.load_lora(lora, key_map)
        finally:
            logging.getLogger().removeHandler(lines)
        return sorted(line.split(": ", 1)[1] for line in lines.lines if line.startswith("lora key not loaded: "))

    for name, lora, dropped in (("distill", distill, sorted(k for k in distill if k.endswith(".diff_m"))), ("peft", peft, sorted(peft))):
        fix = fix_keys(lora, key_map)
        renamed = {fix.renamed.get(k, k): v for k, v in lora.items()}
        check(f"LoraLoaderKeyFix: core leaves out the {len(dropped)} {name} keys as they are, none once renamed",
              not_loaded(lora) == dropped and not_loaded(renamed) == [] and sorted(fix.renamed) == dropped and fix.unmapped == [],
              f"{not_loaded(lora)} {not_loaded(renamed)} {fix}")

    node = node_class()
    (patched,) = node.load(patcher, "distill.safetensors", 0.5)
    (patched,) = node.load(patched, "peft.safetensors", 0.5)

    def base(key):
        return comfy.utils.get_attr(patcher.model, key).float()

    def merged(key):
        """The weight core's calculate_weight makes from the base weight and the patches on `key`."""
        return comfy.lora.calculate_weight(patched.patches[key], base(key).clone(), key)

    blocks = ("diffusion_model.blocks.0", "diffusion_model.blocks.1")
    check("LoraLoaderKeyFix: patches on every modulation, the plain pair's weight and both k weights",
          set(patched.patches) == {f"{b}.modulation" for b in blocks} | {f"{b}.cross_attn.k.weight" for b in blocks}
          | {"diffusion_model.head.modulation", "diffusion_model.blocks.0.self_attn.q.weight"}, f"{sorted(patched.patches)}")
    check("LoraLoaderKeyFix: a modulation is the base + 0.5 x its .diff_m (blocks and head)",
          all(torch.allclose(merged(f"{b}.modulation"), base(f"{b}.modulation") + 0.5 * distill[f"{b}.diff_m"], atol=1e-6)
              for b in blocks + ("diffusion_model.head",)))
    check("LoraLoaderKeyFix: a PEFT k weight is the base + 0.5 x B @ A",
          all(torch.allclose(merged(f"diffusion_model.blocks.{i}.cross_attn.k.weight"),
                             base(f"diffusion_model.blocks.{i}.cross_attn.k.weight")
                             + 0.5 * peft[f"blocks.{i}.cross_attn.k.lora_B.default.weight"] @ peft[f"blocks.{i}.cross_attn.k.lora_A.default.weight"],
                             atol=1e-5) for i in range(2)))

    def same(a, b):
        """Equal patch structures: tensors of one dtype and shape with every value equal, adapters of
        one type with equal fields, dicts with the same keys in the same order."""
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, WeightAdapterBase):
            return type(a) is type(b) and same(vars(a), vars(b))
        return a == b

    def logged(fn):
        """fn()'s result and the console lines it logged, INFO and up."""
        lines, root_logger = Lines(), logging.getLogger()
        level = root_logger.level
        root_logger.addHandler(lines)
        root_logger.setLevel(logging.INFO)
        try:
            return fn(), lines.lines
        finally:
            root_logger.removeHandler(lines)
            root_logger.setLevel(level)

    # A LoRA core loads whole as it is (Wan Fun's lora_unet__ keys, which core's conversion renames, a
    # PEFT pair, a .diff and a .diff_b), saved with metadata. Before the key fix Power Lora Loader
    # handed core the loaded file and no metadata; its patches now are exactly those.
    head, head_bias = base("diffusion_model.head.head.weight"), base("diffusion_model.head.head.bias")
    clean = {
        "lora_unet__blocks_0_cross_attn_k.lora_down.weight": torch.randn(4, 128),
        "lora_unet__blocks_0_cross_attn_k.lora_up.weight": torch.randn(128, 4),
        "lora_unet__blocks_0_cross_attn_k.alpha": torch.tensor(2.0),
        "diffusion_model.blocks.1.self_attn.q.lora_A.weight": torch.randn(4, 128),
        "diffusion_model.blocks.1.self_attn.q.lora_B.weight": torch.randn(128, 4),
        "diffusion_model.head.head.diff": torch.randn(head.shape),
        "diffusion_model.head.head.diff_b": torch.randn(head_bias.shape),
    }
    clean_path = os.path.join(root, "loras", "clean.safetensors")
    save_file(clean, clean_path, metadata={"ss_network_dim": "4"})
    check("PowerLoraLoader: core loads every key of the clean LoRA as it is, nothing to rename",
          not_loaded(comfy.lora_convert.convert_lora(dict(clean))) == [] and fix_keys(comfy.lora_convert.convert_lora(dict(clean)), key_map).renamed == {})
    before, _ = comfy.sd.load_lora_for_models(patcher, None, comfy.utils.load_torch_file(clean_path, safe_load=True), 0.5, 0)
    power = nodes.NODE_CLASS_MAPPINGS["BC_PowerLoraLoader"]()
    (after,), printed = logged(lambda: power.load_loras(model=patcher, lora_1={"on": True, "lora": "clean.safetensors", "strength": 0.5}))
    check("PowerLoraLoader: a LoRA with nothing to rename gives the patches of the code before the key fix, no metadata, no line",
          len(before.patches) == 4 and same(before.patches, after.patches) and "lora_metadata" not in after.attachments
          and before.attachments.keys() == after.attachments.keys() and not any("Power Lora Loader" in line for line in printed),
          f"{sorted(before.patches)} {sorted(after.patches)} {list(after.attachments)} {printed}")
    (keyfix_clean,) = node.load(patcher, "clean.safetensors", 0.5)
    check("LoraLoaderKeyFix: the same patches for the clean LoRA, with the file's metadata attached",
          same(before.patches, keyfix_clean.patches) and keyfix_clean.get_attachment("lora_metadata") == {"ss_network_dim": "4"})

    # The two LoRAs core loads in part, as two rows: the key-fix node's patches, one line per row.
    (power_fixed,), printed = logged(lambda: power.load_loras(model=patcher, lora_1={"on": True, "lora": "distill.safetensors", "strength": 0.5},
                                                              lora_2={"on": True, "lora": "peft.safetensors", "strength": 0.5}))
    check("PowerLoraLoader: the .diff_m and prefix-less PEFT rows renamed and applied, the key-fix node's patches",
          same(power_fixed.patches, patched.patches), f"{sorted(power_fixed.patches)}")
    check("PowerLoraLoader: one console line per row with renames, nothing left out by core",
          [line for line in printed if line.startswith("[BCNodes] Power Lora Loader")]
          == ["[BCNodes] Power Lora Loader: distill.safetensors: 6 keys, 3 renamed (3 .diff_m -> .modulation.diff, "
              "0 given the diffusion_model. prefix), 0 match no module of this model",
              "[BCNodes] Power Lora Loader: peft.safetensors: 4 keys, 4 renamed (0 .diff_m -> .modulation.diff, "
              "4 given the diffusion_model. prefix), 0 match no module of this model"]
          and not any(line.startswith("lora key not loaded") for line in printed), f"{printed}")

    # Through the executor: core's UNETLoader, then two key-fix loaders chained; the console lines
    # count the renames, core reports no key left out, and the second queue is served from the cache.
    lines = Lines()
    root_logger = logging.getLogger()
    level = root_logger.level
    root_logger.addHandler(lines)
    root_logger.setLevel(logging.INFO)
    try:
        out, executed = await run_twice({
            "1": N("UNETLoader", unet_name="tiny_wan.safetensors", weight_dtype="default"),
            "2": N("BC_LoraLoaderKeyFix", model=["1", 0], lora_name="distill.safetensors", strength=1.0),
            "3": N("BC_LoraLoaderKeyFix", model=["2", 0], lora_name="peft.safetensors", strength=1.0),
            "4": N("PreviewAny", source=["3", 0]),
        }, "lora-key-fix")
    finally:
        root_logger.removeHandler(lines)
        root_logger.setLevel(level)
    ours = [line for line in lines.lines if line.startswith("[BCNodes] Lora Loader (Key Fix)")]
    check("LoraLoaderKeyFix: UNETLoader -> two loaders validates, runs, and is cached on the second queue",
          out is not None and executed == [], f"{executed}")
    check("LoraLoaderKeyFix: one console line per LoRA with its renames, nothing left out by core",
          ours == ["[BCNodes] Lora Loader (Key Fix): distill.safetensors: 6 keys, 3 renamed (3 .diff_m -> .modulation.diff, "
                   "0 given the diffusion_model. prefix), 0 match no module of this model",
                   "[BCNodes] Lora Loader (Key Fix): peft.safetensors: 4 keys, 4 renamed (0 .diff_m -> .modulation.diff, "
                   "4 given the diffusion_model. prefix), 0 match no module of this model"]
          and not any(line.startswith("lora key not loaded") for line in lines.lines), f"{lines.lines}")


async def vae_loader():
    """VAE Loader ("Wan 2.1") through ComfyUI itself, on tiny Wan 2.1 VAEs (dim 16, core's WanVAE at its
    own init) saved under a temp vae folder: a plain one (decoder head 3 channels) and a packed one (12,
    as Wan2.1-VAE-upscale2x), plus a tiny TAESD for another model's file. The latent is seeded noise.

    The plain file decodes exactly as through core's Load VAE, with no console line. The packed file,
    through core's VAE Decode, is core's own decoder output for that file (core's Load VAE hands its 12
    channels on as they are) unpacked pixel by pixel into the 2x image, channel c x 4 + i x 2 + j to
    pixel (2h + i, 2w + j); VAE Decode (Tiled) gives the same image in one tile and the 2x size in
    several; VAE Encode is core's own; one console line names the file. The TAESD file is refused."""
    import comfy.ldm.wan.vae
    import folder_paths
    import torch
    from comfy.taesd.taesd import TAESD
    from safetensors.torch import save_file

    root = tempfile.mkdtemp(prefix="bcnodes_vae_")
    folder_paths.add_model_folder_path("vae", root)
    torch.manual_seed(0)
    for name, conv_out in (("tiny_wan.safetensors", 3), ("tiny_wan_2x.safetensors", 12)):
        # core builds the VAE without initialising its weights: seeded noise, RMS norm scales at 1
        net = comfy.ldm.wan.vae.WanVAE(dim=16, z_dim=16, temperal_downsample=[False, True, True], conv_out_channels=conv_out)
        save_file({key: torch.ones(t.shape) if key.endswith("gamma") else torch.randn(t.shape) * 0.05
                   for key, t in net.state_dict().items()}, os.path.join(root, name))
    save_file({key: t.contiguous() for key, t in TAESD().state_dict().items()}, os.path.join(root, "tiny_taesd.safetensors"))
    kept = {}

    class RandLatent:
        """A test-only node: a seeded noise Wan latent, 1 frame of 12 x 16."""
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "make"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def make(self):
            return ({"samples": torch.randn(1, 16, 1, 12, 16, generator=torch.Generator().manual_seed(1))},)

    def keeper(kind):
        class Keep:
            """A test-only output node: keeps the tensor it gets under `key`."""
            RETURN_TYPES = ()
            FUNCTION = "keep"
            OUTPUT_NODE = True
            CATEGORY = "testing"

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": (kind,), "key": ("STRING", {"default": ""})}}

            def keep(self, value, key):
                kept[key] = (value["samples"] if kind == "LATENT" else value).clone()
                return {}
        return Keep

    test_nodes = {"BCTest_RandLatent": RandLatent, "BCTest_KeepImage": keeper("IMAGE"), "BCTest_KeepLatent2": keeper("LATENT")}
    nodes.NODE_CLASS_MAPPINGS.update(test_nodes)
    lines = Lines()
    root_logger = logging.getLogger()
    level = root_logger.level
    root_logger.addHandler(lines)
    root_logger.setLevel(logging.INFO)
    try:
        tiled = dict(temporal_size=64, temporal_overlap=8)
        out, _ = await run({
            "1": N("BCTest_RandLatent"),
            "2": N("BC_VAELoader", model="Wan 2.1", vae_name="tiny_wan_2x.safetensors"),
            "3": N("VAEDecode", samples=["1", 0], vae=["2", 0]), "4": N("BCTest_KeepImage", value=["3", 0], key="ours"),
            "5": N("VAELoader", vae_name="tiny_wan_2x.safetensors"),
            "6": N("VAEDecode", samples=["1", 0], vae=["5", 0]), "7": N("BCTest_KeepImage", value=["6", 0], key="core"),
            "8": N("VAEDecodeTiled", samples=["1", 0], vae=["2", 0], tile_size=512, overlap=64, **tiled),
            "9": N("BCTest_KeepImage", value=["8", 0], key="tiled_one"),
            "10": N("VAEDecodeTiled", samples=["1", 0], vae=["2", 0], tile_size=64, overlap=32, **tiled),
            "11": N("BCTest_KeepImage", value=["10", 0], key="tiled_many"),
            "12": N("VAEEncode", pixels=["3", 0], vae=["2", 0]), "13": N("BCTest_KeepLatent2", value=["12", 0], key="enc_ours"),
            "14": N("VAEEncode", pixels=["3", 0], vae=["5", 0]), "15": N("BCTest_KeepLatent2", value=["14", 0], key="enc_core"),
            "16": N("BC_VAELoader", model="Wan 2.1", vae_name="tiny_wan.safetensors"),
            "17": N("VAEDecode", samples=["1", 0], vae=["16", 0]), "18": N("BCTest_KeepImage", value=["17", 0], key="plain_ours"),
            "19": N("VAELoader", vae_name="tiny_wan.safetensors"),
            "20": N("VAEDecode", samples=["1", 0], vae=["19", 0]), "21": N("BCTest_KeepImage", value=["20", 0], key="plain_core"),
        }, "vae-loader")
        refused_server = Server()
        refused, _ = await run({"1": N("BC_VAELoader", model="Wan 2.1", vae_name="tiny_taesd.safetensors"),
                                "2": N("PreviewAny", source=["1", 0])}, "vae-loader-taesd", {"client_id": "test"}, refused_server)
    finally:
        root_logger.removeHandler(lines)
        root_logger.setLevel(level)
        for key in test_nodes:
            del nodes.NODE_CLASS_MAPPINGS[key]

    check("VAELoader: the prompt validates and runs", out is not None)
    if out is None:
        return
    check("VAELoader: every output is finite (the comparisons below mean something)",
          all(bool(torch.isfinite(t).all()) for t in kept.values()), f"{[k for k, t in kept.items() if not torch.isfinite(t).all()]}")
    core = kept["core"]
    reference = torch.empty(1, 12 * 8 * 2, 16 * 8 * 2, 3, dtype=core.dtype)
    for c in range(3):
        for i in range(2):
            for j in range(2):
                reference[:, i::2, j::2, c] = core[..., c * 4 + i * 2 + j]
    ours = kept["ours"]
    check("VAELoader: core's Load VAE hands the packed file's 12 channels on at 1x", core.shape == (1, 96, 128, 12), f"{core.shape}")
    check("VAELoader: the packed file decodes to core's decoder output unpacked into the 2x image",
          ours.shape == (1, 192, 256, 3) and torch.equal(ours, reference), f"{ours.shape}")
    check("VAELoader: VAE Decode (Tiled) in one tile gives the same image",
          torch.allclose(kept["tiled_one"], ours, atol=1e-6), f"{(kept['tiled_one'] - ours).abs().max()}")
    check("VAELoader: VAE Decode (Tiled) in several tiles gives the 2x image",
          kept["tiled_many"].shape == (1, 192, 256, 3) and bool(torch.isfinite(kept["tiled_many"]).all()), f"{kept['tiled_many'].shape}")
    check("VAELoader: VAE Encode is core's own", torch.equal(kept["enc_ours"], kept["enc_core"]),
          f"{kept['enc_ours'].dtype} {kept['enc_core'].dtype} {(kept['enc_ours'] - kept['enc_core']).abs().max()}")
    check("VAELoader: the plain file decodes exactly as through core's Load VAE",
          torch.equal(kept["plain_ours"], kept["plain_core"]) and kept["plain_ours"].shape == (1, 96, 128, 3),
          f"{kept['plain_ours'].dtype} {kept['plain_core'].dtype} {(kept['plain_ours'] - kept['plain_core']).abs().max()}")
    ours_lines = [line for line in lines.lines if line.startswith("[BCNodes] VAE Loader")]
    check("VAELoader: one console line, for the packed file only",
          ours_lines == ["[BCNodes] VAE Loader: tiny_wan_2x.safetensors: the decoder packs a 2x image into 12 channels: "
                         "it decodes at 2x"], f"{ours_lines}")
    errors = [data.get("exception_message", "") for event, data in refused_server.events if event == "execution_error"]
    check("VAELoader: another model's file is refused naming it",
          refused is None and any(e.startswith("tiny_taesd.safetensors is not a Wan 2.1 VAE") for e in errors), f"{errors}")


async def latent_upscale():
    """Latent Upscale ("Wan 2.1 2x") through ComfyUI itself, on a seeded random LatentModel3d saved as a file the
    node picks from a temp latent_upscale_models folder. A video-shaped Wan latent (1, 16, 2, 6, 8) and Empty SD3
    Latent Image's (1, 16, 6, 8) come out doubled on height and width, equal to the network run directly, with
    the latent's other fields kept."""
    import importlib

    import folder_paths
    import torch
    from safetensors.torch import save_file

    loader = pack_module(".models.wan_latent_upscale.loader")
    arch = importlib.import_module(loader.__name__.rsplit(".", 1)[0] + ".arch")  # imported on the first load
    root = tempfile.mkdtemp(prefix="bcnodes_latent_upscale_")
    folder_paths.add_model_folder_path("latent_upscale_models", root)
    torch.manual_seed(0)
    net = arch.LatentModel3d().eval()
    save_file({key: t.contiguous() for key, t in net.state_dict().items()}, os.path.join(root, "tiny_upscaler.safetensors"))
    kept = {}

    class VideoLatent:
        """A test-only node: a seeded noise Wan latent, 2 frames of 6 x 8."""
        RETURN_TYPES = ("LATENT",)
        FUNCTION = "make"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def make(self):
            return ({"samples": torch.randn(1, 16, 2, 6, 8, generator=torch.Generator().manual_seed(2)), "batch_index": [0]},)

    class KeepLatent:
        """A test-only output node: keeps the latent dict it gets under `key`."""
        RETURN_TYPES = ()
        FUNCTION = "keep"
        OUTPUT_NODE = True
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"value": ("LATENT",), "key": ("STRING", {"default": ""})}}

        def keep(self, value, key):
            kept[key] = dict(value)
            return {}

    test_nodes = {"BCTest_VideoLatent": VideoLatent, "BCTest_KeepLatentDict": KeepLatent}
    nodes.NODE_CLASS_MAPPINGS.update(test_nodes)
    try:
        out, _ = await run({
            "1": N("BCTest_VideoLatent"),
            "2": N("BC_LatentUpscale", samples=["1", 0], model="Wan 2.1 2x", upscale_model_name="tiny_upscaler.safetensors"),
            "3": N("BCTest_KeepLatentDict", value=["2", 0], key="video"), "4": N("BCTest_KeepLatentDict", value=["1", 0], key="video_in"),
            "5": N("EmptySD3LatentImage", width=64, height=48, batch_size=1),
            "6": N("BC_LatentUpscale", samples=["5", 0], model="Wan 2.1 2x", upscale_model_name="tiny_upscaler.safetensors"),
            "7": N("BCTest_KeepLatentDict", value=["6", 0], key="image"), "8": N("BCTest_KeepLatentDict", value=["5", 0], key="image_in"),
        }, "latent-upscale")
    finally:
        for key in test_nodes:
            del nodes.NODE_CLASS_MAPPINGS[key]
        loader.unload()

    check("LatentUpscale: the prompt validates and runs", out is not None)
    if out is None:
        return
    with torch.no_grad():
        video, image = kept["video"]["samples"], kept["image"]["samples"]
        video_ref = net(kept["video_in"]["samples"])
        image_ref = net(kept["image_in"]["samples"].unsqueeze(2)).squeeze(2)
        # float32 rounding only: the executor's CPU convolutions differ from this call's in the last bits (4e-6 seen)
        check("LatentUpscale: a video latent comes out doubled, the network's own output",
              video.shape == (1, 16, 2, 12, 16) and torch.allclose(video, video_ref, rtol=0, atol=1e-5),
              f"{video.shape} {video.dtype} {video_ref.dtype} max|diff| {(video - video_ref).abs().max()}")
        check("LatentUpscale: an image latent (4D) comes out doubled, the network on its single frame",
              image.shape == (1, 16, 12, 16) and torch.allclose(image, image_ref, rtol=0, atol=1e-5),
              f"{image.shape} {image.dtype} max|diff| {(image - image_ref).abs().max()}")
    check("LatentUpscale: the latent's other fields are kept",
          kept["video"]["batch_index"] == [0] and {k for k in kept["image"] if k != "samples"} == {k for k in kept["image_in"] if k != "samples"},
          f"{set(kept['image'])} {set(kept['image_in'])}")


def tiny_wan(name):
    """A tiny Wan 2.1 T2V (one block of 128, random weights) saved as diffusion_models/<name>, and the
    test-only nodes BCTest_WanCond (a zero Wan text conditioning, 8 tokens of 4096) and BCTest_KeepLatent
    (an output node keeping the latent it gets) registered. -> the list KeepLatent appends to; the caller
    deletes the two nodes."""
    import comfy.ops
    import folder_paths
    import torch
    from comfy.ldm.wan.model import WanModel
    from safetensors.torch import save_file

    root = tempfile.mkdtemp(prefix="bcnodes_wan_")
    os.makedirs(os.path.join(root, "diffusion_models"))
    folder_paths.add_model_folder_path("diffusion_models", os.path.join(root, "diffusion_models"))
    net = WanModel(model_type="t2v", dim=128, ffn_dim=256, num_heads=1, num_layers=1, operations=comfy.ops.disable_weight_init)
    save_file({key: torch.randn(t.shape) * 0.02 for key, t in net.state_dict().items()},
              os.path.join(root, "diffusion_models", name))
    kept = []

    class WanCond:
        """A test-only node: a zero Wan text conditioning (8 tokens of 4096)."""
        RETURN_TYPES = ("CONDITIONING",)
        FUNCTION = "make"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

        def make(self):
            return ([[torch.zeros(1, 8, 4096), {}]],)

    class KeepLatent:
        """A test-only output node: keeps the latent it gets."""
        RETURN_TYPES = ()
        FUNCTION = "keep"
        OUTPUT_NODE = True
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"samples": ("LATENT",)}}

        def keep(self, samples):
            kept.append(samples["samples"].clone())
            return {}

    nodes.NODE_CLASS_MAPPINGS["BCTest_WanCond"] = WanCond
    nodes.NODE_CLASS_MAPPINGS["BCTest_KeepLatent"] = KeepLatent
    return kept


async def sage_attention():
    """Patch Sage Attention through ComfyUI itself. No SageAttention kernel runs here (no GPU): the
    kernel is a stand-in, torch SDPA with SageAttention's call signature, in an Active whose device
    type is the one the tensors are on (the kernel path runs) or another (every call is declined).

    Core's entry points, called the way the models call them, reach the override: Qwen-Image's
    optimized_attention_masked with (B, H, N, D) and skip_reshape, with and without its text mask;
    Wan's self-attention with AttentionTensorContainers; Wan's image cross-attention with
    low_precision_attention=False; Wan's audio attention with skip_output_reshape. A call the kernel
    takes matches core's own attention, a declined one is core's own result.

    Through validate_prompt + PromptExecutor: UNETLoader (a tiny Wan 2.1 T2V, one block, fp16 on
    ComfyUI's device) -> Patch Sage Attention -> KSampler (2 steps, cfg 1: 2 attention calls per step). With the
    kernel path every call runs the stand-in and the run summary says so; with every call declined the
    summary is a warning naming the reason, and the latent is bit-identical to mode "disabled"; the
    real resolve here (no sageattention, or no CUDA under --cpu) leaves the model unpatched with a warning."""
    import importlib

    import comfy.ldm.modules.attention as attention
    import comfy.model_management
    import torch
    import torch.nn.functional as F

    node_class = nodes.NODE_CLASS_MAPPINGS["BC_PatchSageAttention"]
    lib = importlib.import_module(node_class.__module__.rsplit(".", 2)[0] + ".libs.sage_attention")

    def kernel(q, k, v, tensor_layout="HND", is_causal=False, sm_scale=None, **kwargs):
        if tensor_layout == "NHD":
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=sm_scale).to(q.dtype)
        return out.transpose(1, 2) if tensor_layout == "NHD" else out

    def stand_in(device_type):
        return lib.Active(lib.KERNELS[lib.AUTO], "stand-in", kernel, {}, "CPU", (0, 0), "test", device_type=device_type)

    # --- core's entry points -----------------------------------------------------------------------
    override = lib.SageOverride(stand_in("cpu"), None, True, False)
    patched = {"optimized_attention_override": override}
    torch.manual_seed(0)

    def close(a, b):
        return a.shape == b.shape and torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)

    q, k, v = (torch.randn(1, 4, 96, 128, dtype=torch.float16) for _ in range(3))  # Qwen: (B, H, N, D)
    text_mask = torch.zeros(1, 1, 96, dtype=torch.float16)
    text_mask[..., 80:] = -65504.0
    qwen = [attention.optimized_attention_masked(q, k, v, 4, mask, transformer_options=to, skip_reshape=True)
            for mask in (None, text_mask) for to in ({}, patched)]
    b, s, n, d = 1, 64, 2, 128
    wq, wk, wv = (torch.randn(b, s, n * d, dtype=torch.float16) for _ in range(3))  # Wan: (B, N, H*D)
    wan_self = [attention.optimized_attention(*(attention.AttentionTensorContainer(t.clone()) for t in (wq, wk, wv)),
                                              heads=n, transformer_options=to) for to in ({}, patched)]
    kimg, vimg = (torch.randn(b, 257, n * d, dtype=torch.float16) for _ in range(2))
    wan_img = [attention.optimized_attention(wq, kimg, vimg, heads=n, transformer_options=to, low_precision_attention=False)
               for to in ({}, patched)]
    aq, ak, av = (torch.randn(4, n, 16, d, dtype=torch.float16) for _ in range(3))
    wan_audio = [attention.optimized_attention(aq, ak, av, heads=n, skip_reshape=True, skip_output_reshape=True,
                                               transformer_options=to) for to in ({}, patched)]
    check("PatchSageAttention: Qwen-Image's call (HND, skip_reshape) runs the kernel and matches core's attention",
          close(qwen[0], qwen[1]), f"{qwen[0].shape} {qwen[1].shape}")
    check("PatchSageAttention: Qwen-Image's call with its text mask is core's own result",
          torch.equal(qwen[2], qwen[3]))
    check("PatchSageAttention: Wan's self-attention (AttentionTensorContainers) runs the kernel and matches core's",
          close(wan_self[0], wan_self[1]))
    check("PatchSageAttention: Wan's image cross-attention (low_precision_attention=False) is core's own result",
          torch.equal(wan_img[0], wan_img[1]))
    check("PatchSageAttention: Wan's audio attention (skip_output_reshape) runs the kernel and matches core's",
          close(wan_audio[0], wan_audio[1]))
    check("PatchSageAttention: the entry points counted 3 kernel calls, 2 declined (mask, low precision)",
          (override.stats.sage, override.stats.fallbacks) == (3, {"mask": 1, "low_precision_attention=False": 1}),
          f"{override.stats.sage} {override.stats.fallbacks}")

    # --- through the executor ----------------------------------------------------------------------
    kept = tiny_wan("tiny_sage_wan.safetensors")

    def prompt(mode):
        return {
            "1": N("UNETLoader", unet_name="tiny_sage_wan.safetensors", weight_dtype="default"),
            "2": N("BC_PatchSageAttention", model=["1", 0], mode=mode, qk_quant="auto", nan_guard=True, log_level="once"),
            "3": N("BCTest_WanCond"),
            "4": N("EmptyHunyuanLatentVideo", width=32, height=32, length=1, batch_size=1),
            "5": N("KSampler", model=["2", 0], seed=0, steps=2, cfg=1.0, sampler_name="euler", scheduler="simple",
                   positive=["3", 0], negative=["3", 0], latent_image=["4", 0], denoise=1.0),
            "6": N("BCTest_KeepLatent", samples=["5", 0]),
        }

    async def sample(mode, active=None):
        """-> (the latent or None, our console lines with their levels)."""
        lines = []

        class Ours(logging.Handler):
            def emit(self, record):
                if record.getMessage().startswith(lib.PREFIX):
                    lines.append((record.levelname, record.getMessage()[len(lib.PREFIX) + 1:]))

        handler = Ours()
        root_logger = logging.getLogger()
        level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        real_resolve = lib.resolve
        if active is not None:
            lib.resolve = lambda mode, quant, device: active
        kept.clear()
        try:
            out, _ = await run(prompt(mode), f"sage-{mode}")
        finally:
            lib.resolve = real_resolve
            root_logger.removeHandler(handler)
            root_logger.setLevel(level)
        return (kept[0] if out is not None and kept else None), lines

    device = comfy.model_management.get_torch_device().type  # where KSampler runs the model
    other = "cpu" if device == "cuda" else "cuda"
    try:
        base, base_lines = await sample("disabled")
        ran, ran_lines = await sample("auto", stand_in(device))
        declined, declined_lines = await sample("auto", stand_in(other))
        real, real_lines = await sample("auto")
    finally:
        del nodes.NODE_CLASS_MAPPINGS["BCTest_WanCond"], nodes.NODE_CLASS_MAPPINGS["BCTest_KeepLatent"]
    check("PatchSageAttention: mode disabled samples, no console line", base is not None and base_lines == [], f"{base_lines}")
    check("PatchSageAttention: kernel path: the run's start names the kernel, its end counts 4 kernel calls",
          ran_lines == [("INFO", "stand-in [] on CPU (sm_00), sageattention test, nan_guard on"),
                        ("INFO", "run summary: 4 attention calls, 4 ran stand-in, 0 ran the original attention")],
          f"{ran_lines}")
    check("PatchSageAttention: kernel path: the latent matches mode disabled's",
          ran is not None and base is not None and close(ran, base))
    check("PatchSageAttention: every call declined: the summary is a warning with the reason, and the latent is "
          "bit-identical to mode disabled's",
          declined_lines[-1:] == [("WARNING", "run summary: 4 attention calls, 0 ran stand-in, 4 ran the original "
                                              f"attention (not on {other.upper()}: 4)")]
          and declined is not None and base is not None and torch.equal(declined, base), f"{declined_lines}")
    check("PatchSageAttention: the real resolve without sageattention / CUDA leaves the model unpatched with a warning",
          len(real_lines) == 1 and real_lines[0][0] == "WARNING" and real_lines[0][1].endswith("the model keeps ComfyUI's attention.")
          and real is not None and base is not None and torch.equal(real, base), f"{real_lines}")


async def detail_boost():
    """Detail Boost through ComfyUI's own sampler: UNETLoader (a tiny Wan 2.1, a flow model) -> a test-only
    node putting a model_function_wrapper on the model (it records the sigma each model call gets and the
    run's schedule) -> Detail Boost -> KSampler (dpmpp_2m_sde, beta, 20 steps, cfg 2, denoise 0.45), through
    validate_prompt + PromptExecutor. Without the node and with weight 0 the model gets the schedule's sigmas
    and the latents are bit-identical; with weight 1 (hard, eta 0.5, steps 3 to 10) the model_function_wrapper
    still runs, the model gets the lowered sigma at steps 3 to 9 only, and the latent differs."""
    import importlib

    import torch

    node_class = nodes.NODE_CLASS_MAPPINGS["BC_DetailBoost"]
    lib = importlib.import_module(node_class.__module__.rsplit(".", 2)[0] + ".libs.detail_boost")
    kept = tiny_wan("tiny_boost_wan.safetensors")
    seen = []  # (sigma the model got, the run's schedule) per model call

    class RecordSigma:
        """A test-only node: a model_function_wrapper recording each model call's sigma and schedule."""
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "patch"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model": ("MODEL",)}}

        def patch(self, model):
            def record(apply_model, args):
                seen.append((float(args["timestep"].max()), args["c"]["transformer_options"]["sample_sigmas"].tolist()))
                return apply_model(args["input"], args["timestep"], **args["c"])

            patched = model.clone()
            patched.set_model_unet_function_wrapper(record)
            return (patched,)

    nodes.NODE_CLASS_MAPPINGS["BCTest_RecordSigma"] = RecordSigma

    def prompt(weight):
        boost = {} if weight is None else {
            "3": N("BC_DetailBoost", model=["2", 0], weight=weight, mode="hard", eta=0.5, start_step=3, end_step=10)}
        return {
            "1": N("UNETLoader", unet_name="tiny_boost_wan.safetensors", weight_dtype="default"),
            "2": N("BCTest_RecordSigma", model=["1", 0]),
            **boost,
            "4": N("BCTest_WanCond"),
            "5": N("EmptyHunyuanLatentVideo", width=32, height=32, length=1, batch_size=1),
            "6": N("KSampler", model=["3" if boost else "2", 0], seed=0, steps=20, cfg=2.0, sampler_name="dpmpp_2m_sde",
                   scheduler="beta", positive=["4", 0], negative=["4", 0], latent_image=["5", 0], denoise=0.45),
            "7": N("BCTest_KeepLatent", samples=["6", 0]),
        }

    async def sample(weight):
        """-> (the latent or None, [(sigma, schedule)] per model call)."""
        kept.clear()
        seen.clear()
        out, _ = await run(prompt(weight), f"detail-boost-{weight}")
        return (kept[0] if out is not None and kept else None), list(seen)

    try:
        base, base_calls = await sample(None)
        off, off_calls = await sample(0.0)
        on, on_calls = await sample(1.0)
    finally:
        for key in ("BCTest_WanCond", "BCTest_KeepLatent", "BCTest_RecordSigma"):
            del nodes.NODE_CLASS_MAPPINGS[key]

    schedule = base_calls[0][1] if base_calls else []
    per_step = len(base_calls) // 20 or 1  # cond and uncond in one batch, or one call each
    check("DetailBoost: the run is 20 steps over a 21-point schedule, every call has it",
          len(schedule) == 21 and len(base_calls) in (20, 40)
          and all(s == schedule for calls in (base_calls, off_calls, on_calls) for _, s in calls),
          f"{len(schedule)} {len(base_calls)}")
    got = [sigma for sigma, _ in base_calls[::per_step]]
    check("DetailBoost: without the node the model gets the schedule's sigmas",
          len(got) == 20 and all(math.isclose(g, s, rel_tol=1e-6) for g, s in zip(got, schedule)), f"{got}")
    check("DetailBoost: weight 0: the same sigmas and a bit-identical latent",
          off_calls == base_calls and off is not None and base is not None and torch.equal(off, base))
    expected = [lib.lowered_sigma("hard", s, schedule[i - 1], 0.5, True, 1.0) if 3 <= i < 10 else s
                for i, s in enumerate(schedule[:20])]
    got = [sigma for sigma, _ in on_calls[::per_step]]
    check("DetailBoost: weight 1: the model_function_wrapper still runs, the lowered sigma at steps 3 to 9 only",
          len(on_calls) == len(base_calls) and len(got) == 20
          and all(math.isclose(g, e, rel_tol=1e-5) for g, e in zip(got, expected))
          and all(g < s for g, s in zip(got[3:10], schedule[3:10])), f"{got} {expected}")
    check("DetailBoost: weight 1: the latent differs", on is not None and base is not None and not torch.equal(on, base))


async def qwen_image_shift():
    """Qwen Image Shift inside ComfyUI, a tiny Wan 2.1 (a flow model) standing in for Qwen-Image.
    fixed, shift_terminal off, against core's own ModelSamplingAuraFlow at the same shift: every scheduler
    at 1 to 50 steps and KSampler's sigmas (denoise below 1, a sampler that drops the penultimate sigma)
    are equal tensors, percent_to_sigma agrees, no wrapper; one KSampler run each through validate_prompt +
    PromptExecutor gets the same schedule and a bit-identical latent, and at shift 3 that schedule is
    Qwen-Image-Lightning's diffusers schedule. dynamic (the defaults: the official config) on one patched
    model in two KSamplers (512x384 at 8 steps, then LatentUpscaleBy 1.25 at 4 steps, denoise 0.5):
    each run's schedule is diffusers' for its size (pass 2: the tail of its 8-step schedule), the model is
    called at it, model_sampling has the run's shift during the run and 1 after, one console line per run.
    The references are diffusers' own (tests/_diffusers_sigmas.py). Refine passes, refine_threshold 500 so the
    512x384 latent (768 tokens) is above it: a KSampler at denoise 1 keeps diffusers' schedule, the next one at
    denoise 0.5 (new noise) has mu at 500 + 0.17 x (768 - 500) tokens; a KSamplerAdvanced split at step 4 of 8
    (add_noise off in the later part) keeps the latent's own shift in both parts."""
    import comfy.samplers
    import torch
    from comfy.patcher_extension import WrappersMP

    from _diffusers_sigmas import ATOL, DIFFUSERS

    shift_node = nodes.NODE_CLASS_MAPPINGS["BC_QwenImageShift"]()
    aura_node = nodes.NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
    kept = tiny_wan("tiny_shift_wan.safetensors")
    seen = []  # (sigma the model got, the run's schedule, model_sampling's shift) per model call
    models = []  # the model each RecordRun got

    def patch(model, mode, shift, shift_terminal):
        return shift_node.patch(model, mode, shift, 1.0, shift_terminal, 0.02, 256, 8192, 0.5, 0.9)[0]

    def same(a, b):
        return a.shape == b.shape and torch.allclose(a, b, rtol=0, atol=0, equal_nan=True)

    def close(got, expected):
        return len(got) == len(expected) and all(abs(g - e) <= ATOL for g, e in zip(got, expected))

    # fixed, shift_terminal off == ModelSamplingAuraFlow, on the sigma path alone
    model = nodes.UNETLoader().load_unet("tiny_shift_wan.safetensors", "default")[0]
    mismatched, counted = [], 0
    for shift in (1.73, 2.3, 3.0):
        ours, aura = patch(model, "fixed", shift, False), aura_node.patch_aura(model, shift)[0]
        ms_ours, ms_aura = ours.get_model_object("model_sampling"), aura.get_model_object("model_sampling")
        if ours.get_wrappers(WrappersMP.OUTER_SAMPLE, "bcnodes_qwen_image_shift"):
            mismatched.append((shift, "wrapper"))
        if type(ms_ours).__mro__[1:] != type(ms_aura).__mro__[1:] or ms_ours.noise_scale != ms_aura.noise_scale:
            mismatched.append((shift, "class"))
        for percent in (0.0, 0.2, 0.5, 0.9, 1.0):
            if ms_ours.percent_to_sigma(percent) != ms_aura.percent_to_sigma(percent):
                mismatched.append((shift, "percent", percent))
        for name in comfy.samplers.SCHEDULER_NAMES:
            for steps in (1, 3, 8, 20, 30, 50):
                counted += 1
                # exact; NaN equal to NaN: core's kl_optimal at 1 step divides 0 by 0 on both
                if not same(comfy.samplers.calculate_sigmas(ms_ours, name, steps),
                            comfy.samplers.calculate_sigmas(ms_aura, name, steps)):
                    mismatched.append((shift, name, steps))
            for sampler in ("euler", "dpm_2"):
                for denoise in (1.0, 0.45):
                    counted += 1
                    a = comfy.samplers.KSampler(ours, 20, "cpu", sampler=sampler, scheduler=name, denoise=denoise).sigmas
                    b = comfy.samplers.KSampler(aura, 20, "cpu", sampler=sampler, scheduler=name, denoise=denoise).sigmas
                    if not same(a, b):
                        mismatched.append((shift, name, sampler, denoise))
    check(f"QwenImageShift: fixed, shift_terminal off == ModelSamplingAuraFlow ({counted} schedules over "
          f"{len(comfy.samplers.SCHEDULER_NAMES)} schedulers, 3 shifts): equal tensors, percent_to_sigma, no wrapper",
          not mismatched, f"{mismatched[:8]}")

    class RecordRun:
        """A test-only node: a model_function_wrapper recording each model call's sigma, the run's schedule and
        the shift of the model_sampling the model runs with; keeps the model it got."""
        RETURN_TYPES = ("MODEL",)
        FUNCTION = "patch"
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"model": ("MODEL",)}}

        def patch(self, model):
            def record(apply_model, args):
                seen.append((float(args["timestep"].max()), args["c"]["transformer_options"]["sample_sigmas"].tolist(),
                             apply_model.__self__.model_sampling.shift))
                return apply_model(args["input"], args["timestep"], **args["c"])

            models.append(model)
            patched = model.clone()
            patched.set_model_unet_function_wrapper(record)
            return (patched,)

    nodes.NODE_CLASS_MAPPINGS["BCTest_RecordRun"] = RecordRun

    def one_pass(shift_node_inputs):
        return {
            "1": N("UNETLoader", unet_name="tiny_shift_wan.safetensors", weight_dtype="default"),
            "2": shift_node_inputs,
            "3": N("BCTest_RecordRun", model=["2", 0]),
            "4": N("BCTest_WanCond"),
            "5": N("EmptyHunyuanLatentVideo", width=512, height=384, length=1, batch_size=1),
            "6": N("KSampler", model=["3", 0], seed=0, steps=8, cfg=1.0, sampler_name="euler", scheduler="simple",
                   positive=["4", 0], negative=["4", 0], latent_image=["5", 0], denoise=1.0),
            "7": N("BCTest_KeepLatent", samples=["6", 0]),
        }

    async def sample(prompt, label):
        """-> (the latents kept, [(sigma, schedule, shift)] per model call, console lines)."""
        kept.clear()
        seen.clear()
        models.clear()
        lines, root_logger = Lines(), logging.getLogger()
        level = root_logger.level
        root_logger.addHandler(lines)
        root_logger.setLevel(logging.INFO)
        try:
            out, _ = await run(prompt, label)
        finally:
            root_logger.removeHandler(lines)
            root_logger.setLevel(level)
        return (list(kept) if out is not None else []), list(seen), [line for line in lines.lines if "Qwen Image Shift" in line]

    try:
        ours_latents, ours_calls, _ = await sample(one_pass(N(
            "BC_QwenImageShift", model=["1", 0], mode="fixed", shift=3.0, multiplier=1.0, shift_terminal=False, terminal=0.02,
            base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9)), "qwen-shift-fixed")
        aura_latents, aura_calls, _ = await sample(one_pass(N("ModelSamplingAuraFlow", model=["1", 0], shift=3.0)), "qwen-shift-aura")

        two_pass = one_pass(N("BC_QwenImageShift", model=["1", 0], mode="dynamic", shift=3.0, multiplier=1.0, shift_terminal=True,
                              terminal=0.02, base_seq_len=256, max_seq_len=8192, base_shift=0.5, max_shift=0.9))
        two_pass["8"] = N("LatentUpscaleBy", samples=["6", 0], upscale_method="nearest-exact", scale_by=1.25)
        two_pass["9"] = N("KSampler", model=["3", 0], seed=1, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                          positive=["4", 0], negative=["4", 0], latent_image=["8", 0], denoise=0.5)
        two_pass["10"] = N("BCTest_KeepLatent", samples=["9", 0])
        dyn_latents, dyn_calls, dyn_lines = await sample(two_pass, "qwen-shift-dynamic")
        dyn_models = list(models)

        def refine_shift(shift_terminal):
            return N("BC_QwenImageShift", model=["1", 0], mode="dynamic", shift=3.0, multiplier=1.0,
                     shift_terminal=shift_terminal, terminal=0.02, base_seq_len=256, max_seq_len=8192, base_shift=0.5,
                     max_shift=0.9, refine_threshold=500, refine_rate=0.17)

        refine = one_pass(refine_shift(True))
        refine["9"] = N("KSampler", model=["3", 0], seed=1, steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                        positive=["4", 0], negative=["4", 0], latent_image=["6", 0], denoise=0.5)
        refine["10"] = N("BCTest_KeepLatent", samples=["9", 0])
        _, ref_calls, ref_lines = await sample(refine, "qwen-shift-refine")

        def advanced(latent, add_noise, start, end, leftover):
            return N("KSamplerAdvanced", model=["3", 0], add_noise=add_noise, noise_seed=0, steps=8, cfg=1.0,
                     sampler_name="euler", scheduler="simple", positive=["4", 0], negative=["4", 0], latent_image=latent,
                     start_at_step=start, end_at_step=end, return_with_leftover_noise=leftover)

        split = one_pass(refine_shift(False))
        split["6"] = advanced(["5", 0], "enable", 0, 4, "enable")
        split["9"] = advanced(["6", 0], "disable", 4, 10000, "disable")
        split["10"] = N("BCTest_KeepLatent", samples=["9", 0])
        _, split_calls, split_lines = await sample(split, "qwen-shift-split")
    finally:
        for key in ("BCTest_WanCond", "BCTest_KeepLatent", "BCTest_RecordRun"):
            del nodes.NODE_CLASS_MAPPINGS[key]

    lightning = DIFFUSERS["lightning_1328x1328_8"][5]
    schedule = ours_calls[0][1] if ours_calls else []
    check("QwenImageShift: fixed 3, shift_terminal off: one KSampler run has ModelSamplingAuraFlow's schedule and "
          "model calls, and a bit-identical latent",
          len(ours_calls) == 8 and ours_calls == aura_calls and len(ours_latents) == 1 and len(aura_latents) == 1
          and torch.equal(ours_latents[0], aura_latents[0]), f"{ours_calls} {aura_calls}")
    check("QwenImageShift: fixed 3, shift_terminal off is Qwen-Image-Lightning's diffusers schedule (simple, 8 steps)",
          close(schedule, lightning), f"{schedule} {lightning}")

    _, _, _, seq1, mu1, ref1 = DIFFUSERS["512x384_8_terminal"]
    _, _, _, seq2, mu2, ref2 = DIFFUSERS["640x480_8_terminal"]
    first, second = dyn_calls[:8], dyn_calls[8:]
    check("QwenImageShift: dynamic, two passes on one model: 8 + 4 model calls, both latents kept",
          len(first) == 8 and len(second) == 4 and sorted(tuple(x.shape[-2:]) for x in dyn_latents) == [(48, 64), (60, 80)],
          f"{len(dyn_calls)} {[tuple(x.shape) for x in dyn_latents]}")
    check("QwenImageShift: dynamic, pass 1 (512x384, 768 tokens) has diffusers' schedule and the model is called at it",
          len(first) == 8 and all(close(s, ref1) for _, s, _ in first)
          and all(math.isclose(sigma, ref1[i], abs_tol=ATOL) for i, (sigma, _, _) in enumerate(first)),
          f"{first and first[0][1]} {ref1}")
    check("QwenImageShift: dynamic, pass 2 (1.25x, 1200 tokens, denoise 0.5) has the tail of diffusers' 8-step schedule",
          len(second) == 4 and all(close(s, ref2[-5:]) for _, s, _ in second)
          and all(math.isclose(sigma, ref2[-5 + i], abs_tol=ATOL) for i, (sigma, _, _) in enumerate(second)),
          f"{second and second[0][1]} {ref2[-5:]}")
    check("QwenImageShift: dynamic: model_sampling has each run's shift (e^mu) during it and 1 after",
          len(dyn_calls) == 12 and all(math.isclose(s, math.exp(mu1), rel_tol=1e-12) for _, _, s in first)
          and all(math.isclose(s, math.exp(mu2), rel_tol=1e-12) for _, _, s in second)
          and len(dyn_models) == 1 and dyn_models[0].get_model_object("model_sampling").shift == 1.0,
          f"{[s for _, _, s in dyn_calls]} {math.exp(mu1)} {math.exp(mu2)}")
    check("QwenImageShift: dynamic: one console line per run with its size, refine state, tokens, mu and shift",
          dyn_lines == [f"[BCNodes] Qwen Image Shift: latent 64x48, refine: no, tokens {seq1}, mu {mu1:.4f}, "
                        f"shift {math.exp(mu1):.4f}",
                        f"[BCNodes] Qwen Image Shift: latent 80x60, refine: yes, tokens {seq2} <= threshold 9600, "
                        f"mu {mu2:.4f}, shift {math.exp(mu2):.4f}"],
          f"{dyn_lines}")

    def line_mu(tokens):
        return 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)

    def shifted(shift, t, terminal):
        """The flow shift of the shift-1 grid t (ending in 0), then the stretch to `terminal` (or None)."""
        out = [shift * x / (1 + (shift - 1) * x) for x in t]
        if terminal is None:
            return out
        factor = (1 - out[-2]) / (1 - terminal)
        return [1 - (1 - x) / factor for x in out[:-1]] + [0.0]

    t8 = [1 - i / 8 for i in range(8)] + [0.0]
    mu_r = line_mu(500 + 0.17 * (seq1 - 500))
    refine_ref = shifted(math.exp(mu_r), t8[-5:], 0.02)
    first, second = ref_calls[:8], ref_calls[8:]
    check("QwenImageShift: refine_threshold 500: denoise 1 (768 tokens) keeps diffusers' schedule and its own shift",
          len(first) == 8 and all(close(s, ref1) and math.isclose(shift, math.exp(mu1), rel_tol=1e-12)
                                  for _, s, shift in first), f"{first and first[0]} {ref1}")
    check("QwenImageShift: refine_threshold 500: denoise 0.5 on new noise is a refine pass, mu at 500 + 0.17 x "
          "(768 - 500) tokens, the model called at its schedule",
          len(second) == 4 and all(close(s, refine_ref) and math.isclose(shift, math.exp(mu_r), rel_tol=1e-12)
                                   for _, s, shift in second)
          and all(math.isclose(sigma, refine_ref[i], abs_tol=ATOL) for i, (sigma, _, _) in enumerate(second)),
          f"{second and second[0]} {refine_ref} {math.exp(mu_r)}")
    check("QwenImageShift: refine pass console lines",
          ref_lines == [f"[BCNodes] Qwen Image Shift: latent 64x48, refine: no, tokens {seq1}, mu {mu1:.4f}, "
                        f"shift {math.exp(mu1):.4f}",
                        f"[BCNodes] Qwen Image Shift: latent 64x48, refine: yes, tokens {seq1}, threshold 500, rate 0.17, "
                        f"mu {mu_r:.4f}, shift {math.exp(mu_r):.4f}"], f"{ref_lines}")
    own = shifted(math.exp(mu1), t8, None)
    head, tail = split_calls[:4], split_calls[4:]
    check("QwenImageShift: KSamplerAdvanced split at step 4 of 8 (add_noise off later): no refine pass, both parts "
          "at the latent's own shift and schedule",
          len(head) == 4 and len(tail) == 4
          and all(close(s, own[:5]) and math.isclose(shift, math.exp(mu1), rel_tol=1e-12) for _, s, shift in head)
          and all(close(s, own[4:]) and math.isclose(shift, math.exp(mu1), rel_tol=1e-12) for _, s, shift in tail)
          and split_lines == [f"[BCNodes] Qwen Image Shift: latent 64x48, refine: no, tokens {seq1}, mu {mu1:.4f}, "
                              f"shift {math.exp(mu1):.4f}"] * 2,
          f"{head and head[0]} {tail and tail[0]} {own} {split_lines}")


def tiny_zimage(kind="ZImage"):
    """A tiny Z-Image on the CPU: core's ZImage config (or another of core's configs of the same model: `kind`) and
    Lumina 2 NextDiT with Z-Image's RoPE (axes 32 / 48 / 48 of one 128-wide head, theta 256, axes_lens 1536 / 512 /
    512) and modulation, dim 256, one layer of each kind, seeded random weights; a ModelPatcher."""
    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models
    import torch

    config = getattr(comfy.supported_models, kind)({
        "image_model": "lumina2", "patch_size": 2, "in_channels": 16, "dim": 256, "n_layers": 1, "n_refiner_layers": 1,
        "n_heads": 2, "n_kv_heads": 2, "multiple_of": 32, "ffn_dim_multiplier": 8.0 / 3.0, "norm_eps": 1e-5, "qk_norm": True,
        "cap_feat_dim": 32, "axes_dims": [32, 48, 48], "axes_lens": [1536, 512, 512], "rope_theta": 256.0,
        "z_image_modulation": True, "time_scale": 1000.0, "pad_tokens_multiple": 32})
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.Lumina2(config, device="cpu")
    generator = torch.Generator().manual_seed(0)
    model.diffusion_model.load_state_dict({key: torch.randn(t.shape, generator=generator) * 0.05
                                           for key, t in model.diffusion_model.state_dict().items()})
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


QWEN_UNET = {"image_model": "qwen_image", "in_channels": 64, "out_channels": 16, "num_layers": 1, "attention_head_dim": 128,
             "num_attention_heads": 1, "joint_attention_dim": 32}


def seeded(module, seed=0):
    """`module` with seeded random weights (x 0.05), in float32 on the CPU."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    module.load_state_dict({key: torch.randn(t.shape, generator=generator) * 0.05 for key, t in module.state_dict().items()})
    return module


def tiny_qwen_image():
    """A tiny Qwen-Image on the CPU: core's QwenImage config and QwenImageTransformer2DModel with Qwen-Image's RoPE (axes
    16 / 56 / 56 of one 128-wide head, theta 10000), one block, captions of 32 features, seeded random weights; a
    ModelPatcher. Its latent is the Qwen VAE's: 16 channels with a frame axis."""
    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models
    import torch

    config = comfy.supported_models.QwenImage(dict(QWEN_UNET))
    config.set_inference_dtype(torch.float32, None)
    model = comfy.model_base.QwenImage(config, device="cpu")
    seeded(model.diffusion_model)
    return comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


class StandInVAE:
    """A stand-in VAE at 8 px per cell: decodes 3 latent channels through a sigmoid, encodes their inverse into 16
    channels. frames: the latent has a frame axis, (B, C, 1, h, w), and the decode one too, (B, 1, H, W, 3), as core's
    Wan 2.1 VAE gives Qwen-Image."""
    output_channels = 3

    def __init__(self, frames):
        self.frames = frames

    def decode(self, samples):
        import torch

        return torch.sigmoid(samples[:, :3]).repeat_interleave(8, -2).repeat_interleave(8, -1).movedim(1, -1).contiguous()

    def encode(self, pixels):
        import torch
        import torch.nn.functional as F

        x = torch.logit(F.avg_pool2d(pixels[..., :3].movedim(-1, 1), 8).clamp(1e-3, 1 - 1e-3))
        x = x.repeat(1, 6, 1, 1)[:, :16].contiguous()
        return x.unsqueeze(2) if self.frames else x

    def spacial_compression_encode(self):
        return 8

    def spacial_compression_decode(self):
        return 8

    def vae_output_dtype(self):
        import torch

        return torch.float32


def maker(name, rtype, value):
    """A test-only node class giving `value` as its one `rtype` output."""
    return type(name, (), {"RETURN_TYPES": (rtype,), "FUNCTION": "make", "CATEGORY": "testing",
                           "INPUT_TYPES": classmethod(lambda cls: {"required": {}}), "make": lambda self: (value,)})


TEST_NODES = ("BCTest_DModel", "BCTest_DCond", "BCTest_DVAE", "BCTest_DKeep", "BCTest_DControl", "BCTest_DPatch", "BCTest_DImage")


def dype_test_nodes(patcher, caption, vae, kept, control=None, patch=None, image=None):
    """Registers the test-only nodes of the DyPE sections: the model, a caption conditioning, the VAE, a control net, a
    model patch, an image, and BCTest_DKeep (an output node keeping a LATENT's samples or an IMAGE in `kept` under its
    tag). -> the base prompt (model "1", conditioning "2", VAE "3")."""
    class Keep:
        RETURN_TYPES = ()
        FUNCTION = "keep"
        OUTPUT_NODE = True
        CATEGORY = "testing"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"tag": ("STRING", {"default": ""})}, "optional": {"latent": ("LATENT",), "image": ("IMAGE",)}}

        def keep(self, tag, latent=None, image=None):
            kept[tag] = (latent["samples"] if latent is not None else image).clone()
            return {}

    nodes.NODE_CLASS_MAPPINGS.update({
        "BCTest_DModel": maker("Model", "MODEL", patcher), "BCTest_DCond": maker("Cond", "CONDITIONING", [[caption, {}]]),
        "BCTest_DVAE": maker("VAE", "VAE", vae), "BCTest_DKeep": Keep, "BCTest_DControl": maker("Control", "CONTROL_NET", control),
        "BCTest_DPatch": maker("Patch", "MODEL_PATCH", patch), "BCTest_DImage": maker("Image", "IMAGE", image)})
    return {"1": N("BCTest_DModel"), "2": N("BCTest_DCond"), "3": N("BCTest_DVAE")}


def drop_test_nodes():
    for key in TEST_NODES:
        nodes.NODE_CLASS_MAPPINGS.pop(key, None)


def ksampler_node(model, latent, seed, denoise, positive=("2", 0), negative=("2", 0)):
    return N("KSampler", model=model, seed=seed, steps=2, cfg=1.0, sampler_name="dpmpp_sde", scheduler="beta",
             positive=list(positive), negative=list(negative), latent_image=latent, denoise=denoise)


def dual_sampler_node(blend, dype="auto", positive=("2", 0), negative=("2", 0), **kw):
    return N("BC_DualSampler", model_1=["1", 0], model_2=["1", 0], positive=list(positive), negative=list(negative),
             latent_image=["4", 0], vae=["3", 0], seed=3, steps_1=2, cfg_1=1.0, sampler_1="dpmpp_sde", scheduler_1="beta",
             steps_2=2, cfg_2=1.0, sampler_2="dpmpp_sde", scheduler_2="beta", denoise_2=0.45, upscale_method="bislerp",
             scale_by=2.0, blend_factor=blend, pixel_upscale_method="lanczos", second_seed="increment", dype=dype, **kw)


async def dype_models():
    """The DyPE sections: each model DyPE takes (dype_model), a model it does not (dype_unsupported), the
    ControlNets (dype_controlnets) and Dual Sampler's start sigma under a per-run shift (dype_run_start)."""
    await dype_model("Z-Image", tiny_zimage(), StandInVAE(frames=False), "rope_embedder")
    await dype_model("Qwen-Image", tiny_qwen_image(), StandInVAE(frames=True), "pe_embedder")
    await dype_unsupported()
    await dype_controlnets()
    await dype_run_start()


async def dype_run_start():
    """Dual Sampler's DyPE start sigma under Qwen Image Shift, on tiny Qwen-Image: a 1024 x 1536 px first pass, the
    second at 2048 x 3072 px (dype auto). The second pass is stopped at its first model call (the model is not run at
    that size); there the start sigma DyPE uses must equal the sigma that call is given. dynamic (shift_terminal on):
    the run's own first sigma (the second pass is a refine pass: the shift at refine_rate 0.17 from refine_threshold's
    mu toward the second pass's latent's, then the terminal stretch), which core's KSampler
    schedule outside the run (shift 1, no stretch) does not give: the start sigma was read from that schedule before.
    fixed, shift 3, shift_terminal off: the same sigma as that schedule's first, as before. fixed with shift_terminal on:
    the run's stretched first sigma (that schedule does not give it either)."""
    import comfy.samplers
    import torch

    class Stop(Exception):
        pass

    qwen = tiny_qwen_image()
    seen = {}

    def stop_at_the_second_pass(module, args, kwargs):
        if args[0].shape[-1] * 8 == 2048:
            seen["call"] = float(args[1].flatten()[0])          # the call's timestep: its sigma (Qwen's multiplier 1)
            seen["start"] = module.pe_embedder.wrapper.step[2].start_sigma  # the DyPE window's Settings for the step
            raise Stop("stopped at the second pass's first model call")

    hook = qwen.model.diffusion_model.register_forward_pre_hook(stop_at_the_second_pass, with_kwargs=True)
    shift = nodes.NODE_CLASS_MAPPINGS["BC_QwenImageShift"]()
    try:
        for label, mode, terminal in (("dynamic", "dynamic", True), ("fixed 3, no terminal", "fixed", False),
                                      ("fixed 3, terminal", "fixed", True)):
            shifted = shift.patch(qwen, mode, 3.0, 1.0, terminal, 0.02, 256, 8192, 0.5, 0.9)[0]
            grid = comfy.samplers.KSampler(shifted, steps=2, device="cpu", sampler="dpmpp_sde", scheduler="beta",
                                           denoise=0.45).sigmas.tolist()
            before = grid[0]  # how the start sigma was read before
            seen.clear()
            kept = {}
            base = dype_test_nodes(shifted, torch.zeros(1, 8, 32), StandInVAE(frames=True), kept)
            try:
                drop_dead_models()
                await run({**base, "4": N("EmptySD3LatentImage", width=1024, height=1536, batch_size=1),
                           "5": dual_sampler_node(0.0, "auto"), "k": N("BCTest_DKeep", tag="latent", latent=["5", 2])},
                          f"dype-run-start-{mode}-{terminal}")
            finally:
                drop_test_nodes()
            ok = "call" in seen and seen["start"] == seen["call"]
            if mode == "dynamic":
                # a refine pass (denoise 0.45 on new noise) of 24576 tokens: refine_rate 0.17 from 9600 tokens' mu
                firsts = {}
                for rate in (0.17, 1.0):
                    s = math.exp(0.5 + 0.4 * (9600 + rate * (24576 - 9600) - 256) / 7936)
                    t = [s * x / (1 + (s - 1) * x) for x in grid]
                    firsts[rate] = 1 - (1 - t[0]) / ((1 - t[-2]) / (1 - 0.02))
                check(f"Dual Sampler under Qwen Image Shift {label}: the start sigma at 2048 x 3072 is the second pass's "
                      "first model call's, the refine pass's shift (refine_rate 0.17, not the latent's own), not the "
                      "schedule core's KSampler gives outside the run",
                      ok and seen["call"] != before and abs(seen["call"] - firsts[0.17]) < 1e-6
                      and abs(seen["call"] - firsts[1.0]) > 1e-3, f"{seen} {firsts} schedule outside the run {before}")
            elif terminal:
                check(f"Dual Sampler under Qwen Image Shift {label}: the start sigma at 2048 x 3072 is the second pass's "
                      "first model call's, not the schedule core's KSampler gives outside the run",
                      ok and seen["call"] != before, f"{seen} schedule outside the run {before}")
            else:
                check(f"Dual Sampler under Qwen Image Shift {label}: the start sigma at 2048 x 3072 is the second pass's "
                      "first model call's, the same as before", ok and seen["call"] == before, f"{seen} before {before}")
    finally:
        hook.remove()


async def dype_model(name, patcher, vae, attribute):
    """DyPE Patcher and Dual Sampler inside ComfyUI on a tiny `name` (CPU), through validate_prompt + PromptExecutor,
    with test-only nodes giving the model, a caption conditioning and a stand-in VAE (a fixed decode / encode at 8 px
    per cell; Qwen-Image's latent with its frame axis):
    - DyPE Patcher in auto mode on a 1920 px latent (the model's auto_threshold): the latent bit-identical to the
      unpatched model's;
    - enable on a 2048 px latent: a different latent, the model's own rotary embedder (core's EmbedND) on it after the
      run, and the unpatched model's run after it bit-identical to the one before (nothing left on the model);
    - Dual Sampler (1024 x 64 px, x2: DyPE on in the second pass) against the core chain KSampler -> Upscale Latent By
      -> KSampler(DyPE Patcher at the second pass's first sigma) -> VAE Decode x2: bit-identical images and latent at
      blend_factor 0; at 1 against VAE Decode -> ImageScale lanczos -> VAE Encode; at 0.5 against core's Latent Blend
      of the two; with dype disable another second latent (DyPE was on)."""
    import comfy.samplers
    import torch

    caption = torch.randn(1, 8, 32, generator=torch.Generator().manual_seed(1))
    kept = {}
    base = dype_test_nodes(patcher, caption, vae, kept)

    async def one_pass(width, mode=None):
        kept.clear()
        patch = {} if mode is None else {"5": N("BC_DyPEPatcher", model=["1", 0], mode=mode, dype_start_sigma=1.0)}
        out, _ = await run({**base, "4": N("EmptySD3LatentImage", width=width, height=64, batch_size=1), **patch,
                            "6": ksampler_node(["5" if patch else "1", 0], ["4", 0], 3, 1.0),
                            "7": N("BCTest_DKeep", tag="latent", latent=["6", 0])}, f"dype-{name}-{width}-{mode}")
        return kept.get("latent") if out is not None else None

    try:
        plain, auto = await one_pass(1920), await one_pass(1920, "auto")
        check(f"DyPE Patcher {name}: auto on a 1920 px latent leaves the model's output bit-identical",
              plain is not None and auto is not None and torch.equal(plain, auto))
        before, dyped = await one_pass(2048), await one_pass(2048, "enable")
        embedder = type(getattr(patcher.model.diffusion_model, attribute)).__name__
        after = await one_pass(2048)
        check(f"DyPE Patcher {name}: on a 2048 px latent the output changes, and the model's own embedder is back after "
              "the run", before is not None and dyped is not None and not torch.equal(before, dyped) and embedder == "EmbedND",
              embedder)
        check(f"DyPE Patcher {name}: the unpatched model's next run is bit-identical to its run before DyPE (nothing left "
              "on it)", after is not None and torch.equal(before, after))

        start = float(comfy.samplers.KSampler(patcher, steps=2, device="cpu", sampler="dpmpp_sde", scheduler="beta", denoise=0.45).sigmas[0])

        def chain(blend):
            prompt = {**base, "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1),
                      "5": ksampler_node(["1", 0], ["4", 0], 3, 1.0),
                      "6": N("LatentUpscaleBy", samples=["5", 0], upscale_method="bislerp", scale_by=2.0),
                      "9": N("VAEDecode", samples=["5", 0], vae=["3", 0]),
                      "11": N("ImageScale", image=["9", 0], upscale_method="lanczos", width=2048, height=128, crop="disabled"),
                      "12": N("VAEEncode", pixels=["11", 0], vae=["3", 0]),
                      "13": N("LatentBlend", samples1=["12", 0], samples2=["6", 0], blend_factor=blend),
                      "7": N("BC_DyPEPatcher", model=["1", 0], mode="auto", dype_start_sigma=start),
                      "8": ksampler_node(["7", 0], [{0.0: "6", 1.0: "12"}.get(blend, "13"), 0], 4, 0.45),
                      "10": N("VAEDecode", samples=["8", 0], vae=["3", 0]),
                      "k1": N("BCTest_DKeep", tag="first", image=["9", 0]), "k2": N("BCTest_DKeep", tag="second", image=["10", 0]),
                      "k3": N("BCTest_DKeep", tag="latent", latent=["8", 0])}
            if blend == 0.0:
                for unused in ("11", "12", "13"):
                    del prompt[unused]
            elif blend == 1.0:
                del prompt["6"], prompt["13"]
            return prompt

        def dual(blend, dype="auto"):
            return {**base, "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1), "5": dual_sampler_node(blend, dype),
                    "k1": N("BCTest_DKeep", tag="first", image=["5", 0]), "k2": N("BCTest_DKeep", tag="second", image=["5", 1]),
                    "k3": N("BCTest_DKeep", tag="latent", latent=["5", 2])}

        got = {}
        for blend in (0.0, 1.0, 0.5):
            kept.clear()
            out_a, _ = await run(chain(blend), f"dype-{name}-chain-{blend}")
            want = dict(kept)
            kept.clear()
            out_b, _ = await run(dual(blend), f"dype-{name}-dual-{blend}")
            got = dict(kept)
            same = out_a is not None and out_b is not None and set(want) == set(got) == {"first", "second", "latent"} and all(
                torch.equal(want[k], got[k]) for k in want)
            check(f"Dual Sampler {name}: blend_factor {blend} bit-identical to the core chain (first and second image, "
                  "latent)", same, f"{ {k: (tuple(want[k].shape), tuple(got[k].shape)) for k in want if k in got} }")
        latent = got.get("latent")
        check(f"Dual Sampler {name}: the second latent is 2048 x 128 px (x2, a multiple of 16), in the model's layout",
              latent is not None and tuple(latent.shape[-2:]) == (16, 256) and latent.ndim == (5 if vae.frames else 4),
              f"{None if latent is None else tuple(latent.shape)}")
        kept.clear()
        out, _ = await run(dual(0.5, "disable"), f"dype-{name}-dual-off")
        check(f"Dual Sampler {name}: DyPE was on in the second pass (dype disable gives another second latent)",
              out is not None and "latent" in kept and latent is not None and not torch.equal(kept["latent"], latent))
    finally:
        drop_test_nodes()


async def dype_unsupported():
    """Dual Sampler and DyPE Patcher on a model DyPE does not take (core's Lumina 2 config on the tiny NextDiT): any mode
    but disable stops with the error naming the models DyPE takes, Dual Sampler before its first pass (the model never
    runs); disable runs as the model is. And a DyPE Config set for another model: the error names both."""
    import torch

    patcher = tiny_zimage("Lumina2")
    calls = []
    hook = patcher.model.diffusion_model.register_forward_pre_hook(lambda module, args: calls.append(1))
    kept = {}
    base = dype_test_nodes(patcher, torch.zeros(1, 8, 32), StandInVAE(frames=False), kept)

    async def errors(prompt, label):
        server = Server()
        out, _ = await run(prompt, label, extra_data={"client_id": "test"}, server=server)
        return out, [data.get("exception_message", "") for event, data in server.events if event == "execution_error"]

    try:
        prompt = {**base, "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1), "5": dual_sampler_node(0.0),
                  "k": N("BCTest_DKeep", tag="latent", latent=["5", 2])}
        out, errs = await errors(prompt, "dype-unsupported-dual")
        check("Dual Sampler: dype auto on a model DyPE does not take stops before the first pass, naming the models",
              out is None and not calls and any("DyPE supports Z-Image, Qwen-Image models; this one is Lumina2" in e for e in errs),
              f"{errs} model calls {len(calls)}")
        prompt["5"]["inputs"]["dype"] = "disable"
        kept.clear()
        out, errs = await errors(prompt, "dype-unsupported-dual-off")
        check("Dual Sampler: dype disable on that model runs both passes", out is not None and "latent" in kept and calls,
              f"{errs}")
        out, errs = await errors({**base, "5": N("BC_DyPEPatcher", model=["1", 0], mode="auto", dype_start_sigma=1.0),
                                  "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1),
                                  "6": ksampler_node(["5", 0], ["4", 0], 3, 1.0),
                                  "k": N("BCTest_DKeep", tag="latent", latent=["6", 0])}, "dype-unsupported-patcher")
        check("DyPE Patcher: auto on a model DyPE does not take is an error naming the models", out is None and any(
            "DyPE supports Z-Image, Qwen-Image models" in e for e in errs), f"{errs}")
    finally:
        hook.remove()
        drop_test_nodes()

    base = dype_test_nodes(tiny_zimage(), torch.zeros(1, 8, 32), StandInVAE(frames=False), kept)
    try:
        out, errs = await errors({**base, "c": N("BC_DyPEConfig", model="Qwen-Image", method="vision_yarn", base_resolution=1328,
                                                 dype_scale=2.0, dype_exponent=1.0, auto_threshold=1920, yarn_alt_scaling=False,
                                                 edited=""),
                                  "5": N("BC_DyPEPatcher", model=["1", 0], mode="auto", dype_start_sigma=1.0, dype_config=["c", 0]),
                                  "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1),
                                  "6": ksampler_node(["5", 0], ["4", 0], 3, 1.0),
                                  "k": N("BCTest_DKeep", tag="latent", latent=["6", 0])}, "dype-config-mismatch")
        check("DyPE Patcher: a DyPE Config set for Qwen-Image on a Z-Image model is an error naming both", out is None and any(
            "DyPE Config is set for Qwen-Image, the model is Z-Image" in e for e in errs), f"{errs}")
    finally:
        drop_test_nodes()


def drop_dead_models():
    """Collects an earlier run's freed patchers (they sit in reference cycles until a collection) and drops the entries
    of core's loaded models that core's LoadedModel.is_dead() then calls dead (the patcher freed, the model still
    held)."""
    import gc

    import comfy.model_management as mm

    gc.collect()
    mm.current_loaded_models[:] = [m for m in mm.current_loaded_models if not m.is_dead()]


class RopeLog:
    """Which rotary embedder computes inside a module's forward: while on, core's EmbedND.forward logs "core" and the
    pack's DyPE embedder (libs/dype/embed.py, every model's) "dype". watch(module) records, for each forward of
    `module`, (the latent's width in cells, the set of kinds computed during it)."""

    def __init__(self):
        import importlib

        import comfy.ldm.flux.layers

        pkg = nodes.NODE_CLASS_MAPPINGS["BC_DyPEPatcher"].__module__.rsplit(".", 2)[0]
        self.classes = {"core": comfy.ldm.flux.layers.EmbedND, "dype": importlib.import_module(f"{pkg}.libs.dype.embed").DyPEEmbedND}
        self.held = {kind: cls.forward for kind, cls in self.classes.items()}
        self.open = []    # the kind lists of the forwards in progress
        self.calls = []   # (module label, width in cells, kinds)
        self.hooks = []

    def __enter__(self):
        for kind, cls in self.classes.items():
            def logged(this, *args, _forward=self.held[kind], _kind=kind, **kwargs):
                for kinds in self.open:
                    kinds.add(_kind)
                return _forward(this, *args, **kwargs)
            cls.forward = logged
        return self

    def __exit__(self, *exc):
        for kind, cls in self.classes.items():
            cls.forward = self.held[kind]
        for hook in self.hooks:
            hook.remove()

    def watch(self, module, label):
        def enter(m, args, kwargs):
            x = kwargs.get("x", args[0] if args else None)
            self.open.append(set())
            self.calls.append((label, None if x is None else int(x.shape[-1]), self.open[-1]))

        def leave(m, args, output):
            self.open.pop()

        self.hooks += [module.register_forward_pre_hook(enter, with_kwargs=True), module.register_forward_hook(leave, always_call=True)]

    def of(self, label):
        return [(width, kinds) for name, width, kinds in self.calls if name == label]


async def dype_controlnets():
    """ControlNets with DyPE on (DyPE Patcher enable, a 2048 x 64 px latent) inside ComfyUI, each checked by the rotary
    embedder that computes during it (RopeLog): on tiny Qwen-Image core's Fun ControlNet (QwenFunControlNet: the
    diffusion model's embedder, through Apply ControlNet with the stand-in VAE), the InstantX ControlNet (its own
    embedder) and DiffSynth's block-wise model patch (QwenImageDiffsynthControlnet: inside the model); on tiny Z-Image
    DiffSynth's Fun ControlNet model patch (ZImageFunControlnet: the model's rotations). Each computes with DyPE's
    embedder in every step, with core's EmbedND without DyPE (auto on a 1920 px latent), and the embedders are core's
    after the run. Dual Sampler with the Fun ControlNet: core's in the first pass (1024 px), DyPE's in the second
    (2048 px). An area-conditioned crop (Conditioning Set Area, 1024 of the 2048 px on Z-Image, enable): its model
    calls compute with core's (not larger than base_resolution), the full latent's with DyPE's.

    Before each run the loaded-model entries core's own is_dead() calls dead are dropped: an earlier run's chained
    clones (DiffSynth's, then DyPE's on it) are freed together, so core cannot switch their entry to a parent, and
    core's ControlNet hint re-encode (loaded_models(only_currently_used=True) -> load_models_gpu) would meet it as
    None."""
    import comfy.controlnet
    import comfy.latent_formats
    import comfy.ldm.lumina.controlnet
    import comfy.ldm.qwen_image.controlnet
    import comfy.model_patcher
    import comfy.ops
    import torch

    from comfy_extras.nodes_model_patch import QwenImageBlockWiseControlNet

    ops, cpu = comfy.ops.disable_weight_init, torch.device("cpu")
    image = torch.rand(1, 64, 512, 3, generator=torch.Generator().manual_seed(2))

    def qwen_control(kind, log):
        if kind == "fun":
            model = seeded(comfy.ldm.qwen_image.controlnet.QwenImageFunControlNetModel(
                control_in_features=132, inner_dim=128, num_attention_heads=1, attention_head_dim=128, num_control_blocks=1,
                main_model_double=1, injection_layers=(0,), dtype=torch.float32, device=cpu, operations=ops), 3)
            log.watch(model, "control")
            return comfy.controlnet.QwenFunControlNet(model, compression_ratio=1, latent_format=comfy.latent_formats.Wan21(),
                                                      concat_mask=False, load_device=cpu, manual_cast_dtype=None, extra_conds=[])
        model = seeded(comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel(
            extra_condition_channels=0, dtype=torch.float32, device=cpu, operations=ops, **QWEN_UNET), 4)
        log.watch(model, "control")
        return comfy.controlnet.ControlNet(model, compression_ratio=1, latent_format=comfy.latent_formats.Wan21(), concat_mask=False,
                                           load_device=cpu, manual_cast_dtype=None, extra_conds=[])

    async def sample(label, patcher, vae, width, mode, apply, control=None, patch=None):
        """One KSampler run of `patcher` under DyPE Patcher (mode) with the ControlNet `apply` puts on: "cn" core's
        Apply ControlNet on the conditioning, "qwen" / "zimage" core's DiffSynth node on the model."""
        drop_dead_models()
        kept = {}
        base = dype_test_nodes(patcher, torch.zeros(1, 8, 32), vae, kept, control=control, patch=patch, image=image)
        try:
            prompt = {**base, "i": N("BCTest_DImage"), "4": N("EmptySD3LatentImage", width=width, height=64, batch_size=1),
                      "k": N("BCTest_DKeep", tag="latent", latent=["6", 0])}
            model, cond = ["1", 0], (["2", 0], ["2", 0])
            if apply == "cn":
                prompt["a"] = N("ControlNetApplyAdvanced", positive=["2", 0], negative=["2", 0], control_net=["BCc", 0],
                                image=["i", 0], strength=1.0, start_percent=0.0, end_percent=1.0, vae=["3", 0])
                prompt["BCc"] = N("BCTest_DControl")
                cond = (["a", 0], ["a", 1])
            else:
                prompt["p"] = N("BCTest_DPatch")
                if apply == "qwen":
                    prompt["m"] = N("QwenImageDiffsynthControlnet", model=["1", 0], model_patch=["p", 0], vae=["3", 0],
                                    image=["i", 0], strength=1.0)
                else:
                    prompt["m"] = N("ZImageFunControlnet", model=["1", 0], model_patch=["p", 0], vae=["3", 0], strength=1.0,
                                    image=["i", 0])
                model = ["m", 0]
            prompt["5"] = N("BC_DyPEPatcher", model=model, mode=mode, dype_start_sigma=1.0)
            prompt["6"] = ksampler_node(["5", 0], ["4", 0], 3, 1.0, *cond)
            out, _ = await run(prompt, label)
            return out is not None and "latent" in kept
        finally:
            drop_test_nodes()

    def kinds_of(calls):
        return sorted({kind for _, kinds in calls for kind in kinds}) if calls else None

    qwen = tiny_qwen_image()
    for kind, what in (("fun", "Fun ControlNet (the diffusion model's embedder)"), ("instantx", "InstantX ControlNet (its own embedder)")):
        for width, mode, want in ((2048, "enable", ["dype"]), (1920, "auto", ["core"])):
            with RopeLog() as log:
                control = qwen_control(kind, log)
                ran = await sample(f"dype-qwen-{kind}-{width}", qwen, StandInVAE(frames=True), width, mode, "cn", control=control)
            got = kinds_of(log.of("control"))
            check(f"DyPE Qwen-Image {what}: on a {width} px latent ({mode}) it computes with {want[0]}", ran and got == want,
                  f"ran {ran} computed {got} in {len(log.of('control'))} calls")
        held = control.control_model.pe_embedder if kind == "instantx" else qwen.model.diffusion_model.pe_embedder
        check(f"DyPE Qwen-Image {what}: the embedder is core's after the runs", type(held).__name__ == "EmbedND",
              type(held).__name__)

    blockwise = seeded(QwenImageBlockWiseControlNet(num_layers=1, in_dim=64, additional_in_dim=0, dim=128, device=cpu,
                                                    dtype=torch.float32, operations=ops), 5)
    held_block = blockwise.control_block
    patched_calls = []
    blockwise.control_block = lambda img, cond, block_id: (patched_calls.append(1), held_block(img, cond, block_id))[1]
    patch = comfy.model_patcher.CoreModelPatcher(blockwise, load_device=cpu, offload_device=cpu)
    for width, mode, want in ((2048, "enable", ["dype"]), (1920, "auto", ["core"])):
        patched_calls.clear()
        with RopeLog() as log:
            log.watch(qwen.model.diffusion_model, "model")
            ran = await sample(f"dype-qwen-diffsynth-{width}", qwen, StandInVAE(frames=True), width, mode, "qwen", patch=patch)
        got = kinds_of(log.of("model"))
        check(f"DyPE Qwen-Image DiffSynth ControlNet: on a {width} px latent ({mode}) the model it runs in computes with "
              f"{want[0]}", ran and patched_calls and got == want, f"ran {ran} patch calls {len(patched_calls)} computed {got}")

    zimage = tiny_zimage()
    fun = seeded(comfy.ldm.lumina.controlnet.ZImage_Control(dim=256, n_heads=2, n_kv_heads=2, multiple_of=32,
                                                             ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
                                                             n_control_layers=1, control_in_dim=16, dtype=torch.float32,
                                                             device=cpu, operations=ops), 6)
    patch = comfy.model_patcher.CoreModelPatcher(fun, load_device=cpu, offload_device=cpu)
    for width, mode, want in ((2048, "enable", ["dype"]), (1920, "auto", ["core"])):
        with RopeLog() as log:
            log.watch(zimage.model.diffusion_model, "model")
            log.watch(fun, "control")
            ran = await sample(f"dype-zimage-diffsynth-{width}", zimage, StandInVAE(frames=False), width, mode, "zimage", patch=patch)
        got = kinds_of(log.of("model"))
        check(f"DyPE Z-Image DiffSynth Fun ControlNet: on a {width} px latent ({mode}) the model it runs in computes with "
              f"{want[0]}", ran and log.of("control") and got == want, f"ran {ran} control calls {len(log.of('control'))} "
              f"computed {got}")

    drop_dead_models()
    kept = {}
    with RopeLog() as log:
        control = qwen_control("fun", log)
        base = dype_test_nodes(qwen, torch.zeros(1, 8, 32), StandInVAE(frames=True), kept, control=control, image=image)
        try:
            prompt = {**base, "i": N("BCTest_DImage"), "BCc": N("BCTest_DControl"),
                      "a": N("ControlNetApplyAdvanced", positive=["2", 0], negative=["2", 0], control_net=["BCc", 0], image=["i", 0],
                             strength=1.0, start_percent=0.0, end_percent=1.0, vae=["3", 0]),
                      "4": N("EmptySD3LatentImage", width=1024, height=64, batch_size=1),
                      "5": dual_sampler_node(0.0, "auto", positive=("a", 0), negative=("a", 1)),
                      "k": N("BCTest_DKeep", tag="latent", latent=["5", 2])}
            out, _ = await run(prompt, "dype-qwen-fun-dual")
        finally:
            drop_test_nodes()
    by_width = {}
    for width, kinds in log.of("control"):
        by_width.setdefault(width, set()).update(kinds)
    check("Dual Sampler Qwen-Image Fun ControlNet: core's embedder in the first pass (1024 px), DyPE's in the second "
          "(2048 px)", out is not None and by_width == {128: {"core"}, 256: {"dype"}}, f"{by_width}")

    drop_dead_models()
    kept = {}
    base = dype_test_nodes(zimage, torch.randn(1, 8, 32, generator=torch.Generator().manual_seed(1)), StandInVAE(frames=False), kept)
    try:
        prompt = {**base, "s": N("ConditioningSetArea", conditioning=["2", 0], width=1024, height=64, x=0, y=0, strength=1.0),
                  "c": N("ConditioningCombine", conditioning_1=["2", 0], conditioning_2=["s", 0]),
                  "4": N("EmptySD3LatentImage", width=2048, height=64, batch_size=1),
                  "5": N("BC_DyPEPatcher", model=["1", 0], mode="enable", dype_start_sigma=1.0),
                  "6": ksampler_node(["5", 0], ["4", 0], 3, 1.0, ["c", 0], ["2", 0]),
                  "k": N("BCTest_DKeep", tag="latent", latent=["6", 0])}
        with RopeLog() as log:
            log.watch(zimage.model.diffusion_model, "model")
            out, _ = await run(prompt, "dype-zimage-area")
    finally:
        drop_test_nodes()
    by_width = {}
    for width, kinds in log.of("model"):
        by_width.setdefault(width, set()).update(kinds)
    check("DyPE Z-Image area conditioning: the 1024 px crop's model calls compute with core's embedder, the 2048 px "
          "latent's with DyPE's", out is not None and by_width == {128: {"core"}, 256: {"dype"}}, f"{by_width}")


def detailer_zimage_patch():
    """The detailer's cut of Z-Image's Fun ControlNet (pipelines/detailer/conditioning.crop_model) on a model
    patched by core's own ZImageFunControlnet node, with an inpaint image and a mask: a Union 2.1-shaped
    ZImage_Control (inpaint channels) built on the meta device, no weights (the node reads its class and
    additional_in_dim; a call past the last control block returns before the model runs) and a stand-in VAE.
    Core's patch is found although core loads its module under the file's path; on a clone, its hint,
    inpaint image and mask are cut to the crop, and core's own call encodes the cut at the crop's sampling
    size; the model connected keeps its lists, its patch and the patch's encode."""
    import importlib

    import comfy.ldm.lumina.controlnet
    import comfy.model_patcher
    import comfy.ops
    import torch

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_Detailer"].__module__.rsplit(".", 2)[0]
    cond = importlib.import_module(f"{pkg}.pipelines.detailer.conditioning")
    crop = importlib.import_module(f"{pkg}.pipelines.detailer.region").Crop(32, 16, 96, 80, 512, 512)

    class VAE:
        def __init__(self):
            self.encoded = []

        def encode(self, pixels):
            self.encoded.append(tuple(pixels.shape))
            return torch.zeros(pixels.shape[0], 16, pixels.shape[1] // 8, pixels.shape[2] // 8)

        def spacial_compression_encode(self):
            return 8

    control = comfy.ldm.lumina.controlnet.ZImage_Control(dim=64, n_heads=2, n_kv_heads=2, n_control_layers=3, additional_in_dim=17,
                                                         refiner_control=True, device="meta", operations=comfy.ops.manual_cast)
    model = comfy.model_patcher.ModelPatcher(torch.nn.Linear(1, 1), torch.device("cpu"), torch.device("cpu"))
    image, mask, vae = torch.rand(1, 96, 128, 3), torch.zeros(1, 96, 128), VAE()
    mask[:, 20:60, 40:90] = 1.0
    patched = nodes.NODE_CLASS_MAPPINGS["ZImageFunControlnet"]().diffsynth_controlnet(
        model, types.SimpleNamespace(model=control), vae, image=image, strength=0.8, inpaint_image=image, mask=mask)[0]
    lists = patched.model_options["transformer_options"]["patches"]
    patch = lists["double_block"][0]
    held = (patch.image, patch.inpaint_image, patch.mask, patch.encoded_image)
    entries = {name: list(entry) for name, entry in lists.items()}

    out = cond.crop_model(patched, crop, 128, 96)
    cut_lists = out.model_options["transformer_options"]["patches"]
    cut = cut_lists["double_block"][0]
    check("Detailer Z-Image ControlNet: core's patch found and copied on a clone, one copy in both of core's lists",
          out is not patched and type(cut) is type(patch) and cut is not patch and cut_lists["noise_refiner"] == [cut]
          and len(cut_lists["double_block"]) == 1, f"{ {k: [type(p).__name__ for p in v] for k, v in cut_lists.items()} }")
    check("Detailer Z-Image ControlNet: the hint, the inpaint image and core's mask cut to the crop, no encode kept",
          torch.equal(cut.image, image[:, 16:80, 32:96]) and torch.equal(cut.inpaint_image, image[:, 16:80, 32:96])
          and cut.mask.shape == (1, 1, 1, 64, 64) and torch.equal(cut.mask, patch.mask[..., 16:80, 32:96])
          and cut.encoded_image is None and cut.temp_data is None,
          f"{tuple(cut.image.shape)} {tuple(cut.mask.shape)} {cut.encoded_image is None}")

    vae.encoded.clear()
    latent = torch.zeros(1, 16, crop.height // 8, crop.width // 8)
    cut({"x": latent, "img": None, "img_input": None, "txt": None, "pe": None, "vec": None, "block_index": 30, "block_type": "double"})
    check("Detailer Z-Image ControlNet: core's call encodes the cut hint and inpaint image at the crop's sampling size",
          vae.encoded == [(1, 512, 512, 3)] * 2 and tuple(cut.encoded_image.shape) == (1, 33, 64, 64), f"{vae.encoded}")
    check("Detailer Z-Image ControlNet: the model connected keeps its lists, its patch and the patch's hint and encode",
          {name: list(entry) for name, entry in lists.items()} == entries and lists["double_block"][0] is patch
          and all(a is b for a, b in zip((patch.image, patch.inpaint_image, patch.mask, patch.encoded_image), held))
          and tuple(patch.encoded_image.shape) == (1, 33, 12, 16), f"{tuple(patch.encoded_image.shape)}")


def detailer_qwen_fun_controlnet():
    """The detailer's cut of a ControlNet in the conditioning (pipelines/detailer/conditioning.crop_conditioning)
    on Qwen-Image's Fun ControlNet as core's Apply ControlNet (ControlNetApplyAdvanced) puts it there: a
    QwenFunControlNet configured as core's loader configures it (load_controlnet_qwen_fun), no weights, and a
    stand-in VAE that records what it is given and stops get_control before the control model would run.
    Each link is copied through core's own copy(), so the copy keeps its class and settings and drops the
    hint core cached for another latent size; the hint is cut to the crop, and core scales the cut to the
    crop's sampling size before its VAE encode."""
    import importlib

    import comfy.controlnet
    import comfy.latent_formats
    import comfy.utils
    import torch

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_Detailer"].__module__.rsplit(".", 2)[0]
    cond = importlib.import_module(f"{pkg}.pipelines.detailer.conditioning")
    crop = importlib.import_module(f"{pkg}.pipelines.detailer.region").Crop(32, 16, 96, 80, 512, 512)

    class Stop(Exception):
        pass

    class VAE:
        def __init__(self):
            self.encoded = []

        def encode(self, pixels):
            self.encoded.append(pixels.clone())
            raise Stop  # the control model (no weights) would run next

        def spacial_compression_encode(self):
            return 8

    vae = VAE()
    control = comfy.controlnet.QwenFunControlNet(None, compression_ratio=1, latent_format=comfy.latent_formats.Wan21(), concat_mask=False,
                                                 load_device=torch.device("cpu"), extra_conds=[])
    control.control_model_wrapped = None  # no weights: copy() only passes it on
    image = torch.rand(1, 96, 128, 3)
    positive, negative = nodes.ControlNetApplyAdvanced().apply_controlnet([[torch.zeros(1, 2, 4), {}]], [[torch.zeros(1, 2, 4), {}]],
                                                                          control, image, 1.0, 0.0, 1.0, vae=vae)
    applied = positive[0][1]["control"]
    applied.cond_hint = torch.zeros(1, 16, 1, 12, 16)  # the hint core cached in a run at the image's own size
    held = (applied.cond_hint_original, applied.cond_hint, dict(positive[0][1]), dict(negative[0][1]))

    done = {}
    cut_positive = cond.crop_conditioning(positive, crop, 128, 96, done)
    cut_negative = cond.crop_conditioning(negative, crop, 128, 96, done)
    cut = cut_positive[0][1]["control"]
    check("Detailer Qwen Fun ControlNet: the copy keeps core's class", type(cut) is comfy.controlnet.QwenFunControlNet
          and cut is not applied, f"{type(cut).__name__}")
    check("Detailer Qwen Fun ControlNet: the positive and the negative share one copy, as Apply ControlNet shares one link",
          cut_negative[0][1]["control"] is cut)
    check("Detailer Qwen Fun ControlNet: the copy keeps the VAE, the latent format and the mask settings",
          cut.vae is vae and cut.latent_format is applied.latent_format and cut.concat_mask is False and cut.extra_concat_orig == []
          and cut.extra_conds == [] and cut.compression_ratio == 1 and cut.upscale_algorithm == applied.upscale_algorithm
          and (cut.strength, cut.timestep_percent_range) == (1.0, (0.0, 1.0)), f"{cut.vae} {cut.concat_mask} {cut.extra_concat_orig}")
    check("Detailer Qwen Fun ControlNet: the copy has no cached hint", cut.cond_hint is None)
    check("Detailer Qwen Fun ControlNet: the hint cut to the crop", tuple(cut.cond_hint_original.shape) == (1, 3, 64, 64)
          and torch.equal(cut.cond_hint_original, image.movedim(-1, 1)[..., 16:80, 32:96]), f"{tuple(cut.cond_hint_original.shape)}")

    cut.control_model = types.SimpleNamespace(dtype=torch.float32)  # get_control reads its dtype before the hint
    try:
        cut.get_control(torch.zeros(1, 16, crop.height // 8, crop.width // 8), torch.tensor([0.5]), {"c_crossattn": torch.zeros(1, 2, 4)},
                        [1], {})
    except Stop:
        pass
    want = comfy.utils.common_upscale(cut.cond_hint_original, crop.width, crop.height, cut.upscale_algorithm, "center").movedim(1, -1)
    check("Detailer Qwen Fun ControlNet: core's get_control hands the VAE the cut hint at the crop's sampling size",
          len(vae.encoded) == 1 and torch.equal(vae.encoded[0], want), f"{[tuple(e.shape) for e in vae.encoded]}")
    check("Detailer Qwen Fun ControlNet: the conditioning and the ControlNet connected are left as they are",
          positive[0][1] == held[2] and negative[0][1] == held[3] and positive[0][1]["control"] is applied
          and applied.cond_hint_original is held[0] and applied.cond_hint is held[1] and applied.previous_controlnet is None)


def lm_core():
    """The LM runtime against core's own Qwen code, no weights:

    - tokenization: for ordinary text (no `embedding:`, no `\\(`, no combining marks) the token structure
      models/common/lm/core_backend.tokenize builds equals what core's Qwen3.5 (qwen35_9b) and Qwen3-VL
      (qwen3vl_8b) tokenizers return from tokenize_with_weights, for prompts of every template, thinking
      on and off, with a prefill and 0 / 1 / 3 images: the same ids, weights 1.0, and image entries
      holding the very frames in order. With combining marks ours gives the official Qwen3.5 ids (literals
      from the official tokenizer.json) where core's differ; Qwen3-VL's tokenizer is used as it is. On
      `\\(` and `embedding:` core changes the ids; ours keeps the HF tokenizer's own. A stray
      <|image_pad|> in the text, which core leaves as an id with no image, is refused;
    - LoRA (libs/lm_lora): core's CLIP models built on the meta device give the real key maps. Every PEFT
      key form (ForCausalLM and ForConditionalGeneration module names, `.default` factors, embed_tokens,
      lm_head, the vision tower) lands on its weight; weight_shapes covers every weight the key map
      addresses; an adapter of Qwen3.5-9B's structure (200 modules) passes check_base on qwen35_9b and is
      refused on qwen35_4b and qwen35_27b (every name there, no shape fits) and on qwen3vl_8b (72 modules
      missing, where core alone would apply 120 of the 200); on qwen35_4b, whose core build ties lm_head to
      embed_tokens (no lm_head weight), a LoRA on embed_tokens or lm_head is refused for the tie; core's
      load_lora reads every tensor of the plan, alpha included;
    - the cuDNN attention fallback's footing in core: its SDPA wrapper reads SDPA_BACKEND_PRIORITY at call
      time, and the executor's cleanup it calls exists."""
    import importlib
    import types

    import comfy.lora
    import comfy.lora_convert
    import comfy.text_encoders.qwen35 as qwen35
    import comfy.text_encoders.qwen3vl as qwen3vl
    import torch

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_ImageResize"].__module__.rsplit(".", 2)[0]

    def pack(name):
        return importlib.import_module(f"{pkg}.{name}")

    core_backend, chat, lm_lora = pack("models.common.lm.core_backend"), pack("models.qwen_lm.chat"), pack("libs.lm_lora")
    family = pack("models.qwen_lm").FAMILY
    catalog = pack("models.common.lm.catalog").load_catalog(family.catalog_path, [], family.templates,
                                                           default_backend=family.default_backend)
    PromptParts = pack("models.common.lm.family").PromptParts
    tokenizers = {"qwen35_9b": qwen35.tokenizer(model_type="qwen35_9b")(embedding_directory=None),
                  "qwen3vl_8b": qwen3vl.tokenizer(model_type="qwen3vl_8b")(embedding_directory=None)}

    def ours(tok, prompt, images):
        return core_backend.tokenize(types.SimpleNamespace(tokenizer=tok), prompt, images)

    def entry_equal(x, y):
        if isinstance(x, dict):
            return (isinstance(y, dict) and x.keys() == y.keys() == {"type", "data", "original_type"}
                    and x["type"] == y["type"] == "image" and x["original_type"] == y["original_type"]
                    and x["data"].data_ptr() == y["data"].data_ptr() and x["data"].shape == y["data"].shape)
        return type(x) is type(y) and x == y

    def same(a, b):
        """Equal token structures: one key, the same rows of (id or image entry, weight)."""
        return a.keys() == b.keys() and all(
            len(a[k]) == len(b[k]) and all(len(r) == len(s) and all(entry_equal(x[0], y[0]) and x[1:] == y[1:] for x, y in zip(r, s))
                                           for r, s in zip(a[k], b[k])) for k in a)

    texts = ["Describe this.", "  Two spaces, a tab\tand\nnew\n\nlines.  \n", 'JSON: {"a": [1, 2.5e-3], "b": null}',
             "def f(x):\n    return x ** 2  # code \\d+ C:\\path", "Unicode: çğıöşü İ 日本語 中文 العربية 👩‍👩‍👧 é",
             "(weight:1.2) [brackets] <angle> 100% & 'q'"]
    total = equal = 0
    for model_name, tok_name in (("Qwen3.5-9B", "qwen35_9b"), ("Qwen3.8-27B", "qwen35_9b"), ("Qwen3-VL-8B Instruct", "qwen3vl_8b")):
        model, tok = catalog[model_name], tokenizers[tok_name]
        for system in ("", "You are a helpful assistant.", texts[1]):
            for user in texts:
                for prefill in ("", '{"name": "', "  Sure,\n"):
                    for n in (0, 1, 3):
                        for thinking in ((False, True) if model.thinking else (False,)):
                            if thinking and prefill:
                                continue
                            prompt = chat.build_prompt(model, PromptParts(system=system, user=user, assistant=prefill, n_images=n, thinking=thinking))
                            images = torch.rand((n, 4, 5, 3)) if n else None
                            frames = [] if images is None else [images[i:i + 1] for i in range(n)]
                            total += 1
                            equal += same(ours(tok, prompt, images), tok.tokenize_with_weights(prompt, False, images=frames))
    check(f"LM: the token structure of {total} ordinary prompts (no combining marks) equals core's tokenize_with_weights "
          "(Qwen3.5 / 3.8 on qwen35_9b's tokenizer, Qwen3-VL on qwen3vl_8b's; 0 / 1 / 3 images)", equal == total,
          f"{equal} of {total}")

    # Core's Qwen3.5 tokenizer splits words with Qwen2's regex (no \p{M}); ours with the official one, so a word with
    # combining marks gets the official ids. The ids below come from the official tokenizer.json of Qwen/Qwen3.5-9B
    # (Qwen/Qwen3.8-27B's gives the same).
    official = {
        "नमस्ते दुनिया, यह एक परीक्षण है।": [58069, 84237, 150104, 153348, 184642, 235886, 11, 175089, 171549, 155171, 42201,
                                       242749, 170046, 154845],
        "สวัสดีครับ ภาษาไทย": [34469, 168607, 153295, 212495, 149826],
        "مَرْحَبًا بِالْعَالَم": [9873, 164865, 28850, 150765, 150021, 222153, 187581, 155815, 152397],
        "আমি বাংলায় গান গাই": [168955, 148653, 59283, 166781, 148787, 152485, 149321, 152485, 172980],
        "<|im_start|>user\nவணக்கம் உலகம்<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n":
            [248045, 846, 198, 149608, 150229, 152246, 150075, 160321, 149255, 148574, 150075, 248046, 198, 248045, 74455,
             198, 248068, 271, 248069, 271],
        "Describe a cat on a sofa. It's 12:30, isn't it?": [72240, 264, 7993, 383, 264, 30051, 13, 1049, 579, 220, 16, 17,
                                                            25, 18, 15, 11, 4290, 914, 424, 30],
        "Şu görseli ayrıntılı anlat, lütfen. İĞÜŞÖÇ ığüşöç": [168981, 84, 29200, 5259, 11942, 215583, 398, 50420, 204408, 11,
                                                         324, 2314, 8665, 268, 13, 37146, 169384, 50724, 168981, 61222,
                                                         44382, 220, 89393, 153675, 2862, 3031],
    }
    tok = tokenizers["qwen35_9b"]
    got = {text: [t for t, _ in ours(tok, text, None)[tok.clip_name][0]] for text in official}
    check("LM: Qwen3.5 / 3.8: ours gives the official tokenizer.json ids (Hindi, Thai, Arabic, Bengali, a Tamil chat "
          "prompt, English, Turkish)", got == official, f"{[t for t in official if got[t] != official[t]]}")
    marked = list(official)[:5]
    differ, vl_equal = 0, 0
    for text in marked:
        for model_name, tok_name in (("Qwen3.5-9B", "qwen35_9b"), ("Qwen3-VL-8B Instruct", "qwen3vl_8b")):
            prompt = chat.build_prompt(catalog[model_name], PromptParts(system="", user=text, assistant="", n_images=0, thinking=False))
            tok = tokenizers[tok_name]
            same_as_core = same(ours(tok, prompt, None), tok.tokenize_with_weights(prompt, False))
            differ += tok_name == "qwen35_9b" and not same_as_core
            vl_equal += tok_name == "qwen3vl_8b" and same_as_core
    check("LM: with combining marks core's Qwen3.5 tokenize departs from the official ids; Qwen3-VL's (no "
          "pretokenize_regex, Qwen2's split is official there) is used as it is", differ == vl_equal == len(marked),
          f"{differ} {vl_equal} of {len(marked)}")

    model = catalog["Qwen3.5-9B"]
    for label, user in (("\\( \\)", "Keep \\(x\\) as written."), ("embedding:", "Use embedding:foo here.")):
        kept = changed = True
        for tok_name, tok in tokenizers.items():
            prompt = chat.build_prompt(model, PromptParts(system="", user=user, assistant="", n_images=0, thinking=False))
            canonical = getattr(tok, tok.clip).tokenizer(prompt)["input_ids"]
            kept &= [t for t, _ in ours(tok, prompt, None)[tok.clip_name][0]] == canonical
            changed &= [t for t, _ in tok.tokenize_with_weights(prompt, False)[tok.clip_name][0]] != canonical
        check(f"LM: on {label} core's tokenize changes the ids; ours keeps the HF tokenizer's", kept and changed, f"{kept} {changed}")
    tok = tokenizers["qwen35_9b"]
    # build_prompt refuses a typed pad (chat._check); one is written in here to test the backend's own guard
    prompt = chat.build_prompt(model, PromptParts(system="", user="see", assistant="", n_images=1, thinking=False)).replace(
        "see<|im_end|>", "see <|image_pad|><|im_end|>")
    image = torch.rand((1, 4, 5, 3))
    stray = sum(1 for t, _ in tok.tokenize_with_weights(prompt, False, images=[image])[tok.clip_name][0] if t == 248056)
    try:
        ours(tok, prompt, image)
        refused = ""
    except ValueError as e:
        refused = str(e)
    check("LM: a stray <|image_pad|> in a prompt is refused by the backend too (core leaves it as an id with no image)",
          stray == 1 and "2 <|image_pad|> token(s) for 1 image(s)" in refused, f"{stray} {refused!r}")

    built = {}
    for name, te in (("qwen35_9b", qwen35.te(model_type="qwen35_9b", mtp=True)), ("qwen35_4b", qwen35.te(model_type="qwen35_4b", mtp=True)),
                     ("qwen35_27b", qwen35.te(model_type="qwen35_27b", mtp=True)), ("qwen3vl_8b", qwen3vl.te(model_type="qwen3vl_8b"))):
        with torch.device("meta"):
            clip_model = te(device="meta", dtype=torch.bfloat16)
        built[name] = (comfy.lora.model_lora_keys_clip(clip_model, {}), core_backend.weight_shapes(clip_model), clip_model)
    check("LM: weight_shapes holds every weight core's key map addresses, with the state dict's shapes",
          all(set(km.values()) <= ws.keys() and all(ws[k] == tuple(v.shape) for k, v in m.state_dict().items() if k.endswith(".weight"))
              for km, ws, m in built.values()))

    def source(tensors):
        return lm_lora.LoraSource("synthetic", tensors, {}, {"r": 2, "lora_alpha": 4, "peft_type": "LORA"}, "adapter_config.json")

    def pair(name, out_in, a="lora_A.weight", b="lora_B.weight"):
        out, inn = out_in
        return {f"{name}.{a}": torch.zeros(2, inn), f"{name}.{b}": torch.zeros(out, 2)}

    # PEFT module name -> the state-dict key it must patch, per model
    forms = {
        "qwen35_9b": {"base_model.model.model.layers.0.linear_attn.in_proj_qkv": "model.layers.0.linear_attn.in_proj_qkv",
                      "base_model.model.model.language_model.layers.3.self_attn.q_proj": "model.layers.3.self_attn.q_proj",
                      "base_model.model.model.language_model.layers.5.mlp.down_proj": "model.layers.5.mlp.down_proj",
                      "base_model.model.lm_head": "model.lm_head",
                      "base_model.model.model.visual.blocks.0.attn.qkv": "visual.blocks.0.attn.qkv",
                      "base_model.model.model.visual.merger.linear_fc1": "visual.merger.linear_fc1"},
        "qwen3vl_8b": {"base_model.model.model.layers.0.self_attn.k_proj": "model.layers.0.self_attn.k_proj",
                       "base_model.model.model.language_model.layers.35.mlp.up_proj": "model.layers.35.mlp.up_proj",
                       "base_model.model.lm_head": "model.lm_head",
                       "base_model.model.model.visual.blocks.26.mlp.linear_fc2": "visual.blocks.26.mlp.linear_fc2",
                       "base_model.model.model.visual.deepstack_merger_list.0.linear_fc1": "visual.deepstack_merger_list.0.linear_fc1"},
    }
    for name, names in forms.items():
        km, ws, _ = built[name]
        target = {peft: f"{name}.transformer.{module}.weight" for peft, module in names.items()}
        tensors = {}
        for i, (peft, key) in enumerate(target.items()):
            a, b = ("lora_A.default.weight", "lora_B.default.weight") if i % 2 else ("lora_A.weight", "lora_B.weight")
            tensors.update(pair(peft, ws[key], a, b))
        for embed in ("base_model.model.model.embed_tokens", "base_model.model.model.language_model.embed_tokens"):
            vocab, hidden = ws[f"{name}.transformer.model.embed_tokens.weight"]
            plan = lm_lora.plan_lora(source({**tensors, f"{embed}.lora_embedding_A": torch.zeros(2, vocab),
                                             f"{embed}.lora_embedding_B": torch.zeros(hidden, 2)}))
            try:
                lm_lora.check_base(plan, km, ws)
                landed = {km.get(m) for m in plan.modules}
            except lm_lora.LoraError as e:
                landed = str(e)
            check(f"LM: {name}: the PEFT key forms ({embed.split('.')[-2]}.embed_tokens among them) land on their weights and pass check_base",
                  landed == set(target.values()) | {f"{name}.transformer.model.embed_tokens.weight"}, f"{landed}")

    # qwen35_4b has no lm_head weight: core's logits (and MTP head) use embed_tokens.weight, so a LoRA on either
    # would patch both; told from the weight shapes
    km4, ws4, _ = built["qwen35_4b"]
    vocab, hidden = ws4["qwen35_4b.transformer.model.embed_tokens.weight"]
    layer = pair("base_model.model.model.language_model.layers.3.self_attn.q_proj", ws4["qwen35_4b.transformer.model.layers.3.self_attn.q_proj.weight"])
    verdicts = {}
    for label, extra in (("none", {}), ("embed_tokens", {"base_model.model.model.embed_tokens.lora_embedding_A": torch.zeros(2, vocab),
                                                         "base_model.model.model.embed_tokens.lora_embedding_B": torch.zeros(hidden, 2)}),
                         ("language_model.embed_tokens", {"base_model.model.model.language_model.embed_tokens.lora_embedding_A": torch.zeros(2, vocab),
                                                          "base_model.model.model.language_model.embed_tokens.lora_embedding_B": torch.zeros(hidden, 2)}),
                         ("lm_head", pair("base_model.model.lm_head", (vocab, hidden)))):
        try:
            lm_lora.check_base(lm_lora.plan_lora(source({**layer, **extra})), km4, ws4)
            verdicts[label] = "pass"
        except lm_lora.LoraError as e:
            verdicts[label] = str(e)
    check("LM: qwen35_4b (tied head): a LoRA on embed_tokens or lm_head is refused for the tie, one without them passes",
          verdicts["none"] == "pass" and all(verdicts[k].startswith("this model ties lm_head to embed_tokens") and "another base" not in verdicts[k]
                                             for k in ("embed_tokens", "language_model.embed_tokens", "lm_head")), f"{verdicts}")

    # Qwen3.5-9B's adapter structure: 24 linear-attention layers x 3, 8 full-attention layers x 4, 32 MLPs x 3
    km9, ws9, _ = built["qwen35_9b"]
    modules = [f"layers.{i}.{kind}.{p}" for i in range(32)
               for kind, projections in ((("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")) if i % 4 == 3 else
                                          ("linear_attn", ("in_proj_qkv", "in_proj_z", "out_proj"))), ("mlp", ("gate_proj", "up_proj", "down_proj")))
               for p in projections]
    tensors = {}
    for module in modules:
        tensors.update(pair(f"base_model.model.model.language_model.{module}", ws9[f"qwen35_9b.transformer.model.{module}.weight"]))
    plan = lm_lora.plan_lora(source(tensors))
    verdicts, counts = {}, {}
    for name, (km, ws, _) in built.items():
        try:
            lm_lora.check_base(plan, km, ws)
            verdicts[name] = "pass"
        except lm_lora.LoraError as e:
            verdicts[name] = str(e)
        named = [m for m in plan.modules if m in km]
        delta = {m: (plan.tensors[f"{m}.lora_B.weight"].shape[0], plan.tensors[f"{m}.lora_A.weight"].shape[1]) for m in named}
        counts[name] = (len(named), sum(1 for m in named if ws[km[m]] == delta[m]))  # (names core finds, shapes that fit)
    check("LM: a Qwen3.5-9B adapter (200 modules) passes check_base on qwen35_9b only; on qwen3vl_8b core alone would apply 120",
          len(plan.modules) == 200 and verdicts["qwen35_9b"] == "pass"
          and counts == {"qwen35_9b": (200, 200), "qwen35_4b": (200, 0), "qwen35_27b": (200, 0), "qwen3vl_8b": (128, 120)}
          and all("200 of 200" in verdicts[n] for n in ("qwen35_4b", "qwen35_27b")) and "72 of 200" in verdicts["qwen3vl_8b"],
          f"{len(plan.modules)} {counts} {verdicts}")
    lines = Lines()
    logging.getLogger().addHandler(lines)
    try:
        patches = comfy.lora.load_lora(comfy.lora_convert.convert_lora(dict(plan.tensors)), km9)
    finally:
        logging.getLogger().removeHandler(lines)
    check("LM: core's load_lora reads every tensor of the plan (alpha 4 for rank 2: scale 2)",
          len(patches) == 200 and not any("not loaded" in line for line in lines.lines)
          and all(p.weights[2] == 4.0 for p in patches.values()), f"{len(patches)} {lines.lines[:3]}")

    # The cuDNN attention fallback (core_backend.generate_ids) relies on core's SDPA wrapper reading the module global
    # at every call, and calls the executor's own after-node cleanup between the attempts.
    import inspect

    import comfy.memory_management
    import comfy.model_management
    import comfy.model_prefetch
    import comfy.ops
    check("LM: core's SDPA wrapper reads comfy.ops.SDPA_BACKEND_PRIORITY at call time; the cleanup the cuDNN fallback "
          "calls exists",
          "with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True)" in inspect.getsource(comfy.ops)
          and callable(getattr(comfy.model_prefetch, "cleanup_prefetch_queues", None))
          and callable(getattr(comfy.model_management, "reset_cast_buffers", None))
          and isinstance(getattr(comfy.memory_management, "aimdo_enabled", None), bool))


async def lm_nodes():
    """Qwen LM and LM Config inside ComfyUI, the "core" LM backend replaced by a recording stand-in (no
    weights): GET /bcnodes/lm/catalog is on the server and answers per family; the Qwen-LM folder is
    registered with the .safetensors filter; a prompt LM Config -> Qwen LM -> Preview Any validates and runs,
    the backend gets the official Qwen3.5 rendering, the config's edited fields over the model's defaults and
    a seed of 2**64 - 1, and the same prompt queued again comes from the cache (seed fixed: the backend is not
    called again); thinking on a model without it fails the run with the template's message; an LM Config field
    fed by a link (a PrimitiveFloat) is applied with edited empty, the node reading the link from its hidden
    PROMPT."""
    import importlib

    import folder_paths
    import server as comfy_server

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_QwenLM"].__module__.rsplit(".", 2)[0]
    registry = importlib.import_module(f"{pkg}.models.common.registry")
    handler = next((r.handler for r in comfy_server.PromptServer.instance.routes
                    if r.method == "GET" and r.path == "/bcnodes/lm/catalog"), None)
    body = json.loads((await handler(None)).body) if handler else {}
    family = body.get("families", {}).get("qwen_lm", {})
    check("LM: GET /bcnodes/lm/catalog answers per family (name, node, error, models with thinking / mtp / "
          "precisions / defaults, loras) and names the config and loader nodes",
          body.get("config_node") == "BC_LMConfig" and body.get("loader_node") == "BC_LMLoader"
          and family.get("name") == "Qwen LM" and family.get("node") == "BC_QwenLM" and family.get("error") is None
          and family.get("models", {}).get("Qwen3.8-27B", {}).get("precisions") == ["BF16", "INT8 ConvRot", "W4A8"]
          and list(family["models"]["Qwen3-VL-8B Instruct"]["defaults"]) == ["thinking_off"]
          and family.get("loras", [None])[0] == "None", f"{str(body)[:300]}")
    paths, exts = folder_paths.folder_names_and_paths.get("Qwen-LM", ([], set()))
    check("LM: the Qwen-LM folder is registered (models/Qwen-LM) with the .safetensors filter",
          os.path.join(folder_paths.models_dir, "Qwen-LM") in paths and exts == {".safetensors"}, f"{paths} {exts}")

    qwen = tempfile.mkdtemp(prefix="bcnodes_qwen_lm_")
    open(os.path.join(qwen, "Qwen3.5-9B_int8_convrot.safetensors"), "wb").close()
    folder_paths.add_model_folder_path("Qwen-LM", qwen, is_default=True)
    calls = []

    class StandIn:
        name = "core"

        def load(self, path, lora):
            calls.append(("load", path, lora))
            return "handle"

        def generate(self, handle, prompt, images, params):
            calls.append(("generate", prompt, params))
            return " A red cube.\n"

        def unload(self):
            return {}

    real = registry.get(registry.LM_BACKEND, "core")
    registry._FAMILIES[registry.LM_BACKEND]["core"] = StandIn()
    try:
        def lm_prompt(model, thinking):
            return {
                "1": N("BC_LMConfig", do_sample=True, temperature=0.3, top_k=20, top_p=0.8, min_p=0.0, repetition_penalty=1.0,
                       presence_penalty=1.5, mtp="off", edited="temperature, mtp"),
                "2": N("BC_QwenLM", model=model, precision="INT8 ConvRot", lora="None", lora_strength=1.0, thinking=thinking,
                       max_new_tokens=32768, seed=2 ** 64 - 1, keep_model_loaded=True, system="You are a helpful assistant.",
                       user="Describe this.", assistant="", config=["1", 0]),
                "3": N("PreviewAny", source=["2", 0]),
                "4": N("PreviewAny", source=["2", 1]),
            }
        out, executed = await run_twice(lm_prompt("Qwen3.5-9B", False), "lm")
        generated = [c for c in calls if c[0] == "generate"]
        params = generated[0][2] if generated else None
        check("LM: LM Config -> Qwen LM validates and runs; text and thinking reach the outputs",
              out is not None and out["3"]["text"] == ["A red cube."] and out["4"]["text"] == [""], f"{out}")
        check("LM: the backend gets the official Qwen3.5 rendering and the file from the Qwen-LM folder",
              calls and calls[0][1] == os.path.join(qwen, "Qwen3.5-9B_int8_convrot.safetensors") and calls[0][2] is None
              and generated and generated[0][1] == "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
              "Describe this.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", f"{calls}")
        check("LM: the config's edited fields over the model's defaults, seed 2**64 - 1",
              params is not None and (params.temperature, params.mtp, params.top_p, params.presence_penalty, params.seed,
                                      params.max_new_tokens) == (0.3, "off", 0.8, 1.5, 2 ** 64 - 1, 32768), f"{params}")
        check("LM: the same prompt queued again comes from the cache (the backend is not called again)",
              executed == [] and len(generated) == 1, f"{executed} {len(generated)}")
        out, valid = await run(lm_prompt("Qwen3-VL-8B Instruct", True), "lm-thinking")
        check("LM: thinking on a model without it fails the run", out is None and valid[0] and len(generated) == 1, f"{valid[1]}")
        linked = lm_prompt("Qwen3.5-9B", False)
        linked["5"] = N("PrimitiveFloat", value=0.42)
        linked["1"]["inputs"].update(temperature=["5", 0], edited="")  # a field fed by a link, edited empty
        out, valid = await run(linked, "lm-linked")
        params = [c for c in calls if c[0] == "generate"][-1][2]
        check("LM: a linked LM Config field is applied with edited empty; the other fields keep the model's defaults",
              out is not None and (params.temperature, params.top_p, params.top_k, params.presence_penalty, params.mtp)
              == (0.42, 0.8, 20, 1.5, "auto"), f"{valid[1] if not valid[0] else params}")
    finally:
        registry._FAMILIES[registry.LM_BACKEND]["core"] = real
        folder_paths.folder_names_and_paths["Qwen-LM"][0].remove(qwen)


async def claude_nodes():
    """Claude and Claude Config inside ComfyUI, the SDK client replaced by a recording stand-in (no network, no
    key): the Anthropic key goes through the downloader's token route and only its presence comes back; a prompt
    Claude Config -> Claude with image_1 and image_2 (a slot the node does not declare, as web/js/claude.js adds it)
    validates and runs; the stand-in gets every frame labelled in slot order before the prompt, the config's
    fields, no system field for an empty system prompt and no fallback with on_refusal error; text and thinking
    reach the outputs; queued again with the seed fixed, the answer comes from the cache."""
    import importlib

    import server as comfy_server

    routes = {(r.method, r.path): r.handler for r in comfy_server.PromptServer.instance.routes}
    post = routes.get(("POST", "/bcnodes/downloader/tokens"))

    class KeyRequest:
        async def json(self):
            return {"anthropic": "sk-ant-runtime"}

    body = json.loads((await post(KeyRequest())).body) if post else {}
    check("Claude: the token route stores the Anthropic key and answers only that one is stored",
          body.get("tokens", {}).get("anthropic") is True and "sk-ant-runtime" not in json.dumps(body), f"{body}")

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_Claude"].__module__.rsplit(".", 2)[0]
    claude = importlib.import_module(f"{pkg}.pipelines.claude.messages")
    calls = []
    final = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking="Two colours.", signature="s"),
                 types.SimpleNamespace(type="text", text="A red image and two green ones.")],
        model="claude-haiku-5-5", stop_reason="end_turn", stop_details=None,
        usage=types.SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
                                    cache_creation_input_tokens=0))

    class StandIn:
        request_id = "req_runtime"

        def __init__(self):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self.stream))

        def stream(self, **params):
            calls.append(params)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield types.SimpleNamespace(type="message_start")

        def get_final_message(self):
            return final

    real = claude.make_client
    claude.make_client = lambda config: StandIn()
    try:
        out, executed = await run_twice({
            "1": N("EmptyImage", width=64, height=48, batch_size=1, color=0xFF0000),
            "2": N("EmptyImage", width=32, height=32, batch_size=2, color=0x00FF00),
            "3": N("BC_ClaudeConfig", model_id="", stop_sequences="END", on_refusal="error", speed="standard",
                   image_format="png", timeout=600, max_retries=2),
            "4": N("BC_Claude", prompt="Describe.", model="Haiku 5.5", max_tokens=1000, effort="low", seed=0,
                   system_prompt="", config=["3", 0], image_1=["1", 0], image_2=["2", 0]),
            "5": N("PreviewAny", source=["4", 0]),
            "6": N("PreviewAny", source=["4", 1]),
        }, "claude")
        params = calls[0] if calls else {}
        content = params.get("messages", [{}])[0].get("content", [])
        check("Claude: Claude Config -> Claude with a grown image_2 validates and runs; text and thinking reach the "
              "outputs", out is not None and out["5"]["text"] == ["A red image and two green ones."]
              and out["6"]["text"] == ["Two colours."], f"{out}")
        check("Claude: every frame labelled in slot order before the prompt; the config's fields; no system, no "
              "fallback, no tools",
              [b.get("text") for b in content if b["type"] == "text"] == ["Image 1:", "Image 2:", "Image 3:", "Describe."]
              and [b["type"] for b in content].count("image") == 3 and params.get("model") == "claude-haiku-5-5"
              and params.get("stop_sequences") == ["END"] and params.get("output_config") == {"effort": "low"}
              and not {"system", "fallbacks", "tools", "temperature"} & set(params), f"{str(params)[:400]}")
        check("Claude: the same prompt queued again comes from the cache (no second request)",
              executed == [] and len(calls) == 1, f"{executed} {len(calls)}")
    finally:
        claude.make_client = real


async def claude_chat_routes():
    """Claude Chat's routes from the real PromptServer's route table, served by aiohttp's test server, the SDK client
    replaced by a stand-in (no network): create, list (with the settings' choices), send (the NDJSON events as they
    come: thinking, text, done; the turn saved with the thinking signature), get, update, a bad setting (400), a
    second turn sending the history first, delete."""
    import importlib

    import anthropic.types.beta as bt
    import server as comfy_server
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_Claude"].__module__.rsplit(".", 2)[0]
    messages = importlib.import_module(f"{pkg}.pipelines.claude.messages")
    calls = []

    class StandIn:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self.stream))

        def stream(self, **params):
            calls.append(params)
            return self

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            yield types.SimpleNamespace(type="content_block_delta",
                                        delta=types.SimpleNamespace(type="thinking_delta", thinking="Hmm."))
            yield types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type="text_delta", text="Hi."))

        def get_final_message(self):
            return types.SimpleNamespace(
                content=[bt.BetaThinkingBlock(type="thinking", thinking="Hmm.", signature="sig"),
                         bt.BetaTextBlock(type="text", text="Hi.")],
                model="claude-opus-5-5", stop_reason="end_turn", stop_details=None,
                usage=types.SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                            cache_creation_input_tokens=10))

    app = web.Application()
    app.router.add_routes([r for r in comfy_server.PromptServer.instance.routes if r.path.startswith("/bcnodes/claude/")])
    client = TestClient(TestServer(app))
    await client.start_server()
    real = messages.make_client
    messages.make_client = lambda config: StandIn()
    try:
        created = await (await client.post("/bcnodes/claude/chats", json={"system": "Be brief."})).json()
        sid = created.get("id", "")
        listed = await (await client.get("/bcnodes/claude/chats")).json()
        check("Claude Chat: a session is created and listed with the settings' choices",
              created.get("system") == "Be brief." and [s["id"] for s in listed.get("sessions", [])][:1] == [sid]
              and listed.get("models", [None])[0] == "Opus 5.5" and listed.get("max_tokens") == 128000, f"{listed}")
        res = await client.post(f"/bcnodes/claude/chats/{sid}/send", json={"text": "Hello", "images": []})
        lines = [json.loads(line) for line in (await res.text()).splitlines() if line.strip()]
        check("Claude Chat: a turn streams its events as NDJSON (thinking, text, done)",
              res.headers.get("Content-Type", "").startswith("application/x-ndjson")
              and [e["type"] for e in lines] == ["thinking", "text", "done"] and lines[1]["text"] == "Hi.", f"{lines}")
        saved = await (await client.get(f"/bcnodes/claude/chats/{sid}")).json()
        check("Claude Chat: the turn is saved, the assistant's thinking signature kept, the title from the message",
              saved.get("title") == "Hello" and saved["messages"][1]["content"][0] == {
                  "type": "thinking", "thinking": "Hmm.", "signature": "sig"}, f"{str(saved)[:300]}")
        changed = await (await client.post(f"/bcnodes/claude/chats/{sid}", json={"effort": "low"})).json()
        bad = await client.post(f"/bcnodes/claude/chats/{sid}", json={"effort": "huge"})
        check("Claude Chat: a setting changes; an unknown value is a 400 that names the choices",
              changed.get("effort") == "low" and bad.status == 400 and "no effort" in (await bad.json())["error"],
              f"{changed.get('effort')} {bad.status}")
        # read the whole stream: the turn runs on a worker thread after the response has started
        await (await client.post(f"/bcnodes/claude/chats/{sid}/send", json={"text": "Again", "images": []})).text()
        check("Claude Chat: the second turn sends the saved history first, then the new message",
              len(calls) == 2 and calls[1]["messages"][:2] == saved["messages"]
              and calls[1]["messages"][2]["content"] == [{"type": "text", "text": "Again"}]
              and calls[1]["output_config"] == {"effort": "low"} and calls[1]["cache_control"] == {"type": "ephemeral"},
              f"{str(calls[1:])[:300]}")
        gone = await client.post(f"/bcnodes/claude/chats/{sid}/delete", json={})
        missing = await client.get(f"/bcnodes/claude/chats/{sid}")
        check("Claude Chat: a deleted session is gone (400)", gone.status == 200 and missing.status == 400,
              f"{gone.status} {missing.status}")
    finally:
        messages.make_client = real
        await client.close()


async def preflight_nodes():
    """The PreFlight nodes inside ComfyUI, the "core" LM backend replaced by a recording stand-in that answers
    with an observation (no weights): EmptyImage + LM Loader -> PreFlight Observe -> PreFlight Report -> Show Text
    validates and runs, the backend gets the frames then the prompt and greedy params (seed 42, 300 tokens) and
    unloads once with the loader's keep_model_loaded off, the prediction lands in
    <user dir>/BCNodes/preflight/feedback.jsonl (its observation stamped with the loader's model), the
    record list goes to the browsers (bcnodes.preflight.records) and GET /bcnodes/preflight/records answers it,
    the image passes through; the same prompt queued again comes from the cache (no second model run, nothing
    logged again). PreFlight Outcome logs on every queue, by the list's label and by record_id_override with a
    label the list does not hold (it validates). PreFlight Calibrate -> Show Text runs once, comes from the cache
    while the store is unchanged and runs again after an outcome is logged."""
    import importlib

    import folder_paths
    import server as comfy_server

    pkg = nodes.NODE_CLASS_MAPPINGS["BC_PreFlightObserve"].__module__.rsplit(".", 2)[0]
    registry = importlib.import_module(f"{pkg}.models.common.registry")
    observe = importlib.import_module(f"{pkg}.pipelines.preflight.observe")
    store = os.path.join(folder_paths.get_user_directory(), "BCNodes", "preflight", "feedback.jsonl")
    answer = json.dumps({"subject_appears_under_18": False, "garment": "bikini", "setting": "beach_pool",
                         "framing": "full_body", "pose": "neutral", "see_through_or_wet": False, "exposure": "mild",
                         "nudity_or_sexual_act": False, "motion_flags": "none", "visible_text": "", "confidence": "high"})

    qwen = tempfile.mkdtemp(prefix="bcnodes_preflight_")
    open(os.path.join(qwen, "Qwen3.5-9B_int8_convrot.safetensors"), "wb").close()
    folder_paths.add_model_folder_path("Qwen-LM", qwen, is_default=True)
    calls = []

    class StandIn:
        name = "core"

        def load(self, path, lora):
            calls.append(("load", path, lora))
            return "handle"

        def generate(self, handle, prompt, images, params):
            calls.append(("generate", prompt, images, params))
            return answer

        def unload(self):
            calls.append(("unload",))
            return {}

    ps = comfy_server.PromptServer.instance
    sent = []
    real_send = ps.send_sync

    def send_sync(event, data, sid=None):
        if event == "bcnodes.preflight.records":
            sent.append(data)
        else:
            real_send(event, data, sid)

    async def queue(ex, server, prompt):
        """One queue on `ex`: (outputs or None, node ids executed, validation tuple)."""
        prompt_id = str(uuid.uuid4())
        valid = await execution.validate_prompt(prompt_id, prompt, None)
        if not valid[0]:
            return None, None, valid
        server.events.clear()
        await ex.execute_async(copy.deepcopy(prompt), prompt_id, {"client_id": "test"}, valid[2])
        if not ex.success:
            errors = [m for m in ex.status_messages if m[0] == "execution_error"]
            print(f"  [preflight] execution failed: {json.dumps(errors, default=str)[:600]}")
            return None, None, valid
        return ex.history_result["outputs"], executed_nodes(server), valid

    def executor():
        server = Server()
        return execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
                                         cache_args={"lru": 0, "ram": 0, "ram_inactive": 0}), server

    def lines():
        with open(store, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    real = registry.get(registry.LM_BACKEND, "core")
    registry._FAMILIES[registry.LM_BACKEND]["core"] = StandIn()
    ps.send_sync = send_sync
    try:
        flow = {
            "1": N("EmptyImage", width=96, height=64, batch_size=3, color=0x808080),
            "8": N("BC_LMLoader", family="Qwen LM", model="Qwen3.5-9B", precision="INT8 ConvRot", lora="None",
                   lora_strength=1.0, keep_model_loaded=False),
            "2": N("BC_PreFlightObserve", image=["1", 0], lm=["8", 0], max_frames=6),
            "3": N("BC_PreFlightReport", observations_json=["2", 0], caption="summer drop", log_prediction=True,
                   image=["1", 0]),
            "4": N("BC_ShowText", text=["3", 1]),
            "5": N("PreviewAny", source=["3", 2]),
            "6": N("BC_MathExpression", expression="a.width", a=["3", 3]),
            "7": N("PreviewAny", source=["6", 0]),
        }
        ex, server = executor()
        out, executed, valid = await queue(ex, server, flow)
        rid = out["5"]["text"][0] if out else ""
        summary = out["4"]["text"][0] if out else ""
        check("PreFlight: Observe -> Report -> Show Text validates and runs; the summary holds the verdicts and the record",
              out is not None and len(rid) == 8 and summary.startswith("Instagram:") and summary.endswith("record: " + rid)
              and out["6"]["value"] == [96], f"{valid[1] if not valid[0] else out}")
        generated = [c for c in calls if c[0] == "generate"]
        params = generated[0][3] if generated else None
        check("PreFlight: the backend gets the 3 frames then the prompt (no system turn, thinking off), greedy, seed 42, "
              "300 tokens, and unloads once (the loader's keep_model_loaded off)",
              len(generated) == 1 and generated[0][2].shape[0] == 3
              and generated[0][1] == ("<|im_start|>user\n" + "<|vision_start|><|image_pad|><|vision_end|>" * 3
                                      + observe.prompts.OBSERVATION_PROMPT
                                      + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
              and (params.do_sample, params.presence_penalty, params.repetition_penalty, params.seed, params.max_new_tokens)
              == (False, 0.0, 1.0, 42, 300) and [c[0] for c in calls] == ["load", "generate", "unload"],
              f"{[c[0] for c in calls]} {params}")
        check("PreFlight: the prediction is in <user dir>/BCNodes/preflight/feedback.jsonl, its observation stamped",
              os.path.isfile(store) and [r["id"] for r in lines()] == [rid]
              and {k: lines()[0]["observations"]["meta"][k] for k in ("model_name", "precision", "lora")}
              == {"model_name": "Qwen3.5-9B", "precision": "INT8 ConvRot", "lora": "None"}, f"{store}")
        check("PreFlight: the record list goes to the browsers once, the new prediction first",
              len(sent) == 1 and sent[0]["records"][0].startswith(rid + " "), f"{sent}")
        handler = next((r.handler for r in ps.routes if r.method == "GET" and r.path == "/bcnodes/preflight/records"), None)
        body = json.loads((await handler(None)).body) if handler else {}
        check("PreFlight: GET /bcnodes/preflight/records answers the same list", body == sent[0] if sent else False, f"{body}")
        out, executed, _ = await queue(ex, server, flow)
        check("PreFlight: the same prompt queued again comes from the cache (no model run, nothing logged again)",
              out is not None and executed == [] and len([c for c in calls if c[0] == "generate"]) == 1 and len(lines()) == 1,
              f"{executed} {len(lines())}")

        label = body.get("records", [""])[0]
        outcome = {"1": N("BC_PreFlightOutcome", record=label, platform="tiktok", result="removed", record_id_override=""),
                   "2": N("PreviewAny", source=["1", 0])}
        out, executed = await run_twice(outcome, "preflight-outcome")
        check("PreFlight: Outcome logs the list's record, on every queue",
              out is not None and out["2"]["text"] == [f"logged: {rid} tiktok=removed"] and "1" in (executed or [])
              and [r["type"] for r in lines()] == ["prediction", "outcome", "outcome"], f"{out} {executed}")
        stale = {"1": N("BC_PreFlightOutcome", record="ffff0000 · a label the list does not hold", platform="x",
                        result="clean", record_id_override=rid),
                 "2": N("PreviewAny", source=["1", 0])}
        out, valid = await run(stale, "preflight-outcome-stale")
        check("PreFlight: a record label the list does not hold validates; record_id_override wins",
              out is not None and out["2"]["text"] == [f"logged: {rid} x=clean"], f"{valid[1] if not valid[0] else out}")

        calibrate = {"1": N("BC_PreFlightCalibrate"), "2": N("BC_ShowText", text=["1", 0])}
        ex, server = executor()
        out, executed, valid = await queue(ex, server, calibrate)
        text = out["2"]["text"][0] if out else ""
        check("PreFlight: Calibrate -> Show Text gives the calibration tables of the store",
              "predictions: 1   with at least one outcome: 1" in text and "base.bikini" in text and executed == ["1", "2"],
              f"{valid[1] if not valid[0] else text[:200]} {executed}")
        out, executed, _ = await queue(ex, server, calibrate)
        cached = executed
        await run(outcome, "preflight-outcome-again")
        out, executed, _ = await queue(ex, server, calibrate)
        check("PreFlight: Calibrate comes from the cache while the store is unchanged and runs again after an outcome",
              cached == [] and executed == ["1", "2"], f"{cached} {executed}")
    finally:
        registry._FAMILIES[registry.LM_BACKEND]["core"] = real
        del ps.send_sync
        folder_paths.folder_names_and_paths["Qwen-LM"][0].remove(qwen)


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

    # the pack's MONITOR was stopped by process_monitor_default: off, it keeps its hook and its last sample
    arm = handlers[("POST", "/bcnodes/monitor/arm")]
    response = await arm(types.SimpleNamespace(json=lambda: asyncio.sleep(0, {"armed": True})))  # the route reads only its body
    monitor = pack_module(".nodes.process_monitor").MONITOR
    check("ProcessMonitor: arming is refused while the monitor is off, saying to turn it on",
          response.status == 409 and "The monitor is off" in json.loads(response.text).get("error", "")
          and not monitor.armed_next, f"{response.status} {response.text}")
    live = handlers[("GET", "/bcnodes/monitor/live")]
    body = json.loads((await live(make_mocked_request("GET", "/bcnodes/monitor/live"))).text)
    check("ProcessMonitor: the live route has no sample while the monitor is off", body == {"enabled": False, "sample": None}, f"{body}")


if __name__ == "__main__":
    asyncio.run(main())

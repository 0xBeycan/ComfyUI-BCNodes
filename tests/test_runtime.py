"""Backend nodes through ComfyUI's real prompt validation and executor, headless.

Needs a ComfyUI checkout whose requirements are installed in the current
Python. It is found through $COMFYUI_DIR or ../ComfyUI next to this pack.
No server is started and no browser is needed; only core CPU nodes are used.

    COMFYUI_DIR=/path/to/ComfyUI python tests/test_runtime.py
"""

import asyncio
import copy
import json
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
    # Output nodes write files; keep them out of the checkout's output/ and temp/.
    import folder_paths
    scratch = tempfile.mkdtemp(prefix="bcnodes_runtime_")
    folder_paths.set_output_directory(os.path.join(scratch, "output"))
    folder_paths.set_temp_directory(os.path.join(scratch, "temp"))
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    assert await nodes.load_custom_node(PACK), "pack failed to load"
    print("loaded:", ", ".join(sorted(k for k in nodes.NODE_CLASS_MAPPINGS if k.startswith("BC_"))))


async def run(prompt, label, extra_data=None):
    """-> (outputs by node id or None, validation tuple)"""
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        return None, valid
    ex = execution.PromptExecutor(Server(), cache_type=execution.CacheType.CLASSIC, cache_args={"lru": 0, "ram": 0, "ram_inactive": 0})
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
        executed = sorted(data["node"] for event, data in server.events if event == "executing" and data.get("node") is not None)
        if i == 0:
            outputs = ex.history_result["outputs"]
            if not executed:
                print(f"  [{label}] run 1 executed nothing; the executor is not reporting")
                return None, None
    return outputs, executed


async def main():
    await setup()

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

    # Video Comparer: both batches in ONE mp4 in temp, A | B side by side, cut
    # to the shorter clip; the info tells the widget how to split it.
    out, _ = await run({
        "1": N("EmptyImage", width=65, height=33, batch_size=6, color=0),   # odd sizes get trimmed to even
        "2": N("EmptyImage", width=64, height=32, batch_size=4, color=16777215),
        "3": N("BC_VideoComparer", fps=12.0, video_a=["1", 0], video_b=["2", 0]),
        "4": N("BC_VideoComparer", fps=12.0, video_b=["2", 0]),
    }, "videocomparer")
    info = out["3"]["bc_video"][0] if out and out["3"]["bc_video"] else None
    ok = info is not None and info["sides"] == ["A", "B"] and info["frames"] == {"A": 6, "B": 4} and info["fps"] == 12.0 and info["audio"] is False
    if ok:
        import av
        import folder_paths
        with av.open(os.path.join(folder_paths.get_temp_directory(), info["filename"])) as clip:
            v = clip.streams.video[0]
            ok = (v.width, v.height, v.frames) == (128, 32, 4) and not clip.streams.audio
    check("VideoComparer: A | B -> one 128x32 mp4, 4 frames, no audio", ok)
    check("VideoComparer: one side only", out is not None and out["4"]["bc_video"][0]["sides"] == ["B"])

    # Audio is muxed in as AAC and cut to the clip; an unequal size is letterboxed.
    import torch
    vc = sys.modules[nodes.NODE_CLASS_MAPPINGS["BC_VideoComparer"].__module__]
    audio = {"waveform": torch.zeros((1, 2, 48000)), "sample_rate": 16000}
    info = vc._encode([("A", torch.zeros((12, 8, 16, 3))), ("B", torch.ones((12, 16, 32, 3)))], 12.0, audio)
    with av.open(os.path.join(folder_paths.get_temp_directory(), info["filename"])) as clip:
        v, a = clip.streams.video[0], clip.streams.audio[0]
        ok = info["audio"] is True and (v.width, v.height, v.frames) == (64, 16, 12) and a.codec_context.name == "aac" and a.sample_rate == 16000 and abs(float(a.duration * a.time_base) - 1.0) < 0.15
    check("VideoComparer: audio -> AAC track, cut to the clip; smaller side letterboxed", ok)

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
        "ShowText": {"1": N("BC_PromptList", prepend_text="", multiline_text="x\ny", append_text="", start_index=0, max_rows=10), "2": N("BC_ShowText", text=["1", 0]), "3": N("PreviewAny", source=["2", 0])},
        "ImageComparer": {"1": image, "2": N("BC_ImageComparer", image_a=["1", 0], image_b=["1", 0])},
        "VideoComparer": {"1": N("EmptyImage", width=16, height=8, batch_size=4, color=0), "2": N("BC_VideoComparer", fps=12.0, video_a=["1", 0])},
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
    dataset = tempfile.mkdtemp(prefix="bcnodes_captions_")
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

    # Auto Model Downloader answers NaN on purpose: whether a file exists is
    # decided on disk, not by the inputs. It has no outputs, so nothing below
    # it can be dragged along.
    out, executed = await run_twice({"1": N("BC_AutoModelDownloader", entries="[]")}, "cache-downloader")
    check("AutoModelDownloader: runs every queue by design", out is not None and executed == ["1"], f"{executed}")

    print(f"\n{results['pass']} passed, {results['fail']} failed")
    sys.exit(1 if results["fail"] else 0)


if __name__ == "__main__":
    asyncio.run(main())

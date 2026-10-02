"""The monitor controller and the execution.py hook (pipelines/process_monitor/{monitor,hook}.py)
against fakes: a fake execution module with ComfyUI's execute signature, fake caches and execution
list, a fake RAM reader and a fake prompt queue. The real ComfyUI path runs in tests/test_runtime.py.
"""

import asyncio
import enum
import sys
import threading
import time
import types
import weakref
from collections import namedtuple

import pytest
import torch

Entry = namedtuple("CacheEntry", "ui outputs")


class Result(enum.Enum):
    SUCCESS = 0
    FAILURE = 1
    PENDING = 2


class FakeRam:
    kind, version, limit = "fake", None, 1000

    def __init__(self, value=100, kills=0):
        self.value, self.kills = value, kills

    def describe(self):
        return "fake RAM"

    def read(self):
        return {"ram": self.value, "ram_limit": self.limit}

    def oom_kills(self):
        return self.kills

    def open_peak_window(self):
        return None


class Probe:
    def __init__(self):
        self.prompt, self.pushed = None, []

    def running(self):
        return (self.prompt, {"extra_pnginfo": {"workflow": {"id": "wf-1"}}}) if self.prompt else None

    def status(self, prompt_id):
        return "success"

    def last_node(self):
        return None

    def push(self, payload):
        self.pushed.append(payload)


class Cache:
    def __init__(self):
        self.cache, self.subcaches = {}, {}

    def get_local(self, uid):
        return self.cache.get(uid)


class Dyn:
    def __init__(self, prompt):
        self.prompt = prompt

    def get_node(self, uid):
        return self.prompt[uid]

    def get_original_prompt(self):
        return self.prompt

    def get_display_node_id(self, uid):
        return uid


PROMPT = {"1": {"class_type": "Load", "inputs": {}}, "2": {"class_type": "Grow", "inputs": {"image": ["1", 0], "n": 3}}}


def fake_execution(behaviour):
    """A module whose execute has ComfyUI's signature; behaviour(uid, caches, executed) -> Result."""
    async def execute(server, dynprompt, caches, current_item, extra_data, executed, prompt_id, execution_list,
                      pending_subgraph_results, pending_async_nodes, ui_outputs, asset_manager):
        return (behaviour(current_item, caches, executed, execution_list), None, None)
    return types.SimpleNamespace(execute=execute)


def produce(uid, caches, executed, execution_list):
    """Node 1 makes a 2 x 4 x 4 x 3 image; node 2 reads it and makes one twice as tall."""
    out = torch.zeros(2, 4 if uid == "1" else 8, 4, 3)
    entry = Entry(None, [[out]])
    caches.outputs.cache[uid] = entry
    execution_list.execution_cache.setdefault("2", {})[uid] = entry
    executed.add(uid)
    return Result.SUCCESS


def call_execute(module, uid, caches, executed, execution_list, prompt_id="p1"):
    extra = {"extra_pnginfo": {"workflow": {"id": "wf-1"}}}
    return asyncio.run(module.execute(None, Dyn(PROMPT), caches, uid, extra, executed, prompt_id, execution_list, {}, {}, {}, None))


@pytest.fixture
def mon(bcnodes, monkeypatch):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "current_loaded_models", [], raising=False)  # ComfyUI's list; the stub lacks it
    return bcnodes["pipelines.process_monitor.monitor"]


@pytest.fixture
def hk(bcnodes):
    return bcnodes["pipelines.process_monitor.hook"]


def make_monitor(mon, tmp_path, ram=None):
    m = mon.Monitor(str(tmp_path), Probe(), platform="linux")
    m.ram, m.gpu = ram or FakeRam(), None
    return m


def records(bcnodes, tmp_path):
    bb = bcnodes["pipelines.process_monitor.blackbox"]
    return bb.read_records(bb.run_files(str(tmp_path / "runs"))[0])


def test_hook_find_needs_the_coroutine_and_get_local(hk, monkeypatch):
    module, reason = hk.find()
    assert module is None and "cannot be imported" in reason  # no ComfyUI under the stubs
    caching = types.ModuleType("comfy_execution.caching")
    caching.BasicCache = type("BasicCache", (), {"get_local": lambda self, uid: None})
    monkeypatch.setitem(sys.modules, "comfy_execution", types.ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.caching", caching)
    good = fake_execution(produce)
    monkeypatch.setitem(sys.modules, "execution", good)
    assert hk.find() == (good, None)
    monkeypatch.setitem(sys.modules, "execution", types.SimpleNamespace(execute=lambda server, dynprompt: None))
    assert "not a coroutine" in hk.find()[1]

    async def short(server, dynprompt, caches, current_item):
        return None
    monkeypatch.setitem(sys.modules, "execution", types.SimpleNamespace(execute=short))
    assert "extra_data, executed, prompt_id, execution_list" in hk.find()[1]
    caching.BasicCache = type("BasicCache", (), {})
    monkeypatch.setitem(sys.modules, "execution", good)
    assert "get_local" in hk.find()[1]


def test_hook_installs_passes_results_through_and_uninstalls(hk):
    module = fake_execution(produce)
    original = module.execute
    seen = []

    class Recorder:
        def node_start(self, call):
            seen.append(("start", call.current_item, call.prompt_id))
            return call.current_item

        def node_end(self, token, call, result, seconds):
            seen.append(("end", token, result[0].name))

    hook = hk.Hook(module, Recorder())
    hook.install()
    assert module.execute is hook.wrapper
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    assert call_execute(module, "1", caches, executed, el)[0] is Result.SUCCESS
    assert seen == [("start", "1", "p1"), ("end", "1", "SUCCESS")] and hook.seconds > 0
    hook.uninstall()
    assert module.execute is original


def test_hook_error_stops_measurement_never_the_node(hk, caplog):
    module = fake_execution(produce)

    class Broken:
        def node_start(self, call):
            raise RuntimeError("boom")

    hook = hk.Hook(module, Broken())
    hook.install()
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    assert call_execute(module, "1", caches, executed, el)[0] is Result.SUCCESS and "1" in executed
    assert not hook.active and "per-node measurement stopped" in hook.error and "boom" in hook.error
    assert "boom" in caplog.text


def test_hook_stays_in_a_later_wrapper_chain(hk):
    module = fake_execution(produce)
    hook = hk.Hook(module, object())
    hook.install()
    later = hook.wrapper

    async def other_pack(*a, **k):
        return await later(*a, **k)
    module.execute = other_pack
    hook.uninstall()
    assert module.execute is other_pack and not hook.active  # a pass-through inside the other pack's chain


def test_armed_run_records_nodes(mon, hk, bcnodes, tmp_path):
    m = make_monitor(mon, tmp_path)
    module = fake_execution(produce)
    m.hook = hk.Hook(module, m)
    m.hook.install()
    m.arm(True)
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    m.probe.prompt = "p1"
    call_execute(module, "1", caches, executed, el)
    m.ram.value = 300
    call_execute(module, "2", caches, executed, el)
    with m._lock:
        m._end_run("success")
    recs = records(bcnodes, tmp_path)
    kinds = [r["type"] for r in recs]
    assert kinds == ["start", "cached", "node", "node_end", "node", "node_end", "end"]
    start, n2, end2 = recs[0], recs[4], recs[5]
    assert recs[1]["nodes"] == []  # nothing was in the cache
    assert start["armed"] and start["workflow_id"] == "wf-1" and start["oom_kills"] == 0
    assert n2["inputs"] == [{"name": "image", "bytes": 2 * 4 * 4 * 3 * 4, "count": 1,
                             "tensors": [{"shape": [2, 4, 4, 3], "dtype": "float32", "device": "cpu", "bytes": 384}]}]
    assert n2["cache"] == 384 and n2["ram"] == 300
    assert end2["state"] == "executed" and end2["output_bytes"] == 768 and end2["cache"] == 384 + 768
    assert end2["peak_source"].startswith("sampler maximum") and end2["ram_peak"] == 300
    assert recs[-1]["monitor"]["hook_s"] > 0
    assert m.measurement("wf-1")["nodes"]["2"]["output_bytes"] == 768  # kept for Emulate's calibration
    assert m.measurement("../settings") is None and m.measurement(None) is None  # an id is a plain token, never a path
    assert not m.armed_next  # one run only


def test_cached_and_pending_nodes(mon, hk, bcnodes, tmp_path):
    m = make_monitor(mon, tmp_path)
    calls = {"n": 0}

    def behaviour(uid, caches, executed, el):
        if uid == "1":  # served from the cache: SUCCESS without being executed
            caches.outputs.cache["1"] = Entry(None, [[torch.zeros(3)]])
            return Result.SUCCESS
        calls["n"] += 1
        if calls["n"] == 1:
            return Result.PENDING
        return produce(uid, caches, executed, el)

    module = fake_execution(behaviour)
    m.hook = hk.Hook(module, m)
    m.hook.install()
    m.arm(True)
    m.probe.prompt = "p1"
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    for uid in ("1", "2", "2"):
        call_execute(module, uid, caches, executed, el)
    with m._lock:
        m._end_run("success")
    ends = [r for r in records(bcnodes, tmp_path) if r["type"] == "node_end"]
    assert [(e["node"], e["state"]) for e in ends] == [("1", "cached"), ("2", "executed")]
    assert "seconds" not in ends[0] and ends[0]["output_bytes"] == 12  # from cache, not measured


def test_sampler_runs_and_threshold_snapshot(mon, bcnodes, tmp_path, monkeypatch):
    import comfy.model_management as mm

    stopped = []
    monkeypatch.setattr(mm, "interrupt_current_processing", lambda value=True: stopped.append(value), raising=False)
    m = make_monitor(mon, tmp_path)
    m.update_settings({"threshold": 0.85, "stop_at_threshold": True})
    m.probe.prompt = "p1"
    m._tick(live=True)  # starts the run, logs a sample, pushes the live bars
    m.ram.value = 900  # crosses 85 % of 1000
    m._tick(live=False)
    m._tick(live=False)  # once per run
    m.probe.prompt = None
    m._tick(live=False)  # the queue is empty: the run ends with the queue's status
    recs = records(bcnodes, tmp_path)
    assert [r["type"] for r in recs] == ["start", "sample", "sample", "snapshot", "snapshot", "sample", "end"]
    quick, full = recs[3], recs[4]
    assert "census" not in quick and quick["stopped"]  # the stack record, written before the census
    assert full["ram"] == 900 and full["stopped"] and stopped == [True] and isinstance(full["census"]["groups"], list)
    assert recs[-1]["status"] == "success" and m.run is None
    assert m.probe.pushed[0]["run"]["prompt_id"] == "p1" and m.probe.pushed[0]["ram"] == 100


def test_samples_carry_the_execution_threads_line(mon, tmp_path):
    m = make_monitor(mon, tmp_path)
    ready, done = threading.Event(), threading.Event()

    def node_code():  # stands in for a node's function: this file is not in site-packages
        ready.set()
        done.wait(5)

    t = threading.Thread(target=node_code)
    t.start()
    ready.wait(5)
    m.probe.prompt = "p1"
    with m._lock:
        m._start_run("p1", {})
    m.exec_thread = t.ident  # what the hook records at node start
    sample = m._sample(m.run)
    done.set()
    t.join()
    assert "test_pipe_process_monitor_monitor.py" in sample["line"] and sample["line"].endswith("node_code")
    assert "threading.py" in sample["top"]  # the innermost frame is the stdlib's Event.wait


def test_snapshot_keeps_no_local_of_the_execution_thread_alive(mon, bcnodes, tmp_path):
    """Reading a running frame's locals from another thread leaves a copy of them on the frame: a
    tensor the node then deletes would stay alive until its function returns. The snapshot reads the
    stack from code and line numbers only, written before the census."""
    m = make_monitor(mon, tmp_path)
    ready, snapped, freed = threading.Event(), threading.Event(), []

    def node_code():
        stacked = torch.zeros(16, 1024, 1024, dtype=torch.uint8)  # 16 MiB
        ref = weakref.ref(stacked)
        ready.set()
        snapped.wait(5)
        del stacked
        freed.append(ref() is None)

    t = threading.Thread(target=node_code)
    t.start()
    ready.wait(5)
    try:
        m.probe.prompt = "p1"
        with m._lock:
            m._start_run("p1", {})
        m.exec_thread = t.ident  # what the hook records at node start
        m.update_settings({"threshold": 0.5})
        m.ram.value = 600
        m._tick(live=False)
    finally:
        snapped.set()
        t.join()
    assert freed == [True]
    stack, full = [r for r in records(bcnodes, tmp_path) if r["type"] == "snapshot"]
    assert "census" not in stack and stack["line"].endswith("node_code") and any("in node_code" in line for line in stack["stack"])
    assert [g["count"] for g in full["census"]["groups"] if g["shape"] == [16, 1024, 1024]] == [1]


def fill_tuples(stop, errors):
    """tuple(genexpr) whose items release the GIL, as ComfyUI's LoRA weight prefetch does around a copy."""
    def item(i):
        time.sleep(0)
        return i
    while not stop.is_set():
        try:
            tuple(item(i) for i in range(20))
        except SystemError as e:
            errors.append(e)


def test_threshold_snapshot_races_no_tuple_builder(mon, bcnodes, tmp_path):
    """A sampler died of the snapshot: its pass over all objects held a reference to a tuple the
    execution thread was still filling, and the tuple's resize failed (SystemError: bad argument to
    internal function). The monitor's snapshot runs on a background thread, as in ComfyUI, while two
    threads fill tuples from generators: one stands in for the execution thread, one for any other."""
    m = make_monitor(mon, tmp_path)
    m.probe.prompt = "p1"
    with m._lock:
        run = m._start_run("p1", {})
    held = [torch.zeros(16, 1024, 257) for _ in range(3)]  # a shape of its own, 16 MiB each
    stop, errors = threading.Event(), []
    builders = [threading.Thread(target=fill_tuples, args=(stop, errors)) for _ in range(2)]
    for t in builders:
        t.start()
    m.exec_thread = builders[0].ident
    sample = {"ram": 900, "ram_limit": 1000, "node": "1", "class_type": "Grow", "line": None}
    monitor = threading.Thread(target=lambda: [m._snapshot(run, sample) for _ in range(8)])
    try:
        monitor.start()
        monitor.join()
    finally:
        stop.set()
        for t in builders:
            t.join()
    assert errors == []
    full = [r for r in records(bcnodes, tmp_path) if r["type"] == "snapshot"][-1]
    assert [g["count"] for g in full["census"]["groups"] if g["shape"] == [16, 1024, 257]] == [3]
    del held


def test_crash_after_restart(mon, bcnodes, tmp_path):
    first = make_monitor(mon, tmp_path, FakeRam(kills=0))
    first.probe.prompt = "p-dead"
    first._tick(live=False)
    first.ram.value = 950
    first._tick(live=False)  # killed here: no end record
    assert first.crash_flag() is None  # its own running run is no crash
    second = make_monitor(mon, tmp_path, FakeRam(kills=1))  # a new process: another session
    assert second.crash_flag() == "p-dead"
    report = second.crash()
    assert report["cause"]["kind"] == "oom_kill" and report["ram_last"] == 950 and report["snapshot"]["ram"] == 950
    assert second.last_run() is None


def test_toggle_installs_and_removes_everything(mon, hk, tmp_path, monkeypatch):
    module = fake_execution(produce)
    original = module.execute
    monkeypatch.setattr(mon, "find", lambda: (module, None))
    m = mon.Monitor(str(tmp_path / "pm"), Probe())
    assert m.settings.enabled and not m.enabled  # on by default, but only start() starts it
    assert not (tmp_path / "pm").exists()  # not started: nothing written
    m.set_enabled(True)
    try:
        assert m.enabled and module.execute is m.hook.wrapper and m._thread.is_alive()
        assert m.status()["hook"]["available"] and m.status()["settings"]["enabled"]
    finally:
        m.set_enabled(False)
    assert not m.enabled and module.execute is original and m._thread is None
    assert mon.Monitor(str(tmp_path / "pm"), Probe()).settings.enabled is False  # persisted


def test_no_hook_point_disables_per_node_measurement_with_a_message(mon, tmp_path, monkeypatch):
    monkeypatch.setattr(mon, "find", lambda: (None, "execution.execute is missing or not a coroutine"))
    m = mon.Monitor(str(tmp_path), Probe())
    m.set_enabled(True)
    try:
        with pytest.raises(RuntimeError, match="needs ComfyUI 0.17.0 or newer: execution.execute is missing"):
            m.arm(True)
        assert m.status()["hook"]["available"] is False
    finally:
        m.set_enabled(False)

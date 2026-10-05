"""The monitor controller and the execution.py hook (pipelines/process_monitor/{monitor,hook}.py)
against fakes: a fake execution module with ComfyUI's execute signature, fake caches and execution
list, a fake RAM reader and a fake prompt queue. The real ComfyUI path runs in tests/test_runtime.py.
"""

import asyncio
import enum
import gc
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

    def __init__(self, value=100, kills=0, container="boot:1"):
        self.value, self.kills, self.container = value, kills, container

    def describe(self):
        return "fake RAM"

    def read(self):
        return {"ram": self.value, "ram_limit": self.limit}

    def oom_kills(self):
        return self.kills

    def identity(self):
        return self.container

    def open_peak_window(self):
        return None


class RawPeak:
    """memory.peak through a descriptor: usage with page cache, above the working set."""

    def __init__(self, raw):
        self.raw = raw

    def close(self):
        return self.raw


class FakeGpu:
    """torch's CUDA peak counter: max since the last reset."""

    def __init__(self):
        self.now, self.top = 0, 0

    def read(self):
        return {"vram": self.now}

    def alloc(self, n):
        self.now = n
        self.top = max(self.top, n)

    def reset_peak(self):
        self.top = self.now

    def peak(self):
        return {"vram_peak": self.top}


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


def test_node_ram_peak_is_the_working_set_and_memory_peak_stays_apart(mon, hk, bcnodes, tmp_path):
    """ram_start, ram_end and the sampled RAM are the working set; cgroup memory.peak also counts page cache,
    so the node's ram_peak is the working set's peak and memory.peak is ram_peak_raw. Emulate's transient
    (ram_peak - ram_start - outputs) then counts no page cache."""
    ram = FakeRam(100)
    ram.open_peak_window = lambda: RawPeak(5000)  # files read during the node filled the page cache
    m = make_monitor(mon, tmp_path, ram)

    def grows(uid, caches, executed, execution_list):
        ram.value = 400
        m._tick(live=False)  # a sample during the node
        ram.value = 200
        return produce(uid, caches, executed, execution_list)
    module = fake_execution(grows)
    m.hook = hk.Hook(module, m)
    m.hook.install()
    m.arm(True)
    m.probe.prompt = "p1"
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    call_execute(module, "1", caches, executed, el)
    with m._lock:
        m._end_run("success")
    end = [r for r in records(bcnodes, tmp_path) if r["type"] == "node_end"][0]
    assert (end["ram_start"], end["ram_peak"], end["ram_end"], end["ram_peak_raw"]) == (100, 400, 200, 5000)
    assert end["peak_source"].startswith("sampler maximum")


def test_run_vram_peak_is_torchs_peak_across_the_node_resets(mon, hk, bcnodes, tmp_path):
    """An armed run resets torch's peak at every node start for the node's own peak; the run's peak keeps
    each node's and what came between."""
    m = make_monitor(mon, tmp_path)
    m.gpu = gpu = FakeGpu()
    peaks = {"1": 700, "2": 300}

    def spikes(uid, caches, executed, execution_list):
        gpu.alloc(peaks[uid])
        gpu.alloc(50)  # freed before the sampler saw it
        return produce(uid, caches, executed, execution_list)
    module = fake_execution(spikes)
    m.hook = hk.Hook(module, m)
    m.hook.install()
    m.arm(True)
    m.probe.prompt = "p1"
    caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
    gpu.alloc(900)  # before the run: not the run's
    gpu.alloc(10)
    call_execute(module, "1", caches, executed, el)
    gpu.alloc(800)  # between the nodes: before node 2's reset
    gpu.alloc(50)
    call_execute(module, "2", caches, executed, el)
    with m._lock:
        m._end_run("success")
    recs = records(bcnodes, tmp_path)
    assert [r["vram_peak"] for r in recs if r["type"] == "node_end"] == [700, 300]
    assert recs[-1]["type"] == "end" and recs[-1]["vram_peak"] == 800
    bb = bcnodes["pipelines.process_monitor.blackbox"]
    assert bb.run_report(recs)["vram_peak"] == 800


def test_a_sampler_error_ends_the_run_with_its_end_record(mon, hk, bcnodes, tmp_path, monkeypatch, caplog):
    """The sampler thread that dies of an error writes the run's end record first: a run without one
    reads as a crash at the next start. The hook stops too: no run starts without the sampler."""
    module = fake_execution(produce)
    ram = FakeRam(100)
    monkeypatch.setattr(mon, "PERIOD", 0.001)
    monkeypatch.setattr(mon, "find", lambda: (module, None))
    monkeypatch.setattr(mon.memory_sources, "ram_source", lambda: ram)
    monkeypatch.setattr(mon.memory_sources, "gpu_source", lambda: None)
    m = mon.Monitor(str(tmp_path), TickingProbe(), platform="linux")
    m.start()
    try:
        m.probe.prompt = "p1"
        m.probe.sampled()
        assert m.run is not None

        def broken():
            raise OSError("cgroup file gone")
        ram.read = broken
        m._thread.join(5)
        assert not m._thread.is_alive() and "cgroup file gone" in m.error
    finally:
        m.stop()
    end = records(bcnodes, tmp_path)[-1]
    assert end["type"] == "end" and end["status"].startswith("monitor stopped during the run")
    assert module.execute is not m.hook.wrapper and m.run is None


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


class TickingProbe(Probe):
    """Counts the sampler's ticks: each one calls running() first."""

    def __init__(self):
        super().__init__()
        self.ticks = 0

    def running(self):
        self.ticks += 1
        return super().running()

    def sampled(self, timeout=5.0):
        """Returns once a whole tick ran after this call: the sampler read the caller's stack."""
        end, deadline = self.ticks + 2, time.monotonic() + timeout
        while self.ticks < end:
            if time.monotonic() > deadline:
                raise TimeoutError("the monitor's sampler did not tick")
            time.sleep(0.0005)


def run_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch, node, ram, hooked=True):
    """Runs node(probe) as ComfyUI runs a node, under the whole monitor: inside execute through the
    hook, the run armed (hooked; else as on a ComfyUI without the hook point: the sampler looks for the
    execution thread on every thread's stack at every tick), the black box on, the sampler thread
    ticking every millisecond. The garbage collector is paused, as it practically is in a GPU loop that
    allocates as much as it frees: only reference counting frees. ram is a FakeRam or its value. Returns
    the run's records."""
    def behaviour(uid, caches, executed, execution_list):
        node(probe)
        executed.add(uid)
        return Result.SUCCESS

    module, probe = fake_execution(behaviour), TickingProbe()
    monkeypatch.setattr(mon, "PERIOD", 0.001)
    monkeypatch.setattr(mon, "find", lambda: (module, None) if hooked else (None, "no hook point"))
    ram = ram if isinstance(ram, FakeRam) else FakeRam(ram)
    monkeypatch.setattr(mon.memory_sources, "ram_source", lambda: ram)
    monkeypatch.setattr(mon.memory_sources, "gpu_source", lambda: None)
    m = mon.Monitor(str(tmp_path), probe, platform="linux")
    m.start()
    gc.disable()
    try:
        if hooked:
            m.arm(True)
        probe.prompt = "p1"
        caches, executed, el = types.SimpleNamespace(outputs=Cache()), set(), types.SimpleNamespace(execution_cache={})
        assert call_execute(module, "2", caches, executed, el)[0] is Result.SUCCESS
        probe.prompt = None
        probe.sampled()  # the sampler ends the run
    finally:
        gc.enable()
        m.stop()
    return records(bcnodes, tmp_path)


def slice_loop(survived, slices=20, ram=None):
    """A node's loop over time slices, as SeedVR2's encode: each iteration makes its tensors in a call of
    its own (the model's encode) while the monitor's sampler reads the stack, then appends to survived
    how many of them are still alive when the iteration ends. With ram (a FakeRam), the first encode
    raises it past the threshold before the sampler reads the stack: the snapshot is taken inside encode."""
    def encode(x, refs, probe):  # the model call: its activations live in this frame
        h = x * 2
        refs.append(weakref.ref(h))
        if ram is not None:
            ram.value = 900
        probe.sampled()  # the sampler reads the stack while h is alive
        return h.sum(dim=0)

    def node(probe):
        for i in range(slices):
            refs = []
            x = torch.full((64, 64), float(i))  # the slice
            latent = encode(x, refs, probe)
            refs += [weakref.ref(x), weakref.ref(latent)]
            del x, latent
            survived.append(sum(r() is not None for r in refs))
    return node


def test_a_node_loop_frees_every_iteration_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch):
    """A SeedVR2 encode loop (one time slice per iteration) held flat VRAM without the monitor and grew
    slice after slice with it on, to an OOM: the sampler's stack reads kept the frames it saw in a
    reference cycle, and every call of the node that ended while kept kept its locals until the garbage
    collector ran. Under the whole monitor, with the RAM threshold's snapshot taken during the loop,
    every tensor of an iteration is freed when the iteration ends, as without the monitor; the records
    are all there. RAM crosses the threshold inside the first encode: a RAM over it from the start would
    take the snapshot at the run's first sample, wherever the execution thread is then (the hook's node
    start, the loop between two encodes), and the stack would miss encode. A sample can also land before
    the node's start and between its end and the run's end (all three made this test flaky)."""
    survived, ram = [], FakeRam(100)
    recs = run_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch, slice_loop(survived, ram=ram), ram=ram)
    assert survived == [0] * 20, f"tensors alive after each iteration: {survived}"
    # the sampler may log a sample between the node's end and the run's end (the queue still runs it)
    assert [r["type"] for r in recs if r["type"] not in ("sample", "snapshot")] == ["start", "cached", "node", "node_end", "end"]
    end = [r for r in recs if r["type"] == "node_end"][0]
    assert end["state"] == "executed" and end["node"] == "2"
    # the sampler may start the run from the queue before the hook's node start: those samples name no node
    node_t = [r for r in recs if r["type"] == "node"][0]["t"]
    samples = [r for r in recs if r["type"] == "sample" and r["t"] >= node_t]
    assert len(samples) >= 40 and all(s["node"] == "2" and s["class_type"] == "Grow" for s in samples)
    assert any(s.get("line", "").endswith(" sampled") and "test_pipe_process_monitor_monitor.py" in s["line"] for s in samples)
    stack, full = [r for r in recs if r["type"] == "snapshot"]
    assert any("in encode" in line for line in stack["stack"]) and any("in node" in line for line in stack["stack"])
    assert stack["stack"] == full["stack"] and isinstance(full["census"]["groups"], list)


def test_the_execution_thread_search_frees_every_iteration(mon, bcnodes, tmp_path, monkeypatch):
    """Without the hook the sampler reads every thread's stack at every tick to find the execution
    thread: that read keeps no frame either."""
    survived = []
    recs = run_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch, slice_loop(survived), ram=100, hooked=False)
    assert survived == [0] * 20, f"tensors alive after each iteration: {survived}"
    assert [r["type"] for r in recs if r["type"] != "sample"] == ["start", "end"] and len(recs) > 40


def test_a_long_call_frees_every_step_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch):
    """KSampler's case: one long call (the sampler) runs the steps, and each step's tensors are made in
    the calls below it while the monitor's sampler reads the stack. Each step's tensors, its input
    among them, are freed when the step ends."""
    survived = []

    def model(x, refs, probe):
        h = x + 1
        refs.append(weakref.ref(h))
        probe.sampled()
        return h * 0.5

    def step(x, refs, probe):
        noise = model(x, refs, probe)
        refs.append(weakref.ref(noise))
        return x - 0.1 * noise

    def sample(x, steps, probe):  # the long call
        for _ in range(steps):
            refs = [weakref.ref(x)]  # the step's input, replaced by its output
            x = step(x, refs, probe)
            survived.append(sum(r() is not None for r in refs))
        return x

    def node(probe):
        sample(torch.zeros(64, 64), 20, probe)

    recs = run_under_the_monitor(mon, bcnodes, tmp_path, monkeypatch, node, ram=100)
    assert survived == [0] * 20, f"tensors alive after each iteration: {survived}"
    assert [r["type"] for r in recs if r["type"] != "sample"] == ["start", "cached", "node", "node_end", "end"]


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
    recreated = make_monitor(mon, tmp_path, FakeRam(kills=0, container="boot:2"))  # a new container: counter at 0 again
    assert recreated.crash()["cause"]["kind"] == "recreated"


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

"""The full clear (pipelines/process_monitor/clear.py) against fakes: a fake probe for ComfyUI's
prompt queue and worker, a fake RAM source, the stub comfy.model_management. Each freeing step is
checked on its own; the route through ComfyUI's real PromptServer and prompt queue runs in
tests/test_runtime.py, the effect on a real process's memory in the proofs (Linux container, macOS).
"""

import gc
import threading
import types
import weakref

import pytest
import torch

GIB = 1 << 30


@pytest.fixture
def cl(bcnodes, monkeypatch):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "current_loaded_models", [], raising=False)  # ComfyUI's list; the stub lacks it
    return bcnodes["pipelines.process_monitor.clear"]


class Ram:
    """A RAM source whose reading falls by `step` bytes each time it is read."""
    kind, version, limit = "fake", None, 64 * GIB

    def __init__(self, start=30 * GIB, step=GIB):
        self.value, self.step = start, step

    def describe(self):
        return "fake RAM"

    def read(self):
        self.value -= self.step
        return {"ram": self.value + self.step, "ram_limit": self.limit}


class Probe:
    """ComfyUI's queue and prompt worker: busy() answers `busy`; the free is done after `polls` checks."""

    def __init__(self, busy=None, polls=2):
        self.busy_message, self.polls = busy, polls
        self.requested = 0
        self.checks = 0

    def busy(self):
        return self.busy_message

    def request_free(self):
        self.requested += 1

    def free_done(self):
        self.checks += 1
        return self.checks > self.polls


def full_clear(cl, probe=None, ram=None, timeout=5.0):
    ram = ram or Ram()
    return cl.FullClear(probe or Probe(), lambda: (ram, None), free_timeout=timeout)


def test_refused_while_a_prompt_runs_or_waits(cl, bcnodes):
    sam3 = bcnodes["models.sam3.loader"]
    sam3._Sam3.name, sam3._Sam3.model = "held.safetensors", object()
    probe = Probe(busy="1 prompt(s) running and 0 queued: wait until the queue is empty, then clear again.")
    with pytest.raises(cl.Busy, match="wait until the queue is empty"):
        full_clear(cl, probe).run()
    assert probe.requested == 0 and sam3._Sam3.model is not None  # nothing done
    sam3.unload()


def test_refused_while_another_clear_runs(cl):
    clear = full_clear(cl)
    assert clear._lock.acquire(blocking=False)
    try:
        with pytest.raises(cl.Busy, match="already running"):
            clear.run()
    finally:
        clear._lock.release()


def test_report_baseline_before_after_and_each_step(cl):
    ram = Ram(start=30 * GIB, step=GIB)
    probe = Probe()
    clear = full_clear(cl, probe, ram)
    clear.record_baseline()  # read at startup: 30 GiB
    report = clear.run()
    assert set(report) == {"baseline", "before", "after", "steps", "remaining"}
    assert [s["name"] for s in report["steps"]] == ["comfyui_free", "pack_models", "garbage", "torch_caches", "malloc_trim"]
    assert report["baseline"]["ram"] == 30 * GIB and report["before"]["ram"] == 29 * GIB and report["after"]["ram"] == 24 * GIB
    for step in report["steps"]:
        assert step["text"] and isinstance(step["found"], str) and step["seconds"] >= 0
        assert step["freed"]["ram"] == GIB  # each reading one GiB below the one before it
    assert {"rss", "rss_anon", "rss_file", "uss", "malloc_free", "comfy_pinned", "cgroup"} <= set(report["after"])
    assert set(report["remaining"]) >= {"groups", "total_bytes"}
    assert probe.requested == 1


def test_comfyui_free_waits_for_the_worker(cl):
    probe = Probe(polls=3)
    found = cl._comfyui_free(probe, timeout=5.0)
    assert probe.requested == 1 and probe.checks == 4  # three answers "not yet", then done
    assert found == "no model was loaded"


def test_comfyui_free_names_the_models_it_unloads(cl, monkeypatch):
    import comfy.model_management as mm

    class Loaded:
        def __init__(self):
            self.model = types.SimpleNamespace(model=type("WAN21", (), {})())

        def model_loaded_memory(self):
            return 2 * GIB

    monkeypatch.setattr(mm, "current_loaded_models", [Loaded()])
    assert cl._comfyui_free(Probe(polls=0), timeout=5.0) == "1 model(s) unloaded: WAN21 (2.00 GB on the device)"


def test_comfyui_free_times_out_with_a_message(cl):
    with pytest.raises(RuntimeError, match="did not finish its free within"):
        cl._comfyui_free(Probe(polls=10 ** 9), timeout=0.2)


def test_a_prompt_started_during_the_free_stops_the_clear(cl):
    probe = Probe(polls=10 ** 9)
    probe.free_done = lambda: setattr(probe, "busy_message", "1 prompt(s) running and 0 queued: wait") or False
    with pytest.raises(cl.Busy, match="a prompt started during the full clear"):
        cl._comfyui_free(probe, timeout=5.0)


def test_pack_models_drops_every_slot(cl, bcnodes):
    birefnet, dav2 = bcnodes["models.birefnet.loader"], bcnodes["models.depth_anything_v2.loader"]
    da3, sam3 = bcnodes["models.depth_anything_3.loader"], bcnodes["models.sam3.loader"]
    weights = [torch.zeros(8) for _ in range(5)]
    refs = [weakref.ref(w) for w in weights]
    birefnet._Loaded.name, birefnet._Loaded.model = "BiRefNet-general", weights[0]
    dav2._Loaded.model = weights[1]
    da3._Loaded.name, da3._Loaded.patcher = "v3-small", weights[2]
    sam3._Sam3.name, sam3._Sam3.model, sam3._Sam3.clip = "sam3.safetensors", weights[3], weights[4]
    del weights
    found = cl._pack_models()
    assert found == "dropped: BiRefNet-general, Depth Anything V2 Small, v3-small, sam3.safetensors"
    assert all(r() is None for r in refs)  # nothing else held them: freed at once
    assert (birefnet._Loaded.model, dav2._Loaded.model, da3._Loaded.patcher, sam3._Sam3.model, sam3._Sam3.clip) == (None,) * 5
    assert cl._pack_models() == "none held"


def test_garbage_frees_what_only_a_cycle_keeps(cl):
    class Node:
        pass

    a, b = Node(), Node()
    a.other, b.other, a.tensor = b, a, torch.zeros(1024)
    ref = weakref.ref(a.tensor)
    del a, b
    gc.disable()  # the automatic collector must not get there first
    try:
        assert ref() is not None
        found = cl._garbage()
    finally:
        gc.enable()
    assert ref() is None and found.endswith("unreachable objects collected")


def test_torch_caches_resets_comfyuis_cast_buffers(cl, monkeypatch):
    import comfy.model_management as mm

    calls = []
    monkeypatch.setattr(mm, "STREAM_CAST_BUFFERS", {None: torch.zeros(GIB // 1024, dtype=torch.int8)}, raising=False)
    monkeypatch.setattr(mm, "reset_cast_buffers", lambda: calls.append("reset"), raising=False)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda force=False: calls.append("empty"))
    assert "ComfyUI's cast buffers held 0.00 GB" in cl._torch_caches() and calls == ["reset"]
    monkeypatch.delattr(mm, "reset_cast_buffers")
    calls.clear()
    cl._torch_caches()
    assert calls == ["empty"]  # an older ComfyUI: its empty_cache alone


class FakeGlibc:
    def __init__(self, free):
        self.free, self.trimmed = free, 0

    def free_bytes(self):
        return self.free

    def trim(self):
        self.trimmed += 1
        return True


def test_malloc_trim_returns_glibcs_free_blocks(cl, monkeypatch):
    lib = FakeGlibc(3 * GIB)
    monkeypatch.setattr(cl.memory_sources, "glibc", lambda: lib)
    assert cl._malloc_trim() == "glibc held 3.00 GB in free blocks; pages released" and lib.trimmed == 1
    monkeypatch.setattr(cl.memory_sources, "glibc", lambda: None)
    assert cl._malloc_trim().startswith("not glibc")


def test_reading_takes_glibc_and_comfyuis_pinned_memory(cl, monkeypatch):
    import comfy.model_management as mm

    monkeypatch.setattr(cl.memory_sources, "glibc", lambda: FakeGlibc(5))
    monkeypatch.setattr(mm, "TOTAL_PINNED_MEMORY", 7, raising=False)
    r = cl.reading(Ram(), None)
    assert r["malloc_free"] == 5 and r["comfy_pinned"] == 7 and r["cgroup"] is None and r["rss"] > 0


def test_freed_is_before_minus_after_where_both_exist(cl):
    before = {"ram": 10, "rss": 8, "vram": None, "malloc_free": 3}
    after = {"ram": 4, "rss": 9, "vram": 1}
    assert cl._freed(before, after) == {"ram": 6, "rss": -1}


def test_two_clears_at_once_run_one(cl):
    """The second of two simultaneous clicks is refused while the first waits for ComfyUI's worker."""
    gate = threading.Event()

    class SlowProbe(Probe):
        def free_done(self):
            gate.wait(5)
            return True

    clear = full_clear(cl, SlowProbe())
    first = threading.Thread(target=clear.run)
    first.start()
    try:
        while not clear._lock.locked():
            pass
        with pytest.raises(cl.Busy, match="already running"):
            clear.run()
    finally:
        gate.set()
        first.join()

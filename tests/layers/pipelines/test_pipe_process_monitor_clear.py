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
    """ComfyUI's queue, prompt worker and full-clear hooks: busy() answers `busy`; the free is done after
    `polls` checks; `held` is True while the queue is held."""

    def __init__(self, busy=None, polls=2, hooks=()):
        self.busy_message, self.polls, self.hooks = busy, polls, list(hooks)
        self.requested = 0
        self.checks = 0
        self.held = False

    def hold(self):
        probe = self

        class Held:
            def __enter__(self):
                probe.held = True

            def __exit__(self, *exc):
                probe.held = False

        return Held()

    def full_clear_hooks(self):
        return self.hooks

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


def with_probe(cl, probe, timeout=5.0):
    """What a step function is handed: the clear, for its probe and timeout."""
    return full_clear(cl, probe, timeout=timeout)


def test_refused_while_a_prompt_runs_or_waits(cl):
    called = []
    probe = Probe(busy="1 prompt(s) running and 0 queued: wait until the queue is empty, then clear again.",
                  hooks=[("BCNodes ours", lambda: called.append(1) or {})])
    with pytest.raises(cl.Busy, match="wait until the queue is empty"):
        full_clear(cl, probe).run()
    assert probe.requested == 0 and called == []  # nothing done


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
        assert step["text"] and isinstance(step["found"], str) and step["seconds"] >= 0 and step["detail"] == []
        assert step["freed"]["ram"] == GIB  # each reading one GiB below the one before it
    assert {"rss", "rss_anon", "rss_file", "uss", "malloc_free", "comfy_pinned", "cgroup"} <= set(report["after"])
    assert set(report["remaining"]) >= {"groups", "by_device"}
    assert probe.requested == 1


def test_the_steps_after_comfyuis_free_run_with_the_queue_held(cl, monkeypatch):
    """ComfyUI's free runs on its prompt worker, so the queue is open then; every later step runs with
    the queue held, so no prompt starts while the caches are reset."""
    seen = {}
    probe = Probe(hooks=[("BCNodes ours", lambda: seen.setdefault("pack_models", probe.held) and {})])
    probe.request_free = lambda: seen.setdefault("comfyui_free", probe.held)
    for name in ("_garbage", "_torch_caches", "_malloc_trim"):
        monkeypatch.setattr(cl, name, lambda clear, name=name: seen.setdefault(name, probe.held) and ("", []))
    monkeypatch.setattr(cl, "STEPS", tuple((n, t, getattr(cl, f.__name__)) for n, t, f in cl.STEPS))
    full_clear(cl, probe).run()
    assert seen == {"comfyui_free": False, "pack_models": True, "_garbage": True, "_torch_caches": True, "_malloc_trim": True}
    assert probe.held is False  # released at the end


def test_a_prompt_queued_during_comfyuis_free_stops_the_clear_before_the_held_steps(cl):
    called = []
    probe = Probe(hooks=[("BCNodes ours", lambda: called.append(1) or {})])
    done = probe.free_done
    probe.free_done = lambda: done() and not setattr(probe, "busy_message", "0 prompt(s) running and 1 queued: wait")
    with pytest.raises(cl.Busy, match="the clear stopped after it"):
        full_clear(cl, probe).run()
    assert probe.requested == 1 and called == [] and probe.held is False


def test_comfyui_free_waits_for_the_worker(cl):
    probe = Probe(polls=3)
    found, detail = cl._comfyui_free(with_probe(cl, probe))
    assert probe.requested == 1 and probe.checks == 4  # three answers "not yet", then done
    assert found == "no model was loaded" and detail == []


def test_comfyui_free_names_the_models_it_unloads(cl, monkeypatch):
    import comfy.model_management as mm

    class Loaded:
        def __init__(self):
            self.model = types.SimpleNamespace(model=type("WAN21", (), {})())

        def model_loaded_memory(self):
            return 2 * GIB

    monkeypatch.setattr(mm, "current_loaded_models", [Loaded()])
    assert cl._comfyui_free(with_probe(cl, Probe(polls=0)))[0] == "1 model(s) unloaded: WAN21 (2.00 GiB on the device)"


def test_comfyui_free_times_out_with_a_message(cl):
    with pytest.raises(RuntimeError, match="did not finish its free within"):
        cl._comfyui_free(with_probe(cl, Probe(polls=10 ** 9), timeout=0.2))


def test_a_prompt_started_during_the_free_stops_the_clear(cl):
    probe = Probe(polls=10 ** 9)
    probe.free_done = lambda: setattr(probe, "busy_message", "1 prompt(s) running and 0 queued: wait") or False
    with pytest.raises(cl.Busy, match="a prompt started during the full clear"):
        cl._comfyui_free(with_probe(cl, probe))


def linear(n):
    """A module of n float32 weights and no bias: 4 n bytes."""
    return torch.nn.Linear(n, 1, bias=False)


def test_release_pack_models_drops_every_slot_and_counts_its_bytes(cl, bcnodes):
    birefnet, dav2 = bcnodes["models.birefnet.loader"], bcnodes["models.depth_anything_v2.loader"]
    da3, sam3 = bcnodes["models.depth_anything_3.loader"], bcnodes["models.sam3.loader"]
    modules = [linear(n) for n in (100, 200, 300, 400, 500)]
    refs = [weakref.ref(m.weight) for m in modules]
    birefnet._Loaded.name, birefnet._Loaded.model = "BiRefNet-general", modules[0]
    dav2._Loaded.model = modules[1]
    da3._Loaded.name, da3._Loaded.patcher = "v3-small", types.SimpleNamespace(model=modules[2])  # a ModelPatcher
    sam3._Sam3.name, sam3._Sam3.model = "sam3.safetensors", types.SimpleNamespace(model=modules[3])
    sam3._Sam3.clip = types.SimpleNamespace(cond_stage_model=modules[4])  # comfy.sd.CLIP
    del modules
    assert cl.release_pack_models() == {"BiRefNet-general": 400, "Depth Anything V2 Small": 800, "v3-small": 1200,
                                        "sam3.safetensors": 1600 + 2000}
    assert all(r() is None for r in refs)  # nothing else held them: freed at once
    assert (birefnet._Loaded.model, dav2._Loaded.model, da3._Loaded.patcher, sam3._Sam3.model, sam3._Sam3.clip) == (None,) * 5
    assert cl.release_pack_models() == {}  # idempotent


def test_pack_models_calls_every_hook_and_reports_a_failing_one(cl):
    calls = []

    def ours():
        calls.append("ours")
        return {"BiRefNet-general": GIB}

    def broken():
        calls.append("broken")
        raise RuntimeError("its loader is gone")

    def theirs():
        calls.append("theirs")
        return {}

    def wrong():
        return ["not", "a", "dict"]

    probe = Probe(hooks=[("BCNodes ours", ours), ("Other broken", broken), ("BCVideoNodes theirs", theirs), ("Other wrong", wrong)])
    found, rows = cl._pack_models(with_probe(cl, probe))
    assert calls == ["ours", "broken", "theirs"]  # a failing hook does not stop the next one
    assert rows[0] == {"hook": "BCNodes ours", "freed": {"BiRefNet-general": GIB}, "error": None}
    assert rows[1] == {"hook": "Other broken", "freed": {}, "error": "RuntimeError: its loader is gone"}
    assert rows[2] == {"hook": "BCVideoNodes theirs", "freed": {}, "error": None}
    assert rows[3]["error"].startswith("TypeError: returned list, expected a dict")
    assert found == ("BCNodes ours: BiRefNet-general 1.00 GiB; Other broken: failed (RuntimeError: its loader is gone); "
                     "BCVideoNodes theirs: nothing loaded; Other wrong: failed (TypeError: returned list, expected a dict of model name -> bytes)")
    assert cl._pack_models(with_probe(cl, Probe())) == ("no pack registered a hook", [])


def test_the_report_carries_each_hooks_result(cl):
    probe = Probe(hooks=[("BCNodes release_pack_models", lambda: {"v3-small": 7})])
    report = full_clear(cl, probe).run()
    step = next(s for s in report["steps"] if s["name"] == "pack_models")
    assert step["detail"] == [{"hook": "BCNodes release_pack_models", "freed": {"v3-small": 7}, "error": None}]


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
        found = cl._garbage(None)
    finally:
        gc.enable()
    assert ref() is None and found[0].endswith("unreachable objects collected")


def test_torch_caches_resets_comfyuis_cast_buffers(cl, monkeypatch):
    import comfy.model_management as mm

    calls = []
    monkeypatch.setattr(mm, "STREAM_CAST_BUFFERS", {None: torch.zeros(GIB // 1024, dtype=torch.int8)}, raising=False)
    monkeypatch.setattr(mm, "reset_cast_buffers", lambda: calls.append("reset"), raising=False)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda force=False: calls.append("empty"))
    assert "ComfyUI's cast buffers held 0.00 GiB" in cl._torch_caches(None)[0] and calls == ["reset"]
    # dynamic VRAM: comfy_aimdo's VRAMBuffer, its committed bytes from size()
    aimdo = types.SimpleNamespace(size=lambda: 3 * GIB)
    monkeypatch.setattr(mm, "STREAM_AIMDO_CAST_BUFFERS", {None: aimdo}, raising=False)
    assert "ComfyUI's cast buffers held 3.00 GiB" in cl._torch_caches(None)[0]
    monkeypatch.delattr(mm, "reset_cast_buffers")
    calls.clear()
    cl._torch_caches(None)
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
    assert cl._malloc_trim(None) == ("glibc held 3.00 GiB in free blocks; pages released", []) and lib.trimmed == 1
    monkeypatch.setattr(cl.memory_sources, "glibc", lambda: None)
    assert cl._malloc_trim(None)[0].startswith("not glibc")


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

"""Drop stale outputs (pipelines/process_monitor/drop_stale.py) against a fake LRU cache class and
fake ComfyUI modules. The real ComfyUI is never touched.
"""

import asyncio
import json
import sys
import types

import pytest


@pytest.fixture
def ds(bcnodes):
    return bcnodes["pipelines.process_monitor.drop_stale"]


@pytest.fixture
def settings_mod(bcnodes):
    return bcnodes["pipelines.process_monitor.settings"]


# -- drop stale outputs ---------------------------------------------------------------------------

class KeySet:
    def __init__(self, keys):
        self.keys = {node: f"k{node}" for node in keys}

    def get_used_keys(self):
        return self.keys.values()


def cache_classes():
    class LRUCache:
        def __init__(self):
            self.cache, self.used_generation, self.children = {}, {}, {}
            self.calls = 0

        async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
            self.calls += 1
            self.cache_key_set = KeySet(node_ids)

    class RAMPressureCache(LRUCache):
        def __init__(self):
            super().__init__()
            self.timestamps = {}

        async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
            await super().set_prompt(dynprompt, node_ids, is_changed_cache)

    return LRUCache, RAMPressureCache


def filled(cls):
    cache = cls()
    for key in ("ka", "kb", "kc", "kd", "ke"):
        cache.cache[key] = object()
        cache.used_generation[key] = 1
        if hasattr(cache, "timestamps"):
            cache.timestamps[key] = 0.0
    cache.children["ka"] = ["kc"]  # an expansion child of a kept node
    cache.children["kd"] = ["ke"]  # the child of a stale node goes with it
    return cache


@pytest.mark.parametrize("which", [0, 1])
def test_drop_stale_removes_exactly_the_entries_the_prompt_does_not_use(ds, which):
    lru, ram = cache_classes()
    cls = (lru, ram)[which]
    hook = ds.DropStale(lru)
    hook.install()
    cache = filled(cls)
    asyncio.run(cache.set_prompt(None, ["a", "b"], None))
    assert cache.calls == 1
    assert set(cache.cache) == {"ka", "kb", "kc"}
    assert set(cache.used_generation) == {"ka", "kb", "kc"}
    assert set(cache.children) == {"ka"}
    if which:
        assert set(cache.timestamps) == {"ka", "kb", "kc"}
    hook.uninstall()
    assert lru.__dict__["set_prompt"] is hook.original
    cache = filled(cls)
    asyncio.run(cache.set_prompt(None, ["a"], None))
    assert len(cache.cache) == 5  # off: nothing dropped


def test_drop_stale_error_turns_it_off_not_the_prompt(ds):
    lru, _ = cache_classes()
    hook = ds.DropStale(lru)
    hook.install()
    cache = lru()
    cache.cache = None  # drop_stale fails on it
    asyncio.run(cache.set_prompt(None, ["a"], None))
    assert cache.calls == 1 and not hook.active and "drop stale outputs stopped" in hook.error
    hook.uninstall()


def test_drop_stale_hook_detection(ds, monkeypatch):
    lru, _ = cache_classes()
    caching = types.ModuleType("comfy_execution.caching")
    caching.LRUCache = lru
    pkg = types.ModuleType("comfy_execution")
    pkg.caching = caching
    monkeypatch.setitem(sys.modules, "comfy_execution", pkg)
    monkeypatch.setitem(sys.modules, "comfy_execution.caching", caching)
    hook, reason = ds.drop_stale_hook()
    assert reason is None and hook.cache_class is lru
    del caching.LRUCache
    hook, reason = ds.drop_stale_hook()
    assert hook is None and "cannot be imported" in reason


# -- the setting ---------------------------------------------------------------------------------

def test_settings_default_and_an_old_file(settings_mod, tmp_path):
    assert settings_mod.MonitorSettings().drop_stale_outputs is False
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"enabled": False, "keep_runs": 5, "retired_setting": True}))  # an unknown key is ignored
    s = settings_mod.load(str(path))
    assert (s.enabled, s.keep_runs, s.drop_stale_outputs) == (False, 5, False)
    with pytest.raises(ValueError, match="drop_stale_outputs must be true or false"):
        settings_mod.apply(s, {"drop_stale_outputs": 1})


class FakeHook:
    def __init__(self):
        self.active, self.log = False, []

    def describe(self):
        return "fake"

    def install(self):
        self.active = True
        self.log.append("install")

    def uninstall(self):
        self.active = False
        self.log.append("uninstall")


def test_setting_applies_independently_of_enabled(ds, settings_mod):
    hook = FakeHook()
    resolved = []
    setting = ds.DropStaleSetting(lambda: resolved.append(1) or (hook, None))
    off = settings_mod.MonitorSettings(enabled=False)
    setting.apply(off)
    assert not hook.active and resolved == []  # not resolved while off
    setting.apply(settings_mod.apply(off, {"drop_stale_outputs": True}))
    assert hook.active
    setting.apply(settings_mod.apply(off, {"drop_stale_outputs": True, "enabled": True}))
    setting.apply(off)
    assert not hook.active
    setting.apply(settings_mod.apply(off, {"drop_stale_outputs": True}))
    assert resolved == [1]  # resolved once
    assert hook.log == ["install", "uninstall", "install"]


def test_setting_with_no_hook_does_nothing(ds, settings_mod):
    setting = ds.DropStaleSetting(lambda: (None, "no cache"))
    setting.apply(settings_mod.MonitorSettings(drop_stale_outputs=True))
    setting.apply(settings_mod.MonitorSettings())

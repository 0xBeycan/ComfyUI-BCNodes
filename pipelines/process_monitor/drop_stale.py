"""Drop stale outputs (`drop_stale_outputs`, off by default), a Process Monitor setting that works
with the monitor on or off: applied at startup from the saved settings, and live on every change.
No ComfyUI file is edited; turning the setting off restores what was there.

The classic cache's behaviour on the RAM-pressure and LRU caches: at the start of each prompt,
every output-cache entry the prompt does not use is dropped, so the tensors of earlier inputs are
freed at once. The classic cache does it already and the null cache holds nothing. Models are not
unloaded.

Nothing here imports ComfyUI at module level: the cache class is found when the hook is resolved.
"""

import asyncio
import functools
import logging

_ORIGINAL = "__bcnodes_drop_stale_original__"


def drop_stale(cache):
    """Drops every entry of an LRU / RAM-pressure output cache that the current prompt does not use:
    its key set's keys, plus the expansion children an entry it keeps lists (the classic cache keeps
    those in the kept node's subcache). Returns the number dropped."""
    keep = set(cache.cache_key_set.get_used_keys())
    children = getattr(cache, "children", {})
    pending = list(keep)
    while pending:
        for child in children.get(pending.pop(), ()):
            if child not in keep:
                keep.add(child)
                pending.append(child)
    stale = [key for key in cache.cache if key not in keep]
    for key in stale:
        del cache.cache[key]
        for side in ("used_generation", "timestamps", "children"):
            getattr(cache, side, {}).pop(key, None)
    return len(stale)


class DropStale:
    """Wraps `set_prompt` of ComfyUI's LRU cache class (RAMPressureCache's set_prompt calls it through
    super()), which the executor calls once per prompt, before any node runs. The wrapper awaits the
    original, then drops the stale entries; an error in that step turns the setting's effect off with
    a message, never the prompt."""

    def __init__(self, cache_class):
        self.cache_class = cache_class
        current = cache_class.__dict__["set_prompt"]
        self.original = getattr(current, _ORIGINAL, current)
        self.active = False
        self.error = None
        self.wrapper = self._wrap()

    def describe(self):
        return "drop stale outputs on: each prompt frees the cached outputs it does not use"

    def install(self):
        self.error = None
        self.active = True
        if self.cache_class.__dict__.get("set_prompt") is not self.wrapper:
            self.cache_class.set_prompt = self.wrapper

    def uninstall(self):
        """Restores the original; when another pack wrapped set_prompt after this one, the wrapper
        stays in its chain as a plain pass-through."""
        self.active = False
        if self.cache_class.__dict__.get("set_prompt") is self.wrapper:
            self.cache_class.set_prompt = self.original

    def _wrap(self):
        original, hook = self.original, self

        @functools.wraps(original)
        async def set_prompt(cache, *args, **kwargs):
            result = await original(cache, *args, **kwargs)
            if hook.active:
                try:
                    dropped = drop_stale(cache)
                    if dropped:
                        logging.info("[BCNodes] drop stale outputs: %d cached output(s) of earlier prompts freed", dropped)
                except Exception as e:  # ours, never the prompt's
                    hook.active = False
                    hook.error = f"drop stale outputs stopped after an error ({e!r}); turn the setting off and on to retry"
                    logging.exception("[BCNodes] %s", hook.error)
            return result

        setattr(set_prompt, _ORIGINAL, original)
        return set_prompt


def drop_stale_hook():
    """(DropStale, None) over ComfyUI's LRUCache, or (None, why it cannot hook)."""
    try:
        from comfy_execution.caching import LRUCache
    except Exception as e:  # a foreign or broken ComfyUI
        return None, f"ComfyUI's output cache cannot be imported ({e})"
    fn = LRUCache.__dict__.get("set_prompt")
    fn = getattr(fn, _ORIGINAL, fn)
    if fn is None or not asyncio.iscoroutinefunction(fn):
        return None, "comfy_execution.caching.LRUCache has no set_prompt coroutine"
    return DropStale(LRUCache), None


class DropStaleSetting:
    """Applies `drop_stale_outputs` from MonitorSettings. The hook is resolved the first time the
    setting is on (once: ComfyUI does not change while it runs); a hook that cannot be made is
    reported once and the setting has no effect."""

    def __init__(self, resolve=drop_stale_hook):
        self._resolve = resolve
        self._hook = None
        self._resolved = False
        self._applied = None

    def apply(self, settings):
        """Installs or uninstalls the hook to match settings, one log line per change."""
        wanted = bool(settings.drop_stale_outputs)
        if self._applied == wanted:
            return
        self._applied = wanted
        if wanted and not self._resolved:
            self._resolved = True
            self._hook, reason = self._resolve()
            if self._hook is None:
                logging.info("[BCNodes] drop stale outputs is off: %s.", reason)
        if self._hook is None:
            return
        if wanted:
            self._hook.install()
            logging.info("[BCNodes] %s.", self._hook.describe())
        else:
            self._hook.uninstall()
            logging.info("[BCNodes] drop stale outputs off.")

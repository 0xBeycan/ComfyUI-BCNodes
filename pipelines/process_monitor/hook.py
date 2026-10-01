"""The execution.py hook point: ComfyUI's per-node coroutine `execution.execute`.

PromptExecutor.execute_async looks `execute` up as a module global at every node, so replacing the
module attribute puts the wrapper around every node of every prompt; restoring it removes it. The
wrapper always awaits the original and returns its result untouched: the monitor's own work runs
before and after it, and an error in that work turns the measurement off with a visible message,
never the node.

Detection is by feature, not by version string: the coroutine with the parameters read here, and
the output cache's side-effect-free `get_local` (ComfyUI 0.17.0). Without them the hook layers
(node-start records, per-node measurement) are off and the status says why.
"""

import functools
import inspect
import logging
import time
from typing import NamedTuple

MIN_COMFYUI = "0.17.0"
_PARAMS = ("dynprompt", "caches", "current_item", "extra_data", "executed", "prompt_id", "execution_list")
_ORIGINAL = "__bcnodes_monitor_original__"


class Call(NamedTuple):
    """The arguments of one execute call the monitor reads."""
    dynprompt: object
    caches: object
    current_item: str
    extra_data: dict
    executed: set
    prompt_id: str
    execution_list: object


def find():
    """(execution module, None) when the hook point is there, else (None, reason)."""
    try:
        import execution
        from comfy_execution.caching import BasicCache
    except Exception as e:  # anything a foreign or broken ComfyUI raises on import
        return None, f"ComfyUI's execution module cannot be imported ({e})"
    fn = getattr(execution, "execute", None)
    fn = getattr(fn, _ORIGINAL, fn)
    if fn is None or not inspect.iscoroutinefunction(fn):
        return None, "execution.execute is missing or not a coroutine"
    names = list(inspect.signature(fn).parameters)
    missing = [p for p in _PARAMS if p not in names]
    if missing:
        return None, f"execution.execute has no parameter(s) {', '.join(missing)}"
    if not hasattr(BasicCache, "get_local"):
        return None, "the output cache has no get_local"
    return execution, None


class Hook:
    """Wraps execution.execute around a recorder with node_start(call) -> token and
    node_end(token, call, result, seconds)."""

    def __init__(self, execution, recorder):
        self.module = execution
        self.original = getattr(execution.execute, _ORIGINAL, execution.execute)
        names = list(inspect.signature(self.original).parameters)
        self.index = [names.index(p) for p in _PARAMS]
        self.recorder = recorder
        self.active = False
        self.error = None
        self.seconds = 0.0  # the monitor's own time inside the wrapper
        self.wrapper = self._wrap()

    def install(self):
        self.error = None
        self.active = True
        if self.module.execute is not self.wrapper:
            self.module.execute = self.wrapper

    def uninstall(self):
        """Restores the original. When another pack wrapped execute after this one, the wrapper stays
        in its chain as a plain pass-through instead of cutting that pack out."""
        self.active = False
        if self.module.execute is self.wrapper:
            self.module.execute = self.original

    def _failed(self, stage, error):
        self.active = False
        self.error = (f"per-node measurement stopped after an error in the monitor ({stage}: {error!r}); "
                      "the workflow is not affected. Turn the monitor off and on to retry.")
        logging.exception("[BCNodes] Process Monitor: %s", self.error)

    def _call(self, args, kwargs):
        return Call(*(args[i] if i < len(args) else kwargs[p] for i, p in zip(self.index, _PARAMS)))

    def _wrap(self):
        original, hook = self.original, self

        @functools.wraps(original)
        async def execute(*args, **kwargs):
            if not hook.active:
                return await original(*args, **kwargs)
            t0 = time.perf_counter()
            call = token = result = None
            try:
                call = hook._call(args, kwargs)
                token = hook.recorder.node_start(call)
            except Exception as e:  # the monitor's error, never the node's
                hook._failed("node start", e)
            t1 = time.perf_counter()
            try:
                result = await original(*args, **kwargs)
                return result
            finally:
                t2 = time.perf_counter()
                if token is not None and hook.active:
                    try:
                        hook.recorder.node_end(token, call, result, t2 - t1)
                    except Exception as e:
                        hook._failed("node end", e)
                hook.seconds += (t1 - t0) + (time.perf_counter() - t2)

        setattr(execute, _ORIGINAL, original)
        return execute

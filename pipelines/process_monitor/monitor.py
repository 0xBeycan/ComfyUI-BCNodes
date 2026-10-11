"""The Process Monitor controller.

Off: no thread, no hook, no file write. On: one sampler thread wakes every 100 ms. Idle, it only
looks whether a prompt runs and reads RAM / VRAM once a second for the live bars. During a run it
reads every tick and, with the black box on, logs a line (RAM, VRAM, the executing node, the line
the execution thread is on). With the hook, every node start is logged too; an armed run adds the
per-node measurement. When RAM crosses the threshold, the execution thread's stack is logged at once,
and the census of the live tensors when the running node ends (on the execution thread, through the
hook).

The sampler thread frees nothing of another thread: whatever it held of another thread's objects (a
frame, a tensor, a model patcher) would become the last reference when that thread dropped its own,
and be freed on the sampler, during a CUDA graph capture of the execution thread too (an abort). So it
reads stacks as strings (thread_stack), never takes the census or lists ComfyUI's loaded models, and
makes no CUDA call but the counter reads it is entered for (CudaMemory.enter_thread); the run's end
record, which reads torch's peak, is written on this thread or the execution thread.

Run boundaries come from the prompt queue (`probe.running()`), which every ComfyUI version has; the
hook starts a run as soon as its first node runs. The monitor counts its own time: the hook's time
on the execution thread, the sampler's CPU time and the snapshot.

The probe is the node layer's adapter over ComfyUI's server: running() -> (prompt_id, extra_data)
or None, status(prompt_id) -> str, last_node() -> node id or None, push(payload).
"""

import ast
import contextlib
import gc
import json
import logging
import os
import re
import sys
import sysconfig
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from functools import lru_cache, partial
from itertools import accumulate, chain, repeat, starmap, takewhile
from operator import is_not, methodcaller

from ...libs import memory_sources
from ...libs.tensor_census import Covered, census, sized, storage_bytes
from . import blackbox, settings as settings_mod
from .hook import MIN_COMFYUI, Hook, find

PERIOD = 0.1
LIVE_EVERY = 10  # ticks: the live bars update once a second
_LIBRARY_DIRS = tuple(p for p in {sysconfig.get_paths().get("stdlib"), sysconfig.get_paths().get("platstdlib")} if p)
# the stack read (thread_stack): C callables only
_ONCE = ((),)  # starmap's one call
_IS_FRAME = partial(is_not, None)
_BACK = repeat("f_back")
_COLLECTS_ON_ALLOCATION = sys.version_info < (3, 12)  # since 3.12 a collection waits for the next bytecode
_FRAME_REPR = re.compile(r"<frame at 0x[0-9a-fA-F]+, file (.+), line (-?\d+), code (.*)>", re.DOTALL)


def _entry_text(filename, line, name):
    return f"{filename}:{line} {name}"


def _is_library(filename):
    return ("site-packages" in filename or "dist-packages" in filename or filename.startswith("<")
            or filename.startswith(_LIBRARY_DIRS))


def _gil():
    """False on a free-threaded interpreter running without its GIL (3.13t+), where other threads run
    during any call."""
    enabled = getattr(sys, "_is_gil_enabled", None)
    return enabled is None or enabled()


@contextlib.contextmanager
def _no_collection():
    """Before Python 3.12 an allocation can run the cyclic collector at once, and the finalizers it runs
    are Python code, during which other threads run: the collector is paused around a stack read there."""
    paused = _COLLECTS_ON_ALLOCATION and gc.isenabled()
    if paused:
        gc.disable()
    try:
        yield
    finally:
        if paused:
            gc.enable()


@lru_cache(maxsize=4096)
def _filename(quoted):
    """A file name from its repr(), as a frame's repr shows it (decoded once per file)."""
    return ast.literal_eval(quoted)


def _entry(text):
    """(filename, line, name) of a frame's repr: <frame at 0x.., file '/x.py', line 12, code fn>."""
    match = _FRAME_REPR.fullmatch(text)
    return _filename(match[1]), int(match[2]), match[3]


def thread_stack(ident):
    """((filename, line, name), ...) of a thread's stack, innermost first; () for a thread that is gone,
    for ident None, and without the GIL (_gil).

    The only place the monitor reads a stack. The frames are taken and dropped inside ONE C-level call
    (the tuple() below): sys._current_frames(), the dict's get, each f_back and each frame's repr are C
    functions that run no bytecode and never release the GIL, and the call drops its last frame
    reference before it returns. So no other thread runs while this thread holds a frame, every frame it
    holds belongs to a call that is still running, and that call's own reference outlives this one: the
    read frees nothing. A frame held in Python (a loop over f_back) is still held when its call returns;
    it then keeps the call's locals, and dropping it frees them on this thread (the RTX 3090 crash: the
    LM's CUDA tensors freed on the sampler thread during a CUDA graph capture, an abort). Only strings
    leave the call. The repr, not f_code: reading f_code raises an audit event, and a Python audit hook
    (filelock installs one, and ComfyUI imports it) is bytecode, during which other threads run;
    sys._current_frames() audits too, but before it takes a frame. The dict it returns lives inside the
    call: it holds the caller's own frame, and a name bound to it would make a reference cycle that keeps
    every frame of it, and their locals, until the cyclic collector runs."""
    if not _gil():
        return ()
    with _no_collection():
        frames = tuple(map(repr, takewhile(_IS_FRAME, accumulate(chain(
            map(methodcaller("get", ident), starmap(sys._current_frames, _ONCE)), _BACK), getattr))))
    return tuple(map(_entry, frames))


def thread_stacks():
    """{thread id: stack} of every thread, each read as thread_stack reads one (the ids in one C-level
    call of their own)."""
    with _no_collection():
        idents = list(chain.from_iterable(starmap(sys._current_frames, _ONCE)))
    return {ident: thread_stack(ident) for ident in idents}


def code_lines(stack):
    """(line, top) of a stack: the innermost entry outside the stdlib and installed packages (the line
    of the node's own code, e.g. the torch.stack call), and the innermost entry when it is another one."""
    own = next((i for i, (filename, _, _) in enumerate(stack) if not _is_library(filename)), None)
    return _entry_text(*stack[own or 0]), (_entry_text(*stack[0]) if own != 0 else None)


def find_execution_thread(stacks):
    """The id of the thread running a prompt (stacks as thread_stacks gives them): the one with
    PromptExecutor's frame on its stack."""
    for ident, stack in stacks.items():
        if any(name in ("execute_async", "execute") and filename.endswith("execution.py") for filename, _, name in stack):
            return ident
    return None


def output_cache_bytes(outputs_cache):
    """Bytes held by ComfyUI's output cache (every entry of every subcache, each byte once)."""
    covered, total, stack = Covered(), 0, [outputs_cache]
    while stack:
        cache = stack.pop()
        for entry in list(getattr(cache, "cache", {}).values()):
            total += storage_bytes(getattr(entry, "outputs", entry), covered)
        stack.extend(list(getattr(cache, "subcaches", {}).values()))
    return total


def link_inputs(node, execution_list, unique_id):
    """[{"name", "bytes", "count", "tensors"}] of the node's linked inputs that hold tensors, read from
    the execution list's own references (no cache access with side effects)."""
    held = getattr(execution_list, "execution_cache", {}).get(unique_id, {})
    out = []
    for name, value in node.get("inputs", {}).items():
        if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)):
            continue
        entry = held.get(value[0])
        outputs = getattr(entry, "outputs", entry)
        if outputs is None or value[1] >= len(outputs):
            continue
        nbytes, parts = sized(outputs[value[1]])
        if parts:
            out.append({"name": name, "bytes": nbytes, "count": len(parts), "tensors": parts[:4]})
    return out


def cached_nodes(dynprompt, outputs_cache):
    """[{"node", "class_type", "output_bytes"}] of the prompt's nodes whose outputs are in the cache when
    the run starts: ComfyUI never calls execute for most of them, so the run's records would miss them."""
    out = []
    for nid, node in dynprompt.get_original_prompt().items():
        entry = outputs_cache.get_local(nid)
        if entry is not None:
            out.append({"node": nid, "class_type": node["class_type"],
                        "output_bytes": storage_bytes(getattr(entry, "outputs", entry))})
    return out


def loaded_models():
    """[{"id", "name", "bytes"}] of ComfyUI's loaded models (bytes on their device), [] outside
    ComfyUI. Only on the execution thread between nodes, or with the prompt queue held and empty:
    a LoadedModel holds its patcher by a weak reference, so the strong one taken here is the last when
    the execution thread drops the patcher meanwhile, and the patcher, its weights and its finalizers
    (cleanup_models) then go on the calling thread."""
    try:
        import comfy.model_management as mm
    except ImportError:
        return []
    out = []
    for lm in list(mm.current_loaded_models):
        patcher = lm.model
        if patcher is None:
            continue
        inner = getattr(patcher, "model", patcher)
        out.append({"id": id(patcher), "name": type(inner).__name__, "bytes": lm.model_loaded_memory()})
    return out


def workflow_id(extra_data):
    wf = ((extra_data or {}).get("extra_pnginfo") or {}).get("workflow") or {}
    return wf.get("id") if isinstance(wf, dict) else None


def _measurement_path(directory, workflow):
    """The calibration file of a workflow id, or None for an id that is not a plain token: the id comes
    from the client and names a file."""
    if not isinstance(workflow, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", workflow):
        return None
    return os.path.join(directory, f"{workflow}.json")


class Run:
    def __init__(self, prompt_id, extra_data, armed, log, hook_s):
        self.prompt_id = prompt_id
        self.workflow_id = workflow_id(extra_data)
        self.t0 = time.time()
        self.armed = armed
        self.log = log
        self.hook_s0 = hook_s
        self.sampler_s = 0.0
        self.snapshot_s = 0.0
        self.node = None
        self.class_type = None
        self.node_peak = 0
        self.vram_peak = 0  # torch's peak over the run, folded in before every per-node reset
        self.snapshot_done = False
        self.census_due = None  # the threshold's stack record, until the execution thread adds the census
        self.models_known = False  # ComfyUI's loaded models at the start are in the log
        self.cached_listed = False
        self.pending = {}  # node id -> seconds of its PENDING execute calls


class Monitor:
    def __init__(self, directory, probe, platform=sys.platform):
        self.directory = directory
        self.probe = probe
        self.platform = platform
        self.session = uuid.uuid4().hex
        self.settings_path = os.path.join(directory, "settings.json")
        self.runs_dir = os.path.join(directory, "runs")
        self.measure_dir = os.path.join(directory, "measurements")
        self.settings = settings_mod.load(self.settings_path)
        self.ram = self.gpu = None
        self.hook = None
        self.hook_reason = None
        self.run = None
        self.armed_next = False
        self.exec_thread = None
        self.error = None
        self.last = None  # the newest reading, for the live route
        self.idle_sampler_s = 0.0
        self.node_cost = []  # the hook's seconds per node of finished armed runs (for Emulate)
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()

    # -- switching -------------------------------------------------------------------------------

    @property
    def enabled(self):
        return self._thread is not None

    def set_enabled(self, enabled):
        """Applies the toggle live and persists it."""
        if enabled != self.settings.enabled:
            self.settings = settings_mod.apply(self.settings, {"enabled": enabled})
            settings_mod.save(self.settings_path, self.settings)
        if enabled:
            self.start()
        else:
            self.stop()

    def start(self):
        if self._thread is not None:
            return
        self.ram = memory_sources.ram_source()
        self.gpu = memory_sources.gpu_source()
        module, self.hook_reason = find()
        if module is not None and self.hook is None:
            self.hook = Hook(module, self)
        if self.hook is not None:
            self.hook.install()
        self.error = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bcnodes-process-monitor", daemon=True)
        self._thread.start()
        crash = self.crash_flag()
        if crash:
            logging.warning("[BCNodes] Process Monitor: the last run (prompt %s) ended without an end record; "
                            "see the monitor's Crash tab.", crash)

    def stop(self):
        """Removes the hook, then stops the sampler, which ends a running run on its way out (the end
        record reads torch's peak: a CUDA call the calling thread, a route's, must not make)."""
        if self._thread is None:
            return
        if self.hook is not None:
            self.hook.uninstall()
        self._stop.set()
        self._thread.join(timeout=5)
        self._thread = None

    def update_settings(self, changes):
        self.settings = settings_mod.apply(self.settings, {k: v for k, v in changes.items() if k != "enabled"})
        settings_mod.save(self.settings_path, self.settings)
        return self.settings

    def arm(self, armed=True):
        if armed and self.hook is None:
            raise RuntimeError(self.hook_message())
        self.armed_next = bool(armed)

    def hook_message(self):
        if self.hook is not None and self.hook.error:
            return self.hook.error
        if self.hook is None:
            return (f"per-node measurement needs ComfyUI {MIN_COMFYUI} or newer: {self.hook_reason or 'the monitor is off'}. "
                    "The live bars, Emulate and the black box work without it.")
        return None

    # -- the sampler thread ----------------------------------------------------------------------

    def _loop(self):
        status = "monitor turned off during the run"
        try:
            if self.gpu is not None:
                try:
                    self.gpu.enter_thread()  # before this thread's first CUDA read
                except Exception:
                    self.gpu = None  # no CUDA read is safe on this thread, not even the end record's
                    raise
            tick = 0
            while not self._stop.wait(PERIOD):
                t0 = time.thread_time()
                self._tick(tick % LIVE_EVERY == 0)
                spent = time.thread_time() - t0
                run = self.run
                if run is not None:
                    run.sampler_s += spent
                else:
                    self.idle_sampler_s += spent
                tick += 1
        except Exception as e:  # keep the workflow running; the status shows the error
            self.error = f"the sampler stopped after an error: {e!r}"
            logging.exception("[BCNodes] Process Monitor: %s", self.error)
            if self.hook is not None:
                self.hook.uninstall()  # no run starts without the sampler that ends it
            status = f"monitor stopped during the run: {self.error}"  # the run did not crash, the monitor did
        with self._lock:
            if self.run is not None:  # its end record, written here: it reads torch's peak
                self._end_run(status)

    def _tick(self, live):
        running = self.probe.running()
        with self._lock:
            if self.run is not None and (running is None or running[0] != self.run.prompt_id):
                self._end_run(self.probe.status(self.run.prompt_id))
            if running is not None and self.run is None:
                self._start_run(*running)
            run = self.run
        if run is None or not (run.log or run.armed):
            if live:
                self.last = self._read()
                self.probe.push(self._live(self.last))
            return
        sample = self._sample(run)
        if run.log is not None:
            run.log.write(sample)
            if not run.snapshot_done and sample.get("ram") and sample["ram"] >= self.settings.threshold * sample["ram_limit"]:
                self._snapshot(run, sample)
        if live:
            self.probe.push(self._live(sample))

    def _read(self):
        out = dict(self.ram.read())
        if self.gpu is not None:
            out.update(self.gpu.read())
        return out

    def _sample(self, run):
        sample = {"type": "sample", "t": time.time(), **self._read(),
                  "node": run.node or self.probe.last_node(), "class_type": run.class_type}
        stack = self._execution_stack()
        if stack:
            sample["line"], top = code_lines(stack)
            if top:
                sample["top"] = top
        if sample.get("ram") is not None:
            run.node_peak = max(run.node_peak, sample["ram"])
        self.last = sample
        return sample

    def _execution_stack(self):
        """The execution thread's stack as thread_stack gives it; () when no thread runs a prompt."""
        stack = thread_stack(self.exec_thread)
        if not stack:
            stacks = thread_stacks()
            self.exec_thread = find_execution_thread(stacks)
            stack = stacks.get(self.exec_thread, ())
        return stack

    def _live(self, sample):
        run = self.run
        return {**sample, "enabled": True, "armed": self.armed_next or bool(run and run.armed),
                "run": None if run is None else {"prompt_id": run.prompt_id, "node": sample.get("node") or run.node,
                                                 "class_type": run.class_type, "elapsed": time.time() - run.t0}}

    def _snapshot(self, run, sample):
        """Once per run, on the sampler thread: the execution thread's stack, its 40 innermost calls as
        traceback.format_stack prints them (thread_stack), a record written at once: the kill can be a
        fraction of a second away. The census of the live tensors is not taken here: it holds each
        tensor while it describes it, and a tensor the node drops meanwhile would be freed on this
        thread. The execution thread adds it when the running node ends (_census)."""
        run.snapshot_done = True
        stack = traceback.format_list([(filename, line, name, None)
                                       for filename, line, name in reversed(self._execution_stack()[:40])])
        stopped = False
        if self.settings.stop_at_threshold:
            try:
                import comfy.model_management as mm
                mm.interrupt_current_processing(True)
                stopped = True
            except ImportError:
                pass
        record = {"type": "snapshot", "t": time.time(), "ram": sample["ram"], "ram_limit": sample["ram_limit"],
                  "threshold": self.settings.threshold, "node": sample.get("node"), "class_type": sample.get("class_type"),
                  "line": sample.get("line"), "stack": stack, "stopped": stopped}
        run.log.write(record)
        run.census_due = record

    def _census(self, run, where):
        """On the execution thread, at a node boundary (no CUDA graph is captured there, and what the
        census frees it frees on the thread that owns it): the threshold's stack record again, with
        every live tensor of the process (tensor_census.census). Its time is the run's snapshot_s, not
        the hook's."""
        record, run.census_due = run.census_due, None
        t0 = time.perf_counter()
        run.log.write({**record, "census": census(), "scope": f"whole process, {where}", "t": time.time(),
                       "seconds": round(time.perf_counter() - t0, 3)})
        run.snapshot_s += time.perf_counter() - t0

    # -- runs ------------------------------------------------------------------------------------

    def _start_run(self, prompt_id, extra_data, models=None):
        """models: loaded_models(), read by the caller on the execution thread; None from the sampler
        thread, which never reads them: the run's first node start logs them then."""
        armed, self.armed_next = self.armed_next and self.hook is not None, False
        log = None
        if self.settings.black_box or armed:
            os.makedirs(self.runs_dir, exist_ok=True)
            blackbox.rotate(self.runs_dir, self.settings.keep_runs)
            log = blackbox.RunLog(blackbox.run_path(self.runs_dir, time.time(), prompt_id))
        run = Run(prompt_id, extra_data, armed, log, self.hook.seconds if self.hook else 0.0)
        run.models_known = models is not None
        self.run = run
        self.exec_thread = None
        if self.gpu is not None:
            self.gpu.reset_peak()
        if log is not None:
            log.write({"type": "start", "v": blackbox.FORMAT, "session": self.session, "prompt_id": prompt_id,
                       "workflow_id": run.workflow_id, "t": run.t0, "pid": os.getpid(), "platform": self.platform,
                       "ram_source": self.ram.describe(), "ram_limit": self.ram.limit, "oom_kills": self.ram.oom_kills(),
                       "container": self.ram.identity(),
                       "models_loaded": None if models is None else [{"name": m["name"], "bytes": m["bytes"]} for m in models],
                       "armed": armed, "threshold": self.settings.threshold})
        return run

    def _end_run(self, status):
        """On the sampler thread or the execution thread only: _fold_vram_peak is a CUDA call."""
        run, self.run = self.run, None
        hook_s = (self.hook.seconds if self.hook else 0.0) - run.hook_s0 - run.snapshot_s  # the census ran inside the hook
        seconds = time.time() - run.t0
        monitor = {"hook_s": round(hook_s, 4), "sampler_s": round(run.sampler_s, 4), "snapshot_s": round(run.snapshot_s, 4),
                   "total_s": round(hook_s + run.sampler_s + run.snapshot_s, 4)}
        if run.log is None:
            return
        end = {"type": "end", "t": time.time(), "status": status, "seconds": round(seconds, 3), "monitor": monitor}
        if self._fold_vram_peak(run):
            end["vram_peak"] = run.vram_peak
        run.log.finish(end)
        if run.armed:
            self._keep_measurement(run, hook_s)

    def _fold_vram_peak(self, run):
        """Folds torch's peak since its last reset into the run's; False when the device keeps none (MPS)."""
        peak = self.gpu.peak().get("vram_peak") if self.gpu is not None else None
        if peak is None:
            return False
        run.vram_peak = max(run.vram_peak, peak)
        return True

    def _ensure_run(self, prompt_id, extra_data):
        with self._lock:
            if self.run is not None and self.run.prompt_id != prompt_id:
                self._end_run(self.probe.status(self.run.prompt_id))
            return self.run or self._start_run(prompt_id, extra_data, loaded_models())

    def _keep_measurement(self, run, hook_s):
        """The armed run's per-node records, per workflow, for Emulate's calibration."""
        records = blackbox.read_records(run.log.path)
        starts, nodes = {}, []
        for r in records:
            if r.get("type") == "node":
                starts[r["node"]] = r
            elif r.get("type") == "node_end":
                if "cache" in r and r["node"] in starts:  # the bytes the node added to the cache (a passed-on input adds none)
                    r["new_bytes"] = max(0, r["cache"] - starts[r["node"]]["cache"])
                nodes.append(r)
        if nodes:
            self.node_cost.append(hook_s / len(nodes))
        path = _measurement_path(self.measure_dir, run.workflow_id)
        if path is None or not nodes:
            return
        os.makedirs(self.measure_dir, exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"prompt_id": run.prompt_id, "t": run.t0, "nodes": {r["node"]: r for r in nodes}}, f)
        os.replace(path + ".tmp", path)

    # -- the hook's recorder ---------------------------------------------------------------------

    def node_start(self, call):
        run = self._ensure_run(call.prompt_id, call.extra_data)
        node = call.dynprompt.get_node(call.current_item)
        run.node, run.class_type = call.current_item, node["class_type"]
        self.exec_thread = threading.get_ident()
        if run.log is None:
            return None
        if run.census_due is not None:
            self._census(run, f"before node {call.current_item} started")
        if run.armed and not run.cached_listed:
            run.cached_listed = True
            run.log.write({"type": "cached", "t": time.time(), "nodes": cached_nodes(call.dynprompt, call.caches.outputs)})
        ram = self.ram.read()["ram"]
        record = {"type": "node", "t": time.time(), "node": call.current_item,
                  "display": call.dynprompt.get_display_node_id(call.current_item), "class_type": node["class_type"],
                  "ram": ram, "cache": output_cache_bytes(call.caches.outputs),
                  "inputs": link_inputs(node, call.execution_list, call.current_item)}
        if not run.models_known:  # the sampler started the run
            run.models_known = True
            record["models_at_start"] = [{"name": m["name"], "bytes": m["bytes"]} for m in loaded_models()]
        run.log.write(record)
        if not run.armed:
            return None
        if self.gpu is not None:
            self._fold_vram_peak(run)  # the reset below starts the node's peak: the run's keeps what came before
            self.gpu.reset_peak()
        run.node_peak = ram or 0
        return {"run": run, "record": record, "window": self.ram.open_peak_window(), "models": loaded_models()}

    def node_end(self, token, call, result, seconds):
        """After every node; token is node_start's (None when it measures nothing)."""
        run = self.run if token is None else token["run"]
        if run is not None and run.census_due is not None:
            self._census(run, f"when node {call.current_item} ended")
        if token is None:
            return
        start, window = token["record"], token["window"]
        state = getattr(result[0], "name", "FAILURE") if result else "FAILURE"
        uid = call.current_item
        if state == "PENDING":  # runs again later (lazy inputs, subgraph expansion, async): time adds up
            run.pending[uid] = run.pending.get(uid, 0.0) + seconds
            if window is not None:
                window.close()
            return
        seconds += run.pending.pop(uid, 0.0)
        executed = uid in call.executed
        record = {"type": "node_end", "t": time.time(), "node": uid, "display": start["display"],
                  "class_type": start["class_type"], "ram_start": start["ram"]}
        entry = call.caches.outputs.get_local(uid)
        nbytes, parts = sized(getattr(entry, "outputs", entry)) if entry is not None else (0, [])
        record.update(outputs=parts[:8], output_bytes=nbytes)
        if state == "SUCCESS" and not executed:
            record["state"] = "cached"  # from cache, not measured: no time, no peak
            if window is not None:
                window.close()
            run.log.write(record)
            return
        ram_end = self.ram.read()["ram"]
        # ram_start, ram_end and the sampled ram are the working set; memory.peak also counts page cache
        # (files read meanwhile), so it is kept apart as ram_peak_raw
        record.update(state="executed" if state == "SUCCESS" else "failed", seconds=round(seconds, 4), ram_end=ram_end,
                      ram_peak=max(run.node_peak, ram_end or 0), peak_source=f"sampler maximum ({int(PERIOD * 1000)} ms)",
                      cache=output_cache_bytes(call.caches.outputs))
        if window is not None:
            record["ram_peak_raw"] = window.close()
        if self.gpu is not None:
            record.update(self.gpu.peak())
            self._fold_vram_peak(run)
        after = loaded_models()
        before_ids = {m["id"] for m in token["models"]}
        after_ids = {m["id"] for m in after}
        record["models_loaded"] = [{"name": m["name"], "bytes": m["bytes"]} for m in after if m["id"] not in before_ids]
        record["models_unloaded"] = [{"name": m["name"], "bytes": m["bytes"]} for m in token["models"] if m["id"] not in after_ids]
        run.log.write(record)

    # -- reports for the routes ------------------------------------------------------------------

    def crash_flag(self):
        """The prompt id of the newest run of an earlier process when it ended without an end record, else
        None (blackbox.latest_crash)."""
        path = blackbox.latest_crash(self.runs_dir, self.session)
        return blackbox.read_records(path)[0]["prompt_id"] if path else None

    def crash(self):
        path = blackbox.latest_crash(self.runs_dir, self.session)
        if path is None:
            return None
        ram = self.ram or memory_sources.ram_source()
        return blackbox.crash_report(blackbox.read_records(path), ram.oom_kills(), ram.identity(), self.platform)

    def last_run(self):
        records = blackbox.latest_finished(self.runs_dir)
        return blackbox.run_report(records) if records else None

    def measurement(self, workflow):
        path = _measurement_path(self.measure_dir, workflow)
        if path is None or not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def node_cost_s(self):
        """The hook's mean seconds per node over this session's armed runs, or None."""
        return sum(self.node_cost) / len(self.node_cost) if self.node_cost else None

    def emulate(self, prompt, workflow, env):
        from . import emulate as emulate_mod  # with its cost profiles: only an estimate needs them, not the startup

        ram = self.ram or memory_sources.ram_source()
        return emulate_mod.emulate(prompt, workflow, env, ram.limit, ram.read()["ram"], self.measurement(workflow.get("id")),
                                   self.node_cost_s())

    def status(self):
        ram, run = self.ram, self.run  # read once: the sampler ends runs meanwhile
        return {
            "enabled": self.enabled, "settings": asdict(self.settings), "armed": self.armed_next,
            "hook": {"available": self.hook is not None and not self.hook.error, "message": self.hook_message(),
                     "min_version": MIN_COMFYUI},
            "sources": {"ram": ram.describe() if ram else None,
                        "vram": self.gpu.describe() if self.gpu is not None else "no CUDA or MPS device"},
            "error": self.error, "crash": self.crash_flag(), "running": run.prompt_id if run else None,
            "idle_sampler_s": round(self.idle_sampler_s, 4),
            "node_cost_s": self.node_cost_s(),
        }

"""Full clear: the process's RAM and VRAM back to where they were right after ComfyUI started,
without a restart, measured against a reading taken at startup (the baseline).

The steps, in order, each measured on its own (a reading before and after it):

  comfyui_free  ComfyUI's own free, what POST /free with unload_models and free_memory asks for, run
                by its prompt worker and waited for: every model unloaded and dropped from model
                management, the executor's caches (every node output and node object of the earlier
                prompts) replaced by empty ones, then its gc.collect and empty_cache.
  pack_models   this pack's model slots (BiRefNet, Depth Anything V2 and 3, SAM 3): they keep a model
                between runs outside ComfyUI's caches, so its free cannot drop them. Each loads again
                on its node's next run.
  garbage       gc.collect(): what only reference cycles kept after the steps above.
  torch_caches  ComfyUI's cast buffers (the VRAM it keeps for weights cast on the fly), the free blocks
                of torch's caching allocator (CUDA or MPS, through ComfyUI's soft_empty_cache) and
                torch's pinned host cache.
  malloc_trim   glibc keeps freed blocks in its arenas for reuse (one arena per thread: ComfyUI's
                prompt worker is a thread); malloc_trim(0) gives every whole free page back. Linux with
                glibc only.

Then one census of the live tensors (libs/tensor_census.census): what something still references
after the clear. What a restart would free and the clear cannot: memory held by another pack's own
cache (it shows in the census), libraries and kernels loaded during the run (the CUDA context grows on
first use). Page cache (files read or mapped, the cgroup's `file`) is reported, not dropped: the
kernel reclaims it under pressure, and a restart of the process does not free it either.

The probe is the node layer's adapter over ComfyUI's server: busy() -> message or None,
request_free(), free_done() -> bool.
"""

import gc
import threading
import time
from typing import TypedDict

import torch

from ...libs import memory_sources
from ...libs.tensor_census import census
from .monitor import loaded_models

FREE_TIMEOUT_S = 300.0  # ComfyUI's free offloads every loaded model: a big one takes a while
POLL_S = 0.05
# the counters of a reading a step can free
COUNTERS = ("ram", "rss", "rss_anon", "rss_file", "uss", "malloc_free", "pinned_cache", "comfy_pinned",
            "vram", "vram_reserved", "vram_device")


class Busy(RuntimeError):
    """The clear was refused: a prompt runs or waits in the queue, or another clear runs."""


class Step(TypedDict):
    name: str
    text: str  # what the step does
    found: str  # what it found to free
    seconds: float
    freed: dict  # counter -> bytes, the reading before the step minus the one after (negative: grew)


class Report(TypedDict):
    baseline: dict | None  # the reading at startup; None when the server's startup was not seen
    before: dict
    after: dict
    steps: list  # [Step]
    remaining: dict  # the census of the live tensors after the clear


def _gb(n):
    return f"{n / 2 ** 30:.2f} GB"


def reading(ram, gpu):
    """Everything the clear compares, in bytes: the monitor's RAM and VRAM (what the bars show: the
    cgroup's working set in a container, else the process RSS) and what the RAM is made of (the
    process's own memory and its mapped files, the cgroup's anon / file split, glibc's free blocks,
    pinned host memory). None where this system has no such counter."""
    import comfy.model_management as mm

    out = {"t": time.time(), **ram.read(), **memory_sources.process_memory(),
           "cgroup": ram.breakdown() if ram.kind == "cgroup" else None}
    lib = memory_sources.glibc()
    out["malloc_free"] = lib.free_bytes() if lib is not None else None
    out["comfy_pinned"] = getattr(mm, "TOTAL_PINNED_MEMORY", None)  # host memory ComfyUI pinned for its models
    if gpu is not None:
        out.update(gpu.read())
        out["pinned_cache"] = gpu.pinned_cache()
    return out


def _comfyui_free(probe, timeout):
    models = loaded_models()
    probe.request_free()
    deadline = time.monotonic() + timeout
    while not probe.free_done():
        message = probe.busy()
        if message:
            raise Busy(f"a prompt started during the full clear: ComfyUI frees after it ends. {message}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"ComfyUI's prompt worker did not finish its free within {timeout:.0f} s; "
                               "look at the server log, then clear again.")
        time.sleep(POLL_S)
    if not models:
        return "no model was loaded"
    return f"{len(models)} model(s) unloaded: " + ", ".join(f"{m['name']} ({_gb(m['bytes'])} on the device)" for m in models)


def _pack_models():
    from ...models.birefnet import loader as birefnet
    from ...models.depth_anything_3 import loader as depth_anything_3
    from ...models.depth_anything_v2 import loader as depth_anything_v2
    from ...models.sam3 import loader as sam3

    held = [name for name in (birefnet.unload(), depth_anything_v2.unload(), depth_anything_3.unload(), sam3.unload()) if name]
    return "dropped: " + ", ".join(held) if held else "none held"


def _garbage():
    return f"{gc.collect()} unreachable objects collected"


def _torch_caches():
    import comfy.model_management as mm

    found = []
    cuda = torch.cuda.is_available() and torch.cuda.is_initialized()
    if cuda:
        found.append(f"the CUDA allocator kept {_gb(torch.cuda.memory_reserved() - torch.cuda.memory_allocated())} unused")
    if hasattr(mm, "reset_cast_buffers"):  # newer ComfyUI
        held = sum(t.nbytes for t in list(getattr(mm, "STREAM_CAST_BUFFERS", {}).values()))
        mm.reset_cast_buffers()  # ends with soft_empty_cache
        found.append(f"ComfyUI's cast buffers held {_gb(held)}")
    else:
        mm.soft_empty_cache()  # CUDA empty_cache and ipc_collect, MPS empty_cache; nothing on the CPU
    if cuda:
        empty_host = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None) or getattr(torch._C, "_host_emptyCache", None)
        if empty_host is not None:
            empty_host()
            found.append("torch's pinned host cache emptied")
    return "; ".join(found) or "no GPU cache on this device"


def _malloc_trim():
    lib = memory_sources.glibc()
    if lib is None:
        return "not glibc (macOS, musl): nothing to trim"
    held = lib.free_bytes()
    released = lib.trim()
    return (f"glibc held {_gb(held)} in free blocks" if held is not None else "glibc older than 2.33: free blocks not counted") + \
        ("; pages released" if released else "; nothing to release")


STEPS = (
    ("comfyui_free", "ComfyUI's own free (POST /free): every model unloaded, every cached node output dropped", None),
    ("pack_models", "this pack's model slots dropped (BiRefNet, Depth Anything, SAM 3); they load again when needed", _pack_models),
    ("garbage", "Python's garbage collector: objects only reference cycles kept", _garbage),
    ("torch_caches", "ComfyUI's cast buffers, the GPU allocator's cache and torch's pinned host cache emptied", _torch_caches),
    ("malloc_trim", "glibc's malloc_trim(0): freed memory its arenas keep for reuse goes back to the system", _malloc_trim),
)


def _freed(before, after):
    return {k: before[k] - after[k] for k in COUNTERS if before.get(k) is not None and after.get(k) is not None}


class FullClear:
    """The full clear and its baseline. `sources()` -> (RAM source, GPU source or None): the monitor's
    own when it runs, else fresh ones (the clear works with the monitor off)."""

    def __init__(self, probe, sources, free_timeout=FREE_TIMEOUT_S):
        self.probe = probe
        self.sources = sources
        self.free_timeout = free_timeout
        self.baseline = None
        self._lock = threading.Lock()

    def record_baseline(self):
        """The reading right after ComfyUI started: its custom nodes loaded, no prompt run yet."""
        self.baseline = reading(*self.sources())

    def run(self):
        """-> Report. Raises Busy (nothing done) while a prompt runs or waits, or another clear runs."""
        message = self.probe.busy()
        if message:
            raise Busy(message)
        if not self._lock.acquire(blocking=False):
            raise Busy("a full clear is already running; wait for its report.")
        try:
            ram, gpu = self.sources()
            before = reading(ram, gpu)
            steps, last = [], before
            for name, text, fn in STEPS:
                t0 = time.perf_counter()
                found = _comfyui_free(self.probe, self.free_timeout) if fn is None else fn()
                now = reading(ram, gpu)
                steps.append(Step(name=name, text=text, found=found, seconds=round(time.perf_counter() - t0, 3),
                                  freed=_freed(last, now)))
                last = now
            return Report(baseline=self.baseline, before=before, after=last, steps=steps, remaining=census(top=12))
        finally:
            self._lock.release()

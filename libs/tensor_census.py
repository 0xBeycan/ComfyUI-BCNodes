"""Tensor sizes from metadata: shape, dtype, device and storage bytes, never a copy.

  - describe / storage_bytes / sized: what a ComfyUI value holds (IMAGE, MASK, a LATENT dict, a CONDITIONING list),
    tensors found through lists, tuples and dicts only (a model object is not walked: its weights
    are reported as loaded models, not as node outputs);
  - census: the live torch tensors of the process grouped by shape / dtype / device, found through
    the garbage collector (or the tensors and numpy arrays of given values).

Bytes are memory, counted once: a tensor's storage is a range of addresses [start, end) on its device,
and the ranges are merged. Views, the same tensor reached twice and storages that start at different
addresses inside one buffer (slices of a pinned host buffer, of a memory-mapped file, of one untyped
storage) add no byte twice.
"""

import bisect
import gc
from itertools import chain, compress, starmap

import numpy as np
import torch

_MAPS = "/proc/self/maps"
# mappings with a path that are still RAM: shared memory, devices, memfd
_NOT_FILES = ("/dev/", "/memfd:", "/SYSV")


class Covered:
    """Address ranges per device, merged. Pass one Covered over several values to count memory once
    across all of them."""

    def __init__(self):
        self._ranges = {}  # device -> ([starts], [ends]), sorted, disjoint

    def claim(self, device, start, end):
        """Adds [start, end) on device; returns how many of its bytes were not covered yet."""
        if end <= start:
            return 0
        starts, ends = self._ranges.setdefault(device, ([], []))
        i = bisect.bisect_left(ends, start)  # the first range that ends at or after start
        j = bisect.bisect_right(starts, end)  # past the last range that begins at or before end
        covered = sum(min(e, end) - max(s, start) for s, e in zip(starts[i:j], ends[i:j]))
        new = end - start - covered
        if i < j:
            start, end = min(start, starts[i]), max(end, ends[j - 1])
        starts[i:j], ends[i:j] = [start], [end]
        return new


def _span(ptr, shape, strides, itemsize):
    """[start, end) of the bytes a strided array's elements occupy (strides in bytes)."""
    if any(n == 0 for n in shape):
        return ptr, ptr
    low = sum((n - 1) * s for n, s in zip(shape, strides) if s < 0)
    high = sum((n - 1) * s for n, s in zip(shape, strides) if s > 0)
    return ptr + low, ptr + high + itemsize


def _memory(t):
    """(device, storage, view) of t: the address ranges of the memory it lives in (its storage, or a
    numpy array's owning array) and of its own elements. A tensor without a plain storage (sparse,
    nested, a wrapper subclass) is a range of its own size, keyed by the object."""
    if isinstance(t, np.ndarray):
        view = _span(t.__array_interface__["data"][0], t.shape, t.strides, t.itemsize)
        owner = t.base if isinstance(t.base, np.ndarray) else None
        if owner is None:
            return "cpu", view, view
        start = owner.__array_interface__["data"][0]
        return "cpu", (start, start + owner.nbytes), view
    try:
        s = t.untyped_storage()
        size, ptr = t.element_size(), t.data_ptr()
        if t.is_contiguous():  # most tensors: no walk over the strides
            view = (ptr, ptr + t.numel() * size)
        else:
            view = _span(ptr, t.shape, [st * size for st in t.stride()], size)
        return str(t.device), (s.data_ptr(), s.data_ptr() + s.nbytes()), view
    except (RuntimeError, NotImplementedError):
        own = (0, t.numel() * t.element_size())
        return ("object", id(t)), own, own


def describe(t):
    """{"shape", "dtype", "device", "bytes"} of a tensor or numpy array (bytes of its own view)."""
    if isinstance(t, np.ndarray):
        return {"shape": list(t.shape), "dtype": str(t.dtype), "device": "cpu (numpy)", "bytes": int(t.nbytes)}
    return {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""), "device": str(t.device),
            "bytes": t.numel() * t.element_size()}


def tensors_in(value, depth=6):
    """Every tensor and numpy array reachable from value through lists, tuples and dicts."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        yield value
    elif depth > 0 and isinstance(value, (list, tuple)):
        for v in value:
            yield from tensors_in(v, depth - 1)
    elif depth > 0 and isinstance(value, dict):
        for v in value.values():
            yield from tensors_in(v, depth - 1)


def storage_bytes(value, covered=None):
    """Bytes of the memory value's tensors live in, each byte counted once. Pass the same Covered
    over several values to count a byte once across all of them."""
    covered = Covered() if covered is None else covered
    total = 0
    for t in tensors_in(value):
        device, storage, _ = _memory(t)
        total += covered.claim(device, *storage)
    return total


def sized(value):
    """(storage_bytes(value), [describe(t) ...])."""
    return storage_bytes(value), [describe(t) for t in tensors_in(value)]


def _maps():
    """The process's memory mappings, sorted: ([starts], [ends], [file-backed]); None without
    /proc/self/maps (macOS, Windows). File-backed: mapped from a file on a filesystem, so its pages
    are page cache, not the process's own RAM; shared memory and devices are not."""
    try:
        with open(_MAPS, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    starts, ends, files = [], [], []
    for line in lines:
        fields = line.split(maxsplit=5)
        low, high = fields[0].split("-")
        path = fields[5] if len(fields) > 5 else ""
        starts.append(int(low, 16))
        ends.append(int(high, 16))
        files.append(path.startswith("/") and not path.startswith(_NOT_FILES))
    return starts, ends, files


def _backing(maps, device, address):
    """"file" or "ram" for memory on the cpu, by the mapping that holds address; "unknown" without
    the mappings or outside all of them; None on another device."""
    if device != "cpu":
        return None
    if maps is None:
        return "unknown"
    starts, ends, files = maps
    i = bisect.bisect_right(starts, address) - 1
    if i < 0 or address >= ends[i]:
        return "unknown"
    return "file" if files[i] else "ram"


def census(values=None, min_bytes=1 << 20, top=20):
    """The live tensors grouped by (shape, dtype, device, backing), largest first:
    {"groups": [{"count", "shape", "dtype", "device", "backing", "bytes_each", "bytes"}], "small_bytes",
    "total_bytes", "file_bytes"}.

    "bytes" counts memory once: each byte goes to the first tensor that claims it, the tensors in the
    order of their own elements' size, largest first, and the rest of each storage (what no live view
    covers) after all of them. So three views of one tensor show count 3 and the bytes of one, and
    slices of one buffer share it out by what each slice holds.
    "backing" is "file" for memory mapped from a file (page cache, not the process's own RAM), "ram"
    for the rest of the cpu memory, "unknown" without /proc/self/maps (macOS), None on a GPU;
    "file_bytes" sums the file-backed bytes, None when unknown.
    The tensors are those reachable from `values` through lists, tuples and dicts (numpy arrays too),
    or, with None, every torch tensor of the process, found through the garbage collector: a pass over
    all of the process's objects, the slow part. Any thread may call it."""
    if values is not None:
        found = {id(t): t for t in tensors_in(values)}
    else:
        # The pass must stay ONE C-level call, with no bytecode while the list of all objects is alive.
        # gc.get_objects() also returns objects other threads are still building: a tuple filled from a
        # generator (tuple(genexpr), f(*genexpr)) is tracked from its allocation, and its resize
        # requires a reference count of 1. A Python loop over the list lets the interpreter switch
        # threads every 5 ms while the list holds a second reference, and the builder's resize fails
        # with "SystemError: bad argument to internal function" (a sampler died of it). Inside the
        # list() call below the list is made, filtered and freed in C: no other thread runs while it
        # exists, and no reference to it outlives the call (`again` is run to its end and the holder
        # cleared, so no iterator made before the list keeps it in a cycle). The collector is paused
        # around the call, so no finalizer runs Python code inside it. A tensor is told by its type
        # alone (type.__subclasscheck__ in C): isinstance reads __class__, which can run Python code.
        # Guarded by test_threshold_snapshot_races_no_tuple_builder (tests/layers/pipelines/
        # test_pipe_process_monitor_monitor.py) and test_census_scan_keeps_no_reference.
        holder = []
        every, again = chain.from_iterable(holder), chain.from_iterable(holder)  # two passes over the list in holder
        scan = chain(
            filter(None, map(holder.append, starmap(gc.get_objects, [()]))),  # the list of all objects into holder
            compress(every, map(torch.Tensor.__subclasscheck__, map(type, again))),  # its tensors
            filter(None, again),  # `again` reaches its end and drops the list
            filter(None, starmap(holder.clear, [()])),  # holder drops it: the list is freed here
        )
        collecting = gc.isenabled()
        gc.disable()
        try:
            tensors = list(scan)
        finally:
            if collecting:
                gc.enable()
        found = {id(t): t for t in tensors}
    maps = _maps()
    rows = []
    for t in found.values():
        if isinstance(t, torch.Tensor) and t.device.type == "meta":
            continue
        device, storage, view = _memory(t)
        rows.append({"d": describe(t), "device": device, "storage": storage, "view": view,
                     "backing": _backing(maps, device, storage[0]), "bytes": 0})
    rows.sort(key=lambda r: r["view"][1] - r["view"][0], reverse=True)
    covered = Covered()
    for part in ("view", "storage"):
        for r in rows:
            r["bytes"] += covered.claim(r["device"], *r[part])
    groups, small = {}, 0
    for r in rows:
        d = r["d"]
        if d["bytes"] < min_bytes:
            small += r["bytes"]
            continue
        key = (tuple(d["shape"]), d["dtype"], d["device"], r["backing"])
        g = groups.setdefault(key, {"count": 0, "shape": d["shape"], "dtype": d["dtype"], "device": d["device"],
                                    "backing": r["backing"], "bytes_each": d["bytes"], "bytes": 0})
        g["count"] += 1
        g["bytes"] += r["bytes"]
    ordered = sorted(groups.values(), key=lambda g: g["bytes"] or g["bytes_each"], reverse=True)
    file_bytes = None if maps is None else sum(r["bytes"] for r in rows if r["backing"] == "file")
    return {"groups": ordered[:top], "small_bytes": small,
            "total_bytes": small + sum(g["bytes"] for g in groups.values()), "file_bytes": file_bytes}

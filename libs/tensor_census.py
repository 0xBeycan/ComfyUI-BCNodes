"""Tensor sizes from metadata: shape, dtype, device and storage bytes, never a copy.

  - describe / storage_bytes / sized: what a ComfyUI value holds (IMAGE, MASK, a LATENT dict, a CONDITIONING list),
    tensors found through lists, tuples and dicts only (a model object is not walked: its weights
    are reported as loaded models, not as node outputs);
  - census: the live torch tensors of the process grouped by shape / dtype / device, found through
    the garbage collector, plus the tensors and numpy arrays the given stack frames hold.

Bytes are counted once per storage: views and the same tensor reached twice add nothing.
"""

import gc

import numpy as np
import torch


def _storage(t):
    """(key, bytes) of the memory t lives in: its storage, or a numpy array's base buffer."""
    if isinstance(t, np.ndarray):
        base = t if t.base is None else t.base
        return ("numpy", id(base)), int(getattr(base, "nbytes", t.nbytes))
    try:
        s = t.untyped_storage()
        return (s.data_ptr(), s.nbytes(), str(t.device)), s.nbytes()
    except (RuntimeError, NotImplementedError):  # tensors without a plain storage (sparse, nested)
        return (id(t),), t.numel() * t.element_size()


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


def storage_bytes(value, seen=None):
    """Bytes of the memory value's tensors live in, each storage counted once. Pass the same `seen`
    set over several values to count a storage once across all of them."""
    seen = set() if seen is None else seen
    total = 0
    for t in tensors_in(value):
        key, nbytes = _storage(t)
        if key not in seen:
            seen.add(key)
            total += nbytes
    return total


def sized(value):
    """(storage_bytes(value), [describe(t) ...])."""
    return storage_bytes(value), [describe(t) for t in tensors_in(value)]


def census(frames=(), min_bytes=1 << 20, top=20, whole_process=True):
    """The live tensors grouped by (shape, dtype, device), largest first:
    {"groups": [{"count", "shape", "dtype", "device", "bytes_each", "bytes"}], "small_bytes",
    "total_bytes"}. "bytes" counts a group's distinct storages, so three views of one tensor show
    count 3 and the bytes of one. Tensors and numpy arrays are found through the locals of `frames`
    (a list there is looked into) and, with whole_process, every tensor through the garbage
    collector: a pass over all of the process's objects, the slow part."""
    found = {}
    if whole_process:
        for obj in gc.get_objects():
            if issubclass(type(obj), torch.Tensor):  # not isinstance: it reads __class__, which some module objects warn on
                found[id(obj)] = obj
    for frame in frames:
        for value in list(frame.f_locals.values()):
            for t in tensors_in(value, depth=2):
                found.setdefault(id(t), t)
    groups, storages, small = {}, set(), 0
    for t in found.values():
        if isinstance(t, torch.Tensor) and t.device.type == "meta":
            continue
        d = describe(t)
        key = (tuple(d["shape"]), d["dtype"], d["device"])
        storage, nbytes = _storage(t)
        nbytes = 0 if storage in storages else nbytes
        storages.add(storage)
        if d["bytes"] < min_bytes:
            small += nbytes
            continue
        g = groups.setdefault(key, {"count": 0, "shape": d["shape"], "dtype": d["dtype"], "device": d["device"],
                                    "bytes_each": d["bytes"], "bytes": 0})
        g["count"] += 1
        g["bytes"] += nbytes
    ordered = sorted(groups.values(), key=lambda g: g["bytes"] or g["bytes_each"], reverse=True)
    return {"groups": ordered[:top], "small_bytes": small,
            "total_bytes": small + sum(g["bytes"] for g in groups.values())}

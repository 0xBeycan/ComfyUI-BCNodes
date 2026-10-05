"""Tensor sizes from metadata (libs/tensor_census.py) and weights from safetensors headers
(libs/safetensors_info.py): bytes written out by hand per shape and dtype; memory counted once, also
across storages that share one buffer; the live census groups copies apart from views and tells
file-backed memory from RAM by the process's mappings; the header reader against files the
safetensors library writes.
"""

import ctypes
import gc
import mmap
import sys
import warnings
import weakref

import numpy as np
import pytest
import torch


@pytest.fixture
def tc(bcnodes):
    return bcnodes["libs.tensor_census"]


def test_image_mask_latent_conditioning_bytes(tc):
    image = torch.zeros(2, 4, 5, 3)
    mask = torch.zeros(2, 4, 5, dtype=torch.float16)
    latent = {"samples": torch.zeros(1, 16, 3, 4, 4), "batch_index": [0]}
    cond = [[torch.zeros(1, 7, 8), {"pooled_output": torch.zeros(1, 8)}]]
    assert tc.storage_bytes(image) == 2 * 4 * 5 * 3 * 4
    assert tc.storage_bytes(mask) == 2 * 4 * 5 * 2
    assert tc.storage_bytes(latent) == 16 * 3 * 4 * 4 * 4
    assert tc.storage_bytes(cond) == (7 * 8 + 8) * 4
    nbytes, parts = tc.sized([image, mask])
    assert nbytes == 480 + 80
    assert parts == [{"shape": [2, 4, 5, 3], "dtype": "float32", "device": "cpu", "bytes": 480},
                     {"shape": [2, 4, 5], "dtype": "float16", "device": "cpu", "bytes": 80}]


def test_a_storage_counts_once(tc):
    image = torch.zeros(10, 4, 4, 3)
    view = image[2:5]
    assert tc.storage_bytes([image, view, image]) == image.numel() * 4
    covered = tc.Covered()
    assert tc.storage_bytes(image, covered) == image.numel() * 4
    assert tc.storage_bytes(view, covered) == 0  # already counted through `covered`
    # a numpy array and its view: one base buffer
    arr = np.zeros((6, 8), dtype=np.float32)
    assert tc.storage_bytes([arr, arr[1:3]]) == 6 * 8 * 4
    # torch.from_numpy shares the buffer: the array and the tensor are one memory
    assert tc.storage_bytes(torch.from_numpy(arr)) == 6 * 8 * 4
    assert tc.storage_bytes([arr, torch.from_numpy(arr[2:])]) == 6 * 8 * 4


MB = 1 << 20


def census_of(tc, *held):
    """The census of `held` alone."""
    return tc.census(held, min_bytes=1, top=100)


def test_covered_merges_address_ranges_per_device(tc):
    c = tc.Covered()
    assert c.claim("cpu", 100, 200) == 100
    assert c.claim("cpu", 150, 250) == 50  # overlaps the first
    assert c.claim("cpu", 120, 180) == 0  # inside
    assert c.claim("cpu", 250, 260) == 10  # touches the end
    assert c.claim("cpu", 0, 400) == 400 - 160  # around all of it
    assert c.claim("cpu", 300, 300) == 0
    assert c.claim("cuda:0", 100, 200) == 100  # another device, another memory


def test_slices_of_a_growing_host_buffer_count_once(tc):
    # a pinned host buffer hands out slices of a fresh frombuffer over the buffer as long as it is
    # so far: every slice's storage starts at the buffer and reports the buffer's length then
    buf = (ctypes.c_uint8 * (24 * MB))()
    addr = ctypes.addressof(buf)
    pins, off = [], 0
    for size in (4 * MB, 6 * MB, 4 * MB, 6 * MB):
        whole = torch.frombuffer((ctypes.c_uint8 * (off + size)).from_address(addr), dtype=torch.uint8)
        pins.append(whole[off:off + size])
        off += size
    assert [p.untyped_storage().data_ptr() for p in pins] == [addr] * 4
    assert [p.untyped_storage().nbytes() for p in pins] == [4 * MB, 10 * MB, 14 * MB, 20 * MB]
    assert tc.storage_bytes(pins) == 20 * MB  # not 48 MB
    c = census_of(tc, pins)
    groups = {g["shape"][0]: g for g in c["groups"]}
    # each slice holds its own part of the buffer
    assert (groups[4 * MB]["count"], groups[4 * MB]["bytes"]) == (2, 8 * MB)
    assert (groups[6 * MB]["count"], groups[6 * MB]["bytes"]) == (2, 12 * MB)
    assert c["by_device"] == {"cpu": 20 * MB}


def test_weights_over_one_memory_map_and_slices_of_one_storage(tc, tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(bytes(12 * MB))
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mv = memoryview(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the map is read-only
        weights = [torch.frombuffer(mv[a:b], dtype=torch.float16) for a, b in ((0, 4 * MB), (4 * MB, 12 * MB))]
    assert tc.storage_bytes(weights) == 12 * MB
    assert tc.storage_bytes(weights + [weights[1][MB:]]) == 12 * MB
    storage = torch.UntypedStorage(8 * MB)
    parts = [torch.empty(0, dtype=torch.uint8).set_(storage[a:b]) for a, b in ((0, 4 * MB), (2 * MB, 6 * MB))]
    assert tc.storage_bytes(parts) == 6 * MB  # [0, 4) and [2, 6) MB
    whole = torch.empty(0, dtype=torch.uint8).set_(storage)
    assert tc.storage_bytes(parts + [whole]) == 8 * MB
    groups = {g["shape"][0]: g for g in census_of(tc, parts, whole)["groups"]}
    # the whole storage claims its bytes before the slices into it
    assert (groups[8 * MB]["bytes"], groups[4 * MB]["bytes"]) == (8 * MB, 0)
    del weights
    mv.release()
    m.close()


def maps_line(start, end, path=""):
    return f"{start:x}-{end:x} r--p 00000000 00:00 {1234 if path else 0} {path}".rstrip()


def test_file_backed_memory_is_told_apart_by_the_mappings(tc, tmp_path, monkeypatch):
    path = tmp_path / "w.bin"
    path.write_bytes(bytes(4 * MB))
    with open(path, "rb") as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    mv = memoryview(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight = torch.frombuffer(mv, dtype=torch.float16)
    anonymous = torch.ones(3 * MB, dtype=torch.uint8)
    shared = torch.ones(5 * MB, dtype=torch.uint8)
    ranges = [(weight, "/models/w.safetensors"), (anonymous, ""), (shared, "/dev/shm/torch_1")]
    lines = sorted((t.data_ptr(), maps_line(t.data_ptr(), t.data_ptr() + t.numel(), p)) for t, p in ranges)
    (tmp_path / "maps").write_text("\n".join(line for _, line in lines) + "\n")
    monkeypatch.setattr(tc, "_MAPS", str(tmp_path / "maps"))
    c = census_of(tc, weight, anonymous, shared)
    backing = {g["shape"][0]: g["backing"] for g in c["groups"]}
    assert backing == {2 * MB: "file", 3 * MB: "ram", 5 * MB: "ram"}
    assert c["file_bytes"] == 4 * MB and c["by_device"] == {"cpu": 12 * MB}
    del weight, ranges
    mv.release()
    m.close()


def test_backing_is_unknown_without_the_mappings(tc, tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_MAPS", str(tmp_path / "no-such-file"))  # macOS has no /proc/self/maps
    c = census_of(tc, torch.ones(7 * MB, dtype=torch.uint8))
    assert c["groups"][0]["backing"] == "unknown" and c["file_bytes"] is None


def test_census_groups_copies_and_views(tc):
    shape_a, shape_b = (3, 17, 19, 3), (5, 23, 29)
    copies = [torch.ones(shape_a) for _ in range(3)]  # three tensors of one shape: three storages
    base = torch.ones(shape_b)
    views = [base, base.view(shape_b)]  # same storage twice
    c = tc.census(min_bytes=1, top=10_000)
    groups = {tuple(g["shape"]): g for g in c["groups"]}
    a, b = groups[shape_a], groups[shape_b]
    each_a = 3 * 17 * 19 * 3 * 4
    assert (a["count"], a["bytes_each"], a["bytes"]) == (3, each_a, 3 * each_a)
    assert (b["count"], b["bytes_each"], b["bytes"]) == (2, 5 * 23 * 29 * 4, 5 * 23 * 29 * 4)
    del copies, views


def test_census_finds_numpy_arrays_in_values(tc):
    held = np.ones((31, 37), dtype=np.float64)  # numpy arrays are not tracked by the garbage collector
    c = tc.census([{"frames": [held]}], min_bytes=1, top=10_000)
    g = [g for g in c["groups"] if g["shape"] == [31, 37]]
    assert g and g[0]["device"] == "cpu (numpy)" and g[0]["bytes"] == 31 * 37 * 8
    del held


def test_census_scan_keeps_no_reference(tc):
    """The pass over all objects leaves no reference behind: the list of all objects is freed inside
    the one call that makes it (a reference that outlives it, even in a cycle the collector would free
    later, is what fails another thread's tuple resize). The collector's state is restored."""
    probe = [object()]  # tracked by the collector, so in the list of all objects
    before = sys.getrefcount(probe)
    tc.census(min_bytes=1 << 40)
    assert sys.getrefcount(probe) == before and gc.isenabled()
    gc.disable()
    try:
        tc.census(min_bytes=1 << 40)
        assert not gc.isenabled() and sys.getrefcount(probe) == before
    finally:
        gc.enable()


def test_census_holds_no_tensor_while_it_describes_the_others(tc, monkeypatch):
    """The census runs when RAM is near its limit, while the node goes on: a tensor the node frees
    meanwhile is freed at once, not kept until the census returns. The census holds one tensor at a
    time, the one it describes."""
    shapes = {(11, 13, 17), (11, 13, 19)}  # shapes of their own
    held = [torch.zeros(shape) for shape in sorted(shapes)]
    refs = [weakref.ref(t) for t in held]
    alive = []
    describe = tc.describe

    def describing(t):
        if tuple(t.shape) in shapes and not alive:
            held.clear()  # the node frees both while the census describes one of them
            alive.append(sum(r() is not None for r in refs))
        return describe(t)

    monkeypatch.setattr(tc, "describe", describing)
    tc.census(min_bytes=1, top=10_000)
    assert alive == [1]


def test_census_puts_small_tensors_aside(tc):
    tiny = torch.ones(3)
    c = tc.census(min_bytes=1 << 40)
    assert c["groups"] == [] and c["small_bytes"] >= 12 and sum(c["by_device"].values()) == c["small_bytes"]
    del tiny


class Wrapped(torch.Tensor):
    """A wrapper subclass as comfy_kitchen's QuantizedTensor is one: the logical shape and dtype
    (bfloat16) over inner tensors that hold the memory (int8 data and a float32 scale)."""

    @staticmethod
    def __new__(cls, qdata, scale):
        return torch.Tensor._make_wrapper_subclass(cls, qdata.shape, dtype=torch.bfloat16, device=qdata.device)

    def __init__(self, qdata, scale):
        self._qdata, self._scale = qdata, scale

    def __tensor_flatten__(self):
        return ["_qdata", "_scale"], None

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(func)


def test_a_wrapper_subclass_is_its_inner_tensors(tc):
    qdata, scale = torch.ones(4 * MB, dtype=torch.int8), torch.ones(1)
    weight = Wrapped(qdata, scale)
    assert tc.storage_bytes(weight) == 4 * MB + 4  # not the 8 MB of 4M bfloat16 values
    assert tc.storage_bytes([weight, qdata]) == 4 * MB + 4
    _, parts = tc.sized(weight)
    assert [(p["dtype"], p["bytes"]) for p in parts] == [("int8", 4 * MB), ("float32", 4)]
    module = torch.nn.Module()
    module.register_buffer("weight", weight)
    assert tc.module_bytes(module) == 4 * MB + 4
    c = census_of(tc, weight)
    assert c["by_device"] == {"cpu": 4 * MB + 4} and {g["dtype"] for g in c["groups"]} == {"int8", "float32"}
    # through the garbage collector: the wrapper and its inner tensors are all found, each byte once
    c = tc.census(min_bytes=1, top=10_000)
    assert not [g for g in c["groups"] if g["dtype"] == "bfloat16" and g["shape"] == [4 * MB]]
    mine = [g for g in c["groups"] if g["shape"] == [4 * MB] and g["dtype"] == "int8"]
    assert [(g["count"], g["bytes"]) for g in mine] == [(1, 4 * MB)]
    del weight, module


def test_sparse_and_nested_tensors_are_their_parts(tc):
    sparse = torch.eye(1000).to_sparse()  # 1000 values and 2 x 1000 int64 indices, not 1000 x 1000
    assert tc.storage_bytes(sparse) == 1000 * 4 + 2 * 1000 * 8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # sparse CSR is in beta
        csr = torch.eye(1000).to_sparse_csr()
    parts = (csr.crow_indices(), csr.col_indices(), csr.values())
    assert tc.storage_bytes(csr) == sum(p.untyped_storage().nbytes() for p in parts) < 1000 * 1000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # nested tensors are a prototype
        nested = torch.nested.nested_tensor([torch.ones(3, 2), torch.ones(5, 2)])
        jagged = torch.nested.nested_tensor([torch.ones(3, 2), torch.ones(5, 2)], layout=torch.jagged)
    assert tc.storage_bytes(nested) == 16 * 4  # one buffer
    assert tc.storage_bytes(jagged) >= 16 * 4
    c = census_of(tc, sparse, csr, nested, jagged)
    assert c["unsized"] == 0 and c["by_device"]["cpu"] == tc.storage_bytes([sparse, csr, nested, jagged])
    held = [sparse, nested, jagged]
    tc.census(min_bytes=1, top=10_000)  # through the garbage collector: no tensor kind raises
    del held


def test_census_counts_each_device_apart(tc):
    cpu = torch.ones(3 * MB, dtype=torch.uint8)
    arr = np.ones(2 * MB, dtype=np.uint8)
    meta = torch.ones(5 * MB, dtype=torch.uint8, device="meta")
    assert census_of(tc, cpu, arr, meta)["by_device"] == {"cpu": 5 * MB}
    if torch.backends.mps.is_available():
        gpu = torch.ones(MB, dtype=torch.uint8, device="mps")
        assert census_of(tc, cpu, gpu)["by_device"] == {"cpu": 3 * MB, str(gpu.device): MB}


def test_safetensors_header(bcnodes, tmp_path):
    from safetensors.torch import save_file

    si = bcnodes["libs.safetensors_info"]
    tensors = {"a": torch.zeros(3, 4), "b": torch.zeros(5, dtype=torch.bfloat16), "c": torch.zeros(2, 2, 2, dtype=torch.float16)}
    path = tmp_path / "w.safetensors"
    save_file(tensors, str(path), metadata={"format": "pt"})
    info = si.weights_info(str(path))
    assert info == {"bytes": 48 + 10 + 16, "params": 12 + 5 + 8, "dtypes": {"F32": 48, "BF16": 10, "F16": 16}, "source": "header"}


def test_weights_of_other_formats_are_their_file_size(bcnodes, tmp_path):
    si = bcnodes["libs.safetensors_info"]
    path = tmp_path / "w.ckpt"
    path.write_bytes(b"x" * 1234)
    assert si.weights_info(str(path)) == {"bytes": 1234, "params": None, "dtypes": {}, "source": "file size"}
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes((2 ** 40).to_bytes(8, "little") + b"{}")
    with pytest.raises(ValueError, match="not a safetensors header"):
        si.weights_info(str(bad))


def test_module_bytes_counts_each_weight_once(tc):
    a = torch.nn.Linear(10, 10)  # 100 + 10 float32
    b = torch.nn.Linear(10, 10, bias=False)
    b.weight = a.weight  # tied: the same storage in both
    meta = torch.nn.Linear(1000, 1000, device="meta")  # no memory
    assert tc.module_bytes(a) == 110 * 4
    assert tc.module_bytes(a, b, None, meta) == 110 * 4
    a.register_buffer("stats", torch.zeros(5, dtype=torch.float64))
    assert tc.module_bytes(a) == 110 * 4 + 40

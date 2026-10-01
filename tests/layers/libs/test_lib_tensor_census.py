"""Tensor sizes from metadata (libs/tensor_census.py) and weights from safetensors headers
(libs/safetensors_info.py): bytes written out by hand per shape and dtype; storages counted once;
the live census groups copies apart from views; the header reader against files the safetensors
library writes.
"""

import sys

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
    seen = set()
    assert tc.storage_bytes(image, seen) == image.numel() * 4
    assert tc.storage_bytes(view, seen) == 0  # already counted through `seen`
    # a numpy array and its view: one base buffer
    arr = np.zeros((6, 8), dtype=np.float32)
    assert tc.storage_bytes([arr, arr[1:3]]) == 6 * 8 * 4
    # torch.from_numpy shares the buffer; the census sees the tensor's storage size
    assert tc.storage_bytes(torch.from_numpy(arr)) == 6 * 8 * 4


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


def test_census_finds_numpy_arrays_through_frames(tc):
    held = np.ones((31, 37), dtype=np.float64)  # numpy arrays are not tracked by the garbage collector
    c = tc.census(frames=[sys._getframe()], min_bytes=1, top=10_000)
    g = [g for g in c["groups"] if g["shape"] == [31, 37]]
    assert g and g[0]["device"] == "cpu (numpy)" and g[0]["bytes"] == 31 * 37 * 8
    del held


def test_census_puts_small_tensors_aside(tc):
    tiny = torch.ones(3)
    c = tc.census(min_bytes=1 << 40)
    assert c["groups"] == [] and c["small_bytes"] >= 12 and c["total_bytes"] == c["small_bytes"]
    del tiny


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

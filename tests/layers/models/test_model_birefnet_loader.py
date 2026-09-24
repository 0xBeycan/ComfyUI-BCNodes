"""Golden: the BiRefNet loader and weight resolution (plan 8.2 row 16; ruling D9 b).

models/birefnet/loader.py:load(name) keeps one model in a single-slot
cache: on a miss it imports lazily, evicts the cached model (soft_empty_cache), looks the name
up, resolves the weights, constructs BiRefNet(backbone), loads the state dict strict, calls
eval() and moves it to the device. models/birefnet/weights.py:weights_path(name)
registers the folder, returns a file already present, and otherwise downloads
the checkpoint from its Hugging Face repo. Pinned:
  - the ordered log of a miss, a hit on the second call (same objects, no new entry) and a
    switch to another name (eviction first);
  - an unknown name with an empty cache and with a model cached (D9 b): the KeyError, the
    soft_empty_cache count and the _Loaded fields afterwards (the cached model is evicted
    before the lookup raises);
  - weight resolution for all 11 checkpoint names: the hops (URL, headers), the path, the
    add_model_folder_path / get_full_path calls, the stdout decile lines and the file; plus a
    file already present (primary and extra folder) and an unknown name with and without a
    file.

TinyNet stands in for the architecture class (looked up at call time by the lazy import) and
`safetensors.torch.load_file` is a stub; every request is answered by _golden.fake_http and
time.monotonic is a counter. Temp paths are masked as <tmp>. Recorded with
BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import hashlib
import itertools
import os
import sys
import time
import types

import pytest
import torch

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'load/miss_hit_switch': '1b151447c143d9c60cf8f242ac986981',
    'load/unknown_empty_cache': '93399012fed2877e26dd00245c6208d7',
    'load/unknown_model_cached': '7500b4af04d5ccaed5e794b0e3ac8e8e',
    'weights/download/BiRefNet-general': 'b953f4bb214f297ad548fe6c555dfc03',
    'weights/download/BiRefNet_512x512': 'c3db1947e1170c8374a3d273235f5a48',
    'weights/download/BiRefNet-HR': '5aa18eed433a2c8c5f4cb5ce89a03212',
    'weights/download/BiRefNet-portrait': 'de32f5ee9881ecaeb57ddd337b7fd24f',
    'weights/download/BiRefNet-matting': '08df6694fc5a5610b1cfc9beec78fff6',
    'weights/download/BiRefNet-HR-matting': '33fd4e5cec7a54b510045d5f40abc80f',
    'weights/download/BiRefNet_lite': '55a9a5747dabd0152e1ad9844a69d022',
    'weights/download/BiRefNet_lite-2K': '6369ca00edeca57c5a93124bda035736',
    'weights/download/BiRefNet_dynamic': '176e5404c98856e501ef49642bc0d7ed',
    'weights/download/BiRefNet_lite-matting': 'f2d504a1bf2a362d9dc20776940e37b7',
    'weights/download/Lucida': 'c67b9a69eeb493adc079e2924111104c',
    'weights/present_primary': '240c8c3feef38558feb82f871b9db202',
    'weights/present_extra_folder': '17723f40f201d0c2807277d2252a6eae',
    'weights/unknown_absent': 'db715657085aff291aaea4ba138494b1',
    'weights/unknown_present': 'c8dfea7d7f4167f2a09d49b59ec39fa6',
}

WHERE = Where({
    "load": "models.birefnet.loader:load",
    "weights_path": "models.birefnet.weights:weights_path",
    "Loaded": "models.birefnet.loader:_Loaded",
    "Loaded.name": "models.birefnet.loader:_Loaded.name",
    "Loaded.model": "models.birefnet.loader:_Loaded.model",
    "Loaded.device": "models.birefnet.loader:_Loaded.device",
    "Loaded.dtype": "models.birefnet.loader:_Loaded.dtype",
    "arch.BiRefNet": "models.birefnet.arch:BiRefNet",
})

NAMES = [
    "BiRefNet-general", "BiRefNet_512x512", "BiRefNet-HR", "BiRefNet-portrait", "BiRefNet-matting",
    "BiRefNet-HR-matting", "BiRefNet_lite", "BiRefNet_lite-2K", "BiRefNet_dynamic", "BiRefNet_lite-matting",
    "Lucida",
]
BODY = bytes(range(256)) * (10 * 1024)  # 2.5 MiB: three 1 MiB chunks, so three decile lines
LOG = []


class TinyNet(torch.nn.Module):
    """Stands in for BiRefNet and logs what the loader does to it."""

    def __init__(self, backbone="swin_v1_l"):
        super().__init__()
        LOG.append(("construct", backbone))
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        LOG.append(("load_state_dict", sorted(state_dict), strict))
        return super().load_state_dict(state_dict, strict=strict)

    def eval(self):
        LOG.append(("eval",))
        return super().eval()

    def to(self, *args, **kwargs):
        LOG.append(("to", args, kwargs))
        return super().to(*args, **kwargs)


def _mask(x, tmp):
    if isinstance(x, str):
        return x.replace(tmp, "<tmp>")
    if isinstance(x, dict):
        return {_mask(k, tmp): _mask(v, tmp) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_mask(v, tmp) for v in x)
    return x


@pytest.fixture
def seams(bcnodes, monkeypatch, tmp_path):
    """Fresh cache, recorders on the stubs, TinyNet, per-test models/user dirs; returns
    (tmp, hops_of(script), weights(name, folder))."""
    tmp = str(tmp_path)
    LOG.clear()
    for field in ("name", "model", "device", "dtype"):
        WHERE.patch(monkeypatch, f"Loaded.{field}", None)
    WHERE.patch(monkeypatch, "arch.BiRefNet", TinyNet)

    def load_file(path):
        LOG.append(("load_file", path))
        return {"weight": torch.tensor([1.5, -2.0, 0.75]), "bias": torch.tensor([0.1])}

    st = types.ModuleType("safetensors.torch")
    st.load_file = load_file
    monkeypatch.setitem(sys.modules, "safetensors.torch", st)

    mm = sys.modules["comfy.model_management"]
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: LOG.append(("soft_empty_cache",)), raising=True)
    get_device = mm.get_torch_device
    monkeypatch.setattr(mm, "get_torch_device", lambda: (LOG.append(("get_torch_device",)), get_device())[1], raising=True)

    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", os.path.join(tmp, "models"), raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: os.path.join(tmp, "user"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    add, full = fp.add_model_folder_path, fp.get_full_path

    def add_model_folder_path(name, path, is_default=False):
        LOG.append(("add_model_folder_path", name, path, is_default))
        return add(name, path, is_default)

    def get_full_path(name, filename):
        found = full(name, filename)
        LOG.append(("get_full_path", name, filename, found))
        return found

    monkeypatch.setattr(fp, "add_model_folder_path", add_model_folder_path, raising=True)
    monkeypatch.setattr(fp, "get_full_path", get_full_path, raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)

    def weights(name, folder=os.path.join(tmp, "models", "background_removal")):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"{name}.safetensors"), "wb"):
            pass

    return tmp, lambda script: _golden.fake_http(monkeypatch, script), weights


def _loaded():
    c = WHERE["Loaded"]
    return {"name": c.name, "model": type(c.model).__name__, "device": c.device, "dtype": c.dtype}


def _load(name):
    try:
        model, device, dtype = WHERE["load"](name)
        return {"returned": (type(model).__name__, device, dtype)}
    except Exception as e:
        return {"exception": (type(e).__name__, str(e))}


def test_load_miss_hit_switch(seams):
    check_env(ENV, "torch")
    tmp, _, weights = seams
    weights("BiRefNet_lite")
    weights("BiRefNet-general")
    steps = {}
    first = WHERE["load"]("BiRefNet_lite")
    steps["miss"] = {"log": list(LOG), "returned": (type(first[0]).__name__, first[1], first[2]), "loaded": _loaded()}
    LOG.clear()
    second = WHERE["load"]("BiRefNet_lite")
    steps["hit"] = {"log": list(LOG), "same_objects": all(a is b for a, b in zip(first, second)), "loaded": _loaded()}
    LOG.clear()
    steps["switch"] = {**_load("BiRefNet-general"), "log": list(LOG), "loaded": _loaded()}
    check(GOLDEN, "load/miss_hit_switch", digest(_mask(steps, tmp)))


def test_unknown_name_empty_cache(seams):
    check_env(ENV, "torch")
    tmp, _, _ = seams
    out = {**_load("nope"), "log": list(LOG), "loaded": _loaded()}
    check(GOLDEN, "load/unknown_empty_cache", digest(_mask(out, tmp)))


def test_unknown_name_model_cached(seams):
    check_env(ENV, "torch")
    tmp, _, weights = seams
    weights("BiRefNet_lite")
    WHERE["load"]("BiRefNet_lite")
    LOG.clear()
    out = {**_load("nope"), "log": list(LOG), "loaded": _loaded()}
    check(GOLDEN, "load/unknown_model_cached", digest(_mask(out, tmp)))


@pytest.mark.parametrize("name", NAMES)
def test_weights_path_download(name, seams, capsys):
    check_env(ENV, "torch")
    tmp, hops_of, _ = seams
    hops = hops_of({"huggingface.co": (200, {"Content-Type": "application/octet-stream",
                                            "Content-Length": str(len(BODY))}, BODY)})
    path = WHERE["weights_path"](name)
    with open(path, "rb") as f:
        data = f.read()
    os.remove(path)
    out = {"path": path, "log": list(LOG), "hops": hops, "stdout": capsys.readouterr().out,
           "file": (len(data), hashlib.md5(data).hexdigest())}
    check(GOLDEN, f"weights/download/{name}", digest(_mask(out, tmp)))


@pytest.mark.parametrize("case", ["present_primary", "present_extra_folder", "unknown_absent", "unknown_present"])
def test_weights_path_without_download(case, seams, capsys):
    check_env(ENV, "torch")
    tmp, hops_of, weights = seams
    hops = hops_of({})
    name = "nope" if case.startswith("unknown") else "BiRefNet_lite"
    if case == "present_extra_folder":
        extra = os.path.join(tmp, "extra", "background_removal")
        sys.modules["folder_paths"].add_model_folder_path("background_removal", extra)
        weights(name, extra)
        LOG.clear()
    elif case in ("present_primary", "unknown_present"):
        weights(name)
    try:
        out = {"path": WHERE["weights_path"](name)}
    except Exception as e:
        out = {"exception": (type(e).__name__, str(e))}
    out.update(log=list(LOG), hops=hops, stdout=capsys.readouterr().out)
    check(GOLDEN, f"weights/{case}", digest(_mask(out, tmp)))

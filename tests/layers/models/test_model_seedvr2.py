"""Golden: the SeedVR2 frame-shape rules and the VAE room maker (plan 8.2 row 23).

Pinned:
  - frames_to_4n1(t) for t = 1..13 (models/seedvr2/frames.py);
  - side_resize, incl. the max_resolution cap pass (models/seedvr2/frames.py);
  - divisible_pad with its default multiple (PAD_MULTIPLE) and with multiple=16, and the
    already-divisible input returned as the same object (models/seedvr2/frames.py);
  - make_room_for_vae, both branches: the free memory is enough, or everything is unloaded
    first; the calls on comfy.model_management and the log lines
    (models/seedvr2/vae.py:make_room_for_vae).

comfy.model_management is the stub_comfy module with recorders; get_free_memory is scripted.
Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import logging
import sys
import types

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'torchvision': '0.29.0',
}

GOLDEN = {
    'frames_to_4n1': '8ed509bda587a85c658a4cde49aab695',
    'side_resize/landscape_up': '04e82b4c88c57fd9c1ea5a4242db26df',
    'side_resize/landscape_up_cap': 'c50b942f6cf43928de9989c2b0c33887',
    'side_resize/landscape_up_cap_not_hit': '04e82b4c88c57fd9c1ea5a4242db26df',
    'side_resize/landscape_down': '9075c2b35437079d74a3889774f92e36',
    'side_resize/portrait_up_cap': '083e34a6e123fd8800a42a3f9704728d',
    'side_resize/square_same': '4f9fd8c3d3fe7c42156b546f687001ce',
    'divisible_pad/default': '59c906b0ccfcf49aaf86754f20b13c85',
    'divisible_pad/multiple_16': '59c906b0ccfcf49aaf86754f20b13c85',
    'divisible_pad/multiple_8': 'bf25d5894eb59b0ad72d481c3c12e30e',
    'divisible_pad/only_height': '55839bbafbb394e4d7ca383b2d7149ea',
    'divisible_pad/divisible_default': '2eb2ab34c426a2643f5934a2260ccb38',
    'make_room/models_stay': '414d0b2cc47b1a3bfcccc5a6c670f694',
    'make_room/unload_all': '17508c84e6b9e356b9827fa4a59c6117',
}

WHERE = Where({
    "frames_to_4n1": "models.seedvr2.frames:frames_to_4n1",
    "side_resize": "models.seedvr2.frames:side_resize",
    "divisible_pad": "models.seedvr2.frames:divisible_pad",
    "make_room_for_vae": "models.seedvr2.vae:make_room_for_vae",
})

GiB = 2 ** 30


def _frames(h, w):
    return torch.rand((2, 3, h, w), generator=torch.Generator().manual_seed(13))


def test_frames_to_4n1(bcnodes):
    check_env(ENV, "torch", "torchvision")
    check(GOLDEN, "frames_to_4n1", digest([WHERE["frames_to_4n1"](t) for t in range(1, 14)]))


@pytest.mark.parametrize("name, h, w, resolution, max_resolution", [
    ("landscape_up", 30, 44, 60, 0),
    ("landscape_up_cap", 30, 44, 60, 80),
    ("landscape_up_cap_not_hit", 30, 44, 60, 200),
    ("landscape_down", 30, 44, 17, 0),
    ("portrait_up_cap", 44, 30, 75, 96),
    ("square_same", 32, 32, 32, 0),
])
def test_side_resize(name, h, w, resolution, max_resolution, bcnodes):
    check_env(ENV, "torch", "torchvision")
    out = WHERE["side_resize"](_frames(h, w), resolution, max_resolution)
    check(GOLDEN, f"side_resize/{name}", digest(out))


@pytest.mark.parametrize("name, h, w, kwargs", [
    ("default", 30, 36, {}),
    ("multiple_16", 30, 36, {"multiple": 16}),
    ("multiple_8", 30, 36, {"multiple": 8}),
    ("only_height", 30, 48, {}),
    ("divisible_default", 32, 48, {}),
])
def test_divisible_pad(name, h, w, kwargs, bcnodes):
    check_env(ENV, "torch", "torchvision")
    frames = _frames(h, w)
    out = WHERE["divisible_pad"](frames, **kwargs)
    check(GOLDEN, f"divisible_pad/{name}", digest({"out": out, "same_object": out is frames}))


@pytest.mark.parametrize("name, free, disable_offload", [
    ("models_stay", [40 * GiB], True),
    ("unload_all", [3 * GiB, 29 * GiB], False),
])
def test_make_room_for_vae(name, free, disable_offload, bcnodes, monkeypatch, caplog):
    check_env(ENV, "torch", "torchvision")
    calls = []
    scripted = list(free)
    mm = sys.modules["comfy.model_management"]

    def get_free_memory(*args, **kwargs):
        calls.append(("get_free_memory", args, kwargs))
        return scripted.pop(0)

    monkeypatch.setattr(mm, "get_free_memory", get_free_memory, raising=True)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: calls.append(("soft_empty_cache",)), raising=True)
    monkeypatch.setattr(mm, "unload_all_models", lambda: calls.append(("unload_all_models",)), raising=True)
    monkeypatch.setattr(mm, "load_models_gpu", lambda *a, **kw: calls.append(("load_models_gpu", a, kw)), raising=True)
    vae = types.SimpleNamespace(device=torch.device("cpu"), patcher="patcher", disable_offload=disable_offload)
    caplog.set_level(logging.INFO)
    WHERE["make_room_for_vae"](vae, 17_000 * 1024 * 1024)
    logs = [(r.levelname, r.getMessage()) for r in caplog.records]
    check(GOLDEN, f"make_room/{name}", digest({"calls": calls, "logs": logs, "left": scripted}))

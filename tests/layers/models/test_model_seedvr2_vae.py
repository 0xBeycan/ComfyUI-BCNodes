"""make_room_for_vae (models/seedvr2/vae.py) under the stub model management: with less free VRAM
than asked for it unloads every model first, with enough it keeps them; either way it loads the VAE with
that amount as memory_required."""

import types

import pytest
import torch

GIB = 2 ** 30


class _VAE:
    device = torch.device("cpu")
    patcher = object()
    disable_offload = False


@pytest.mark.parametrize("free_gib, unloads", [(20.7, True), (31.0, False)])
def test_make_room_unloads_only_when_short(bcnodes, monkeypatch, free_gib, unloads):
    import comfy.model_management as mm

    calls = types.SimpleNamespace(unloaded=0, loaded=[])
    monkeypatch.setattr(mm, "get_free_memory", lambda device=None, torch_free_too=False: int(free_gib * GIB))
    monkeypatch.setattr(mm, "unload_all_models", lambda: setattr(calls, "unloaded", calls.unloaded + 1))
    monkeypatch.setattr(mm, "load_models_gpu", lambda models, **kw: calls.loaded.append(kw))
    needed = int(29.8 * GIB)
    bcnodes["models.seedvr2.vae"].make_room_for_vae(_VAE(), needed)
    assert calls.unloaded == (1 if unloads else 0)
    assert calls.loaded == [{"memory_required": needed, "force_full_load": False}]

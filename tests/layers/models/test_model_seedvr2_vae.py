"""make_room_for_vae (models/seedvr2/vae.py) under the stub model management: with less free VRAM
than asked for it unloads every model first, with enough it keeps them; either way it loads the VAE with
that amount as memory_required.

tile_for: a set tile_size is used as it is; 0 (auto) is the largest multiple of 32, from the one covering
the frame down, whose working set (fixed + bytes per pixel x min(H, t) x min(W, t)) fits the card's total
less 768 MiB. Worked out by hand, 1088x1920 (720p -> 1080p) and 2160x3840 (1080p -> 4K):
  - 32 GB (31.36 GiB, 30.61 left): encoder 1568 (1088 x 1568 x 15,440 + 6.36e9 = 30.45 GiB; 1600 is 30.96),
    decoder 1024 (29.81; 1056 is 31.23); at 4K encoder 1280 (29.48; 1312 is 30.68), decoder 1024;
  - 96 GB (94.97 GiB, 94.22 left): one 1920 tile each at 1080p (35.96 and 52.03); at 4K encoder 2816
    (2160 x 2816: 93.39; 2848 is 94.38), decoder 1984 (91.50; 2016 is 94.23);
  - 24 GB (23.65 GiB, 22.90 left): encoder 1056 (21.96), decoder 832 (22.19), at both sizes;
  - 48 GB (47.50 GiB, 46.75 left): encoder 1920 (one tile), decoder 1664 (46.08); at 4K 1664 (45.74) and 1344
    (46.00)."""

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


ENCODER = (6_360_000_000, 15_440)
DECODER = (7_950_000_000, 22_940)


@pytest.mark.parametrize("total_gib, height, width, part, expected", [
    (31.36, 1088, 1920, ENCODER, 1568), (31.36, 1088, 1920, DECODER, 1024),
    (31.36, 2160, 3840, ENCODER, 1280), (31.36, 2160, 3840, DECODER, 1024),
    (94.97, 1088, 1920, ENCODER, 1920), (94.97, 1088, 1920, DECODER, 1920),
    (94.97, 2160, 3840, ENCODER, 2816), (94.97, 2160, 3840, DECODER, 1984),
    (23.65, 1088, 1920, ENCODER, 1056), (23.65, 1088, 1920, DECODER, 832),
    (47.50, 1088, 1920, ENCODER, 1920), (47.50, 1088, 1920, DECODER, 1664),
    (47.50, 2160, 3840, ENCODER, 1664), (47.50, 2160, 3840, DECODER, 1344),
])
def test_auto_tile(bcnodes, monkeypatch, total_gib, height, width, part, expected):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: int(total_gib * GIB), raising=False)
    assert bcnodes["models.seedvr2.vae"].tile_for(0, height, width, *part, torch.device("cpu"), "test") == expected


def test_auto_tile_covering_size_is_a_multiple_of_32(bcnodes, monkeypatch):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: 1000 * GIB, raising=False)
    vae = bcnodes["models.seedvr2.vae"]
    assert vae.tile_for(0, 720, 1296, *ENCODER, torch.device("cpu"), "test") == 1312
    assert vae.tile_for(0, 32, 48, *ENCODER, torch.device("cpu"), "test") == 64  # never below the widget's smallest tile


def test_a_card_too_small_gets_the_smallest_tile_and_a_warning(bcnodes, monkeypatch, caplog):
    import logging

    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: 4 * GIB, raising=False)
    with caplog.at_level(logging.WARNING):
        assert bcnodes["models.seedvr2.vae"].tile_for(0, 1088, 1920, *DECODER, torch.device("cpu"), "test") == 64
    assert "may run out of memory" in caplog.text


@pytest.mark.parametrize("tile", [64, 1024, 4096])
def test_a_set_tile_is_kept(bcnodes, monkeypatch, tile):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: pytest.fail("a set tile reads no card"),
                        raising=False)
    assert bcnodes["models.seedvr2.vae"].tile_for(tile, 1088, 1920, *DECODER, torch.device("cpu"), "test") == tile

"""make_room_for_vae (models/seedvr2/vae.py) under the stub model management: with less free VRAM
than asked for it unloads every model first, with enough it keeps them; either way it loads the VAE with
that amount as memory_required.

tile_for: a typed tile_size is the same tile on both axes, its overlap cut as VAE Encode / Decode (Tiled) cut
it. 0 (auto) picks the tile sides (multiples of 32) computing the fewest pixels (every tile in full, the
overlaps again) whose working set (fixed + bytes per pixel x the largest tile's pixels) fits the card's total
less 768 MiB, the overlap (256) kept: a side that does not cover the frame is at least 512. Worked out by hand
for 32 GB (31.36 GiB, 30.61 left) at 1088x1920:
  - encoder: the whole frame (6.36e9 + 15,440 x 1088 x 1920 = 35.96 GiB) does not fit; two full-height strips
    compute 1088 x (1920 + 256) = 2,367,488 pixels whatever their width, the narrowest pair is 1088 wide
    (1088 + 1088 - 256 = 1920): 22.95 GiB. Two rows would compute (1088 + 256) x 1920, more;
  - decoder: two strips of 1088 x 1088 need 32.69 GiB; three, at least (1920 + 512) / 3 = 811 -> 832 wide,
    need 7.95e9 + 22,940 x 1088 x 832 = 26.74 GiB and compute 1088 x 2432 = 2,646,016 pixels, a fifth less
    than the 1024 square's six tiles ((1024 + 320) x (1024 + 1024 + 384) = 3,268,608).
The other rows of the table come out of the same arithmetic."""

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
CPU = torch.device("cpu")


def tile(bcnodes, total_gib, height, width, part, tile_size=0, overlap=256):
    vae, tiling = bcnodes["models.seedvr2.vae"], bcnodes["models.seedvr2.tiling"]
    axis, cell = (tiling.encode_axis, 1) if part is ENCODER else (tiling.decode_axis, 8)
    return vae.tile_for(tile_size, height, width, overlap, axis, cell, *part, CPU, "test"), cell


def card(monkeypatch, total_gib):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: int(total_gib * GIB), raising=False)


# card (GiB total), frame, part -> tile sides (rows, columns) in pixels, tiles (rows, columns), pixels computed
@pytest.mark.parametrize("total_gib, height, width, part, sides, tiles, computed", [
    (23.65, 1088, 1920, ENCODER, (1088, 832), (1, 3), 2646016),
    (23.65, 1088, 1920, DECODER, (1088, 608), (1, 5), 3203072),
    (23.65, 2160, 3840, ENCODER, (896, 1152), (3, 4), 12312576),
    (23.65, 2160, 3840, DECODER, (896, 768), (3, 7), 14364672),
    (31.36, 1088, 1920, ENCODER, (1088, 1088), (1, 2), 2367488),
    (31.36, 1088, 1920, DECODER, (1088, 832), (1, 3), 2646016),
    (31.36, 2160, 3840, ENCODER, (1216, 1152), (2, 4), 11132928),
    (31.36, 2160, 3840, DECODER, (896, 1152), (3, 4), 12312576),
    (47.50, 1088, 1920, ENCODER, (1088, 1920), (1, 1), 2088960),
    (47.50, 1088, 1920, DECODER, (1088, 1088), (1, 2), 2367488),
    (47.50, 2160, 3840, ENCODER, (1216, 2048), (2, 2), 9895936),
    (47.50, 2160, 3840, DECODER, (1216, 1472), (2, 3), 10514432),
    (94.97, 1088, 1920, ENCODER, (1088, 1920), (1, 1), 2088960),
    (94.97, 1088, 1920, DECODER, (1088, 1920), (1, 1), 2088960),
    (94.97, 2160, 3840, ENCODER, (2176, 2048), (1, 2), 8847360),
    (94.97, 2160, 3840, DECODER, (2176, 1472), (1, 3), 9400320),
])
def test_auto_tile(bcnodes, monkeypatch, total_gib, height, width, part, sides, tiles, computed):
    card(monkeypatch, total_gib)
    ((rows, ov_h), (cols, ov_w)), cell = tile(bcnodes, total_gib, height, width, part)
    assert (rows * cell, cols * cell) == sides and (ov_h, ov_w) == (256 // cell, 256 // cell)  # the overlap kept
    ranges, _ = bcnodes["models.seedvr2.tiling"].tile_plan(height // cell, width // cell, (rows, cols), (ov_h, ov_w), CPU)
    assert (len({r[:2] for r in ranges}), len({r[2:] for r in ranges})) == tiles
    assert sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in ranges) * cell * cell == computed
    largest = min(height, sides[0]) * min(width, sides[1])
    assert part[0] + part[1] * largest <= total_gib * GIB - 768 * 2 ** 20


def test_auto_tile_keeps_the_overlap_where_the_square_would_cut_it(bcnodes, monkeypatch):
    """24 GB, decoder: a typed 832 tile gets VAE Decode (Tiled)'s quarter (208, 26 cells); the auto strips keep 256."""
    card(monkeypatch, 23.65)
    assert bcnodes["models.seedvr2.tiling"].decode_axis(832, 256) == (104, 26)
    ((rows, ov_h), (cols, ov_w)), _ = tile(bcnodes, 23.65, 1088, 1920, DECODER)
    assert (ov_h, ov_w) == (32, 32) and cols * 8 == 608


def test_auto_tile_without_a_tiling_keeping_the_overlap(bcnodes, monkeypatch, caplog):
    """No side of 2 x overlap fits: the largest square that does, its overlap cut as a typed one; with none
    at all, the smallest tile and a warning."""
    import logging

    # 13.5 GiB, 12.75 left (13.69e9 bytes): the decoder's 7.95e9 + 22,940 per pixel holds a 480 square (13.24e9),
    # not the smallest tile keeping the overlap, 512 x 512 (13.96e9)
    card(monkeypatch, 13.5)
    assert tile(bcnodes, 13.5, 1088, 1920, DECODER)[0] == ((60, 15), (60, 15))  # 480 // 8, (480 // 4) // 8
    card(monkeypatch, 4.0)
    with caplog.at_level(logging.WARNING):
        assert tile(bcnodes, 4.0, 1088, 1920, DECODER)[0] == ((8, 2), (8, 2))  # 64 // 8, (64 // 4) // 8
    assert "may run out of memory" in caplog.text


def test_auto_tile_small_frames_and_odd_sizes(bcnodes, monkeypatch):
    card(monkeypatch, 1000)
    assert tile(bcnodes, 1000, 720, 1296, ENCODER)[0] == ((736, 256), (1312, 256))  # one tile: each side a multiple of 32 over the frame
    assert tile(bcnodes, 1000, 32, 48, ENCODER)[0] == ((64, 256), (64, 256))  # never below the widget's smallest tile


@pytest.mark.parametrize("size", [64, 1024, 4096])
def test_a_typed_tile_is_today_square(bcnodes, monkeypatch, size):
    import comfy.model_management as mm

    monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: pytest.fail("a typed tile reads no card"),
                        raising=False)
    tiling = bcnodes["models.seedvr2.tiling"]
    assert tile(bcnodes, None, 1088, 1920, ENCODER, size)[0] == (tiling.encode_axis(size, 256),) * 2
    assert tile(bcnodes, None, 1088, 1920, DECODER, size)[0] == (tiling.decode_axis(size, 256),) * 2

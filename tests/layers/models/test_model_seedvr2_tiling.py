"""The tile geometry of models/seedvr2/tiling.py against ComfyUI's own:

  - tile_plan with a tile and an overlap per axis gives the tiles of comfy/ldm/seedvr/vae.py tiled_vae's
    loop (lines 137-146, written out below), square or not;
  - encode_axis / decode_axis cut a typed tile's overlap as VAE Encode (Tiled) and VAE Decode (Tiled) on
    the SeedVR2 VAE do (comfy/sd.py, nodes.py VAEDecodeTiled's quarter, the wrapper's tile - 8, tiled_vae's
    tile - 1), worked out by hand."""

import pytest
import torch

CPU = torch.device("cpu")


def tiled_vae_ranges(h, w, ti_h, ti_w, ov_h, ov_w):
    stride_h, stride_w = max(1, ti_h - ov_h), max(1, ti_w - ov_w)
    tile_ranges = []
    for y_idx in range(0, h, stride_h):
        y_end = min(y_idx + ti_h, h)
        if y_idx > 0 and (y_end - y_idx) <= ov_h:
            continue
        for x_idx in range(0, w, stride_w):
            x_end = min(x_idx + ti_w, w)
            if x_idx > 0 and (x_end - x_idx) <= ov_w:
                continue
            tile_ranges.append((y_idx, y_end, x_idx, x_end))
    return tile_ranges


@pytest.mark.parametrize("h, w, tile, overlap", [
    (136, 240, (128, 128), (32, 32)),  # 1080p latent, the 1024 square: 2 x 3
    (136, 240, (136, 104), (32, 32)),  # the auto strips on 32 GB: one row of 3
    (1088, 1920, (1088, 1088), (256, 256)),  # the encoder's two strips
    (270, 480, (112, 144), (32, 32)),
    (48, 160, (64, 96), (16, 16)),
    (37, 53, (16, 9), (3, 1)),
])
def test_tile_plan_is_tiled_vae(bcnodes, h, w, tile, overlap):
    ranges, _ = bcnodes["models.seedvr2.tiling"].tile_plan(h, w, tile, overlap, CPU)
    assert ranges == tiled_vae_ranges(h, w, *tile, *overlap)


def test_spans(bcnodes):
    spans = bcnodes["models.seedvr2.tiling"].spans
    assert spans(240, 104, 32) == [(0, 104), (72, 176), (144, 240)]
    assert spans(240, 128, 32) == [(0, 128), (96, 224), (192, 240)]
    assert spans(136, 128, 32) == [(0, 128), (96, 136)]  # 40 rows left: kept
    assert spans(100, 80, 32) == [(0, 80), (48, 100)]  # a third tile from 96 would be 4 long, inside the overlap: dropped
    assert spans(48, 64, 256) == [(0, 48)]


@pytest.mark.parametrize("tile, overlap, expected", [
    (1024, 256, (1024, 256)), (64, 256, (64, 56)), (8, 256, (8, 0)),
])
def test_encode_axis(bcnodes, tile, overlap, expected):
    assert bcnodes["models.seedvr2.tiling"].encode_axis(tile, overlap) == expected


@pytest.mark.parametrize("tile, overlap, expected", [
    (1024, 256, (128, 32)),  # 1024 is four overlaps: kept
    (832, 256, (104, 26)),   # a quarter: 208
    (64, 256, (8, 2)),       # a quarter: 16
    (2048, 256, (256, 32)),
    (16, 64, (2, 0)),        # a quarter: 4, cut to 8 - 8 = 0
])
def test_decode_axis(bcnodes, tile, overlap, expected):
    assert bcnodes["models.seedvr2.tiling"].decode_axis(tile, overlap) == expected

"""The spatial tiles of the SeedVR2 VAE and their blend weights.

A tiling is one tile size and overlap per axis, (rows, columns), as tiled_vae takes them
(comfy/ldm/seedvr/vae.py `tile_size=(ti_h, ti_w)`, `tile_overlap=(ov_h, ov_w)`): a tile_size typed into a
node is the same on both axes, the auto tile may differ (models/seedvr2/vae.py tile_for)."""

import torch


def spans(length, tile, overlap):
    """tiled_vae's tiles along one axis of `length` cells (lines 137-146): starts `tile - overlap` apart, a
    last tile no longer than the overlap dropped. -> [(start, end)]"""
    stride = max(1, tile - overlap)
    out = []
    for start in range(0, length, stride):
        end = min(start + tile, length)
        if start > 0 and end - start <= overlap:
            continue
        out.append((start, end))
    return out


def encode_axis(tile, overlap):
    """One axis of VAE Encode (Tiled) on the SeedVR2 VAE with a typed tile, in pixels: (tile, overlap),
    the overlap cut to the tile less 8 (vae.py encode_tiled)."""
    return tile, min(overlap, max(0, tile - 8))


def decode_axis(tile, overlap):
    """One axis of VAE Decode (Tiled) on the SeedVR2 VAE with a typed tile, in latent cells: (tile, overlap),
    the overlap first cut to a quarter of the tile (nodes.py VAEDecodeTiled), then as vae.py decode_tiled ->
    tiled_vae(encode=False) takes it."""
    if tile < overlap * 4:
        overlap = tile // 4
    cells = max(1, tile // 8)
    return cells, min(max(0, min(overlap, max(0, tile - 8)) // 8), cells - 1)


def tile_plan(h, w, tile, overlap, device):
    """The spatial tiles of tiled_vae on a grid of `h` x `w` cells, `tile` and `overlap` per axis as
    (rows, columns), and the cosine ramp used for their blend weights."""
    ranges = [(y, y_end, x, x_end) for y, y_end in spans(h, tile[0], overlap[0]) for x, x_end in spans(w, tile[1], overlap[1])]

    ramps = {}

    def ramp(steps):
        if steps not in ramps:
            t = torch.linspace(0, 1, steps=steps, device=device, dtype=torch.float32)
            ramps[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramps[steps]

    return ranges, ramp


def tile_weight(y, y_end, x, x_end, h, w, th, tw, edge_h, edge_w, ramp, device):
    """tiled_vae lines 182-207: cosine fades on interior edges only, separable."""
    cur_ov_h = max(0, min(edge_h, th // 2))
    cur_ov_w = max(0, min(edge_w, tw // 2))
    w_h = torch.ones((th,), device=device)
    w_w = torch.ones((tw,), device=device)
    if cur_ov_h > 0:
        r = ramp(cur_ov_h)
        if y > 0:
            w_h[:cur_ov_h] = r
        if y_end < h:
            w_h[-cur_ov_h:] = 1.0 - r
    if cur_ov_w > 0:
        r = ramp(cur_ov_w)
        if x > 0:
            w_w[:cur_ov_w] = r
        if x_end < w:
            w_w[-cur_ov_w:] = 1.0 - r
    return w_h.view(1, 1, 1, -1, 1) * w_w.view(1, 1, 1, 1, -1)

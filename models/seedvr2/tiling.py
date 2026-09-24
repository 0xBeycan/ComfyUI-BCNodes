"""The spatial tiles of the SeedVR2 VAE and their blend weights."""

import torch


def tile_plan(h, w, tile, overlap, device):
    """The spatial tiles of comfy/ldm/seedvr/vae.py tiled_vae (lines 137-146) on a grid of
    `h` x `w` cells, and the cosine ramp used for their blend weights."""
    stride = max(1, tile - overlap)
    ranges = []
    for y in range(0, h, stride):
        y_end = min(y + tile, h)
        if y > 0 and (y_end - y) <= overlap:
            continue
        for x in range(0, w, stride):
            x_end = min(x + tile, w)
            if x > 0 and (x_end - x) <= overlap:
                continue
            ranges.append((y, y_end, x, x_end))

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

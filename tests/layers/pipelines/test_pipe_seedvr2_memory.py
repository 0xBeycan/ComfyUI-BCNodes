"""The SeedVR2 flows hold no clip-sized buffer besides their outputs, and give what the whole-clip
way gives, bit for bit:

  - SeedVR2 Resize downscales four frames at a time: its outputs equal the lanczos downscale of the
    whole clip (8-bit PIL, as comfy.utils.lanczos) resized afterwards;
  - SeedVR2 VAE Encode writes every latent slice into the float32 latent it returns: equal to the
    slices concatenated in the VAE dtype (one tile), or to the float32 tile blend cast to the VAE
    dtype (tiles), then cast to float32 and scaled;
  - SeedVR2 VAE Decode sums its tiles in the output itself: equal to a float16 (B, 3, T, H, W) sum
    of the whole clip normalised afterwards, a chunk at a time.

The stand-in VAE is local in time and space, so one call over a whole tile equals the slices the
flows feed it; a ramp across each tile makes overlapping tiles disagree, so the blend is exercised.
Memory is the CPU allocator's, as the profiler traces it. What a flow holds besides its outputs
(its transient) is measured at two clip lengths: a clip-sized buffer makes it grow by one frame of
that buffer per frame, so the bounds below are a fraction of one frame.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

ACTIVE = 2  # comfy.ldm.seedvr.vae.MemoryState, as tests/_harness.py stubs it
SCALE = 0.9152  # BYTEDANCE_VAE_SCALING_FACTOR, as tests/_harness.py stubs it (shift 0)
CPU = torch.device("cpu")


class VideoAutoencoderKLWrapper:
    """8 x 8 pixels <-> one latent cell, 4 frames <-> one latent frame (the first frame alone)."""

    use_slicing = True
    slicing_sample_min_size = 4
    slicing_latent_min_size = 1
    temporal_downsample_factor = 4

    def _encode(self, x, memory_state=0, memory_cache=None):
        if memory_state != ACTIVE:  # the first frame alone makes the first latent frame
            x = torch.cat((x[:, :, :1].expand(-1, -1, 3, -1, -1), x), dim=2)
        mean = torch.nn.functional.avg_pool3d(x.float(), (4, 8, 8))
        mean = mean + torch.linspace(0.0, 0.25, mean.shape[-1])
        mean = mean.repeat(1, 6, 1, 1, 1)[:, :16]
        return torch.cat((mean, torch.zeros_like(mean)), dim=1).to(x.dtype)

    def _decode(self, z, memory_state=0, memory_cache=None):
        x = z[:, :3].repeat_interleave(4, dim=2).repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)
        if memory_state != ACTIVE:  # the first latent frame makes one frame
            x = x[:, :, 3:]
        return (x + torch.linspace(-0.25, 0.25, x.shape[-1])).to(z.dtype)


class VAE:
    first_stage_model = VideoAutoencoderKLWrapper()
    vae_dtype = torch.float16
    device = CPU
    patcher = None
    disable_offload = True
    process_output = staticmethod(lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0))  # comfy/sd.py


def peak_bytes(fn):
    """fn() and the most CPU tensor memory it allocated at once: the CPU allocator's running total
    over its allocations and frees during the call, from the profiler's trace, less the total it
    started from."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        result = fn()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            events = json.load(f)["traceEvents"]
    memory = sorted((e for e in events if e.get("name") == "[memory]" and e["args"]["Device Type"] == 0), key=lambda e: e["ts"])
    start = memory[0]["args"]["Total Allocated"] - memory[0]["args"]["Bytes"]
    return result, max(e["args"]["Total Allocated"] for e in memory) - start


def nbytes(*tensors):
    return sum(t.numel() * t.element_size() for t in tensors)


def transient_growth(make, run, sizes):
    """Bytes the transient (the peak minus the outputs) of `run(make(size))` grows by per output frame,
    from sizes[0] to sizes[1]; the input is made before the measurement, and a first untimed run takes
    the one-time allocations (lazy imports) out of it. `run` returns (outputs, output frame count)."""
    run(make(sizes[0]))
    held = []
    for size in sizes:
        given = make(size)
        (outputs, count), peak = peak_bytes(lambda: run(given))
        held.append((peak - nbytes(*outputs), count))
    (t0, c0), (t1, c1) = held
    return (t1 - t0) / (c1 - c0)


@pytest.fixture
def stubbed(bcnodes, monkeypatch):
    for name in ("pipelines.seedvr2.encode", "pipelines.seedvr2.decode"):
        monkeypatch.setattr(bcnodes[name], "make_room_for_vae", lambda vae, needed: None)
    return bcnodes


# -- SeedVR2 Resize -------------------------------------------------------------------------------

def _clip(n, h=48, w=80):
    return torch.rand(n, h, w, 3, generator=torch.Generator().manual_seed(n))


def _lanczos_whole_clip(image, factor):
    """ImageScaleBy(lanczos): every frame through 8-bit PIL LANCZOS, the list stacked."""
    from PIL import Image

    size = (round(image.shape[2] * factor), round(image.shape[1] * factor))
    frames = [np.array(Image.fromarray(np.clip(255.0 * f.numpy(), 0, 255).astype(np.uint8)).resize(size, Image.Resampling.LANCZOS))
              for f in image]
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def test_resize_equals_the_whole_clip_downscale(bcnodes):
    flow = bcnodes["pipelines.seedvr2.resize"]
    image = _clip(10)
    # 48x80 -> 0.5 lanczos 24x40; resolution from the original: 48 * 1.5 = 72, which is 24 * 3 on the downscaled clip
    out = flow.resize(image, 1.5, 0.5, 0, False)
    expected = flow.resize(_lanczos_whole_clip(image, 0.5), 3.0, 1.0, 0, False)
    assert out[0].shape == (13, 80, 128, 3) and out[1].shape == (10, 72, 120, 3)
    assert torch.equal(out[0], expected[0]) and torch.equal(out[1], expected[1])


def test_resize_holds_no_downscaled_clip(bcnodes, monkeypatch):
    flow = bcnodes["pipelines.seedvr2.resize"]
    seen = []
    lanczos = flow.lanczos_scale_by

    def spy(frames, factor):
        seen.append(frames.shape[0])
        return lanczos(frames, factor)

    monkeypatch.setattr(flow, "lanczos_scale_by", spy)
    growth = transient_growth(_clip, lambda image: (flow.resize(image, 1.5, 0.5, 0, False), image.shape[0]), (32, 64))
    assert max(seen) == 4  # four frames at a time
    assert growth < 24 * 40 * 3 * 4 / 4, growth  # a quarter of one downscaled float32 frame


# -- SeedVR2 VAE Encode ---------------------------------------------------------------------------

def _pixels(n, h, w):
    return torch.rand(n, h, w, 3, generator=torch.Generator().manual_seed(n)).to(torch.float16)


def _encoded_whole(model, pixels):
    """The tile's mode in one call, as the slices give it: (1, 16, T, h, w) in the VAE dtype."""
    x = (pixels.movedim(-1, 1).permute(1, 0, 2, 3).unsqueeze(0) * 2.0 - 1.0).to(torch.float16)
    return torch.chunk(model._encode(x), 2, dim=1)[0]


def _encode_whole_clip(bcnodes, pixels, tile, overlap):
    """The slices in the VAE dtype (one tile) or the float32 tile blend cast to it (tiles), then
    cropped, cast to float32 and scaled."""
    tiling = bcnodes["models.seedvr2.tiling"]
    model = VAE.first_stage_model
    n, height, width = pixels.shape[:3]
    target_t, target_h, target_w = (n + 3) // 4, (height + 7) // 8, (width + 7) // 8
    overlap = min(overlap, max(0, tile - 8))
    if height <= tile and width <= tile:
        z = _encoded_whole(model, pixels)
    else:
        ranges, ramp = tiling.tile_plan(height, width, tile, overlap, CPU)
        result = torch.zeros((1, 16, target_t, target_h, target_w))
        count = torch.zeros((1, 1, 1, target_h, target_w))
        for y0, y1, x0, x1 in ranges:
            part = _encoded_whole(model, pixels[:, y0:y1, x0:x1])[:, :, :target_t]
            th, tw = part.shape[3], part.shape[4]
            weight = tiling.tile_weight(y0, y1, x0, x1, height, width, th, tw, overlap // 8, overlap // 8, ramp, CPU)
            result[:, :, :, y0 // 8:y0 // 8 + th, x0 // 8:x0 // 8 + tw] += part.mul_(weight).float()
            count[:, :, :, y0 // 8:y0 // 8 + th, x0 // 8:x0 // 8 + tw] += weight
        z = (result / count.clamp(min=1e-6)).to(torch.float16)
    return z[:, :, :target_t, :target_h, :target_w].to(torch.float32).contiguous() * SCALE


@pytest.mark.parametrize("tile", [128, 32])
def test_encode_equals_the_whole_clip_way(stubbed, tile):
    pixels = _pixels(13, 48, 80)  # 13 frames: slices of 5, 4 and 4; tile 32 / overlap 16: 2 x 4 tiles
    z = stubbed["pipelines.seedvr2.encode"].encode(pixels, VAE(), tile, 16)[0]["samples"]
    assert z.shape == (1, 16, 4, 6, 10) and z.dtype == torch.float32 and z.is_contiguous()
    assert torch.equal(z, _encode_whole_clip(stubbed, pixels, tile, 16))


@pytest.mark.parametrize("tile", [128, 32])
def test_encode_holds_the_latent_once(stubbed, tile):
    encode = stubbed["pipelines.seedvr2.encode"].encode

    def run(pixels):
        z = encode(pixels, VAE(), tile, 16)[0]["samples"]
        return (z,), z.shape[2]

    # long enough for the latent to outweigh one slice's working set; in float32 latent frames: one tile
    # and tiles hold nothing (tiles round their sum through float16 a chunk of latent frames at a time;
    # the whole sum cast at once would be 0.5, the slices listed or summed, cast, copied to float32 and
    # scaled into a new tensor 2.5)
    assert transient_growth(lambda n: _pixels(n, 48, 80), run, (513, 1025)) < 0.25 * 16 * 6 * 10 * 4


# -- SeedVR2 VAE Decode ---------------------------------------------------------------------------

def _latent(t, h=12, w=20):
    return torch.rand(1, 16, t, h, w, generator=torch.Generator().manual_seed(t)) * 2.0 - 1.0


def _decode_whole_clip(bcnodes, z, tile, overlap):
    """The tiles summed into a float16 (B, 3, T, H, W) clip, then normalised in float32, cast
    through the VAE dtype, put in range and stored channels last."""
    tiling = bcnodes["models.seedvr2.tiling"]
    model = VAE.first_stage_model
    latent = z.to(torch.float16) / SCALE + 0.0
    b, _, t, h, w = latent.shape
    if tile < overlap * 4:
        overlap = tile // 4
    tile_lat = tile // 8
    ov_lat = min(min(overlap, tile - 8) // 8, tile_lat - 1)
    result = torch.zeros((b, 3, 4 * t - 3, h * 8, w * 8), dtype=torch.float16)
    count = torch.zeros((1, 1, 1, h * 8, w * 8))
    ranges, ramp = tiling.tile_plan(h, w, tile_lat, ov_lat, CPU)
    for y0, y1, x0, x1 in ranges:
        decoded = model._decode(latent[:, :, :, y0:y1, x0:x1])
        th, tw = decoded.shape[3], decoded.shape[4]
        weight = tiling.tile_weight(y0, y1, x0, x1, h, w, th, tw, ov_lat * 8, ov_lat * 8, ramp, CPU)
        result[:, :, :, y0 * 8:y0 * 8 + th, x0 * 8:x0 * 8 + tw] += decoded.mul_(weight)
        count[:, :, :, y0 * 8:y0 * 8 + th, x0 * 8:x0 * 8 + tw] += weight
    out = (result.float() / count.clamp(min=1e-6)).to(torch.float16)
    out = VAE.process_output(out.float()).to(torch.float16).movedim(1, -1)
    return out.reshape(-1, h * 8, w * 8, 3)


def test_decode_equals_the_whole_clip_way(stubbed):
    z = _latent(5)  # 17 frames of 96x160; tile 64 / overlap 16: latent tiles of 8 with 2 overlapping, 2 x 3 tiles
    out = stubbed["pipelines.seedvr2.decode"].decode({"samples": z}, VAE(), 64, 16)[0]
    expected = _decode_whole_clip(stubbed, z, 64, 16)
    assert out.shape == (17, 96, 160, 3) and out.dtype == torch.float16
    assert torch.equal(out, expected)
    assert out.float().std() > 0.1  # the stand-in is not a constant


@pytest.mark.parametrize("tile", [256, 64])
def test_decode_holds_the_clip_once(stubbed, tile):
    decode = stubbed["pipelines.seedvr2.decode"].decode

    def run(z):
        out = decode({"samples": z}, VAE(), tile, 16)[0]
        return (out,), out.shape[0]

    assert transient_growth(_latent, run, (9, 17)) < 96 * 160 * 3 * 2 / 4  # a quarter of one output frame


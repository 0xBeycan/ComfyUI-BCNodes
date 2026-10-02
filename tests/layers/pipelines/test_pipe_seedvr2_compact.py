"""SeedVR2 Preprocess (Compact) -> sampler -> SeedVR2 PostProcess (Compact) gives what today's chain
(Resize -> VAE Encode -> sampler -> VAE Decode -> PostProcess) gives, bit for bit, and holds less:

  - equal outputs (torch.equal, same dtype) for every colour-correction method, downscale 0.5 / 0.75 /
    1.0, upscale 1.5 / 2.0, a max_resolution cap, frame counts that are padded to 4n+1 and sizes
    padded to /16 (rows and columns cut again at the end), one-tile and tiled encode and decode;
  - an 8-bit clip given as float16 gives the frames the same clip gives as float32, through both
    chains (a float16 IMAGE is requantized to k/255 before the resize);
  - VAE Decode's `keep` stores exactly the frames, rows and columns of the full decode it keeps;
  - Preprocess (Compact) returns the latent and a few numbers, and nothing else stays allocated
    once it returns; while it runs it holds the padded clip once (no reference next to it);
  - PostProcess (Compact) holds no clip-sized buffer besides its output.

The VAE is the stand-in of test_pipe_seedvr2_memory.py; the colour transfers are reference-dependent
stand-ins (the stub ones pass the content through), so a wrong reference shows in the output.
"""

import pytest
import torch

from test_pipe_seedvr2_memory import VAE, nbytes, peak_bytes, stubbed, transient_growth  # noqa: F401  (stubbed: a fixture)

METHODS = ["lab", "wavelet", "adain", "none"]
TRANSFERS = {
    "lab_color_transfer": lambda content, style: content * 0.75 + style * 0.25,
    "wavelet_color_transfer": lambda content, style: content - content.mean() + style.mean(dim=(2, 3), keepdim=True),
    "adain_color_transfer": lambda content, style: (content - content.mean()) / (content.std() + 1e-5) * style.std() + style.mean(),
}


@pytest.fixture
def m(stubbed, monkeypatch):
    import sys

    for name, fn in TRANSFERS.items():
        monkeypatch.setattr(sys.modules["comfy.ldm.seedvr.color_fix"], name, fn)
    return stubbed


def sampler(latent):
    """The sampler stand-in: every latent value changed, deterministically."""
    z = latent["samples"]
    return {"samples": z * 0.8 + torch.linspace(-0.2, 0.2, z.shape[-1])}


def old_chain(m, image, up, down, max_res, tile, methods=METHODS):
    """Today's chain through its four flows: {method: frames}."""
    pixels, reference = m["pipelines.seedvr2.resize"].resize(image, up, down, max_res, False)
    latent = m["pipelines.seedvr2.encode"].encode(pixels, VAE(), tile, 16)[0]
    images = m["pipelines.seedvr2.decode"].decode(sampler(latent), VAE(), tile, 16)[0]
    return {method: m["pipelines.seedvr2.postprocess"].process(images, reference, method)[0] for method in methods}


def compact_chain(m, image, up, down, max_res, tile, methods=METHODS):
    """The compact pair: {method: frames}."""
    flow = m["pipelines.seedvr2.compact"]
    latent, plan = flow.preprocess(image, VAE(), up, down, max_res, False, tile, 16)
    return {method: flow.postprocess(sampler(latent), VAE(), image, plan, tile, 16, method)[0] for method in methods}


def _clip(n, h, w):
    return torch.rand(n, h, w, 3, generator=torch.Generator().manual_seed(n * 1000 + h))


# frames, height, width, upscale, downscale, max_resolution, tile (512 covers every frame here: one tile;
# 32: tiled encode and decode). The resized size, the padded one and the reference (= output) size:
CASES = [
    (7, 30, 52, 1.5, 0.5, 0, 512),  # 15x26 -> 45x78, padded 48x80 and 9 frames; output 44x78, 7 frames
    (10, 30, 52, 1.5, 0.75, 0, 32),  # 22x39 -> 45x79, padded 48x80 and 13 frames; output 44x78
    (9, 36, 60, 1.5, 1.0, 0, 32),  # 54x90, padded 64x96, 9 frames; output 54x90
    (13, 20, 34, 2.0, 0.5, 0, 512),  # 10x17 -> 40x68, padded 48x80; output 40x68
    (6, 22, 30, 2.0, 1.0, 50, 32),  # 44x60 capped to 37x50, padded 48x64 and 9 frames; output 36x50
    (1, 30, 52, 2.0, 0.75, 0, 32),  # one frame: 22x39 -> 60x106, padded 64x112; output 60x106
]


@pytest.mark.parametrize("n, h, w, up, down, max_res, tile", CASES)
def test_the_compact_chain_equals_todays(m, n, h, w, up, down, max_res, tile):
    image = _clip(n, h, w)
    old, new = old_chain(m, image, up, down, max_res, tile), compact_chain(m, image, up, down, max_res, tile)
    for method in METHODS:
        assert new[method].dtype == old[method].dtype == torch.float16, method
        assert new[method].shape == old[method].shape and new[method].shape[0] == n, method
        assert torch.equal(new[method], old[method]), method
    assert not torch.equal(old["lab"], old["none"]) and not torch.equal(old["adain"], old["wavelet"])  # the reference matters


def test_an_8_bit_clip_as_float16_gives_the_float32_frames(m):
    levels = torch.randint(0, 256, (6, 30, 52, 3), generator=torch.Generator().manual_seed(1))
    as32 = levels.float() / 255
    as16 = as32.to(torch.float16)
    assert not torch.equal(as16.float(), as32)  # float16(k/255) is not float32(k/255)
    for down in (0.5, 1.0):
        old32 = old_chain(m, as32, 1.5, down, 0, 32)
        for chain in (old_chain, compact_chain):
            got = chain(m, as16, 1.5, down, 0, 32)
            assert all(torch.equal(got[method], old32[method]) for method in METHODS), (chain.__name__, down)


@pytest.mark.parametrize("tile", [512, 32])
def test_decode_keep_stores_the_cut_of_the_full_decode(m, tile):
    decode = m["pipelines.seedvr2.decode"].decode
    z = torch.rand(1, 16, 4, 6, 10, generator=torch.Generator().manual_seed(4)) * 2.0 - 1.0  # 13 frames of 48x80
    full = decode({"samples": z}, VAE(), tile, 16)[0]
    for keep in ((11, 44, 78), (13, 48, 80), (20, 60, 90), (1, 2, 2)):
        cut = decode({"samples": z}, VAE(), tile, 16, keep=keep)[0]
        expected = full[:keep[0], :keep[1], :keep[2]]
        assert cut.shape == expected.shape and cut.is_contiguous() and torch.equal(cut, expected), keep


def test_postprocess_checks_its_inputs_against_the_plan(m):
    flow = m["pipelines.seedvr2.compact"]
    image = _clip(5, 30, 52)
    latent, plan = flow.preprocess(image, VAE(), 1.5, 0.5, 0, False, 512, 16)
    assert plan == {"frames": 5, "height": 30, "width": 52, "downscale_factor": 0.5, "resolution": 45, "max_resolution": 0,
                    "out_height": 44, "out_width": 78}
    with pytest.raises(ValueError, match="the plan was made from"):
        flow.postprocess(latent, VAE(), image[:4], plan, 512, 16, "lab")
    with pytest.raises(ValueError, match="one video"):
        flow.postprocess({"samples": latent["samples"].repeat(2, 1, 1, 1, 1)}, VAE(), image, plan, 512, 16, "lab")
    with pytest.raises(ValueError, match="BC_SeedVR2PostProcessCompact: unknown color_correction_method"):
        flow.postprocess(latent, VAE(), image, plan, 512, 16, "hsv")
    with pytest.raises(ValueError, match="needs the SeedVR2 VAE"):
        flow.preprocess(image, object(), 1.5, 0.5, 0, False, 512, 16)


# -- memory ---------------------------------------------------------------------------------------

def held_after(fn):
    """fn() and the CPU tensor memory still allocated when it has returned (its result included), from the
    profiler's trace, less the total it started from."""
    import json
    import os
    import tempfile

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
    return result, memory[-1]["args"]["Total Allocated"] - start


@pytest.mark.parametrize("tile", [512, 32])
def test_preprocess_returns_the_latent_and_frees_the_padded_clip(m, tile):
    flow = m["pipelines.seedvr2.compact"]
    image = _clip(33, 30, 52)
    flow.preprocess(image[:5], VAE(), 1.5, 0.5, 0, False, tile, 16)  # lazy imports out of the measurement
    (latent, plan), held = held_after(lambda: flow.preprocess(image, VAE(), 1.5, 0.5, 0, False, tile, 16))
    assert set(latent) == {"samples"} and all(isinstance(v, (int, float)) for v in plan.values())
    assert held == nbytes(latent["samples"])  # the padded clip and every working buffer are gone

    def run(image):
        latent, _ = flow.preprocess(image, VAE(), 1.5, 0.5, 0, False, tile, 16)
        return (latent["samples"],), image.shape[0]

    # while it runs: the padded float16 clip once (48x80 per frame), no reference clip beside it
    padded_frame = 48 * 80 * 3 * 2
    growth = transient_growth(lambda n: _clip(n, 30, 52), run, (65, 129))
    assert padded_frame * 0.75 < growth < padded_frame * 1.25, growth


@pytest.mark.parametrize("tile, method", [(512, "lab"), (32, "adain"), (32, "none")])
def test_postprocess_holds_only_its_output(m, tile, method):
    flow = m["pipelines.seedvr2.compact"]

    def make(n):
        image = _clip(n, 30, 52)
        latent, plan = flow.preprocess(image, VAE(), 1.5, 0.5, 0, False, tile, 16)
        return sampler(latent), image, plan

    def run(given):
        out = flow.postprocess(given[0], VAE(), given[1], given[2], tile, 16, method)[0]
        return (out,), out.shape[0]

    assert transient_growth(make, run, (17, 33)) < 44 * 78 * 3 * 2 / 4  # a quarter of one output frame

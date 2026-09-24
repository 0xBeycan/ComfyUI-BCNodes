"""Golden: the four SeedVR2 nodes end to end on CPU (plan 8.2 row 22).

Every case enters through a node class, which stays in nodes/seedvr2.py, so there is no WHERE.
  - Resize for (upscale, downscale, max_resolution, emulate_bf16) = (1.0, 0.5, 0, F),
    (1.0, 0.5, 96, F), (3.2, 1.0, 0, F), (2.0, 1.0, 0, T) and the widget defaults
    (2.0, 0.5, 4096, T): image and reference digests and the log lines (the CPU warning of
    emulate_bf16). Every downscale below 1 runs lanczos_scale_by.
  - VAE Encode on a FakeVAE whose latents follow its input: the samples digest, the call log
    (every _encode input digest, memory state and cache, soft_empty_cache, get_free_memory,
    unload_all_models, load_models_gpu args, ProgressBar), the _Progress / make-room log lines,
    and the model's device attribute afterwards; tile/overlap 4096/256, 32/16, 40/64 (overlap
    clamp), a 69-frame clip (crosses TRIM_EVERY), use_slicing False, and an RGBA clip with
    the free memory short: 1 MiB free against a ~20 MB working set, so make-room unloads all
    models and reads the free memory again (30 GiB).
  - VAE Decode likewise, incl. the VAEDecodeTiled overlap rule (64/32 -> 16), a batch of 2, an
    18-latent-frame clip (69 frames, crosses TRIM_EVERY), use_slicing False, both RuntimeError
    texts (a decoder that returns one frame too many / too few per call) and the ValueError
    texts, incl. a wrong VAE with a malformed LATENT (the VAE check comes first).
  - PostProcess with distinct stub transfers (lab 0.5c+0.5s, wavelet c-0.1s, adain 0.9s) on the
    five synthetic cases of tests/parity_seedvr2_video.py postprocess_cases (its `clip`
    generator and Generator(0) chain, copied here so this file's inputs never change) x the
    four methods. The cases are built with one intra-op thread: on this machine the bytes of
    `clip`'s bilinear interpolate differ between 1 and 5 threads, and the thread count is not
    what the golden pins. The node call runs at the ambient thread count.
  - The IMAGE-batch checks (C20): (type, message) for None, a 3-D tensor and an empty batch
    into Resize `image`, Encode `pixels`, and PostProcess `images` and
    `original_resized_images`.

comfy.model_management, comfy.utils.ProgressBar and comfy.ldm.seedvr.color_fix are the
stub_comfy modules with recorders; time.monotonic is a counter. Recorded with
BCNODES_GOLDEN_RECORD=1 on FIXED_BASE.
"""

import itertools
import logging
import sys
import time

import pytest
import torch
import torch.nn.functional as F

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
    'torchvision': '0.29.0',
}

GOLDEN = {
    'resize/1.0/0.5/0/False': '8ac8352c1ef2d965d534ae2ba3664e47',
    'resize/1.0/0.5/96/False': '7124613e162116e019fdc713709d753a',
    'resize/3.2/1.0/0/False': 'a492b53c9270566bd30ae8a326bce488',
    'resize/2.0/1.0/0/True': 'ebaef04243c8bcd560b9ece44a9e8dea',
    'resize/2.0/0.5/4096/True': '4b10070395d64938ea7de31cb03468cb',
    'encode/single_4096_256': '523f4dcd58c82744842ffa6af82645e1',
    'encode/tiled_32_16': '79f7cd92ab68a8a3928f444b6b00aace',
    'encode/tiled_40_64_overlap_clamp': 'b5046b1e5f3e3bfd45482b8de957b0ae',
    'encode/clip_69_frames': '9a1c4a4daeb6fe2fe2cbe5ca125a5f50',
    'encode/no_slicing': 'a30801a4b52688b4ac4307115b583aee',
    'encode/rgba_free_memory_short': 'a933ce6148a1cc2909abc8d918d25f12',
    'decode/single_4096_256': '63f07d856a5d2e828a65d19550bd597c',
    'decode/tiled_32_16_batch_2': '779eea64c90f11824781fd9581a6d424',
    'decode/tiled_64_32_overlap_rule': '173e195a38d837318d89b73429af18c9',
    'decode/clip_18_latent_frames': 'e17e458fa4814bc06607ebee47924032',
    'decode/no_slicing': '9b8d230132d877f9c37cfe024654265a',
    'decode/error_one_frame_too_many': '14e8c408713ce8fb8b50d1b815867949',
    'decode/error_one_frame_too_few': '5b4b32c6eb2e9d6837bcd298d06663ca',
    'decode/latent_4d': '2effed7ad614d61a7238d39b92cb4127',
    'decode/latent_8_channels': '5b1e47249d6ea3300fad8fdc4332c2d3',
    'decode/wrong_vae': 'b1eb53fc94dfb29ae45ee8e4ad234ae1',
    'decode/wrong_vae_and_latent_4d': 'b1eb53fc94dfb29ae45ee8e4ad234ae1',
    'postprocess/5_frames_vs_3_ref/lab': '8e20fe931f44655c5fdf29495aa9ad53',
    'postprocess/5_frames_vs_3_ref/wavelet': '0d693618e3b677f443cef18dc61cb7f1',
    'postprocess/5_frames_vs_3_ref/adain': '170ce1c5ad7b5ab6fb2b4ad331bbbb46',
    'postprocess/5_frames_vs_3_ref/none': '997c70d754a469b0dae03a2dc06a52ef',
    'postprocess/odd_sizes/lab': 'e864ad4602ebcce35dc147ca8d0c72ab',
    'postprocess/odd_sizes/wavelet': '6e7dea1beb91d99271a02018d2056f37',
    'postprocess/odd_sizes/adain': '037b05759a764e88ca7ad9d38fc01d9e',
    'postprocess/odd_sizes/none': '875697138c7d9e38ef3964c481591e0d',
    'postprocess/reference_larger/lab': '52faa1cdf01efd0e8686e80be2be73cd',
    'postprocess/reference_larger/wavelet': '145719323a5f3d311cdaa3523d8d2287',
    'postprocess/reference_larger/adain': '23ce3cb3882dcbc17bf348221d5ce755',
    'postprocess/reference_larger/none': '1e1f55ddd0031cb72532aa5138dd35b2',
    'postprocess/alpha_reference/lab': 'f2f5a3c122de5655a7d3da1b81886a6e',
    'postprocess/alpha_reference/wavelet': '686464382aa083e4d6cf1738ae56dc54',
    'postprocess/alpha_reference/adain': 'a685d26f86e982778cd7670fda6a34ec',
    'postprocess/alpha_reference/none': 'e7f22e67825d5fc02a52eca135fa835c',
    'postprocess/float16_decoded/lab': 'f19d2c61cbdcc5572424104ab9529413',
    'postprocess/float16_decoded/wavelet': '1b1e055f42eb63872dea3e88946ef9c0',
    'postprocess/float16_decoded/adain': 'c5340b7c365bcb5d94a265402f720167',
    'postprocess/float16_decoded/none': '6dd7139b8bd72e4b8c55e83fca1f4fc4',
    'postprocess/unknown_method': '02b86241db4a51db32ea4357d220a4c9',
    'batch_check/resize/image/none': '8730277bac3f168803eca833f6a3a4a8',
    'batch_check/resize/image/3d': '8730277bac3f168803eca833f6a3a4a8',
    'batch_check/resize/image/empty': '8730277bac3f168803eca833f6a3a4a8',
    'batch_check/encode/pixels/none': 'eb773cd8b2b5fd69256dc06181d27e99',
    'batch_check/encode/pixels/3d': 'eb773cd8b2b5fd69256dc06181d27e99',
    'batch_check/encode/pixels/empty': 'eb773cd8b2b5fd69256dc06181d27e99',
    'batch_check/postprocess/images/none': '393b7e9c2cd9fe9ffa777aead4d3d9f5',
    'batch_check/postprocess/images/3d': '393b7e9c2cd9fe9ffa777aead4d3d9f5',
    'batch_check/postprocess/images/empty': '393b7e9c2cd9fe9ffa777aead4d3d9f5',
    'batch_check/postprocess/original_resized_images/none': '366122cb253ae6e7072f6a357d20c580',
    'batch_check/postprocess/original_resized_images/3d': '366122cb253ae6e7072f6a357d20c580',
    'batch_check/postprocess/original_resized_images/empty': '366122cb253ae6e7072f6a357d20c580',
}

GiB = 2 ** 30
CALLS = []


def _state(name):
    return getattr(sys.modules["comfy.ldm.seedvr.vae"].MemoryState, name)


class VideoAutoencoderKLWrapper:
    """Stands in for comfy's SeedVR2 VAE model: 4 pixel frames <-> 1 latent frame (a first
    slice of 1 + 4k <-> 1 + k), 8x spatial. Its outputs follow its inputs, the slice's memory
    cache and the position in the tile, so a changed slice, cache or blend shows."""

    def __init__(self, use_slicing=True, frame_delta=0):
        self.use_slicing = use_slicing
        self.slicing_sample_min_size = 4
        self.slicing_latent_min_size = 1
        self.temporal_downsample_factor = 4
        self.frame_delta = frame_delta
        self.device = "unset"
        self.caches = []  # kept alive, so identity numbers are never reused

    def _cache(self, memory_cache):
        if memory_cache is None:
            return None
        for i, c in enumerate(self.caches):
            if c is memory_cache:
                return i
        self.caches.append(memory_cache)
        return len(self.caches) - 1

    def _step(self, memory_cache):
        if memory_cache is None:
            return 0
        n = memory_cache.get("n", 0)
        memory_cache["n"] = n + 1
        return n

    def _encode(self, x, memory_state=None, memory_cache=None):
        CALLS.append(("_encode", tuple(x.shape), str(x.dtype), str(x.device), memory_state, self._cache(memory_cache), digest(x)))
        t = x.shape[2] // 4 if memory_state == _state("ACTIVE") else (x.shape[2] + 3) // 4
        pooled = F.adaptive_avg_pool3d(x.float(), (t, x.shape[3] // 8, x.shape[4] // 8))
        mode = torch.stack([pooled[:, k % 3] * (0.5 + k / 16) - k / 32 for k in range(16)], dim=1)
        mode = mode + 0.01 * self._step(memory_cache)
        return torch.cat([mode, torch.full_like(mode, -3.0)], dim=1).to(torch.float16)

    def _decode(self, z, memory_state=None, memory_cache=None):
        CALLS.append(("_decode", tuple(z.shape), str(z.dtype), str(z.device), memory_state, self._cache(memory_cache), digest(z)))
        zf = z.float()
        base = torch.stack([zf[:, 0] + 0.5 * zf[:, 3], zf[:, 1] - 0.25 * zf[:, 7], 0.75 * zf[:, 2] + 0.1 * zf[:, 15]], dim=1)
        frames = base.repeat_interleave(4, dim=2)
        if memory_state != _state("ACTIVE"):
            frames = frames[:, :, 3:]
        frames = frames.repeat_interleave(8, dim=3).repeat_interleave(8, dim=4)
        frames = (frames + torch.linspace(-0.2, 0.2, frames.shape[-1]).view(1, 1, 1, 1, -1)
                  + 0.02 * torch.arange(frames.shape[2]).view(1, 1, -1, 1, 1) + 0.005 * self._step(memory_cache))
        if self.frame_delta > 0:
            frames = torch.cat([frames, frames[:, :, -1:].expand(-1, -1, self.frame_delta, -1, -1)], dim=2)
        elif self.frame_delta < 0:
            frames = frames[:, :, :self.frame_delta]
        return frames.tanh().to(torch.float16)


class FakeVAE:
    def __init__(self, **model_kwargs):
        self.first_stage_model = VideoAutoencoderKLWrapper(**model_kwargs)
        self.vae_dtype = torch.float16
        self.device = torch.device("cpu")
        self.patcher = "patcher"
        self.disable_offload = True

    @staticmethod
    def process_output(image):  # comfy/sd.py VAE.process_output
        return torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)


class ProgressBar:
    def __init__(self, total):
        CALLS.append(("ProgressBar", total))

    def update(self, n):
        CALLS.append(("ProgressBar.update", n))


@pytest.fixture
def seams(bcnodes, monkeypatch, caplog):
    """Recorders on the comfy stubs, a monotonic counter and INFO logging; returns
    (node module, free-memory script setter, logs())."""
    CALLS.clear()
    free = []
    mm = sys.modules["comfy.model_management"]

    def get_free_memory(*args, **kwargs):
        CALLS.append(("get_free_memory", args, kwargs))
        return free.pop(0) if free else 2 ** 40

    monkeypatch.setattr(mm, "get_free_memory", get_free_memory, raising=True)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: CALLS.append(("soft_empty_cache",)), raising=True)
    monkeypatch.setattr(mm, "unload_all_models", lambda: CALLS.append(("unload_all_models",)), raising=True)
    monkeypatch.setattr(mm, "load_models_gpu", lambda *a, **kw: CALLS.append(("load_models_gpu", a, kw)), raising=True)
    monkeypatch.setattr(sys.modules["comfy.utils"], "ProgressBar", ProgressBar, raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    caplog.set_level(logging.INFO)
    return bcnodes["seedvr2"], free.extend, lambda: [(r.levelname, r.getMessage()) for r in caplog.records]


def _call(fn, *args):
    try:
        return {"out": fn(*args)}
    except Exception as e:
        return {"exception": (type(e).__name__, str(e))}


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("up, down, max_res, bf16", [
    (1.0, 0.5, 0, False), (1.0, 0.5, 96, False), (3.2, 1.0, 0, False), (2.0, 1.0, 0, True), (2.0, 0.5, 4096, True),
])
def test_resize(up, down, max_res, bf16, seams):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, logs = seams
    image = torch.rand((6, 48, 104, 4), generator=torch.Generator().manual_seed(1))
    out = module.SeedVR2Resize().resize(image, up, down, max_res, bf16)
    check(GOLDEN, f"resize/{up}/{down}/{max_res}/{bf16}", digest({"out": out, "logs": logs()}))


# ---------------------------------------------------------------------------
# VAE Encode
# ---------------------------------------------------------------------------

ENCODE_CASES = {  # name -> (frames, h, w, channels, tile, overlap, use_slicing, free memory script)
    "single_4096_256": (9, 48, 64, 3, 4096, 256, True, []),
    "tiled_32_16": (9, 48, 64, 3, 32, 16, True, []),
    "tiled_40_64_overlap_clamp": (9, 48, 64, 3, 40, 64, True, []),
    "clip_69_frames": (69, 16, 16, 3, 4096, 256, True, []),
    "no_slicing": (9, 48, 64, 3, 4096, 256, False, []),
    "rgba_free_memory_short": (5, 32, 40, 4, 4096, 256, True, [2 ** 20, 30 * GiB]),
}


@pytest.mark.parametrize("name", list(ENCODE_CASES))
def test_encode(name, seams):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, set_free, logs = seams
    n, h, w, c, tile, overlap, use_slicing, free = ENCODE_CASES[name]
    set_free(free)
    pixels = torch.rand((n, h, w, c), generator=torch.Generator().manual_seed(1)).half()
    vae = FakeVAE(use_slicing=use_slicing)
    out = module.SeedVR2VAEEncode().encode(pixels, vae, tile, overlap)
    check(GOLDEN, f"encode/{name}", digest({"out": out, "calls": CALLS, "logs": logs(),
                                             "model_device": vae.first_stage_model.device}))


# ---------------------------------------------------------------------------
# VAE Decode
# ---------------------------------------------------------------------------

DECODE_CASES = {  # name -> (latent shape (b, t, h, w), tile, overlap, model kwargs)
    "single_4096_256": ((1, 3, 10, 12), 4096, 256, {}),
    "tiled_32_16_batch_2": ((2, 3, 10, 12), 32, 16, {}),
    "tiled_64_32_overlap_rule": ((1, 3, 10, 12), 64, 32, {}),
    "clip_18_latent_frames": ((1, 18, 4, 4), 4096, 256, {}),
    "no_slicing": ((1, 3, 10, 12), 4096, 256, {"use_slicing": False}),
    "error_one_frame_too_many": ((1, 3, 4, 4), 4096, 256, {"frame_delta": 1}),
    "error_one_frame_too_few": ((1, 3, 4, 4), 4096, 256, {"frame_delta": -1}),
}


def _latent(b, t, h, w):
    return torch.randn((b, 16, t, h, w), generator=torch.Generator().manual_seed(2)) * 0.8


@pytest.mark.parametrize("name", list(DECODE_CASES))
def test_decode(name, seams):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, logs = seams
    shape, tile, overlap, model_kwargs = DECODE_CASES[name]
    vae = FakeVAE(**model_kwargs)
    out = _call(module.SeedVR2VAEDecode().decode, {"samples": _latent(*shape)}, vae, tile, overlap)
    check(GOLDEN, f"decode/{name}", digest({**out, "calls": CALLS, "logs": logs(),
                                             "model_device": vae.first_stage_model.device}))


@pytest.mark.parametrize("name, samples, vae", [
    ("latent_4d", {"samples": torch.zeros(1, 16, 4, 4)}, FakeVAE),
    ("latent_8_channels", {"samples": torch.zeros(1, 8, 2, 4, 4)}, FakeVAE),
    ("wrong_vae", {"samples": torch.zeros(1, 16, 2, 4, 4)}, object),
    ("wrong_vae_and_latent_4d", {"samples": torch.zeros(1, 16, 4, 4)}, object),
])
def test_decode_value_errors(name, samples, vae, seams):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, _ = seams
    out = _call(module.SeedVR2VAEDecode().decode, samples, vae(), 4096, 256)
    check(GOLDEN, f"decode/{name}", digest({**out, "calls": CALLS}))


# ---------------------------------------------------------------------------
# PostProcess
# ---------------------------------------------------------------------------

def clip(frames, height, width, g, channels=3):
    """tests/parity_seedvr2_video.py `clip`, verbatim."""
    coarse = torch.rand(frames, channels, height // 8 + 1, width // 8 + 1, generator=g)
    smooth = torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    noise = torch.rand(frames, channels, height, width, generator=g) * 0.1
    return (smooth * 0.9 + noise).clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def postprocess_cases():
    """The five cases of parity_seedvr2_video.postprocess_cases, built in its order with one
    intra-op thread."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        g = torch.Generator().manual_seed(0)
        return {
            "5_frames_vs_3_ref": (clip(5, 64, 96, g), clip(3, 64, 96, g)),
            "odd_sizes": (clip(2, 67, 99, g), clip(2, 65, 97, g)),
            "reference_larger": (clip(1, 64, 96, g), clip(1, 80, 120, g)),
            "alpha_reference": (clip(2, 64, 96, g), clip(2, 64, 96, g, channels=4)),
            "float16_decoded": (clip(2, 64, 96, g).half(), clip(2, 64, 96, g)),
        }
    finally:
        torch.set_num_threads(threads)


TRANSFERS = {
    "lab_color_transfer": lambda c, s: 0.5 * c + 0.5 * s,
    "wavelet_color_transfer": lambda c, s: c - 0.1 * s,
    "adain_color_transfer": lambda c, s: 0.9 * s,
}


@pytest.fixture
def transfers(monkeypatch):
    color_fix = sys.modules["comfy.ldm.seedvr.color_fix"]
    for name, fn in TRANSFERS.items():
        def recorded(content, style, name=name, fn=fn):
            CALLS.append((name, tuple(content.shape), str(content.dtype), tuple(style.shape), str(style.dtype)))
            return fn(content, style)

        monkeypatch.setattr(color_fix, name, recorded, raising=True)


@pytest.mark.parametrize("method", ["lab", "wavelet", "adain", "none"])
@pytest.mark.parametrize("case", list(postprocess_cases()))
def test_postprocess(case, method, seams, transfers):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, _ = seams
    images, reference = postprocess_cases()[case]
    out = module.SeedVR2PostProcess().process(images, reference, method)
    check(GOLDEN, f"postprocess/{case}/{method}", digest({"out": out, "calls": CALLS}))


def test_postprocess_unknown_method(seams, transfers):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, _ = seams
    images, reference = postprocess_cases()["odd_sizes"]
    check(GOLDEN, "postprocess/unknown_method", digest(_call(module.SeedVR2PostProcess().process, images, reference, "hsv")))


# ---------------------------------------------------------------------------
# The IMAGE-batch checks (C20)
# ---------------------------------------------------------------------------

BAD = {"none": lambda: None, "3d": lambda: torch.rand(8, 8, 3), "empty": lambda: torch.zeros(0, 8, 8, 3)}
ENTRIES = {
    "resize/image": lambda m, x: m.SeedVR2Resize().resize(x, 2.0, 0.5, 4096, True),
    "encode/pixels": lambda m, x: m.SeedVR2VAEEncode().encode(x, FakeVAE(), 4096, 256),
    "postprocess/images": lambda m, x: m.SeedVR2PostProcess().process(x, torch.rand(1, 8, 8, 3), "lab"),
    "postprocess/original_resized_images": lambda m, x: m.SeedVR2PostProcess().process(torch.rand(1, 8, 8, 3), x, "lab"),
}


@pytest.mark.parametrize("bad", list(BAD))
@pytest.mark.parametrize("entry", list(ENTRIES))
def test_image_batch_check(entry, bad, seams, transfers):
    check_env(ENV, "torch", "numpy", "Pillow", "torchvision")
    module, _, _ = seams
    out = _call(ENTRIES[entry], module, BAD[bad]())
    assert "exception" in out
    check(GOLDEN, f"batch_check/{entry}/{bad}", digest({**out, "calls": CALLS}))

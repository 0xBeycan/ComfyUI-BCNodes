"""Dual Sampler's flow (pipelines/dual_sampler.py):

- the upscaled size: each side times scale_by, rounded to the nearest multiple of 16 px (2 cells of 8 px; one cell
  of a 16 or 32 px VAE);
- DyPE first: model_2 patched (the mode, the config, and no start sigma: DyPE reads the second pass's first model
  call's) before the first pass, so a model DyPE does not take stops the run before any sampling; no schedule is
  computed before the run; at denoise 0 no DyPE;
- the first pass: model_1 at denoise 1.0 with the seed, on the latent given;
- the second pass: model_2's DyPE clone, the seed itself or the next one (the last seed's next is 0);
- blend_factor 0: core's Upscale Latent By (common_upscale, no crop) alone, no decode or encode before the second
  pass, the decodes after it, only those wanted (an unwanted image a 0-frame tensor of the decode's shape);
- blend_factor 1: the pixel way alone (decode -> resize at the upscaled size in px -> encode), no latent upscale;
  the first image is that decode, never decoded twice, even when unwanted;
- between: core's Latent Blend, pixel * b + latent * (1 - b); the latent's other fields kept;
- a 5-D latent (Qwen-Image's: a frame axis before height and width) goes through both ways and the blend as it is,
  its pixel way encoding each image of a batch on its own (core's Wan 2.1 VAE takes a batch for one video's frames);
- a VAE whose latent does not match the model's, batch included: an error naming both.

The sampler, core's latent upscale and schedule, DyPE's patch, the resize and the VAE are stand-ins recording calls.
"""

import sys
import types

import pytest
import torch

LAST_SEED = 0xFFFFFFFFFFFFFFFF
CONFIG = {"model": "Z-Image", "values": {"method": "ntk"}}


class VAE:
    def __init__(self, channels=16):
        self.channels = channels
        self.decoded, self.encoded = [], []

    def decode(self, samples):
        self.decoded.append(samples.clone())
        return torch.full((samples.shape[0], samples.shape[-2] * 8, samples.shape[-1] * 8, 3), 0.5)

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        return torch.full((pixels.shape[0], self.channels, pixels.shape[1] // 8, pixels.shape[2] // 8), 20.0)

    def spacial_compression_encode(self):
        return 8

    def spacial_compression_decode(self):
        return 8

    output_channels = 3

    def vae_output_dtype(self):
        return torch.float32


@pytest.fixture
def flow(bcnodes, monkeypatch, comfy_stubs):
    seen = {"passes": [], "upscales": [], "patches": [], "resizes": [], "order": []}

    def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
        seen["order"].append("pass")
        seen["passes"].append(dict(model=model, seed=seed, steps=steps, cfg=cfg, sampler=sampler_name, scheduler=scheduler,
                                   latent=latent, denoise=denoise))
        return {**latent, "samples": latent["samples"] + 1}

    def common_upscale(samples, width, height, method, crop):
        seen["upscales"].append((width, height, method, crop))
        return torch.full((*samples.shape[:-2], height, width), 10.0)

    def patch(model, mode, start_sigma, config):
        seen["order"].append("dype")
        seen["patches"].append((model, mode, start_sigma, config))
        if model == "unsupported" and mode != "disable":
            raise ValueError("DyPE supports Z-Image, Qwen-Image models; this one is Other.")
        return ("dype", model)

    def resize_image(image, mask, width, height, method, keep, *args, **kwargs):
        seen["resizes"].append((tuple(image.shape), width, height, method, keep))
        return torch.full((image.shape[0], height, width, 3), 0.25), width, height, None

    monkeypatch.setattr(bcnodes["libs.sampling"], "ksampler", ksampler)
    monkeypatch.setattr(sys.modules["comfy.utils"], "common_upscale", common_upscale, raising=False)
    monkeypatch.setattr(bcnodes["models.common.dype"], "patch", patch)
    monkeypatch.setattr(bcnodes["libs.resize"], "resize_image", resize_image)
    return bcnodes["pipelines.dual_sampler"], seen


def settings(module, **kw):
    values = dict(seed=5, first=module.Pass(9, 1.0, "dpmpp_sde", "beta", 1.0), second=module.Pass(8, 1.5, "euler", "simple", 0.45),
                  upscale_method="bislerp", scale_by=2.0, blend_factor=0.0, pixel_upscale_method="lanczos",
                  second_seed="increment", dype="auto", dype_config=CONFIG)
    values.update(kw)
    return module.Settings(**values)


def latent():
    return {"samples": torch.zeros(1, 16, 128, 192), "batch_index": [0]}


def test_the_upscaled_size(flow):
    module, _ = flow
    assert module.upscaled_size(128, 192, 2.0, 8) == (256, 384)
    assert module.upscaled_size(128, 192, 1.25, 8) == (160, 240)
    assert module.upscaled_size(100, 75, 1.3, 8) == (130, 98)  # 97.5 cells: 48.75 multiples of 2 -> 49
    assert module.upscaled_size(3, 3, 0.1, 8) == (2, 2)        # at least one multiple
    assert module.upscaled_size(64, 96, 1.5, 16) == (96, 144)   # 16 px cells: one cell
    assert module.upscaled_size(33, 47, 1.3, 32) == (43, 61)    # 32 px cells: one cell, not 0
    assert module.upscaled_size(1, 1, 0.1, 32) == (1, 1)


def test_blend_zero_is_the_core_chain(flow):
    module, seen = flow
    vae = VAE()
    first, second, out = module.run("m1", "m2", "pos", "neg", latent(), vae, settings(module))
    one, two = seen["passes"]
    assert (one["model"], one["seed"], one["steps"], one["cfg"], one["sampler"], one["scheduler"], one["denoise"]) == (
        "m1", 5, 9, 1.0, "dpmpp_sde", "beta", 1.0)
    assert seen["upscales"] == [(384, 256, "bislerp", "disabled")] and not seen["resizes"] and not vae.encoded
    assert seen["patches"] == [("m2", "auto", None, CONFIG)]
    assert seen["order"] == ["dype", "pass", "pass"]  # DyPE before any sampling
    assert (two["model"], two["seed"], two["steps"], two["cfg"], two["sampler"], two["scheduler"], two["denoise"]) == (
        ("dype", "m2"), 6, 8, 1.5, "euler", "simple", 0.45)
    assert torch.equal(two["latent"]["samples"], torch.full((1, 16, 256, 384), 10.0)) and two["latent"]["batch_index"] == [0]
    assert len(vae.decoded) == 2 and torch.equal(vae.decoded[0], torch.ones(1, 16, 128, 192))  # both after the second pass
    assert first.shape == (1, 1024, 1536, 3) and second.shape == (1, 2048, 3072, 3)
    assert torch.equal(out["samples"], torch.full((1, 16, 256, 384), 11.0))


def test_unwanted_images_are_not_decoded(flow):
    module, _ = flow
    vae = VAE()
    first, second, _ = module.run("m1", "m2", "p", "n", latent(), vae, settings(module), want_first=False, want_second=False)
    assert not vae.decoded
    assert first.shape == (0, 1024, 1536, 3) and second.shape == (0, 2048, 3072, 3) and first.dtype == torch.float32


def test_blend_one_is_the_pixel_way_alone(flow):
    module, seen = flow
    vae = VAE()
    first, _, _ = module.run("m1", "m2", "p", "n", latent(), vae, settings(module, blend_factor=1.0, pixel_upscale_method="bicubic"),
                             want_first=False)
    assert not seen["upscales"] and len(vae.decoded) == 2  # the first latent once (the pixel way), the second once
    assert seen["resizes"] == [((1, 1024, 1536, 3), 3072, 2048, "bicubic", "stretch")] and vae.encoded == [(1, 2048, 3072, 3)]
    assert torch.equal(seen["passes"][1]["latent"]["samples"], torch.full((1, 16, 256, 384), 20.0))
    assert first.shape == (1, 1024, 1536, 3)  # decoded for the pixel way; dropped by the node when unwanted


def test_blend_between_is_cores_latent_blend(flow):
    module, seen = flow
    vae = VAE()
    module.run("m1", "m2", "p", "n", latent(), vae, settings(module, blend_factor=0.3))
    assert seen["upscales"] == [(384, 256, "bislerp", "disabled")] and len(vae.encoded) == 1
    assert torch.equal(seen["passes"][1]["latent"]["samples"], torch.full((1, 16, 256, 384), 20.0) * 0.3 + torch.full((1, 16, 256, 384), 10.0) * (1 - 0.3))
    assert len(vae.decoded) == 2  # the first image is the pixel way's decode


def test_seeds_and_no_dype_at_denoise_zero(flow):
    module, seen = flow
    module.run("m1", "m2", "p", "n", latent(), VAE(), settings(module, second_seed="fixed"))
    assert seen["passes"][1]["seed"] == 5
    seen["passes"].clear()
    module.run("m1", "m2", "p", "n", latent(), VAE(), settings(module, seed=LAST_SEED))
    assert seen["passes"][1]["seed"] == 0
    seen["passes"].clear()
    seen["patches"].clear()
    module.run("m1", "m2", "p", "n", latent(), VAE(), settings(module, second=module.Pass(8, 1.0, "euler", "simple", 0.0)))
    assert not seen["patches"] and seen["passes"][1]["model"] == "m2"


def test_a_vae_of_another_latent_is_an_error(flow):
    module, _ = flow
    with pytest.raises(ValueError, match=r"latent of \(1, 4, 256, 384\).*the model's latent is \(1, 16, 256, 384\)"):
        module.run("m1", "m2", "p", "n", latent(), VAE(channels=4), settings(module, blend_factor=0.5))


def test_a_model_dype_does_not_take_stops_before_the_first_pass(flow):
    module, seen = flow
    with pytest.raises(ValueError, match="DyPE supports Z-Image, Qwen-Image models"):
        module.run("m1", "unsupported", "p", "n", latent(), VAE(), settings(module))
    assert seen["order"] == ["dype"] and not seen["passes"]
    seen["order"].clear()
    module.run("m1", "unsupported", "p", "n", latent(), VAE(), settings(module, dype="disable", dype_config=None))
    assert seen["order"] == ["dype", "pass", "pass"] and seen["passes"][1]["model"] == ("dype", "unsupported")


class VAE5D(VAE):
    """Qwen-Image's VAE (core's Wan 2.1 VAE): a latent with a frame axis, (B, C, 1, h, w); decodes to (B, 1, H, W, 3)."""

    def decode(self, samples):
        self.decoded.append(samples.clone())
        return torch.full((samples.shape[0], samples.shape[2], samples.shape[-2] * 8, samples.shape[-1] * 8, 3), 0.5)

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        return torch.full((pixels.shape[0], self.channels, 1, pixels.shape[1] // 8, pixels.shape[2] // 8), 20.0)


def test_a_5d_latent_goes_through_both_ways(flow):
    module, seen = flow
    lat = {"samples": torch.zeros(1, 16, 1, 128, 192)}
    for blend, want in ((0.0, torch.full((1, 16, 1, 256, 384), 10.0)), (1.0, torch.full((1, 16, 1, 256, 384), 20.0)),
                        (0.25, torch.full((1, 16, 1, 256, 384), 20.0) * 0.25 + torch.full((1, 16, 1, 256, 384), 10.0) * 0.75)):
        seen["passes"].clear()
        vae = VAE5D()
        first, second, out = module.run("m1", "m2", "p", "n", lat, vae, settings(module, blend_factor=blend))
        assert torch.equal(seen["passes"][1]["latent"]["samples"], want), blend
        assert first.shape == (1, 1024, 1536, 3) and second.shape == (1, 2048, 3072, 3) and out["samples"].ndim == 5
    with pytest.raises(ValueError, match=r"latent of \(1, 4, 1, 256, 384\).*the model's latent is \(1, 16, 1, 256, 384\)"):
        module.run("m1", "m2", "p", "n", lat, VAE5D(channels=4), settings(module, blend_factor=0.5))


class VideoVAE(VAE5D):
    """Core's VAE Encode with the Wan 2.1 VAE: an image batch is one video, its frames floored to 4n + 1 latent frames;
    each latent the mean of the pixels it encodes."""

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        frames = 1 + (pixels.shape[0] - 1) // 4
        return torch.full((1, self.channels, frames, pixels.shape[1] // 8, pixels.shape[2] // 8), float(pixels.mean()))


def test_a_5d_batch_is_encoded_image_by_image(flow, monkeypatch, bcnodes):
    module, seen = flow

    def resize_image(image, mask, width, height, *args, **kwargs):  # image i -> pixels of value i
        return torch.arange(image.shape[0], dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, height, width, 3), width, height, None

    monkeypatch.setattr(bcnodes["libs.resize"], "resize_image", resize_image)
    vae = VideoVAE()
    lat = {"samples": torch.zeros(3, 16, 1, 128, 192)}
    module.run("m1", "m2", "p", "n", lat, vae, settings(module, blend_factor=1.0))
    assert vae.encoded == [(1, 2048, 3072, 3)] * 3  # one image at a time, never the batch as a video
    got = seen["passes"][1]["latent"]["samples"]
    assert got.shape == (3, 16, 1, 256, 384) and [float(got[i].mean()) for i in range(3)] == [0.0, 1.0, 2.0]


def test_a_vae_dropping_the_batch_is_an_error(flow):
    module, _ = flow

    class OneLatent(VAE):
        def encode(self, pixels):
            return super().encode(pixels[:1])

    with pytest.raises(ValueError, match=r"latent of \(1, 16, 256, 384\).*the model's latent is \(2, 16, 256, 384\)"):
        module.run("m1", "m2", "p", "n", {"samples": torch.zeros(2, 16, 128, 192)}, OneLatent(), settings(module, blend_factor=0.5))

"""ZiT Dual Sampler's flow (pipelines/zit_dual_sampler.py):

- the upscaled size: each side times scale_by, rounded to the nearest multiple of 16 px (2 cells of 8 px);
- the first pass: model_1 at denoise 1.0 with the seed, on the latent given;
- the second pass: model_2 with DyPE (the mode, the config and the first sigma of the second pass's own schedule:
  core's KSampler at model_2, steps_2, sampler_2, scheduler_2, denoise_2), the seed itself or the next one (the last
  seed's next is 0); at denoise 0 no DyPE;
- blend_factor 0: core's Upscale Latent By (common_upscale, no crop) alone, no decode or encode before the second
  pass, the decodes after it, only those wanted (an unwanted image a 0-frame tensor of the decode's shape);
- blend_factor 1: the pixel way alone (decode -> resize at the upscaled size in px -> encode), no latent upscale;
  the first image is that decode, never decoded twice, even when unwanted;
- between: core's Latent Blend, pixel * b + latent * (1 - b); the latent's other fields kept;
- a VAE whose latent does not match the model's: an error naming both.

The sampler, core's latent upscale and schedule, DyPE's patch, the resize and the VAE are stand-ins recording calls.
"""

import sys
import types

import pytest
import torch

LAST_SEED = 0xFFFFFFFFFFFFFFFF


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
    seen = {"passes": [], "upscales": [], "schedules": [], "patches": [], "resizes": []}

    def ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
        seen["passes"].append(dict(model=model, seed=seed, steps=steps, cfg=cfg, sampler=sampler_name, scheduler=scheduler,
                                   latent=latent, denoise=denoise))
        return {**latent, "samples": latent["samples"] + 1}

    def common_upscale(samples, width, height, method, crop):
        seen["upscales"].append((width, height, method, crop))
        return torch.full((samples.shape[0], samples.shape[1], height, width), 10.0)

    class KSampler:
        def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
            seen["schedules"].append((model, steps, sampler, scheduler, denoise))
            self.sigmas = torch.tensor([0.6944, 0.5, 0.0])

    def patch(model, mode, start_sigma, config):
        seen["patches"].append((model, mode, start_sigma, config))
        return ("dype", model)

    def resize_image(image, mask, width, height, method, keep, *args, **kwargs):
        seen["resizes"].append((tuple(image.shape), width, height, method, keep))
        return torch.full((image.shape[0], height, width, 3), 0.25), width, height, None

    monkeypatch.setattr(bcnodes["libs.sampling"], "ksampler", ksampler)
    monkeypatch.setattr(sys.modules["comfy.utils"], "common_upscale", common_upscale, raising=False)
    samplers = types.ModuleType("comfy.samplers")
    samplers.KSampler = KSampler
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setattr(sys.modules["comfy"], "samplers", samplers, raising=False)
    monkeypatch.setattr(bcnodes["models.common.dype"], "patch", patch)
    monkeypatch.setattr(bcnodes["libs.resize"], "resize_image", resize_image)
    return bcnodes["pipelines.zit_dual_sampler"], seen


def settings(module, **kw):
    values = dict(seed=5, first=module.Pass(9, 1.0, "dpmpp_sde", "beta", 1.0), second=module.Pass(8, 1.5, "euler", "simple", 0.45),
                  upscale_method="bislerp", scale_by=2.0, blend_factor=0.0, pixel_upscale_method="lanczos",
                  second_seed="increment", dype="auto", dype_config={"method": "ntk"})
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


def test_blend_zero_is_the_core_chain(flow):
    module, seen = flow
    vae = VAE()
    first, second, out = module.run("m1", "m2", "pos", "neg", latent(), vae, settings(module))
    one, two = seen["passes"]
    assert (one["model"], one["seed"], one["steps"], one["cfg"], one["sampler"], one["scheduler"], one["denoise"]) == (
        "m1", 5, 9, 1.0, "dpmpp_sde", "beta", 1.0)
    assert seen["upscales"] == [(384, 256, "bislerp", "disabled")] and not seen["resizes"] and not vae.encoded
    assert seen["schedules"] == [("m2", 8, "euler", "simple", 0.45)]
    assert seen["patches"] == [("m2", "auto", pytest.approx(0.6944), {"method": "ntk"})]
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
    with pytest.raises(ValueError, match=r"latent of \(4, 256, 384\).*the model's latent is \(16, 256, 384\)"):
        module.run("m1", "m2", "p", "n", latent(), VAE(channels=4), settings(module, blend_factor=0.5))

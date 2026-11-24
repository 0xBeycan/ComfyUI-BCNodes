"""One sampler run as core's common_ksampler makes it (libs/sampling.py): the empty-latent fix gets the latent's
downscale ratios, the noise its seed and batch_index, the sampler the latent's noise_mask, the denoise, the seed,
the preview callback and the progress-bar flag; the returned LATENT keeps the input's other fields, drops the
downscale ratios and holds the samples. comfy.sample and latent_preview are stand-ins recording their calls."""

import sys
import types

import pytest
import torch


@pytest.fixture
def calls(monkeypatch, comfy_stubs):
    seen = {}
    sample = types.ModuleType("comfy.sample")

    def fix(model, latent, spacial=None, temporal=None):
        seen["fix"] = (model, spacial, temporal)
        return latent + 1

    def noise(latent, seed, batch_inds=None):
        seen["noise"] = (seed, batch_inds)
        return torch.zeros_like(latent)

    def run(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent, **kw):
        seen["sample"] = dict(kw, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, latent=latent)
        return latent * 2

    sample.fix_empty_latent_channels, sample.prepare_noise, sample.sample = fix, noise, run
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)
    monkeypatch.setattr(sys.modules["comfy"], "sample", sample, raising=False)
    monkeypatch.setattr(sys.modules["comfy.utils"], "PROGRESS_BAR_ENABLED", False, raising=False)
    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda model, steps: ("callback", steps)
    monkeypatch.setitem(sys.modules, "latent_preview", preview)
    return seen


def test_ksampler_is_cores_common_ksampler(bcnodes, calls):
    mask = torch.ones(1, 1, 4, 4)
    latent = {"samples": torch.zeros(1, 16, 4, 4), "batch_index": [3], "noise_mask": mask,
              "downscale_ratio_spacial": 8, "downscale_ratio_temporal": 4, "other": "kept"}
    out = bcnodes["libs.sampling"].ksampler("model", 7, 9, 1.0, "dpmpp_sde", "beta", "pos", "neg", latent, 0.45)
    assert calls["fix"] == ("model", 8, 4) and calls["noise"] == (7, [3])
    s = calls["sample"]
    assert (s["steps"], s["cfg"], s["sampler_name"], s["scheduler"], s["denoise"], s["seed"]) == (9, 1.0, "dpmpp_sde", "beta", 0.45, 7)
    assert s["noise_mask"] is mask and s["callback"] == ("callback", 9) and s["disable_pbar"] is True
    assert set(out) == {"samples", "batch_index", "noise_mask", "other"} and torch.equal(out["samples"], torch.full((1, 16, 4, 4), 2.0))
    assert "downscale_ratio_spacial" in latent  # the input is not changed


def test_a_plain_latent(bcnodes, calls):
    out = bcnodes["libs.sampling"].ksampler("model", 0, 1, 1.0, "euler", "simple", "p", "n", {"samples": torch.zeros(1, 4, 2, 2)})
    assert calls["fix"] == ("model", None, None) and calls["noise"] == (0, None)
    assert calls["sample"]["noise_mask"] is None and calls["sample"]["denoise"] == 1.0 and set(out) == {"samples"}

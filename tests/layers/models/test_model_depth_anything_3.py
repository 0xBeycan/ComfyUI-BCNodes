"""Depth Anything 3 over ComfyUI core, with core's loader, preprocessing and net replaced by
stand-ins (no weights, no downloads):

  - weights: a file in models/geometry_estimation is used as is; without it the Comfy-Org
    repackage URL is fetched into that folder, one file per v3 model;
  - loader: core's comfy.sd.load_diffusion_model on that file, one load for two calls, and a
    switch to another model drops the first;
  - predictor: the model is loaded onto the GPU once; per frame core's preprocess_image gets the
    frame (1, H, W, 3) with the short side `resolution` (lower_bound_resize), the net gets it in
    its own dtype, the depth is resized bilinearly (align_corners=False) to the size asked for,
    inverted (near = larger) and clipped to its 2nd..98th percentiles (numpy's np.percentile as
    the reference, as the authors' visualize_depth); the sky clip runs only for a model with a
    sky output, at the model's resolution;
  - the percentile equals np.percentile (linear interpolation) on odd sizes, ties and one value;
  - a ComfyUI without core's Depth Anything 3 stops with an error that says what to do, before
    any download or load.
"""

import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

PKG = "models.depth_anything_3"
SMALL_URL = ("https://huggingface.co/Comfy-Org/Depth-Anything-3/resolve/main/geometry_estimation/"
             "depth_anything_3_small.safetensors")


@pytest.fixture
def weights(bcnodes):
    return bcnodes[f"{PKG}.weights"]


@pytest.fixture
def loader(bcnodes):
    return bcnodes[f"{PKG}.loader"]


@pytest.fixture
def inference(bcnodes):
    return bcnodes[f"{PKG}.inference"]


@pytest.fixture
def models_dir(loader, monkeypatch, tmp_path):
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    monkeypatch.setattr(loader._Loaded, "name", None, raising=True)
    monkeypatch.setattr(loader._Loaded, "patcher", None, raising=True)
    return tmp_path / "models" / "geometry_estimation"


class _Net:
    """Stands in for core's DepthAnything3Net: depth grows from the top row (near) to the bottom
    (far), plus a sky output when `sky` is set; records what it is given."""

    def __init__(self, dtype, sky):
        self.dtype, self.sky, self.seen = dtype, sky, []

    def __call__(self, x):
        self.seen.append((tuple(x.shape), x.dtype))
        _, _, h, w = x.shape
        out = {"depth": torch.linspace(1.0, 4.0, h).view(1, h, 1).expand(1, h, w).to(x.dtype)}
        if self.sky:
            out["sky"] = torch.zeros((1, h, w), dtype=x.dtype)
        return out


class _Patcher:
    """Stands in for core's ModelPatcher: `model` is an nn.Module (core's BaseModel) holding the net."""

    def __init__(self, path, net):
        self.path = path
        self.model = torch.nn.Module()
        self.model.diffusion_model = net


@pytest.fixture
def core(monkeypatch):
    """comfy.sd, comfy.ldm.depth_anything_3.preprocess and model_management.load_model_gpu as
    recording stand-ins; returns the record."""
    rec = types.SimpleNamespace(loads=[], gpu=[], pre=[], sky=[], net=_Net(torch.float32, sky=False))
    sd = types.ModuleType("comfy.sd")

    def load_diffusion_model(path, model_options=None):
        rec.loads.append((os.path.basename(path), os.path.basename(os.path.dirname(path)), model_options))
        return _Patcher(path, rec.net)

    sd.load_diffusion_model = load_diffusion_model
    pre = types.ModuleType("comfy.ldm.depth_anything_3.preprocess")

    def preprocess_image(image, process_res=504, method="upper_bound_resize"):
        rec.pre.append((tuple(image.shape), process_res, method))
        _, h, w, _ = image.shape
        k = process_res / min(h, w)
        return torch.zeros((1, 3, round(h * k / 14) * 14, round(w * k / 14) * 14))

    def apply_sky_aware_clip(depth, sky):
        rec.sky.append((tuple(depth.shape), tuple(sky.shape)))
        return depth.clamp_max(3.0)

    pre.preprocess_image = preprocess_image
    pre.apply_sky_aware_clip = apply_sky_aware_clip
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    monkeypatch.setitem(sys.modules, "comfy.ldm.depth_anything_3", types.ModuleType("comfy.ldm.depth_anything_3"))
    monkeypatch.setitem(sys.modules, "comfy.ldm.depth_anything_3.preprocess", pre)
    monkeypatch.setattr(sys.modules["comfy"], "sd", sd, raising=False)
    monkeypatch.setattr(sys.modules["comfy.model_management"], "load_model_gpu", rec.gpu.append, raising=False)
    return rec


def test_weights_in_geometry_estimation(models_dir, core, loader, bcnodes, monkeypatch):
    models_dir.mkdir(parents=True)
    (models_dir / "depth_anything_3_base.safetensors").write_bytes(b"")
    monkeypatch.setattr(bcnodes["models.common.download"], "fetch_with_progress",
                        lambda *a: pytest.fail("downloaded a file that is there"), raising=True)
    patcher = loader.load("v3-base")
    assert loader.load("v3-base") is patcher
    assert core.loads == [("depth_anything_3_base.safetensors", "geometry_estimation", None)]


def test_weights_download_and_one_slot(models_dir, core, loader, bcnodes, monkeypatch):
    fetched = []
    monkeypatch.setattr(bcnodes["models.common.download"], "fetch_with_progress",
                        lambda url, path, label: fetched.append((url, path, label)) or path, raising=True)
    small = loader.load("v3-small")
    assert fetched == [(SMALL_URL, str(models_dir / "depth_anything_3_small.safetensors"), "depth_anything_3_small.safetensors")]
    metric = loader.load("v3-metric-large")
    assert metric is not small and loader._Loaded.patcher is metric
    assert [f[2] for f in fetched] == ["depth_anything_3_small.safetensors", "depth_anything_3_metric_large.safetensors"]
    assert loader.load("v3-metric-large") is metric and len(core.loads) == 2


def test_weights_table(weights):
    assert weights.FILES == {
        "v3-small": "depth_anything_3_small.safetensors",
        "v3-base": "depth_anything_3_base.safetensors",
        "v3-mono-large": "depth_anything_3_mono_large.safetensors",
        "v3-metric-large": "depth_anything_3_metric_large.safetensors",
    }


@pytest.mark.parametrize("sky", [False, True])
def test_predict(sky, models_dir, core, inference, monkeypatch, bcnodes):
    monkeypatch.setattr(bcnodes["models.common.download"], "fetch_with_progress", lambda url, path, label: path, raising=True)
    core.net = _Net(torch.float16, sky=sky)
    predict = inference.predictor("v3-mono-large" if sky else "v3-small")
    assert len(core.gpu) == 1
    frame = torch.rand((30, 50, 3), generator=torch.Generator().manual_seed(3))
    a = predict(frame, 28, (19, 31))
    b = predict(frame, 28, (19, 31))
    assert len(core.gpu) == 1, "the model is loaded onto the GPU once per run, not per frame"
    assert core.pre == [((1, 30, 50, 3), 28, "lower_bound_resize")] * 2
    assert core.net.seen == [((1, 3, 28, 42), torch.float16)] * 2
    assert core.sky == ([((28, 42), (28, 42))] * 2 if sky else [])
    depth = torch.linspace(1.0, 4.0, 28).view(28, 1).expand(28, 42).half().float()
    if sky:
        depth = depth.clamp_max(3.0)
    inverse = F.interpolate(depth[None, None], size=(19, 31), mode="bilinear", align_corners=False)[0, 0].reciprocal()
    lo, hi = (float(np.percentile(inverse.numpy(), q)) for q in (2, 98))
    assert a.shape == (19, 31) and a.dtype == torch.float32
    assert torch.allclose(a, inverse.clamp(lo, hi), rtol=0, atol=1e-6) and torch.equal(a, b)
    assert abs(float(a.max()) - hi) < 1e-6 and abs(float(a.min()) - lo) < 1e-6
    assert bool((a[0] >= a[-1]).all()) and bool((a[0] > a[-1]).any()), "near (top) must come back larger"


@pytest.mark.parametrize("values", [
    torch.rand(1001, generator=torch.Generator().manual_seed(5)) ** 4,
    torch.rand(7, generator=torch.Generator().manual_seed(6)),
    torch.tensor([3.0, 1.0, 1.0, 1.0, 2.0, 9.0]),
    torch.tensor([4.0]),
])
@pytest.mark.parametrize("q", [2.0, 98.0, 50.0])
def test_percentile_is_numpys(values, q, inference):
    got = float(inference._percentile(values, q))
    assert abs(got - float(np.percentile(values.double().numpy(), q))) < 1e-6


def test_comfyui_without_depth_anything_3(inference, monkeypatch):
    monkeypatch.delitem(sys.modules, "comfy.ldm.depth_anything_3", raising=False)
    monkeypatch.delitem(sys.modules, "comfy.ldm.depth_anything_3.preprocess", raising=False)
    monkeypatch.setattr(inference, "load", lambda name: pytest.fail("loaded before the core check"), raising=True)
    with pytest.raises(RuntimeError, match="update ComfyUI, or choose v2-small"):
        inference.predictor("v3-small")

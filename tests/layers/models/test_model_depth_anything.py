"""Depth Anything V2 Small: the vendored architecture, the preprocessing size rule, the
predictor and the loader, without the real weights (no downloads).

  - arch: DepthAnythingV2() from random init runs a forward on a small input whose sides are
    multiples of 14 and returns (B, H, W), >= 0; its state dict has the ViT-S layout the
    released checkpoint loads strict into (key count, a few shapes, no extra heads);
  - attention: the DINOv2 Attention (torch's scaled_dot_product_attention) equals the upstream
    formula written out, proj(softmax(q k^T * head_dim^-0.5) v), with no dropout in eval;
  - round_resolution / net_size: the authors' Resize(lower_bound, keep_aspect_ratio,
    ensure_multiple_of=14) worked out by hand for a few shapes;
  - predictor (the "v2-small" entry of the depth family): with a stand-in net, the net sees the
    rounded short side and the prediction comes back at the size asked for, unnormalised, as
    the authors' bilinear resize (align_corners=True);
  - the depth family lists v2-small first, then the four v3 models;
  - loader: the weights come from models/depthanything through torch.load(weights_only=True),
    a strict load, one load for two calls; without the file the authors' HF URL is fetched.
"""

import os

import pytest
import torch

PKG = "models.depth_anything_v2"


@pytest.fixture
def arch(bcnodes):
    return bcnodes[f"{PKG}.arch"]


@pytest.fixture
def inference(bcnodes):
    return bcnodes[f"{PKG}.inference"]


@pytest.fixture
def loader(bcnodes):
    return bcnodes[f"{PKG}.loader"]


def test_arch_forward_random_init(arch):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = arch.DepthAnythingV2().eval()
    x = torch.rand((2, 3, 28, 42), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 28, 42)
    assert out.dtype == torch.float32
    assert bool((out >= 0).all())


def test_arch_state_dict_layout(arch):
    with torch.random.fork_rng(devices=[]):
        sd = arch.DepthAnythingV2().state_dict()
    assert len(sd) == 239  # the key count of depth_anything_v2_vits.pth
    assert tuple(sd["pretrained.pos_embed"].shape) == (1, 1370, 384)
    assert tuple(sd["pretrained.patch_embed.proj.weight"].shape) == (384, 3, 14, 14)
    assert tuple(sd["pretrained.blocks.11.ls2.gamma"].shape) == (384,)
    assert tuple(sd["depth_head.projects.3.weight"].shape) == (384, 384, 1, 1)
    assert tuple(sd["depth_head.scratch.output_conv2.2.weight"].shape) == (1, 32, 1, 1)
    assert not any(k.startswith("depth_head.readout_projects") for k in sd)
    assert "pretrained.blocks.12.norm1.weight" not in sd


@pytest.mark.parametrize("attn_drop", [0.0, 0.5])
def test_attention_equals_explicit_softmax(attn_drop, bcnodes):
    layers = bcnodes[f"{PKG}.arch.dinov2_layers"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        attn = layers.Attention(384, num_heads=6, qkv_bias=True, attn_drop=attn_drop).eval()
    x = torch.randn((2, 37, 384), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        got = attn(x)
        q, k, v = attn.qkv(x).reshape(2, 37, 3, 6, 64).permute(2, 0, 3, 1, 4)
        weights = ((q * 64**-0.5) @ k.transpose(-2, -1)).softmax(dim=-1)
        expected = attn.proj((weights @ v).transpose(1, 2).reshape(2, 37, 384))
    assert got.shape == (2, 37, 384)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("value, rounded", [(14, 14), (1, 14), (20, 14), (21, 28), (518, 518), (520, 518),
                                             (525, 532), (1024, 1022), (2044, 2044)])
def test_round_resolution(value, rounded, inference):
    assert inference.round_resolution(value) == rounded


@pytest.mark.parametrize("h, w, resolution, size", [
    (480, 640, 518, (518, 686)),    # landscape: short side 518, 640 * 518 / 480 = 690.7 -> 686
    (640, 480, 518, (686, 518)),    # portrait
    (512, 512, 518, (518, 518)),    # square, upscaled
    (1080, 1920, 518, (518, 924)),  # 1920 * 518 / 1080 = 920.9 -> 924
    (100, 1000, 14, (14, 140)),
    (40, 60, 28, (28, 42)),
])
def test_net_size(h, w, resolution, size, inference):
    got = inference.net_size(h, w, resolution)
    assert got == size
    assert all(side % 14 == 0 and side >= resolution for side in got)


class _NearLeft(torch.nn.Module):
    """Stands in for the model: inverse depth falls from the left edge (near) to the right
    (far); records the input shape it is given."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(tuple(x.shape))
        b, _, h, w = x.shape
        return torch.linspace(10.0, 1.0, w).view(1, 1, w).expand(b, h, w).clone()


def test_predictor_sizes_and_direction(inference, monkeypatch):
    net = _NearLeft()
    monkeypatch.setattr(inference, "load", lambda: (net, torch.device("cpu")), raising=True)
    predict = inference.predictor()
    frame = torch.rand((30, 50, 3), generator=torch.Generator().manual_seed(1))
    pred = predict(frame, 30, (17, 29))
    assert net.seen == [(1, 3, 28, 42)]  # 30 -> 28; 50 * 28 / 30 = 46.7 -> 42
    assert pred.shape == (17, 29) and pred.dtype == torch.float32
    expected = torch.nn.functional.interpolate(torch.linspace(10.0, 1.0, 42).view(1, 1, 1, 42).expand(1, 1, 28, 42),
                                               size=(17, 29), mode="bilinear", align_corners=True)[0, 0]
    assert torch.allclose(pred, expected)
    assert float(pred[0, 0]) == 10.0 and float(pred[0, -1]) == 1.0  # not normalised: the pipeline does that


def test_depth_family_order(bcnodes):
    registry = bcnodes["models.common.registry"]
    assert registry.names(registry.DEPTH) == ["v2-small", "v3-small", "v3-base", "v3-mono-large", "v3-metric-large"]
    assert registry.get(registry.DEPTH, "v2-small") is bcnodes[f"{PKG}.inference"].predictor


@pytest.fixture
def weights_dir(loader, monkeypatch, tmp_path):
    fp = __import__("sys").modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    monkeypatch.setattr(loader._Loaded, "model", None, raising=True)
    monkeypatch.setattr(loader._Loaded, "device", None, raising=True)
    return tmp_path / "models" / "depthanything"


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def test_loader_reads_depthanything_folder(weights_dir, arch, loader, monkeypatch):
    weights_dir.mkdir(parents=True)
    (weights_dir / "depth_anything_v2_vits.pth").write_bytes(b"")
    calls = []

    def fake_load(path, map_location=None, weights_only=None):
        calls.append((os.path.basename(path), os.path.basename(os.path.dirname(path)), map_location, weights_only))
        return {"w": torch.ones(1)}

    monkeypatch.setattr(torch, "load", fake_load, raising=True)
    monkeypatch.setattr(arch, "DepthAnythingV2", _Tiny, raising=True)
    model, device = loader.load()
    again, _ = loader.load()
    assert again is model
    assert calls == [("depth_anything_v2_vits.pth", "depthanything", "cpu", True)]
    assert float(model.w.detach()) == 1.0 and not model.training
    assert device == torch.device("cpu")


def test_weights_download_url(weights_dir, arch, loader, bcnodes, monkeypatch):
    fetched = []
    monkeypatch.setattr(bcnodes["models.common.download"], "fetch_with_progress",
                        lambda url, path, label: fetched.append((url, path, label)) or path, raising=True)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"w": torch.ones(1)}, raising=True)
    monkeypatch.setattr(arch, "DepthAnythingV2", _Tiny, raising=True)
    loader.load()
    assert fetched == [("https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
                        str(weights_dir / "depth_anything_v2_vits.pth"), "depth_anything_v2_vits.pth")]

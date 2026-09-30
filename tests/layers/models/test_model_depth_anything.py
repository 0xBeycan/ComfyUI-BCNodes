"""Depth Anything V2 Small: the vendored architecture, the preprocessing size rule, the
normalisation, the inference loop and the loader, without the real weights (no downloads).

  - arch: DepthAnythingV2() from random init runs a forward on a small input whose sides are
    multiples of 14 and returns (B, H, W), >= 0; its state dict has the ViT-S layout the
    released checkpoint loads strict into (key count, a few shapes, no extra heads);
  - round_resolution / net_size: the authors' Resize(lower_bound, keep_aspect_ratio,
    ensure_multiple_of=14) worked out by hand for a few shapes;
  - normalize: per-frame min-max with the largest inverse depth (nearest) at 1, a flat frame 0;
  - estimate: with a stand-in net, the output is (B, H, W) at the input size in 0..1, near = 1,
    and the net sees the rounded short side;
  - loader: the weights come from models/depthanything through torch.load(weights_only=True),
    a strict load, one load for two calls; without the file the authors' HF URL is fetched.
"""

import os

import pytest
import torch

from _golden import Where

WHERE = Where({
    "arch.DepthAnythingV2": "models.depth_anything_v2.arch:DepthAnythingV2",
    "round_resolution": "models.depth_anything_v2.inference:round_resolution",
    "net_size": "models.depth_anything_v2.inference:net_size",
    "normalize": "models.depth_anything_v2.inference:normalize",
    "estimate": "models.depth_anything_v2.inference:estimate",
    "inference.load": "models.depth_anything_v2.inference:load",
    "load": "models.depth_anything_v2.loader:load",
    "Loaded.model": "models.depth_anything_v2.loader:_Loaded.model",
    "Loaded.device": "models.depth_anything_v2.loader:_Loaded.device",
    "fetch_with_progress": "models.common.download:fetch_with_progress",
})


def test_arch_forward_random_init(bcnodes):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        net = WHERE["arch.DepthAnythingV2"]().eval()
    x = torch.rand((2, 3, 28, 42), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = net(x)
    assert out.shape == (2, 28, 42)
    assert out.dtype == torch.float32
    assert bool((out >= 0).all())


def test_arch_state_dict_layout(bcnodes):
    with torch.random.fork_rng(devices=[]):
        sd = WHERE["arch.DepthAnythingV2"]().state_dict()
    assert len(sd) == 239  # the key count of depth_anything_v2_vits.pth
    assert tuple(sd["pretrained.pos_embed"].shape) == (1, 1370, 384)
    assert tuple(sd["pretrained.patch_embed.proj.weight"].shape) == (384, 3, 14, 14)
    assert tuple(sd["pretrained.blocks.11.ls2.gamma"].shape) == (384,)
    assert tuple(sd["depth_head.projects.3.weight"].shape) == (384, 384, 1, 1)
    assert tuple(sd["depth_head.scratch.output_conv2.2.weight"].shape) == (1, 32, 1, 1)
    assert not any(k.startswith("depth_head.readout_projects") for k in sd)
    assert "pretrained.blocks.12.norm1.weight" not in sd


@pytest.mark.parametrize("value, rounded", [(14, 14), (1, 14), (20, 14), (21, 28), (518, 518), (520, 518),
                                             (525, 532), (1024, 1022), (2044, 2044)])
def test_round_resolution(value, rounded, bcnodes):
    assert WHERE["round_resolution"](value) == rounded


@pytest.mark.parametrize("h, w, resolution, size", [
    (480, 640, 518, (518, 686)),    # landscape: short side 518, 640 * 518 / 480 = 690.7 -> 686
    (640, 480, 518, (686, 518)),    # portrait
    (512, 512, 518, (518, 518)),    # square, upscaled
    (1080, 1920, 518, (518, 924)),  # 1920 * 518 / 1080 = 920.9 -> 924
    (100, 1000, 14, (14, 140)),
    (40, 60, 28, (28, 42)),
])
def test_net_size(h, w, resolution, size, bcnodes):
    got = WHERE["net_size"](h, w, resolution)
    assert got == size
    assert all(side % 14 == 0 and side >= resolution for side in got)


def test_normalize_near_is_white(bcnodes):
    inv = torch.tensor([[[1.0, 2.0], [3.0, 5.0]], [[4.0, 4.0], [4.0, 4.0]]])
    out = WHERE["normalize"](inv)
    assert torch.equal(out[0], torch.tensor([[0.0, 0.25], [0.5, 1.0]]))
    assert torch.equal(out[1], torch.zeros(2, 2))


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


def test_estimate_shape_and_direction(bcnodes, monkeypatch):
    net = _NearLeft()
    WHERE.patch(monkeypatch, "inference.load", lambda: (net, torch.device("cpu")))
    rgb = torch.rand((2, 30, 50, 3), generator=torch.Generator().manual_seed(1))
    out = WHERE["estimate"](rgb, 30)
    assert out.shape == (2, 30, 50)
    assert out.dtype == torch.float32
    assert net.seen == [(1, 3, 28, 42), (1, 3, 28, 42)]  # 30 -> 28; 50 * 28 / 30 = 46.7 -> 42
    assert float(out.min()) == 0.0 and float(out.max()) == 1.0
    assert bool((out[:, :, 0] > out[:, :, -1]).all()), "the near (left) side must be the bright one"


@pytest.fixture
def weights_dir(bcnodes, monkeypatch, tmp_path):
    fp = __import__("sys").modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    WHERE.patch(monkeypatch, "Loaded.model", None)
    WHERE.patch(monkeypatch, "Loaded.device", None)
    return tmp_path / "models" / "depthanything"


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def test_loader_reads_depthanything_folder(weights_dir, monkeypatch):
    weights_dir.mkdir(parents=True)
    (weights_dir / "depth_anything_v2_vits.pth").write_bytes(b"")
    calls = []

    def fake_load(path, map_location=None, weights_only=None):
        calls.append((os.path.basename(path), os.path.basename(os.path.dirname(path)), map_location, weights_only))
        return {"w": torch.ones(1)}

    monkeypatch.setattr(torch, "load", fake_load, raising=True)
    WHERE.patch(monkeypatch, "arch.DepthAnythingV2", _Tiny)
    model, device = WHERE["load"]()
    again, _ = WHERE["load"]()
    assert again is model
    assert calls == [("depth_anything_v2_vits.pth", "depthanything", "cpu", True)]
    assert float(model.w.detach()) == 1.0 and not model.training
    assert device == torch.device("cpu")


def test_weights_download_url(weights_dir, monkeypatch):
    fetched = []
    WHERE.patch(monkeypatch, "fetch_with_progress", lambda url, path, label: fetched.append((url, path, label)) or path)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"w": torch.ones(1)}, raising=True)
    WHERE.patch(monkeypatch, "arch.DepthAnythingV2", _Tiny)
    WHERE["load"]()
    assert fetched == [("https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
                        str(weights_dir / "depth_anything_v2_vits.pth"), "depth_anything_v2_vits.pth")]

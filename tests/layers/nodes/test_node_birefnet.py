"""Golden: BC_BiRefNetRemoveBackground end to end on a tiny deterministic net (plan 8.2 row 13).

`remove_background` loads the model, resizes each frame to the checkpoint's `res`, normalises
it, runs the net under torch.no_grad(), resizes the matte back and hands it to `finish`. Real
weights are out of reach (no downloads), so the architecture class the loader imports lazily
(`from .arch import BiRefNet` in models/birefnet/loader.py) is replaced by
TinyNet, a per-pixel linear map with weights from the `safetensors.torch.load_file` stub.
Each case pins:
  - the digests of the three outputs (shape, dtype, requires_grad, bytes);
  - what the net received per frame: the input digest (pins the resize, the clamp, the
    ImageNet normalisation and the dtype), its dtype, and torch.is_grad_enabled() inside
    forward (pins the torch.no_grad() block);
  - the load log (construct(backbone), strict load, eval, .to) and the load_file path.
The empty and None guards return zero-batch outputs without loading anything.

The weights file exists under the stub models dir, so no download runs (the download branch
is row 16). Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import os
import sys
import types

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'defaults': 'b996939ab5196376002c3e7b0089d2f8',
    'color': '93a8f698a78915b164f811c005852b10',
    'guard/none': 'f6f10efd661264b307b53f35f5c45d9a',
    'guard/empty_batch': 'f6f10efd661264b307b53f35f5c45d9a',
}

WHERE = Where({
    "node": "nodes.birefnet:BiRefNetRemoveBackground",
    "arch.BiRefNet": "models.birefnet.arch:BiRefNet",
    "Loaded.name": "models.birefnet.loader:_Loaded.name",
    "Loaded.model": "models.birefnet.loader:_Loaded.model",
    "Loaded.device": "models.birefnet.loader:_Loaded.device",
    "Loaded.dtype": "models.birefnet.loader:_Loaded.dtype",
})

MODEL = "BiRefNet-general"  # the widget default
WEIGHTS = {"weight": torch.tensor([1.5, -2.0, 0.75]), "bias": torch.tensor([0.1])}
LOG = []


class TinyNet(torch.nn.Module):
    """Stands in for BiRefNet: logits = w . rgb + b per pixel, (1, 3, H, W) -> (1, 1, H, W)."""

    def __init__(self, backbone="swin_v1_l"):
        super().__init__()
        LOG.append(("construct", backbone))
        self.weight = torch.nn.Parameter(torch.zeros(3))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, strict=True):
        LOG.append(("load_state_dict", sorted(state_dict), strict))
        return super().load_state_dict(state_dict, strict=strict)

    def eval(self):
        LOG.append(("eval",))
        return super().eval()

    def to(self, *args, **kwargs):
        LOG.append(("to", args, kwargs))
        return super().to(*args, **kwargs)

    def forward(self, x):
        LOG.append(("forward", digest(x), str(x.dtype), torch.is_grad_enabled()))
        w = self.weight
        return x[:, 0:1] * w[0] + x[:, 1:2] * w[1] + x[:, 2:3] * w[2] + self.bias


@pytest.fixture
def setup(bcnodes, monkeypatch, tmp_path):
    """Fresh cache, TinyNet as the architecture, the load_file stub, and the weights file of
    MODEL under a per-test models dir."""
    LOG.clear()
    for field in ("name", "model", "device", "dtype"):
        WHERE.patch(monkeypatch, f"Loaded.{field}", None)
    WHERE.patch(monkeypatch, "arch.BiRefNet", TinyNet)

    def load_file(path):
        LOG.append(("load_file", os.path.relpath(path, tmp_path).replace(os.sep, "/")))
        return {k: v.clone() for k, v in WEIGHTS.items()}

    st = types.ModuleType("safetensors.torch")
    st.load_file = load_file
    monkeypatch.setitem(sys.modules, "safetensors.torch", st)
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    weights = tmp_path / "models" / "background_removal" / f"{MODEL}.safetensors"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"")


def _image():
    return torch.rand((2, 48, 40, 4), generator=torch.Generator().manual_seed(31))


@pytest.mark.parametrize("name, kwargs", [
    ("defaults", {}),
    ("color", {"background": "Color"}),
])
def test_remove_background(name, kwargs, setup):
    check_env(ENV, "torch")
    out = WHERE["node"]().remove_background(_image(), MODEL, **kwargs)
    check(GOLDEN, name, digest({"outputs": out, "log": LOG}))


@pytest.mark.parametrize("name, image", [
    ("none", None),
    ("empty_batch", torch.zeros((0, 48, 40, 4))),
])
def test_empty_guard(name, image, setup):
    check_env(ENV, "torch")
    out = WHERE["node"]().remove_background(image, MODEL)
    assert LOG == [], "the guard loaded the model"
    check(GOLDEN, f"guard/{name}", digest(out))

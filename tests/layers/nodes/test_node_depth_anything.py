"""BC_DepthAnythingV2 through the node method, the model replaced by a stand-in (no weights):
the output is one IMAGE at the input's size with the depth on all three channels, the
resolution widget reaches the net, and None / an empty batch return an empty IMAGE without
loading the model."""

import pytest
import torch


@pytest.fixture
def node(bcnodes):
    return bcnodes["depth_anything"].DepthAnythingV2


@pytest.fixture
def inference(bcnodes):
    return bcnodes["models.depth_anything_v2.inference"]


class _NearTop(torch.nn.Module):
    """Inverse depth falls from the top row (near) to the bottom (far)."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, x):
        self.seen.append(tuple(x.shape))
        b, _, h, w = x.shape
        return torch.linspace(5.0, 0.0, h).view(1, h, 1).expand(b, h, w).clone()


@pytest.fixture
def net(inference, monkeypatch):
    stand_in = _NearTop()
    monkeypatch.setattr(inference, "load", lambda: (stand_in, torch.device("cpu")), raising=True)
    return stand_in


def test_output_matches_input_size(net, node):
    image = torch.rand((3, 37, 61, 3), generator=torch.Generator().manual_seed(7))
    (depth,) = node().estimate_depth(image, 56)
    assert depth.shape == (3, 37, 61, 3)
    assert depth.dtype == torch.float32
    assert torch.equal(depth[..., 0], depth[..., 1]) and torch.equal(depth[..., 0], depth[..., 2])
    assert float(depth.min()) == 0.0 and float(depth.max()) == 1.0
    assert bool((depth[:, 0] > depth[:, -1]).all()), "near (top) must be white"
    assert net.seen == [(1, 3, 56, 98)] * 3  # short side 56; 61 * 56 / 37 = 92.3 -> 98 (nearest multiple of 14)


@pytest.mark.parametrize("image", [None, torch.zeros((0, 16, 16, 3))])
def test_empty_guard(image, node, inference, monkeypatch):
    def fail():
        raise AssertionError("the guard loaded the model")

    monkeypatch.setattr(inference, "load", fail, raising=True)
    (depth,) = node().estimate_depth(image, 518)
    assert depth.shape == (0, 64, 64, 3)

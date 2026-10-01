"""BC_DepthAnythingV2 (Depth Anything) through the node method, the V2 model replaced by a
stand-in (no weights): the surface (resolution first as before, then the appended optional
`model` widget defaulting to v2-small and the optional width / height sockets), the output at the
short side `resolution` with the depth on all three channels, exactly width x height when both
are connected, an error when only one is, and None / an empty batch returning an empty IMAGE
without loading the model."""

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


def test_surface(node, bcnodes):
    types = node.INPUT_TYPES()
    assert list(types["required"]) == ["image", "resolution"]
    assert types["required"]["resolution"][1]["default"] == 518
    assert list(types["optional"]) == ["model", "width", "height"]
    names, opts = types["optional"]["model"]
    assert names == ["v2-small", "v3-small", "v3-base", "v3-mono-large", "v3-metric-large"]
    assert opts["default"] == "v2-small"
    for side in ("width", "height"):
        kind, opts = types["optional"][side]
        assert kind == "INT" and opts["forceInput"] is True
    assert bcnodes["depth_anything"].NODE_DISPLAY_NAME_MAPPINGS == {"BC_DepthAnythingV2": "Depth Anything"}


def test_output_at_short_side_resolution(net, node):
    image = torch.rand((3, 37, 61, 3), generator=torch.Generator().manual_seed(7))
    (depth,) = node().estimate_depth(image, 56)
    assert depth.shape == (3, 56, 92, 3)  # short side 56; 61 * 56 / 37 = 92.3 -> 92
    assert depth.dtype == torch.float32
    assert torch.equal(depth[..., 0], depth[..., 1]) and torch.equal(depth[..., 0], depth[..., 2])
    assert float(depth.min()) == 0.0 and float(depth.max()) == 1.0
    assert bool((depth[:, 0] > depth[:, -1]).all()), "near (top) must be white"
    assert net.seen == [(1, 3, 56, 98)] * 3  # the net: short side 56; 61 * 56 / 37 = 92.3 -> 98 (nearest multiple of 14)


def test_width_height(net, node):
    image = torch.rand((2, 37, 61, 3), generator=torch.Generator().manual_seed(8))
    (depth,) = node().estimate_depth(image, 56, "v2-small", 48, 64)
    assert depth.shape == (2, 64, 48, 3)
    assert bool((depth[:, 0] > depth[:, -1]).all()), "near (top) must be white"


@pytest.mark.parametrize("width, height", [(64, None), (None, 64)])
def test_one_side_connected_is_an_error(width, height, node, inference, monkeypatch):
    monkeypatch.setattr(inference, "load", lambda: pytest.fail("loaded the model"), raising=True)
    with pytest.raises(ValueError, match="connect both width and height, or neither"):
        node().estimate_depth(torch.rand((1, 8, 8, 3)), 56, "v2-small", width, height)


@pytest.mark.parametrize("image", [None, torch.zeros((0, 16, 16, 3))])
def test_empty_guard(image, node, inference, monkeypatch):
    def fail():
        raise AssertionError("the guard loaded the model")

    monkeypatch.setattr(inference, "load", fail, raising=True)
    (depth,) = node().estimate_depth(image, 518)
    assert depth.shape == (0, 64, 64, 3)

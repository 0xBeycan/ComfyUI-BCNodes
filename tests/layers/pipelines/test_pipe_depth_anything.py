"""The Depth Anything flow with a stand-in entry of the depth family (no model):

  - no target size: the depth map's short side is `resolution`, the long side keeps the aspect
    (libs/geometry.short_side_size), the model is asked for exactly that size, every frame is
    normalised on its own to 0..1 with the nearest at 1, a flat frame is 0, and the three
    channels are equal;
  - a target size: exactly height x width; on an aspect mismatch the model is asked for the size
    that covers it with the input's aspect, and the output is the centred window of that map,
    normalised over the whole map before the crop;
  - the entry is loaded once per run.
"""

import pytest
import torch


@pytest.fixture
def pipe(bcnodes):
    return bcnodes["pipelines.depth_anything"]


class _Entry:
    """A depth-family entry: predict returns, at the size asked for, an inverse depth that grows
    with the column (near = right) plus the frame's mean, so frames differ; a frame whose first
    pixel is 0 comes back flat."""

    def __init__(self):
        self.loads, self.asked = 0, []

    def __call__(self):
        self.loads += 1

        def predict(frame, resolution, size):
            self.asked.append((tuple(frame.shape), resolution, size))
            h, w = size
            if float(frame[0, 0, 0]) == 0.0:
                return torch.full((h, w), 2.0)
            return torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w) * 3.0 + float(frame.mean())

        return predict


@pytest.fixture
def entry(bcnodes, monkeypatch):
    registry = bcnodes["models.common.registry"]
    stand_in = _Entry()
    monkeypatch.setitem(registry._FAMILIES[registry.DEPTH], "stand-in", stand_in)
    return stand_in


def test_short_side_output(pipe, entry):
    rgb = torch.rand((3, 37, 61, 3), generator=torch.Generator().manual_seed(7))
    rgb[1] = 0.0
    out = pipe.estimate(rgb, "stand-in", 56)
    size = (56, 92)  # 61 * 56 / 37 = 92.32 (libs/geometry.short_side_size)
    assert out.shape == (3, 56, 92, 3) and out.dtype == torch.float32 and out.device.type == "cpu"
    assert entry.loads == 1
    assert entry.asked == [((37, 61, 3), 56, size)] * 3
    assert torch.equal(out[..., 0], out[..., 1]) and torch.equal(out[..., 0], out[..., 2])
    for i in (0, 2):
        assert float(out[i].min()) == 0.0 and float(out[i].max()) == 1.0
        assert bool((out[i, :, -1, 0] > out[i, :, 0, 0]).all()), "near (right) must be white"
        assert torch.allclose(out[i, 0, :, 0], torch.linspace(0.0, 1.0, 92))
    assert torch.equal(out[1], torch.zeros(56, 92, 3)), "a flat frame is 0"


@pytest.mark.parametrize("size, cover, window", [
    ((40, 60), (40, 60), (0, 0)),       # the input's aspect: no crop
    ((32, 96), (64, 96), (16, 0)),      # wider than the input: x 1.6 covers it, rows cropped
    ((60, 20), (60, 90), (0, 35)),      # taller: columns cropped
    ((33, 70), (47, 70), (7, 0)),       # 40 * 70 / 60 = 46.67 -> 47; (47 - 33) // 2
])
def test_target_size_cover_and_centre_crop(size, cover, window, pipe, entry):
    rgb = torch.rand((2, 40, 60, 3), generator=torch.Generator().manual_seed(2))
    out = pipe.estimate(rgb, "stand-in", 28, size)
    height, width = size
    assert out.shape == (2, height, width, 3)
    assert entry.asked == [((40, 60, 3), 28, cover)] * 2
    top, left = window
    ch, cw = cover
    for i in range(2):
        full = torch.arange(cw, dtype=torch.float32).view(1, cw).expand(ch, cw) * 3.0 + float(rgb[i].mean())
        full = (full - full.min()) / (full.max() - full.min())
        assert torch.allclose(out[i, ..., 0], full[top:top + height, left:left + width])

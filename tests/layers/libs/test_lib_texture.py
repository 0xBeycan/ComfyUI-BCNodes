"""libs/texture.py, one frame at a time:

  - pore_noise hands out the frames of one torch.randn((b, 1, h, w)) draw from the seed (written out
    here), when a frame holds a multiple of 16 values and when it does not;
  - apply_texture of a batch equals apply_texture of each of its frames, so Skin Texture's frame loop
    gives what the batch at once gave: bit for bit when a frame holds a multiple of 16 pixels; else the
    sRGB transfer's pow, which torch vectorises in blocks, rounds the frame's few pixels that fall in
    another block differently in the last bit (under 1e-7, a ten-thousandth of an 8-bit level).
"""

import pytest
import torch


@pytest.fixture
def texture(bcnodes):
    return bcnodes["libs.texture"]


@pytest.mark.parametrize("b, h, w", [(1, 37, 51), (3, 32, 24), (3, 37, 51), (2, 64, 64)])
def test_pore_noise_is_the_whole_draw_frame_by_frame(texture, b, h, w):
    whole = torch.randn((b, 1, h, w), generator=torch.Generator().manual_seed(9))
    frames = list(texture.pore_noise(b, h, w, 9))
    assert len(frames) == b and all(f.shape == (1, 1, h, w) for f in frames)
    assert torch.equal(torch.cat(frames), whole)


@pytest.mark.parametrize("h, w, exact", [(32, 24, True), (33, 48, True), (37, 51, False)])
def test_apply_texture_frame_by_frame_is_the_batch(texture, h, w, exact):
    g = torch.Generator().manual_seed(4)
    image, mask = torch.rand(3, h, w, 3, generator=g), torch.rand(3, h, w, generator=g)
    noise = torch.randn(3, 1, h, w, generator=g)
    args = dict(texture=0.8, detail=0.6, pore_scale=1.3, gate=0.9)
    whole = texture.apply_texture(image, mask, noise=noise, **args)
    frames = [texture.apply_texture(image[i:i + 1], mask[i:i + 1], noise=noise[i:i + 1], **args) for i in range(3)]
    assert torch.equal(torch.cat(frames), whole) if exact else float((torch.cat(frames) - whole).abs().max()) < 1e-7

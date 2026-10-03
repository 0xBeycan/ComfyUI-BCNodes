"""SeedVR2 Chunk Size's pick (pipelines/seedvr2/chunk_size.py): the largest 4n+1 chunk with
8.36 GiB + 0.6075 GiB per megapixel per latent frame x (1 + safety_margin) inside the card's total, at most
the clip. The expected picks are worked out by hand from that formula:

  - 720p -> 1080p (the DiT at 1088x1920: latent 136 x 240, 2.08896 Mpx): a latent frame costs
    0.6075 x 2.08896 x 1.64 = 2.0812 GiB, so a 32 GB card (31.36 GiB) holds (31.36 - 8.36) / 2.0812 = 11.05
    -> 11 latent frames = 41 pixel frames, and a 96 GB card (94.97 GiB) 41.6 -> 41 = 161: the longest chunk
    measured to run on each card's 900-frame clip (49 failed on the first, 281 on the second);
  - 1080p -> 4K (2160x3840: latent 270 x 480, 8.2944 Mpx): 8.2637 GiB a latent frame, 2.78 -> 2 = 5 frames on
    the 32 GB card, 10.48 -> 10 = 37 on the 96 GB one.
"""

import logging

import pytest
import torch

GIB = 2 ** 30


@pytest.fixture
def card(bcnodes, monkeypatch):
    import comfy.model_management as mm

    def set_total(gib):
        monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: int(gib * GIB), raising=False)
    return set_total


def pick(bcnodes, frames, h, w, margin=0.64, batch=1):
    # only the shape is read: a zero-stride view, no clip-sized buffer
    latent = {"samples": torch.zeros(()).expand(batch, 16, (frames - 1) // 4 + 1, h, w)}
    return bcnodes["pipelines.seedvr2.chunk_size"].frames_per_chunk(latent, margin)[0]


@pytest.mark.parametrize("total_gib, h, w, expected", [
    (31.36, 136, 240, 41),   # 5090, 1080p: 11 latent frames
    (94.97, 136, 240, 161),  # RTX PRO 6000, 1080p: 41 latent frames
    (23.65, 136, 240, 25),   # 24 GB, 1080p: (23.65 - 8.36) / 2.0812 = 7.35 -> 7
    (47.50, 136, 240, 69),   # 48 GB, 1080p: 18.81 -> 18
    (31.36, 270, 480, 5),    # 5090, 4K
    (94.97, 270, 480, 37),   # RTX PRO 6000, 4K
])
def test_picks_by_card_and_size(bcnodes, card, total_gib, h, w, expected):
    card(total_gib)
    assert pick(bcnodes, 901, h, w) == expected


def test_at_most_the_clip(bcnodes, card):
    card(94.97)
    assert pick(bcnodes, 81, 136, 240) == 81  # 161 fit, the clip has 81
    assert pick(bcnodes, 1, 136, 240) == 1


def test_the_margin_scales_the_part_per_frame(bcnodes, card):
    card(31.36)
    # margin 0: (31.36 - 8.36) / (0.6075 x 2.08896) = 18.1 -> 18 latent frames = 69 frames
    assert pick(bcnodes, 901, 136, 240, margin=0.0) == 69
    # margin 1: 23.0 / 2.5381 = 9.06 -> 9 = 33 frames
    assert pick(bcnodes, 901, 136, 240, margin=1.0) == 33


def test_a_batch_counts_as_more_pixels(bcnodes, card):
    card(31.36)
    # two videos of 1080p: 4.1779 Mpx, 4.1625 GiB a latent frame: 5.5 -> 5 = 17 frames
    assert pick(bcnodes, 901, 136, 240, batch=2) == 17


def test_a_card_too_small_gets_one_frame_and_a_warning(bcnodes, card, caplog):
    card(8.0)  # below the fixed part
    with caplog.at_level(logging.WARNING):
        assert pick(bcnodes, 901, 136, 240) == 1
    assert "lower the resolution" in caplog.text


def test_not_a_seedvr2_latent(bcnodes, card):
    card(31.36)
    chunk = bcnodes["pipelines.seedvr2.chunk_size"]
    with pytest.raises(ValueError, match="SeedVR2 latent"):
        chunk.frames_per_chunk({"samples": torch.zeros(1, 4, 64, 64)}, 0.64)
    with pytest.raises(ValueError, match="SeedVR2 latent"):
        chunk.frames_per_chunk({"samples": torch.zeros(1, 4, 5, 64, 64)}, 0.64)
    with pytest.raises(ValueError, match="safety_margin"):
        chunk.frames_per_chunk({"samples": torch.zeros(1, 16, 5, 8, 8)}, -0.1)

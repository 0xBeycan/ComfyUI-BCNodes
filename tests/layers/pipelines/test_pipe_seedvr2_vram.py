"""The VRAM the SeedVR2 VAE flows ask make_room_for_vae for, on a stand-in VAE (the flow stops there):

  - SeedVR2 VAE Decode, a 1088x1920 portrait frame (latent 136 x 240), tile 1024 / 256: the decoder's
    fixed part plus its bytes per pixel of the 1024 x 1024 tile, more than the 20.7 GiB a 32 GB card had
    free with the DiT resident (the decode then ran out of memory) and less than the 31.0 GiB it has free
    with nothing else loaded;
  - a 2048 tile covers that frame: one tile of 1088 x 1920 pixels, more than a 32 GB card holds, for the
    decoder and for the encoder;
  - SeedVR2 VAE Encode, the same frame, tile 1024: the encoder's fixed part plus its bytes per pixel of
    the tile. The frame count does not enter.
"""

import pytest
import torch

GIB = 2 ** 30


class _Stop(Exception):
    pass


class VideoAutoencoderKLWrapper:
    """Only its type name matters here: the flows check it first."""


class _VAE:
    first_stage_model = VideoAutoencoderKLWrapper()
    device = torch.device("cpu")
    vae_dtype = torch.float16


@pytest.fixture
def asked(bcnodes, monkeypatch):
    seen = []

    def make_room(vae, needed):
        seen.append(needed)
        raise _Stop

    for name in ("pipelines.seedvr2.decode", "pipelines.seedvr2.encode"):
        monkeypatch.setattr(bcnodes[name], "make_room_for_vae", make_room)
    return seen


def _decode(bcnodes, tile, frames=1):
    with pytest.raises(_Stop):
        bcnodes["pipelines.seedvr2.decode"].decode({"samples": torch.zeros(1, 16, frames, 240, 136)}, _VAE(), tile, 256)


def _encode(bcnodes, tile, frames=1):
    with pytest.raises(_Stop):
        bcnodes["pipelines.seedvr2.encode"].encode(torch.zeros(frames, 1920, 1088, 3, dtype=torch.float16), _VAE(), tile, 256)


def test_decode_1024_tile(bcnodes, asked):
    _decode(bcnodes, 1024)
    assert asked == [7_950_000_000 + 22_940 * 1024 * 1024]  # 32,004,333,440 bytes = 29.8 GiB
    assert 20.7 * GIB < asked[0] < 31.0 * GIB


def test_decode_ignores_the_frame_count(bcnodes, asked):
    _decode(bcnodes, 1024, frames=1)
    _decode(bcnodes, 1024, frames=21)
    assert asked[0] == asked[1]


def test_decode_2048_tile_is_one_tile_larger_than_the_card(bcnodes, asked):
    _decode(bcnodes, 2048)
    assert asked == [7_950_000_000 + 22_940 * 1088 * 1920]
    assert asked[0] > 32 * GIB


def test_encode_1024_tile(bcnodes, asked):
    _encode(bcnodes, 1024)
    assert asked == [6_360_000_000 + 15_440 * 1024 * 1024]  # 22,550,013,440 bytes = 21.0 GiB


def test_encode_2048_tile_is_one_tile_larger_than_the_card(bcnodes, asked):
    _encode(bcnodes, 2048)
    assert asked == [6_360_000_000 + 15_440 * 1920 * 1088]
    assert asked[0] > 31.37 * GIB


@pytest.fixture
def card(monkeypatch):
    import comfy.model_management as mm

    def set_total(gib):
        monkeypatch.setattr(mm, "get_total_memory", lambda dev=None, torch_total_too=False: int(gib * GIB), raising=False)
    return set_total


def test_auto_tile_on_a_5090(bcnodes, asked, card):
    """tile_size 0 on a 31.36 GiB card (30.61 GiB less the driver's 768 MiB): the encoder's whole 1088x1920 frame
    (36.0 GiB) does not fit, the 1568 tile (1088 x 1568: 30.45 GiB) does; the decoder's 1024 (29.8 GiB) fits and
    1056 (31.2 GiB) does not."""
    card(31.36)
    _encode(bcnodes, 0)
    _decode(bcnodes, 0)
    assert asked == [6_360_000_000 + 15_440 * 1088 * 1568, 7_950_000_000 + 22_940 * 1024 * 1024]


def test_auto_tile_on_a_96_gb_card(bcnodes, asked, card):
    """94.97 GiB: one tile covering the frame for both (the smallest multiple of 32 over 1920)."""
    card(94.97)
    _encode(bcnodes, 0)
    _decode(bcnodes, 0)
    assert asked == [6_360_000_000 + 15_440 * 1088 * 1920, 7_950_000_000 + 22_940 * 1088 * 1920]


def test_a_set_tile_ignores_the_card(bcnodes, asked, card):
    card(8.0)  # far too small: a set tile_size is used as it is, as before
    _encode(bcnodes, 1024)
    _decode(bcnodes, 1024)
    assert asked == [6_360_000_000 + 15_440 * 1024 * 1024, 7_950_000_000 + 22_940 * 1024 * 1024]

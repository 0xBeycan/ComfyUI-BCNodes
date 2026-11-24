"""models/wan_vae (VAE Loader's "Wan 2.1"): prepare() on stand-ins for core's VAE.

A plain Wan 2.1 VAE (decoder gives the image's 3 channels) comes back untouched. A decoder giving
3 x r^2 channels has its decode and decode_tiled outputs unpacked into an r-times larger image,
channel c x r^2 + i x r + j going to pixel (h x r + i, w x r + j) of channel c: torch's pixel_shuffle
order, the one Wan2.1-VAE-upscale2x's model card unpacks with. The 3D tiled decode hands core's tiler
the decoder's own channel count, and a clone keeps the unpacking. Any other VAE, or a channel count
that is not 3 x a square, is refused naming the file.
"""

import pytest
import torch


class _CoreVAE:
    """Core's VAE as prepare() and the subclass touch it: decode / decode_tiled return `out` channels-last."""

    def __init__(self, conv_out=3, latent_channels=16, out=None):
        self.output_channels = 3
        self.latent_channels = latent_channels
        if conv_out is not None:
            self.conv_out_channels = conv_out
        self.out = out

    def decode(self, samples_in, vae_options={}):
        return self.out

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self.out

    def clone(self):
        copy = _CoreVAE.__new__(_CoreVAE)  # as core: a new plain VAE with the fields copied
        copy.__dict__.update({k: v for k, v in self.__dict__.items() if k != "unpack_scale"})
        return copy


def reference_unpack(packed, scale):
    """(..., H, W, 3 x scale^2) -> (..., H x scale, W x scale, 3), pixel by pixel."""
    *lead, height, width, _ = packed.shape
    out = torch.empty(*lead, height * scale, width * scale, 3)
    for c in range(3):
        for i in range(scale):
            for j in range(scale):
                out[..., i::scale, j::scale, c] = packed[..., c * scale * scale + i * scale + j]
    return out


def wan_vae(bcnodes):
    return bcnodes["models.wan_vae.loader"]


def test_registered_as_wan_2_1(bcnodes):
    registry = bcnodes["models.common.registry"]
    assert registry.names(registry.VAE) == ["Wan 2.1"]
    assert registry.get(registry.VAE, "Wan 2.1") is wan_vae(bcnodes).load


def test_plain_wan_vae_untouched(bcnodes):
    vae = _CoreVAE(conv_out=3)
    got, note = wan_vae(bcnodes).prepare(vae, "wan_2.1_vae.safetensors")
    assert got is vae and type(got) is _CoreVAE and note == ""


@pytest.mark.parametrize("scale", [2, 3])
@pytest.mark.parametrize("lead", [(1,), (2, 1)], ids=["image", "video"])
@pytest.mark.parametrize("method", ["decode", "decode_tiled"])
def test_packed_decoder_unpacked(bcnodes, scale, lead, method):
    packed = torch.rand(*lead, 4, 5, 3 * scale * scale)
    vae, note = wan_vae(bcnodes).prepare(_CoreVAE(conv_out=3 * scale * scale, out=packed), "up.safetensors")
    assert note == f"the decoder packs a {scale}x image into {3 * scale * scale} channels: it decodes at {scale}x"
    got = getattr(vae, method)(torch.zeros(1))
    assert got.shape == (*lead, 4 * scale, 5 * scale, 3)
    assert torch.equal(got, reference_unpack(packed, scale))


def test_tiled_3d_keeps_the_packed_channels(bcnodes, monkeypatch):
    import comfy.utils

    calls = []

    def tiled_scale_multidim(samples, function, **kw):
        calls.append(kw)
        return torch.full((1, 12, 1, 8, 8), -1.0)

    monkeypatch.setattr(comfy.utils, "tiled_scale_multidim", tiled_scale_multidim, raising=False)
    vae, _ = wan_vae(bcnodes).prepare(_CoreVAE(conv_out=12), "up.safetensors")
    vae.upscale_ratio, vae.upscale_index_formula, vae.output_device = "ratio", (4, 8, 8), "cpu"
    vae.process_output = lambda image: image.add_(1.0)
    got = vae.decode_tiled_3d(torch.zeros(1, 16, 1, 8, 8), tile_t=5, tile_x=16, tile_y=24, overlap=(1, 4, 4))
    assert calls == [{"tile": (5, 16, 24), "overlap": (1, 4, 4), "upscale_amount": "ratio", "out_channels": 12,
                      "index_formulas": (4, 8, 8), "output_device": "cpu"}]
    assert got.shape == (1, 12, 1, 8, 8) and torch.equal(got, torch.zeros_like(got))  # process_output applied


def test_clone_keeps_the_unpacking(bcnodes):
    packed = torch.rand(1, 2, 2, 12)
    vae, _ = wan_vae(bcnodes).prepare(_CoreVAE(conv_out=12, out=packed), "up.safetensors")
    copy = vae.clone()
    assert type(copy) is type(vae) and copy.unpack_scale == 2
    assert torch.equal(copy.decode(None), reference_unpack(packed, 2))


@pytest.mark.parametrize("vae", [_CoreVAE(conv_out=None), _CoreVAE(conv_out=12, latent_channels=48)],
                         ids=["no Wan 2.1 layout", "48 latent channels"])
def test_other_vae_refused(bcnodes, vae):
    with pytest.raises(ValueError, match=r"^ae\.safetensors is not a Wan 2\.1 VAE .*pick its model, or load it with Load VAE"):
        wan_vae(bcnodes).prepare(vae, "ae.safetensors")


@pytest.mark.parametrize("conv_out", [4, 6, 15, 24])  # not a multiple of 3; 3 x 2, 3 x 5, 3 x 8: no square
def test_channels_not_a_whole_image_refused(bcnodes, conv_out):
    with pytest.raises(ValueError, match=rf"^odd\.safetensors: its decoder gives {conv_out} channels, neither 3 .* nor 3 x a square"):
        wan_vae(bcnodes).prepare(_CoreVAE(conv_out=conv_out), "odd.safetensors")

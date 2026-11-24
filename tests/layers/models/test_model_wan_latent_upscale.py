"""models/wan_latent_upscale (Latent Upscale's "Wan 2.1 2x") on seeded random LatentModel3d weights.

Registered in the latent upscale family. A 5D latent (B, 16, F, H, W) comes back (B, 16, F, 2H, 2W) as the
network gives it; a 4D one (B, 16, H, W) is the same network on its single frame. The input's dtype comes
back, on ComfyUI's intermediate device. A latent with other than 16 channels is refused before any load.
The weights file is the one asked for: kept loaded while it is asked for again, replaced when another is;
a file whose weights do not fit the model is refused naming it. The network's output is the
nearest-upscaled input plus what the network adds.
"""

import pytest
import torch


@pytest.fixture
def upscaler(bcnodes, monkeypatch):
    import comfy.model_management as mm

    arch = bcnodes["models.wan_latent_upscale.arch"]
    loader = bcnodes["models.wan_latent_upscale.loader"]
    torch.manual_seed(0)
    net = arch.LatentModel3d(hidden_size=8, depth=2).eval()
    monkeypatch.setattr(loader._Loaded, "path", "/models/latent_upscale_models/up.safetensors")
    monkeypatch.setattr(loader._Loaded, "model", net)
    monkeypatch.setattr(loader._Loaded, "device", torch.device("cpu"))
    monkeypatch.setattr(mm, "intermediate_device", lambda: torch.device("cpu"), raising=False)
    return bcnodes["models.wan_latent_upscale.inference"], net, "/models/latent_upscale_models/up.safetensors"


def test_registered_as_wan_2_1_2x(bcnodes):
    registry = bcnodes["models.common.registry"]
    assert registry.names(registry.LATENT_UPSCALE) == ["Wan 2.1 2x"]
    assert registry.get(registry.LATENT_UPSCALE, "Wan 2.1 2x") is bcnodes["models.wan_latent_upscale.inference"].upscale


def test_video_latent_doubled(upscaler):
    inference, net, path = upscaler
    x = torch.randn(2, 16, 3, 5, 7)
    got = inference.upscale(x, path)
    with torch.no_grad():
        assert got.shape == (2, 16, 3, 10, 14) and torch.equal(got, net(x))


def test_image_latent_is_its_single_frame(upscaler):
    inference, net, path = upscaler
    x = torch.randn(1, 16, 6, 4)
    got = inference.upscale(x, path)
    with torch.no_grad():
        assert got.shape == (1, 16, 12, 8) and torch.equal(got, net(x.unsqueeze(2)).squeeze(2))


def test_input_dtype_comes_back(upscaler):
    inference, _, path = upscaler
    assert inference.upscale(torch.randn(1, 16, 1, 4, 4, dtype=torch.float16), path).dtype == torch.float16


def test_other_channel_count_refused_before_loading(bcnodes, monkeypatch):
    loader = bcnodes["models.wan_latent_upscale.loader"]
    monkeypatch.setattr(loader._Loaded, "model", None)
    with pytest.raises(ValueError, match=r"^Wan 2\.1 2x takes 16-channel latents .*this latent has 48\."):
        bcnodes["models.wan_latent_upscale.inference"].upscale(torch.zeros(1, 48, 1, 4, 4), "/x/up.safetensors")
    assert loader._Loaded.model is None


def save_weights(arch, path, seed, **config):
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    net = arch.LatentModel3d(**config)
    save_file({k: t.contiguous() for k, t in net.state_dict().items()}, str(path))
    return net


def test_the_file_asked_for_is_loaded_and_kept(bcnodes, tmp_path):
    arch, loader = bcnodes["models.wan_latent_upscale.arch"], bcnodes["models.wan_latent_upscale.loader"]
    loader.unload()
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    net_a, net_b = save_weights(arch, a, 1), save_weights(arch, b, 2)
    model, _ = loader.load(str(a))
    assert torch.equal(model.conv_in.weight, net_a.conv_in.weight)
    assert loader.load(str(a))[0] is model  # asked again: kept
    model_b, _ = loader.load(str(b))
    assert model_b is not model and torch.equal(model_b.conv_in.weight, net_b.conv_in.weight)
    assert loader._Loaded.path == str(b)
    loader.unload()


def test_a_file_of_another_model_is_refused_naming_it(bcnodes, tmp_path):
    arch, loader = bcnodes["models.wan_latent_upscale.arch"], bcnodes["models.wan_latent_upscale.loader"]
    loader.unload()
    other = tmp_path / "other.safetensors"
    save_weights(arch, other, 3, hidden_size=8, depth=1)
    with pytest.raises(ValueError, match=r"^other\.safetensors is not a Wan 2\.1 2x latent upscaler .*pick its model, or another file"):
        loader.load(str(other))
    assert loader._Loaded.model is None


def test_network_adds_to_the_nearest_upscaled_input(bcnodes):
    net = bcnodes["models.wan_latent_upscale.arch"].LatentModel3d(hidden_size=8, depth=1).eval()
    with torch.no_grad():
        for p in net.conv_out.parameters():
            p.zero_()
        x = torch.randn(1, 16, 2, 3, 3)
        assert torch.equal(net(x), x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4))


def test_unload_counts_and_drops(bcnodes):
    loader = bcnodes["models.wan_latent_upscale.loader"]
    loader._Loaded.path, loader._Loaded.device = "/m/latent_upscale_models/up.safetensors", torch.device("cpu")
    loader._Loaded.model = torch.nn.Linear(10, 10, bias=False)
    assert loader.unload() == {"up.safetensors": 400}
    assert (loader._Loaded.path, loader._Loaded.model, loader._Loaded.device) == (None, None, None) and loader.unload() == {}

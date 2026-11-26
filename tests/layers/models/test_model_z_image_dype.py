"""Z-Image's DyPE embedder and adapter (models/z_image/):

- the output is core's EmbedND layout (B, 1, N, sum(axes_dim) / 2, 2, 2), float32 on the ids' device, the tables
  computed in float64 (on the CPU when the device has no float64, as core's rope);
- the time / sequence axis is core's rope, bit for bit, whatever the method;
- vision_yarn: the image axes take the Vision YaRN table times the attention scale 1 + 0.05 ln(s) t_norm^lambda_t
  (t_norm 1 above start_sigma, t / start_sigma below): the full scale above start_sigma, 1 at t 0; a caption token
  (row and column 0) gets that scale on its image axes, as the original;
- yarn: the YaRN table at the stretched length int(512 s), its scale times the attention scale;
- ntk: the NTK table at (s^(d / (d - 2)))^k_t; base: at s^(d / (d - 2)), whatever t;
- pi: the image positions over s^k_t, the plain table (DY-PI);
- the adapter takes core's ZImage config only (not a subclass: Ming-Image, the pixel-space Z-Image), no ControlNet
  model as one with its own embedder, reads theta and axes_dim from the model's own rotary embedder (given, not the
  attribute, which holds the step's during a step) and the longer
  image-axis RoPE length from axes_lens, and the settings' method, start sigma and dype_scale; its automatic values
  are vision_yarn, 1024 px, 2.0, auto_threshold 1920 px, and no yarn_alt_scaling (its yarn has no such option).

core's rope is restated in tests/_rope.py.
"""

import math
import sys
import types

import pytest
import torch

from _rope import core_rope, mm, rotation  # noqa: F401  (mm: a fixture)

THETA, AXES, LENS = 256.0, (32, 48, 48), (1536, 512, 512)


@pytest.fixture
def embedder(bcnodes, mm):
    def make(method="vision_yarn", scale=3.0, t=1.0, start_sigma=1.0, dype_scale=2.0, dype_exponent=1.0):
        return bcnodes["models.z_image.dype.embedder"].DyPERope(THETA, AXES, 512, method, scale, t, start_sigma, dype_scale,
                                                                dype_exponent)
    return make


def ids(rows=6, cols=10, t=33.0):
    """Image position ids as core makes them: (1, rows * cols, 3), the time axis at t."""
    pos = torch.zeros(1, rows * cols, 3)
    pos[..., 0] = t
    pos[..., 1] = torch.arange(rows, dtype=torch.float32).view(-1, 1).repeat(1, cols).flatten()
    pos[..., 2] = torch.arange(cols, dtype=torch.float32).view(1, -1).repeat(rows, 1).flatten()
    return pos


def axes(out):
    """The output split per axis: [(1, N, d / 2, 2, 2)]."""
    parts, start = [], 0
    for dim in AXES:
        parts.append(out[:, 0, :, start:start + dim // 2])
        start += dim // 2
    return parts


def test_layout_dtype_and_the_time_axis_is_cores(embedder):
    pos = ids()
    for method in ("vision_yarn", "yarn", "ntk", "pi", "base"):
        out = embedder(method)(pos)
        assert out.shape == (1, 1, 60, sum(AXES) // 2, 2, 2) and out.dtype == torch.float32, method
        assert torch.equal(axes(out)[0], core_rope(pos[..., 0], AXES[0], THETA)), method


def test_vision_yarn_and_its_attention_scale(bcnodes, embedder):
    rope = bcnodes["libs.dype.rope"]
    pos = ids()
    for t, start, expected in ((1.0, 1.0, 1 + 0.05 * math.log(3)), (0.8, 0.7, 1 + 0.05 * math.log(3)),
                               (0.35, 0.7, 1 + 0.05 * math.log(3) * 0.5 ** 2), (0.0, 0.7, 1.0)):
        e = embedder("vision_yarn", 3.0, t, start, 2.0, 2.0)
        assert e.mscale == pytest.approx(expected)
        table = rope.vision_yarn(48, THETA, 512, 3.0, 3.0, 2.0 * t ** 2.0, "cpu")
        angles = torch.einsum("...n,d->...nd", pos[..., 1], table)
        want = rotation(torch.cos(angles) * e.mscale, torch.sin(angles) * e.mscale)
        assert torch.equal(axes(e(pos))[1], want), (t, start)


def test_a_caption_token_gets_the_attention_scale_on_its_image_axes(embedder):
    caption = torch.zeros(1, 4, 3)
    caption[..., 0] = torch.arange(1, 5, dtype=torch.float32)
    e = embedder("vision_yarn", 3.0, 1.0)
    for part in axes(e(caption))[1:]:
        assert torch.allclose(part[..., 0, 0], torch.full_like(part[..., 0, 0], e.mscale)) and not part[..., 1, 0].any()


def test_yarn_ntk_base_and_pi(bcnodes, embedder):
    rope = bcnodes["libs.dype.rope"]
    pos, s, t = ids(), 2.5, 0.6
    k_t = 2.0 * t ** 1.0

    def image_axis(method):
        return axes(embedder(method, s, t, 1.0, 2.0, 1.0)(pos))[2]

    def plain(freqs, p=pos[..., 2], scale=1.0):
        angles = torch.einsum("...n,d->...nd", p, freqs)
        return rotation(torch.cos(angles) * scale, torch.sin(angles) * scale)

    table, yarn_scale = rope.yarn(48, THETA, 512, int(512 * s), k_t, "cpu")
    mscale = 1 + 0.05 * math.log(s) * (t / 1.0) ** 1.0  # t below start_sigma 1.0: decayed
    assert torch.equal(image_axis("yarn"), plain(table, scale=yarn_scale * mscale))
    factor = s ** (48 / 46)
    assert torch.equal(image_axis("ntk"), plain(rope.ntk(48, THETA, factor ** k_t, "cpu")))
    assert torch.equal(image_axis("base"), plain(rope.ntk(48, THETA, factor, "cpu")))
    assert torch.equal(axes(embedder("base", s, 0.1, 1.0, 2.0, 1.0)(pos))[2], image_axis("base"))  # no modulation
    assert torch.equal(image_axis("pi"), plain(rope.base(48, THETA, "cpu"), pos[..., 2] / s ** k_t))


def test_no_fp64_device_computes_on_the_cpu(bcnodes, embedder, monkeypatch, mm):
    seen = []
    monkeypatch.setattr(mm, "supports_fp64", lambda device: seen.append(device) or False)
    out = embedder()(ids())
    assert seen and out.device.type == "cpu" and out.dtype == torch.float32


class ZImage:
    pass


class MingImage(ZImage):
    pass


@pytest.fixture
def adapter(bcnodes, monkeypatch, comfy_stubs):
    models = types.ModuleType("comfy.supported_models")
    models.ZImage = ZImage
    monkeypatch.setitem(sys.modules, "comfy.supported_models", models)
    return bcnodes["models.z_image.adapter"].ADAPTER


def patcher(config):
    return types.SimpleNamespace(model=types.SimpleNamespace(model_config=config))


def test_the_adapter_takes_z_image_only(adapter):
    assert adapter.matches(patcher(ZImage()))
    assert not adapter.matches(patcher(MingImage())) and not adapter.matches(patcher(object()))
    assert adapter.defaults == {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0, "auto_threshold": 1920}
    assert adapter.latent_scale == 8 and adapter.attribute == "rope_embedder"
    assert not adapter.own_embedder(types.SimpleNamespace(rope_embedder=object()))


def test_the_adapter_reads_the_models_own_rope(adapter, bcnodes, mm):
    rope = types.SimpleNamespace(theta=THETA, axes_dim=list(AXES))
    dm = types.SimpleNamespace(rope_embedder="a step's embedder", axes_lens=[1536, 512, 640])
    s = bcnodes["models.common.dype"].Settings(mode="enable", method="ntk", base_resolution=1024, dype_scale=1.5,
                                               dype_exponent=None, auto_threshold=1920, yarn_alt_scaling=True,
                                               start_sigma=0.6)
    e = adapter.embedder(dm, rope, s, 1536, 2048, 2.0, 0.5, 1.0)
    assert (e.theta, list(e.axes_dim), e.base_len, e.method, e.scale, e.t) == (THETA, list(AXES), 640, "ntk", 2.0, 0.5)
    assert e.k_t == 1.5 * 0.5 and e.mscale == pytest.approx(1 + 0.05 * math.log(2.0) * (0.5 / 0.6))

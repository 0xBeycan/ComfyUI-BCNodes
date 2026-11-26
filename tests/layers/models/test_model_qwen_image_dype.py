"""Qwen-Image's DyPE embedder and adapter (models/qwen_image/):

- the output is core's EmbedND layout (B, 1, N, sum(axes_dim) / 2, 2, 2), float32 on the ids' device, the tables
  computed in float64 (on the CPU when the device has no float64, as core's rope);
- the time / sequence axis is core's rope, bit for bit, whatever the method;
- every token takes the same tables: a caption token (core places it after the image on all three axes) gets on its
  image axes what an image token at that position gets;
- vision_yarn: each image axis takes the Vision YaRN table with its own linear scale (its side over base_resolution,
  at least 1) and the NTK scale of the image (the longer side's), times the attention scale 1 + 0.1 ln(s)
  t_norm^lambda_t (t_norm 1 above start_sigma, t / start_sigma below);
- yarn: both image axes stretched to the longer side's length, YaRN's own attention scale (no decay); with
  yarn_alt_scaling each axis stretched to its own length with the attention scale 1 + 0.1 ln(s_axis), an axis not over
  base_resolution the plain table;
- ntk, base and pi: as every model's (the image's scale);
- the adapter takes core's QwenImage config only (not Qwen-Image 2.1 or a subclass), takes the InstantX ControlNet
  model (core's QwenImageControlNetModel) as one with its own embedder, reads theta and axes_dim from the module's own
  embedder, YaRN's original length as base_resolution in tokens (8 px cells, the module's patch size) and each image
  axis's scale from the latent's size; its automatic values are vision_yarn, 1328 px, 2.0, auto_threshold 1920 px, no
  yarn_alt_scaling.

core's rope is restated in tests/_rope.py.
"""

import math
import sys
import types

import pytest
import torch

from _rope import core_rope, mm, rotation  # noqa: F401  (mm: a fixture)

THETA, AXES, BASE_LEN = 10000.0, (16, 56, 56), 83


@pytest.fixture
def embedder(bcnodes, mm):
    def make(method="vision_yarn", scales=(1.5, 2.5), t=1.0, start_sigma=1.0, dype_scale=2.0, dype_exponent=1.0,
             alt=False):
        return bcnodes["models.qwen_image.dype.embedder"].DyPERope(THETA, AXES, BASE_LEN, method, scales, t, start_sigma,
                                                                   dype_scale, dype_exponent, alt)
    return make


def ids(rows=6, cols=10, caption=0):
    """Position ids as core's Qwen-Image makes them: (1, caption + rows * cols, 3), the caption first, at positions
    from max(rows, cols) // 2 on all three axes; the image at time 0, its rows and columns centred."""
    img = torch.zeros(1, rows * cols, 3)
    img[..., 1] = (torch.arange(rows, dtype=torch.float32) - rows // 2).view(-1, 1).repeat(1, cols).flatten()
    img[..., 2] = (torch.arange(cols, dtype=torch.float32) - cols // 2).view(1, -1).repeat(rows, 1).flatten()
    start = max(rows // 2, cols // 2)
    txt = torch.arange(start, start + caption, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, 3)
    return torch.cat([txt, img], dim=1)


def axes(out):
    """The output split per axis: [(1, N, d / 2, 2, 2)]."""
    parts, start = [], 0
    for dim in AXES:
        parts.append(out[:, 0, :, start:start + dim // 2])
        start += dim // 2
    return parts


def plain(freqs, pos, scale=1.0):
    angles = torch.einsum("...n,d->...nd", pos, freqs)
    return rotation(torch.cos(angles) * scale, torch.sin(angles) * scale)


def test_layout_dtype_and_the_time_axis_is_cores(embedder):
    pos = ids(caption=5)
    for method in ("vision_yarn", "yarn", "ntk", "pi", "base"):
        for alt in (False, True):
            out = embedder(method, alt=alt)(pos)
            assert out.shape == (1, 1, 65, sum(AXES) // 2, 2, 2) and out.dtype == torch.float32, method
            assert torch.equal(axes(out)[0], core_rope(pos[..., 0], AXES[0], THETA)), method


def test_a_caption_token_takes_the_image_tokens_tables(embedder):
    pos = ids(caption=4)
    image_like = pos[:, :4].clone()
    image_like[..., 0] = 0  # the caption's positions as image tokens (time 0) would have them
    for method in ("vision_yarn", "yarn", "ntk", "pi"):
        for alt in (False, True):
            e = embedder(method, t=0.6, alt=alt)
            caption, image = axes(e(pos)), axes(e(image_like))
            assert torch.equal(caption[1][:, :4], image[1]) and torch.equal(caption[2][:, :4], image[2]), method


def test_vision_yarn_and_its_attention_scale(bcnodes, embedder):
    rope = bcnodes["libs.dype.rope"]
    pos = ids()
    for t, start, decay in ((1.0, 1.0, 1.0), (0.8, 0.7, 1.0), (0.35, 0.7, 0.5 ** 2), (0.0, 0.7, 0.0)):
        e = embedder("vision_yarn", (0.8, 1.6), t, start, 2.0, 2.0)  # rows under base_resolution, columns 1.6x
        assert e.scale == 1.6 and e.mscale == pytest.approx(1 + 0.1 * math.log(1.6) * decay)
        k_t = 2.0 * t ** 2.0
        out = axes(e(pos))
        rows = rope.vision_yarn(56, THETA, BASE_LEN, 1.0, 1.6, k_t, "cpu")     # linear 1 (0.8 clamped), NTK at 1.6
        cols = rope.vision_yarn(56, THETA, BASE_LEN, 1.6, 1.6, k_t, "cpu")
        assert torch.equal(out[1], plain(rows, pos[..., 1], e.mscale)), (t, start)
        assert torch.equal(out[2], plain(cols, pos[..., 2], e.mscale)), (t, start)
        assert not torch.equal(rows, cols)


def test_yarn_with_and_without_alt_scaling(bcnodes, embedder):
    rope = bcnodes["libs.dype.rope"]
    pos, k_t = ids(), 2.0 * 0.6
    out = axes(embedder("yarn", (0.8, 1.6), 0.6, 1.0, 2.0, 1.0)(pos))
    table, yarn_scale = rope.yarn(56, THETA, BASE_LEN, BASE_LEN * 1.6, k_t, "cpu")  # both to the longer side
    assert yarn_scale == pytest.approx(1 + 0.1 * math.log(1.6) / math.sqrt(1.6))
    assert torch.equal(out[1], plain(table, pos[..., 1], yarn_scale)) and torch.equal(out[2], plain(table, pos[..., 2], yarn_scale))
    late = embedder("yarn", (0.8, 1.6), 0.0, 1.0, 2.0, 1.0)  # at t 0: the ramps plain, the attention scale not decayed
    assert torch.equal(axes(late(pos))[2], plain(rope.yarn(56, THETA, BASE_LEN, BASE_LEN * 1.6, 0.0, "cpu")[0], pos[..., 2],
                                                 yarn_scale))
    alt = axes(embedder("yarn", (0.8, 1.6), 0.6, 1.0, 2.0, 1.0, alt=True)(pos))
    assert torch.equal(alt[1], plain(rope.base(56, THETA, "cpu"), pos[..., 1]))  # rows not over base_resolution
    cols, _ = rope.yarn(56, THETA, BASE_LEN, BASE_LEN * 1.6, k_t, "cpu")
    assert torch.equal(alt[2], plain(cols, pos[..., 2], 1 + 0.1 * math.log(1.6)))
    both = axes(embedder("yarn", (1.2, 1.6), 0.6, 1.0, 2.0, 1.0, alt=True)(pos))
    rows, _ = rope.yarn(56, THETA, BASE_LEN, BASE_LEN * 1.2, k_t, "cpu")
    assert torch.equal(both[1], plain(rows, pos[..., 1], 1 + 0.1 * math.log(1.2)))


def test_ntk_base_and_pi_at_the_images_scale(bcnodes, embedder):
    rope = bcnodes["libs.dype.rope"]
    pos, t = ids(), 0.6
    k_t = 2.0 * t ** 1.0

    def image_axis(method):
        return axes(embedder(method, (1.0, 2.5), t, 1.0, 2.0, 1.0)(pos))[1]

    factor = 2.5 ** (56 / 54)
    assert torch.equal(image_axis("ntk"), plain(rope.ntk(56, THETA, factor ** k_t, "cpu"), pos[..., 1]))
    assert torch.equal(image_axis("base"), plain(rope.ntk(56, THETA, factor, "cpu"), pos[..., 1]))
    assert torch.equal(image_axis("pi"), plain(rope.base(56, THETA, "cpu"), pos[..., 1] / 2.5 ** k_t))


def test_no_fp64_device_computes_on_the_cpu(embedder, monkeypatch, mm):
    seen = []
    monkeypatch.setattr(mm, "supports_fp64", lambda device: seen.append(device) or False)
    out = embedder()(ids())
    assert seen and out.device.type == "cpu" and out.dtype == torch.float32


class QwenImage:
    pass


class QwenImage21:
    pass


class Subclass(QwenImage):
    pass


class InstantX:  # core's QwenImageControlNetModel
    pass


@pytest.fixture
def adapter(bcnodes, monkeypatch, comfy_stubs):
    models = types.ModuleType("comfy.supported_models")
    models.QwenImage, models.QwenImage21 = QwenImage, QwenImage21
    monkeypatch.setitem(sys.modules, "comfy.supported_models", models)
    controlnet = types.ModuleType("comfy.ldm.qwen_image.controlnet")
    controlnet.QwenImageControlNetModel = InstantX
    monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image.controlnet", controlnet)
    return bcnodes["models.qwen_image.adapter"].ADAPTER


def patcher(config):
    return types.SimpleNamespace(model=types.SimpleNamespace(model_config=config))


def test_the_adapter_takes_qwen_image_only(adapter):
    assert adapter.matches(patcher(QwenImage()))
    assert not any(adapter.matches(patcher(c)) for c in (QwenImage21(), Subclass(), object()))
    assert adapter.defaults == {"method": "vision_yarn", "base_resolution": 1328, "dype_scale": 2.0, "auto_threshold": 1920,
                                "yarn_alt_scaling": False}
    assert adapter.latent_scale == 8 and adapter.attribute == "pe_embedder"


def test_the_instantx_controlnet_carries_its_own_embedder(adapter):
    assert adapter.own_embedder(InstantX()) and adapter.own_embedder(type("Sub", (InstantX,), {})())
    assert not adapter.own_embedder(object())  # the Fun ControlNet: it reads the diffusion model's


def test_the_adapter_reads_the_modules_own_rope(adapter, bcnodes, mm):
    rope = types.SimpleNamespace(theta=THETA, axes_dim=list(AXES))
    module = types.SimpleNamespace(pe_embedder="a step's embedder", patch_size=2)
    s = bcnodes["models.common.dype"].Settings(mode="enable", method="yarn", base_resolution=1328, dype_scale=1.5,
                                               dype_exponent=None, auto_threshold=1920, yarn_alt_scaling=True,
                                               start_sigma=0.6)
    e = adapter.embedder(module, rope, s, 928, 2656, 2.0, 0.5, 1.0)
    assert (e.theta, list(e.axes_dim), e.base_len, e.method, e.scales, e.scale, e.t, e.yarn_alt_scaling) == (
        THETA, list(AXES), 83, "yarn", (1.0, 2.0), 2.0, 0.5, True)
    assert e.k_t == 1.5 * 0.5 and e.mscale == pytest.approx(1 + 0.1 * math.log(2.0) * (0.5 / 0.6))

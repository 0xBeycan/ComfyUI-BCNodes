"""The detailer core (pipelines/detailer: region, conditioning, core), with stand-ins for core's sampler,
VAE, Differential Diffusion and node_helpers:

  - the crop: `context` times the region's box around its centre, at the sampling aspect (long side
    `resolution`, both sides multiples of 64), moved inside and cut to the image; None when the crop
    would not be enlarged;
  - ControlNet hints cut to the crop: same size, another size, another aspect (core's centre cut), a
    chain copied link by link and shared by the positive and the negative, a 1x1 extra concat kept, the
    input conditioning untouched;
  - core's inpaint conditioning: the pixels under the rounded mask grey in the concat latent, the mask
    as concat_mask, the latent of the pixels;
  - the paste: every pixel outside the mask keeps its exact value, a colour drift is matched away, the
    patch's own detail lands inside;
  - the run: two passes (denoise then refine_denoise, seed then seed + 1, a pass at 0 skipped) on a
    Differential Diffusion clone, the soft noise mask at the sampling size, the cut hint in the
    conditioning, a video VAE's 5-D decode, only masked pixels changed, a region too large skipped, the
    mask output, the image unlinked (nothing sampled), half precision.
"""

import sys
import types

import pytest
import torch


class FakeVAE:
    """The 'latent' is the pixels themselves (1, 3, H, W); decode gives a video VAE's (1, 1, H, W, 3)."""

    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(pixels.clone())
        return pixels.movedim(-1, 1).clone()

    def decode(self, latent):
        return latent.movedim(1, -1)[:, None]


class FakeControl:
    """What the cut reads and writes of a core ControlBase: copy() as core's (the previous link not copied)."""

    def __init__(self, hint, extra=()):
        self.cond_hint_original = hint
        self.extra_concat_orig = list(extra)
        self.previous_controlnet = None

    def copy(self):
        return FakeControl(self.cond_hint_original, self.extra_concat_orig)

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self


def conditioning_set_values(conditioning, values={}, append=False):
    """node_helpers.conditioning_set_values as core writes it (no append)."""
    return [[t[0], dict(t[1], **values)] for t in conditioning]


@pytest.fixture
def det(bcnodes):
    return types.SimpleNamespace(region=bcnodes["pipelines.detailer.region"], cond=bcnodes["pipelines.detailer.conditioning"],
                                 core=bcnodes["pipelines.detailer.core"])


@pytest.fixture
def sampled(monkeypatch):
    """Stand-ins for comfy.sample, latent_preview, Differential Diffusion and node_helpers: -> the list of
    sampler calls (their keyword arguments, model and positive included). The sampler inverts the latent."""
    calls = []
    sample = types.ModuleType("comfy.sample")
    sample.fix_empty_latent_channels = lambda model, latent: latent
    sample.prepare_noise = lambda latent, seed: torch.zeros_like(latent)

    def run(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent, **kw):
        calls.append(dict(kw, model=model, steps=steps, cfg=cfg, sampler_name=sampler_name, scheduler=scheduler, positive=positive,
                          negative=negative, latent=latent))
        return 1.0 - latent

    sample.sample = run
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)
    monkeypatch.setattr(sys.modules["comfy"], "sample", sample, raising=False)
    monkeypatch.setattr(sys.modules["comfy.utils"], "PROGRESS_BAR_ENABLED", True, raising=False)
    preview = types.ModuleType("latent_preview")
    preview.prepare_callback = lambda model, steps: None
    monkeypatch.setitem(sys.modules, "latent_preview", preview)
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = conditioning_set_values
    monkeypatch.setitem(sys.modules, "node_helpers", helpers)
    extras = types.ModuleType("comfy_extras")
    dd = types.ModuleType("comfy_extras.nodes_differential_diffusion")

    class DifferentialDiffusion:
        @classmethod
        def execute(cls, model, strength=1.0):
            return types.SimpleNamespace(args=(("dd", model, strength),))

    dd.DifferentialDiffusion = DifferentialDiffusion
    monkeypatch.setitem(sys.modules, "comfy_extras", extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_differential_diffusion", dd)
    return calls


def settings(det, vae=None, positive=None, **kw):
    values = dict(vae=vae or FakeVAE(), positive=positive or [[torch.zeros(1, 2, 4), {}]], negative=[[torch.zeros(1, 2, 4), {}]],
                  seed=7, steps=3, cfg=2.5, sampler_name="euler", scheduler="simple", denoise=0.5, refine_denoise=0.2,
                  resolution=128, context=2.0)
    values.update(kw)
    return det.core.Settings(**values)


def picture(seed, h=64, w=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g) * 0.6 + 0.2


def square(h=64, w=64, top=20, left=24, size=10):
    m = torch.zeros(1, h, w)
    m[0, top:top + size, left:left + size] = 1.0
    return m


# --- the crop ----------------------------------------------------------------------------------------

def test_bounds(det):
    hard = torch.zeros(10, 12, dtype=torch.bool)
    assert det.region.bounds(hard) is None
    hard[2:5, 3:9] = True
    assert det.region.bounds(hard) == (3, 2, 9, 5)


def test_crop_around_the_centre_at_the_sampling_size(det):
    crop = det.region.plan((450, 450, 550, 550), 1000, 1000, 3.0, 1024)
    assert crop == det.region.Crop(350, 350, 650, 650, 1024, 1024)
    # a wide box: the long side is the resolution, both sides multiples of 64, the crop at that aspect
    crop = det.region.plan((400, 480, 600, 520), 1000, 1000, 2.0, 1024)
    assert (crop.width, crop.height) == (1024, 192)
    assert abs((crop.x1 - crop.x0) / (crop.y1 - crop.y0) - 1024 / 192) < 0.1
    assert crop.x1 - crop.x0 >= 400 and crop.y1 - crop.y0 >= 80


def test_crop_moved_inside_and_cut_to_the_image(det):
    crop = det.region.plan((0, 0, 50, 50), 1000, 800, 3.0, 1024)
    assert (crop.x0, crop.y0, crop.x1, crop.y1) == (0, 0, 150, 150)
    crop = det.region.plan((940, 760, 1000, 800), 1000, 800, 3.0, 1024)
    assert (crop.x1, crop.y1) == (1000, 800)
    crop = det.region.plan((100, 100, 300, 140), 400, 200, 10.0, 1024)  # context past the image: cut to it
    assert 0 <= crop.x0 and crop.x1 <= 400 and 0 <= crop.y0 and crop.y1 <= 200


def test_no_crop_when_it_would_not_be_enlarged(det):
    assert det.region.plan((300, 300, 700, 700), 2000, 2000, 3.0, 1024) is None  # 1200 px crop for 1024
    assert det.region.plan((300, 300, 642, 642), 2000, 2000, 3.0, 1024) is None  # 1026 px for 1024
    assert det.region.plan((300, 300, 600, 600), 2000, 2000, 3.0, 1024) is not None  # 900 px


# --- ControlNet hints --------------------------------------------------------------------------------

def hint(h, w):
    return torch.arange(h * w, dtype=torch.float32).reshape(1, 1, h, w).repeat(1, 3, 1, 1)


def test_hint_cut_to_the_crop(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    same = det.cond._cut(hint(100, 100), crop, 100, 100)
    assert torch.equal(same, hint(100, 100)[..., 20:40, 10:30])
    half = det.cond._cut(hint(50, 50), crop, 100, 100)
    assert torch.equal(half, hint(50, 50)[..., 10:20, 5:15])
    # a hint twice as wide as the image: core uses its centre 100 x 100
    wide = det.cond._cut(hint(100, 200), crop, 100, 100)
    assert torch.equal(wide, hint(100, 200)[..., 20:40, 60:80])


def test_control_chain_cut_once_for_positive_and_negative(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    first = FakeControl(hint(100, 100))
    second = FakeControl(hint(100, 100), extra=[torch.ones(1, 1, 100, 100), torch.tensor([[[[1.0]]]])])
    second.set_previous_controlnet(first)
    positive = [[torch.zeros(1), {"control": second, "other": 1}]]
    negative = [[torch.zeros(1), {"control": second}], [torch.zeros(1), {}]]
    done = {}
    pos = det.cond.crop_conditioning(positive, crop, 100, 100, done)
    neg = det.cond.crop_conditioning(negative, crop, 100, 100, done)
    cut = pos[0][1]["control"]
    assert neg[0][1]["control"] is cut and cut is not second and pos[0][1]["other"] == 1
    assert cut.cond_hint_original.shape[-2:] == (20, 20) and cut.extra_concat_orig[0].shape[-2:] == (20, 20)
    assert cut.extra_concat_orig[1].shape[-2:] == (1, 1)
    assert cut.previous_controlnet is not first and cut.previous_controlnet.cond_hint_original.shape[-2:] == (20, 20)
    # the input is untouched; an entry without a ControlNet keeps its dict
    assert positive[0][1]["control"] is second and second.cond_hint_original.shape[-2:] == (100, 100)
    assert second.previous_controlnet is first and neg[1][1] is negative[1][1]


def test_inpaint_conditioning(det, sampled):
    vae = FakeVAE()
    pixels = torch.rand(1, 64, 64, 3)
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., 10:20, 10:20] = 0.6  # rounds to 1
    mask[..., 30:40, 30:40] = 0.4  # rounds to 0
    positive, negative, latent = det.cond.inpaint([[torch.zeros(1), {"a": 1}]], [[torch.zeros(1), {}]], vae, pixels, mask)
    masked = vae.encoded[0]
    assert torch.equal(masked[0, 10:20, 10:20], torch.full((10, 10, 3), 0.5))
    assert torch.allclose(masked[0, 30:40, 30:40], pixels[0, 30:40, 30:40], atol=1e-6)
    assert torch.equal(latent, pixels.movedim(-1, 1))
    for cond in (positive, negative):
        assert torch.equal(cond[0][1]["concat_latent_image"], masked.movedim(-1, 1)) and cond[0][1]["concat_mask"] is mask
    assert positive[0][1]["a"] == 1


# --- the paste ---------------------------------------------------------------------------------------

def test_paste_changes_only_the_mask(det):
    base = picture(1)[0].permute(2, 0, 1)
    hard = torch.zeros(64, 64, dtype=torch.bool)
    hard[16:48, 12:52] = True
    checker = (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2 * 0.1 - 0.05
    patch = (base + 0.15 + checker).clamp(0, 1)  # a colour drift plus fine detail
    out = det.core.paste(base, patch, hard)
    assert torch.equal(out[:, ~hard], base[:, ~hard])
    inner = (slice(None), slice(28, 36), slice(28, 36))
    # the drift matched away: the inside keeps the base's colour, with the patch's detail on it
    assert abs(float((out[inner] - base[inner]).mean())) < 0.02
    assert torch.allclose(out[inner] - base[inner], checker[28:36, 28:36].expand(3, 8, 8), atol=0.03)


# --- the run -----------------------------------------------------------------------------------------

def test_two_passes_on_a_differential_diffusion_clone(det, sampled):
    image, mask = picture(2), square()
    out, changed = det.core.run(image, mask, "model", settings(det))
    assert [(c["denoise"], c["seed"]) for c in sampled] == [(0.5, 7), (0.2, 8)]
    assert all(c["model"] == ("dd", "model", 1.0) for c in sampled)
    assert (sampled[0]["steps"], sampled[0]["cfg"], sampled[0]["sampler_name"], sampled[0]["scheduler"]) == (3, 2.5, "euler", "simple")
    # the refine pass samples the structure pass's latent
    assert torch.equal(sampled[1]["latent"], 1.0 - sampled[0]["latent"])
    noise_mask = sampled[0]["noise_mask"]
    assert noise_mask.shape == (1, 1, 128, 128) and noise_mask is sampled[1]["noise_mask"]
    assert float(noise_mask[0, 0, 64, 64]) > 0.99 and float(noise_mask[0, 0, 0, 0]) < 0.01
    assert bool(((noise_mask > 0.05) & (noise_mask < 0.95)).any())  # a soft edge for Differential Diffusion
    assert sampled[0]["positive"][0][1]["concat_mask"] is noise_mask
    assert out.dtype == torch.float32 and changed.dtype == torch.float32


@pytest.mark.parametrize("denoise,refine,passes", [(0.5, 0.0, [(0.5, 7)]), (0.0, 0.3, [(0.3, 8)])])
def test_a_pass_at_zero_does_not_run(det, sampled, denoise, refine, passes):
    det.core.run(picture(2), square(), "model", settings(det, denoise=denoise, refine_denoise=refine))
    assert [(c["denoise"], c["seed"]) for c in sampled] == passes


def test_refine_seed_wraps(det, sampled):
    det.core.run(picture(2), square(), "model", settings(det, seed=0xFFFFFFFFFFFFFFFF))
    assert [c["seed"] for c in sampled] == [0xFFFFFFFFFFFFFFFF, 0]


def test_only_masked_pixels_change(det, sampled):
    image, mask = picture(3), square()
    out, changed = det.core.run(image, mask, "model", settings(det))
    inside = mask[0] > 0.5
    assert torch.equal(out[0][~inside], image[0][~inside])
    assert not torch.allclose(out[0][inside], image[0][inside], atol=1e-3)
    assert torch.equal(changed, mask)


def test_controlnet_hint_reaches_the_sampler_cut(det, sampled):
    control = FakeControl(hint(64, 64))
    det.core.run(picture(2), square(), "model", settings(det, positive=[[torch.zeros(1), {"control": control}]]))
    # Crop(19, 15, 39, 35): box (24, 20, 34, 30) times 2 around its centre
    cut = sampled[0]["positive"][0][1]["control"]
    assert torch.equal(cut.cond_hint_original, hint(64, 64)[..., 15:35, 19:39])
    assert control.cond_hint_original.shape[-2:] == (64, 64)


def test_regions_in_batch_order_and_skipped_when_too_large(det, sampled):
    image = picture(4)
    masks = torch.cat([square(top=4, left=4, size=8), square(top=40, left=40, size=8), torch.ones(1, 64, 64)])
    out, changed = det.core.run(image, masks, "model", settings(det, resolution=64))
    assert len(sampled) == 4  # two regions, two passes each; the whole-image mask would not be enlarged
    assert torch.equal(changed[0], (masks[0] + masks[1]).clamp(0, 1))
    assert torch.equal(out[0][changed[0] == 0], image[0][changed[0] == 0])


def test_image_unlinked_samples_nothing(det, sampled):
    image, mask = picture(5).repeat(2, 1, 1, 1), square()
    full_out, full_changed = det.core.run(image, mask, "model", settings(det))
    sampled.clear()
    out, changed = det.core.run(image, mask, "model", settings(det), want_image=False)
    assert sampled == [] and out.shape == (0, 64, 64, 3) and torch.equal(changed, full_changed)
    assert full_out.shape == (2, 64, 64, 3) and torch.equal(full_changed[1], mask[0])


def test_half_precision_in_and_out(det, sampled):
    image = picture(6).half()
    out, changed = det.core.run(image, square().half(), "model", settings(det))
    assert out.dtype == torch.float16 and changed.dtype == torch.float16
    inside = square()[0] > 0.5
    assert torch.equal(out[0][~inside], image[0][~inside])

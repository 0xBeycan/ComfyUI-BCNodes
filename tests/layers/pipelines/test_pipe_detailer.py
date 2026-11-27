"""The detailer core (pipelines/detailer: region, conditioning, core), with stand-ins for core's sampler,
VAE, Differential Diffusion and node_helpers:

  - the crop: sampled at `crop_size` px on its longer side with the region's longer side at `size` px inside
    it (the box at its aspect, grown around its centre; a crop_size under size is size), cut to the image
    and moved inside it at the same zoom (each sampling side rounded up to a multiple of 64, the crop taken
    at that aspect, the zoom raised only where the crop would not fit);
    wide, tall and square boxes; None when the longer side is `size` or more;
  - ControlNet hints cut to the crop: same size, another size, another aspect (core's centre cut, or
    scaled as it is), channels last; a chain copied link by link and shared by the positive and the
    negative, a 1x1 extra concat kept, the input conditioning untouched; a link that is no ControlBase an
    error, before any encode;
  - core's model-patch ControlNets, one stand-in per kind (each with the fields core's class sets): Z-Image's
    Fun ControlNet (hint, inpaint image and mask cut with the centre cut, no encoded hint kept, one copy per
    region on a clone of the Differential Diffusion clone), the DiffSynth ControlNet (the mask scaled as it
    is, the encode's size dropped), Qwen-Image 2.1's Fun ControlNet (scaled as it is; its wrapper and block
    patches pointed at the copy, two stacked), Wan's Uni3C, Anima's LLLite (its attention and MLP patches
    pointed at the copy) and MiniMax H3's Fun ControlNet; every other setting kept, the model connected and
    its patches untouched; a model without any sampled as it is; SUPIR's patch an error, before any encode;
  - core's inpaint conditioning: the pixels under the rounded mask grey in the concat latent, the mask
    as concat_mask, the latent of the pixels;
  - the paste: every pixel outside the mask keeps its exact value, a colour drift is matched away, the
    patch's own detail lands inside;
  - the run: one sampling call per region (denoise, seed) on a Differential Diffusion clone, the soft
    noise mask at the sampling size, the cut hint in the conditioning, a video VAE's 5-D decode, only
    masked pixels changed, a region of `size` px or more skipped, a region over max_size skipped (at it
    detailed), the mask output, the image unlinked (nothing sampled), half precision;
  - denoise 0: every region left as it is (no Differential Diffusion clone, no VAE call, no sampling),
    the image bit-identical in float32 and float16, the mask output empty.
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


class ControlBase:
    """comfy.controlnet.ControlBase, the class the cut checks each link of a chain against (the det fixture
    installs it as comfy.controlnet)."""


class FakeControl(ControlBase):
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


def copy_lists(obj):
    """comfy.utils.deepcopy_list_dict: dicts and lists copied, every other object shared."""
    if isinstance(obj, dict):
        return {k: copy_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [copy_lists(v) for v in obj]
    return obj


class FakeModel:
    """What the detailer reads and calls of core's ModelPatcher: model_options (transformer_options' patch
    lists and replace patches), wrappers, and clone() as core's (model_options through deepcopy_list_dict,
    each wrapper list copied, `parent` the model cloned)."""

    def __init__(self, patches=None, tag=None, replace=None, wrappers=None):
        to = {} if patches is None else {"patches": patches}
        if replace is not None:
            to["patches_replace"] = replace
        self.model_options, self.wrappers = {"transformer_options": to}, wrappers or {}
        self.tag, self.parent = tag, None

    def clone(self):
        c = FakeModel(tag=("clone", self.tag))
        c.model_options, c.parent = copy_lists(self.model_options), self
        c.wrappers = {kind: {key: list(w) for key, w in by_key.items()} for kind, by_key in self.wrappers.items()}
        return c


class Patch:
    """A stand-in of one of core's model-patch ControlNets (the cut finds it by its class name): the settings
    core's classes share, `fields` (what core's __init__ sets), and the method core registers as a wrapper."""

    def __init__(self, **fields):
        self.model_patch, self.vae, self.strength, self.sigma_range = object(), object(), 0.8, (1.0, 0.75)
        self.__dict__.update(fields)

    def diffusion_model_wrapper(self, executor, *args):
        return executor(*args)


class ZImageControlPatch(Patch):
    """core's ZImageControlPatch (comfy_extras/nodes_model_patch.py): the hint and the inpaint image
    (B, H, W, C), the mask as core keeps it ((B, 1, 1, H, W)), the encoded hint and the state between
    blocks."""

    def __init__(self, image, inpaint_image=None, mask=None):
        super().__init__(image=image, inpaint_image=inpaint_image, mask=mask, encoded_image=torch.zeros(1, 33, 8, 8), temp_data=None)


class DiffSynthCnetPatch(Patch):
    """core's DiffSynthCnetPatch: the hint (B, H, W, C), the mask as Z-Image's, the hint encoded at
    construction and the size it was encoded at."""

    def __init__(self, image, mask=None):
        super().__init__(image=image, mask=mask, encoded_image=torch.zeros(1, 16, 8, 8), encoded_image_size=tuple(image.shape[1:3]))


class QwenImage21FunControlPatch(Patch):
    """core's QwenImage21FunControlPatch: the hint and the inpaint image (B, H, W, C), the node's mask, the
    blocks it follows, and what it computes (the control latent, the stream between blocks)."""

    def __init__(self, image, inpaint_image=None, mask=None):
        super().__init__(image=image, inpaint_image=inpaint_image, mask=mask, active=False, injection_layers=[0, 2],
                         control=torch.zeros(1, 129, 4, 4), stream=None, pristine=None)


class QwenImage21FunControlBlockPatch:
    """core's block patch of it: the control patch, the block, the replace patch it wraps."""

    def __init__(self, control_patch, block_index, previous):
        self.control_patch, self.block_index, self.previous = control_patch, block_index, previous


class WanUni3CCnetPatch(Patch):
    """core's WanUni3CCnetPatch: the render video (T, H, W, C) and its encode."""

    def __init__(self, render_video):
        super().__init__(render_video=render_video, sigma_start=1.0, sigma_end=0.0, prepared_render=torch.zeros(1, 16, 1, 8, 8),
                         temp_data=None)


class AnimaLLLitePatch(Patch):
    """core's AnimaLLLitePatch (comfy/ldm/anima/lllite.py): the hint (B, H, W, C) and the mask (B, H, W)."""

    def __init__(self, image, mask=None):
        super().__init__(image=image, mask=mask, sigma_start=1.0, sigma_end=0.0)


class AnimaLLLiteAttentionPatch:
    """core's attention patch of it: the patch it reads and the projections it patches."""

    def __init__(self, patch, targets):
        self.patch, self.targets = patch, targets


class AnimaLLLiteMLPPatch:
    def __init__(self, patch):
        self.patch = patch


class MiniMaxH3FunControlPatch(Patch):
    """core's MiniMaxH3FunControlPatch (comfy_extras/nodes_minimax_h3.py): the control and source videos
    (T, C, H, W), the mask, and what it computes."""

    def __init__(self, control_video, mask=None, source_video=None):
        super().__init__(control_video=control_video, mask=mask, source_video=source_video, sigma_start=1.0, sigma_end=0.0,
                         control_latent=torch.zeros(1, 49, 1, 4, 4), control_latent_shape=(1, 48, 1, 4, 4), control_stream=None,
                         pristine_stream=None, active=False)


class MiniMaxH3FunControlBlockPatch(QwenImage21FunControlBlockPatch):
    pass


class SUPIRPatch:
    """core's SUPIRPatch (comfy/ldm/supir/supir_patch.py): registered as three of its methods."""

    def middle_after(self, kwargs):
        return kwargs

    def output_block(self, h, hsp, transformer_options):
        return h, hsp

    def pre_upsample(self, layer, x, *args, **kwargs):
        return layer(x)


def conditioning_set_values(conditioning, values={}, append=False):
    """node_helpers.conditioning_set_values as core writes it (no append)."""
    return [[t[0], dict(t[1], **values)] for t in conditioning]


@pytest.fixture
def det(bcnodes, monkeypatch):
    controlnet = types.ModuleType("comfy.controlnet")
    controlnet.ControlBase = ControlBase
    monkeypatch.setitem(sys.modules, "comfy.controlnet", controlnet)
    return types.SimpleNamespace(region=bcnodes["pipelines.detailer.region"], cond=bcnodes["pipelines.detailer.conditioning"],
                                 core=bcnodes["pipelines.detailer.core"])


@pytest.fixture
def sampled(monkeypatch):
    """Stand-ins for comfy.sample, latent_preview, Differential Diffusion and node_helpers: -> the list of
    sampler calls (their keyword arguments, model and positive included). The sampler inverts the latent."""
    calls = []
    sample = types.ModuleType("comfy.sample")
    sample.fix_empty_latent_channels = lambda model, latent, *ratios: latent
    sample.prepare_noise = lambda latent, seed, batch_inds=None: torch.zeros_like(latent)

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
            dd = model.clone() if isinstance(model, FakeModel) else FakeModel()
            dd.tag = ("dd", model, strength)
            return types.SimpleNamespace(args=(dd,))

    dd.DifferentialDiffusion = DifferentialDiffusion
    monkeypatch.setitem(sys.modules, "comfy_extras", extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_differential_diffusion", dd)
    return calls


def settings(det, vae=None, positive=None, **kw):
    values = dict(vae=vae or FakeVAE(), positive=positive or [[torch.zeros(1, 2, 4), {}]], negative=[[torch.zeros(1, 2, 4), {}]],
                  seed=7, steps=3, cfg=2.5, sampler_name="euler", scheduler="simple", denoise=0.5, size=64, crop_size=128)
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


def test_region_at_size_inside_a_crop_sampled_at_crop_size(det):
    Crop = det.region.Crop
    # an 80 x 100 box (a face) at 640 in a 1280 crop: a 160 x 200 crop around its centre, sampled at 1024 x 1280
    assert det.region.plan((450, 400, 530, 500), 1000, 1000, 640, 1280) == Crop(410, 350, 570, 550, 1024, 1280)
    # crop_size at size: the box itself; three times size: a box's width and height on every side
    assert det.region.plan((100, 100, 200, 180), 1000, 1000, 640, 640) == Crop(100, 100, 200, 180, 640, 512)
    assert det.region.plan((450, 450, 550, 550), 1000, 1000, 320, 960) == Crop(350, 350, 650, 650, 960, 960)
    # a crop_size under size is size: the box itself
    assert det.region.plan((100, 100, 200, 180), 1000, 1000, 640, 256) == Crop(100, 100, 200, 180, 640, 512)
    # size 512, crop_size 1280: the model samples at exactly 1280 with the face's 100 px side at 512 inside (zoom 5.12,
    # a 200 x 250 crop)
    assert det.region.plan((450, 400, 530, 500), 1000, 1000, 512, 1280) == Crop(390, 325, 590, 575, 1024, 1280)


def test_no_crop_when_the_region_is_size_or_larger(det):
    assert det.region.plan((300, 300, 380, 400), 2000, 2000, 100, 200) is None  # its longer side at size
    assert det.region.plan((300, 300, 380, 401), 2000, 2000, 100, 200) is None  # above it
    assert det.region.plan((300, 300, 401, 380), 2000, 2000, 100, 200) is None  # the longer side is the width
    crop = det.region.plan((300, 300, 380, 399), 2000, 2000, 100, 200)  # one px under: sampled, enlarged
    assert crop is not None and crop.height > crop.y1 - crop.y0 and crop.width > crop.x1 - crop.x0


def test_crop_moved_inside_and_cut_to_the_image(det):
    Crop = det.region.Crop
    # a crop past the top-left and the bottom-right corners: moved inside, its size kept (100 x 160 at zoom
    # 8 is 800 x 1280; 800 rounded up to 832, so 104 columns)
    assert det.region.plan((0, 0, 50, 80), 1000, 800, 640, 1280) == Crop(0, 0, 104, 160, 832, 1280)
    assert det.region.plan((950, 720, 1000, 800), 1000, 800, 640, 1280) == Crop(896, 640, 1000, 800, 832, 1280)
    # a crop larger than the whole image: the crop is the image, sampled smaller than crop_size at the same zoom
    assert det.region.plan((100, 50, 300, 150), 400, 200, 640, 1920) == Crop(0, 0, 400, 200, 1280, 640)
    # ... and where the sampling sides, rounded up (1400 -> 1408, 700 -> 704), would need more than the image, the
    # zoom rises to fit: the region at 704 px instead of 700
    assert det.region.plan((100, 50, 300, 150), 400, 200, 700, 2100) == Crop(0, 0, 400, 200, 1408, 704)


def test_non_square_regions(det):
    Crop = det.region.Crop
    # a wide box and a tall one: the longer side sampled at size, the crop at the box's aspect
    assert det.region.plan((400, 480, 600, 520), 1000, 1000, 640, 1280) == Crop(300, 460, 700, 540, 1280, 256)
    assert det.region.plan((480, 400, 520, 600), 1000, 1000, 640, 1280) == Crop(460, 300, 540, 700, 256, 1280)
    # a 103 x 47 box: its 206 x 94 crop at 640 / 103 is 1280 x 584; 584 is rounded up to 640, and the crop taken at
    # that aspect and the same zoom (103 rows, not 94): the rounding only adds surroundings, the region stays at 640
    crop = det.region.plan((450, 450, 553, 497), 1000, 1000, 640, 1280)
    assert (crop.width, crop.height, crop.x1 - crop.x0, crop.y1 - crop.y0) == (1280, 640, 206, 103)
    assert crop.x0 <= 450 and crop.x1 >= 553 and crop.y0 <= 450 and crop.y1 >= 497


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
    # ... or, scaled as it is (no centre cut), all of it at twice the image's scale across
    stretched = det.cond._cut(hint(100, 200), crop, 100, 100, centre=False)
    assert torch.equal(stretched, hint(100, 200)[..., 20:40, 20:60])
    # channels last (..., H, W, C): the same rows and columns
    assert torch.equal(det.cond._cut(hwc(100, 200), crop, 100, 100, channels_last=True), hwc(100, 200)[:, 20:40, 60:80])
    assert torch.equal(det.cond._cut(hwc(100, 200), crop, 100, 100, False, True), hwc(100, 200)[:, 20:40, 20:60])


def hwc(h, w):
    return hint(h, w).movedim(1, -1)


def test_zimage_patch_cut_to_the_crop(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    patch = ZImageControlPatch(hwc(100, 100), inpaint_image=hwc(50, 100), mask=hint(200, 100)[:, :1, None])
    cut = det.cond._crop_patch(patch, crop, 100, 100)
    assert torch.equal(cut.image, hwc(100, 100)[:, 20:40, 10:30])
    # as wide as the image, half as tall: core uses its centre 50 x 50, at half the image's scale
    assert torch.equal(cut.inpaint_image, hwc(50, 100)[:, 10:20, 30:40])
    # twice as tall as the image: core uses its centre 100 x 100
    assert cut.mask.shape == (1, 1, 1, 20, 20) and torch.equal(cut.mask, hint(200, 100)[:, :1, None, 70:90, 10:30])
    assert cut.encoded_image is None and cut.temp_data is None
    assert (cut.model_patch, cut.vae, cut.strength) == (patch.model_patch, patch.vae, patch.strength)
    # core's node takes the hint, the inpaint image and the mask each as optional
    bare = det.cond._crop_patch(ZImageControlPatch(None, inpaint_image=hwc(100, 100)), crop, 100, 100)
    assert bare.image is None and bare.mask is None and torch.equal(bare.inpaint_image, hwc(100, 100)[:, 20:40, 10:30])


def test_diffsynth_patch_cut_to_the_crop(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    patch = DiffSynthCnetPatch(hwc(100, 200), mask=hint(50, 100)[:, :1, None])
    cut = det.cond._crop_patch(patch, crop, 100, 100)
    # the hint through core's centre cut; the mask scaled onto the latent as it is (core's crop "none")
    assert torch.equal(cut.image, hwc(100, 200)[:, 20:40, 60:80])
    assert torch.equal(cut.mask, hint(50, 100)[:, :1, None, 10:20, 10:30])
    # the encode kept for core's to(), its size dropped: the first call encodes the cut at the crop's size
    assert cut.encoded_image is patch.encoded_image and cut.encoded_image_size is None and patch.encoded_image_size == (100, 200)
    assert (cut.model_patch, cut.vae, cut.strength, cut.sigma_range) == (patch.model_patch, patch.vae, 0.8, (1.0, 0.75))


def test_qwen_image_21_patch_cut_with_its_wrapper_and_block_patches(det):
    """core registers Qwen-Image 2.1's Fun ControlNet as its bound diffusion_model_wrapper and a block patch per
    block it follows (each wrapping the replace patch already there): two stacked, the second's block 0 wrapping
    the first's, which wraps another pack's."""
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    first = QwenImage21FunControlPatch(hwc(100, 200), inpaint_image=hwc(50, 100), mask=hint(100, 100)[:, 0])
    second = QwenImage21FunControlPatch(hwc(100, 100))
    other, unrelated = types.SimpleNamespace(), lambda executor, *args: executor(*args)
    block = QwenImage21FunControlBlockPatch
    dit = {("single_block", 0): block(second, 0, block(first, 0, other)), ("single_block", 2): block(first, 2, None)}
    wrappers = {"diffusion_model": {None: [first.diffusion_model_wrapper, second.diffusion_model_wrapper, unrelated]}}
    model = FakeModel(replace={"dit": dict(dit)}, wrappers=wrappers)
    out = det.cond.crop_model(model, crop, 100, 100)
    assert out is not model and out.parent is model
    w = out.wrappers["diffusion_model"][None]
    cut, cut2 = w[0].__self__, w[1].__self__
    assert type(cut) is QwenImage21FunControlPatch and cut is not first and cut2 is not second and w[2] is unrelated
    assert w[0].__func__ is QwenImage21FunControlPatch.diffusion_model_wrapper
    odit = out.model_options["transformer_options"]["patches_replace"]["dit"]
    b0, b2 = odit[("single_block", 0)], odit[("single_block", 2)]
    assert b0.control_patch is cut2 and b0.previous.control_patch is cut and b0.previous.previous is other and b0.block_index == 0
    assert b2.control_patch is cut and b2.previous is None and b2.block_index == 2
    # core scales all three onto the latent as they are (common_upscale's crop "disabled")
    assert torch.equal(cut.image, hwc(100, 200)[:, 20:40, 20:60]) and torch.equal(cut.inpaint_image, hwc(50, 100)[:, 10:20, 10:30])
    assert torch.equal(cut.mask, hint(100, 100)[:, 0, 20:40, 10:30])
    assert cut.control is None and cut.stream is None and cut.pristine is None
    assert (cut.strength, cut.sigma_range, cut.injection_layers, cut.vae) == (0.8, (1.0, 0.75), first.injection_layers, first.vae)
    assert cut2.inpaint_image is None and cut2.mask is None and torch.equal(cut2.image, hwc(100, 100)[:, 20:40, 10:30])
    # the model connected keeps its wrappers, its block patches and the patches' own images
    assert [f.__self__ for f in model.wrappers["diffusion_model"][None][:2]] == [first, second]
    assert model.model_options["transformer_options"]["patches_replace"]["dit"] == dit
    assert dit[("single_block", 0)].control_patch is second and dit[("single_block", 0)].previous.control_patch is first
    assert first.image.shape == (1, 100, 200, 3) and first.control is not None


def test_uni3c_patch_cut_on_a_clone(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    video = torch.cat([hwc(100, 100), hwc(100, 100) + 1])  # (T, H, W, C)
    patch = WanUni3CCnetPatch(video)
    out = det.cond.crop_model(FakeModel({"double_block": [patch]}), crop, 100, 100)
    cut = out.model_options["transformer_options"]["patches"]["double_block"][0]
    assert cut is not patch and torch.equal(cut.render_video, video[:, 20:40, 10:30])
    assert cut.prepared_render is None and cut.temp_data is None and patch.prepared_render is not None
    assert (cut.strength, cut.sigma_start, cut.sigma_end, cut.vae) == (0.8, 1.0, 0.0, patch.vae)


def test_anima_lllite_patch_cut_with_its_attention_and_mlp_patches(det):
    """core's AnimaLLLiteApply: the patch in post_input, and three patches that read what it computes by the
    patch object itself: each copy must point at the cut."""
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    patch = AnimaLLLitePatch(hwc(100, 200), mask=hint(100, 100)[:, 0])
    attn1, attn2 = AnimaLLLiteAttentionPatch(patch, {"q": "self_attn_q_proj"}), AnimaLLLiteAttentionPatch(patch, {})
    mlp = AnimaLLLiteMLPPatch(patch)
    model = FakeModel({"post_input": [patch], "attn1_patch": [attn1], "attn2_patch": [attn2], "mlp_patch": [mlp]})
    out = det.cond.crop_model(model, crop, 100, 100)
    lists = out.model_options["transformer_options"]["patches"]
    cut = lists["post_input"][0]
    assert cut is not patch and torch.equal(cut.image, hwc(100, 200)[:, 20:40, 60:80])
    assert torch.equal(cut.mask, hint(100, 100)[:, 0, 20:40, 10:30])
    assert (cut.strength, cut.sigma_start, cut.sigma_end) == (0.8, 1.0, 0.0)
    for name, kept in (("attn1_patch", attn1), ("attn2_patch", attn2), ("mlp_patch", mlp)):
        copied = lists[name][0]
        assert type(copied) is type(kept) and copied is not kept and copied.patch is cut and kept.patch is patch
    assert lists["attn1_patch"][0].targets is attn1.targets
    assert model.model_options["transformer_options"]["patches"]["attn1_patch"] == [attn1]


def test_minimax_h3_patch_cut_with_its_wrapper_and_block_patches(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    video = torch.cat([hint(100, 100), hint(100, 100) + 1])  # (T, C, H, W), as core's node hands it on
    patch = MiniMaxH3FunControlPatch(video, mask=hint(100, 200)[:, 0])
    blocks = {("double_block", 0): MiniMaxH3FunControlBlockPatch(patch, 0, None)}
    model = FakeModel(replace={"dit": blocks}, wrappers={"diffusion_model": {None: [patch.diffusion_model_wrapper]}})
    out = det.cond.crop_model(model, crop, 100, 100)
    cut = out.wrappers["diffusion_model"][None][0].__self__
    assert out.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)].control_patch is cut
    assert torch.equal(cut.control_video, video[..., 20:40, 10:30]) and torch.equal(cut.mask, hint(100, 200)[:, 0, 20:40, 60:80])
    assert cut.source_video is None and cut.control_latent is None and cut.control_latent_shape is None
    assert cut.control_stream is None and cut.pristine_stream is None and patch.control_latent is not None
    assert (cut.strength, cut.sigma_start, cut.sigma_end) == (0.8, 1.0, 0.0)


SUPIR = SUPIRPatch()


@pytest.mark.parametrize("patches", [{"middle_block_after_patch": [SUPIR.middle_after]}, {"output_block_patch": [SUPIR.output_block]},
                                     {"forward_timestep_embed_patch": [(object, SUPIR.pre_upsample)]}])
def test_supir_patch_is_an_error(det, patches):
    with pytest.raises(ValueError, match="SUPIRPatch, which the Detailer cannot use: it holds the latent of the whole image"):
        det.cond.crop_model(FakeModel(dict(patches)), det.region.Crop(10, 20, 30, 40, 128, 128), 100, 100)


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


class OtherControl:
    """A ControlNet of another kind: a hint the cut could reach, but no ControlBase."""

    def __init__(self, hint):
        self.cond_hint_original, self.extra_concat_orig, self.previous_controlnet = hint, [], None


def test_a_controlnet_that_is_no_controlbase_is_an_error(det):
    crop = det.region.Crop(10, 20, 30, 40, 128, 128)
    with pytest.raises(ValueError, match="ControlNet of class OtherControl, which the Detailer cannot cut"):
        det.cond.crop_conditioning([[torch.zeros(1), {"control": OtherControl(hint(100, 100))}]], crop, 100, 100, {})
    chained = FakeControl(hint(100, 100)).set_previous_controlnet(OtherControl(hint(100, 100)))  # an earlier link of the chain
    with pytest.raises(ValueError, match="OtherControl"):
        det.cond.crop_conditioning([[torch.zeros(1), {"control": chained}]], crop, 100, 100, {})


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

def test_one_sampling_on_a_differential_diffusion_clone(det, sampled):
    image, mask = picture(2), square()
    out, changed = det.core.run(image, mask, "model", settings(det), 64)
    assert [(c["denoise"], c["seed"]) for c in sampled] == [(0.5, 7)]  # one call: no second pass
    assert sampled[0]["model"].tag == ("dd", "model", 1.0)
    assert (sampled[0]["steps"], sampled[0]["cfg"], sampled[0]["sampler_name"], sampled[0]["scheduler"]) == (3, 2.5, "euler", "simple")
    noise_mask = sampled[0]["noise_mask"]
    assert noise_mask.shape == (1, 1, 128, 128)
    assert float(noise_mask[0, 0, 64, 64]) > 0.99 and float(noise_mask[0, 0, 0, 0]) < 0.01
    assert bool(((noise_mask > 0.05) & (noise_mask < 0.95)).any())  # a soft edge for Differential Diffusion
    assert sampled[0]["positive"][0][1]["concat_mask"] is noise_mask
    assert out.dtype == torch.float32 and changed.dtype == torch.float32


class UnusedVAE:
    """A VAE that must not be called."""

    def encode(self, pixels):
        raise AssertionError("VAE encode at denoise 0")

    def decode(self, latent):
        raise AssertionError("VAE decode at denoise 0")


def test_denoise_zero_leaves_every_region_as_it_is(det, sampled, monkeypatch):
    def unused(model):
        raise AssertionError("Differential Diffusion clone at denoise 0")

    monkeypatch.setattr(det.core, "differential", unused)
    masks = torch.cat([square(top=4, left=4, size=8), square(top=40, left=40, size=8)])
    for dtype in (torch.float32, torch.float16):
        image = picture(2).repeat(2, 1, 1, 1).to(dtype)
        out, changed = det.core.run(image, masks.to(dtype), "model", settings(det, vae=UnusedVAE(), denoise=0.0), 64)
        # no encode, sampling, decode or paste: every pixel keeps its exact value, and no region is marked
        assert sampled == [] and out.dtype == dtype and torch.equal(out, image)
        assert changed.dtype == dtype and changed.shape == (2, 64, 64) and not bool(changed.any())


def test_only_masked_pixels_change(det, sampled):
    image, mask = picture(3), square()
    out, changed = det.core.run(image, mask, "model", settings(det), 64)
    inside = mask[0] > 0.5
    assert torch.equal(out[0][~inside], image[0][~inside])
    assert not torch.allclose(out[0][inside], image[0][inside], atol=1e-3)
    assert torch.equal(changed, mask)


def test_controlnet_hint_reaches_the_sampler_cut(det, sampled):
    control = FakeControl(hint(64, 64))
    det.core.run(picture(2), square(), "model", settings(det, positive=[[torch.zeros(1), {"control": control}]]), 64)
    # Crop(19, 15, 39, 35): box (24, 20, 34, 30) times 2 around its centre
    cut = sampled[0]["positive"][0][1]["control"]
    assert torch.equal(cut.cond_hint_original, hint(64, 64)[..., 15:35, 19:39])
    assert control.cond_hint_original.shape[-2:] == (64, 64)


def zimage_model(patch, other=None):
    """A model as core's ZImageFunControlnet patches it, `patch` in both lists; `other` (another pack's
    patch, unhashable) after it in double_block."""
    return FakeModel({"noise_refiner": [patch], "double_block": [patch] + ([] if other is None else [other])})


def test_zimage_patch_cut_on_a_clone(det, sampled):
    patch, other = ZImageControlPatch(hwc(64, 64), inpaint_image=hwc(64, 64), mask=hint(64, 64)[:, :1, None]), types.SimpleNamespace()
    image, inpaint_image, mask, encoded = patch.image, patch.inpaint_image, patch.mask, patch.encoded_image
    model = zimage_model(patch, other)
    det.core.run(picture(2), square(), model, settings(det), 64)
    m = sampled[0]["model"]
    assert len(sampled) == 1 and m.parent.tag == ("dd", model, 1.0)  # a clone of the DD clone
    patches = m.model_options["transformer_options"]["patches"]
    cut = patches["noise_refiner"][0]
    assert cut is not patch and patches["double_block"][0] is cut and patches["double_block"][1] is other
    # Crop(19, 15, 39, 35), as for the ControlNet's hint above
    assert torch.equal(cut.image, hwc(64, 64)[:, 15:35, 19:39]) and torch.equal(cut.inpaint_image, hwc(64, 64)[:, 15:35, 19:39])
    assert torch.equal(cut.mask, hint(64, 64)[:, :1, None, 15:35, 19:39]) and cut.encoded_image is None
    # the model connected and the DD clone keep the patch, and the patch keeps its hint and its encode
    for kept in (model, m.parent):
        lists = kept.model_options["transformer_options"]["patches"]
        assert lists["noise_refiner"] == [patch] and lists["double_block"][0] is patch and lists["double_block"][1] is other
    assert patch.image is image and patch.inpaint_image is inpaint_image and patch.mask is mask and patch.encoded_image is encoded


def test_zimage_patch_cut_once_per_region(det, sampled):
    """Each region its own copy: core keeps one encode per latent size, which two regions sampled at the same
    size would otherwise share."""
    patch = ZImageControlPatch(hwc(64, 64))
    masks = torch.cat([square(top=4, left=4, size=8), square(top=40, left=40, size=8)])
    s = settings(det)
    det.core.run(picture(4), masks, zimage_model(patch), s, 64)
    cuts = [c["model"].model_options["transformer_options"]["patches"]["double_block"][0] for c in sampled]
    assert len(cuts) == 2 and cuts[0] is not cuts[1]
    for cut, mask in zip(cuts, masks):
        crop, _ = det.core.locate(mask > 0.5, s)
        assert (crop.width, crop.height) == (128, 128) and torch.equal(cut.image, hwc(64, 64)[:, crop.y0:crop.y1, crop.x0:crop.x1])


def test_a_model_without_the_zimage_patch_is_sampled_as_it_is(det, sampled):
    other = types.SimpleNamespace()
    model = FakeModel({"double_block": [other]})
    det.core.run(picture(2), square(), model, settings(det), 64)
    assert all(c["model"].tag == ("dd", model, 1.0) for c in sampled)  # the DD clone itself, not a clone of it
    assert sampled[0]["model"].model_options["transformer_options"]["patches"]["double_block"][0] is other


@pytest.mark.parametrize("where", ["conditioning", "model"])
def test_a_controlnet_it_cannot_cut_stops_the_run_before_any_encode(det, sampled, where):
    positive = [[torch.zeros(1), {"control": OtherControl(hint(64, 64))}]] if where == "conditioning" else None
    model = FakeModel({"middle_block_after_patch": [SUPIR.middle_after]}) if where == "model" else "model"
    with pytest.raises(ValueError, match="OtherControl" if where == "conditioning" else "SUPIRPatch"):
        det.core.run(picture(2), square(), model, settings(det, vae=UnusedVAE(), positive=positive), 64)
    assert sampled == []


def test_regions_in_batch_order_and_skipped_when_too_large(det, sampled):
    image = picture(4)
    masks = torch.cat([square(top=4, left=4, size=8), square(top=40, left=40, size=8), torch.ones(1, 64, 64)])
    out, changed = det.core.run(image, masks, "model", settings(det), 64)
    assert len(sampled) == 2  # one call per region; the whole-image mask is 64 px, the size: left as it is
    assert torch.equal(changed[0], (masks[0] + masks[1]).clamp(0, 1))
    assert torch.equal(out[0][changed[0] == 0], image[0][changed[0] == 0])


def test_region_over_max_size_left_as_it_is(det, sampled):
    masks = torch.cat([square(top=4, left=4, size=8), square(top=40, left=40, size=10)])
    out, changed = det.core.run(picture(4), masks, "model", settings(det), 9)  # the 10 px region just over max_size: skipped
    assert len(sampled) == 1 and torch.equal(changed[0], masks[0])
    sampled.clear()
    out, changed = det.core.run(picture(4), masks, "model", settings(det), 10)  # at exactly max_size: detailed
    assert len(sampled) == 2 and torch.equal(changed[0], (masks[0] + masks[1]).clamp(0, 1))


def test_image_unlinked_samples_nothing(det, sampled):
    image, mask = picture(5).repeat(2, 1, 1, 1), square()
    full_out, full_changed = det.core.run(image, mask, "model", settings(det), 64)
    sampled.clear()
    out, changed = det.core.run(image, mask, "model", settings(det), 64, want_image=False)
    assert sampled == [] and out.shape == (0, 64, 64, 3) and torch.equal(changed, full_changed)
    assert full_out.shape == (2, 64, 64, 3) and torch.equal(full_changed[1], mask[0])


def test_half_precision_in_and_out(det, sampled):
    image = picture(6).half()
    out, changed = det.core.run(image, square().half(), "model", settings(det), 64)
    assert out.dtype == torch.float16 and changed.dtype == torch.float16
    inside = square()[0] > 0.5
    assert torch.equal(out[0][~inside], image[0][~inside])

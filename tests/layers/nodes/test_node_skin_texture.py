"""Golden: Skin Texture's run(), through the node method.

Mask path (SAM 3 never called): the widget defaults, a mask of another size, (B, 1, H, W), a
mask batch longer than the image batch, a half exclude mask (and one of another size), feather
8, a batch of 3 with one mask, texture / detail 0, other pore scales and seed, a black image,
fp16 and RGBA images, None and an empty batch. At 80 px the scaled detail radius is below the
blur cut-off (0.2 px), so the detail boost is a no-op there; pore_scale 2.5 / 4 and a 512 px wide
image run it.

SAM 3 path: comfy.sd.load_checkpoint_guess_config and comfy_extras.nodes_sam3.SAM3_Detect are
recording stubs, SAM3_Detect answers a scripted mask per prompt text, and the clip is a fake
that records tokenize / encode. Pinned: both outputs and the call log (checkpoint path and
flags, prompt texts, threshold, refine_iterations=2, individual_masks=False), the face gate as
the minimum over the batch, and the single-slot model cache (a second run with the same name
loads nothing, another name loads again). The cache class `_Sam3` (WHERE) is emptied before
each case and restored afterwards.

Input: Generator(5) (1, 64, 80, 3) * 0.5 + 0.25.
"""

import sys
import types

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
}

GOLDEN = {
    'mask/mask_defaults': 'c710ea71c2be3dfd35fa2d8a0f692e41',
    'mask/mask_ones': 'f913d1678f11befb8857a8cbed608268',
    'mask/mask_other_size': '9d40c99ebdd207de37db39cb5eabf34c',
    'mask/mask_b1hw': 'dcda1e8b939666ffae5e1a9a06039e71',
    'mask/mask_batch_longer': '1282cefef933907640625680916e1767',
    'mask/half_exclude': 'fe3bb7ed4bfe472ac25210c4dbf4acc5',
    'mask/exclude_other_size': 'e19d7f2cfd9ed7e5d95f22e6ae0276b5',
    'mask/feather_8': 'e89cb8fee642295f0bff8b08d753d452',
    'mask/batch_3_one_mask': '2fb79a856c45af578fb28991505e8d4a',
    'mask/texture_0': 'a55c48146f42860899e65641f19f9777',
    'mask/detail_0': 'f913d1678f11befb8857a8cbed608268',
    'mask/texture_detail_0': 'a55c48146f42860899e65641f19f9777',
    'mask/pore_scale_2.5_seed_7': '172564b69624792318daf83e6234498c',
    'mask/pore_scale_4_detail_2': '1d73669acef719748f79726c9c649ca2',
    'mask/wide_image_defaults': 'de3faa9c4b9d2a8ea1cf4e925e21ba2a',
    'mask/black_image': 'e18fec8d267abd1aede49e8beccd2641',
    'mask/fp16_image': '90c7fc12eda5ca867ec997f15b054c90',
    'mask/rgba_image': '52d91433834caf5d3b64a4adf1cf763e',
    'mask/none_image': '3f66dec676a37f9f3870fe12889ada6d',
    'mask/empty_batch': '3f66dec676a37f9f3870fe12889ada6d',
    'sam3/defaults': 'a138863c5327ad6c3a8fb64f52b8e5e5',
    'sam3/batch_gate_is_the_min': '11c21d29d360c4ff6e4dcace7e6f9d6c',
    'sam3/no_face': 'f2834c34856dfa0e9482038d7e5d1d7d',
    'sam3/threshold_and_exclude': '6fd72f11f20f58dc8b946d9e2239d942',
    'sam3/other_checkpoint': '5da73331346d7eef827119034de85d57',
    'sam3/cache': '81c1d9d7a027a666c0c0fac5d48dc989',
    'sam3/missing_checkpoint': 'ef9d00c4ce7c40f29ef250ea1faf284c',
}

WHERE = Where({
    "Sam3": "models.sam3.loader:_Sam3",
})

DEFAULT_SAM3 = "sam3.1_multiplex_fp16.safetensors"
OTHER_SAM3 = "sam3_other.safetensors"
CHECKPOINTS = {DEFAULT_SAM3, OTHER_SAM3}

DEFAULTS = dict(sam3_model=DEFAULT_SAM3, texture=0.35, detail=0.45, pore_scale=1.0, feather=8, seed=0, threshold=0.5)
KW = dict(DEFAULTS, texture=0.5, detail=0.6, feather=0, seed=3)


def _src(b=1, seed=5, c=3):
    return torch.rand((b, 64, 80, c), generator=torch.Generator().manual_seed(seed)) * 0.5 + 0.25


def _soft(shape, seed=6):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed))


def _half(shape=(1, 64, 80)):
    half = torch.zeros(shape)
    half[..., : shape[-1] // 2] = 1.0
    return half


def _rows(b, spans, dtype=torch.float32):
    """(b, 64, 80) with frame i set to 1 over rows spans[i] = (top, bottom) in columns 20:60."""
    out = torch.zeros((b, 64, 80), dtype=dtype)
    for i, span in enumerate(spans):
        if span:
            out[i, span[0]:span[1], 20:60] = 1
    return out


@pytest.fixture
def sam3(monkeypatch):
    """Recording SAM 3 stubs; masks[prompt text] -> (B, H, W) mask. Returns (masks, calls)."""
    calls = []
    masks = {}

    class Clip:
        def tokenize(self, text):
            calls.append(("tokenize", text))
            return {"text": text}

        def encode_from_tokens_scheduled(self, tokens):
            calls.append(("encode_from_tokens_scheduled", tokens))
            return [["cond", {"text": tokens["text"]}]]

    def load_checkpoint_guess_config(path, **kwargs):
        calls.append(("load_checkpoint_guess_config", path, kwargs))
        return (f"model:{path.rsplit('/', 1)[-1]}", Clip(), None, None)

    class SAM3_Detect:
        @classmethod
        def execute(cls, model, image, **kwargs):
            text = kwargs["conditioning"][0][1]["text"]
            calls.append(("SAM3_Detect.execute", model, digest(image), kwargs))
            return types.SimpleNamespace(args=(masks[text], [f"boxes:{text}"]))

    def get_full_path(folder, name):
        calls.append(("get_full_path", folder, name))
        return f"/path/to/checkpoints/{name}" if folder == "checkpoints" and name in CHECKPOINTS else None

    sd = types.ModuleType("comfy.sd")
    sd.load_checkpoint_guess_config = load_checkpoint_guess_config
    extras = types.ModuleType("comfy_extras")
    nodes_sam3 = types.ModuleType("comfy_extras.nodes_sam3")
    nodes_sam3.SAM3_Detect = SAM3_Detect
    extras.nodes_sam3 = nodes_sam3
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    monkeypatch.setattr(sys.modules["comfy"], "sd", sd, raising=False)
    monkeypatch.setitem(sys.modules, "comfy_extras", extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sam3", nodes_sam3)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_full_path", get_full_path, raising=True)
    cache = WHERE["Sam3"]
    for attr in ("name", "model", "clip"):
        monkeypatch.setattr(cache, attr, None, raising=True)
    return masks, calls


SKIN, EXCLUDE, FACE = "skin:6", "eyes:4, eyebrows:4, lips:2, teeth:2", "face:2"

MASK_CASES = {
    "mask_defaults": lambda: (_src(), dict(DEFAULTS, mask=_soft((1, 64, 80)))),
    "mask_ones": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)))),
    "mask_other_size": lambda: (_src(), dict(KW, mask=_soft((1, 32, 40)))),
    "mask_b1hw": lambda: (_src(), dict(KW, mask=_soft((1, 1, 64, 80)))),
    "mask_batch_longer": lambda: (_src(), dict(KW, mask=torch.cat([_half(), _soft((1, 64, 80))]))),
    "half_exclude": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), exclude_mask=_half())),
    "exclude_other_size": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), exclude_mask=_soft((1, 16, 20)))),
    "feather_8": lambda: (_src(), dict(KW, mask=_half(), feather=8)),
    "batch_3_one_mask": lambda: (_src(b=3), dict(KW, mask=_soft((1, 64, 80)))),
    "texture_0": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), texture=0.0)),
    "detail_0": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), detail=0.0)),
    "texture_detail_0": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), texture=0.0, detail=0.0)),
    "pore_scale_2.5_seed_7": lambda: (_src(), dict(KW, mask=torch.ones((1, 64, 80)), pore_scale=2.5, seed=7)),
    "pore_scale_4_detail_2": lambda: (_src(), dict(KW, mask=_soft((1, 64, 80)), pore_scale=4.0, detail=2.0)),
    "wide_image_defaults": lambda: (torch.rand((1, 48, 512, 3), generator=torch.Generator().manual_seed(5)),
                                    dict(DEFAULTS, mask=_soft((1, 48, 512)))),
    "black_image": lambda: (torch.zeros((1, 64, 80, 3)), dict(KW, mask=torch.ones((1, 64, 80)))),
    "fp16_image": lambda: (_src().half(), dict(KW, mask=_half())),
    "rgba_image": lambda: (_src(c=4), dict(KW, mask=_half())),
    "none_image": lambda: (None, dict(KW)),
    "empty_batch": lambda: (torch.zeros((0, 64, 80, 3)), dict(KW, mask=torch.ones((1, 64, 80)))),
}


@pytest.mark.parametrize("name", list(MASK_CASES))
def test_mask_path(name, bcnodes, sam3):
    check_env(ENV, "torch")
    _, calls = sam3
    image, kwargs = MASK_CASES[name]()
    out = bcnodes["skin_texture"].SkinTexture().run(image, **kwargs)
    assert calls == []
    check(GOLDEN, f"mask/{name}", digest(out))


def _script(masks, b, skin_spans, drop_spans, face_spans):
    masks[SKIN] = _rows(b, skin_spans, torch.float16)
    masks[EXCLUDE] = _rows(b, drop_spans, torch.bool)
    masks[FACE] = _rows(b, face_spans)


# name -> (batch, skin rows, exclude rows, face rows per frame, image, run() keywords)
SAM3_CASES = {
    "defaults": (1, [(4, 60)], [(20, 26)], [(10, 22)], lambda: _src(), lambda: dict(DEFAULTS)),
    "batch_gate_is_the_min": (2, [(4, 60), (0, 64)], [(20, 26), None], [(5, 35), (30, 38)], lambda: _src(b=2), lambda: dict(KW)),
    "no_face": (1, [(4, 60)], [None], [None], lambda: _src(), lambda: dict(KW)),
    "threshold_and_exclude": (1, [(0, 64)], [(20, 26)], [(0, 40)], lambda: _src(),
                              lambda: dict(KW, threshold=0.3, exclude_mask=_half())),
    "other_checkpoint": (1, [(4, 60)], [(20, 26)], [(10, 22)], lambda: _src(), lambda: dict(KW, sam3_model=OTHER_SAM3)),
}


@pytest.mark.parametrize("name", list(SAM3_CASES))
def test_sam3_path(name, bcnodes, sam3):
    check_env(ENV, "torch")
    masks, calls = sam3
    b, skin, drop, face, image, kwargs = SAM3_CASES[name]
    _script(masks, b, skin, drop, face)
    out = bcnodes["skin_texture"].SkinTexture().run(image(), **kwargs())
    check(GOLDEN, f"sam3/{name}", digest({"out": out, "calls": calls}))


def test_sam3_model_cache(bcnodes, sam3):
    masks, calls = sam3
    _script(masks, 1, [(4, 60)], [(20, 26)], [(10, 22)])
    node = bcnodes["skin_texture"].SkinTexture()
    runs = [node.run(_src(), **dict(KW, sam3_model=name)) for name in (DEFAULT_SAM3, DEFAULT_SAM3, OTHER_SAM3, OTHER_SAM3)]
    loads = [c for c in calls if c[0] in ("load_checkpoint_guess_config", "get_full_path")]
    check(GOLDEN, "sam3/cache", digest({"runs": runs, "loads": loads, "calls": len(calls)}))


def test_sam3_missing_checkpoint(bcnodes, sam3):
    _, calls = sam3
    with pytest.raises(FileNotFoundError) as e:
        bcnodes["skin_texture"].SkinTexture().run(_src(), **dict(KW, sam3_model="sam3_missing.safetensors"))
    check(GOLDEN, "sam3/missing_checkpoint", digest({"error": str(e.value), "calls": calls}))

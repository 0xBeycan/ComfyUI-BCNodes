"""Golden: the five PostFx nodes end to end, on the postfx package of the ENV.

- PostFx Apply on a Generator(3) (2, 32, 40, 3) batch: the default theme (the INPUT_TYPES
  default) / neutral / 1.0 / seed 0 / fixed; night_flash with increment; a (1, 16, 20) half mask
  (resized); a (1, 32, 40) mask broadcast to the 2 frames; an all-zero mask, which is ignored
  (same output as no mask); a connected look (a LUT theme) overriding the theme; 'none', which
  returns the input object itself.
- PostFx Theme / Custom Look / LUT look dicts as json.dumps(sort_keys=True, default=str) with the
  postfx package dir, the repo dir and the test's tmp dir replaced by <postfx>, <repo>, <tmp>:
  Theme for the default theme and a LUT theme; Custom Look with every knob moved and with every
  knob neutral, each on look=None and on a theme look; LUT through lut_path (an identity .cube in
  the tmp dir), with look=None and with a theme look, and with no LUT at all. For every call with
  a look, the input dict is unchanged afterwards (C72: both branches of the base-look rule at
  both sites, incl. the deepcopy).
- LUT through LUTS_DIR: PostFxLut().build("bcnodes_golden_absent.cube", 1.0, look={}) raises
  FileNotFoundError naming <repo>/luts/..., <repo> being _harness.PKG_DIR (computed without
  LUTS_DIR), so LUTS_DIR (pipelines/postfx.py) is pinned.
- PostFx Signature Sheet: pixel digest of the sheet for a Generator(4) (1, 48, 64, 3) image.

postfx must be importable (D1): check_env fails, never skips, when it is not.
"""

import copy
import json
import os

import pytest
import torch

from _golden import check, check_env, digest
from _harness import PKG_DIR

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
    'cv2': '5.0.0',
    'postfx': '1.1.0',
    'postfx_py_md5': '5fa0d1ce9ebacc627bfc660eed20e424',
}

GOLDEN = {
    'apply/default': '641a5ff4ac2adc6ca269621c6195dc2b',
    'apply/night_flash_increment': 'b8f82b8480f1d85303b32915459b13e6',
    'apply/half_mask_16x20': '9a3861e7ec5b52fefa507e1ca02529f9',
    'apply/mask_32x40_two_frames': 'd11dfe18d6fee9f7f51cacfac9c0d704',
    'apply/all_zero_mask': '641a5ff4ac2adc6ca269621c6195dc2b',
    'apply/look_overrides_theme': '7b36deee319ea0f4c5b49056af2ae42c',
    'default_theme': 'signature/01_portra_400',
    'theme/default': '6ca33371560bbfb8c6f5a933ec596b9c',
    'theme/luts/punch_overlay': '1cf462e415cacb39f30e812f67cdf312',
    'custom_look/moved/none': '803a6f0df8b3c1dd61b170bd4a003f74',
    'custom_look/moved/theme': 'bdb28a718cf5ab83162ec9a6ccf0d126',
    'custom_look/neutral/none': '8625fba9969bc4009bca3e6fe4816ed3',
    'custom_look/neutral/theme': '6ca33371560bbfb8c6f5a933ec596b9c',
    'lut/identity_lut_path/none': '4b20bb77dbfc5b777040069569cd0de5',
    'lut/identity_lut_path/theme': '4a3d9a1f584b5e29e2d7e6c6ea09f984',
    'lut/no_lut/none': '8625fba9969bc4009bca3e6fe4816ed3',
    'lut/no_lut/theme': '6ca33371560bbfb8c6f5a933ec596b9c',
    'lut/luts_dir_absent': "LUT file not found: '<repo>/luts/bcnodes_golden_absent.cube'",
    'signature_sheet': '95dc630a5683ecc3d0c314add8d272c6',
}

LUT_THEME = "luts/punch_overlay"
IDENTITY_CUBE = "TITLE \"identity\"\nLUT_3D_SIZE 2\n" + "".join(
    f"{r} {g} {b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1))

KNOBS_MOVED = dict(temp=0.25, tint=-0.1, exposure=0.5, contrast=0.3, vibrance=0.2, saturation=1.2, grain=0.02,
                   grain_size=2.0, vignette=0.3, halation=0.5, clarity=-0.2)
KNOBS_NEUTRAL = dict(temp=0.0, tint=0.0, exposure=0.0, contrast=0.0, vibrance=0.0, saturation=1.0, grain=0.0,
                     grain_size=1.5, vignette=0.0, halation=0.0, clarity=0.0)


def _env():
    check_env(ENV, "torch", "numpy", "Pillow", "cv2", "postfx")


def _image():
    return torch.rand((2, 32, 40, 3), generator=torch.Generator().manual_seed(3))


def _scrub(text, tmp=None):
    import postfx

    text = text.replace(os.path.dirname(postfx.__file__), "<postfx>").replace(PKG_DIR, "<repo>")
    return text.replace(tmp, "<tmp>") if tmp else text


def _look_json(look, tmp=None):
    return _scrub(json.dumps(look, sort_keys=True, default=str), tmp)


@pytest.fixture
def m(bcnodes):
    return bcnodes["postfx"]


def _default_theme(m):
    return m.PostFxApply.INPUT_TYPES()["required"]["theme"][1]["default"]


# name -> kwargs of apply() over the default theme / neutral / 1.0 / 0 / fixed
APPLY_CASES = {
    "default": {},
    "night_flash_increment": dict(condition="night_flash", batch_seed="increment"),
    "half_mask_16x20": dict(mask="half_16x20"),
    "mask_32x40_two_frames": dict(mask="half_32x40"),
    "all_zero_mask": dict(mask="zeros"),
    "look_overrides_theme": dict(look=LUT_THEME, seed=7, strength=0.75),
}


def _mask(kind):
    if kind == "zeros":
        return torch.zeros(1, 32, 40)
    h, w = (16, 20) if kind == "half_16x20" else (32, 40)
    mask = torch.zeros(1, h, w)
    mask[:, :, w // 2:] = 1.0
    return mask


@pytest.mark.parametrize("name", list(APPLY_CASES))
def test_apply(name, m):
    _env()
    kwargs = dict(theme=_default_theme(m), condition="neutral", strength=1.0, seed=0, batch_seed="fixed")
    kwargs.update(APPLY_CASES[name])
    if "mask" in kwargs:
        kwargs["mask"] = _mask(kwargs["mask"])
    if "look" in kwargs:
        kwargs["look"] = m.PostFxTheme().load(kwargs["look"])[0]
    (out,) = m.PostFxApply().apply(_image(), **kwargs)
    if name == "all_zero_mask":
        (unmasked,) = m.PostFxApply().apply(_image(), theme=_default_theme(m), condition="neutral", strength=1.0,
                                            seed=0, batch_seed="fixed")
        assert torch.equal(out, unmasked)
    check(GOLDEN, f"apply/{name}", digest(out))


def test_apply_none_returns_the_input(m):
    _env()
    image = _image()
    (out,) = m.PostFxApply().apply(image, "none", "neutral", 1.0, 0, "fixed")
    assert out is image


def test_default_theme_label(m):
    _env()
    check(GOLDEN, "default_theme", _default_theme(m))


@pytest.mark.parametrize("theme", ["default", LUT_THEME])
def test_theme(theme, m):
    _env()
    (look,) = m.PostFxTheme().load(_default_theme(m) if theme == "default" else theme)
    check(GOLDEN, f"theme/{theme}", digest(_look_json(look)))


@pytest.mark.parametrize("base", ["none", "theme"])
@pytest.mark.parametrize("knobs", ["moved", "neutral"])
def test_custom_look(knobs, base, m):
    _env()
    look = m.PostFxTheme().load(_default_theme(m))[0] if base == "theme" else None
    before = copy.deepcopy(look)
    (out,) = m.PostFxCustomLook().build(**(KNOBS_MOVED if knobs == "moved" else KNOBS_NEUTRAL), look=look)
    assert look == before, "Custom Look changed its input look"
    assert out is not look
    check(GOLDEN, f"custom_look/{knobs}/{base}", digest(_look_json(out)))


@pytest.mark.parametrize("base", ["none", "theme"])
@pytest.mark.parametrize("lut", ["identity_lut_path", "no_lut"])
def test_lut(lut, base, m, tmp_path):
    _env()
    cube = tmp_path / "identity.cube"
    cube.write_text(IDENTITY_CUBE, encoding="utf-8")
    look = m.PostFxTheme().load(_default_theme(m))[0] if base == "theme" else None
    before = copy.deepcopy(look)
    lut_path = f"  {cube}  " if lut == "identity_lut_path" else ""
    (out,) = m.PostFxLut().build("none", 0.6, lut_path=lut_path, look=look)
    assert look == before, "PostFx LUT changed its input look"
    assert out is not look
    check(GOLDEN, f"lut/{lut}/{base}", digest(_look_json(out, str(tmp_path))))


def test_lut_through_luts_dir(m):
    _env()
    assert not os.path.exists(os.path.join(PKG_DIR, "luts", "bcnodes_golden_absent.cube"))
    with pytest.raises(FileNotFoundError) as exc:
        m.PostFxLut().build("bcnodes_golden_absent.cube", 1.0, look={})
    check(GOLDEN, "lut/luts_dir_absent", _scrub(str(exc.value)))


def test_signature_sheet(m):
    _env()
    image = torch.rand((1, 48, 64, 3), generator=torch.Generator().manual_seed(4))
    (sheet,) = m.PostFxSignatureSheet().build(image, "signature", "neutral", 1.0, 5)
    check(GOLDEN, "signature_sheet", digest(sheet))

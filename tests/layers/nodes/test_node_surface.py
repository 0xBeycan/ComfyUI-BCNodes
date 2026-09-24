"""Golden: the ComfyUI surface of the pack, as ComfyUI loads it.

The root __init__ is executed (the way ComfyUI's loader does it) under the stub_comfy set, as
its own package binding. Pinned per node key, in NODE_CLASS_MAPPINGS order: display name, class
name (not its module), CATEGORY, FUNCTION, OUTPUT_NODE, RETURN_TYPES, RETURN_NAMES, INPUT_IS_LIST,
OUTPUT_IS_LIST, OUTPUT_TOOLTIPS, SEARCH_ALIASES, DESCRIPTION, whether IS_CHANGED exists, and the
full INPUT_TYPES() incl. tooltips. Also WEB_DIRECTORY, the bytes of every web/js file and of
locales/en/main.json, SeedVR2PostProcess.METHODS, and where PostFx LUT looks for .cube files.

INPUT_TYPES is dumped with a FlexibleOptionalInputType as {"__flex__": socket type, "known":
declared keys} and the AnyType wildcard as "*". The LUT combo lists whatever .cube files sit in
<repo>/luts (user files, ignored by git), so it is masked in that node's digest and checked
against an independent listing instead. The theme and condition combos come from postfx, so
ENV pins its version and source.

Seams: importlib.util.find_spec answers None for the AVIF/JXL Pillow plugins (Save Image's
format list), folder_paths.get_filename_list answers [] (the SAM 3 combo).
"""

import hashlib
import importlib.util
import os
import sys

import pytest

import _harness
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'postfx': '1.1.0',
    'postfx_py_md5': '5fa0d1ce9ebacc627bfc660eed20e424',
}

GOLDEN = {
    'BC_LogicBoolean': '21dc5e32a3061c057a1e672d89f68517',
    'BC_IsMaskEmpty': '0f8961815cfdf09c21d664b457a95b5c',
    'BC_MaskFillHoles': '21f8ad23ea243f112eb1db4176ce3a98',
    'BC_MaskGrow': '644f612f985f0a02f90d3cb8f4bc3088',
    'BC_ImageScaleByAspectRatio': 'f357c762601b83da27f354cfb099df7b',
    'BC_JoinImageLists': 'ebda12a10bd7372c8e3094cc7cce0c72',
    'BC_BiRefNetRemoveBackground': '4e949b91eba906a0d07d011e0ff6eca5',
    'BC_AutoModelDownloader': '30ea250cdf9a226a68d8be2af865474c',
    'BC_MathExpression': '88d06eb78ae19f0c5eda96547b07bf89',
    'BC_PromptList': 'adfd46af27533d12828452f8d3cd2c4a',
    'BC_AnySwitch': 'dffefc7ed9139df25f6b7fff91be6f58',
    'BC_Seed': '8a1891ba4da1e3dd95e3e2c21967a3b7',
    'BC_ShowText': '18274655500c11d820c4ba4f1fe00053',
    'BC_ImageComparer': 'f362bb2269b8f5441e46dc1a9b345d11',
    'BC_VideoComparer': '3db43d47cab0362979b0ca4ac0702e11',
    'BC_PowerLoraLoader': '1e8418f18280700fc9c63d0b8c0d19c9',
    'BC_AnythingEverywhere': '3baaf3d9df09eadff0c8d11eaf9051b6',
    'BC_FastGroupsBypasser': 'da53f0f5c2e7a04c0965bd58c230f012',
    'BC_SeedVR2Resize': '210a3228ef5bf7caa16cb65e45979fa9',
    'BC_SeedVR2VAEEncode': '0b667ef96354b171dff34b3ca87b96bd',
    'BC_SeedVR2VAEDecode': 'bc3ebed7d51a1b6cb53940b40c9d0d12',
    'BC_SeedVR2PostProcess': '64dbe14cd16830fb2dac0d465365ec5f',
    'BC_PostFxApply': 'dada65847ad66e20196319d1d9df9a13',
    'BC_PostFxTheme': 'ea644fff26913031a5e52b332191e8fb',
    'BC_PostFxCustomLook': '458621485eeb7012dc45863c6a7614e2',
    'BC_PostFxLut': 'c8578af94831fe10bac0e2a9c0d54e95',
    'BC_PostFxSignatureSheet': '2dc94e3d862460a2f9becbd02440a6a5',
    'BC_CaptionAudit': 'e40ece3e35c41d02dc87964ed3a2dcc5',
    'BC_SocialMediaExport': 'ca47d4a8e0a34d3fc443077368567030',
    'BC_ImageQualityGate': '8920f3c2f59783b81c46432debc8b851',
    'BC_SaveImage': '4b0c4c167fe208ac4e6c7d0e4d85fae0',
    'BC_SkinTexture': 'f18437b7ece7b517851d49ce6649582d',
    'WEB_DIRECTORY': './web',
    'frontend files': ['web/js/align.js', 'web/js/any_switch.js', 'web/js/anything_everywhere.js', 'web/js/auto_bypass.js', 'web/js/auto_model_downloader.js', 'web/js/comparer.js', 'web/js/fast_groups_bypasser.js', 'web/js/join_image_lists.js', 'web/js/math_expression.js', 'web/js/power_lora_loader.js', 'web/js/save_image.js', 'web/js/seed.js', 'web/js/show_text.js', 'locales/en/main.json'],
    'web/js/align.js': '46e083c5a6b641fde808f0fe574ce5b6',
    'web/js/any_switch.js': 'c22ed943fcdfac53b8552c6fe88b0101',
    'web/js/anything_everywhere.js': '07fb8ee723dd7bde4a2b93e6f0f8fb82',
    'web/js/auto_bypass.js': 'f08fdca2429222dbc47247ea4c125f88',
    'web/js/auto_model_downloader.js': '4f887ced9ee3d7cbaa35df6ae03961f4',
    'web/js/comparer.js': '5d0b032aadee6fe9d22466c17db37f6e',
    'web/js/fast_groups_bypasser.js': 'f6fd7e073a87d9ebc19fe74bb9955723',
    'web/js/join_image_lists.js': '1a9061e56dddb9e9570fcd8c3b481ee9',
    'web/js/math_expression.js': '821147e2f1dc281397bb71c2df1a9728',
    'web/js/power_lora_loader.js': 'd79b6135172ffb056a3390ebbc8e4257',
    'web/js/save_image.js': '2427f24fb6b601c9e67842958af0b1fa',
    'web/js/seed.js': '5d6e4c5066a55bbde8105c1644ba8a78',
    'web/js/show_text.js': 'd2dcf7634221e0e44423eb7e1bedd185',
    'locales/en/main.json': '200e6c5790ea53ee55ee4c645b420d7a',
    'SeedVR2PostProcess.METHODS': ['lab', 'wavelet', 'adain', 'none'],
}

WHERE = Where({
    "LUTS_DIR": "pipelines.postfx:LUTS_DIR",
})

ROOT = "bcnodes_surface_under_test"

KEYS = [
    "BC_LogicBoolean", "BC_IsMaskEmpty", "BC_MaskFillHoles", "BC_MaskGrow", "BC_ImageScaleByAspectRatio",
    "BC_JoinImageLists", "BC_BiRefNetRemoveBackground", "BC_AutoModelDownloader", "BC_MathExpression",
    "BC_PromptList", "BC_AnySwitch", "BC_Seed", "BC_ShowText", "BC_ImageComparer", "BC_VideoComparer",
    "BC_PowerLoraLoader", "BC_AnythingEverywhere", "BC_FastGroupsBypasser", "BC_SeedVR2Resize",
    "BC_SeedVR2VAEEncode", "BC_SeedVR2VAEDecode", "BC_SeedVR2PostProcess", "BC_PostFxApply", "BC_PostFxTheme",
    "BC_PostFxCustomLook", "BC_PostFxLut", "BC_PostFxSignatureSheet", "BC_CaptionAudit", "BC_SocialMediaExport",
    "BC_ImageQualityGate", "BC_SaveImage", "BC_SkinTexture",
]

ATTRS = ("CATEGORY", "FUNCTION", "OUTPUT_NODE", "RETURN_TYPES", "RETURN_NAMES", "INPUT_IS_LIST", "OUTPUT_IS_LIST",
         "OUTPUT_TOOLTIPS", "SEARCH_ALIASES", "DESCRIPTION")
ABSENT = "<absent>"
LUTS_MASK = "<luts/*.cube>"


@pytest.fixture(scope="module")
def root(comfy_stubs):
    """The root package executed from its __init__.py, as ComfyUI's loader does; every module of
    this binding leaves sys.modules afterwards."""
    spec = importlib.util.spec_from_file_location(ROOT, os.path.join(_harness.PKG_DIR, "__init__.py"),
                                                  submodule_search_locations=[_harness.PKG_DIR])
    module = importlib.util.module_from_spec(spec)
    sys.modules[ROOT] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        for key in [k for k in sys.modules if k == ROOT or k.startswith(ROOT + ".")]:
            del sys.modules[key]


@pytest.fixture
def seams(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name in ("pillow_avif", "pillow_jxl"):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec, raising=True)
    monkeypatch.setattr(sys.modules["folder_paths"], "get_filename_list", lambda name: [], raising=True)


def _norm(x):
    """INPUT_TYPES / RETURN_TYPES as plain data: the flexible optional mapping and the wildcard
    type spelled out, containers kept as their own type (a tuple stays a tuple)."""
    kind = type(x).__name__
    if kind == "FlexibleOptionalInputType":
        return {"__flex__": _norm(x.socket_type), "known": _norm(dict(x.known))}
    if kind == "AnyType":
        return "*"
    if isinstance(x, dict):
        return {k: _norm(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_norm(v) for v in x)
    return x


def _surface(root, key):
    cls = root.NODE_CLASS_MAPPINGS[key]
    input_types = _norm(cls.INPUT_TYPES())
    if key == "BC_PostFxLut":
        lut = input_types["required"]["lut"]
        input_types["required"]["lut"] = (LUTS_MASK,) + lut[1:]
    out = {"display_name": root.NODE_DISPLAY_NAME_MAPPINGS.get(key, ABSENT), "class_name": cls.__name__}
    out.update({attr: _norm(getattr(cls, attr, ABSENT)) for attr in ATTRS})
    out["has_IS_CHANGED"] = hasattr(cls, "IS_CHANGED")
    out["INPUT_TYPES"] = input_types
    return out


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def test_keys_in_order(root):
    assert list(root.NODE_CLASS_MAPPINGS) == KEYS
    assert list(root.NODE_DISPLAY_NAME_MAPPINGS) == KEYS
    assert root.__all__ == ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


@pytest.mark.parametrize("key", KEYS)
def test_node_surface(key, root, seams):
    check_env(ENV, "postfx")
    check(GOLDEN, key, digest(_surface(root, key)))


def test_web_directory(root):
    check(GOLDEN, "WEB_DIRECTORY", root.WEB_DIRECTORY)


def test_frontend_files():
    web = os.path.join(_harness.PKG_DIR, "web", "js")
    files = [f"web/js/{name}" for name in sorted(os.listdir(web))] + ["locales/en/main.json"]
    check(GOLDEN, "frontend files", files)
    for rel in files:
        check(GOLDEN, rel, _md5(os.path.join(_harness.PKG_DIR, *rel.split("/"))))


def test_lut_combo_lists_the_repo_luts_folder(root, seams):
    luts = os.path.join(_harness.PKG_DIR, "luts")
    cubes = sorted(name for name in (os.listdir(luts) if os.path.isdir(luts) else [])
                   if os.path.splitext(name)[1].lower() == ".cube")
    assert list(root.NODE_CLASS_MAPPINGS["BC_PostFxLut"].INPUT_TYPES()["required"]["lut"][0]) == ["none"] + cubes


def test_luts_dir_is_the_repo_luts_folder(bcnodes):
    assert os.path.realpath(WHERE["LUTS_DIR"]) == os.path.realpath(os.path.join(_harness.PKG_DIR, "luts"))


def test_seedvr2_postprocess_methods(root):
    check(GOLDEN, "SeedVR2PostProcess.METHODS", root.NODE_CLASS_MAPPINGS["BC_SeedVR2PostProcess"].METHODS)

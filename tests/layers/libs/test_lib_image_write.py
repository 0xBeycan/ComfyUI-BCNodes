"""Golden: Save Image's format list, output_extensions (target libs/image_write.py).

output_extensions() is the base formats, with ".avif" / ".jxl" put in front when the Pillow
plugin module pillow_avif / pillow_jxl can be found. importlib.util.find_spec (looked up at call
time) is patched per case to report the plugins present or absent; the other module names pass
through to the real one. Each case pins the returned list and the plugin names asked for, in
order.
"""

import importlib.machinery
import importlib.util

import pytest

from _golden import Where, check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'no_plugins': "(['.webp', '.png', '.jpg', '.jpeg', '.j2k', '.jp2', '.gif', '.tiff', '.bmp'], ['pillow_avif', 'pillow_jxl'])",
    'avif': "(['.avif', '.webp', '.png', '.jpg', '.jpeg', '.j2k', '.jp2', '.gif', '.tiff', '.bmp'], ['pillow_avif', 'pillow_jxl'])",
    'jxl': "(['.jxl', '.webp', '.png', '.jpg', '.jpeg', '.j2k', '.jp2', '.gif', '.tiff', '.bmp'], ['pillow_avif', 'pillow_jxl'])",
    'avif_and_jxl': "(['.jxl', '.avif', '.webp', '.png', '.jpg', '.jpeg', '.j2k', '.jp2', '.gif', '.tiff', '.bmp'], ['pillow_avif', 'pillow_jxl'])",
}

WHERE = Where({
    "output_extensions": "libs.image_write:output_extensions",
})

PLUGIN_MODULES = ("pillow_avif", "pillow_jxl")

CASES = {
    "no_plugins": (),
    "avif": ("pillow_avif",),
    "jxl": ("pillow_jxl",),
    "avif_and_jxl": ("pillow_avif", "pillow_jxl"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_output_extensions(name, bcnodes, monkeypatch):
    check_env(ENV)
    present = CASES[name]
    real_find_spec = importlib.util.find_spec
    asked = []

    def find_spec(module, package=None):
        if module in PLUGIN_MODULES:
            asked.append(module)
            return importlib.machinery.ModuleSpec(module, None) if module in present else None
        return real_find_spec(module, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec, raising=True)
    check(GOLDEN, name, repr((WHERE["output_extensions"](), asked)))

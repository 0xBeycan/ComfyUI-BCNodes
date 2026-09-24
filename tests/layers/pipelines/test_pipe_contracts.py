"""Key order of the TypedDict contracts (plan 6.2).

The TypedDicts document dicts that stay dicts (served as JSON by the routes, indexed by the
existing tests); these cases hold each one to the keys, in order, that its producer builds:
  - pipelines/model_download.py: resolve_entry -> ResolvedEntry; describe -> DescribedItem for
    an entry that resolves (missing and present files), DescribedError for one that does not.
  - pipelines/social_export.py: render -> RenderMeta for each of its four strategies (the pad
    producer and the other one); PlatformSpec -> the "_schema" keys of nodes/social_specs.json,
    and every shipped platform uses only those keys (besides its '_' notes).

folder_paths is the stub_comfy module with a per-test models dir; URLs are placeholders.
"""

import json
import os
import sys

import pytest

from _golden import Where
from _harness import PKG_DIR

WHERE = Where({
    "ResolvedEntry": "pipelines.model_download:ResolvedEntry",
    "DescribedItem": "pipelines.model_download:DescribedItem",
    "DescribedError": "pipelines.model_download:DescribedError",
    "resolve_entry": "pipelines.model_download:resolve_entry",
    "describe": "pipelines.model_download:describe",
    "RenderMeta": "pipelines.social_export:RenderMeta",
    "PlatformSpec": "pipelines.social_export:PlatformSpec",
    "render": "pipelines.social_export:render",
})

HF = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
CIVITAI = "https://civitai.com/api/download/models/12345"

# Entries that resolve: keep the URL name, rename, both flags, the legacy flag of each service.
RESOLVED = [
    {"url": HF, "dir": "checkpoints"},
    {"url": HF, "dir": "loras/renamed.safetensors", "hf": True},
    {"url": HF, "dir": "checkpoints", "hf": True, "civitai": True},
    {"url": HF, "dir": "checkpoints", "token": True},
    {"url": CIVITAI, "dir": "loras/model_a.safetensors", "token": True},
    {"url": "https://huggingface.co/org/repo/resolve/main/present.safetensors", "dir": "vae"},
]

# Entries that do not: a host outside the list, no file name, an empty url, no fields.
UNRESOLVED = [
    {"url": " https://example.com/model.safetensors ", "dir": " checkpoints ", "hf": True, "token": True},
    {"url": CIVITAI, "dir": "loras", "civitai": True},
    {"url": "", "dir": "checkpoints"},
    {},
]


@pytest.fixture
def models(bcnodes, monkeypatch, tmp_path):
    """A per-test models/ with vae/present.safetensors, so describe sees a present file too."""
    monkeypatch.setattr(sys.modules["folder_paths"], "models_dir", str(tmp_path / "models"), raising=True)
    os.makedirs(tmp_path / "models" / "vae")
    (tmp_path / "models" / "vae" / "present.safetensors").write_bytes(b"x" * 7)
    return str(tmp_path)


def _keys(contract):
    return list(WHERE[contract].__annotations__)


def test_resolved_entry_key_order(models):
    for entry in RESOLVED:
        assert list(WHERE["resolve_entry"](entry)) == _keys("ResolvedEntry"), entry


def test_described_item_key_order(models):
    items = WHERE["describe"](RESOLVED)
    assert [i["exists"] for i in items] == [False] * 5 + [True]
    for item in items:
        assert list(item) == _keys("DescribedItem"), item


def test_described_error_key_order(models):
    items = WHERE["describe"](UNRESOLVED)
    assert all("error" in i for i in items)
    for item in items:
        assert list(item) == _keys("DescribedError"), item


def test_described_item_extends_resolved_entry(bcnodes):
    assert _keys("DescribedItem") == _keys("ResolvedEntry") + ["exists", "size"]


# A single-point 9:16 band. Each master forces one strategy: 9:16 is in the band; 10:16 crops
# 10% of the pixels (under the 15% budget); 16:9 crops 68%, over the budget, so the overflow
# strategy decides.
STORY = {"ar_min": 0.5625, "ar_max": 0.5625, "max_w": 1080, "max_h": 1920, "format": "jpg"}
RENDER = [
    ((90, 160), "error", "scale"),
    ((100, 160), "error", "crop"),
    ((160, 90), "crop", "crop!"),
    ((160, 90), "pad", "pad"),
]


@pytest.mark.parametrize("size, overflow, strategy", RENDER)
def test_render_meta_key_order(bcnodes, size, overflow, strategy):
    from PIL import Image
    _, meta = WHERE["render"](Image.new("RGB", size), STORY, overflow_strategy=overflow, platform="platform_a")
    assert meta["strategy"] == strategy
    assert list(meta) == _keys("RenderMeta")


def test_platform_spec_keys_follow_the_shipped_schema(bcnodes):
    with open(os.path.join(PKG_DIR, "nodes", "social_specs.json"), encoding="utf-8") as fh:
        specs = json.load(fh)
    assert [k for k in specs["_schema"] if not k.startswith("_")] == _keys("PlatformSpec")
    for name, spec in specs.items():
        if not name.startswith("_"):
            assert {k for k in spec if not k.startswith("_")} <= set(_keys("PlatformSpec")), name

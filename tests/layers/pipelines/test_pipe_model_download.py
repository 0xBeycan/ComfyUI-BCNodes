"""Golden: the downloader's entry grammar and "already asked" store (plan 8.2 row 20).

pipelines/model_download.py:
  - parse_entries, resolve_entry and describe: the resolved dict (key order included) or the
    error message for a rename, a blob link, the legacy token flag for both services,
    subdomains, backslashes, a symlinked subdirectory, a symlinked file that leaves it, `../`,
    no extension, encoded separators, a non-http scheme, a host outside the list, empty or
    missing fields; describe's error items (which skip the legacy token mapping, 12 B-4) and
    a non-dict entry;
  - entries_key; missing_tokens (explicit tokens and the stored ones) and its message;
  - the seen store: read_seen / mark_seen on hex keys, a garbage or non-list file, and the
    mixed-type sequence on an empty store (12 B C30): mark_seen(5) writes [5]; mark_seen("abc")
    raises TypeError from sorted() after open(path, "w") truncated the file, which is left
    empty; mark_seen("abc") again reads that as an empty store and writes ["abc"].

folder_paths is the stub_comfy module with a per-test models and user dir; temp paths are
masked as <tmp>. Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import json
import os
import re
import sys

import pytest

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'parse_entries/empty': '1172643cb3261823831abc539fe61b0f',
    'parse_entries/blank': '1172643cb3261823831abc539fe61b0f',
    'parse_entries/none': '1172643cb3261823831abc539fe61b0f',
    'parse_entries/empty_list': '1172643cb3261823831abc539fe61b0f',
    'parse_entries/blank_line': '1172643cb3261823831abc539fe61b0f',
    'parse_entries/not_a_list': '4681eab563607a153df487b3abd09ea5',
    'parse_entries/not_json': 'e75ebc61ffa90348994a601e574b496c',
    'parse_entries/mixed': '61d687e80eac54c4be100c26b3844566',
    'resolve_entry/keep_name': '1956667d23d8d24b3b6543fa2967db98',
    'resolve_entry/rename': '9f87a7846ea178b078dc5ad5ba43441d',
    'resolve_entry/rename_upper_ext': '2cd2f3cda699524818829fc7f22b9da1',
    'resolve_entry/rename_yaml': 'fc5d38add667700bbc383082ab0ff649',
    'resolve_entry/blob_to_resolve': '1956667d23d8d24b3b6543fa2967db98',
    'resolve_entry/blob_with_query': '38df51152907113022b30256c73adea3',
    'resolve_entry/url_with_fragment': 'c2b689866ce51cbbe8b5801af2f3d8f8',
    'resolve_entry/legacy_token_hf': '7dac84c2ce6e3116c86fbc694a85ad8f',
    'resolve_entry/legacy_token_civitai': 'dccaf1413d511b8d2c3a1a25de77852d',
    'resolve_entry/flags_hf_and_civitai': '25daaacc52a32d87e15f372950b5fe06',
    'resolve_entry/civitai_flag': 'dccaf1413d511b8d2c3a1a25de77852d',
    'resolve_entry/subdomain': '092b277c0acc4a6c015bc2e38a7dbf85',
    'resolve_entry/uppercase_scheme_and_host': 'ff5c190424b547f980949489888ee95d',
    'resolve_entry/backslashes': '00b7cbacc15c49593fe43e6e4ce0b183',
    'resolve_entry/slashes_and_spaces': '00b7cbacc15c49593fe43e6e4ce0b183',
    'resolve_entry/absolute_dir': 'fea0645a664781b187c20f89fd1d1476',
    'resolve_entry/dotdot_dir': 'ff37bb8fc538cd86603e84d431a306fc',
    'resolve_entry/dotdot_inside': '76333c74a282d0773d33c9bdfaf399b3',
    'resolve_entry/dotdot_that_stays': '8a2954bc5d3b85b4135fd40cb1745c35',
    'resolve_entry/dotdot_in_renamed_file': '3ba2b66d4570c90b39ed1765a7eda267',
    'resolve_entry/no_extension': 'e52fca3175489d9c2abe03b8b8ff5dad',
    'resolve_entry/encoded_slash': '477119e6502c850e0c0c8f23fc1d443c',
    'resolve_entry/encoded_backslash': 'ffc03def5665b90708b7fc5c0219a857',
    'resolve_entry/encoded_dotdot': '5c6f297e58e2fdb17c957eec2dace1c4',
    'resolve_entry/non_http': '1e5ab5bc857b3d8ac84340edb1dd48a5',
    'resolve_entry/host_outside': '7425e6df5880ab9c381d3bbab9ecb1d1',
    'resolve_entry/lookalike_host': '59738050941ae914714f8b4648d4b3e0',
    'resolve_entry/empty_url': '03835a05ca7dbb2f057aca21f6b2924e',
    'resolve_entry/blank_url': '03835a05ca7dbb2f057aca21f6b2924e',
    'resolve_entry/empty_dir': 'fdc35d0a5826bc0ac67450cbeb436b54',
    'resolve_entry/none_fields': '03835a05ca7dbb2f057aca21f6b2924e',
    'resolve_entry/missing_fields': '03835a05ca7dbb2f057aca21f6b2924e',
    'resolve_entry/symlinked_subdir': 'd3f7dc38a35c8f6aedc8a3bb36c52fa4',
    'resolve_entry/symlinked_subdir_rename': '3f317afffcde088e3afff05da24b9272',
    'resolve_entry/symlinked_file_leaves_dir': '7934c82bab369aad4951c5ee6427a56a',
    'describe/lines': 'f2dd3702b8c30b4c51ffeb5683f13d4d',
    'describe/non_dict': '2cc0597b06ee20ba437c4a1930b8e1da',
    'entries_key': 'b41bce2eb86bedfb58088c6196e47ab7',
    'missing_tokens/stored_none': '39143f435737d549c8e9c8af2ec979aa',
    'missing_tokens/stored_hf': 'c9f15967034f095045db4aac8c826d33',
    'missing_tokens/stored_both': '6388882cfb01ff4e0156e86155b0f1b6',
    'missing_tokens/explicit_civitai_only': 'a668af72763f27b58308c44f7af9f125',
    'missing_tokens/explicit_empty_values': '39143f435737d549c8e9c8af2ec979aa',
    'seen/hex_keys': 'e1f69aa81c1e69dafbc3140c15e6e4ba',
    'seen/bad_file/garbage': 'b361e757e116821e3cc13e67fb40ae08',
    'seen/bad_file/non_list': 'b361e757e116821e3cc13e67fb40ae08',
    'seen/bad_file/empty_file': 'b361e757e116821e3cc13e67fb40ae08',
    'seen/mixed_types': 'bb8ca10d1c3f3b6c8196825a0ecb772a',
}

WHERE = Where({
    "parse_entries": "pipelines.model_download:parse_entries",
    "resolve_entry": "pipelines.model_download:resolve_entry",
    "describe": "pipelines.model_download:describe",
    "entries_key": "pipelines.model_download:entries_key",
    "missing_tokens": "pipelines.model_download:missing_tokens",
    "missing_tokens_message": "pipelines.model_download:missing_tokens_message",
    "read_seen": "pipelines.model_download:read_seen",
    "mark_seen": "pipelines.model_download:mark_seen",
    "seen_file": "pipelines.model_download:seen_file",
    "save_tokens": "libs.download:save_tokens",
})

HF = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
CIVITAI = "https://civitai.com/api/download/models/12345"


@pytest.fixture
def dirs(bcnodes, monkeypatch, tmp_path):
    """Per-test models/ and user/ under tmp_path; returns tmp_path as a string."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path / "user"), raising=True)
    (tmp_path / "models").mkdir()
    return str(tmp_path)


def _mask(x, tmp):
    if isinstance(x, str):
        return x.replace(tmp, "<tmp>")
    if isinstance(x, dict):
        return {_mask(k, tmp): _mask(v, tmp) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_mask(v, tmp) for v in x)
    return x


# sorted() over a set holding an int and a str names the two types in the set's iteration
# order, which follows the str hash seed (PYTHONHASHSEED); the pair is put in sorted order.
_UNORDERABLE = re.compile(r"'<' not supported between instances of '(\w+)' and '(\w+)'")


def _exception(e):
    text = str(e)
    m = _UNORDERABLE.fullmatch(text)
    if m:
        text = "'<' not supported between instances of '{}' and '{}'".format(*sorted(m.groups()))
    return type(e).__name__, text


def _call(fn, *args):
    try:
        return {"out": fn(*args)}
    except Exception as e:
        return {"exception": _exception(e)}


# ---------------------------------------------------------------------------
# parse_entries
# ---------------------------------------------------------------------------

TEXTS = {
    "empty": "", "blank": "  \n", "none": None, "empty_list": "[]",
    "blank_line": json.dumps([{"url": " ", "dir": ""}]),
    "not_a_list": json.dumps({"url": HF, "dir": "checkpoints"}),
    "not_json": "not json",
    "mixed": json.dumps([{"url": HF, "dir": "checkpoints"}, "text", 5, None, {"dir": "loras"}, {"url": " x "},
                         {"url": None, "dir": None}, {"other": 1}]),
}


@pytest.mark.parametrize("name", list(TEXTS))
def test_parse_entries(name, bcnodes):
    check_env(ENV)
    check(GOLDEN, f"parse_entries/{name}", digest(_call(WHERE["parse_entries"], TEXTS[name])))


# ---------------------------------------------------------------------------
# resolve_entry
# ---------------------------------------------------------------------------

ENTRIES = {
    "keep_name": {"url": HF, "dir": "checkpoints"},
    "rename": {"url": HF, "dir": "loras/renamed.safetensors"},
    "rename_upper_ext": {"url": HF, "dir": "loras/Renamed.SAFETENSORS"},
    "rename_yaml": {"url": "https://huggingface.co/org/repo/resolve/main/config.yaml", "dir": "configs/model.yaml"},
    "blob_to_resolve": {"url": "https://huggingface.co/org/repo/blob/main/model.safetensors", "dir": "checkpoints"},
    "blob_with_query": {"url": " https://huggingface.co/org/repo/blob/main/sub/model.safetensors?download=true ", "dir": "checkpoints"},
    "url_with_fragment": {"url": HF + "#frag", "dir": "checkpoints"},
    "legacy_token_hf": {"url": HF, "dir": "checkpoints", "token": True},
    "legacy_token_civitai": {"url": CIVITAI, "dir": "loras/model_a.safetensors", "token": True},
    "flags_hf_and_civitai": {"url": HF, "dir": "checkpoints", "hf": True, "civitai": True},
    "civitai_flag": {"url": CIVITAI, "dir": "loras/model_a.safetensors", "civitai": 1},
    "subdomain": {"url": "https://cdn-lfs.huggingface.co/org/model.safetensors", "dir": "checkpoints"},
    "uppercase_scheme_and_host": {"url": "HTTP://HuggingFace.co/org/repo/resolve/main/model.safetensors", "dir": "checkpoints"},
    "backslashes": {"url": HF, "dir": "\\loras\\sub\\"},
    "slashes_and_spaces": {"url": HF, "dir": " /loras/sub/ "},
    "absolute_dir": {"url": HF, "dir": "/etc"},
    "dotdot_dir": {"url": HF, "dir": "../x"},
    "dotdot_inside": {"url": HF, "dir": "loras/../../x"},
    "dotdot_that_stays": {"url": HF, "dir": "loras/../checkpoints"},
    "dotdot_in_renamed_file": {"url": HF, "dir": "loras/a..b.safetensors"},
    "no_extension": {"url": CIVITAI, "dir": "loras"},
    "encoded_slash": {"url": "https://huggingface.co/org/repo/resolve/main/x%2F..%2Fmodel.safetensors", "dir": "loras"},
    "encoded_backslash": {"url": "https://huggingface.co/org/repo/resolve/main/x%5Cmodel.safetensors", "dir": "loras"},
    "encoded_dotdot": {"url": "https://huggingface.co/org/repo/resolve/main/%2E%2E.safetensors", "dir": "loras"},
    "non_http": {"url": "ftp://huggingface.co/org/model.safetensors", "dir": "checkpoints"},
    "host_outside": {"url": "https://example.com/model.safetensors", "dir": "checkpoints"},
    "lookalike_host": {"url": "https://huggingface.co.evil.example/model.safetensors", "dir": "checkpoints"},
    "empty_url": {"url": "", "dir": "checkpoints"},
    "blank_url": {"url": "  ", "dir": "checkpoints"},
    "empty_dir": {"url": HF, "dir": ""},
    "none_fields": {"url": None, "dir": None},
    "missing_fields": {},
}


def _symlinks(tmp):
    """models/linked -> <tmp>/elsewhere (allowed: the check uses the directory's realpath), and
    models/loras/escape.safetensors -> <tmp>/outside.safetensors (refused)."""
    elsewhere = os.path.join(tmp, "elsewhere")
    os.makedirs(elsewhere)
    os.symlink(elsewhere, os.path.join(tmp, "models", "linked"))
    os.makedirs(os.path.join(tmp, "models", "loras"))
    with open(os.path.join(tmp, "outside.safetensors"), "wb"):
        pass
    os.symlink(os.path.join(tmp, "outside.safetensors"), os.path.join(tmp, "models", "loras", "escape.safetensors"))


SYMLINK_ENTRIES = {
    "symlinked_subdir": {"url": HF, "dir": "linked"},
    "symlinked_subdir_rename": {"url": HF, "dir": "linked/renamed.safetensors"},
    "symlinked_file_leaves_dir": {"url": HF, "dir": "loras/escape.safetensors"},
}


@pytest.mark.parametrize("name", list(ENTRIES) + list(SYMLINK_ENTRIES))
def test_resolve_entry(name, dirs):
    check_env(ENV)
    _symlinks(dirs)
    entry = {**ENTRIES, **SYMLINK_ENTRIES}[name]
    check(GOLDEN, f"resolve_entry/{name}", digest(_mask(_call(WHERE["resolve_entry"], entry), dirs)))


# ---------------------------------------------------------------------------
# describe, entries_key, missing_tokens
# ---------------------------------------------------------------------------

LINES = [
    {"url": HF, "dir": "checkpoints", "hf": True},  # missing, needs the HF token
    {"url": CIVITAI, "dir": "loras/model_a.safetensors", "civitai": True},  # missing, needs the Civitai token
    {"url": "https://huggingface.co/org/repo/resolve/main/present.safetensors", "dir": "vae", "hf": True},  # present
    {"url": CIVITAI, "dir": "loras/legacy.safetensors", "token": True},  # legacy flag -> civitai
    {"url": " https://example.com/model.safetensors ", "dir": " checkpoints ", "hf": True, "token": True},  # error item
    {"url": CIVITAI, "dir": "loras", "civitai": True},  # error item: no file name
]


def _present(tmp):
    os.makedirs(os.path.join(tmp, "models", "vae"))
    with open(os.path.join(tmp, "models", "vae", "present.safetensors"), "wb") as f:
        f.write(b"x" * 7)


def test_describe(dirs):
    check_env(ENV)
    _present(dirs)
    items = WHERE["describe"](LINES)
    check(GOLDEN, "describe/lines", digest(_mask(items, dirs)))


def test_describe_non_dict_entry(dirs):
    check_env(ENV)
    check(GOLDEN, "describe/non_dict", digest(_call(WHERE["describe"], [LINES[0], "text"])))


def test_entries_key(dirs):
    check_env(ENV)
    _present(dirs)
    valid = [i for i in WHERE["describe"](LINES) if not i.get("error")]
    keys = [WHERE["entries_key"](valid), WHERE["entries_key"](valid[::-1]), WHERE["entries_key"](valid[:1]),
            WHERE["entries_key"]([])]
    check(GOLDEN, "entries_key", digest(keys))


@pytest.mark.parametrize("name, stored, explicit", [
    ("stored_none", {}, None),
    ("stored_hf", {"huggingface": "hf-placeholder-token"}, None),
    ("stored_both", {"huggingface": "hf-placeholder-token", "civitai": "civitai-placeholder-token"}, None),
    ("explicit_civitai_only", {"huggingface": "hf-placeholder-token"}, {"civitai": "x"}),
    ("explicit_empty_values", {}, {"huggingface": "", "civitai": None}),
])
def test_missing_tokens(name, stored, explicit, dirs):
    check_env(ENV)
    _present(dirs)
    WHERE["save_tokens"](stored)
    items = WHERE["describe"](LINES)
    missing = WHERE["missing_tokens"](items) if explicit is None else WHERE["missing_tokens"](items, explicit)
    check(GOLDEN, f"missing_tokens/{name}", digest([missing, WHERE["missing_tokens_message"](missing)]))


# ---------------------------------------------------------------------------
# Seen store
# ---------------------------------------------------------------------------

def _seen(tmp):
    path = WHERE["seen_file"]()
    data = None
    if os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
    return {"path": _mask(path, tmp), "file": data}


def test_seen_hex_keys(dirs):
    check_env(ENV)
    steps = [("read_empty", WHERE["read_seen"](), _seen(dirs))]
    for key in ("b7e23ec29af22b0b", "0a1b2c3d4e5f6071", "b7e23ec29af22b0b"):
        steps.append((key, WHERE["mark_seen"](key), WHERE["read_seen"](), _seen(dirs)))
    check(GOLDEN, "seen/hex_keys", digest(steps))


@pytest.mark.parametrize("name, content", [
    ("garbage", b"[not json"), ("non_list", b'{"b7e23ec29af22b0b": true}'), ("empty_file", b""),
])
def test_seen_bad_file(name, content, dirs):
    check_env(ENV)
    path = WHERE["seen_file"]()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    out = [WHERE["read_seen"](), WHERE["mark_seen"]("b7e23ec29af22b0b"), _seen(dirs)]
    check(GOLDEN, f"seen/bad_file/{name}", digest(out))


def test_seen_mixed_types_on_empty_store(dirs):
    check_env(ENV)
    steps = []
    for key in (5, "abc", "abc"):
        steps.append((repr(key), _call(WHERE["mark_seen"], key), _seen(dirs)))
    check(GOLDEN, "seen/mixed_types", digest(steps))

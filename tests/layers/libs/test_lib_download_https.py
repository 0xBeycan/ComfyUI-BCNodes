"""F4: a stored token is sent over https only.

A downloader line may be written as http:// (resolve_entry accepts both schemes), and _auth
attaches a stored token whatever the scheme: the Hugging Face Authorization header, or token= in
the Civitai query. Before F4 that first request crossed the network in cleartext; both hosts
redirect to https only after it. F4 sends every request that carries a stored token to https://
on the same host, path and query.

- Property (red on F4's parent): for four http:// lines with a stored token, the first hop goes
  to https:// with the rest of the line unchanged and carries the token. Over every case of this
  file, no http:// hop carries a stored token.
- UNCHANGED (recorded with BCNODES_GOLDEN_RECORD=1 on F4's parent, asserted since): for those
  four cases the file bytes, progress calls and exception text (a message names the line as
  written, F1); for seven cases F4 does not touch, the whole capture, hops included.

Tokens are placeholders; every request is answered by _golden.fake_http.
"""

import itertools
import os
import sys
import time
import urllib.parse

import pytest

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

UNCHANGED = {
    'hf_http_200': 'b7d471908da3f8fbaa404c87744d4a97',
    'hf_mixed_case_302_cdn': 'b7d471908da3f8fbaa404c87744d4a97',
    'civitai_http_query_401': 'f191663bfc27ac43a7d1f22272ae79ac',
    'civitai_http_307_r2': 'b7d471908da3f8fbaa404c87744d4a97',
    'hf_http_no_token': 'd8a1279c7d791d3600fd37e7c2cbc5f8',
    'hf_http_civitai_token_only': 'd8a1279c7d791d3600fd37e7c2cbc5f8',
    'civitai_http_no_token': 'b90adf6b7c0bbbd450af0c07c69902b9',
    'civitai_http_own_token': '61265db9d1664aa6647c0cc20616eaa6',
    'hf_https_token': 'c9233dcce1d26ad04c2c874a2186e188',
    'civitai_https_token': '7fe022c6582dab39205a12e9de246ccd',
    'other_host_http_both_tokens': 'bb5773accd7046939fc63ae5de98bc8b',
}

WHERE = Where({
    "download_file": "libs.download:download_file",
    "save_tokens": "libs.download:save_tokens",
})

HF_TOKEN = "hf-placeholder-token"
CIVITAI_TOKEN = "civitai-placeholder/+=token"
QUOTED = urllib.parse.quote(CIVITAI_TOKEN)
HF_PATH = "huggingface.co/org/repo/resolve/main/model.safetensors"
CIVITAI_PATH = "civitai.com/api/download/models/12345"
CDN_URL = "https://us.aws.cdn.hf.co/xet-bridge-us/placeholder/model?X-Amz-Signature=placeholder"
R2_URL = "https://bucket.r2.cloudflarestorage.com/model?X-Amz-Signature=placeholder"
BINARY = {"Content-Type": "application/octet-stream"}
BODY = b"0123456789"
OK = (200, {**BINARY, "Content-Length": "10"}, BODY)

# name -> (stored tokens, line url, first request F4 must send, fake_http script)
UPGRADED = {
    "hf_http_200": (
        {"huggingface": HF_TOKEN}, f"http://{HF_PATH}?download=true", f"https://{HF_PATH}?download=true",
        {"huggingface.co": OK}),
    "hf_mixed_case_302_cdn": (
        {"huggingface": HF_TOKEN}, "HTTP://HuggingFace.co/org/repo/resolve/main/model.safetensors",
        "https://HuggingFace.co/org/repo/resolve/main/model.safetensors",
        {"huggingface.co": (302, {"Location": CDN_URL}, b""), "us.aws.cdn.hf.co": OK}),
    "civitai_http_query_401": (
        {"civitai": CIVITAI_TOKEN}, f"http://{CIVITAI_PATH}?type=Model&format=SafeTensor",
        f"https://{CIVITAI_PATH}?type=Model&format=SafeTensor&token={QUOTED}",
        {"civitai.com": (401, {}, b"")}),
    "civitai_http_307_r2": (
        {"civitai": CIVITAI_TOKEN}, f"http://{CIVITAI_PATH}", f"https://{CIVITAI_PATH}?token={QUOTED}",
        {"civitai.com": (307, {"Location": R2_URL}, b""), "bucket.r2.cloudflarestorage.com": OK}),
}

# name -> (stored tokens, line url, fake_http script)
UNTOUCHED = {
    "hf_http_no_token": ({}, f"http://{HF_PATH}", {"huggingface.co": OK}),
    "hf_http_civitai_token_only": ({"civitai": CIVITAI_TOKEN}, f"http://{HF_PATH}", {"huggingface.co": OK}),
    "civitai_http_no_token": ({}, f"http://{CIVITAI_PATH}", {"civitai.com": OK}),
    "civitai_http_own_token": (  # the user's own token= in the line (12 B-36): no stored token is attached
        {"civitai": CIVITAI_TOKEN}, f"http://{CIVITAI_PATH}?token=own-placeholder", {"civitai.com": OK}),
    "hf_https_token": ({"huggingface": HF_TOKEN}, f"https://{HF_PATH}", {"huggingface.co": OK}),
    "civitai_https_token": ({"civitai": CIVITAI_TOKEN}, f"https://{CIVITAI_PATH}", {"civitai.com": OK}),
    "other_host_http_both_tokens": (
        {"huggingface": HF_TOKEN, "civitai": CIVITAI_TOKEN}, "http://example.com/model.safetensors", {"example.com": OK}),
}

CASES = {**{name: (t, url, script) for name, (t, url, _, script) in UPGRADED.items()}, **UNTOUCHED}


def run_case(case, monkeypatch, tmp):
    """download_file over the scripted hops; returns {"hops", "progress", "files", "exception"}."""
    tokens, url, script = case
    monkeypatch.setattr(sys.modules["folder_paths"], "get_user_directory", lambda: os.path.join(tmp, "user"), raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    WHERE["save_tokens"](tokens)
    path = os.path.join(tmp, "models", "checkpoints", "model.safetensors")
    hops = _golden.fake_http(monkeypatch, script)
    progress = []
    out = {}
    try:
        WHERE["download_file"](url, path, lambda downloaded, total: progress.append((downloaded, total)))
    except Exception as e:
        out["exception"] = (type(e).__name__, str(e))
    files = {}
    for suffix in ("", ".part"):
        if os.path.isfile(path + suffix):
            with open(path + suffix, "rb") as f:
                files[os.path.basename(path + suffix)] = f.read()
    return {"hops": hops, "progress": progress, "files": files, **out}


@pytest.mark.parametrize("name", list(UPGRADED))
def test_request_with_a_stored_token_goes_to_https(name, bcnodes, monkeypatch, tmp_path):
    first_hop = UPGRADED[name][2]
    hops = run_case(CASES[name], monkeypatch, str(tmp_path))["hops"]
    assert hops[0][1] == first_hop
    if "civitai" not in UPGRADED[name][0]:
        assert hops[0][2].get("Authorization") == f"Bearer {HF_TOKEN}"


@pytest.mark.parametrize("name", list(CASES))
def test_no_http_hop_carries_a_stored_token(name, bcnodes, monkeypatch, tmp_path):
    hops = run_case(CASES[name], monkeypatch, str(tmp_path))["hops"]
    cleartext = [hop[1] for hop in hops if urllib.parse.urlparse(hop[1]).scheme == "http"
                 and any(token in repr(hop[1:3]) for token in (HF_TOKEN, CIVITAI_TOKEN, QUOTED))]
    assert cleartext == [], f"a stored token sent over http: {cleartext}"


@pytest.mark.parametrize("name", list(CASES))
def test_unchanged(name, bcnodes, monkeypatch, tmp_path):
    check_env(ENV)
    capture = run_case(CASES[name], monkeypatch, str(tmp_path))
    if name in UPGRADED:
        del capture["hops"]  # what F4 changes; the property above pins the new first hop
    check(UNCHANGED, name, digest(capture))

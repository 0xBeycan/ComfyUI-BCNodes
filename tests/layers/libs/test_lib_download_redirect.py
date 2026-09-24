"""F2: a stored Hugging Face token does not follow a redirect to another host.

_auth puts `Authorization: Bearer <token>` on every huggingface.co request when an HF token is
stored, and urllib's redirect handler copies every request header to the redirect target,
whatever its host. huggingface.co hands file bodies to a CDN on another host with a 302, so
before F2 the token reached that CDN.

- Property (red on F2's parent): an HF token is stored and huggingface.co answers 302 to
  us.aws.cdn.hf.co: the first hop carries Authorization, the CDN hop none. Over every case of
  this file, no hop whose host differs from the previous hop's carries Authorization.
- UNCHANGED (recorded with BCNODES_GOLDEN_RECORD=1 on F2's parent, asserted since): hop tables,
  file bytes and progress calls of four redirects F2 does not touch.

Tokens are placeholders; every request is answered by _golden.fake_http.
"""

import itertools
import os
import sys
import time

import pytest

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

UNCHANGED = {
    'hf_307_relative_resolve_cache': 'e0ff17c11f16d7c51ce55f8adc816042',
    'civitai_307_r2': 'a7da62568124f4e68746c495e8983a18',
    'hf_302_cdn_no_token': '992b8ff1a01b39431bc7165832dab788',
    'hf_302_cdn_resume': '3820ac61dcb19976c4a903f1f930eca2',
}

WHERE = Where({
    "download_file": "libs.download:download_file",
    "save_tokens": "libs.download:save_tokens",
})

HF_TOKEN = "hf-placeholder-token"
CIVITAI_TOKEN = "civitai-placeholder/+=token"
HF_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
CDN_URL = "https://us.aws.cdn.hf.co/xet-bridge-us/placeholder/model?X-Amz-Signature=placeholder"
CIVITAI_URL = "https://civitai.com/api/download/models/12345"
R2_URL = "https://bucket.r2.cloudflarestorage.com/model?X-Amz-Signature=placeholder"
BINARY = {"Content-Type": "application/octet-stream"}
BODY = b"0123456789"

# name -> (stored tokens, url, .part content or None, fake_http script)
CASES = {
    "hf_302_cdn_stored_token": (
        {"huggingface": HF_TOKEN}, HF_URL, None,
        {"huggingface.co": (302, {"Location": CDN_URL}, b""),
         "us.aws.cdn.hf.co": (200, {**BINARY, "Content-Length": "10"}, BODY)}),
    "hf_307_relative_resolve_cache": (  # HF's same-host redirect for non-LFS files: the token stays
        {"huggingface": HF_TOKEN}, "https://huggingface.co/org/repo/resolve/main/config.json", None,
        {"huggingface.co": [(307, {"Location": "/api/resolve-cache/models/org/repo/0123abcd/config.json"}, b""),
                            (200, {"Content-Type": "application/json", "Content-Length": "10"}, b'{"a": 1}\n\n')]}),
    "civitai_307_r2": (  # the Civitai token rides in the first URL only
        {"civitai": CIVITAI_TOKEN}, CIVITAI_URL, None,
        {"civitai.com": (307, {"Location": R2_URL}, b""),
         "bucket.r2.cloudflarestorage.com": (200, {**BINARY, "Content-Length": "10"}, BODY)}),
    "hf_302_cdn_no_token": (
        {}, HF_URL, None,
        {"huggingface.co": (302, {"Location": CDN_URL}, b""),
         "us.aws.cdn.hf.co": (200, {**BINARY, "Content-Length": "10"}, BODY)}),
    "hf_302_cdn_resume": (  # a .part is present: Range on both hops, 206
        {}, HF_URL, b"01234",
        {"huggingface.co": (302, {"Location": CDN_URL}, b""),
         "us.aws.cdn.hf.co": (206, {**BINARY, "Content-Length": "5", "Content-Range": "bytes 5-9/10"}, b"56789")}),
}


def run_case(case, monkeypatch, tmp):
    """download_file over the scripted hops; returns {"hops", "progress", "files", "exception"}."""
    tokens, url, part, script = case
    monkeypatch.setattr(sys.modules["folder_paths"], "get_user_directory", lambda: os.path.join(tmp, "user"), raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    WHERE["save_tokens"](tokens)
    path = os.path.join(tmp, "models", "checkpoints", url.rsplit("/", 1)[-1])
    if part is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(part)
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


def test_hf_token_stays_on_huggingface(bcnodes, monkeypatch, tmp_path):
    hops = run_case(CASES["hf_302_cdn_stored_token"], monkeypatch, str(tmp_path))["hops"]
    assert [h[0] for h in hops] == ["huggingface.co", "us.aws.cdn.hf.co"]
    assert hops[0][2].get("Authorization") == f"Bearer {HF_TOKEN}"
    assert "Authorization" not in hops[1][2], "the HF token followed the redirect to the CDN"


@pytest.mark.parametrize("name", list(CASES))
def test_no_hop_to_another_host_carries_authorization(name, bcnodes, monkeypatch, tmp_path):
    hops = run_case(CASES[name], monkeypatch, str(tmp_path))["hops"]
    crossed = [hop[1] for prev, hop in zip(hops, hops[1:]) if hop[0] != prev[0] and "Authorization" in hop[2]]
    assert crossed == [], f"Authorization sent to another host: {crossed}"


@pytest.mark.parametrize("name", [n for n in CASES if n != "hf_302_cdn_stored_token"])
def test_unchanged(name, bcnodes, monkeypatch, tmp_path):
    check_env(ENV)
    check(UNCHANGED, name, digest(run_case(CASES[name], monkeypatch, str(tmp_path))))

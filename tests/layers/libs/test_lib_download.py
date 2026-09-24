"""Golden: the download transport and the token store (plan 8.2 row 21).

libs/download.py:
  - service_of: exact host or subdomain of HOSTS, else None; and the constants HOSTS, CHUNK,
    USER_AGENT, SERVICE_NAMES;
  - the token store: save_tokens (merge, strip, delete on empty, unknown services ignored,
    0o600), read_tokens (missing, garbage, non-dict and filtered files), tokens_present, the
    JSON bytes on disk;
  - download_file over scripted hops: 200 (with and without Content-Length, several chunks),
    206 resume, a range the server ignores, 416 with and without a .part, text/html, a short
    body, 401/403 hints for both services and none for another host, 500, URLError, a
    Civitai token appended to a query, a line carrying its own token=. Each case pins the
    per-hop log (hostname, URL, headers incl. User-Agent / Authorization / Range, timeout),
    the progress calls, the files left (size, md5) and the exception text.

Recorded on FIXED_BASE, so no message carries the stored Civitai token (F1); the redirect
cases are pinned by test_lib_download_redirect (F2), http:// lines with a stored token by
test_lib_download_https (F4). Tokens are placeholders; every request is answered by
_golden.fake_http; time.monotonic is a counter; temp paths are masked as <tmp>.
Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import hashlib
import itertools
import os
import stat
import sys
import time
import urllib.error

import pytest

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'service_of/https://huggingface.co/org/repo': '8c6a4c85126c85556bd152af680e1a2f',
    'service_of/https://HuggingFace.CO/org/repo': '8c6a4c85126c85556bd152af680e1a2f',
    'service_of/https://cdn-lfs.huggingface.co/x': '8c6a4c85126c85556bd152af680e1a2f',
    'service_of/https://civitai.com/api/download/models/1': '37cd5471b3f5fd2b9f367b1818e84d8d',
    'service_of/https://www.civitai.com/x': '37cd5471b3f5fd2b9f367b1818e84d8d',
    'service_of/http://civitai.com:8080/x': '37cd5471b3f5fd2b9f367b1818e84d8d',
    'service_of/https://user@civitai.com/x': '37cd5471b3f5fd2b9f367b1818e84d8d',
    'service_of/https://huggingface.co.evil.example/x': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/https://evilhuggingface.co/x': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/https://example.com/x': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/huggingface.co/org/repo': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/not a url': '6adf97f83acf6453d4a6a4b1070f3754',
    'service_of/https://[::1]/x': '6adf97f83acf6453d4a6a4b1070f3754',
    'constants': '5ef760884dba1b386a0b2e741fc78282',
    'tokens/sequence': 'c610c5cda631176c9578e28b0f005514',
    'tokens/read/garbage': 'c137ec60038abe77bdb7988bcdecf449',
    'tokens/read/non_dict': 'c137ec60038abe77bdb7988bcdecf449',
    'tokens/read/filtered': 'c137ec60038abe77bdb7988bcdecf449',
    'tokens/read/partly_valid': '3aa18a92609b5a8e95f9d935d0c35620',
    'tokens/read/empty_file': 'c137ec60038abe77bdb7988bcdecf449',
    'download/hf_200': 'be3270b842e3fbdfb002af73ee45b4d5',
    'download/hf_200_stored_token': '94cc970ab0b09dc16f4c5901958d34ed',
    'download/hf_200_no_length': '3b578c85e88a6af549d1714ad089baeb',
    'download/hf_200_three_chunks': 'fb8d1fa4ed1d00ae22914c59e243d798',
    'download/hf_200_three_chunks_no_length': '9183ee1436efa1e594c2c31fff4ec21e',
    'download/hf_200_no_progress_callback': '5cdc5cd91f07abe562267ad51b813555',
    'download/resume_206': 'a609deece7bbd10787796292b98cf66a',
    'download/resume_range_ignored': '9fb809f17057c3141c9341f205365e62',
    'download/resume_416_part_complete': '373e41e331fe5d9f85bb4a5b18def530',
    'download/416_without_part': '3d6fcc7eace5ed5122709d0f80e0281e',
    'download/text_html': '8a8eeb142422ba2b4b4446c1d001a5f6',
    'download/text_html_resume': 'e66a12d6a19c65e11aa338d280758513',
    'download/short_body': 'e0bc0895f6f736999988aba76e3903f5',
    'download/hf_401_stored_token': 'b15481f2c93d3822e84edbac925703a8',
    'download/hf_403_no_token': 'e7fc475cb64fde7e0e557fa085ee304f',
    'download/hf_500': '9c5d2e713bfdd01964640bf7e89a961b',
    'download/civitai_401_stored_token': '8059e394120e56eda5d294e75e516a0f',
    'download/civitai_403_no_token': 'acd8da38ff8c0ae4d0588c62f5d8ab1a',
    'download/civitai_200_stored_token_query': '76d433a27bb84a8b52c0fd16300a4286',
    'download/civitai_own_token_in_url': '2982b259a9396f1db9e7043cfdec8f09',
    'download/civitai_unreachable_stored_token': '1ead282dee25162c51fd52156f962e51',
    'download/hf_unreachable': '4674a04cddd1cc090e75a7d67c5a51e6',
    'download/other_host_401_both_tokens': '5bc2bc25da44c2fb0559c63d2f88b581',
    'download/hf_token_on_subdomain': '610ebac25410c595ab7676bc587da846',
}

WHERE = Where({
    "service_of": "libs.download:service_of",
    "save_tokens": "libs.download:save_tokens",
    "read_tokens": "libs.download:read_tokens",
    "tokens_file": "libs.download:tokens_file",
    "tokens_present": "libs.download:tokens_present",
    "download_file": "libs.download:download_file",
    "HOSTS": "libs.download:HOSTS",
    "CHUNK": "libs.download:CHUNK",
    "USER_AGENT": "libs.download:USER_AGENT",
    "SERVICE_NAMES": "libs.download:SERVICE_NAMES",
})

HF_TOKEN = "hf-placeholder-token"
CIVITAI_TOKEN = "civitai-placeholder/+=token"
HF_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
CIVITAI_URL = "https://civitai.com/api/download/models/12345"
BINARY = {"Content-Type": "application/octet-stream"}
BODY = b"0123456789"
BIG = bytes(range(256)) * (10 * 1024)  # 2.5 MiB: three CHUNK reads


@pytest.fixture
def user_dir(bcnodes, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["folder_paths"], "get_user_directory", lambda: str(tmp_path / "user"), raising=True)
    return str(tmp_path)


def _mask(x, tmp):
    if isinstance(x, str):
        return x.replace(tmp, "<tmp>")
    if isinstance(x, dict):
        return {_mask(k, tmp): _mask(v, tmp) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_mask(v, tmp) for v in x)
    return x


# ---------------------------------------------------------------------------
# service_of and constants
# ---------------------------------------------------------------------------

URLS = [
    "https://huggingface.co/org/repo", "https://HuggingFace.CO/org/repo", "https://cdn-lfs.huggingface.co/x",
    "https://civitai.com/api/download/models/1", "https://www.civitai.com/x", "http://civitai.com:8080/x",
    "https://user@civitai.com/x", "https://huggingface.co.evil.example/x", "https://evilhuggingface.co/x",
    "https://example.com/x", "huggingface.co/org/repo", "", "not a url", "https://[::1]/x",
]


@pytest.mark.parametrize("url", URLS)
def test_service_of(url, bcnodes):
    check_env(ENV)
    check(GOLDEN, f"service_of/{url}", digest(WHERE["service_of"](url)))


def test_constants(bcnodes):
    check_env(ENV)
    check(GOLDEN, "constants", digest({k: WHERE[k] for k in ("HOSTS", "CHUNK", "USER_AGENT", "SERVICE_NAMES")}))


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------

def _store(tmp):
    path = WHERE["tokens_file"]()
    if not os.path.isfile(path):
        return {"path": _mask(path, tmp), "file": None}
    with open(path, "rb") as f:
        data = f.read()
    return {"path": _mask(path, tmp), "file": data, "mode": oct(stat.S_IMODE(os.stat(path).st_mode))}


def test_token_store_sequence(user_dir):
    check_env(ENV)
    steps = [("read_empty", WHERE["read_tokens"](), _store(user_dir))]
    for update in ({"huggingface": " hf-a "}, {"civitai": "cv-b", "other": "x"}, {"huggingface": "hf-c"},
                   {"huggingface": ""}, {"civitai": None}, {"civitai": "  "}, {}):
        steps.append((repr(update), WHERE["save_tokens"](update), WHERE["tokens_present"](), _store(user_dir)))
    check(GOLDEN, "tokens/sequence", digest(steps))


@pytest.mark.parametrize("name, content", [
    ("garbage", b"{not json"),
    ("non_dict", b'["huggingface", "hf-a"]'),
    ("filtered", b'{"huggingface": 5, "civitai": "", "other": "x"}'),
    ("partly_valid", b'{"huggingface": "hf-a", "civitai": ["cv"], "other": "x"}'),
    ("empty_file", b""),
])
def test_token_store_read(name, content, user_dir):
    check_env(ENV)
    path = WHERE["tokens_file"]()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)
    out = [WHERE["read_tokens"](), WHERE["tokens_present"](), WHERE["save_tokens"]({"civitai": "cv-z"}), _store(user_dir)]
    check(GOLDEN, f"tokens/read/{name}", digest(out))


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------

def _ok(body=BODY, length=True, **headers):
    return (200, {**BINARY, **({"Content-Length": str(len(body))} if length else {}), **headers}, body)


# name -> (stored tokens, url, .part content or None, fake_http script, with on_progress)
CASES = {
    "hf_200": ({}, HF_URL, None, {"huggingface.co": _ok()}, True),
    "hf_200_stored_token": ({"huggingface": HF_TOKEN}, HF_URL, None, {"huggingface.co": _ok()}, True),
    "hf_200_no_length": ({}, HF_URL, None, {"huggingface.co": _ok(length=False)}, True),
    "hf_200_three_chunks": ({}, HF_URL, None, {"huggingface.co": _ok(BIG)}, True),
    "hf_200_three_chunks_no_length": ({}, HF_URL, None, {"huggingface.co": _ok(BIG, length=False)}, True),
    "hf_200_no_progress_callback": ({}, HF_URL, None, {"huggingface.co": _ok()}, False),
    "resume_206": ({}, HF_URL, b"01234", {"huggingface.co": (206, {**BINARY, "Content-Length": "5", "Content-Range": "bytes 5-9/10"}, b"56789")}, True),
    "resume_range_ignored": ({}, HF_URL, b"abc", {"huggingface.co": _ok()}, True),
    "resume_416_part_complete": ({}, HF_URL, BODY, {"huggingface.co": (416, {}, b"")}, True),
    "416_without_part": ({}, HF_URL, None, {"huggingface.co": (416, {}, b"")}, True),
    "text_html": ({}, HF_URL, None, {"huggingface.co": (200, {"Content-Type": "text/html; charset=utf-8"}, b"<html></html>")}, True),
    "text_html_resume": ({}, HF_URL, b"01234", {"huggingface.co": (206, {"Content-Type": "text/html"}, b"<html>")}, True),
    "short_body": ({}, HF_URL, None, {"huggingface.co": (200, {**BINARY, "Content-Length": "100"}, b"x" * 40)}, True),
    "hf_401_stored_token": ({"huggingface": HF_TOKEN}, HF_URL, None, {"huggingface.co": (401, {}, b"")}, True),
    "hf_403_no_token": ({}, HF_URL, None, {"huggingface.co": (403, {}, b"")}, True),
    "hf_500": ({"huggingface": HF_TOKEN}, HF_URL, None, {"huggingface.co": (500, {}, b"")}, True),
    "civitai_401_stored_token": ({"civitai": CIVITAI_TOKEN}, CIVITAI_URL, None, {"civitai.com": (401, {}, b"")}, True),
    "civitai_403_no_token": ({}, CIVITAI_URL, None, {"civitai.com": (403, {}, b"")}, True),
    "civitai_200_stored_token_query": ({"civitai": CIVITAI_TOKEN}, CIVITAI_URL + "?type=Model&format=SafeTensor", None, {"civitai.com": _ok()}, True),
    "civitai_own_token_in_url": ({"civitai": CIVITAI_TOKEN}, CIVITAI_URL + "?token=own-placeholder", None, {"civitai.com": _ok()}, True),
    "civitai_unreachable_stored_token": ({"civitai": CIVITAI_TOKEN}, CIVITAI_URL, None, {"civitai.com": urllib.error.URLError("host unreachable")}, True),
    "hf_unreachable": ({}, HF_URL, None, {"huggingface.co": urllib.error.URLError("host unreachable")}, True),
    "other_host_401_both_tokens": ({"huggingface": HF_TOKEN, "civitai": CIVITAI_TOKEN}, "https://example.com/model.safetensors", None, {"example.com": (401, {}, b"")}, True),
    "hf_token_on_subdomain": ({"huggingface": HF_TOKEN}, "https://cdn-lfs.huggingface.co/org/model.safetensors", None, {"cdn-lfs.huggingface.co": _ok()}, True),
}


def _files(path):
    out = {}
    for suffix in ("", ".part"):
        if os.path.isfile(path + suffix):
            with open(path + suffix, "rb") as f:
                data = f.read()
            out[os.path.basename(path + suffix)] = (len(data), hashlib.md5(data).hexdigest())
    return out


@pytest.mark.parametrize("name", list(CASES))
def test_download_file(name, user_dir, monkeypatch):
    check_env(ENV)
    tokens, url, part, script, with_progress = CASES[name]
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    WHERE["save_tokens"](tokens)
    path = os.path.join(user_dir, "models", "sub", "dir", "model.safetensors")
    if part is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".part", "wb") as f:
            f.write(part)
    hops = _golden.fake_http(monkeypatch, script)
    progress = []
    out = {}
    try:
        result = WHERE["download_file"](url, path, (lambda d, t: progress.append((d, t))) if with_progress else None)
        out["returned"] = result
    except Exception as e:
        out["exception"] = (type(e).__name__, str(e))
    out.update(hops=hops, progress=progress, files=_files(path))
    check(GOLDEN, f"download/{name}", digest(_mask(out, user_dir)))


"""F1: a stored Civitai token is never echoed.

_auth appends a stored Civitai token to the request URL. Before F1, download_file rebound `url`
to that URL, so every message it built carried the token: the job snapshot (/check, /start,
/status), the console, the file_error and done websocket events, and the node exception.

- Property (red on F1's parent): with a Civitai token stored, four failures, each run as a job
  and through the node, carry the token (raw or URL-quoted) in none of those places, and the
  first request still ends with token=<quoted token>.
- UNCHANGED (recorded with BCNODES_GOLDEN_RECORD=1 on F1's parent, asserted since): the same
  captures for four cases F1 does not touch.

Tokens are placeholders; every request is answered by _golden.fake_http.
"""

import copy
import itertools
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.parse

import pytest

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

UNCHANGED = {
    'hf_401_stored_hf_token/job': 'bae0bf3dbf377b916287fbafb6f707a6',
    'hf_401_stored_hf_token/node': '5f7f144cb33abaab955dd49af5182df0',
    'civitai_401_no_stored_token/job': '364e886a8083c718da2628bac0de2739',
    'civitai_401_no_stored_token/node': '8ec6158a27d83819c571fa2121e9ea7c',
    'civitai_url_carries_token/job': '82f887501baf18103ef0353690e2c631',
    'civitai_url_carries_token/node': 'e8e9a84c70b29ba821650e28a7ff997d',
    'hf_200/job': 'e95f211123a286906d851c45e8d0b0eb',
    'hf_200/node': 'eacc623a5ba01f512ce8942bdaa20742',
}

WHERE = Where({
    "save_tokens": "libs.download:save_tokens",
    "PromptServer": "nodes.downloader:PromptServer",
    "Job": "nodes.downloader:_Job",
    "Job.current": "nodes.downloader:_Job.current",
    "start_job": "nodes.downloader:start_job",
    "AutoModelDownloader": "nodes.downloader:AutoModelDownloader",
})

CIVITAI_TOKEN = "civitai-placeholder/+=token"  # "+" and "=" make the quoted form differ
HF_TOKEN = "hf-placeholder-token"
CIVITAI_URL = "https://civitai.com/api/download/models/12345"
HF_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
R2_URL = "https://bucket.r2.cloudflarestorage.com/model?X-Amz-Signature=placeholder"
BINARY = {"Content-Type": "application/octet-stream"}

# name -> (stored tokens, line {"url", "dir", flags}, fake_http script)
LEAK_CASES = {
    "civitai_401": (
        {"civitai": CIVITAI_TOKEN}, {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors", "civitai": True},
        {"civitai.com": (401, {}, b"")}),
    "civitai_unreachable": (
        {"civitai": CIVITAI_TOKEN}, {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors", "civitai": True},
        {"civitai.com": urllib.error.URLError("host unreachable")}),
    "civitai_html": (
        {"civitai": CIVITAI_TOKEN}, {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors", "civitai": True},
        {"civitai.com": (200, {"Content-Type": "text/html; charset=utf-8"}, b"<html>login</html>")}),
    "civitai_short_body_after_307": (
        {"civitai": CIVITAI_TOKEN}, {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors", "civitai": True},
        {"civitai.com": (307, {"Location": R2_URL}, b""),
         "bucket.r2.cloudflarestorage.com": (200, {**BINARY, "Content-Length": "100"}, b"x" * 40)}),
}

UNCHANGED_CASES = {
    "hf_401_stored_hf_token": (
        {"huggingface": HF_TOKEN}, {"url": HF_URL, "dir": "checkpoints", "hf": True},
        {"huggingface.co": (401, {}, b"")}),
    "civitai_401_no_stored_token": (
        {}, {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors"},
        {"civitai.com": (401, {}, b"")}),
    "civitai_url_carries_token": (  # the user's own token= in the line (12 B-36): _auth adds none
        {"civitai": CIVITAI_TOKEN}, {"url": CIVITAI_URL + "?token=own-placeholder", "dir": "loras/model_a.safetensors", "civitai": True},
        {"civitai.com": (401, {}, b"")}),
    "hf_200": (
        {"huggingface": HF_TOKEN}, {"url": HF_URL, "dir": "checkpoints", "hf": True},
        {"huggingface.co": (200, {**BINARY, "Content-Length": "12"}, b"model-bytes!")}),
}


class _Server:
    """Stands in for PromptServer.instance: records every send_sync as sent."""

    def __init__(self):
        self.events = []

    def send_sync(self, event, data):
        self.events.append((event, copy.deepcopy(data)))


def _files(root):
    out = []
    for d, _, names in os.walk(root):
        for name in names:
            path = os.path.join(d, name)
            with open(path, "rb") as f:
                out.append((os.path.relpath(path, root).replace(os.sep, "/"), f.read()))
    return sorted(out)


def _job_item(line, models):
    """What describe() makes of the line: every line here names its file after the directory."""
    rel_dir, filename = (line["dir"].rsplit("/", 1) if line["dir"].endswith(".safetensors")
                         else (line["dir"], line["url"].rsplit("/", 1)[-1]))
    return {"url": line["url"], "dir": rel_dir, "filename": filename, "path": os.path.join(models, rel_dir, filename)}


def run_case(case, entry_point, monkeypatch, tmp, read_stdout):
    """Runs one case as a job or through the node; returns (capture, hops).

    capture: the job snapshot and events, or the node's result or exception; stdout; the files
    under models/. read_stdout() returns what was printed since the last call."""
    tokens, line, script = case
    fp = sys.modules["folder_paths"]
    models = os.path.join(tmp, "models")
    monkeypatch.setattr(fp, "models_dir", models, raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: os.path.join(tmp, "user"), raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    monkeypatch.setattr(time, "time", lambda: 1234567890.0, raising=True)  # the job id
    server = _Server()
    WHERE.patch(monkeypatch, "PromptServer", types.SimpleNamespace(instance=server))
    WHERE.patch(monkeypatch, "Job.current", None)
    WHERE["save_tokens"](tokens)
    hops = _golden.fake_http(monkeypatch, script)
    read_stdout()

    capture = {}
    if entry_point == "job":
        job, started = WHERE["start_job"]([_job_item(line, models)])
        for thread in threading.enumerate():
            if thread.name == "bcnodes-downloader":
                thread.join(10)
        assert started and not job["running"], "the job did not finish"
        capture["snapshot"] = WHERE["Job"].snapshot()
        capture["events"] = server.events
    else:
        try:
            capture["result"] = WHERE["AutoModelDownloader"]().download(json.dumps([line]))
        except Exception as e:
            capture["exception"] = (type(e).__name__, str(e))
    capture["stdout"] = read_stdout()
    capture["files"] = _files(models)
    return capture, hops


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _strings(k)
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


@pytest.mark.parametrize("entry_point", ["job", "node"])
@pytest.mark.parametrize("name", list(LEAK_CASES))
def test_stored_civitai_token_is_not_echoed(name, entry_point, bcnodes, monkeypatch, tmp_path, capsys):
    capture, hops = run_case(LEAK_CASES[name], entry_point, monkeypatch, str(tmp_path), lambda: capsys.readouterr().out)
    quoted = urllib.parse.quote(CIVITAI_TOKEN)
    echoed = [s for s in _strings(capture) if CIVITAI_TOKEN in s or quoted in s]
    assert echoed == [], f"the stored Civitai token is echoed in: {echoed}"
    assert hops[0][1].endswith("token=" + quoted), "the request no longer carries the token"


@pytest.mark.parametrize("entry_point", ["job", "node"])
@pytest.mark.parametrize("name", list(UNCHANGED_CASES))
def test_unchanged(name, entry_point, bcnodes, monkeypatch, tmp_path, capsys):
    check_env(ENV)
    capture, hops = run_case(UNCHANGED_CASES[name], entry_point, monkeypatch, str(tmp_path), lambda: capsys.readouterr().out)
    check(UNCHANGED, f"{name}/{entry_point}", digest({**capture, "hops": hops}))

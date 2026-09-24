"""Golden: BC_AutoModelDownloader, its HTTP routes and the job runner (plan 8.2 row 19).

The routes, the websocket job runner and the node stay in nodes/downloader.py (section 3), so
the cases reach them through the module itself, bound a second time
(bcnodes_routes_under_test) with `server` and `aiohttp` stubbed first: the module then
registers its routes at import, as it does inside ComfyUI. Pinned:
  - the registered (method, path, handler) list, in order;
  - (status, JSON body) of every handler over one scripted session: /status, /tokens get and
    set, /check (incl. an error line and a non-dict entry, which raises: HTTP 500 in aiohttp,
    12 B-4), /dismiss (a key, none, empty; {"key": 5} on a store holding strings, and on an
    empty store followed by a string key: the C30 sequence), /start (nothing to download, a
    missing token, a started job, a second start while it runs);
  - the job runner: start_job + join of the bcnodes-downloader thread -> the event sequence,
    the snapshot, stdout and files, for a download and a failure with a stored Civitai token
    (so no event or error carries it, F1), and start_job while a job runs;
  - node download(): the ui text, stdout, the ProgressBar calls and files, and both
    ValueError texts (bad lines; missing tokens), a download failure, a non-list entries text;
    IS_CHANGED.

A job's download is held inside fake_http until the route has returned its snapshot, so the
snapshot does not depend on thread timing (12 B-5). JSON bodies are what aiohttp's
json_response sends (json.dumps at once). Tokens are placeholders; every request is answered
by _golden.fake_http; time.time (the job id) and time.monotonic are fixed; the handlers are
run by driving their coroutine (none of them awaits anything but request.json()); temp paths
are masked as <tmp>. Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE.
"""

import copy
import hashlib
import itertools
import json
import os
import re
import sys
import threading
import time
import types

import pytest

import _golden
from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'routes/registered': 'e6ed4ab5bfe9e5cf7370381bf8da0fb8',
    'routes/01_status_idle': '9b2c86f54afcdf1f770f4f64f1279e5c',
    'routes/02_tokens_get_none': '74b5f6bd84bd4a1fe018f98e9d1af81a',
    'routes/03_check_lines': '453839dff76f3000bc5e9a3e0c8ae48c',
    'routes/04_check_no_entries': '9bccad1b34714a21489f9703dce0955e',
    'routes/05_check_entries_none': '9bccad1b34714a21489f9703dce0955e',
    'routes/06_check_non_dict_entry': '1c0df101ce85cbf35ee02c63744fd996',
    'routes/07_tokens_set_hf': '7c22f02fef4ae29038acffa307efc6ed',
    'routes/08_start_nothing_missing': 'cbff5b7a6c747c34ef5e672036633226',
    'routes/09_start_missing_civitai_token': '30d7a510c722896f68d4170439bc4cdb',
    'routes/10_dismiss_key': 'ce0eb42e803cc6619165b797fc00d73f',
    'routes/11_dismiss_no_key': 'ce0eb42e803cc6619165b797fc00d73f',
    'routes/12_dismiss_empty_key': 'ce0eb42e803cc6619165b797fc00d73f',
    'routes/13_check_seen': '9c81b20227b6b44335008459edf5d06e',
    'routes/14_tokens_set_civitai': '49891c94b0f4693c4e6ee5b422ca96d5',
    'routes/15_start_job': '21f239d9375614d9025232db8689b918',
    'routes/16_start_while_running': '241e2766d1ebdfa589c6750f2c3bd4e6',
    'routes/17_status_running': 'b634236232b4336e295d4ac9cd93a8f0',
    'routes/18_status_done': '7499f26ecb7131320fc7b4e121fd0ddd',
    'routes/job': '11fc791454c87b800cd4984c514cda81',
    'routes/19_dismiss_int_key_on_string_store': '411a1ad77bc0b42ee75dd5d446c2ff00',
    'routes/20_check_after_failed_dismiss': '343d5639d646ab3fc2eb21018a5ca519',
    'routes_c30/01_dismiss_int_key': 'b053a17df12a0e499198b3b21bcd7b52',
    'routes_c30/02_check_after_int_key': 'b2e9cd4e8e4a194f7cf6ed06238f4ac4',
    'routes_c30/03_dismiss_string_key': '411a1ad77bc0b42ee75dd5d446c2ff00',
    'routes_c30/04_dismiss_string_key_again': 'c982aa987b8c1a634c68ff3adf113cb8',
    'job/download_and_failure': '11e7e8c94d9c45dc6419cf98e90398fb',
    'job/start_while_running': 'e6f4d71eb70bd590078c9b147d8ba6d5',
    'node/empty_text': 'f54a5e173a7d2a62688785cfec4ba9d2',
    'node/blank_text': 'f54a5e173a7d2a62688785cfec4ba9d2',
    'node/none': 'f54a5e173a7d2a62688785cfec4ba9d2',
    'node/empty_list': 'f54a5e173a7d2a62688785cfec4ba9d2',
    'node/all_present': '07ab2a5bb71c0eb17d65ecec2bb94d8a',
    'node/one_missing_hf_token_stored': '8a0a0645f7b8fde0323ad9db46aa3b3a',
    'node/bad_lines': '804a79c62d1138392915f94206be0a39',
    'node/missing_tokens': 'eece194dd11b0e7805353993bd1ac31e',
    'node/missing_civitai_token_only': 'f64a165bd54e25e9189bc22a6703132a',
    'node/download_fails_stored_civitai_token': '0bf84b7472c41a665002f2257028a07c',
    'node/not_a_list': 'ff9355174344518d6e7b587241a9411b',
    'node/IS_CHANGED': 'ac70d8c938881541923ab3bbe3f91de0',
}

WHERE = Where({
    "save_tokens": "libs.download:save_tokens",
})

BINDING = "bcnodes_routes_under_test"
MODULE = "nodes.downloader"  # routes, job runner and node stay here (section 3)

HF_TOKEN = "hf-placeholder-token"
CIVITAI_TOKEN = "civitai-placeholder/+=token"
HF_URL = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
CIVITAI_URL = "https://civitai.com/api/download/models/12345"
PRESENT_URL = "https://huggingface.co/org/repo/resolve/main/present.safetensors"
BINARY = {"Content-Type": "application/octet-stream"}
BIG = bytes(range(256)) * (10 * 1024)  # 2.5 MiB: three 1 MiB chunks

LINE_HF = {"url": HF_URL, "dir": "checkpoints", "hf": True}
LINE_CIVITAI = {"url": CIVITAI_URL, "dir": "loras/model_a.safetensors", "civitai": True}
LINE_PRESENT = {"url": PRESENT_URL, "dir": "vae"}
LINE_ERROR = {"url": "https://example.com/model.safetensors", "dir": "checkpoints"}
LINE_NO_NAME = {"url": CIVITAI_URL, "dir": "loras", "civitai": True}


class _Instance:
    """Stands in for PromptServer.instance: a route table and send_sync."""

    def __init__(self):
        self.registered = []
        self.events = []
        self.routes = types.SimpleNamespace(get=self._route("GET"), post=self._route("POST"))

    def _route(self, method):
        def route(path):
            def register(handler):
                self.registered.append((method, path, handler))
                return handler
            return register
        return route

    def send_sync(self, event, data):
        self.events.append((event, copy.deepcopy(data)))


def _json_response(data, status=200):
    return status, json.dumps(data)


class _ProgressBar:
    calls = []

    def __init__(self, total):
        self.calls.append(("ProgressBar", total))

    def update(self, n):
        self.calls.append(("update", n))


@pytest.fixture
def dl(bcnodes, second_binding, monkeypatch, tmp_path):
    """The downloader module under the stubs, bound as BINDING; per-test models/ and user/
    (with models/vae/present.safetensors), fixed clocks, a ProgressBar recorder."""
    instance = _Instance()
    server = types.ModuleType("server")
    server.PromptServer = type("PromptServer", (), {"instance": instance})
    web = types.ModuleType("aiohttp.web")
    web.json_response = _json_response
    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = web
    monkeypatch.setitem(sys.modules, "server", server)
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp)
    monkeypatch.setitem(sys.modules, "aiohttp.web", web)
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path / "user"), raising=True)
    (tmp_path / "models" / "vae").mkdir(parents=True)
    (tmp_path / "models" / "vae" / "present.safetensors").write_bytes(b"x" * 7)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    monkeypatch.setattr(time, "time", lambda: 1234567890.0, raising=True)
    _ProgressBar.calls = []
    monkeypatch.setattr(sys.modules["comfy.utils"], "ProgressBar", _ProgressBar, raising=True)
    module = second_binding(BINDING)[MODULE]
    return types.SimpleNamespace(module=module, instance=instance, tmp=str(tmp_path),
                                 fake_http=lambda script: _golden.fake_http(monkeypatch, script))


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


def _files(root):
    out = []
    for d, _, names in os.walk(root):
        for name in names:
            path = os.path.join(d, name)
            with open(path, "rb") as f:
                data = f.read()
            out.append((os.path.relpath(path, root).replace(os.sep, "/"),
                        data if name.endswith(".json") else (len(data), hashlib.md5(data).hexdigest())))
    return sorted(out)


def _join_job():
    for thread in threading.enumerate():
        if thread.name == "bcnodes-downloader":
            thread.join(10)
            assert not thread.is_alive(), "the job thread did not finish"


def _held(reply, release):
    """A fake_http reply that waits for `release` first (the job thread blocks there)."""
    def answer(req):
        release.wait(10)
        return reply
    return answer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def test_registered_routes(dl):
    check_env(ENV)
    check(GOLDEN, "routes/registered", digest([(m, p, h.__name__) for m, p, h in dl.instance.registered]))


class _Session:
    """Calls the routes by (method, path) and records each outcome under a numbered key."""

    def __init__(self, dl, prefix):
        self.dl, self.prefix, self.n = dl, prefix, 0
        self.handlers = {(m, p): h for m, p, h in dl.instance.registered}

    def call(self, method, path, body=None):
        class Request:
            async def json(self):
                return copy.deepcopy(body)

        coro = self.handlers[(method, path)](Request())
        try:
            coro.send(None)
        except StopIteration as stop:
            return {"response": stop.value}
        except Exception as e:
            return {"exception": _exception(e)}
        coro.close()
        raise AssertionError(f"{method} {path} awaited something besides request.json()")

    def step(self, label, method, path, body=None):
        self.n += 1
        out = self.call(method, path, body)
        seen = os.path.join(self.dl.tmp, "user", "BCNodes", "downloader_seen.json")
        with_seen = {**out, "seen_file": None}
        if os.path.isfile(seen):
            with open(seen, "rb") as f:
                with_seen["seen_file"] = f.read()
        check(GOLDEN, f"{self.prefix}/{self.n:02d}_{label}", digest(_mask(with_seen, self.dl.tmp)))
        return out


def test_route_session(dl, capsys):
    check_env(ENV)
    s = _Session(dl, "routes")
    lines = [LINE_HF, LINE_CIVITAI, LINE_PRESENT, LINE_ERROR]
    s.step("status_idle", "GET", "/bcnodes/downloader/status")
    s.step("tokens_get_none", "GET", "/bcnodes/downloader/tokens")
    checked = s.step("check_lines", "POST", "/bcnodes/downloader/check", {"entries": lines})
    s.step("check_no_entries", "POST", "/bcnodes/downloader/check", {})
    s.step("check_entries_none", "POST", "/bcnodes/downloader/check", {"entries": None})
    s.step("check_non_dict_entry", "POST", "/bcnodes/downloader/check", {"entries": [LINE_HF, "text"]})
    s.step("tokens_set_hf", "POST", "/bcnodes/downloader/tokens",
           {"huggingface": f" {HF_TOKEN} ", "civitai": "", "other": "x"})
    s.step("start_nothing_missing", "POST", "/bcnodes/downloader/start", {"entries": [LINE_PRESENT]})
    s.step("start_missing_civitai_token", "POST", "/bcnodes/downloader/start", {"entries": lines})
    key = json.loads(checked["response"][1])["key"]
    s.step("dismiss_key", "POST", "/bcnodes/downloader/dismiss", {"key": key})
    s.step("dismiss_no_key", "POST", "/bcnodes/downloader/dismiss", {})
    s.step("dismiss_empty_key", "POST", "/bcnodes/downloader/dismiss", {"key": ""})
    s.step("check_seen", "POST", "/bcnodes/downloader/check", {"entries": lines})
    s.step("tokens_set_civitai", "POST", "/bcnodes/downloader/tokens", {"civitai": CIVITAI_TOKEN})

    release = threading.Event()
    hops = dl.fake_http({
        "huggingface.co": _held((200, {**BINARY, "Content-Length": str(len(BIG))}, BIG), release),
        "civitai.com": (401, {}, b""),
    })
    s.step("start_job", "POST", "/bcnodes/downloader/start", {"entries": lines})
    s.step("start_while_running", "POST", "/bcnodes/downloader/start", {"entries": [LINE_HF]})
    s.step("status_running", "GET", "/bcnodes/downloader/status")
    release.set()
    _join_job()
    s.step("status_done", "GET", "/bcnodes/downloader/status")
    check(GOLDEN, "routes/job", digest(_mask({"events": dl.instance.events, "hops": hops,
                                               "stdout": capsys.readouterr().out,
                                               "files": _files(dl.tmp)}, dl.tmp)))
    s.step("dismiss_int_key_on_string_store", "POST", "/bcnodes/downloader/dismiss", {"key": 5})
    s.step("check_after_failed_dismiss", "POST", "/bcnodes/downloader/check", {"entries": lines})


def test_dismiss_int_key_on_empty_store(dl):
    """C30: /dismiss {"key": 5} writes [5]; the next string key makes sorted() raise after the
    file was truncated; the empty file then reads as an empty store."""
    check_env(ENV)
    s = _Session(dl, "routes_c30")
    s.step("dismiss_int_key", "POST", "/bcnodes/downloader/dismiss", {"key": 5})
    s.step("check_after_int_key", "POST", "/bcnodes/downloader/check", {"entries": [LINE_PRESENT]})
    s.step("dismiss_string_key", "POST", "/bcnodes/downloader/dismiss", {"key": "abc"})
    s.step("dismiss_string_key_again", "POST", "/bcnodes/downloader/dismiss", {"key": "abc"})


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------

ITEMS = {  # what describe() makes of LINE_HF, LINE_CIVITAI, LINE_PRESENT: (url, dir, filename)
    "hf": (HF_URL, "checkpoints", "model.safetensors"),
    "civitai": (CIVITAI_URL, "loras", "model_a.safetensors"),
    "present": (PRESENT_URL, "vae", "present.safetensors"),
}


def _item(dl, name):
    url, rel_dir, filename = ITEMS[name]
    return {"url": url, "dir": rel_dir, "filename": filename, "path": os.path.join(dl.tmp, "models", rel_dir, filename)}


def test_job_download_and_failure(dl, capsys):
    check_env(ENV)
    WHERE["save_tokens"]({"civitai": CIVITAI_TOKEN})
    hops = dl.fake_http({"huggingface.co": (200, {**BINARY, "Content-Length": str(len(BIG))}, BIG),
                         "civitai.com": (401, {}, b"")})
    snapshots = [dl.module._Job.snapshot()]
    job, started = dl.module.start_job([_item(dl, "hf"), _item(dl, "civitai")])
    _join_job()
    snapshots.append(dl.module._Job.snapshot())
    out = {"started": started, "job": job, "snapshots": snapshots, "events": dl.instance.events, "hops": hops,
           "stdout": capsys.readouterr().out, "files": _files(dl.tmp)}
    check(GOLDEN, "job/download_and_failure", digest(_mask(out, dl.tmp)))


def test_job_start_while_running(dl, capsys):
    check_env(ENV)
    release = threading.Event()
    hops = dl.fake_http({"huggingface.co": _held((200, {**BINARY, "Content-Length": "10"}, b"0123456789"), release)})
    first, started_first = dl.module.start_job([_item(dl, "hf")])
    second, started_second = dl.module.start_job([_item(dl, "present")])
    during = dl.module._Job.snapshot()
    release.set()
    _join_job()
    out = {"started": (started_first, started_second), "same_job": second is first, "during": during,
           "after": dl.module._Job.snapshot(), "events": dl.instance.events, "hops": hops,
           "stdout": capsys.readouterr().out, "files": _files(dl.tmp)}
    check(GOLDEN, "job/start_while_running", digest(_mask(out, dl.tmp)))


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

NODE_CASES = {  # name -> (entries text, stored tokens, fake_http script)
    "empty_text": ("", {}, {}),
    "blank_text": ("  ", {}, {}),
    "none": (None, {}, {}),
    "empty_list": ("[]", {}, {}),
    "all_present": (json.dumps([LINE_PRESENT]), {}, {}),
    "one_missing_hf_token_stored": (json.dumps([LINE_HF, LINE_PRESENT]), {"huggingface": HF_TOKEN},
                                    {"huggingface.co": (200, {**BINARY, "Content-Length": str(len(BIG))}, BIG)}),
    "bad_lines": (json.dumps([LINE_HF, LINE_ERROR, LINE_NO_NAME]), {}, {}),
    "missing_tokens": (json.dumps([LINE_HF, LINE_CIVITAI, LINE_PRESENT]), {}, {}),
    "missing_civitai_token_only": (json.dumps([LINE_HF, LINE_CIVITAI]), {"huggingface": HF_TOKEN}, {}),
    "download_fails_stored_civitai_token": (json.dumps([LINE_CIVITAI]), {"civitai": CIVITAI_TOKEN},
                                            {"civitai.com": (401, {}, b"")}),
    "not_a_list": (json.dumps(LINE_HF), {}, {}),
}


@pytest.mark.parametrize("name", list(NODE_CASES))
def test_node_download(name, dl, capsys):
    check_env(ENV)
    text, tokens, script = NODE_CASES[name]
    WHERE["save_tokens"](tokens)
    hops = dl.fake_http(script)
    try:
        out = {"result": dl.module.AutoModelDownloader().download(text)}
    except Exception as e:
        out = {"exception": (type(e).__name__, str(e))}
    out.update(progress_bar=_ProgressBar.calls, hops=hops, stdout=capsys.readouterr().out,
               events=dl.instance.events, files=_files(dl.tmp))
    check(GOLDEN, f"node/{name}", digest(_mask(out, dl.tmp)))


def test_is_changed(dl):
    check_env(ENV)
    check(GOLDEN, "node/IS_CHANGED", digest(repr(dl.module.AutoModelDownloader.IS_CHANGED("[]"))))

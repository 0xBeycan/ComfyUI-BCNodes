"""Golden-test helpers: digest, at + WHERE, ENV, record mode, fake_http.

A golden file pins today's outputs so a move can prove it changed none. It declares:

    ENV = {...}              # the versions its digests depend on: check_env(ENV, "torch", ...)
    GOLDEN = {case: digest}  # recorded values; check(GOLDEN, case, value) asserts one
    WHERE = Where({...})     # its pack seams and direct helper calls; the cases use only WHERE[...]

A move step edits WHERE entries only; ENV, GOLDEN, inputs and assertions stay as recorded.

Record mode: with BCNODES_GOLDEN_RECORD=1, check() and check_env() assert nothing; they collect
the values and the run ends by printing each file's tables (`ENV = {...}`, `GOLDEN = {...}`,
or whatever name the file gave the table) to paste into the file.

Imports only the standard library at module level; torch, numpy and PIL are used when a value
of theirs is digested, so a torch-free test stays torch-free.
"""

import dataclasses
import hashlib
import http.client
import importlib
import io
import os
import pathlib
import platform
import re
import sys
import urllib.parse
import urllib.request

import _harness

RECORD = os.environ.get("BCNODES_GOLDEN_RECORD") == "1"


# ---------------------------------------------------------------------------
# digest
# ---------------------------------------------------------------------------

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _md5(data):
    return hashlib.md5(data).hexdigest()


class _Tag:
    """Stands for a tensor, array, image, file or set inside a digested container."""

    def __init__(self, kind, value):
        self.kind, self.value = kind, value

    def __repr__(self):
        return f"<{self.kind} {self.value}>"


def _leaf(x):
    """(kind, md5) for a tensor, ndarray, PIL image or file path; None for anything else."""
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(x, torch.Tensor):
        t = x.detach().cpu().contiguous()
        h = hashlib.md5(f"{tuple(t.shape)}|{t.dtype}|{x.requires_grad}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        return "tensor", h.hexdigest()
    np = sys.modules.get("numpy")
    if np is not None and isinstance(x, np.ndarray):
        if x.dtype == object:
            raise ValueError("digest of an object array: its bytes are pointers")
        return "ndarray", _md5(f"{x.dtype}|{x.shape}|".encode() + x.tobytes())
    pil = sys.modules.get("PIL.Image")
    if pil is not None and isinstance(x, pil.Image):
        return "image", _md5(f"{x.mode}|{x.size}|".encode() + x.tobytes())
    if isinstance(x, pathlib.PurePath):
        return "file", _file(x)
    return None


def _file(path):
    """Raw bytes md5 | decoded-pixel md5 ("-" when PIL cannot decode it), so a codec change is
    told apart from a pixel change."""
    with open(path, "rb") as f:
        raw = _md5(f.read())
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as im:
            im.load()
            pixels = _leaf(im)[1]
    except UnidentifiedImageError:
        pixels = "-"
    return f"{raw}|{pixels}"


def _canon(x):
    """x with every dataclass as its asdict(), and every tensor, array, image, file and set
    inside a dict, list or tuple as a tag holding its digest.

    A subclass of dict, list or tuple (OrderedDict, defaultdict, a namedtuple, torch.Size,
    torch.return_types) is walked the same way and tagged with its class name: its own repr
    would print a large tensor or array truncated with '...', so different values would digest
    the same. The name only, not the module, so a class that moves keeps its digest."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _canon(dataclasses.asdict(x))
    kind = type(x)
    if isinstance(x, dict):
        out = {k: _canon(v) for k, v in x.items()}
    elif isinstance(x, list):
        out = [_canon(v) for v in x]
    elif isinstance(x, tuple):
        out = tuple(_canon(v) for v in x)
    elif isinstance(x, (set, frozenset)):  # repr order of a set of str changes with the hash seed
        return _Tag(kind.__name__, sorted(repr(_canon(v)) for v in x))
    else:
        leaf = _leaf(x)
        return x if leaf is None else _Tag(*leaf)
    return out if kind in (dict, list, tuple) else _Tag(kind.__name__, out)


def digest(x):
    """md5 hex fingerprint of x.

    - torch tensor: md5 of f"{shape}|{dtype}|{requires_grad}|" + its raw bytes (any dtype incl.
      bf16/fp16; requires_grad makes a lost torch.no_grad() visible);
    - numpy array: dtype, shape, bytes; PIL image: mode, size, tobytes();
    - pathlib path: "<md5 of the file bytes>|<md5 of the decoded pixels>";
    - dataclass instance: digested exactly as dataclasses.asdict(x), so a check dict and the
      dataclass that replaces it digest the same when keys, order, values and types match;
    - anything else: md5 of its repr (exact floats, -0.0, np.float64 vs float, dict insertion
      order all visible), with the kinds above inside dicts, lists and tuples digested first
      (their subclasses too, tagged with the class name: OrderedDict, namedtuple, ...).

    A repr carrying a memory address raises: it could never match again."""
    leaf = _leaf(x)
    if leaf is not None:
        return leaf[1]
    text = repr(_canon(x))
    if _ADDRESS.search(text):
        raise ValueError(f"digest of a value whose repr has a memory address: {text[:300]}")
    return _md5(text.encode())


# ---------------------------------------------------------------------------
# at + WHERE
# ---------------------------------------------------------------------------

def at(path):
    """The object at "module.path:attr.attr" in the bound package (bcnodes_under_test, the
    `bcnodes` fixture), looked up now. "module.path" alone gives the module."""
    module, _, attrs = path.partition(":")
    if _harness.PKG_NAME not in sys.modules:
        raise RuntimeError(f"{_harness.PKG_NAME} is not bound: use the `bcnodes` fixture (tests/conftest.py)")
    obj = importlib.import_module(f"{_harness.PKG_NAME}.{module}")
    for name in filter(None, attrs.split(".")):
        obj = getattr(obj, name)
    return obj


class Where(dict):
    """A golden file's WHERE table. An entry is a location string "module.path:attr[.attr]",
    resolved by at() every time it is read (so a stale location fails where it is used), or an
    adapter callable built on at(), returned as it is."""

    def __getitem__(self, key):
        entry = super().__getitem__(key)
        return at(entry) if isinstance(entry, str) else entry

    def patch(self, monkeypatch, key, value):
        """Replaces the attribute at a location entry for this test only; raising=True, so a
        location that no longer exists fails instead of patching nothing."""
        entry = super().__getitem__(key)
        if not isinstance(entry, str) or ":" not in entry:
            raise TypeError(f"WHERE[{key!r}] is not a 'module:attr' location: {entry!r}")
        module, _, attrs = entry.partition(":")
        head, _, name = attrs.rpartition(".")
        monkeypatch.setattr(at(f"{module}:{head}"), name, value, raising=True)


# ---------------------------------------------------------------------------
# GOLDEN / ENV and record mode
# ---------------------------------------------------------------------------

_RECORDED = {}  # file -> {table name: {case: value}, "ENV": {...}}


def _table_name(table, frame):
    return next((k for k, v in frame.f_globals.items() if v is table), "GOLDEN")


def check(table, case, value):
    """Asserts value == table[case]. Record mode: stores the value instead."""
    frame = sys._getframe(1)
    if RECORD:
        tables = _RECORDED.setdefault(frame.f_globals.get("__file__", "?"), {})
        tables.setdefault(_table_name(table, frame), {})[case] = value
        return
    if case not in table:
        raise AssertionError(f"case {case!r} has no recorded value in {_table_name(table, frame)} (BCNODES_GOLDEN_RECORD=1 prints it)")
    if value != table[case]:
        raise AssertionError(f"{case!r}: got {value!r}, recorded {table[case]!r}")


_MODULE_OF = {
    "torch": "torch", "numpy": "numpy", "Pillow": "PIL", "cv2": "cv2", "scipy": "scipy",
    "torchvision": "torchvision", "av": "av", "postfx": "postfx", "caption_audit": "caption_audit",
}
_SOURCE_MD5 = ("postfx", "caption_audit")


def source_md5(module):
    """md5 over a package's *.py files (sorted relative path + bytes), found next to
    module.__file__, so it works for a source tree on PYTHONPATH and for a pip install alike."""
    root = os.path.dirname(module.__file__)
    files = sorted(
        os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
        for d, _, names in os.walk(root) for f in names if f.endswith(".py")
    )
    h = hashlib.md5()
    for rel in files:
        with open(os.path.join(root, rel), "rb") as f:
            h.update(rel.encode() + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def env(*deps):
    """Platform, Python and the version of each named dependency (keys of _MODULE_OF); for
    postfx and caption_audit also source_md5, since other sessions commit to their source
    trees without a version bump."""
    out = {"platform": f"{sys.platform}-{platform.machine()}", "python": platform.python_version()}
    for dep in deps:
        try:
            module = importlib.import_module(_MODULE_OF[dep])
        except ImportError as e:
            raise AssertionError(
                f"{dep} is not importable ({e}); the goldens need it: pip install -r requirements.txt, "
                "or PYTHONPATH=/path/to/postfx:/path/to/caption-audit for local checkouts") from e
        out[dep] = module.__version__
        if dep in _SOURCE_MD5:
            if getattr(module, "__file__", None) is None:
                raise AssertionError(f"sys.modules[{dep!r}] is a stub; ENV needs the real package")
            out[f"{dep}_py_md5"] = source_md5(module)
    return out


def check_env(recorded, *deps):
    """Fails, naming each differing entry, when env(*deps) is not the recorded ENV: the digests
    were recorded against those versions. Never skips. Record mode: stores env(*deps) instead."""
    current = env(*deps)
    frame = sys._getframe(1)
    if RECORD:
        _RECORDED.setdefault(frame.f_globals.get("__file__", "?"), {})[_table_name(recorded, frame)] = current
        return
    if current != recorded:
        diff = [f"{k}: recorded {recorded.get(k)!r}, now {current.get(k)!r}"
                for k in dict.fromkeys([*recorded, *current]) if recorded.get(k) != current.get(k)]
        raise AssertionError("ENV differs from the one the goldens were recorded with:\n  " + "\n  ".join(diff))


def recorded_tables():
    """The tables collected in record mode, as text to paste; empty when nothing was recorded."""
    lines = []
    for file, tables in _RECORDED.items():
        lines.append(f"# {os.path.relpath(file, _harness.PKG_DIR)}")
        for name, table in tables.items():
            lines.append(f"{name} = {{")
            lines += [f"    {k!r}: {v!r}," for k, v in table.items()]
            lines.append("}")
    return lines


# ---------------------------------------------------------------------------
# fake_http: the one network seam
# ---------------------------------------------------------------------------

class UnscriptedRequest(BaseException):
    """A request fake_http has no reply for. A BaseException, so no `except Exception` in the
    code under test turns it into an ordinary error message (or lets a real download start)."""


class _Wire:
    """The socket http.client reads a scripted reply from."""

    def __init__(self, data):
        self.data = data

    def makefile(self, mode):
        return io.BytesIO(self.data)


def _response(req, status, headers, body):
    """A real http.client.HTTPResponse parsed from the scripted reply, finished the way
    AbstractHTTPHandler.do_open finishes it."""
    items = headers.items() if isinstance(headers, dict) else headers
    head = [f"HTTP/1.1 {status} {http.client.responses.get(status, '')}", *(f"{k}: {v}" for k, v in items)]
    resp = http.client.HTTPResponse(_Wire(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body),
                                    method=req.get_method(), url=req.full_url)
    resp.begin()
    resp.url = req.get_full_url()
    resp.msg = resp.reason
    return resp


def fake_http(monkeypatch, script):
    """Answers every urllib request from `script` instead of the network; returns the hop log.

    Patches urllib.request.HTTPSHandler.https_open and HTTPHandler.http_open (raising=True), so
    urllib's real opener, header processing, error processor and redirect handler all run, for
    urlopen and for an opener built per call alike. `script` maps a hostname to its reply:
      - (status, headers, body): headers a dict or a list of pairs, body bytes; sent as written
        (no Content-Length is added), redirects via a "Location" header;
      - an exception instance (e.g. urllib.error.URLError("...")): raised as the connection error;
      - a list of the above: one per request to that host, in order;
      - a callable(request) returning one of the above.
    Every request is logged first as (hostname, full_url, dict(req.header_items()), req.timeout).
    A host with no reply (or none left) raises UnscriptedRequest."""
    hops = []
    replies = {host: list(r) if isinstance(r, list) else r for host, r in script.items()}

    def open_(handler, req):
        url = req.full_url
        host = urllib.parse.urlparse(url).hostname
        hops.append((host, url, dict(req.header_items()), req.timeout))
        if host not in replies or replies[host] == []:
            raise UnscriptedRequest(f"no scripted reply for {host}: {url}")
        reply = replies[host]
        if isinstance(reply, list):
            reply = reply.pop(0)
        elif callable(reply):
            reply = reply(req)
        if isinstance(reply, BaseException):
            raise reply
        return _response(req, *reply)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", open_, raising=True)
    monkeypatch.setattr(urllib.request.HTTPHandler, "http_open", open_, raising=True)
    return hops

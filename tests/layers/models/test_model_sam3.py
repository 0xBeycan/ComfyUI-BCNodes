"""Golden: where the SAM 3 checkpoint comes from, called directly.

path(name): a checkpoint found under models/checkpoints (default and other name, no request),
a missing non-default name (FileNotFoundError text, no request), and the missing default,
downloaded into the first checkpoints folder: the returned path, stdout (the "downloading"
line and the decile progress lines), every request hop (host, URL, headers, timeout), the file
bytes, and no .part left; a 404 on the download raises with its text. choices(): the default
is offered first when it is not on disk, and not twice when it is.

Seams: folder_paths' checkpoints folders point into tmp dirs (and its user dir, so no token is
stored); every request is answered by _golden.fake_http; time.monotonic counts 1, 2, 3, ... so
every chunk reports progress.
"""

import hashlib
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

GOLDEN = {
    'found_default': "(('returned', '/path/to/models/checkpoints_b/sam3.1_multiplex_fp16.safetensors'), '')",
    'found_other_name': "(('returned', '/path/to/models/checkpoints_a/sub/sam3_other.safetensors'), '')",
    'missing_other_name': "(('raised', 'FileNotFoundError', 'SAM 3 checkpoint sam3_missing.safetensors is not in models/checkpoints'), '')",
    'download/cdn_302': '7cef6426fc9398215abaa4f2b4d2cdde',
    'download/direct_200': '04eaf4e55a87bbe98ca48fb4243518f4',
    'download/not_found_404': 'da61ce81de056a5c5d5bacb6a9f90955',
    'choices/empty': ['sam3.1_multiplex_fp16.safetensors'],
    'choices/others': ['sam3.1_multiplex_fp16.safetensors', 'b.safetensors', 'a.safetensors'],
    'choices/default_listed': ['a.safetensors', 'sam3.1_multiplex_fp16.safetensors', 'z.safetensors'],
}

WHERE = Where({
    "path": "models.sam3.checkpoint:path",
    "choices": "models.sam3.checkpoint:choices",
})

DEFAULT = "sam3.1_multiplex_fp16.safetensors"
URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"
CDN_URL = "https://us.aws.cdn.hf.co/xet-bridge-us/placeholder/sam3?X-Amz-Signature=placeholder"
BODY = bytes(range(256)) * (7 * 1024 * 2)  # 3.5 MiB: four 1 MiB chunks, the last one half
OK = (200, {"Content-Type": "application/octet-stream", "Content-Length": str(len(BODY))}, BODY)


@pytest.fixture
def checkpoints(monkeypatch, tmp_path):
    """Two checkpoints folders (the first is where a download lands), an empty user dir;
    returns (first, second, masker) where masker replaces the tmp root in a text."""
    first, second = tmp_path / "checkpoints_a", tmp_path / "checkpoints_b"
    first.mkdir()
    second.mkdir()
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "folder_names_and_paths", {"checkpoints": ([str(first), str(second)], set())}, raising=True)
    monkeypatch.setattr(fp, "get_folder_paths", lambda name: list(fp.folder_names_and_paths[name][0]), raising=False)
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path / "user"), raising=True)
    ticks = itertools.count(1)
    monkeypatch.setattr(time, "monotonic", lambda: float(next(ticks)), raising=True)
    return str(first), str(second), lambda text: text.replace(str(tmp_path), "/path/to/models")


def _run(name, capsys):
    try:
        out = ("returned", WHERE["path"](name))
    except Exception as e:
        out = ("raised", type(e).__name__, str(e))
    return out, capsys.readouterr().out


def test_found_default(bcnodes, checkpoints, monkeypatch, capsys):
    first, second, mask = checkpoints
    open(os.path.join(second, DEFAULT), "wb").close()
    hops = _golden.fake_http(monkeypatch, {})
    out, stdout = _run(DEFAULT, capsys)
    assert hops == []
    check(GOLDEN, "found_default", mask(repr((out, stdout))))


def test_found_other_name(bcnodes, checkpoints, monkeypatch, capsys):
    first, second, mask = checkpoints
    os.makedirs(os.path.join(first, "sub"))
    open(os.path.join(first, "sub", "sam3_other.safetensors"), "wb").close()
    hops = _golden.fake_http(monkeypatch, {})
    out, stdout = _run("sub/sam3_other.safetensors", capsys)
    assert hops == []
    check(GOLDEN, "found_other_name", mask(repr((out, stdout))))


def test_missing_other_name(bcnodes, checkpoints, monkeypatch, capsys):
    _, _, mask = checkpoints
    hops = _golden.fake_http(monkeypatch, {})
    out, stdout = _run("sam3_missing.safetensors", capsys)
    assert hops == []
    check(GOLDEN, "missing_other_name", mask(repr((out, stdout))))


DOWNLOADS = {
    "cdn_302": {"huggingface.co": (302, {"Location": CDN_URL}, b""), "us.aws.cdn.hf.co": OK},
    "direct_200": {"huggingface.co": OK},
    "not_found_404": {"huggingface.co": (404, {}, b"")},
}


@pytest.mark.parametrize("name", list(DOWNLOADS))
def test_default_download(name, bcnodes, checkpoints, monkeypatch, capsys):
    check_env(ENV)
    first, second, mask = checkpoints
    hops = _golden.fake_http(monkeypatch, DOWNLOADS[name])
    out, stdout = _run(DEFAULT, capsys)
    assert hops[0][1] == URL
    files = {}
    for folder in (first, second):
        for f in sorted(os.listdir(folder)):
            with open(os.path.join(folder, f), "rb") as fh:
                files[mask(os.path.join(folder, f))] = hashlib.md5(fh.read()).hexdigest()
    check(GOLDEN, f"download/{name}", digest({"out": mask(repr(out)), "stdout": mask(stdout), "hops": hops, "files": files}))


CHOICES = {
    "empty": [],
    "others": ["b.safetensors", "a.safetensors"],
    "default_listed": ["a.safetensors", DEFAULT, "z.safetensors"],
}


@pytest.mark.parametrize("name", list(CHOICES))
def test_choices(name, bcnodes, monkeypatch):
    listing = CHOICES[name]
    monkeypatch.setattr(sys.modules["folder_paths"], "get_filename_list",
                        lambda folder: list(listing) if folder == "checkpoints" else ["wrong_folder"], raising=True)
    check(GOLDEN, f"choices/{name}", WHERE["choices"]())

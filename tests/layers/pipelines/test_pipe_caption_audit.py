"""Golden: Caption Audit plumbing (nodes/caption_audit.py, target pipelines/caption_audit/audit.py).

- split_terms over separators, blanks and non-strings.
- build_args: the AuditArgs repr for the defaults and for every field set (incl. string and int
  widget values that it converts, an images_dir inside the roots), and the error for an
  images_dir outside them.
- allowed_roots / resolve_dir: the roots from folder_paths (its attributes set to placeholder
  paths under /path/to/comfyui, with and without base_path / get_input_directory, and with
  folder_paths absent -> the cwd) plus BC_CAPTION_ROOTS (several entries, empty ones, "~" with
  HOME set to a placeholder, duplicates, a relative entry); resolve_dir for a path inside, one
  outside (the message), '' (the cwd), "~", ".." collapse, symlinks out of and into a root.
  The test's tmp dir is replaced by /path/to/tmp in every recorded string.
- dir_fingerprint on a tmp tree with fixed mtimes (os.utime, ns), given as a path relative to
  the cwd so the hashed paths do not depend on the tmp dir: recursive False / True, dot files,
  a dot directory, upper-case .TXT, non-.txt files, a dangling .txt symlink (the error branch),
  a missing directory, None.

BC_CAPTION_ROOTS is set with monkeypatch.setenv.
"""

import os
import sys

import pytest

from _golden import Where, check, check_env

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'split_terms/empty': '[]',
    'split_terms/none': '[]',
    'split_terms/zero': '[]',
    'split_terms/single': "['single']",
    'split_terms/two': "['red scarf', 'studio lighting']",
    'split_terms/blanks': "['a', 'b']",
    'split_terms/only_commas': '[]',
    'split_terms/number': "['12.5']",
    'split_terms/spaces_inside': "['term a', 'term   b']",
    'build_args/defaults': "AuditArgs(dir='/path/to/datasets/dataset', images_dir=None, recursive=False, trigger=None, class_word=[], fuse=[], critical_threshold=0.85, warn_threshold=0.6, info_threshold=0.35, ngram_max=3, jaccard=0.9, no_stopwords=False, list_files=10, max_rows=200, max_interpret=12, no_color=True, format='term', out=None, compare=None, positional_dir=None)",
    'build_args/every_field': "AuditArgs(dir='/path/to/datasets/dataset', images_dir='/path/to/datasets/images', recursive=True, trigger='<trigger>', class_word=['person', 'term_a'], fuse=['red scarf', 'term_b'], critical_threshold=0.9, warn_threshold=1.0, info_threshold=0.25, ngram_max=2, jaccard=0.75, no_stopwords=True, list_files=10, max_rows=50, max_interpret=12, no_color=True, format='term', out=None, compare=None, positional_dir=None)",
    'build_args/blank_strings': "AuditArgs(dir='/path/to/datasets/dataset', images_dir=None, recursive=False, trigger=None, class_word=[], fuse=[], critical_threshold=0.85, warn_threshold=0.6, info_threshold=0.35, ngram_max=3, jaccard=0.9, no_stopwords=False, list_files=10, max_rows=200, max_interpret=12, no_color=True, format='term', out=None, compare=None, positional_dir=None)",
    'build_args/images_dir_outside': "/path/elsewhere/images is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to/datasets). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (':'-separated) and restart ComfyUI.",
    'allowed_roots/folder_paths_only': "['/path/to/comfyui', '/path/to/comfyui/output', '/path/to/comfyui/user']",
    'allowed_roots/base_path_and_input': "['/path/to/comfyui-base', '/path/to/comfyui/input', '/path/to/comfyui/output', '/path/to/comfyui/user']",
    'allowed_roots/env_entries': "['/path/to/comfyui', '/path/to/comfyui/output', '/path/to/comfyui/user', '/path/to/datasets', '/path/to/home/datasets', '/path/to/tmp/cwd/relative/dir', '/path/to/datasets_b']",
    'allowed_roots/folder_paths_absent': "['/path/to/tmp/cwd', '/path/to/datasets']",
    'resolve_dir/inside': "('ok', '/path/to/datasets/dataset')",
    'resolve_dir/inside_trailing_slash': "('ok', '/path/to/datasets/dataset')",
    'resolve_dir/root_itself': "('ok', '/path/to/datasets')",
    'resolve_dir/outside': '(\'ValueError\', "/path/elsewhere/dataset is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to/datasets, /path/to/home/datasets, /path/to/tmp/root, /path/to/tmp/cwd). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (\':\'-separated) and restart ComfyUI.")',
    'resolve_dir/prefix_is_not_inside': '(\'ValueError\', "/path/to/datasets_other is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to/datasets, /path/to/home/datasets, /path/to/tmp/root, /path/to/tmp/cwd). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (\':\'-separated) and restart ComfyUI.")',
    'resolve_dir/dotdot_escapes': '(\'ValueError\', "/path/to/elsewhere is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to/datasets, /path/to/home/datasets, /path/to/tmp/root, /path/to/tmp/cwd). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (\':\'-separated) and restart ComfyUI.")',
    'resolve_dir/dotdot_stays': "('ok', '/path/to/datasets/dataset')",
    'resolve_dir/home': "('ok', '/path/to/home/datasets/dataset')",
    'resolve_dir/empty_is_cwd': "('ok', '/path/to/tmp/cwd')",
    'resolve_dir/none_is_cwd': "('ok', '/path/to/tmp/cwd')",
    'resolve_dir/blank_is_cwd': "('ok', '/path/to/tmp/cwd')",
    'resolve_dir/relative': "('ok', '/path/to/tmp/cwd/sub/dir')",
    'resolve_dir/symlink_out': '(\'ValueError\', "/path/to/tmp/outside is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to/datasets, /path/to/home/datasets, /path/to/tmp/root, /path/to/tmp/cwd). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (\':\'-separated) and restart ComfyUI.")',
    'resolve_dir/symlink_in': "('ok', '/path/to/tmp/root/data')",
    'dir_fingerprint/recursive_False': '7319251117dc40c6957025561664ffbf418549996a289ea339b7e53c632d4f03',
    'dir_fingerprint/recursive_True': 'e0032b70580c95d644343367be267afa2b5cf75e20b4648327861c3181caa958',
    'dir_fingerprint/dangling_link/recursive_False': "dataset|error:[Errno 2] No such file or directory: 'dataset/broken.txt'",
    'dir_fingerprint/dangling_link/recursive_True': "dataset|error:[Errno 2] No such file or directory: 'dataset/broken.txt'",
    "dir_fingerprint/not_a_directory/'missing'": "'missing'",
    "dir_fingerprint/not_a_directory/''": "''",
    'dir_fingerprint/not_a_directory/None': "''",
    "dir_fingerprint/not_a_directory/'a_file.txt'": "'a_file.txt'",
}

WHERE = Where({
    "split_terms": "pipelines.caption_audit.audit:split_terms",
    "build_args": "pipelines.caption_audit.audit:build_args",
    "allowed_roots": "pipelines.caption_audit.audit:allowed_roots",
    "resolve_dir": "pipelines.caption_audit.audit:resolve_dir",
    "dir_fingerprint": "pipelines.caption_audit.audit:dir_fingerprint",
})

ROOTS_ENV = "BC_CAPTION_ROOTS"
MTIME_NS = 1_700_000_000_123_456_789


@pytest.fixture
def placeholders(monkeypatch, tmp_path):
    """folder_paths with placeholder directories, BC_CAPTION_ROOTS unset, HOME a placeholder, the
    cwd an empty tmp dir; returns scrub(text): the tmp dir -> /path/to/tmp."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", "/path/to/comfyui/models", raising=True)
    monkeypatch.setattr(fp, "get_output_directory", lambda: "/path/to/comfyui/output", raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: "/path/to/comfyui/user", raising=True)
    monkeypatch.delenv(ROOTS_ENV, raising=False)
    monkeypatch.setenv("HOME", "/path/to/home")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    real = os.path.realpath(tmp_path)
    return lambda text: str(text).replace(real, "/path/to/tmp").replace(str(tmp_path), "/path/to/tmp")


SPLIT_CASES = {
    "empty": "", "none": None, "zero": 0, "single": "single", "two": "red scarf, studio lighting",
    "blanks": " a ,, b , ", "only_commas": ",,", "number": 12.5, "spaces_inside": "  term a  ,term   b",
}


@pytest.mark.parametrize("name", list(SPLIT_CASES))
def test_split_terms(name, bcnodes):
    check_env(ENV)
    check(GOLDEN, f"split_terms/{name}", repr(WHERE["split_terms"](SPLIT_CASES[name])))


BUILD_CASES = {
    "defaults": dict(),
    "every_field": dict(trigger=" <trigger> ", images_dir=" /path/to/datasets/images ", class_words="person, term_a",
                        fuse="red scarf, ,term_b", critical_threshold="0.9", warn_threshold=1, info_threshold=0.25,
                        ngram_max="2", jaccard=0.75, no_stopwords=1, recursive="yes", max_rows=50.7),
    "blank_strings": dict(trigger="   ", images_dir="   ", class_words=" , ", fuse=None),
}


@pytest.mark.parametrize("name", list(BUILD_CASES))
def test_build_args(name, bcnodes, placeholders, monkeypatch):
    check_env(ENV)
    monkeypatch.setenv(ROOTS_ENV, "/path/to/datasets")
    args = WHERE["build_args"]("/path/to/datasets/dataset", **BUILD_CASES[name])
    check(GOLDEN, f"build_args/{name}", placeholders(repr(args)))


def test_build_args_images_dir_outside(bcnodes, placeholders, monkeypatch):
    check_env(ENV)
    monkeypatch.setenv(ROOTS_ENV, "/path/to/datasets")
    with pytest.raises(ValueError) as exc:
        WHERE["build_args"]("/path/to/datasets/dataset", images_dir="/path/elsewhere/images")
    check(GOLDEN, "build_args/images_dir_outside", placeholders(exc.value))


# name -> (extra folder_paths attributes, BC_CAPTION_ROOTS or None, folder_paths importable)
ROOTS_CASES = {
    "folder_paths_only": ({}, None, True),
    "base_path_and_input": ({"base_path": "/path/to/comfyui-base", "get_input_directory": lambda: "/path/to/comfyui/input"}, None, True),
    "env_entries": ({}, os.pathsep.join(["/path/to/datasets", "", "  ", "~/datasets", "/path/to/datasets/", "relative/dir",
                                         "/path/to/comfyui/output", "/path/to/other/../datasets_b"]), True),
    "folder_paths_absent": ({}, "/path/to/datasets", False),
}


def _roots_setup(name, monkeypatch):
    extra, env_roots, importable = ROOTS_CASES[name]
    fp = sys.modules["folder_paths"]
    for attr, value in extra.items():
        monkeypatch.setattr(fp, attr, value, raising=False)
    if env_roots is not None:
        monkeypatch.setenv(ROOTS_ENV, env_roots)
    if not importable:
        monkeypatch.setitem(sys.modules, "folder_paths", None)


@pytest.mark.parametrize("name", list(ROOTS_CASES))
def test_allowed_roots(name, bcnodes, placeholders, monkeypatch):
    check_env(ENV)
    _roots_setup(name, monkeypatch)
    check(GOLDEN, f"allowed_roots/{name}", placeholders(repr(WHERE["allowed_roots"]())))


RESOLVE_CASES = {
    "inside": "/path/to/datasets/dataset",
    "inside_trailing_slash": "/path/to/datasets/dataset/",
    "root_itself": "/path/to/datasets",
    "outside": "/path/elsewhere/dataset",
    "prefix_is_not_inside": "/path/to/datasets_other",
    "dotdot_escapes": "/path/to/datasets/../elsewhere",
    "dotdot_stays": "/path/to/datasets/x/../dataset",
    "home": "~/datasets/dataset",
    "empty_is_cwd": "",
    "none_is_cwd": None,
    "blank_is_cwd": "   ",
    "relative": "sub/dir",
    "symlink_out": "<tmp>/root/link_out",
    "symlink_in": "<tmp>/outside/link_in",
}


@pytest.mark.parametrize("name", list(RESOLVE_CASES))
def test_resolve_dir(name, bcnodes, placeholders, monkeypatch, tmp_path):
    check_env(ENV)
    root, outside = tmp_path / "root", tmp_path / "outside"
    (root / "data").mkdir(parents=True)
    outside.mkdir()
    (root / "link_out").symlink_to(outside, target_is_directory=True)
    (outside / "link_in").symlink_to(root / "data", target_is_directory=True)
    monkeypatch.setenv(ROOTS_ENV, os.pathsep.join(["/path/to/datasets", "~/datasets", str(root), str(tmp_path / "cwd")]))
    path = RESOLVE_CASES[name]
    if path is not None:
        path = path.replace("<tmp>", str(tmp_path))
    try:
        value = ("ok", WHERE["resolve_dir"](path))
    except ValueError as e:
        value = ("ValueError", str(e))
    check(GOLDEN, f"resolve_dir/{name}", placeholders(repr(value)))


def _tree(base):
    """A caption tree under base/dataset with fixed mtimes; returns its path relative to base."""
    files = {
        "a.txt": b"<trigger>, a person", "B.TXT": b"<trigger>, term_a", "c.txt": b"", ".hidden.txt": b"x",
        "img.png": b"png", "notes.md": b"md", "d.txt.bak": b"bak",
        "sub/e.txt": b"<trigger>, term_b, red scarf", "sub/.f.txt": b"x", "sub/deeper/g.txt": b"g",
        ".dotdir/h.txt": b"h",
    }
    for rel, data in files.items():
        path = base / "dataset" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, ns=(MTIME_NS, MTIME_NS))
    return "dataset"


@pytest.mark.parametrize("recursive", [False, True])
def test_dir_fingerprint(recursive, bcnodes, placeholders, tmp_path):
    check_env(ENV)
    rel = _tree(tmp_path / "cwd")
    check(GOLDEN, f"dir_fingerprint/recursive_{recursive}", placeholders(WHERE["dir_fingerprint"](rel, recursive)))


@pytest.mark.parametrize("recursive", [False, True])
def test_dir_fingerprint_dangling_link(recursive, bcnodes, placeholders, tmp_path):
    check_env(ENV)
    rel = _tree(tmp_path / "cwd")
    (tmp_path / "cwd" / rel / "broken.txt").symlink_to(tmp_path / "missing.txt")
    check(GOLDEN, f"dir_fingerprint/dangling_link/recursive_{recursive}", placeholders(WHERE["dir_fingerprint"](rel, recursive)))


@pytest.mark.parametrize("directory", ["missing", "", None, "a_file.txt"])
def test_dir_fingerprint_not_a_directory(directory, bcnodes, placeholders, tmp_path):
    check_env(ENV)
    (tmp_path / "cwd" / "a_file.txt").write_text("x")
    check(GOLDEN, f"dir_fingerprint/not_a_directory/{directory!r}", repr(WHERE["dir_fingerprint"](directory, True)))

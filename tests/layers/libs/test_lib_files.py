"""libs/files.py walk_once: os.walk top-down with links followed, each real folder walked once (a link
back up the tree or a second link to a folder ends there), subfolders in sorted order, pruning honoured."""

import os

import pytest


@pytest.fixture
def files(bcnodes):
    return bcnodes["libs.files"]


def walked(files, root):
    return [os.path.relpath(dirpath, root) for dirpath, _dirnames, _filenames in files.walk_once(str(root))]


def test_walk_follows_links_once_and_ends_on_a_loop(files, tmp_path):
    root = tmp_path / "root"
    (root / "a" / "inner").mkdir(parents=True)
    (root / "b").mkdir()
    (tmp_path / "elsewhere" / "deep").mkdir(parents=True)
    os.symlink(tmp_path / "elsewhere", root / "c")  # a linked folder is walked
    os.symlink(root, root / "a" / "back")  # a loop back to the top: skipped
    os.symlink(tmp_path / "elsewhere", root / "d")  # a second link to the same folder: skipped
    j = os.path.join
    assert walked(files, root) == [".", "a", j("a", "inner"), "b", "c", j("c", "deep")]


def test_pruned_folders_are_skipped(files, tmp_path):
    (tmp_path / "keep" / "x").mkdir(parents=True)
    (tmp_path / "skip" / "y").mkdir(parents=True)
    order = []
    for dirpath, dirnames, _filenames in files.walk_once(str(tmp_path)):
        order.append(os.path.relpath(dirpath, tmp_path))
        dirnames[:] = [d for d in dirnames if d != "skip"]
    assert order == [".", "keep", os.path.join("keep", "x")]


def test_missing_root_walks_nothing(files, tmp_path):
    assert walked(files, tmp_path / "missing") == []

import sys

import pytest

import _golden
import _harness

# The other test_*.py files here are scripts (they exit or chdir at import); only
# the pytest-style modules are collected.
collect_ignore = ["test_import_time.py", "test_nodes.py", "test_runtime.py"]


@pytest.fixture(scope="session")
def comfy_stubs(tmp_path_factory):
    """The stub_comfy set (folder_paths, comfy, comfy.model_management, comfy.utils,
    comfy.ldm.seedvr.*) in sys.modules for the session; returns the stubs' temp root. Any other
    stub is installed by the file that needs it, with monkeypatch.setitem(sys.modules, ...)."""
    tmp = str(tmp_path_factory.mktemp("comfy"))
    _harness.stub_comfy(tmp)
    return tmp


@pytest.fixture(scope="session")
def bcnodes(comfy_stubs):
    """The pack bound as bcnodes_under_test (no root __init__) under the stub_comfy set: the
    ModuleMap of _harness.load_package. _golden.at() and WHERE resolve against this binding."""
    return _harness.load_package()


@pytest.fixture
def second_binding():
    """bind(name): the pack bound again under another package name, returning its ModuleMap, so
    a module runs its import-time code again under stubs the test installed first (the
    downloader registers its routes at import when `server` and `aiohttp` are stubbed). Every
    module under that name leaves sys.modules after the test."""
    names = []

    def bind(name):
        if name in sys.modules:
            raise ValueError(f"{name} is already bound; a second binding needs its own package name")
        names.append(name)
        return _harness.bind_package(name)

    yield bind
    for name in names:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


def pytest_terminal_summary(terminalreporter):
    if not _golden.RECORD:
        return
    terminalreporter.section("BCNODES_GOLDEN_RECORD=1: nothing was asserted; recorded tables")
    for line in _golden.recorded_tables() or ["(no check() or check_env() call ran)"]:
        terminalreporter.write_line(line)

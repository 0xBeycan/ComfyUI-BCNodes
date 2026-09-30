import pytest

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
    ModuleMap of _harness.load_package."""
    return _harness.load_package()

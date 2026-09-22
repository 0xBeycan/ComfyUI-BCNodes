# The other test_*.py files here are scripts (they exit or chdir at import); only
# the pytest-style modules are collected.
collect_ignore = ["test_import_time.py", "test_nodes.py", "test_runtime.py"]

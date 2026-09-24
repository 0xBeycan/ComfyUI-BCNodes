"""Layer rule: nodes/ -> pipelines/ -> models/ -> libs/, each layer importing only the layers
to its right, plus the lazy-import rule, checked statically.

Every .py file of the four layers and the root __init__.py is parsed with ast, never imported,
so modules the import gate never reaches cold are covered too. Every Import/ImportFrom in a file
counts, wherever it sits; one with a function or lambda around it is lazy, any other is
module-level. The vendored BiRefNet code lives in models/birefnet/arch/; a root birefnet/ must not
come back, and the walk fails if it does.

    python -m pytest tests/test_layers.py
"""

import ast
import os
import sys

from _harness import PKG_DIR

LAYERS = ("nodes", "pipelines", "models", "libs")
VENDORED = ("birefnet",)
PACK_NAMES = LAYERS + VENDORED + ("tests",)
OUTSIDE = "outside the layers"

# Violations present at ae8062d, all fixed by the refactor. The list stays empty for good:
# test_transitional_is_empty fails on any entry, so a new violation is fixed, not allowlisted.
TRANSITIONAL = [
]


def _module_of(rel):
    parts = rel[:-3].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _files():
    """(path relative to the repo, dotted module name) of every file the rule covers."""
    found = [("__init__.py", "")]
    for top in LAYERS + VENDORED:
        for dirpath, dirnames, filenames in os.walk(os.path.join(PKG_DIR, top)):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for f in sorted(filenames):
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dirpath, f), PKG_DIR).replace(os.sep, "/")
                    found.append((rel, _module_of(rel)))
    return found


def _imports(rel):
    """(node, lazy, inside a try) for every Import/ImportFrom of the file."""
    with open(os.path.join(PKG_DIR, rel), encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=rel)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    tries = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            chain, p = [], parents.get(node)
            while p is not None:
                chain.append(p)
                p = parents.get(p)
            lazy = any(isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for a in chain)
            yield node, lazy, any(isinstance(a, tries) for a in chain)


def _targets(rel, module, node):
    """The pack modules a relative import reaches ("" = the root package). `from <dots><module>
    import names` reaches <base>.<module>; with no module, each name that is a file or package
    under <base> is a target, any other name reaches <base>. None: above the package root."""
    package = module.split(".") if module else []
    if not rel.endswith("__init__.py"):
        package = package[:-1]
    up = node.level - 1
    if up > len(package):
        return None
    base = package[:len(package) - up]
    if node.module:
        return [".".join(base + node.module.split("."))]
    targets = []
    for alias in node.names:
        path = os.path.join(PKG_DIR, *base, alias.name)
        sub = os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))
        targets.append(".".join(base + [alias.name] if sub else base))
    return targets


def _layer(module):
    top = module.split(".")[0]
    if module == "":
        return "root"
    if top in LAYERS:
        return top
    return "vendored" if top in VENDORED else "unknown"


def _allowed(src, tgt):
    s, t = src.split("."), tgt.split(".")
    sl, tl = _layer(src), _layer(tgt)
    if sl == "root":
        return tl == "nodes"
    if sl == "nodes":
        return src != "nodes.common" and (tgt == "nodes.common" or tl in ("pipelines", "models", "libs"))
    if sl == "pipelines":
        if tl == "pipelines":
            return len(s) > 1 and len(t) > 1 and s[1] == t[1]
        return tl in ("models", "libs")
    if sl == "models":
        if tl == "libs":
            return True
        if tl != "models" or len(t) < 2:
            return False
        if src == "models":  # the registration hub imports its children
            return True
        return t[1] == "common" or (s[1] != "common" and t[1] == s[1])
    if sl == "libs":
        return tl == "libs"
    return False


def _scan():
    """{kind: {violation key: ["path:line", ...]}} over every file."""
    found = {"edge": {}, "absolute": {}, "module-level": {}}

    def add(kind, key, rel, node):
        found[kind].setdefault(key, []).append(f"{rel}:{node.lineno}")

    for rel, module in _files():
        if _layer(module) == "vendored":
            found["edge"].setdefault((module.split(".")[0], OUTSIDE), []).append(rel)
        for node, lazy, in_try in _imports(rel):
            if isinstance(node, ast.ImportFrom) and node.level:
                for target in _targets(rel, module, node) or ["<above the package root>"]:
                    if not _allowed(module, target):
                        add("edge", (module, target), rel, node)
                continue
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            for name in names:
                top = name.split(".")[0]
                if top in PACK_NAMES and not (
                        rel == "nodes/image_comparer.py" and lazy and isinstance(node, ast.ImportFrom)
                        and name == "nodes" and [a.name for a in node.names] == ["PreviewImage"]):
                    add("absolute", (module, name), rel, node)  # ComfyUI's nodes.PreviewImage is the one exception
                if not lazy and not (
                        top in sys.stdlib_module_names or top in ("torch", "numpy")
                        or (rel == "nodes/downloader.py" and in_try and top in ("server", "aiohttp"))):
                    add("module-level", (module, name), rel, node)
    return found


def _report(found):
    return "\n".join(f"  {key}: {', '.join(where)}" for key, where in found.items())


def test_transitional_is_empty():
    assert TRANSITIONAL == [], f"TRANSITIONAL must stay empty; fix the edge instead: {TRANSITIONAL}"


def test_imports_between_layers_follow_the_rule():
    edges = _scan()["edge"]
    new = {key: where for key, where in edges.items() if key not in TRANSITIONAL}
    stale = [key for key in TRANSITIONAL if key not in edges]
    assert not new, "forbidden import edges (source module, target module):\n" + _report(new)
    assert not stale, f"TRANSITIONAL entries whose violation is gone; delete them: {stale}"


def test_no_absolute_import_of_a_pack_name():
    found = _scan()["absolute"]
    assert not found, "absolute imports of a pack name (use a relative import):\n" + _report(found)


def test_module_level_imports_are_stdlib_torch_numpy_or_relative():
    found = _scan()["module-level"]
    assert not found, "module-level imports that must move inside the function using them:\n" + _report(found)


def test_every_layer_directory_is_a_package():
    missing = set()
    for rel, _ in _files():
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            if not os.path.isfile(os.path.join(PKG_DIR, *parts[:i], "__init__.py")):
                missing.add("/".join(parts[:i]))
    assert not missing, f"directories without __init__.py (they would be namespace packages): {sorted(missing)}"

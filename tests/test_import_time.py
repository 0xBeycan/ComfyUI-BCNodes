"""Import-time gate for ComfyUI-BCNodes.

Run from anywhere with a Python that has torch and numpy:

    python tests/test_import_time.py

Measures, with torch and numpy already loaded (ComfyUI has them loaded long
before custom nodes are imported):

  * the whole package, once, in this process — must stay under BUDGET_S and
    must not pull in any of the HEAVY modules;
  * each node module on its own, cold, in a fresh subprocess.

Exits non-zero when the budget or the heavy-module rule is broken.
"""

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "bcnodes_under_test"
BUDGET_S = 0.1
NODE_MODULES = ["logic", "mask", "image_scale", "lists", "birefnet", "downloader", "math_expression", "prompt_list", "any_switch", "seed", "show_text",
                "image_comparer", "video_comparer", "power_lora_loader", "everywhere", "seedvr2",
                "postfx", "caption_audit", "social_media_export", "image_quality_gate", "save_image", "skin_texture"]
HEAVY = [
    "transformers", "timm", "scipy", "cv2", "PIL", "huggingface_hub",
    "safetensors", "kornia", "einops", "torchvision", "folder_paths",
    "yaml", "postfx", "caption_audit",
]


def bind_package(execute_init):
    """Registers the repo directory as a package under PKG_NAME, the way
    ComfyUI does for custom_nodes. With execute_init=False only the package
    object is created, so a single node module can be imported alone."""
    if execute_init:
        spec = importlib.util.spec_from_file_location(
            PKG_NAME, os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PKG_NAME] = module
        spec.loader.exec_module(module)
        return module
    module = types.ModuleType(PKG_NAME)
    module.__path__ = [PKG_DIR]
    sys.modules[PKG_NAME] = module
    return module


def time_node_module(name):
    code = (
        "import sys, time; import torch, numpy\n"
        f"sys.path.insert(0, {json.dumps(os.path.dirname(os.path.abspath(__file__)))})\n"
        "import test_import_time as t\n"
        "t.bind_package(execute_init=False)\n"
        "before = set(sys.modules)\n"
        "t0 = time.perf_counter()\n"
        f"import importlib; importlib.import_module(t.PKG_NAME + '.nodes.{name}')\n"
        "dt = time.perf_counter() - t0\n"
        "heavy = sorted(m for m in set(sys.modules) - before if m.split('.')[0] in t.HEAVY)\n"
        "print(__import__('json').dumps({'seconds': dt, 'heavy': heavy}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PKG_DIR)
    if out.returncode != 0:
        raise RuntimeError(f"nodes.{name} failed to import:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    import numpy  # noqa: F401
    import torch  # noqa: F401

    failures = []

    before = set(sys.modules)
    t0 = time.perf_counter()
    pkg = bind_package(execute_init=True)
    package_s = time.perf_counter() - t0
    heavy = sorted(m for m in set(sys.modules) - before if m.split(".")[0] in HEAVY)

    print(f"{'module':<28}{'import (s)':>12}  heavy imports")
    print(f"{'package':<28}{package_s:>12.4f}  {', '.join(heavy) or '-'}")
    if package_s > BUDGET_S:
        failures.append(f"package import took {package_s:.3f}s (budget {BUDGET_S}s)")
    if heavy:
        failures.append(f"package import pulled in heavy modules: {', '.join(heavy)}")

    for name in NODE_MODULES:
        r = time_node_module(name)
        print(f"{'nodes.' + name:<28}{r['seconds']:>12.4f}  {', '.join(r['heavy']) or '-'}")
        if r["heavy"]:
            failures.append(f"nodes.{name} pulled in heavy modules: {', '.join(r['heavy'])}")

    expected = {
        "BC_LogicBoolean", "BC_IsMaskEmpty", "BC_MaskFillHoles", "BC_MaskGrow", "BC_ImageScaleByAspectRatio",
        "BC_JoinImageLists", "BC_BiRefNetRemoveBackground", "BC_AutoModelDownloader",
        "BC_MathExpression", "BC_PromptList", "BC_AnySwitch", "BC_Seed", "BC_ShowText",
        "BC_ImageComparer", "BC_VideoComparer", "BC_PowerLoraLoader", "BC_AnythingEverywhere", "BC_FastGroupsBypasser",
        "BC_SeedVR2Resize", "BC_SeedVR2VAEEncode", "BC_SeedVR2VAEDecode", "BC_SeedVR2PostProcess",
        "BC_PostFxApply", "BC_PostFxTheme", "BC_PostFxCustomLook", "BC_PostFxLut", "BC_PostFxSignatureSheet",
        "BC_CaptionAudit", "BC_SocialMediaExport", "BC_ImageQualityGate", "BC_SaveImage", "BC_SkinTexture",
    }
    registered = set(pkg.NODE_CLASS_MAPPINGS)
    if registered != expected:
        failures.append(f"NODE_CLASS_MAPPINGS keys differ: missing={sorted(expected - registered)} extra={sorted(registered - expected)}")
    print(f"\nregistered: {', '.join(sorted(registered))}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()

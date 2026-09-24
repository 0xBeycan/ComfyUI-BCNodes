"""Shared test harness: the ComfyUI stubs and the package binding.

PKG_DIR, PKG_NAME, stub_comfy and load_package are tests/test_nodes.py's harness, moved here
verbatim so every test binds the pack the same way. load_package now returns a ModuleMap: the
same eager node modules by short name, plus any other pack module by its dotted path, imported
on first access (m["libs.video"]), so a later move edits one access line instead of a module
list. Imports only the standard library at module level (stub_comfy imports torch when called).
"""

import importlib
import os
import sys
import types

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "bcnodes_under_test"


def stub_comfy(tmp):
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(tmp, "models")
    fp.get_user_directory = lambda: os.path.join(tmp, "user")
    fp.folder_names_and_paths = {}

    def add_model_folder_path(name, path, is_default=False):
        fp.folder_names_and_paths.setdefault(name, ([], set()))[0].append(path)

    def get_full_path(name, filename):
        for p in fp.folder_names_and_paths.get(name, ([], set()))[0]:
            f = os.path.join(p, filename)
            if os.path.isfile(f):
                return f
        return None

    fp.add_model_folder_path = add_model_folder_path
    fp.get_full_path = get_full_path
    fp.get_filename_list = lambda name: []
    fp.get_output_directory = lambda: os.path.join(tmp, "output")
    fp.get_temp_directory = lambda: os.path.join(tmp, "temp")

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        subfolder, filename = os.path.split(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(output_dir, subfolder)
        return full_output_folder, filename, 1, subfolder, filename_prefix

    fp.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = fp

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    cu = types.ModuleType("comfy.utils")
    import torch

    mm.get_torch_device = lambda: torch.device("cpu")
    mm.vae_device = lambda: torch.device("cpu")
    mm.soft_empty_cache = lambda: None
    mm.load_models_gpu = lambda models, **kw: None
    mm.get_free_memory = lambda device=None, torch_free_too=False: 2 ** 40
    mm.unload_all_models = lambda: None

    import contextlib
    mm.cuda_device_context = lambda device: contextlib.nullcontext()

    # comfy.ldm.seedvr: what the SeedVR2 decode / post-process nodes import lazily.
    ldm = types.ModuleType("comfy.ldm")
    seedvr = types.ModuleType("comfy.ldm.seedvr")
    constants = types.ModuleType("comfy.ldm.seedvr.constants")
    constants.BYTEDANCE_VAE_SCALING_FACTOR = 0.9152
    constants.BYTEDANCE_VAE_SHIFTING_FACTOR = 0.0
    vae_mod = types.ModuleType("comfy.ldm.seedvr.vae")

    class MemoryState:
        DISABLED, INITIALIZING, ACTIVE = 0, 1, 2

    vae_mod.MemoryState = MemoryState
    color_fix = types.ModuleType("comfy.ldm.seedvr.color_fix")
    color_fix.lab_color_transfer = lambda content, style: content
    color_fix.wavelet_color_transfer = lambda content, style: content
    color_fix.adain_color_transfer = lambda content, style: style
    for name, mod in (("comfy.ldm", ldm), ("comfy.ldm.seedvr", seedvr), ("comfy.ldm.seedvr.constants", constants),
                      ("comfy.ldm.seedvr.vae", vae_mod), ("comfy.ldm.seedvr.color_fix", color_fix)):
        sys.modules[name] = mod

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, n):
            pass

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = cu


class ModuleMap(dict):
    """Modules of one package binding. The eager entries are keyed by node module name
    (m["logic"]); any other key is a dotted path under the package, imported when first read
    (m["libs.video"] -> <package>.libs.video)."""

    def __init__(self, package, eager):
        super().__init__(eager)
        self.package = package

    def __missing__(self, key):
        return importlib.import_module(f"{self.package}.{key}")


def load_package():
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [PKG_DIR]
    sys.modules[PKG_NAME] = pkg
    return ModuleMap(PKG_NAME, {name: importlib.import_module(f"{PKG_NAME}.nodes.{name}")
            for name in ("logic", "mask", "image_scale", "lists", "birefnet", "downloader", "math_expression", "prompt_list", "any_switch", "seed", "show_text",
                         "image_comparer", "video_comparer", "power_lora_loader", "everywhere", "seedvr2",
                         "postfx", "caption_audit", "social_media_export", "image_quality_gate", "save_image", "skin_texture")})


def bind_package(name):
    """Binds the repo directory as the package `name`, as load_package does (the root __init__
    is not executed), but imports no module: returns a ModuleMap with no eager entries."""
    pkg = types.ModuleType(name)
    pkg.__path__ = [PKG_DIR]
    sys.modules[name] = pkg
    return ModuleMap(name, {})

"""Lora Loader (Key Fix) (nodes/lora_key_fix.py) through the node method, with stand-ins for the core
functions it calls (comfy.utils.load_torch_file, comfy.lora_convert.convert_lora,
comfy.lora.model_lora_keys_unet, comfy.sd.load_lora_for_models, folder_paths):

- the surface: a MODEL, a LoRA from models/loras and a strength in, a MODEL out, under BCNodes/loaders;
- the file is read with its metadata and goes through core's format conversion first; the key map is
  built from the model's own BaseModel; core gets the renamed dict, the same tensors, no CLIP, the
  strength for the model, 0 for the CLIP and the file's metadata, and its patched model comes out;
- the console line: keys, renames by kind, keys that match no module (a warning naming the first
  five when there are any);
- strength 0 returns the model untouched and reads no file; no model (None) comes out as None.

The renames themselves are tests/layers/libs/test_lib_lora_keys.py; the node against core's real
loader on a real (tiny) Wan model is tests/test_runtime.py.
"""

import logging
import sys
import types

import pytest

KEY_MAP = {"diffusion_model.blocks.0.modulation", "diffusion_model.blocks.0.self_attn.q",
           "lora_unet_blocks_0_self_attn_q", "diffusion_model.head.modulation"}


@pytest.fixture
def module(bcnodes):
    return bcnodes["lora_key_fix"]


@pytest.fixture
def core(monkeypatch, comfy_stubs):
    """Stand-ins for the core functions, recording their calls; `core.file` is what the LoRA file holds."""
    rec = types.SimpleNamespace(file={}, metadata={"format": "pt"}, calls=[], patched=object())
    fp = sys.modules["folder_paths"]
    comfy = sys.modules["comfy"]
    monkeypatch.setattr(fp, "get_filename_list", lambda name: ["a.safetensors", "sub/b.safetensors"] if name == "loras" else [])

    def get_full_path_or_raise(folder, name):
        rec.calls.append(("path", folder, name))
        return f"/loras/{name}"

    def load_torch_file(path, safe_load=False, return_metadata=False):
        rec.calls.append(("load", path, safe_load, return_metadata))
        return dict(rec.file), rec.metadata

    def convert_lora(sd):
        rec.calls.append(("convert", sorted(sd)))
        return {k.replace("lora_unet__", "lora_unet_"): v for k, v in sd.items()}

    def model_lora_keys_unet(model, key_map):
        rec.calls.append(("key_map", model, dict(key_map)))
        key_map.update({k: k for k in KEY_MAP})
        return key_map

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
        rec.calls.append(("apply", model, clip, lora, strength_model, strength_clip, lora_metadata))
        return rec.patched, None

    monkeypatch.setattr(fp, "get_full_path_or_raise", get_full_path_or_raise, raising=False)
    monkeypatch.setattr(sys.modules["comfy.utils"], "load_torch_file", load_torch_file, raising=False)
    for name, fns in (("lora_convert", {"convert_lora": convert_lora}), ("lora", {"model_lora_keys_unet": model_lora_keys_unet}),
                      ("sd", {"load_lora_for_models": load_lora_for_models})):
        mod = types.ModuleType(f"comfy.{name}")
        for fn_name, fn in fns.items():
            setattr(mod, fn_name, fn)
        monkeypatch.setitem(sys.modules, f"comfy.{name}", mod)
        monkeypatch.setattr(comfy, name, mod, raising=False)
    return rec


def test_the_surface(module, core):
    cls = module.LoraLoaderKeyFix
    required = cls.INPUT_TYPES()["required"]
    assert list(required) == ["model", "lora_name", "strength"]
    assert required["model"][0] == "MODEL" and required["lora_name"][0] == ["a.safetensors", "sub/b.safetensors"]
    assert required["strength"][0] == "FLOAT" and required["strength"][1]["default"] == 1.0
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY) == (("MODEL",), ("MODEL",), "load", "BCNodes/loaders")
    assert not getattr(cls, "OUTPUT_NODE", False)
    assert module.NODE_CLASS_MAPPINGS == {"BC_LoraLoaderKeyFix": cls}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"BC_LoraLoaderKeyFix": "Lora Loader (Key Fix)"}


def test_core_gets_the_renamed_lora(module, core, caplog):
    tensors = {k: object() for k in ("diffusion_model.blocks.0.diff_m", "blocks.0.self_attn.q.lora_A.default.weight",
                                     "lora_unet__blocks_0_self_attn_q.lora_down.weight")}
    core.file = tensors
    model = types.SimpleNamespace(model=object())
    with caplog.at_level(logging.INFO):
        out = module.LoraLoaderKeyFix().load(model, "sub/b.safetensors", 0.75)
    assert out == (core.patched,)
    (path, load, convert, key_map, apply) = core.calls
    assert path == ("path", "loras", "sub/b.safetensors")
    assert load == ("load", "/loras/sub/b.safetensors", True, True)
    assert convert == ("convert", sorted(tensors))
    assert key_map == ("key_map", model.model, {})
    _, applied_to, clip, lora, strength, strength_clip, metadata = apply
    assert (applied_to, clip, strength, strength_clip, metadata) == (model, None, 0.75, 0, {"format": "pt"})
    # renamed in the LoRA's order, the converted Wan Fun key (core's own conversion) kept as converted
    assert list(lora) == ["diffusion_model.blocks.0.modulation.diff", "diffusion_model.blocks.0.self_attn.q.lora_A.default.weight",
                          "lora_unet_blocks_0_self_attn_q.lora_down.weight"]
    assert list(lora.values()) == list(tensors.values())
    assert caplog.messages == ["[BCNodes] Lora Loader (Key Fix): sub/b.safetensors: 3 keys, 2 renamed (1 .diff_m -> "
                               ".modulation.diff, 1 given the diffusion_model. prefix), 0 match no module of this model"]
    assert caplog.records[0].levelno == logging.INFO


def test_unmapped_keys_are_a_warning_naming_the_first_five(module, core, caplog):
    unmapped = [f"diffusion_model.img_emb.proj.{i}.lora_A.weight" for i in range(6)]
    core.file = {k: object() for k in ["diffusion_model.head.diff_m"] + unmapped}
    with caplog.at_level(logging.INFO):
        module.LoraLoaderKeyFix().load(types.SimpleNamespace(model=object()), "a.safetensors", 1.0)
    assert caplog.records[0].levelno == logging.WARNING
    assert caplog.messages == ["[BCNodes] Lora Loader (Key Fix): a.safetensors: 7 keys, 1 renamed (1 .diff_m -> .modulation.diff, "
                               "0 given the diffusion_model. prefix), 6 match no module of this model and are not applied: "
                               + ", ".join(unmapped[:5]) + ", ..."]
    # the unmapped keys still go to core, which names each one ("lora key not loaded")
    assert set(core.calls[-1][3]) == {"diffusion_model.head.modulation.diff"} | set(unmapped)


def test_strength_zero_or_no_model_reads_nothing(module, core):
    model = object()
    assert module.LoraLoaderKeyFix().load(model, "a.safetensors", 0.0) == (model,)
    assert module.LoraLoaderKeyFix().load(None, "a.safetensors", 1.0) == (None,)
    assert core.calls == []

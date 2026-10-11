"""Qwen LM, LM Config and LM Loader (nodes/lm.py) through the node classes, on the stub folder_paths (core's
folder semantics: a folder kept once, get_folder_paths a copy, KeyError for an unknown key) and the real Qwen LM
catalog:

  - the surface: BC_QwenLM's and BC_LMConfig's widgets in the brief's order with their defaults, ranges,
    steps, multiline text and the seed's `fixed` control, a tooltip on every input, outputs, category,
    display names, the built-in models the description names; the model / precision / LoRA lists from the
    catalog and the LoRA folders;
  - the folders: models/Qwen-LM registered once (after a folder extra_model_paths.yaml gave the key), its
    empty extension set given .safetensors, text_encoders folders read (none: empty);
  - a broken user models.yaml: the widgets list the built-in catalog and the error is logged; a run raises;
  - LM Config: only the edited fields leave it, typed as pipelines/lm.LMConfigValues says, in edited's
    order, then every other field fed by a link in the prompt (its hidden prompt_graph / unique_id), so a
    linked field is applied with edited empty; no prompt: edited alone; an unknown name is an error naming
    the fields;
  - Qwen LM hands every input to pipelines/lm.run unchanged;
  - LM Loader: family (the families' names, never their registry keys), then Qwen LM's model, precision, lora
    and lora_strength widgets (the same specs, one MODEL_DEFAULTS for both nodes), keep_model_loaded; the lists
    of every registered family (a second family registered for the test); its output, a pipelines/lm.LMSpec of
    the widgets with the family's name mapped to its key, with nothing loaded or looked for; an unknown family
    name, a model of another family, a precision of another model and a LoRA of another family are errors
    naming what to pick (families by name); two families with one name are refused, naming both;
  - GET /bcnodes/lm/catalog: the payload's shape (pipelines/lm.CatalogPayload: each family's name and LoRA
    list, the loader node), every registered family listed, and the route on a stand-in server.
The flow itself is tests/layers/pipelines/test_pipe_lm.py.
"""

import asyncio
import dataclasses
import json
import logging
import os
import sys
import types

import pytest

QWEN_REQUIRED = ["model", "precision", "lora", "lora_strength", "thinking", "max_new_tokens", "seed", "keep_model_loaded",
                 "system", "user", "assistant"]
CONFIG_REQUIRED = ["do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty", "mtp",
                   "edited"]
MODELS = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-9B Abliterated", "Qwen3-VL-8B Instruct", "Qwen3-VL-8B Instruct Abliterated",
          "Qwen3.8-27B"]


@pytest.fixture
def lm(bcnodes):
    return bcnodes["lm"]


@pytest.fixture
def fp(comfy_stubs, monkeypatch, tmp_path):
    """The stub folder_paths with models_dir in tmp_path and one text_encoders folder."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(fp, "folder_names_and_paths", {"text_encoders": ([str(tmp_path / "models" / "text_encoders")],
                                                                         {".safetensors"})})
    return fp


def qwen_folder(tmp_path):
    return str(tmp_path / "models" / "Qwen-LM")


# --- the surface ---------------------------------------------------------------------------------------------


def test_qwen_lm_inputs(lm, fp):
    spec = lm.QwenLM.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(req) == QWEN_REQUIRED and list(opt) == ["image", "config"]
    assert req["model"][0] == MODELS and req["model"][1]["default"] == "Qwen3.5-9B"
    assert req["precision"][0] == ["BF16", "INT8 ConvRot", "W4A8"] and req["precision"][1]["default"] == "INT8 ConvRot"
    assert req["lora"][0] == ["None"] and req["lora"][1]["default"] == "None"
    assert req["lora_strength"] == ("FLOAT", {**req["lora_strength"][1], "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01})
    assert req["thinking"][0] == "BOOLEAN" and req["thinking"][1]["default"] is False
    assert req["max_new_tokens"][0] == "INT" and {k: req["max_new_tokens"][1][k] for k in ("default", "min", "max")} == {
        "default": 32768, "min": 1, "max": 262144}
    seed = req["seed"][1]
    assert req["seed"][0] == "INT" and (seed["default"], seed["min"], seed["max"]) == (0, 0, 0xffffffffffffffff)
    assert seed["control_after_generate"] == "fixed"
    assert req["keep_model_loaded"][0] == "BOOLEAN" and req["keep_model_loaded"][1]["default"] is True
    for name in ("system", "user", "assistant"):
        assert req[name][0] == "STRING" and req[name][1]["default"] == "" and req[name][1]["multiline"] is True
    assert opt["image"][0] == "IMAGE" and opt["config"][0] == "BC_LM_CONFIG"
    assert all(options.get("tooltip") for options in (v[1] for v in (*req.values(), *opt.values())))
    tokens = req["max_new_tokens"][1]["tooltip"]
    assert "1 GiB" in tokens and "2 GiB" in tokens and "4.5 GiB" in tokens  # the KV cache core reserves


def test_lm_config_inputs(lm):
    req = lm.LMConfig.INPUT_TYPES()["required"]
    assert list(req) == CONFIG_REQUIRED and list(lm.LMConfig.INPUT_TYPES()) == ["required", "hidden"]
    assert lm.LMConfig.INPUT_TYPES()["hidden"] == {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}
    assert req["do_sample"] == ("BOOLEAN", {**req["do_sample"][1], "default": True})
    ranges = {"temperature": ("FLOAT", 0.7, 0.01, 2.0, 0.01), "top_k": ("INT", 20, 0, 1000, None),
              "top_p": ("FLOAT", 0.8, 0.0, 1.0, 0.01), "min_p": ("FLOAT", 0.0, 0.0, 1.0, 0.01),
              "repetition_penalty": ("FLOAT", 1.0, 0.0, 5.0, 0.01), "presence_penalty": ("FLOAT", 1.5, 0.0, 5.0, 0.01)}
    for name, (kind, default, lo, hi, step) in ranges.items():
        options = req[name][1]
        assert (req[name][0], options["default"], options["min"], options["max"], options.get("step")) == (kind, default, lo, hi, step)
    assert req["mtp"][0] == ["auto", "off", "2", "3", "4", "5"] and req["mtp"][1]["default"] == "auto"
    assert req["edited"][0] == "STRING" and req["edited"][1]["default"] == "" and not req["edited"][1].get("multiline")
    assert all(v[1].get("tooltip") for v in req.values())
    assert "0 to 2" in req["presence_penalty"][1]["tooltip"]
    mtp = req["mtp"][1]["tooltip"]
    assert "text-only" in mtp and "MTP head" in mtp and "differs from non-MTP output for the same seed" in mtp


def test_config_fields_match_the_widgets_and_the_catalog(lm, bcnodes):
    """pipelines/lm.LMConfigValues (the BC_LM_CONFIG value; its keys are the names `edited` takes, in widget
    order) holds LM Config's widgets but `edited`, typed as the catalog's Sampling fields the pipeline overlays
    them on; the mtp combo is the backend's MTP_CHOICES."""
    req = lm.LMConfig.INPUT_TYPES()["required"]
    fields = bcnodes["pipelines.lm"].LMConfigValues
    assert list(fields.__annotations__) == CONFIG_REQUIRED[:-1] and fields.__total__ is False
    catalog = bcnodes["models.common.lm.catalog"]
    assert fields.__annotations__ == {f.name: f.type for f in dataclasses.fields(catalog.Sampling)}
    assert set(fields.__annotations__) == set(catalog.SAMPLING_FIELDS)
    assert req["mtp"][0] == list(bcnodes["models.common.lm.backend"].MTP_CHOICES)


def test_outputs_and_names(lm):
    assert (lm.QwenLM.RETURN_TYPES, lm.QwenLM.RETURN_NAMES, lm.QwenLM.FUNCTION) == (("STRING", "STRING"), ("text", "thinking"), "generate")
    assert (lm.LMConfig.RETURN_TYPES, lm.LMConfig.RETURN_NAMES, lm.LMConfig.FUNCTION) == (("BC_LM_CONFIG",), ("config",), "build")
    assert lm.QwenLM.CATEGORY == lm.LMConfig.CATEGORY == "BCNodes/lm"
    # the menu order: Qwen LM, LM Config, LM Loader, Captioner (tests/layers/nodes/test_node_captioner.py)
    assert list(lm.NODE_CLASS_MAPPINGS.items()) == [("BC_QwenLM", lm.QwenLM), ("BC_LMConfig", lm.LMConfig),
                                                    ("BC_LMLoader", lm.LMLoader), ("BC_Captioner", lm.Captioner)]
    assert lm.NODE_DISPLAY_NAME_MAPPINGS == {"BC_QwenLM": "Qwen LM", "BC_LMConfig": "LM Config",
                                             "BC_LMLoader": "LM Loader", "BC_Captioner": "Captioner"}
    assert lm.FAMILY_NODES == {"BC_QwenLM": "qwen_lm"}
    for cls in (lm.QwenLM, lm.LMConfig):
        assert cls.DESCRIPTION and cls.SEARCH_ALIASES[0] == "BCNodes" and not getattr(cls, "OUTPUT_NODE", False)


def test_description_names_the_built_in_models(lm):
    """The abliterated builds the description names are the catalog's (MODELS), not one per base model."""
    assert [m.removesuffix(" Abliterated") for m in MODELS if m.endswith(" Abliterated")] == [
        "Qwen3.5-9B", "Qwen3-VL-8B Instruct"]
    assert lm.QwenLM.DESCRIPTION.startswith(
        "Runs a Qwen chat model (Qwen3.5-4B / 9B, Qwen3.8-27B, Qwen3-VL-8B Instruct, and abliterated builds of "
        "Qwen3.5-9B and Qwen3-VL-8B) through ComfyUI core")


def test_lora_list_from_the_loras_folders(lm, fp, tmp_path):
    for rel in ("b.safetensors", os.path.join("sub", "a.safetensors"), os.path.join("peft", "adapter_model.safetensors"),
                os.path.join("peft", "adapter_config.json")):
        path = os.path.join(qwen_folder(tmp_path), "loras", rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()
    assert lm.QwenLM.INPUT_TYPES()["required"]["lora"][0] == ["None", "b.safetensors", "peft", os.path.join("sub", "a.safetensors")]


# --- the folders ---------------------------------------------------------------------------------------------


def test_folders_registered_once(lm, fp, tmp_path):
    first = lm.family_folders("qwen_lm")
    assert first == ((qwen_folder(tmp_path),), (str(tmp_path / "models" / "text_encoders"),))
    assert fp.folder_names_and_paths["Qwen-LM"] == ([qwen_folder(tmp_path)], {".safetensors"})
    assert lm.family_folders("qwen_lm") == first and fp.folder_names_and_paths["Qwen-LM"][0] == [qwen_folder(tmp_path)]


def test_folders_after_extra_model_paths(lm, fp, tmp_path):
    fp.folder_names_and_paths["Qwen-LM"] = (["/elsewhere/Qwen-LM"], set())  # what extra_model_paths.yaml makes
    del fp.folder_names_and_paths["text_encoders"]
    assert lm.family_folders("qwen_lm") == (("/elsewhere/Qwen-LM", qwen_folder(tmp_path)), ())
    assert fp.folder_names_and_paths["Qwen-LM"][1] == {".safetensors"}
    fp.folder_names_and_paths["Qwen-LM"] = (["/x"], {".sft"})
    lm.family_folders("qwen_lm")
    assert fp.folder_names_and_paths["Qwen-LM"][1] == {".sft"}  # a filter already set stays


def test_broken_user_catalog(lm, fp, tmp_path, caplog, monkeypatch):
    os.makedirs(qwen_folder(tmp_path))
    yaml_path = os.path.join(qwen_folder(tmp_path), "models.yaml")
    with open(yaml_path, "w") as f:
        f.write("models: [oops]\n")
    with caplog.at_level(logging.ERROR):
        req = lm.QwenLM.INPUT_TYPES()["required"]
    assert req["model"][0] == MODELS and req["precision"][0] == ["BF16", "INT8 ConvRot", "W4A8"]
    assert f"[BCNodes] Qwen LM: {yaml_path}: the file must hold one key" in caplog.text
    with pytest.raises(ValueError, match="the file must hold one key"):
        lm.QwenLM().generate("Qwen3.5-9B", "INT8 ConvRot", "None", 1.0, False, 8, 0, True, "", "hi", "")


# --- LM Config -----------------------------------------------------------------------------------------------


def build(lm, edited, **values):
    widgets = dict(do_sample=False, temperature=0.25, top_k=7, top_p=0.5, min_p=0.1, repetition_penalty=1.2,
                   presence_penalty=0.0, mtp="3")
    return lm.LMConfig().build(**{**widgets, **values}, edited=edited)[0]


def test_config_nothing_edited(lm):
    assert build(lm, "") == {} and build(lm, " , ,") == {}


def test_config_only_the_edited_fields(lm):
    out = build(lm, " top_k ,temperature,top_k,, mtp")
    assert out == {"top_k": 7, "temperature": 0.25, "mtp": "3"} and list(out) == ["top_k", "temperature", "mtp"]
    every = build(lm, ",".join(CONFIG_REQUIRED[:-1]), top_k=7.0, temperature=1, do_sample=1)
    assert [type(v) for v in every.values()] == [bool, float, int, float, float, float, float, str]


def test_config_linked_fields_are_applied(lm):
    def graph(edited="", **links):
        inputs = dict(do_sample=False, temperature=0.25, top_k=7, top_p=0.5, min_p=0.1, repetition_penalty=1.2,
                      presence_penalty=0.0, mtp="3", edited=edited)
        return {"4": {"class_type": "PrimitiveFloat", "inputs": {"value": 0.25}},
                "9": {"class_type": "BC_LMConfig", "inputs": {**inputs, **links}}}

    def linked(edited="", node="9", prompt=None, **values):
        if prompt is None:
            prompt = graph(edited, temperature=["4", 0], top_k=["5", 0])
        return build(lm, edited, **values, prompt_graph=prompt, unique_id=node)

    # edited empty (an API prompt, or after Reset to model defaults): the linked fields still apply, in widget order
    out = linked(top_k=3.0)
    assert out == {"temperature": 0.25, "top_k": 3} and list(out) == ["temperature", "top_k"] and type(out["top_k"]) is int
    # edited first, a field both edited and linked once
    out = linked("mtp, top_k")
    assert out == {"mtp": "3", "top_k": 7, "temperature": 0.25} and list(out) == ["mtp", "top_k", "temperature"]
    # no prompt (a direct call), a node the prompt does not hold (made by expansion), no link: edited alone
    assert build(lm, "mtp") == {"mtp": "3"}
    assert linked("mtp", node="12") == {"mtp": "3"} and linked("mtp", prompt={}) == {"mtp": "3"}
    assert linked("mtp", prompt=graph("mtp")) == {"mtp": "3"}
    # a linked edited is a value like any other; only the fields count as linked
    assert linked("top_p", prompt=graph(["6", 0])) == {"top_p": 0.5}


def test_config_unknown_name(lm):
    with pytest.raises(ValueError, match="unknown field\\(s\\) temp, seed; the fields are do_sample, temperature, top_k, "
                                         "top_p, min_p, repetition_penalty, presence_penalty, mtp$"):
        build(lm, "temp,top_p,seed")


# --- Qwen LM -> the pipeline ---------------------------------------------------------------------------------


def test_generate_hands_everything_to_the_pipeline(lm, fp, tmp_path, monkeypatch, bcnodes):
    pipe = bcnodes["pipelines.lm"]
    seen = []
    monkeypatch.setattr(pipe, "run", lambda request: seen.append(request) or pipe.LMResult("the text", "the reasoning"))
    image = object()
    out = lm.QwenLM().generate("Qwen3.8-27B", "W4A8", "x.safetensors", 0.5, True, 100, 42, False, "sys", "usr", "",
                               image=image, config={"top_k": 5})
    assert out == ("the text", "the reasoning")
    assert seen == [pipe.LMRequest(
        family="qwen_lm", node="Qwen LM", model="Qwen3.8-27B", precision="W4A8", lora="x.safetensors", lora_strength=0.5,
        thinking=True, max_new_tokens=100, seed=42, keep_model_loaded=False, system="sys", user="usr", assistant="",
        image=image, config={"top_k": 5}, model_folders=(qwen_folder(tmp_path),),
        text_encoder_folders=(str(tmp_path / "models" / "text_encoders"),))]
    lm.QwenLM().generate("Qwen3.5-9B", "BF16", "None", 1.0, False, 8, 0, True, "", "hi", "")
    assert seen[-1].config == {} and seen[-1].image is None


# --- LM Loader -----------------------------------------------------------------------------------------------

LOADER_REQUIRED = ["family", "model", "precision", "lora", "lora_strength", "keep_model_loaded"]
OTHER_CATALOG = """models:
  Other-1B:
    template: qwen3.5
    thinking: false
    mtp: false
    defaults:
      thinking_off: {do_sample: true, temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: auto}
    precisions:
      FP8: {file: Other-1B_fp8.safetensors, url: "https://example.org/Other-1B_fp8.safetensors"}
"""


@pytest.fixture
def other(bcnodes, monkeypatch, tmp_path):
    """A second LM family, other_lm named Other LM: the Qwen LM family's templates over a catalog of one model
    (Other-1B, FP8), its folder models/Other-LM holding one LoRA."""
    registry = bcnodes["models.common.registry"]
    catalog = tmp_path / "other_models.yaml"
    catalog.write_text(OTHER_CATALOG, encoding="utf-8")
    family = dataclasses.replace(bcnodes["models.qwen_lm"].FAMILY, key="other_lm", name="Other LM",
                                 folder="Other-LM", catalog_path=str(catalog))
    monkeypatch.setitem(registry._FAMILIES[registry.LM_FAMILY], "other_lm", family)
    loras = tmp_path / "models" / "Other-LM" / "loras"
    loras.mkdir(parents=True)
    (loras / "other_style.safetensors").write_bytes(b"")
    return family


def test_loader_inputs(lm, fp, bcnodes):
    spec = lm.LMLoader.INPUT_TYPES()
    req = spec["required"]
    assert list(spec) == ["required"] and list(req) == LOADER_REQUIRED
    registry = bcnodes["models.common.registry"]
    # the families' names, in registry order; the registry key stays internal
    names = [registry.get("lm_family", key).name for key in registry.names("lm_family")]
    assert req["family"][0] == names == ["Qwen LM"]
    qwen_name = registry.get("lm_family", lm.QwenLM.FAMILY).name
    assert req["family"][1]["default"] == lm.MODEL_DEFAULTS["family"] == qwen_name
    qwen = lm.QwenLM.INPUT_TYPES()["required"]
    for name in ("model", "precision", "lora_strength"):
        assert req[name] == qwen[name], name  # Qwen LM's widgets: one spec, one tooltip
    assert req["lora"][0] == qwen["lora"][0] and req["lora"][1]["default"] == qwen["lora"][1]["default"]
    assert req["lora"][1]["tooltip"] == qwen["lora"][1]["tooltip"] + " Only the selected family's LoRAs are shown."
    keep = req["keep_model_loaded"]
    assert keep[0] == "BOOLEAN" and keep[1]["default"] is qwen["keep_model_loaded"][1]["default"] is True
    assert all(v[1].get("tooltip") for v in req.values())


def test_one_place_for_the_defaults(lm, fp, bcnodes):
    """Qwen LM's and LM Loader's initial values are nodes/lm.MODEL_DEFAULTS (family: LM Loader's only, the Qwen
    LM family's name); a wrapper node has none."""
    assert lm.MODEL_DEFAULTS == {"family": "Qwen LM", "model": "Qwen3.5-9B", "precision": "INT8 ConvRot",
                                 "lora": "None", "lora_strength": 1.0, "keep_model_loaded": True}
    assert lm.MODEL_DEFAULTS["lora"] == bcnodes["pipelines.lm"].NO_LORA
    assert lm.MODEL_DEFAULTS["family"] == bcnodes["models.qwen_lm"].FAMILY.name
    loader = lm.LMLoader.INPUT_TYPES()["required"]
    assert {name: loader[name][1]["default"] for name in lm.MODEL_DEFAULTS} == lm.MODEL_DEFAULTS
    qwen = lm.QwenLM.INPUT_TYPES()["required"]
    assert "family" not in qwen
    assert {name: qwen[name][1]["default"] for name in lm.MODEL_DEFAULTS if name != "family"} == {
        name: value for name, value in lm.MODEL_DEFAULTS.items() if name != "family"}
    assert not hasattr(bcnodes["pipelines.lm"], "DEFAULT_LM")
    assert all("default" not in v[1] for v in lm.Captioner.INPUT_TYPES()["required"].values() if v[0] == "BC_LM")


def test_loader_lists_every_family(lm, fp, other, tmp_path):
    req = lm.LMLoader.INPUT_TYPES()["required"]
    assert req["family"][0] == ["Qwen LM", "Other LM"]
    assert req["model"][0] == MODELS + ["Other-1B"]
    assert req["precision"][0] == ["BF16", "INT8 ConvRot", "W4A8", "FP8"]
    assert req["lora"][0] == ["None", "other_style.safetensors"]
    # Qwen LM keeps its own family's lists
    qwen = lm.QwenLM.INPUT_TYPES()["required"]
    assert qwen["model"][0] == MODELS and qwen["lora"][0] == ["None"]


def test_loader_output(lm, fp, bcnodes):
    assert (lm.LMLoader.RETURN_TYPES, lm.LMLoader.RETURN_NAMES, lm.LMLoader.FUNCTION) == (("BC_LM",), ("lm",), "load")
    assert lm.LMLoader.CATEGORY == "BCNodes/lm" and not getattr(lm.LMLoader, "OUTPUT_NODE", False)
    assert lm.LMLoader.DESCRIPTION and lm.LMLoader.SEARCH_ALIASES[0] == "BCNodes" and len(lm.LMLoader.OUTPUT_TOOLTIPS) == 1
    pipe = bcnodes["pipelines.lm"]
    out = lm.LMLoader().load("Qwen LM", "Qwen3.8-27B", "W4A8", "None", 0.5, False)
    assert out == (pipe.LMSpec(family="qwen_lm", model="Qwen3.8-27B", precision="W4A8", lora="None",
                               lora_strength=0.5, keep_model_loaded=False),)
    assert list(pipe.LMSpec.__dataclass_fields__) == ["family", "model", "precision", "lora", "lora_strength",
                                                       "keep_model_loaded"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        out[0].model = "Qwen3.5-4B"


def test_loader_loads_nothing(lm, fp, bcnodes, monkeypatch, other):
    """No file is looked for or downloaded, no LoRA read, no backend called: the model file is not on disk."""
    pipe = bcnodes["pipelines.lm"]
    registry = bcnodes["models.common.registry"]
    for name in ("model_path", "lora_spec", "read_lora", "fetch_with_progress"):
        monkeypatch.setattr(pipe, name, lambda *a, _name=name, **k: pytest.fail(f"LM Loader called {_name}"))

    class Untouchable:
        name = "core"

        def __getattr__(self, attr):
            pytest.fail(f"LM Loader used the backend ({attr})")

    monkeypatch.setitem(registry._FAMILIES[registry.LM_BACKEND], "core", Untouchable())
    spec = lm.LMLoader().load("Other LM", "Other-1B", "FP8", "other_style.safetensors", 1.0, True)[0]
    assert (spec.family, spec.model, spec.precision, spec.lora) == ("other_lm", "Other-1B", "FP8",
                                                                     "other_style.safetensors")


@pytest.mark.parametrize("args, match", [
    (("Gemma LM", "Qwen3.5-9B", "INT8 ConvRot", "None"),
     "LM family 'Gemma LM' is not registered; pick one of: Qwen LM, Other LM$"),
    # a registry key is not a family's name
    (("qwen_lm", "Qwen3.5-9B", "INT8 ConvRot", "None"),
     "LM family 'qwen_lm' is not registered; pick one of: Qwen LM, Other LM$"),
    (("Other LM", "Qwen3.5-9B", "INT8 ConvRot", "None"),
     "model 'Qwen3.5-9B' is not in the Other-LM catalog; the models are: Other-1B$"),
    (("Qwen LM", "Other-1B", "FP8", "None"), "model 'Other-1B' is not in the Qwen-LM catalog; the models are: Qwen3.5-4B, "),
    (("Qwen LM", "Qwen3.5-9B", "W4A8", "None"), "Qwen3.5-9B has no precision 'W4A8'; its precisions are: BF16, INT8 ConvRot$"),
    (("Other LM", "Other-1B", "BF16", "None"), "Other-1B has no precision 'BF16'; its precisions are: FP8$"),
    (("Qwen LM", "Qwen3.5-9B", "BF16", "other_style.safetensors"),
     r"LoRA 'other_style.safetensors' is not in the LoRA folders of the Qwen LM family \(.*Qwen-LM.loras\); pick one "
     r"of: None$"),
])
def test_loader_errors_say_what_to_pick(lm, fp, other, args, match):
    with pytest.raises(ValueError, match=match):
        lm.LMLoader().load(*args, 1.0, True)


def test_family_names_map_to_keys(lm, fp, bcnodes, other):
    """pipelines/lm.family_keys: {name: key} in registry order; LM Loader's name maps to the key its spec holds."""
    pipe = bcnodes["pipelines.lm"]
    assert pipe.family_keys() == {"Qwen LM": "qwen_lm", "Other LM": "other_lm"}
    assert (pipe.family_key_of("Qwen LM"), pipe.family_key_of("Other LM")) == ("qwen_lm", "other_lm")
    assert lm.LMLoader().load("Other LM", "Other-1B", "FP8", "None", 1.0, True)[0].family == "other_lm"


def test_two_families_with_one_name_are_refused(lm, fp, bcnodes, monkeypatch):
    registry = bcnodes["models.common.registry"]
    copy = dataclasses.replace(bcnodes["models.qwen_lm"].FAMILY, key="copy_lm")  # keeps the name Qwen LM
    monkeypatch.setitem(registry._FAMILIES[registry.LM_FAMILY], "copy_lm", copy)
    match = "LM families 'qwen_lm' and 'copy_lm' both have the name 'Qwen LM'; give each family its own name"
    with pytest.raises(ValueError, match=match):
        lm.LMLoader.INPUT_TYPES()
    with pytest.raises(ValueError, match=match):
        lm.LMLoader().load("Qwen LM", "Qwen3.5-9B", "INT8 ConvRot", "None", 1.0, True)


def test_an_unknown_family_key_names_the_families(lm, fp, bcnodes, other):
    """A key no family is registered under (a node's code, never a widget) names the families by name."""
    with pytest.raises(ValueError, match="^no LM family is registered as 'gemma_lm'; the families are: Qwen LM, "
                                         "Other LM$"):
        lm.family_folders("gemma_lm")


def test_every_family_has_its_node(lm, bcnodes):
    """The catalog route lists FAMILY_NODES' families: every family LM Loader offers (the registered ones)."""
    registry = bcnodes["models.common.registry"]
    assert sorted(lm.FAMILY_NODES.values()) == sorted(registry.names(registry.LM_FAMILY))


# --- the catalog route ---------------------------------------------------------------------------------------


def test_catalog_payload(lm, fp, bcnodes, tmp_path):
    os.makedirs(os.path.join(qwen_folder(tmp_path), "loras", "sub"))
    for rel in ("b.safetensors", os.path.join("sub", "a.safetensors")):
        open(os.path.join(qwen_folder(tmp_path), "loras", rel), "wb").close()
    payload = lm.catalog_payload()
    assert json.loads(json.dumps(payload)) == payload
    assert list(payload) == ["families", "config_node", "loader_node"] and payload["config_node"] == "BC_LMConfig"
    assert payload["loader_node"] == "BC_LMLoader" and lm.NODE_CLASS_MAPPINGS["BC_LMLoader"] is lm.LMLoader
    pipe = bcnodes["pipelines.lm"]
    assert list(payload) == list(pipe.CatalogPayload.__annotations__)  # the TypedDict, keys in its order
    family = payload["families"]["qwen_lm"]
    assert list(payload["families"]) == ["qwen_lm"] and family["node"] == "BC_QwenLM" and family["error"] is None
    assert family["name"] == "Qwen LM"  # what LM Loader's family widget holds; web/js/lm.js finds the family by it
    assert list(family) == list(pipe.FamilyInfo.__annotations__)
    assert list(family["models"]) == MODELS
    assert family["loras"] == ["None", "b.safetensors", os.path.join("sub", "a.safetensors")]  # the lora widget's list
    assert family["models"]["Qwen3.8-27B"]["precisions"] == ["BF16", "INT8 ConvRot", "W4A8"]
    assert family["models"]["Qwen3.5-9B"] == {
        "thinking": True, "mtp": True, "precisions": ["BF16", "INT8 ConvRot"],
        "defaults": {"thinking_off": {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                                      "presence_penalty": 1.5, "repetition_penalty": 1.0, "mtp": "auto"},
                     "thinking_on": {"do_sample": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                     "presence_penalty": 1.5, "repetition_penalty": 1.0, "mtp": "auto"}}}


def test_catalog_payload_names_every_family(lm, fp, other, monkeypatch):
    monkeypatch.setitem(lm.FAMILY_NODES, "BC_OtherLM", "other_lm")  # the second family's node
    families = lm.catalog_payload()["families"]
    assert {key: (family["name"], family["node"]) for key, family in families.items()} == {
        "qwen_lm": ("Qwen LM", "BC_QwenLM"), "other_lm": ("Other LM", "BC_OtherLM")}
    assert families["other_lm"]["loras"] == ["None", "other_style.safetensors"]
    assert [family["name"] for family in families.values()] == lm.LMLoader.INPUT_TYPES()["required"]["family"][0]


def test_catalog_payload_with_a_broken_user_catalog(lm, fp, tmp_path):
    os.makedirs(qwen_folder(tmp_path))
    with open(os.path.join(qwen_folder(tmp_path), "models.yaml"), "w") as f:
        f.write("models:\n  Qwen3.5-9B:\n    thinking: maybe\n")
    family = lm.catalog_payload()["families"]["qwen_lm"]
    assert family["error"].endswith("models.Qwen3.5-9B: thinking must be true or false") and list(family["models"]) == MODELS


def test_routes_need_a_server(lm, monkeypatch):
    monkeypatch.setitem(sys.modules, "server", None)  # `from server import ...` raises ImportError
    assert lm.register_routes() is None
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace()  # no instance: not a running server
    monkeypatch.setitem(sys.modules, "server", server)
    assert lm.register_routes() is None


def test_catalog_route(lm, fp, monkeypatch):
    web = pytest.importorskip("aiohttp.web")
    routes = web.RouteTableDef()
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
    monkeypatch.setitem(sys.modules, "server", server)
    lm.register_routes()
    [route] = list(routes)
    assert (route.method, route.path) == ("GET", "/bcnodes/lm/catalog")
    response = asyncio.run(route.handler(None))
    assert response.status == 200 and json.loads(response.body) == lm.catalog_payload()

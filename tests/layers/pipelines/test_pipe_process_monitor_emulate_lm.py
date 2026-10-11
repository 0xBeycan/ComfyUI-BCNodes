"""Emulate's profiles of the LM nodes (pipelines/process_monitor/profiles.py); the graph helpers are the
emulate test's.

  - Qwen LM: not counted. Its model widget holds a catalog name, not a file name, so Emulate reads no
    weights for it, and nothing in the prompt sizes the model, the LoRA, the KV cache core reserves for
    prompt + max_new_tokens or the resized images; its two string outputs alone would read as "counted",
    0 bytes. What links into it (an image, LM Config) keeps its own estimate;
  - LM Config: counted, no tensor: its output is a dict of the edited sampling fields;
  - LM Loader: counted, no tensor: its output is a description of the model (pipelines/lm.LMSpec), nothing
    loaded;
  - Captioner: not counted, as Qwen LM (its LM Loader's model is a catalog name), with its 600-token KV cache
    and the 1024 cap named; its outputs are lists (views of the input frames, strings), which Emulate does not
    follow. The image it reads keeps its own estimate.
"""

import pytest

import test_pipe_process_monitor_emulate as emulate
from test_pipe_process_monitor_emulate import N, rows_of


class Env(emulate.Env):
    TYPES = {**emulate.Env.TYPES, "BC_QwenLM": ["STRING", "STRING"], "BC_LMConfig": ["BC_LM_CONFIG"],
             "BC_LMLoader": ["BC_LM"], "BC_Captioner": ["IMAGE", "STRING"], "PreviewAny": []}


@pytest.fixture
def em(bcnodes):
    return bcnodes["pipelines.process_monitor.emulate"]


def config(edited="temperature"):
    return N("BC_LMConfig", do_sample=True, temperature=0.6, top_k=20, top_p=0.8, min_p=0.0, repetition_penalty=1.0,
             presence_penalty=1.5, mtp="auto", edited=edited)


def qwen(**inputs):
    widgets = dict(model="Qwen3.5-9B", precision="INT8 ConvRot", lora="my_lora.safetensors", lora_strength=1.0,
                   thinking=False, max_new_tokens=32768, seed=0, keep_model_loaded=True, system="", user="Describe this.",
                   assistant="")
    return N("BC_QwenLM", **dict(widgets, **inputs))


def test_qwen_lm_is_not_counted_and_lm_config_is_counted(em):
    p = {"9": N("LoadImage", image="ref.png"), "1": config(), "2": qwen(image=["9", 0], config=["1", 0]),
         "3": N("PreviewAny", source=["2", 0])}
    r = em.estimate(p, Env())
    rows = {row["id"]: row for row in r["rows"]}
    lm = rows["2"]
    assert lm["status"] == "not counted" and lm["transient"] is None and lm["outputs"] == [] and lm["output_bytes"] == 0
    assert lm["note"].startswith("the model widget names a catalog entry, not a file")
    for part in ("model file's weights", "the LoRA", "KV cache core reserves for prompt + max_new_tokens", "images resized"):
        assert part in lm["note"], part
    # no model file named: no weights, no VRAM; the LoRA's name is looked up as any file name and not found here
    assert lm["weights"] == [] and lm["vram_need"] == 0 and r["weights_total"] == 0
    cfg = rows["1"]
    assert (cfg["status"], cfg["outputs"], cfg["output_bytes"], cfg["transient"]) == ("counted", [], 0, 0)
    assert "edited sampling fields" in cfg["note"]
    # the image it reads keeps its estimate; the node's output (text) adds nothing downstream
    assert rows["9"]["output_bytes"] == 480 * 640 * 3 * 4 + 480 * 640 * 4
    assert rows["3"]["status"] == "counted" and rows["3"]["output_bytes"] == 0
    assert r["not_counted"] == ["2"]


def test_qwen_lm_is_not_counted_without_inputs_too(em):
    # text only, no LM Config, no LoRA: still nothing in the prompt sizes the model or the KV cache
    row = rows_of(em, {"1": qwen(lora="None")}, Env())["1"]
    assert row["status"] == "not counted" and "catalog entry, not a file" in row["note"]


def test_lm_loader_is_counted_and_captioner_is_not(em):
    p = {"9": N("LoadImage", image="ref.png"),
         "3": N("BC_LMLoader", family="Qwen LM", model="Qwen3.5-9B", precision="INT8 ConvRot",
                lora="my_lora.safetensors", lora_strength=1.0, keep_model_loaded=True),
         "1": N("BC_Captioner", image=["9", 0], lm=["3", 0], style="tags", seed=0),
         "2": N("PreviewAny", source=["1", 1])}
    r = em.estimate(p, Env())
    rows = {row["id"]: row for row in r["rows"]}
    loader = rows["3"]
    assert (loader["status"], loader["outputs"], loader["output_bytes"], loader["transient"]) == ("counted", [], 0, 0)
    assert loader["note"].startswith("a description of the model") and "nothing loaded" in loader["note"]
    assert loader["weights"] == [] and loader["vram_need"] == 0
    row = rows["1"]
    assert row["status"] == "not counted" and row["transient"] is None and row["outputs"] == [] and row["output_bytes"] == 0
    assert row["note"].startswith("the LM Loader's model is a catalog entry, not a file")
    for part in ("model file's weights", "the LoRA", "KV cache core reserves for prompt + 600 tokens",
                 "over 1024 on a side", "lists of views of the input frames"):
        assert part in row["note"], part
    assert row["weights"] == [] and row["vram_need"] == 0
    assert rows["9"]["output_bytes"] == 480 * 640 * 3 * 4 + 480 * 640 * 4
    assert r["not_counted"] == ["1"]

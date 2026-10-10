"""Emulate's profiles of the LM nodes (pipelines/process_monitor/profiles.py); the graph helpers are the
emulate test's.

  - Qwen LM: not counted. Its model widget holds a catalog name, not a file name, so Emulate reads no
    weights for it, and nothing in the prompt sizes the model, the LoRA, the KV cache core reserves for
    prompt + max_new_tokens or the resized images; its two string outputs alone would read as "counted",
    0 bytes. What links into it (an image, LM Config) keeps its own estimate;
  - LM Config: counted, no tensor: its output is a dict of the edited sampling fields.
"""

import pytest

import test_pipe_process_monitor_emulate as emulate
from test_pipe_process_monitor_emulate import N, rows_of


class Env(emulate.Env):
    TYPES = {**emulate.Env.TYPES, "BC_QwenLM": ["STRING", "STRING"], "BC_LMConfig": ["BC_LM_CONFIG"],
             "PreviewAny": []}


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

"""libs/lm_lora.py: LoRA files (PEFT folders and single .safetensors files) turned into the tensors
ComfyUI core applies to a CLIP, the alpha resolved from the file, its metadata or its config (a PEFT
LoRA's never guessed; core's names without one take alpha = rank, core's convention), the base-model
check by key map and shapes, and the LoRA folder listing.

Fixtures are tiny synthetic tensors written to tmp_path. The PEFT key forms are the ones PEFT's
save_pretrained writes (`base_model.model.<HF module>.lora_A.weight`, `lora_embedding_A` for an
nn.Embedding, `base_layer.weight` next to a LoRA on embed_tokens / lm_head; checked against a real
PEFT 0.21.2 save of transformers' Qwen3.5, whose config file PEFT_DEFAULTS reproduces). The deltas are
checked against core's LoRA math written out (comfy/weight_adapter/lora.py: up @ down * alpha / rank,
rank = lora_A's first dimension) and PEFT's (Linear: B @ A * scaling, Embedding: (B @ A).T * scaling,
scaling = alpha / r, or alpha / sqrt(r) with use_rslora).
"""

import json
import math
import os
import re

import pytest
import torch

H, V, R = 6, 10, 2  # hidden size, vocab, rank
P = "base_model.model."
CORE = "text_encoders.transformer.model."
# adapter_config.json as PEFT 0.21.2 writes it for LoraConfig(r=16, lora_alpha=32): every feature at its default
PEFT_DEFAULTS = {
    "alora_invocation_tokens": None, "alpha_pattern": {}, "arrow_config": None, "auto_mapping": None,
    "base_model_name_or_path": None, "bias": "none", "corda_config": None, "ensure_weight_tying": False,
    "eva_config": None, "exclude_modules": None, "fan_in_fan_out": False, "inference_mode": False,
    "init_lora_weights": True, "kasa_config": None, "layer_replication": None, "layers_pattern": None,
    "layers_to_transform": None, "loftq_config": {}, "lora_alpha": 32, "lora_bias": False, "lora_dropout": 0.0,
    "lora_ga_config": None, "megatron_config": None, "megatron_core": "megatron.core", "modules_to_save": None,
    "monteclora_config": None, "peft_type": "LORA", "peft_version": "0.21.2", "qalora_group_size": 16, "r": 16,
    "rank_pattern": {}, "revision": None, "target_modules": ["q_proj"], "target_parameters": None, "task_type": None,
    "trainable_token_indices": None, "use_bdlora": None, "use_dora": False, "use_qalora": False, "use_rslora": False,
    "velora_config": None}


@pytest.fixture
def lm(bcnodes):
    return bcnodes["libs.lm_lora"]


def save(path, tensors, metadata=None):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path), metadata=metadata)
    return str(path)


def peft_folder(tmp_path, tensors, config, name="adapter"):
    folder = tmp_path / name
    folder.mkdir()
    save(folder / "adapter_model.safetensors", tensors)
    if config is not None:
        (folder / "adapter_config.json").write_text(json.dumps(config))
    return str(folder)


def pair(module, rank=R, out=H, inp=H):
    g = torch.Generator().manual_seed(len(module))
    return {f"{module}.lora_A.weight": torch.randn(rank, inp, generator=g),
            f"{module}.lora_B.weight": torch.randn(out, rank, generator=g)}


def core_delta(plan, module):
    up, down = plan.tensors[f"{module}.lora_B.weight"], plan.tensors[f"{module}.lora_A.weight"]
    return plan.tensors[f"{module}.alpha"].item() / down.shape[0] * (up @ down)


def test_peft_causal_lm_folder(lm, tmp_path):
    g = torch.Generator().manual_seed(0)
    emb_a, emb_b = torch.randn(R, V, generator=g), torch.randn(H, R, generator=g)
    tensors = {**pair(P + "model.layers.0.self_attn.q_proj"), **pair(P + "lm_head", out=V),
               P + "model.embed_tokens.lora_embedding_A": emb_a, P + "model.embed_tokens.lora_embedding_B": emb_b,
               P + "model.embed_tokens.base_layer.weight": torch.zeros(V, H), P + "lm_head.base_layer.weight": torch.zeros(V, H)}
    source = lm.read_lora(peft_folder(tmp_path, tensors, {"r": R, "lora_alpha": 8, "peft_type": "LORA"}))
    # the full base weights are never read (an embedding's is 1 G values on a 9B model)
    unread = [k for k, v in source.tensors.items() if v is None]
    assert unread == sorted(k for k in tensors if k.endswith("base_layer.weight"))
    plan = lm.plan_lora(source)
    assert plan.format == "peft" and plan.alpha_source == "adapter_config.json"
    assert set(plan.modules) == {CORE + "layers.0.self_attn.q_proj", CORE + "lm_head", CORE + "embed_tokens"}
    assert sorted(plan.tensors) == sorted(f"{m}.{s}" for m in plan.modules for s in ("lora_A.weight", "lora_B.weight", "alpha"))
    for m in plan.modules:
        assert plan.tensors[f"{m}.alpha"].dtype == torch.float32 and plan.tensors[f"{m}.alpha"].item() == 8.0
    scaling = 8 / R
    q = P + "model.layers.0.self_attn.q_proj"
    assert torch.allclose(core_delta(plan, CORE + "layers.0.self_attn.q_proj"),
                          scaling * tensors[q + ".lora_B.weight"] @ tensors[q + ".lora_A.weight"])
    assert torch.allclose(core_delta(plan, CORE + "embed_tokens"), scaling * (emb_b @ emb_a).T)
    assert core_delta(plan, CORE + "embed_tokens").shape == (V, H)
    assert any("left out: 2 full base weights" in n for n in plan.notes)
    assert "rank 2, alpha 8 from adapter_config.json (scale alpha / rank 4)" in plan.notes[0]


def test_peft_conditional_generation_names_and_the_default_adapter_suffix(lm, tmp_path):
    lang, vis = P + "model.language_model.layers.1.mlp.down_proj", P + "model.visual.blocks.0.attn.qkv"
    tensors = {**pair(lang), **{k.replace(".weight", ".default.weight"): v for k, v in pair(vis).items()}}
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, {"r": R, "lora_alpha": R})))
    assert set(plan.modules) == {CORE + "layers.1.mlp.down_proj", "text_encoders.transformer.visual.blocks.0.attn.qkv"}
    assert torch.equal(plan.tensors["text_encoders.transformer.visual.blocks.0.attn.qkv.lora_A.weight"],
                       tensors[vis + ".lora_A.default.weight"])


def test_peft_language_model_embed_tokens_with_the_adapter_name_kept(lm, tmp_path):
    # a PeftModel's raw state dict keeps the adapter name; an embedding's factors are Parameters (no .weight)
    g = torch.Generator().manual_seed(1)
    emb_a, emb_b = torch.randn(R, V, generator=g), torch.randn(H, R, generator=g)
    tensors = {P + "model.language_model.embed_tokens.lora_embedding_A.default": emb_a,
               P + "model.language_model.embed_tokens.lora_embedding_B.default": emb_b}
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, {"r": R, "lora_alpha": 4})))
    assert plan.modules == (CORE + "embed_tokens",)
    assert torch.allclose(core_delta(plan, CORE + "embed_tokens"), 4 / R * (emb_b @ emb_a).T)


def test_a_new_format_is_one_entry(lm, tmp_path, monkeypatch):
    # e.g. PEFT's keys without the base_model.model. wrapper
    entry = lm.LoraFormat("bare", marks=("model.",), modules=(("model.", CORE),),
                          tensors=(("lora_A.weight", lm.DOWN, False), ("lora_B.weight", lm.UP, False)))
    monkeypatch.setattr(lm, "FORMATS", lm.FORMATS + (entry,))
    path = save(tmp_path / "bare.safetensors", pair("model.layers.0.mlp.up_proj"), {"lora_alpha": "4"})
    plan = lm.plan_lora(lm.read_lora(path))
    assert (plan.format, plan.modules, plan.alpha_source) == ("bare", (CORE + "layers.0.mlp.up_proj",), "metadata")


def test_single_file_with_its_stem_json(lm, tmp_path):
    path = save(tmp_path / "style.safetensors", pair(P + "model.layers.0.mlp.up_proj"))
    (tmp_path / "style.json").write_text(json.dumps({"lora_alpha": 4, "r": R}))
    source = lm.read_lora(path)
    assert source.config_path == str(tmp_path / "style.json")
    plan = lm.plan_lora(source)
    assert plan.alpha_source == "style.json" and plan.tensors[CORE + "layers.0.mlp.up_proj.alpha"].item() == 4.0


def test_an_adapter_model_file_reads_the_adapter_config_beside_it(lm, tmp_path):
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), {"r": R, "lora_alpha": 16})
    plan = lm.plan_lora(lm.read_lora(folder + "/adapter_model.safetensors"))
    assert plan.alpha_source == "adapter_config.json"
    assert plan.tensors[CORE + "layers.0.mlp.up_proj.alpha"].item() == 16.0


@pytest.mark.parametrize("metadata", [
    {"format": "pt", "lora_alpha": "12", "r": "2"},
    {"format": "pt", "peft_config": json.dumps({"r": 2, "lora_alpha": 12, "use_rslora": False})},
    {"lora_alpha": "12", "peft_config": json.dumps({"r": 2, "lora_alpha": 12})},  # both, agreeing
])
def test_alpha_from_the_metadata(lm, tmp_path, metadata):
    path = save(tmp_path / "m.safetensors", pair(P + "model.layers.0.mlp.up_proj"), metadata)
    plan = lm.plan_lora(lm.read_lora(path))
    assert plan.alpha_source == "metadata" and plan.tensors[CORE + "layers.0.mlp.up_proj.alpha"].item() == 12.0


def test_metadata_configs_that_disagree_are_an_error(lm, tmp_path):
    path = save(tmp_path / "m.safetensors", pair(P + "model.layers.0.mlp.up_proj"),
                {"lora_alpha": "12", "peft_config": json.dumps({"lora_alpha": 16})})
    with pytest.raises(lm.LoraError, match="lora_alpha 12.0 and peft_config gives 16; keep one"):
        lm.plan_lora(lm.read_lora(path))


def test_alpha_sources_in_order_tensors_then_metadata_then_config(lm, tmp_path):
    m = CORE + "layers.0.mlp.up_proj"
    path = save(tmp_path / "t.safetensors", {**pair(m), m + ".alpha": torch.tensor(1.0)}, {"lora_alpha": "12"})
    (tmp_path / "t.json").write_text(json.dumps({"lora_alpha": 99}))
    plan = lm.plan_lora(lm.read_lora(path))
    assert (plan.alpha_source, plan.tensors[m + ".alpha"].item()) == ("tensors", 1.0)
    path = save(tmp_path / "m.safetensors", pair(P + "model.layers.0.mlp.up_proj"), {"lora_alpha": "12"})
    (tmp_path / "m.json").write_text(json.dumps({"lora_alpha": 99}))
    plan = lm.plan_lora(lm.read_lora(path))
    assert (plan.alpha_source, plan.tensors[m + ".alpha"].item()) == ("metadata", 12.0)


def test_comfy_names_pass_through_and_alpha_tensors_come_first(lm, tmp_path):
    a, b = CORE + "layers.0.self_attn.k_proj", "lora_te_layers_1_mlp_gate_proj"
    tensors = {a + ".lora_down.weight": torch.ones(R, H), a + ".lora_up.weight": torch.ones(H, R),
               a + ".alpha": torch.tensor(1.0, dtype=torch.bfloat16), **pair(b), b + ".alpha": torch.tensor(3.0)}
    path = save(tmp_path / "c.safetensors", tensors)
    (tmp_path / "c.json").write_text(json.dumps({"lora_alpha": 99}))  # not used: the file carries alphas
    source = lm.read_lora(path)
    plan = lm.plan_lora(source)
    assert (plan.format, plan.alpha_source, set(plan.modules)) == ("comfy", "tensors", {a, b})
    assert plan.tensors[a + ".lora_A.weight"] is source.tensors[a + ".lora_down.weight"]
    assert plan.tensors[a + ".alpha"].item() == 1.0 and plan.tensors[b + ".alpha"].item() == 3.0
    assert plan.tensors[a + ".alpha"].dtype == torch.float32


def test_no_alpha_says_how_to_fix_it(lm, tmp_path):
    path = save(tmp_path / "bare.safetensors", pair(P + "model.layers.0.mlp.up_proj"), {"format": "pt"})
    with pytest.raises(lm.LoraError) as e:
        lm.plan_lora(lm.read_lora(path))
    msg = str(e.value)
    # brief: write <stem>.json next to it containing {"lora_alpha": <alpha>, "r": <rank>} (the values used in
    # training), or keep adapter_config.json next to adapter_model.safetensors; the file's rank is filled in
    assert msg.startswith(path + ": no LoRA alpha")
    assert ('Write bare.json next to it containing {"lora_alpha": <alpha>, "r": 2} (the values used in training), '
            "or keep adapter_config.json next to adapter_model.safetensors and pick that folder") in msg
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), None)
    with pytest.raises(lm.LoraError) as e:
        lm.plan_lora(lm.read_lora(folder))
    assert str(e.value) == (f"{folder}: no LoRA alpha: the folder has no adapter_config.json and the file carries none. "
                            "Put the adapter_config.json saved with this adapter next to adapter_model.safetensors, or "
                            'write one containing {"lora_alpha": <alpha>, "r": 2} (the values used in training)')
    (tmp_path / "bare.json").write_text(json.dumps({"description": "a style"}))  # another tool's sidecar
    with pytest.raises(lm.LoraError) as e:
        lm.plan_lora(lm.read_lora(path))
    assert str(e.value) == (f"{tmp_path / 'bare.json'}: no lora_alpha in it; add \"lora_alpha\": <alpha> (the value used "
                            "in training)")


def test_core_names_without_alpha_take_cores_convention(lm, tmp_path):
    # core reads a module without .alpha at scale 1.0 (comfy/lora.py: alpha None; weight_adapter/lora.py: scale 1.0),
    # and its own LoRA extraction (comfy_extras/nodes_lora_extract.py) writes lora_up / lora_down without alpha
    a, b = "text_encoders.q.transformer.model.layers.0.mlp.up_proj", "lora_te_layers_1_mlp_gate_proj"
    tensors = {a + ".lora_up.weight": torch.ones(H, 4), a + ".lora_down.weight": torch.ones(4, H), **pair(b)}
    path = save(tmp_path / "extracted.safetensors", tensors, {"format": "pt"})
    plan = lm.plan_lora(lm.read_lora(path))
    assert (plan.format, plan.alpha_source) == ("comfy", "rank")
    assert plan.tensors[a + ".alpha"].item() == 4.0 and plan.tensors[b + ".alpha"].item() == float(R)
    assert all(plan.tensors[m + ".alpha"].dtype == torch.float32 for m in plan.modules)
    assert torch.equal(core_delta(plan, a), tensors[a + ".lora_up.weight"] @ tensors[a + ".lora_down.weight"])
    assert plan.notes[0] == "comfy LoRA: 2 modules, rank 2..4, alpha 2..4 from rank (scale alpha / rank 1)"
    assert plan.notes[1] == ("no alpha in the file, its metadata or a config file: alpha = rank on every module, core's "
                             "convention for a LoRA without alpha (scale 1.0; core's own LoRA extraction saves none)")
    # a source that gives an alpha still wins; a config file without lora_alpha is still an error
    (tmp_path / "extracted.json").write_text(json.dumps({"lora_alpha": 8}))
    plan = lm.plan_lora(lm.read_lora(path))
    assert (plan.alpha_source, plan.tensors[a + ".alpha"].item()) == ("extracted.json", 8.0)
    (tmp_path / "extracted.json").write_text(json.dumps({"description": "a style"}))
    with pytest.raises(lm.LoraError, match="extracted.json: no lora_alpha in it"):
        lm.plan_lora(lm.read_lora(path))


def test_no_alpha_with_ranks_that_differ_by_module(lm, tmp_path):
    path = save(tmp_path / "mixed.safetensors", {**pair(P + "model.layers.0.mlp.up_proj"),
                                                 **pair(P + "model.layers.0.mlp.down_proj", rank=4)})
    with pytest.raises(lm.LoraError, match=r'Write mixed.json next to it containing \{"lora_alpha": <alpha>\} \(the '
                                           r"value used in training; the ranks differ by module"):
        lm.plan_lora(lm.read_lora(path))


def test_alpha_tensors_on_some_modules_only_are_an_error(lm, tmp_path):
    a = CORE + "layers.0.mlp.up_proj"
    path = save(tmp_path / "p.safetensors", {**pair(a), a + ".alpha": torch.tensor(2.0), **pair(CORE + "layers.1.mlp.up_proj")})
    with pytest.raises(lm.LoraError, match="1 of 2 modules carry an .alpha tensor"):
        lm.plan_lora(lm.read_lora(path))


def test_alpha_and_rank_patterns_follow_peft_matching(lm, tmp_path):
    # PEFT's get_pattern_key: a pattern matches a module name as `(.*\.)?<pattern>$`; the first wins
    q0, q1, down = (P + "model.layers.0.self_attn.q_proj", P + "model.layers.1.self_attn.q_proj",
                    P + "model.layers.0.mlp.down_proj")
    tensors = {**pair(q0), **pair(q1), **pair(down, rank=1)}
    config = {"r": R, "lora_alpha": 8, "rank_pattern": {"down_proj": 1},
              "alpha_pattern": {"proj": 100, "layers.1.self_attn.q_proj": 3, "mlp.down_proj": 5, "down_proj": 9}}
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, config)))
    alpha = {m: plan.tensors[m + ".alpha"].item() for m in plan.modules}
    # "proj" never matches "q_proj" (no dot before it); "mlp.down_proj" comes before "down_proj"
    assert alpha == {CORE + "layers.0.self_attn.q_proj": 8.0, CORE + "layers.1.self_attn.q_proj": 3.0,
                     CORE + "layers.0.mlp.down_proj": 5.0}
    assert "alpha_pattern: 2 of 3 modules take their own alpha" in plan.notes
    # the whole PEFT module name matches too (the prefix group is optional), and it is matched without base_model.model.
    config["alpha_pattern"] = {"model.layers.0.self_attn.q_proj": 6, "base_model.model.model.layers.1.self_attn.q_proj": 7}
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, config, name="whole")))
    assert plan.tensors[CORE + "layers.0.self_attn.q_proj.alpha"].item() == 6.0
    assert plan.tensors[CORE + "layers.1.self_attn.q_proj.alpha"].item() == 8.0


def test_rslora_scale_is_alpha_over_sqrt_rank(lm, tmp_path):
    m = P + "model.layers.0.mlp.up_proj"
    tensors = pair(m, rank=4)
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, {"r": 4, "lora_alpha": 8, "use_rslora": True})))
    assert plan.tensors[CORE + "layers.0.mlp.up_proj.alpha"].item() == 16.0
    assert torch.allclose(core_delta(plan, CORE + "layers.0.mlp.up_proj"),
                          8 / math.sqrt(4) * tensors[m + ".lora_B.weight"] @ tensors[m + ".lora_A.weight"])


def test_rslora_with_rank_and_alpha_patterns(lm, tmp_path):
    # PEFT: scaling = (alpha_pattern match, else lora_alpha) / sqrt(rank_pattern match, else r)
    up, down = P + "model.layers.0.mlp.up_proj", P + "model.layers.0.mlp.down_proj"
    tensors = {**pair(up, rank=4), **pair(down, rank=9)}
    config = {"r": 4, "lora_alpha": 8, "use_rslora": True,
              "rank_pattern": {"down_proj": 9}, "alpha_pattern": {"down_proj": 6}}
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, config)))
    for module, scaling in ((up, 8 / 2), (down, 6 / 3)):
        assert torch.allclose(core_delta(plan, CORE + module[len(P + "model."):]),
                              scaling * tensors[module + ".lora_B.weight"] @ tensors[module + ".lora_A.weight"])


@pytest.mark.parametrize("field, key", [
    ("alpha_pattern", "up_(proj"),
    ("rank_pattern", "(?i)up_proj"),  # a valid regex alone, an error inside PEFT's (.*\.)?(<key>)$
])
def test_a_pattern_that_is_no_regex_is_named(lm, tmp_path, field, key):
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), {"r": R, "lora_alpha": 8, field: {key: 4}})
    with pytest.raises(lm.LoraError, match=rf"{field} key '{re.escape(key)}' is not a valid pattern"):
        lm.plan_lora(lm.read_lora(folder))


def test_patterns_from_a_metadata_config(lm, tmp_path):
    # a PEFT config held in the metadata carries its patterns and use_rslora like adapter_config.json
    up, down = P + "model.layers.0.mlp.up_proj", P + "model.layers.0.mlp.down_proj"
    config = {"r": 4, "lora_alpha": 8, "use_rslora": True, "alpha_pattern": {"down_proj": 2}}
    path = save(tmp_path / "m.safetensors", {**pair(up, rank=4), **pair(down, rank=4)}, {"config": json.dumps(config)})
    plan = lm.plan_lora(lm.read_lora(path))
    assert plan.alpha_source == "metadata"
    assert {m: plan.tensors[m + ".alpha"].item() for m in plan.modules} == {CORE + "layers.0.mlp.up_proj": 16.0,
                                                                           CORE + "layers.0.mlp.down_proj": 4.0}


@pytest.mark.parametrize("config, message", [
    ({"lora_alpha": "32"}, "lora_alpha is '32', not a number"),
    ({"lora_alpha": True}, "lora_alpha is True, not a number"),
    ({"lora_alpha": float("inf")}, "lora_alpha is inf, not a number"),
    ({"lora_alpha": 8, "r": 2.0}, "r is 2.0, not a positive whole number"),
    ({"lora_alpha": 8, "r": 0}, "r is 0, not a positive whole number"),
    ({"lora_alpha": 8, "use_rslora": "yes"}, "use_rslora is 'yes', not true or false"),
    ({"lora_alpha": 8, "alpha_pattern": ["up_proj"]}, r"alpha_pattern is \['up_proj'\], not an object"),
    ({"lora_alpha": 8, "alpha_pattern": {"up_proj": "4"}}, r"alpha_pattern\['up_proj'\] is '4', not a number"),
])
def test_config_values_that_are_not_what_peft_writes(lm, tmp_path, config, message):
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), config)
    with pytest.raises(lm.LoraError, match=message):
        lm.plan_lora(lm.read_lora(folder))


@pytest.mark.parametrize("metadata, message", [
    ({"lora_alpha": "a lot"}, "metadata lora_alpha is 'a lot', not a number"),
    ({"lora_alpha": "8", "r": "2.5"}, "metadata r is '2.5', not a number"),
])
def test_metadata_values_that_are_no_numbers(lm, tmp_path, metadata, message):
    path = save(tmp_path / "m.safetensors", pair(P + "model.layers.0.mlp.up_proj"), metadata)
    with pytest.raises(lm.LoraError, match=message):
        lm.plan_lora(lm.read_lora(path))


def test_a_config_whose_rank_disagrees_with_the_file_is_another_adapters(lm, tmp_path):
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj", rank=8), {"r": 16, "lora_alpha": 32})
    with pytest.raises(lm.LoraError, match="8 in the file, 16 in the config"):
        lm.plan_lora(lm.read_lora(folder))


@pytest.mark.parametrize("field, value, feature", [
    ("use_dora", True, "DoRA"), ("modules_to_save", ["lm_head"], "modules_to_save"),
    ("trainable_token_indices", [5, 6], "trainable_token_indices"), ("lora_bias", True, "lora_bias"),
    ("fan_in_fan_out", True, "fan_in_fan_out"), ("use_qalora", True, "use_qalora"),
    ("layer_replication", [[0, 2]], "layer_replication"), ("alora_invocation_tokens", [1, 2], "alora_invocation_tokens"),
    ("use_bdlora", {"target_modules_bd_a": ["q_proj"], "nblocks": 2}, "use_bdlora"),
    ("arrow_config", {"top_k": 3}, "arrow_config"), ("kasa_config", {}, "kasa_config"),  # {}: KaSA at its defaults
    ("peft_type", "LOHA", "peft_type is LOHA"),
    # initialisations that rewrite the base weights (PEFT: pissa / corda / astra by prefix, olora, lora_ga, loftq)
    ("init_lora_weights", "pissa", "rewrites the base model's weights"),
    ("init_lora_weights", "pissa_niter_4", "init_lora_weights is 'pissa_niter_4'"),
    ("init_lora_weights", "olora", "olora"), ("init_lora_weights", "corda", "corda"),
    ("init_lora_weights", "astra", "astra"),
    ("init_lora_weights", "lora_ga", "lora_ga"), ("init_lora_weights", "loftq", "LoftQ adapter cannot be converted"),
    # PEFT's off values and plain-LoRA settings are accepted (bias: judged by the tensors, see test_malformed_files)
    ("modules_to_save", [], None), ("trainable_token_indices", {}, None), ("layer_replication", [], None),
    ("alora_invocation_tokens", [], None), ("use_bdlora", False, None),
    ("init_lora_weights", False, None), ("init_lora_weights", "gaussian", None), ("init_lora_weights", "eva", None),
    ("init_lora_weights", "orthogonal", None), ("init_lora_weights", "mica", None), ("peft_type", "lora", None),
    ("bias", "all", None),
])
def test_unsupported_features_are_named(lm, tmp_path, field, value, feature):
    config = {**PEFT_DEFAULTS, "r": R, "lora_alpha": 8, field: value}
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), config)
    if feature is None:
        assert lm.plan_lora(lm.read_lora(folder)).tensors[CORE + "layers.0.mlp.up_proj.alpha"].item() == 8.0
        return
    with pytest.raises(lm.LoraError, match=feature):
        lm.plan_lora(lm.read_lora(folder))


def test_unsupported_features_are_checked_whatever_the_alpha_source(lm, tmp_path):
    path = save(tmp_path / "m.safetensors", pair(P + "model.layers.0.mlp.up_proj"),
                {"peft_config": json.dumps({"lora_alpha": 8, "use_dora": True})})
    with pytest.raises(lm.LoraError, match=r"metadata peft_config: sets DoRA \(use_dora\)"):
        lm.plan_lora(lm.read_lora(path))
    m = CORE + "layers.0.mlp.up_proj"
    path = save(tmp_path / "c.safetensors", {**pair(m), m + ".alpha": torch.tensor(8.0)})
    (tmp_path / "c.json").write_text(json.dumps({"lora_alpha": 8, "fan_in_fan_out": True}))
    with pytest.raises(lm.LoraError, match=r"c.json: sets transposed weights \(fan_in_fan_out\)"):
        lm.plan_lora(lm.read_lora(path))


@pytest.mark.parametrize("tensors, message", [
    ({**pair(P + "model.layers.0.mlp.up_proj"), P + "model.layers.0.mlp.up_proj.lora_magnitude_vector": torch.ones(H)},
     "1 of 3 tensors are not LoRA tensors"),
    # a bias PEFT's bias "lora_only" trained (judged by the tensors: the Qwen LM projections have none)
    ({**pair(P + "model.layers.0.mlp.up_proj"), P + "model.layers.0.mlp.up_proj.base_layer.bias": torch.ones(H)},
     (r"not LoRA tensors this loader reads \(base_model.model.model.layers.0.mlp.up_proj.base_layer.bias\)\. .*biases "
      r"trained with PEFT's bias \"all\" / \"lora_only\".*: use a LoRA trained without them")),
    ({**pair(CORE + "layers.0.mlp.up_proj"), CORE + "layers.0.mlp.up_proj.alpha": torch.ones(2)},
     r"layers.0.mlp.up_proj.alpha holds 2 values, not one; the file is damaged"),
    ({}, "holds no tensors; export the LoRA again"),
    ({**pair(P + "model.layers.0.mlp.up_proj"), **pair(CORE + "layers.1.mlp.up_proj")}, "mixes the comfy and peft layouts"),
    (pair(P + "score"), "module base_model.model.score is not a module of a Qwen LM"),
    ({P + "model.layers.0.mlp.up_proj.lora_A.weight": torch.ones(R, H)}, "has no lora_B"),
    ({P + "model.embed_tokens.base_layer.weight": torch.ones(V, H)}, "has no lora_A and no lora_B"),
    ({**pair(P + "model.layers.0.mlp.up_proj"), P + "model.layers.0.mlp.up_proj.lora_B.weight": torch.ones(H, 3)},
     "do not share a rank"),
    ({**pair(P + "model.layers.0.mlp.up_proj"), **pair(P + "model.language_model.layers.0.mlp.up_proj")},
     "are the same module text_encoders.transformer.model.layers.0.mlp.up_proj"),
    ({**pair(P + "model.layers.0.mlp.up_proj"),
      P + "model.layers.0.mlp.up_proj.lora_A.default.weight": torch.ones(R, H)}, "has its lora_A.weight twice"),
])
def test_malformed_files(lm, tmp_path, tensors, message):
    folder = peft_folder(tmp_path, tensors, {"r": R, "lora_alpha": 8})
    with pytest.raises(lm.LoraError, match=message):
        lm.plan_lora(lm.read_lora(folder))


def test_read_errors(lm, tmp_path):
    with pytest.raises(lm.LoraError, match="LoRA not found"):
        lm.read_lora(str(tmp_path / "nope.safetensors"))
    (tmp_path / "x.ckpt").write_bytes(b"0")
    with pytest.raises(lm.LoraError, match="not a LoRA; pick a .safetensors file or a PEFT folder"):
        lm.read_lora(str(tmp_path / "x.ckpt"))
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "adapter_model.bin").write_bytes(b"0")
    with pytest.raises(lm.LoraError, match="safe_serialization=True"):
        lm.read_lora(str(tmp_path / "old"))
    (tmp_path / "bad.safetensors").write_bytes(b"garbage!garbage")
    with pytest.raises(lm.LoraError, match="not a readable safetensors file"):
        lm.read_lora(str(tmp_path / "bad.safetensors"))
    folder = peft_folder(tmp_path, pair(P + "model.layers.0.mlp.up_proj"), None, name="broken")
    (tmp_path / "broken" / "adapter_config.json").write_text("{lora_alpha: 8")
    with pytest.raises(lm.LoraError, match="adapter_config.json: not valid JSON"):
        lm.read_lora(folder)


def core_key_map(weights):
    """core's key map for one CLIP's weights, as model_lora_keys_clip builds it: each
    `<clip>.transformer.X.weight` under `text_encoders.<clip>.transformer.X`, `text_encoders.transformer.X`
    and, for X = model.Y, `lora_te_<Y, dots as underscores>`."""
    key_map = {}
    for w in weights:
        clip, rest = w[:-len(".weight")].split(".", 1)
        key_map.update({f"text_encoders.{clip}.{rest}": w, f"text_encoders.{rest}": w})
        if rest.startswith("transformer.model."):
            key_map["lora_te_" + rest[len("transformer.model."):].replace(".", "_")] = w
    return key_map


WEIGHTS = {"q.transformer.model.layers.0.self_attn.q_proj.weight": (H, H),
           "q.transformer.model.layers.0.mlp.down_proj.weight": (H, 2 * H),
           "q.transformer.model.embed_tokens.weight": (V, H),
           "q.transformer.model.lm_head.weight": (V, H),  # a head of its own: not tied
           "q.transformer.visual.patch_embed.proj.weight": (4, 3, 2, 2, 2)}  # a Conv3d
KEY_MAP = core_key_map(WEIGHTS)
# core's Qwen3.5-4B: no lm_head weight, the logits come from embed_tokens.weight (tie_word_embeddings)
TIED = {k: v for k, v in WEIGHTS.items() if ".lm_head." not in k}


def plan_of(lm, tmp_path, tensors):
    return lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, {"r": R, "lora_alpha": 8})))


def test_check_base_passes_a_fitting_lora(lm, tmp_path):
    plan = plan_of(lm, tmp_path, {**pair(P + "model.layers.0.self_attn.q_proj"),
                                  **pair(P + "model.layers.0.mlp.down_proj", inp=2 * H),
                                  P + "model.embed_tokens.lora_embedding_A": torch.ones(R, V),
                                  P + "model.embed_tokens.lora_embedding_B": torch.ones(H, R),
                                  # PEFT's Conv3d LoRA: A a conv of the layer's kernel, B a 1x1x1 conv
                                  P + "model.visual.patch_embed.proj.lora_A.weight": torch.ones(R, 3, 2, 2, 2),
                                  P + "model.visual.patch_embed.proj.lora_B.weight": torch.ones(4, R, 1, 1, 1)})
    lm.check_base(plan, KEY_MAP, WEIGHTS)


def test_check_base_refuses_names_or_shapes_of_another_model(lm, tmp_path):
    plan = plan_of(lm, tmp_path, {**pair(P + "model.layers.0.self_attn.q_proj", out=2 * H),
                                  **pair(P + "model.layers.0.mlp.down_proj", inp=2 * H),
                                  **pair(P + "model.layers.7.self_attn.q_proj")})
    with pytest.raises(lm.LoraError) as e:
        lm.check_base(plan, KEY_MAP, WEIGHTS)
    msg = str(e.value)
    assert "1 of 3 modules are not in this model (text_encoders.transformer.model.layers.7.self_attn.q_proj)" in msg
    assert "1 of 3 have another shape (text_encoders.transformer.model.layers.0.self_attn.q_proj: LoRA (12, 6), model (6, 6))" in msg
    assert "pick the model it was trained on" in msg


def test_check_base_refuses_two_modules_on_one_weight(lm, tmp_path):
    path = save(tmp_path / "two.safetensors", {
        **pair("text_encoders.q.transformer.model.layers.0.self_attn.q_proj"), **pair("lora_te_layers_0_self_attn_q_proj"),
        "text_encoders.q.transformer.model.layers.0.self_attn.q_proj.alpha": torch.tensor(1.0),
        "lora_te_layers_0_self_attn_q_proj.alpha": torch.tensor(1.0)})
    with pytest.raises(lm.LoraError, match="both patch q.transformer.model.layers.0.self_attn.q_proj.weight"):
        lm.check_base(lm.plan_lora(lm.read_lora(path)), KEY_MAP, WEIGHTS)


def test_check_base_refuses_embed_tokens_and_lm_head_on_a_tied_model(lm, tmp_path):
    tied = ("this model ties lm_head to embed_tokens in ComfyUI core (it has no lm_head weight: the output head is "
            "embed_tokens.weight, as in Qwen3.5-4B), so a LoRA on embed_tokens / lm_head would patch both ({}): use a "
            "LoRA trained without embed_tokens and lm_head")
    q = pair(P + "model.layers.0.self_attn.q_proj")
    embed = {P + "model.embed_tokens.lora_embedding_A": torch.ones(R, V), P + "model.embed_tokens.lora_embedding_B": torch.ones(H, R)}
    plan = plan_of(lm, tmp_path, {**q, **embed})
    lm.check_base(plan, KEY_MAP, WEIGHTS)  # the same LoRA on a model with a head of its own
    with pytest.raises(lm.LoraError) as e:
        lm.check_base(plan, core_key_map(TIED), TIED)
    assert str(e.value) == tied.format(CORE + "embed_tokens")
    # lm_head has no weight on the tied model: refused for the tie, not as a module of another model
    (tmp_path / "head").mkdir()
    for form in (P + "lm_head", "text_encoders.q.transformer.model.lm_head", CORE + "lm_head", "lora_te_lm_head"):
        path = save(tmp_path / "head" / "h.safetensors", {**pair(form, out=V), form + ".alpha": torch.tensor(1.0)})
        plan = lm.plan_lora(lm.read_lora(path))
        with pytest.raises(lm.LoraError) as e:
            lm.check_base(plan, core_key_map(TIED), TIED)
        assert str(e.value) == tied.format(plan.modules[0]), form
    (tmp_path / "fits").mkdir()
    lm.check_base(plan_of(lm, tmp_path / "fits", q), core_key_map(TIED), TIED)  # neither: it fits
    # another base model's LoRA is reported as that, its head too
    (tmp_path / "other").mkdir()
    plan = plan_of(lm, tmp_path / "other", {**pair(P + "model.layers.0.self_attn.q_proj", out=2 * H), **pair(P + "lm_head", out=V)})
    with pytest.raises(lm.LoraError, match=r"1 of 2 have another shape .* trained on another base model"):
        lm.check_base(plan, core_key_map(TIED), TIED)


def test_list_loras(lm, tmp_path):
    def touch(rel):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    for rel in ("a.safetensors", "b.ckpt", "sub/c.SAFETENSORS",
                "run/adapter_config.json", "run/adapter_model.safetensors", "run/extra.safetensors",
                "run/checkpoint-10/adapter_config.json", "run/checkpoint-10/adapter_model.safetensors",
                "run/logs/x.safetensors", "half/adapter_config.json", "half/y.safetensors",
                "adapter_config.json", "adapter_model.safetensors"):
        touch(rel)
    sep = os.sep
    assert lm.list_loras(str(tmp_path)) == sorted([
        "a.safetensors", "adapter_model.safetensors", f"half{sep}y.safetensors", "run",
        f"run{sep}checkpoint-10", f"sub{sep}c.SAFETENSORS"])
    assert lm.list_loras(str(tmp_path / "missing")) == []


def test_list_loras_ends_on_a_link_loop(lm, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "sub" / "b.safetensors").write_bytes(b"")
    os.symlink(tmp_path, tmp_path / "sub" / "back")  # a link back to the top lists nothing twice
    assert lm.list_loras(str(tmp_path)) == ["a.safetensors", os.path.join("sub", "b.safetensors")]


def qwen_weights(clip, layers, hidden, linear_attention, gated_q):
    """{state-dict key: shape} of a Qwen LM's projections as core names them, at small dims: Qwen3.5
    (`linear_attention`) has a full-attention layer every 4th layer and linear attention in the others,
    and its q_proj carries the output gate (`gated_q`: twice the width); Qwen3-VL has full attention
    in every layer."""
    weights = {}
    for n in range(layers):
        layer = f"{clip}.transformer.model.layers.{n}."
        if linear_attention and n % 4 != 3:
            weights.update({layer + "linear_attn.in_proj_qkv.weight": (2 * hidden, hidden),
                            layer + "linear_attn.in_proj_z.weight": (hidden, hidden),
                            layer + "linear_attn.out_proj.weight": (hidden, hidden)})
        else:
            weights.update({layer + "self_attn.q_proj.weight": ((2 if gated_q else 1) * hidden, hidden),
                            layer + "self_attn.k_proj.weight": (hidden // 2, hidden),
                            layer + "self_attn.v_proj.weight": (hidden // 2, hidden),
                            layer + "self_attn.o_proj.weight": (hidden, hidden)})
        weights.update({layer + "mlp.gate_proj.weight": (3 * hidden, hidden),
                        layer + "mlp.up_proj.weight": (3 * hidden, hidden),
                        layer + "mlp.down_proj.weight": (hidden, 3 * hidden)})
    return weights


# HF module prefix: ...ForConditionalGeneration, ...ForCausalLM
@pytest.mark.parametrize("hf_prefix", ["model.language_model.", "model."])
def test_a_qwen35_9b_lora_fits_its_model_only(lm, tmp_path, hf_prefix):
    # The structure of a real Qwen3.5-9B LoRA (r 16, lora_alpha 32 in adapter_config.json): 24 linear_attn layers x
    # {in_proj_qkv, in_proj_z, out_proj}, 8 full-attention layers (3, 7, ..., 31) x {q, k, v, o}_proj, 32 x MLP.
    # Core would apply 120 of its 200 modules to Qwen3-VL-8B and all 200 names match on Qwen3.5-4B.
    nine = qwen_weights("qwen35_9b", 32, 8, linear_attention=True, gated_q=True)
    g = torch.Generator().manual_seed(0)
    tensors = {}
    for w, (out, inp) in nine.items():
        module = P + hf_prefix + w[len("qwen35_9b.transformer.model."):-len(".weight")]
        tensors[module + ".lora_A.weight"] = torch.randn(16, inp, generator=g)
        tensors[module + ".lora_B.weight"] = torch.randn(out, 16, generator=g)
    plan = lm.plan_lora(lm.read_lora(peft_folder(tmp_path, tensors, PEFT_DEFAULTS)))
    kinds = {}
    for m in plan.modules:
        kind = m.split(".layers.")[1].split(".", 1)[1]
        kinds[kind] = kinds.get(kind, 0) + 1
    assert kinds == {"linear_attn.in_proj_qkv": 24, "linear_attn.in_proj_z": 24, "linear_attn.out_proj": 24,
                     "self_attn.q_proj": 8, "self_attn.k_proj": 8, "self_attn.v_proj": 8, "self_attn.o_proj": 8,
                     "mlp.gate_proj": 32, "mlp.up_proj": 32, "mlp.down_proj": 32}
    assert len(plan.tensors) == 600
    assert sum(k.endswith((".lora_A.weight", ".lora_B.weight")) for k in plan.tensors) == 400
    assert {plan.tensors[m + ".alpha"].item() for m in plan.modules} == {32.0}
    lm.check_base(plan, core_key_map(nine), nine)

    vl = qwen_weights("qwen3vl_8b", 36, 8, linear_attention=False, gated_q=False)
    with pytest.raises(lm.LoraError) as e:
        lm.check_base(plan, core_key_map(vl), vl)
    msg = str(e.value)
    assert "72 of 200 modules are not in this model (text_encoders.transformer.model.layers.0.linear_attn." in msg
    assert ("8 of 200 have another shape (text_encoders.transformer.model.layers.11.self_attn.q_proj: LoRA (16, 8), "
            "model (8, 8)") in msg

    four = qwen_weights("qwen35_4b", 32, 6, linear_attention=True, gated_q=True)
    with pytest.raises(lm.LoraError) as e:
        lm.check_base(plan, core_key_map(four), four)
    assert "200 of 200 have another shape" in str(e.value) and "not in this model" not in str(e.value)

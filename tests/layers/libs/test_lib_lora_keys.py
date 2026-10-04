"""libs/lora_keys.py: the LoRA keys ComfyUI core's loader leaves out, renamed so its key map holds
them; the keys core already maps are never touched, no rename overwrites a tensor, and the keys
still unmapped are listed.

The key map is written out by hand as core builds it for a Wan model (comfy/lora.py
model_lora_keys_unet; Wan has no branch of its own there): each `diffusion_model.<module>.weight`
gives the entries `diffusion_model.<module>` and `lora_unet_<module, dots as underscores>`, each other
parameter (a block's or the head's `modulation`) its own name. That the renamed keys load in core and
the original ones do not is checked against core itself in tests/test_runtime.py.
"""

import pytest

WEIGHTS = [f"blocks.{i}.{m}" for i in range(2) for m in (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2")] + ["head.head", "patch_embedding", "text_embedding.0", "time_embedding.0", "time_projection.1"]
PARAMETERS = ["blocks.0.modulation", "blocks.1.modulation", "head.modulation"]
WAN = ({f"diffusion_model.{m}" for m in WEIGHTS} | {"lora_unet_" + m.replace(".", "_") for m in WEIGHTS}
       | {f"diffusion_model.{p}" for p in PARAMETERS})


@pytest.fixture
def lk(bcnodes):
    return bcnodes["libs.lora_keys"]


def test_lightx2v_diff_m_becomes_the_modulation_diff(lk):
    lora = ["diffusion_model.blocks.0.diff_m", "diffusion_model.blocks.1.diff_m", "diffusion_model.head.diff_m",
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight", "diffusion_model.blocks.0.self_attn.q.lora_up.weight",
            "diffusion_model.blocks.0.self_attn.q.alpha", "diffusion_model.blocks.0.self_attn.q.diff_b"]
    fix = lk.fix_keys(lora, WAN)
    assert fix.renamed == {"diffusion_model.blocks.0.diff_m": "diffusion_model.blocks.0.modulation.diff",
                           "diffusion_model.blocks.1.diff_m": "diffusion_model.blocks.1.modulation.diff",
                           "diffusion_model.head.diff_m": "diffusion_model.head.modulation.diff"}
    assert (fix.unmapped, fix.modulation, fix.prefixed) == ([], 3, 0)


def test_peft_keys_without_the_prefix_get_it(lk):
    lora = ["blocks.0.cross_attn.k.lora_A.default.weight", "blocks.0.cross_attn.k.lora_B.default.weight",
            "blocks.1.ffn.2.lora_A.default.weight", "blocks.1.ffn.2.lora_B.default.weight"]
    fix = lk.fix_keys(lora, WAN)
    assert fix.renamed == {k: "diffusion_model." + k for k in lora}
    assert (fix.unmapped, fix.modulation, fix.prefixed) == ([], 0, 4)


def test_a_diff_m_without_the_prefix_gets_both(lk):
    fix = lk.fix_keys(["blocks.1.diff_m"], WAN)
    assert fix.renamed == {"blocks.1.diff_m": "diffusion_model.blocks.1.modulation.diff"}
    assert (fix.modulation, fix.prefixed) == (1, 1)


def test_keys_core_maps_are_untouched(lk):
    lora = ["lora_unet_blocks_0_self_attn_q.lora_down.weight", "lora_unet_blocks_0_self_attn_q.lora_up.weight",
            "lora_unet_blocks_0_self_attn_q.alpha", "diffusion_model.blocks.1.cross_attn.v.lora_A.weight",
            "diffusion_model.blocks.1.cross_attn.v.lora_B.weight", "diffusion_model.blocks.0.modulation.diff",
            "diffusion_model.head.head.diff_b", "diffusion_model.blocks.0.ffn.0.lora_B.default.weight"]
    fix = lk.fix_keys(lora, WAN)
    assert (fix.renamed, fix.unmapped, fix.modulation, fix.prefixed) == ({}, [], 0, 0)


def test_keys_of_modules_the_model_lacks_are_listed_not_renamed(lk):
    # a Wan 2.1 I2V LoRA on a model without the image branch: k_img is not k, img_emb is absent; a
    # text-encoder key has no module in the diffusion model either
    lora = ["diffusion_model.blocks.0.cross_attn.k_img.lora_A.weight", "diffusion_model.img_emb.proj.1.lora_A.weight",
            "blocks.0.cross_attn.norm_k_img.diff", "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight",
            "diffusion_model.blocks.7.self_attn.q.lora_A.weight", "diffusion_model.blocks.0.diff_x"]
    fix = lk.fix_keys(lora, WAN)
    assert (fix.renamed, fix.unmapped) == ({}, lora)


def test_no_rename_overwrites_a_tensor(lk):
    # the prefixed name is the LoRA's own already
    lora = ["blocks.0.self_attn.q.lora_A.default.weight", "diffusion_model.blocks.0.self_attn.q.lora_A.default.weight"]
    fix = lk.fix_keys(lora, WAN)
    assert (fix.renamed, fix.unmapped) == ({}, ["blocks.0.self_attn.q.lora_A.default.weight"])
    # two keys that would get the same new name: the first one keeps it
    fix = lk.fix_keys(["blocks.0.diff_m", "blocks.0.modulation.diff"], WAN)
    assert (fix.renamed, fix.unmapped) == ({"blocks.0.diff_m": "diffusion_model.blocks.0.modulation.diff"}, ["blocks.0.modulation.diff"])


def test_a_model_that_maps_bare_keys_keeps_them_bare(lk):
    # core maps `<module>` without the prefix for several families (Qwen-Image, LTXV, ...)
    key_map = {"diffusion_model.transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_q"}
    fix = lk.fix_keys(["transformer_blocks.0.attn.to_q.lora_A.weight"], key_map)
    assert (fix.renamed, fix.unmapped) == ({}, [])


@pytest.mark.parametrize("key, entry, addressed", [
    ("diffusion_model.blocks.0.self_attn.q.lora_A.weight", "diffusion_model.blocks.0.self_attn.q", True),
    ("diffusion_model.blocks.0.modulation.diff", "diffusion_model.blocks.0.modulation", True),
    ("lora_unet_blocks_0_ffn_0.hada_w1_a", "lora_unet_blocks_0_ffn_0", True),
    # diffusers' attention-processor LoRA: the entry is followed by `_lora.`
    ("unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q_lora.up.weight",
     "unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.to_q", True),
    # a longer module name is not its prefix's module
    ("diffusion_model.blocks.0.cross_attn.k_img.lora_A.weight", "diffusion_model.blocks.0.cross_attn.k", False),
    ("lora_unet_blocks_0_cross_attn_k_img.lora_down.weight", "lora_unet_blocks_0_cross_attn_k", False),
    # an entry alone is no LoRA tensor
    ("diffusion_model.blocks.0.modulation", "diffusion_model.blocks.0.modulation", False),
])
def test_addressed(lk, key, entry, addressed):
    assert lk.addressed(key, {entry}) is addressed

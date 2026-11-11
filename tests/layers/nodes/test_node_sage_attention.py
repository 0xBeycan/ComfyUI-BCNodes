"""Patch Sage Attention (nodes/sage_attention.py): the output is always a clone and the input model is
never changed; "disabled" and a kernel that cannot run leave the clone unpatched (the latter with a
console warning, no error); a resolved kernel puts a SageOverride on the clone's transformer_options,
chained to the override the model already had, and the two run callbacks on the clone only; a
ComfyUI without the per-model attention override is an error naming the version.

The patcher is a stand-in with ModelPatcher's clone semantics for model_options and callbacks
(comfy.utils.deepcopy_list_dict copies dicts and lists, not their leaves); ComfyUI's attention
module and patcher_extension are stubs.
"""

import logging
import sys
import types

import pytest


class Patcher:
    def __init__(self, transformer_options=None):
        self.load_device = "cuda:0"
        self.model_options = {"transformer_options": dict(transformer_options or {})}
        self.callbacks = {}

    def clone(self):
        n = Patcher()
        n.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
        n.callbacks = {k: {k1: list(c) for k1, c in v.items()} for k, v in self.callbacks.items()}
        return n

    def add_callback(self, call_type, callback):
        self.callbacks.setdefault(call_type, {}).setdefault(None, []).append(callback)


@pytest.fixture
def comfy_api(monkeypatch, comfy_stubs):
    """comfy.ldm.modules.attention with wrap_attn (ComfyUI 0.3.60+) and comfy.patcher_extension."""
    modules = types.ModuleType("comfy.ldm.modules")
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.wrap_attn = lambda func: func
    extension = types.ModuleType("comfy.patcher_extension")
    extension.CallbacksMP = types.SimpleNamespace(ON_PRE_RUN="on_pre_run", ON_CLEANUP="on_cleanup")
    modules.attention = attention
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules", modules)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", attention)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    monkeypatch.setattr(sys.modules["comfy.ldm"], "modules", modules, raising=False)
    return attention


@pytest.fixture
def node(bcnodes):
    return bcnodes["sage_attention"].PatchSageAttention()


@pytest.fixture
def sa(bcnodes):
    return bcnodes["libs.sage_attention"]


def stand_in(sa):
    kernel = sa.KERNELS[sa.AUTO]
    return sa.Active(kernel, "auto (sageattn_qk_int8_pv_fp16_cuda)", lambda *a, **kw: None, kernel.arguments((12, 0)),
                     "GPU", (12, 0), "2.2.0")


def test_mode_combo_and_defaults(node, sa):
    required = node.INPUT_TYPES()["required"]
    assert required["mode"][0] == list(sa.MODES) and required["mode"][1]["default"] == "auto"
    assert required["nan_guard"][1]["default"] is True
    assert required["log_level"][0] == ["once", "verbose"] and required["log_level"][1]["default"] == "once"
    assert (node.RETURN_TYPES, node.CATEGORY) == (("MODEL",), "BCNodes/model")


def test_disabled_returns_an_unpatched_clone(node):
    model = Patcher()
    (out,) = node.patch(model, "disabled", True, "once")
    assert out is not model and "optimized_attention_override" not in out.model_options["transformer_options"]
    assert out.callbacks == {}


def test_no_model_passes_through(node):
    assert node.patch(None, "auto", True, "once") == (None,)


def test_missing_sageattention_warns_and_leaves_the_clone_unpatched(node, sa, comfy_api, monkeypatch, caplog):
    monkeypatch.setattr(sa, "_installed", lambda package: None)
    model = Patcher()
    with caplog.at_level(logging.WARNING):
        (out,) = node.patch(model, "auto", True, "once")
    assert out is not model and "optimized_attention_override" not in out.model_options["transformer_options"]
    assert caplog.records[-1].getMessage() == ("[BCNodes] Patch Sage Attention: auto (sageattn_qk_int8_pv_fp16_cuda): "
                                               "sageattention is not installed; the model keeps ComfyUI's attention.")


def test_a_resolved_kernel_patches_the_clone_only(node, sa, comfy_api, monkeypatch, caplog):
    seen = []
    monkeypatch.setattr(sa, "resolve", lambda mode, device: seen.append((mode, device)) or stand_in(sa))
    earlier = object()
    model = Patcher({"optimized_attention_override": earlier})
    (out,) = node.patch(model, "auto", False, "verbose")
    override = out.model_options["transformer_options"]["optimized_attention_override"]
    assert isinstance(override, sa.SageOverride) and seen == [("auto", "cuda:0")]
    assert override.previous is earlier and override.nan_guard is False and override.stats.verbose is True
    assert model.model_options["transformer_options"] == {"optimized_attention_override": earlier} and model.callbacks == {}
    with caplog.at_level(logging.INFO):
        (start,) = out.callbacks["on_pre_run"][None]
        (end,) = out.callbacks["on_cleanup"][None]
        start(out)
        end(out)
    assert [r.getMessage() for r in caplog.records][-2:] == [
        "[BCNodes] Patch Sage Attention: auto (sageattn_qk_int8_pv_fp16_cuda) [pv_accum_dtype=fp32, smooth_k=True, "
        "qk_quant_gran=per_warp] on GPU (sm_120), sageattention 2.2.0, nan_guard off",
        "[BCNodes] Patch Sage Attention: run summary: 0 attention calls, 0 ran auto (sageattn_qk_int8_pv_fp16_cuda), "
        "0 ran the original attention"]


def test_a_comfyui_without_the_override_is_an_error(node, comfy_api, monkeypatch):
    monkeypatch.delattr(comfy_api, "wrap_attn")
    with pytest.raises(RuntimeError, match="ComfyUI 0.3.60 or newer"):
        node.patch(Patcher(), "auto", True, "once")

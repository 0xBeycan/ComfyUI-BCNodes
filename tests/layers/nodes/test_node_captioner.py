"""Captioner (nodes/lm.py) through the node class, on the stub folder_paths and the real Qwen LM catalog, with a
recording stand-in for the "core" LM backend (no weights):

  - the surface: image, lm (BC_LM, required, no default), style (none / tags / prose, prose), seed (its `fixed`
    control), the instruction a socket (forceInput), then config (BC_LM_CONFIG, optional, appended after
    instruction), a tooltip on every input; no model, precision, lora or
    keep widget; outputs image + caption, both lists; no INPUT_IS_LIST, no heavy output, not an output node;
    category, display name; no family, model or Qwen name in its code;
  - through pipelines/lm.prepare + generate: one prepare per node call from the LMSpec (the model file looked
    for once, the LoRA read once, its console notes once), one generate per frame; the backend's prompt per
    frame (the frame then the style's text; the frame alone for none; the instruction in place of the style),
    its GenerateParams (DECODING without an LM Config, 600 tokens, the seed; with one connected, prepare gets
    its value as it is: the model's thinking-off defaults with its fields over them, DECODING not used), the
    frame the model gets capped at 1024 on
    its long side; the LMSpec's model, precision and LoRA at its strength; a LoRA with tags or prose keeps the
    style; the outputs: one list element per frame, each image a view of the input frame, each caption its
    frame's answer;
  - keep_model_loaded from the LMSpec: on, never unloaded; off, unloaded once after the last frame, after an
    error in a frame too (and in prepare), and ComfyUI's interrupt passes; an empty batch is refused before the
    model is prepared.
The flow itself is tests/layers/pipelines/test_pipe_captioner.py.
"""

import inspect
import logging
import os
import sys

import pytest
import torch

IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
THINK_OFF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
REQUIRED = ["image", "lm", "style", "seed"]


@pytest.fixture
def lm(bcnodes):
    return bcnodes["lm"]


@pytest.fixture
def pipe(bcnodes):
    return bcnodes["pipelines.lm"]


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.captioner.caption"]


@pytest.fixture
def fp(comfy_stubs, monkeypatch, tmp_path):
    """The stub folder_paths with models_dir in tmp_path and one text_encoders folder."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(fp, "folder_names_and_paths", {"text_encoders": ([str(tmp_path / "models" / "text_encoders")],
                                                                         {".safetensors"})})
    return fp


class Backend:
    """The "core" LM backend stand-in: records its calls in `events`; generate answers "caption <n>" for its
    n-th call (from 1), or raises `error` on call `fail_at`."""
    name = "core"

    def __init__(self):
        self.events, self.generated, self.error, self.fail_at = [], [], None, None

    def load(self, path, lora):
        self.events.append(("load", path, lora))
        return "handle"

    def generate(self, handle, prompt, images, params):
        self.events.append(("generate",))
        self.generated.append((prompt, images, params))
        if self.error is not None and len(self.generated) == self.fail_at:
            raise self.error
        return f" caption {len(self.generated)}\n"

    def unload(self):
        self.events.append(("unload",))
        return {}


@pytest.fixture
def backend(bcnodes, monkeypatch, fp, tmp_path):
    """The stand-in backend as "core", and the Qwen3.5-9B INT8 ConvRot file in models/Qwen-LM."""
    registry = bcnodes["models.common.registry"]
    fake = Backend()
    monkeypatch.setitem(registry._FAMILIES[registry.LM_BACKEND], "core", fake)
    folder = tmp_path / "models" / "Qwen-LM"
    folder.mkdir(parents=True)
    (folder / "Qwen3.5-9B_int8_convrot.safetensors").write_bytes(b"")
    return fake


def spec(pipe, **changes):
    """An LM Loader's output: Qwen3.5-9B INT8 ConvRot, no LoRA, kept loaded."""
    fields = dict(family="qwen_lm", model="Qwen3.5-9B", precision="INT8 ConvRot", lora="None", lora_strength=1.0,
                  keep_model_loaded=True)
    return pipe.LMSpec(**{**fields, **changes})


def run(lm, pipe, image, style="prose", keep=True, instruction=None, seed=5, config=None, **changes):
    return lm.Captioner().caption(image, spec(pipe, keep_model_loaded=keep, **changes), style, seed,
                                  instruction=instruction, config=config)


def kinds(backend):
    return [e[0] for e in backend.events]


def save_lora(path):
    """A ComfyUI-named LoRA of one Qwen3.5 module, rank 2, alpha 4."""
    from safetensors.torch import save_file

    module = "text_encoders.transformer.model.layers.0.mlp.down_proj"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file({f"{module}.lora_A.weight": torch.zeros(2, 3), f"{module}.lora_B.weight": torch.zeros(4, 2),
               f"{module}.alpha": torch.tensor(4.0)}, path)


# --- the surface ---------------------------------------------------------------------------------------------


def test_the_inputs_and_their_defaults(lm, fp):
    spec = lm.Captioner.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(spec) == ["required", "optional"] and list(req) == REQUIRED and list(opt) == ["instruction", "config"]
    assert opt["config"][0] == "BC_LM_CONFIG" and "default" not in opt["config"][1]
    assert req["image"][0] == "IMAGE"
    assert req["lm"][0] == "BC_LM" and "default" not in req["lm"][1]  # required: no default model, no fallback
    assert req["style"][0] == ["none", "tags", "prose"] and req["style"][1]["default"] == "prose"
    seed = req["seed"][1]
    assert req["seed"][0] == "INT" and (seed["default"], seed["min"], seed["max"]) == (0, 0, 0xffffffffffffffff)
    assert seed["control_after_generate"] == "fixed"
    assert opt["instruction"][0] == "STRING" and opt["instruction"][1]["forceInput"] is True
    assert "default" not in opt["instruction"][1] and not opt["instruction"][1].get("multiline")
    assert all(v[1].get("tooltip") for v in (*req.values(), *opt.values()))


def test_no_family_or_model_in_the_captioner(lm):
    """The Captioner reaches a model only through its lm input and pipelines/lm: no family key, model name,
    precision or Qwen reference in its code, tooltips, description or search aliases."""
    source = inspect.getsource(lm.Captioner)
    text = repr(lm.Captioner.INPUT_TYPES()) + lm.Captioner.DESCRIPTION + " ".join(lm.Captioner.SEARCH_ALIASES)
    for word in ("qwen", "Qwen", "FAMILY", "INT8", "BF16", "family_key"):
        assert word not in source and word not in text, word


def test_outputs_category_and_name(lm):
    cls = lm.Captioner
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.OUTPUT_IS_LIST, cls.FUNCTION) == (
        ("IMAGE", "STRING"), ("image", "caption"), (True, True), "caption")
    assert len(cls.OUTPUT_TOOLTIPS) == 2 and cls.CATEGORY == "BCNodes/lm"
    assert not getattr(cls, "INPUT_IS_LIST", False) and not getattr(cls, "OUTPUT_NODE", False)
    assert not getattr(cls, "HEAVY_OUTPUTS", ())  # views of the input frames: nothing to drop
    assert lm.NODE_CLASS_MAPPINGS["BC_Captioner"] is cls and lm.NODE_DISPLAY_NAME_MAPPINGS["BC_Captioner"] == "Captioner"
    assert cls.DESCRIPTION and cls.SEARCH_ALIASES[0] == "BCNodes" and "caption" in cls.SEARCH_ALIASES
    assert "BC_Captioner" not in lm.FAMILY_NODES  # the catalog route lists one node per family: Qwen LM


# --- through the LM runtime ----------------------------------------------------------------------------------


def test_one_prepare_then_one_generate_per_frame(lm, pipe, backend, monkeypatch):
    counts = {"prepare": 0, "generate": 0, "model_path": 0}
    for name in counts:
        def counted(*args, _name=name, _fn=getattr(pipe, name), **kwargs):
            counts[_name] += 1
            return _fn(*args, **kwargs)
        monkeypatch.setattr(pipe, name, counted)
    run(lm, pipe, torch.rand(4, 16, 16, 3))
    assert counts == {"prepare": 1, "generate": 4, "model_path": 1}


def test_one_run_per_frame_with_the_style_text(lm, pipe, flow, backend, tmp_path):
    image = torch.rand(3, 40, 56, 3)
    images, captions = run(lm, pipe, image)
    assert captions == ["caption 1", "caption 2", "caption 3"]
    assert len(images) == 3
    for i, frame in enumerate(images):
        assert frame.shape == (1, 40, 56, 3) and frame.data_ptr() == image[i].data_ptr()
    prose = flow.user_text("prose", None)
    for prompt, frames, params in backend.generated:
        assert prompt == "<|im_start|>user\n" + IMAGE_BLOCK + prose + THINK_OFF
        assert frames.shape[0] == 1 and frames.dtype == torch.float32
        assert (params.do_sample, params.temperature, params.top_p, params.top_k, params.min_p, params.presence_penalty,
                params.repetition_penalty, params.seed, params.max_new_tokens) == (True, 0.7, 0.8, 20, 0.0, 0.0, 1.0, 5, 600)
    model = str(tmp_path / "models" / "Qwen-LM" / "Qwen3.5-9B_int8_convrot.safetensors")
    assert backend.events == [("load", model, None), ("generate",)] * 3  # kept loaded: never unloaded


def test_none_sends_the_frame_alone(lm, pipe, backend):
    run(lm, pipe, torch.rand(1, 32, 32, 3), style="none")
    assert backend.generated[0][0] == "<|im_start|>user\n" + IMAGE_BLOCK + THINK_OFF


@pytest.mark.parametrize("style", ["none", "tags", "prose"])
def test_the_instruction_replaces_the_style(lm, pipe, backend, style):
    run(lm, pipe, torch.rand(1, 32, 32, 3), style=style, instruction="Describe the outfit only.")
    assert backend.generated[0][0] == "<|im_start|>user\n" + IMAGE_BLOCK + "Describe the outfit only." + THINK_OFF


def test_a_whitespace_instruction_is_not_connected(lm, pipe, flow, backend):
    run(lm, pipe, torch.rand(1, 32, 32, 3), style="tags", instruction="  \n")
    assert backend.generated[0][0] == "<|im_start|>user\n" + IMAGE_BLOCK + flow.user_text("tags", None) + THINK_OFF


def params_of(backend):
    p = backend.generated[0][2]
    return (p.do_sample, p.temperature, p.top_p, p.top_k, p.min_p, p.presence_penalty, p.repetition_penalty, p.mtp,
            p.seed, p.max_new_tokens)


def test_without_a_config_prepare_gets_the_decoding(lm, pipe, flow, backend, monkeypatch):
    seen = []
    monkeypatch.setattr(pipe, "prepare", lambda *a, _fn=pipe.prepare, **k: seen.append(k["config"]) or _fn(*a, **k))
    run(lm, pipe, torch.rand(1, 16, 16, 3))
    assert seen == [flow.DECODING] and seen[0] is not flow.DECODING  # a copy: the constant is never handed out
    assert params_of(backend) == (True, 0.7, 0.8, 20, 0.0, 0.0, 1.0, "auto", 5, 600)


@pytest.mark.parametrize("config, expected", [
    # nothing edited: the model's thinking-off defaults (Qwen3.5-9B: presence_penalty 1.5), DECODING not used
    ({}, (True, 0.7, 0.8, 20, 0.0, 1.5, 1.0, "auto", 5, 600)),
    # edited fields over the model's defaults, not over DECODING
    ({"temperature": 0.3, "mtp": "off"}, (True, 0.3, 0.8, 20, 0.0, 1.5, 1.0, "off", 5, 600)),
    ({"do_sample": False, "presence_penalty": 0.0}, (False, 0.7, 0.8, 20, 0.0, 0.0, 1.0, "auto", 5, 600)),
])
def test_a_connected_config_reaches_prepare_as_it_is(lm, pipe, backend, monkeypatch, config, expected):
    seen = []
    monkeypatch.setattr(pipe, "prepare", lambda *a, _fn=pipe.prepare, **k: seen.append(k["config"]) or _fn(*a, **k))
    run(lm, pipe, torch.rand(2, 16, 16, 3), config=config)
    assert len(seen) == 1 and seen[0] is config
    assert params_of(backend) == expected and len(backend.generated) == 2
    assert all(g[2] == backend.generated[0][2] for g in backend.generated)


def test_an_lm_config_node_feeds_the_captioner(lm, pipe, backend):
    """LM Config's own output (one field edited) on the Captioner: that field over the model's defaults."""
    (config,) = lm.LMConfig().build(True, 0.42, 20, 0.8, 0.0, 1.0, 1.5, "auto", "temperature")
    run(lm, pipe, torch.rand(1, 16, 16, 3), config=config)
    assert params_of(backend) == (True, 0.42, 0.8, 20, 0.0, 1.5, 1.0, "auto", 5, 600)


def test_the_model_gets_the_frame_capped_at_1024(lm, pipe, backend):
    image = torch.rand(1, 1500, 2000, 3)
    images, _ = run(lm, pipe, image)
    # 2000 x 1500 -> thumbnail 1024 x 768 -> the official sizing: multiples of 32
    assert tuple(backend.generated[0][1].shape) == (1, 768, 1024, 3)
    assert images[0].shape == (1, 1500, 2000, 3) and images[0].data_ptr() == image.data_ptr()


# --- the LMSpec: model, LoRA ---------------------------------------------------------------------------------


def test_the_specs_model_and_precision(lm, pipe, backend, tmp_path):
    (tmp_path / "models" / "Qwen-LM" / "Qwen3-VL-8B-Instruct_bf16.safetensors").write_bytes(b"")
    run(lm, pipe, torch.rand(2, 16, 16, 3), model="Qwen3-VL-8B Instruct", precision="BF16")
    path = str(tmp_path / "models" / "Qwen-LM" / "Qwen3-VL-8B-Instruct_bf16.safetensors")
    assert [e for e in backend.events if e[0] == "load"] == [("load", path, None)] * 2
    # Qwen3-VL's template: no think block; the decoding stays the Captioner's
    assert backend.generated[0][0].endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert backend.generated[0][2].presence_penalty == 0.0 and backend.generated[0][2].max_new_tokens == 600


@pytest.mark.parametrize("style", ["tags", "prose"])
def test_a_lora_keeps_the_style(lm, pipe, flow, backend, tmp_path, monkeypatch, caplog, style):
    """A LoRA with tags or prose: the style's prompt still goes in (none only when selected); the LoRA at the
    spec's strength, read once, the same LoraSpec on every frame, its console notes once per node call."""
    save_lora(str(tmp_path / "models" / "Qwen-LM" / "loras" / "my_lora.safetensors"))
    reads = []
    monkeypatch.setattr(pipe, "read_lora", lambda path, _fn=pipe.read_lora: reads.append(path) or _fn(path))
    with caplog.at_level(logging.INFO):
        run(lm, pipe, torch.rand(3, 16, 16, 3), style=style, lora="my_lora.safetensors", lora_strength=0.6)
    loads = [e for e in backend.events if e[0] == "load"]
    assert len(loads) == 3 and all(e[2] is loads[0][2] for e in loads) and loads[0][2].strength == 0.6
    assert len(reads) == 1
    assert all(prompt == "<|im_start|>user\n" + IMAGE_BLOCK + flow.user_text(style, None) + THINK_OFF
               for prompt, _, _ in backend.generated)
    assert len([r for r in caplog.records if "Captioner: LoRA my_lora.safetensors" in r.getMessage()]) == 1


def test_a_lora_with_none_sends_the_frame_alone(lm, pipe, backend, tmp_path):
    save_lora(str(tmp_path / "models" / "Qwen-LM" / "loras" / "my_lora.safetensors"))
    run(lm, pipe, torch.rand(1, 16, 16, 3), style="none", lora="my_lora.safetensors")
    assert backend.generated[0][0] == "<|im_start|>user\n" + IMAGE_BLOCK + THINK_OFF


# --- keep_model_loaded ---------------------------------------------------------------------------------------


def test_keep_off_unloads_once_after_the_last_frame(lm, pipe, backend):
    run(lm, pipe, torch.rand(3, 16, 16, 3), keep=False)
    assert kinds(backend) == ["load", "generate"] * 3 + ["unload"]


def test_keep_off_unloads_once_after_an_error(lm, pipe, backend):
    backend.error, backend.fail_at = RuntimeError("CUDA out of memory"), 2
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run(lm, pipe, torch.rand(3, 16, 16, 3), keep=False)
    assert kinds(backend) == ["load", "generate", "load", "generate", "unload"]


def test_keep_off_unloads_after_an_error_in_prepare(lm, pipe, backend):
    with pytest.raises(FileNotFoundError, match="LoRA gone.safetensors is not in"):
        run(lm, pipe, torch.rand(2, 16, 16, 3), keep=False, lora="gone.safetensors")
    assert kinds(backend) == ["unload"]


def test_keep_on_leaves_the_model_after_an_error(lm, pipe, backend):
    backend.error, backend.fail_at = RuntimeError("CUDA out of memory"), 1
    with pytest.raises(RuntimeError):
        run(lm, pipe, torch.rand(2, 16, 16, 3), keep=True)
    assert kinds(backend) == ["load", "generate"]


def test_the_interrupt_passes_and_unloads(lm, pipe, backend):
    class InterruptProcessingException(BaseException):  # as comfy.model_management's
        pass

    backend.error, backend.fail_at = InterruptProcessingException(), 1
    with pytest.raises(InterruptProcessingException):
        run(lm, pipe, torch.rand(2, 16, 16, 3), keep=False)
    assert kinds(backend) == ["load", "generate", "unload"]


def test_an_empty_batch_is_an_error_before_the_model_and_still_unloads_with_keep_off(lm, pipe, backend, monkeypatch):
    monkeypatch.setattr(pipe, "prepare", lambda *a, **k: pytest.fail("prepared for an empty batch"))
    with pytest.raises(ValueError, match="image holds no frames"):
        run(lm, pipe, torch.zeros(0, 16, 16, 3), keep=False)
    assert kinds(backend) == ["unload"]

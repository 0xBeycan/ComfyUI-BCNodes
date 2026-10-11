"""Prompter (nodes/lm.py) through the node class, on the stub folder_paths and the real Qwen LM catalog, with a
recording stand-in for the "core" LM backend (no weights) and a stand-in CLIP:

  - the surface: lm (BC_LM, required, no default), text (multiline, ""), instructions (none / include identity /
    exclude identity, include identity), seed (its `fixed` control); optional image, instruction (a socket),
    clip (CLIP), config (BC_LM_CONFIG); the hidden prompt graph and node id; a tooltip on every input; outputs
    text (STRING) and conditioning (CONDITIONING); category, display name, after Captioner in the menu; no
    family, model or Qwen name in its code;
  - through pipelines/lm.prepare + generate: one prepare and one generate per node call; the backend's prompt
    for each row of the table (the official Qwen3.5 rendering: no system turn, no prefill, thinking off, the
    image block first when the image is used); GenerateParams: DECODING without an LM Config, the model's
    thinking-off defaults with the config's fields over them with one (prepare gets it as it is), 600 tokens,
    the seed; the image the model gets capped at 1024;
  - conditioning: with a clip, core's CLIPTextEncode on the prompt (tokenize, then encode_from_tokens_scheduled,
    the result as it is), after the model is unloaded (keep_model_loaded off); without one, an error asking for a
    CLIP before the model is prepared when the prompt links conditioning (and an unload with keep off), else no
    error and an ExecutionBlocker carrying that message (a node linked later after a cached run meets it); a
    link to text only, or no prompt graph, is no error;
  - keep_model_loaded from the LMSpec: on, never unloaded; off, unloaded once after the generate, after an
    error too (an input error before the model included).
The flow itself is tests/layers/pipelines/test_pipe_prompter.py.
"""

import inspect
import sys
import types

import pytest
import torch

IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
THINK_OFF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
USER = "<|im_start|>user\n"
BRIEF = "a woman reading in a cafe"


@pytest.fixture
def lm(bcnodes):
    return bcnodes["lm"]


@pytest.fixture
def pipe(bcnodes):
    return bcnodes["pipelines.lm"]


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.prompter.write"]


@pytest.fixture
def fp(comfy_stubs, monkeypatch, tmp_path):
    """The stub folder_paths with models_dir in tmp_path and one text_encoders folder."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(fp, "folder_names_and_paths", {"text_encoders": ([str(tmp_path / "models" / "text_encoders")],
                                                                         {".safetensors"})})
    return fp


class Blocker:
    """comfy_execution.graph.ExecutionBlocker's stand-in: holds the message."""

    def __init__(self, message):
        self.message = message


@pytest.fixture(autouse=True)
def execution_blocker(monkeypatch):
    graph = types.ModuleType("comfy_execution.graph")
    graph.ExecutionBlocker = Blocker
    monkeypatch.setitem(sys.modules, "comfy_execution", types.ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.graph", graph)


class Backend:
    """The "core" LM backend stand-in: records its calls in `events` (shared with the CLIP stand-in) and answers
    `answer`, or raises `error`."""
    name = "core"

    def __init__(self, events):
        self.events, self.generated, self.error, self.answer = events, [], None, "  A half-body shot of one woman.\n"

    def load(self, path, lora):
        self.events.append(("load", path, lora))
        return "handle"

    def generate(self, handle, prompt, images, params):
        self.events.append(("generate",))
        self.generated.append((prompt, images, params))
        if self.error is not None:
            raise self.error
        return self.answer

    def unload(self):
        self.events.append(("unload",))
        return {}


class Clip:
    """A CLIP stand-in: tokenize and encode_from_tokens_scheduled recorded in `events`."""

    def __init__(self, events):
        self.events = events

    def tokenize(self, text):
        self.events.append(("tokenize", text))
        return {"tokens": text}

    def encode_from_tokens_scheduled(self, tokens):
        self.events.append(("encode", tokens))
        return [["cond", {"pooled_output": None}]]


@pytest.fixture
def events():
    return []


@pytest.fixture
def backend(bcnodes, monkeypatch, fp, tmp_path, events):
    """The stand-in backend as "core", and the Qwen3.5-9B INT8 ConvRot file in models/Qwen-LM."""
    registry = bcnodes["models.common.registry"]
    fake = Backend(events)
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


def run(lm, pipe, text=BRIEF, instructions="include identity", keep=True, seed=5, **inputs):
    return lm.Prompter().write(spec(pipe, keep_model_loaded=keep), text, instructions, seed, **inputs)


def kinds(events):
    return [e[0] for e in events]


def params_of(backend):
    p = backend.generated[0][2]
    return (p.do_sample, p.temperature, p.top_p, p.top_k, p.min_p, p.presence_penalty, p.repetition_penalty, p.mtp,
            p.seed, p.max_new_tokens)


# --- the surface ---------------------------------------------------------------------------------------------


def test_the_inputs_and_their_defaults(lm, fp):
    spec = lm.Prompter.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(spec) == ["required", "optional", "hidden"]
    assert list(req) == ["lm", "text", "instructions", "seed"]
    assert list(opt) == ["image", "instruction", "clip", "config"]
    assert spec["hidden"] == {"prompt_graph": "PROMPT", "unique_id": "UNIQUE_ID"}
    assert req["lm"][0] == "BC_LM" and "default" not in req["lm"][1]
    assert req["text"][0] == "STRING" and req["text"][1]["default"] == "" and req["text"][1]["multiline"] is True
    assert req["instructions"][0] == ["none", "include identity", "exclude identity"]
    assert req["instructions"][1]["default"] == "include identity"
    seed = req["seed"][1]
    assert req["seed"][0] == "INT" and (seed["default"], seed["min"], seed["max"]) == (0, 0, 0xffffffffffffffff)
    assert seed["control_after_generate"] == "fixed"
    assert [opt[name][0] for name in opt] == ["IMAGE", "STRING", "CLIP", "BC_LM_CONFIG"]
    assert opt["instruction"][1]["forceInput"] is True and "default" not in opt["instruction"][1]
    assert all(v[1].get("tooltip") for v in (*req.values(), *opt.values()))


def test_outputs_category_and_name(lm):
    cls = lm.Prompter
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION) == (("STRING", "CONDITIONING"), ("text", "conditioning"),
                                                                 "write")
    assert len(cls.OUTPUT_TOOLTIPS) == 2 and cls.CATEGORY == "BCNodes/lm"
    assert not getattr(cls, "OUTPUT_IS_LIST", ()) and not getattr(cls, "OUTPUT_NODE", False)
    assert not getattr(cls, "HEAVY_OUTPUTS", ())
    assert lm.NODE_CLASS_MAPPINGS["BC_Prompter"] is cls and lm.NODE_DISPLAY_NAME_MAPPINGS["BC_Prompter"] == "Prompter"
    assert list(lm.NODE_CLASS_MAPPINGS)[-2:] == ["BC_Captioner", "BC_Prompter"]
    assert cls.DESCRIPTION and cls.SEARCH_ALIASES[0] == "BCNodes" and "prompt enhancer" in cls.SEARCH_ALIASES
    assert "BC_Prompter" not in lm.FAMILY_NODES


def test_no_family_or_model_in_the_prompter(lm):
    source = inspect.getsource(lm.Prompter)
    text = repr(lm.Prompter.INPUT_TYPES()) + lm.Prompter.DESCRIPTION + " ".join(lm.Prompter.SEARCH_ALIASES)
    for word in ("qwen", "Qwen", "FAMILY", "INT8", "BF16", "family_key"):
        assert word not in source and word not in text, word


# --- the table through the LM runtime ------------------------------------------------------------------------


def test_one_prepare_and_one_generate(lm, pipe, backend, monkeypatch):
    counts = {"prepare": 0, "generate": 0, "model_path": 0}
    for name in counts:
        def counted(*args, _name=name, _fn=getattr(pipe, name), **kwargs):
            counts[_name] += 1
            return _fn(*args, **kwargs)
        monkeypatch.setattr(pipe, name, counted)
    run(lm, pipe)
    assert counts == {"prepare": 1, "generate": 1, "model_path": 1}


@pytest.mark.parametrize("instructions, name", [("include identity", "brief-include.txt"),
                                                ("exclude identity", "brief-exclude.txt")])
@pytest.mark.parametrize("with_image", [False, True])
def test_a_brief_goes_after_the_brief_prompt_and_the_image_is_not_used(lm, pipe, flow, backend, instructions, name,
                                                                       with_image):
    image = torch.rand(1, 32, 32, 3) if with_image else None
    text, _ = run(lm, pipe, instructions=instructions, image=image)
    prompt, images, _ = backend.generated[0]
    assert prompt == USER + flow.instruction_text(instructions, None, "brief") + "\n\n" + BRIEF + THINK_OFF
    assert images is None and IMAGE_BLOCK not in prompt and text == "A half-body shot of one woman."


@pytest.mark.parametrize("instructions", ["include identity", "exclude identity"])
def test_an_image_alone_goes_before_the_image_prompt(lm, pipe, flow, backend, instructions):
    run(lm, pipe, text=" ", instructions=instructions, image=torch.rand(1, 32, 32, 3))
    prompt, images, _ = backend.generated[0]
    assert prompt == USER + IMAGE_BLOCK + flow.instruction_text(instructions, None, "image") + THINK_OFF
    assert images.shape[0] == 1


def test_none_text_only(lm, pipe, backend):
    run(lm, pipe, instructions="none")
    assert backend.generated[0][0] == USER + BRIEF + THINK_OFF and backend.generated[0][1] is None


def test_none_image_only(lm, pipe, backend):
    run(lm, pipe, text="", instructions="none", image=torch.rand(1, 32, 32, 3))
    assert backend.generated[0][0] == USER + IMAGE_BLOCK + THINK_OFF


def test_none_text_and_image(lm, pipe, backend):
    run(lm, pipe, instructions="none", image=torch.rand(1, 32, 32, 3))
    assert backend.generated[0][0] == USER + IMAGE_BLOCK + BRIEF + THINK_OFF


@pytest.mark.parametrize("instructions", ["none", "include identity", "exclude identity"])
def test_the_override_replaces_the_built_in_prompt(lm, pipe, backend, instructions):
    run(lm, pipe, instructions=instructions, instruction="Rewrite as a prompt.")
    assert backend.generated[0][0] == USER + "Rewrite as a prompt.\n\n" + BRIEF + THINK_OFF
    backend.generated.clear()
    run(lm, pipe, text="", instructions=instructions, instruction="Describe it.", image=torch.rand(1, 16, 16, 3))
    assert backend.generated[0][0] == USER + IMAGE_BLOCK + "Describe it." + THINK_OFF


def test_the_model_gets_the_image_capped_at_1024(lm, pipe, backend):
    run(lm, pipe, text="", image=torch.rand(1, 1500, 2000, 3))
    # 2000 x 1500 -> thumbnail 1024 x 768 -> the official sizing: multiples of 32
    assert tuple(backend.generated[0][1].shape) == (1, 768, 1024, 3)


def test_no_brief_and_no_image_is_an_error_before_the_model(lm, pipe, backend, monkeypatch):
    monkeypatch.setattr(pipe, "prepare", lambda *a, **k: pytest.fail("prepared without a brief or an image"))
    with pytest.raises(ValueError, match="Prompter has nothing to work from"):
        run(lm, pipe, text="  ", keep=False)
    assert kinds(backend.events) == ["unload"]


def test_a_batch_is_an_error_before_the_model(lm, pipe, backend, monkeypatch):
    monkeypatch.setattr(pipe, "prepare", lambda *a, **k: pytest.fail("prepared for a batch"))
    with pytest.raises(ValueError, match="Prompter takes one image, got a batch of 3 frames"):
        run(lm, pipe, text="", image=torch.rand(3, 16, 16, 3))
    assert backend.events == []


# --- the decoding and LM Config ------------------------------------------------------------------------------


def test_without_a_config_prepare_gets_the_decoding(lm, pipe, flow, backend, monkeypatch):
    seen = []
    monkeypatch.setattr(pipe, "prepare", lambda *a, _fn=pipe.prepare, **k: seen.append(k) or _fn(*a, **k))
    run(lm, pipe)
    assert seen[0]["config"] == flow.DECODING and seen[0]["config"] is not flow.DECODING
    assert (seen[0]["node"], seen[0]["thinking"], seen[0]["max_new_tokens"], seen[0]["seed"]) == (
        "Prompter", False, 600, 5)
    assert params_of(backend) == (True, 0.7, 0.8, 20, 0.0, 0.0, 1.0, "auto", 5, 600)


@pytest.mark.parametrize("config, expected", [
    ({}, (True, 0.7, 0.8, 20, 0.0, 1.5, 1.0, "auto", 5, 600)),
    ({"temperature": 0.3, "mtp": "off"}, (True, 0.3, 0.8, 20, 0.0, 1.5, 1.0, "off", 5, 600)),
    ({"top_p": 0.95, "presence_penalty": 0.0}, (True, 0.7, 0.95, 20, 0.0, 0.0, 1.0, "auto", 5, 600)),
])
def test_a_connected_config_reaches_prepare_as_it_is(lm, pipe, backend, monkeypatch, config, expected):
    seen = []
    monkeypatch.setattr(pipe, "prepare", lambda *a, _fn=pipe.prepare, **k: seen.append(k["config"]) or _fn(*a, **k))
    run(lm, pipe, config=config)
    assert seen == [config] and seen[0] is config
    assert params_of(backend) == expected


# --- conditioning --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [True, False])
def test_a_clip_encodes_the_prompt_as_clip_text_encode_after_the_unload(lm, pipe, backend, events, keep):
    text, conditioning = run(lm, pipe, keep=keep, clip=Clip(events))
    assert text == "A half-body shot of one woman."
    assert conditioning == [["cond", {"pooled_output": None}]]
    unload = [("unload",)] if not keep else []
    assert events[1:] == [("generate",)] + unload + [("tokenize", text), ("encode", {"tokens": text})]


def graph(**links):
    """A prompt graph: node 7 is the Prompter; node 9 takes `links` {input name: [node, output index]}."""
    return {"7": {"class_type": "BC_Prompter", "inputs": {}}, "9": {"class_type": "KSampler", "inputs": links}}


def test_linked_conditioning_without_a_clip_is_an_error_before_the_model(lm, pipe, backend, monkeypatch):
    monkeypatch.setattr(pipe, "prepare", lambda *a, **k: pytest.fail("prepared although no CLIP is connected"))
    with pytest.raises(ValueError, match="no CLIP is connected: connect a CLIP .* to Prompter's clip input"):
        run(lm, pipe, keep=False, prompt_graph=graph(positive=["7", 1]), unique_id="7")
    assert kinds(backend.events) == ["unload"]


@pytest.mark.parametrize("prompt_graph", [graph(text=["7", 0]), graph(positive=["8", 1]), graph(), None])
def test_unlinked_conditioning_without_a_clip_is_no_error_and_carries_the_blocker(lm, prompt_graph, pipe, backend):
    text, conditioning = run(lm, pipe, prompt_graph=prompt_graph, unique_id="7")
    assert text == "A half-body shot of one woman."
    assert isinstance(conditioning, Blocker) and conditioning.message == lm.NO_CLIP


def test_linked_conditioning_with_a_clip_runs(lm, pipe, backend, events):
    text, conditioning = run(lm, pipe, clip=Clip(events), prompt_graph=graph(positive=["7", 1]), unique_id="7")
    assert conditioning == [["cond", {"pooled_output": None}]] and ("tokenize", text) in events


# --- keep_model_loaded ---------------------------------------------------------------------------------------


def test_keep_on_never_unloads(lm, pipe, backend):
    run(lm, pipe, keep=True)
    assert kinds(backend.events) == ["load", "generate"]


def test_keep_off_unloads_once_after_the_generate(lm, pipe, backend):
    run(lm, pipe, keep=False)
    assert kinds(backend.events) == ["load", "generate", "unload"]


def test_keep_off_unloads_after_an_error(lm, pipe, backend):
    backend.error = RuntimeError("CUDA out of memory")
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        run(lm, pipe, keep=False, clip=Clip(backend.events))
    assert kinds(backend.events) == ["load", "generate", "unload"]  # no encode after the error

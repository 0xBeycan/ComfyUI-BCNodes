"""models/common/lm/core_backend.py with ComfyUI core replaced by recording stand-ins (no weights):

  - load: core's comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None,
    clip_type=CLIPType.STABLE_DIFFUSION); one slot: the base kept by path, a LoRA clone by its key; another
    file unloads everything (through core) before it loads;
  - LoRA: the spec's build gets core's CLIP key map of the base's cond_stage_model and its weight shapes,
    core's load_lora_for_models(None, base, tensors, 0, strength) makes the clone; a build that raises
    leaves the base loaded and no clone;
  - unload: unload_model_and_clones on the base's patcher while the base is still held, then base and
    clone dropped; {file name: bytes of its weights}, {} when empty;
  - generate: our token structure (the CLIP's tokenize is never used), max_length = max_new_tokens, the
    seed always given, mtp mapped to core's True / False / depth; decode with skip_special_tokens;
  - the cuDNN attention fallback (comfy.ops.SDPA_BACKEND_PRIORITY holding cuDNN, as core sets it with CUDA): a
    cuDNN attention error retries once without cuDNN in that list (one warning, the failed attempt's prefetch
    state dropped as the executor does, the same tokens and arguments), the original list object back after
    success and failure, later generates start without cuDNN; any other error passes through; a second
    cuDNN failure says to start ComfyUI with --use-split-cross-attention;
  - tokenize: SDTokenizer's slicing (tokens_start, the end token its HF tokenizer adds) and its start /
    end tokens; one image entry per <|image_pad|>, in order, holding images[i:i + 1]; a tokenizer whose
    init_kwargs carry pretokenize_regex is encoded through a copy split by it (core's instance unchanged,
    the copy cached), one that cannot be copied is used as it is with one warning;
  - weight_shapes: the logical shape a quantized weight reports, tied modules under every name.
The comparison with core's real tokenizers is in tests/test_runtime.py (lm_core).
"""

import re
import sys
import types

import pytest
import torch

PAD, IM_START, IM_END = 900, 901, 902
SPECIAL = {"<|image_pad|>": PAD, "<|im_start|>": IM_START, "<|im_end|>": IM_END}


@pytest.fixture
def cb(bcnodes):
    return bcnodes["models.common.lm.core_backend"]


@pytest.fixture
def backend_params(bcnodes):
    return bcnodes["models.common.lm.backend"]


class _HF:
    """A HF tokenizer stand-in: special tokens by id, any other character by its code point; no
    pretokenize_regex, so it is used as it is (as core's Qwen3-VL tokenizer)."""

    init_kwargs = {}

    def __call__(self, text):
        ids = []
        for piece in re.split(r"(<\|[a-z_]+\|>)", text):
            ids += [SPECIAL[piece]] if piece in SPECIAL else [ord(c) for c in piece]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]


def _tokenizer(name="qwen35_9b", tokens_start=0, adds_end=False, start=None, end=None, hf=None):
    """Core's SD1Tokenizer shape: clip / clip_name and the SDTokenizer under that attribute."""
    inner = types.SimpleNamespace(tokenizer=hf or _HF(), tokens_start=tokens_start, tokenizer_adds_end_token=adds_end,
                                  start_token=start, end_token=end)
    return types.SimpleNamespace(clip=name, clip_name=name, **{name: inner})


class _Clip:
    """core's CLIP: tokenizer, cond_stage_model (one 4x3 float32 Linear), patcher; records generate / decode."""

    def __init__(self, rec, path):
        self.rec, self.path = rec, path
        self.tokenizer = _tokenizer()
        self.cond_stage_model = torch.nn.Module()
        self.cond_stage_model.proj = torch.nn.Linear(3, 4, bias=False)
        self.patcher = types.SimpleNamespace(name=f"patcher of {path}")

    def tokenize(self, *a, **kw):
        pytest.fail("core's tokenize was used: it alters the raw prompt")

    def generate(self, tokens, **kw):
        self.rec.events.append(("generate", self, tokens, kw))
        return [65, 66, IM_END]

    def decode(self, ids, **kw):
        self.rec.events.append(("decode", ids, kw))
        return "AB"


@pytest.fixture
def core(monkeypatch):
    """comfy.sd, comfy.lora and comfy.model_management's unload calls as recording stand-ins."""
    rec = types.SimpleNamespace(events=[], key_map={"text_encoders.proj": "proj.weight"}, backend=None)
    sd = types.ModuleType("comfy.sd")
    sd.CLIPType = types.SimpleNamespace(STABLE_DIFFUSION="CLIPType.STABLE_DIFFUSION", FLUX="CLIPType.FLUX")

    def load_clip(**kw):
        rec.events.append(("load_clip", kw))
        return _Clip(rec, kw["ckpt_paths"][0])

    def load_lora_for_models(model, clip, lora, strength_model, strength_clip):
        rec.events.append(("load_lora_for_models", model, clip, lora, strength_model, strength_clip))
        return None, types.SimpleNamespace(base=clip, lora=lora, strength=strength_clip)

    sd.load_clip = load_clip
    sd.load_lora_for_models = load_lora_for_models
    # comfy.ops without SDPA_BACKEND_PRIORITY (no CUDA); the cuDNN tests give it core's list
    ops = types.ModuleType("comfy.ops")
    memory = types.ModuleType("comfy.memory_management")
    memory.aimdo_enabled = True
    prefetch = types.ModuleType("comfy.model_prefetch")
    prefetch.cleanup_prefetch_queues = lambda: rec.events.append(("cleanup_prefetch_queues",))
    lora = types.ModuleType("comfy.lora")

    def model_lora_keys_clip(model, key_map={}):
        rec.events.append(("model_lora_keys_clip", model, dict(key_map)))
        return dict(rec.key_map)

    lora.model_lora_keys_clip = model_lora_keys_clip
    mm = sys.modules["comfy.model_management"]

    def unload_model_and_clones(patcher):
        # held while core unloads it: the base is dropped only after this returns
        rec.events.append(("unload_model_and_clones", patcher, rec.backend and rec.backend._base is not None))

    monkeypatch.setattr(mm, "unload_model_and_clones", unload_model_and_clones, raising=False)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: rec.events.append(("soft_empty_cache",)), raising=True)
    monkeypatch.setattr(mm, "reset_cast_buffers", lambda: rec.events.append(("reset_cast_buffers",)), raising=False)
    for name, mod in (("comfy.sd", sd), ("comfy.lora", lora), ("comfy.ops", ops), ("comfy.memory_management", memory),
                      ("comfy.model_prefetch", prefetch)):
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(sys.modules["comfy"], name.split(".")[1], mod, raising=False)
    return rec


@pytest.fixture
def backend(cb, core):
    core.backend = cb.CoreBackend()
    return core.backend


def kinds(rec):
    return [e[0] for e in rec.events]


def spec(backend_params, key, strength=0.5, built=None, fail=None):
    """A LoraSpec whose build records its arguments and returns {key: tensor} (or raises `fail`)."""
    calls = [] if built is None else built

    def build(key_map, weight_shapes):
        calls.append((key_map, weight_shapes))
        if fail is not None:
            raise fail
        return {"text_encoders.proj.lora_A.weight": torch.zeros(1, 3), "key": key}
    return backend_params.LoraSpec(key=key, strength=strength, build=build), calls


def test_load_clip_arguments_and_the_cache(backend, core):
    clip = backend.load("/m/Qwen3.5-9B_bf16.safetensors", None)
    assert core.events == [("load_clip", {"ckpt_paths": ["/m/Qwen3.5-9B_bf16.safetensors"], "embedding_directory": None,
                                          "clip_type": "CLIPType.STABLE_DIFFUSION"})]
    assert backend.load("/m/Qwen3.5-9B_bf16.safetensors", None) is clip
    assert kinds(core) == ["load_clip"], "the same path is served from the slot"


def test_another_file_unloads_everything_first(backend, core):
    a = backend.load("/m/a.safetensors", None)
    b = backend.load("/m/b.safetensors", None)
    assert b is not a and b.path == "/m/b.safetensors"
    assert kinds(core) == ["load_clip", "unload_model_and_clones", "soft_empty_cache", "load_clip"]
    assert core.events[1][1] is a.patcher


def test_a_file_that_fails_to_load_leaves_the_slot_empty(backend, core, monkeypatch):
    backend.load("/m/a.safetensors", None)

    def broken(**kw):
        raise RuntimeError("not a model file")

    monkeypatch.setattr(sys.modules["comfy.sd"], "load_clip", broken)
    with pytest.raises(RuntimeError, match="not a model file"):
        backend.load("/m/b.safetensors", None)
    assert kinds(core) == ["load_clip", "unload_model_and_clones", "soft_empty_cache"]
    assert backend.unload() == {} and kinds(core)[-1] == "soft_empty_cache", "nothing held after the failed load"


def test_lora_build_and_clone(backend, backend_params, core):
    lora, built = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.5))
    clone = backend.load("/m/a.safetensors", lora)
    base = backend.load("/m/a.safetensors", None)
    assert kinds(core) == ["load_clip", "model_lora_keys_clip", "load_lora_for_models"]
    _, model, given = core.events[1]
    assert model is base.cond_stage_model and given == {}, "core's key map of the base CLIP's model"
    assert built == [(core.key_map, {"proj.weight": (4, 3)})]
    _, unet, clip, tensors, strength_model, strength_clip = core.events[2]
    assert unet is None and clip is base and strength_model == 0 and strength_clip == 0.5
    assert tensors["key"] == lora.key and clone.lora is tensors


def test_lora_cache_by_key(backend, backend_params, core):
    lora, built = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.5))
    clone = backend.load("/m/a.safetensors", lora)
    same, _ = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.5), built=built)
    assert backend.load("/m/a.safetensors", same) is clone and len(built) == 1, "an equal key is a cache hit"
    other, _ = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.75), strength=0.75, built=built)
    clone2 = backend.load("/m/a.safetensors", other)
    assert clone2 is not clone and len(built) == 2 and clone2.strength == 0.75
    assert kinds(core).count("load_clip") == 1, "the base stays loaded across LoRA changes"
    base = backend.load("/m/a.safetensors", None)
    assert base is clone2.base
    assert backend.load("/m/a.safetensors", other) is not clone2 and len(built) == 3, "None dropped the clone"


def test_lora_on_another_file_reloads_and_rebuilds(backend, backend_params, core):
    lora, built = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.5))
    backend.load("/m/a.safetensors", lora)
    clone = backend.load("/m/b.safetensors", lora)
    assert clone.base.path == "/m/b.safetensors" and len(built) == 2
    assert kinds(core).count("unload_model_and_clones") == 1


def test_a_lora_that_does_not_fit_keeps_the_base(backend, backend_params, core):
    bad, _ = spec(backend_params, ("/l/bad.safetensors", 1, 2, 1.0), fail=ValueError("this LoRA does not fit this model"))
    with pytest.raises(ValueError, match="does not fit"):
        backend.load("/m/a.safetensors", bad)
    base = backend.load("/m/a.safetensors", None)
    assert kinds(core) == ["load_clip", "model_lora_keys_clip"] and base.path == "/m/a.safetensors"
    good, built = spec(backend_params, ("/l/good.safetensors", 1, 2, 1.0))
    assert backend.load("/m/a.safetensors", good).base is base and len(built) == 1


def test_unload(backend, backend_params, core):
    assert backend.unload() == {} and core.events == [], "nothing held: no core call"
    lora, _ = spec(backend_params, ("/l/x.safetensors", 1, 2, 0.5))
    backend.load("/m/dir/Qwen3.5-4B_bf16.safetensors", lora)
    base = backend.load("/m/dir/Qwen3.5-4B_bf16.safetensors", None)
    backend.load("/m/dir/Qwen3.5-4B_bf16.safetensors", lora)
    core.events.clear()
    assert backend.unload() == {"Qwen3.5-4B_bf16.safetensors": 4 * 3 * 4}
    assert core.events == [("unload_model_and_clones", base.patcher, True)]
    assert backend.unload() == {} and len(core.events) == 1
    backend.load("/m/dir/Qwen3.5-4B_bf16.safetensors", None)
    assert kinds(core)[-1] == "load_clip", "unloaded: the next load reads the file again"


@pytest.mark.parametrize("mtp, expected", [("auto", True), ("off", False), ("2", 2), ("3", 3), ("4", 4), ("5", 5)])
def test_core_mtp(cb, mtp, expected):
    got = cb.core_mtp(mtp)
    assert got == expected and type(got) is type(expected)


@pytest.mark.parametrize("bad", ["1", "6", "on", "", True, 3])
def test_core_mtp_refuses_other_values(cb, bad):
    with pytest.raises(ValueError, match="use auto, off, 2, 3, 4 or 5"):
        cb.core_mtp(bad)


def params(backend_params, **over):
    values = dict(do_sample=True, temperature=0.7, top_k=20, top_p=0.8, min_p=0.0, repetition_penalty=1.0,
                  presence_penalty=1.5, seed=123, max_new_tokens=32768, mtp="auto")
    return backend_params.GenerateParams(**{**values, **over})


def test_generate(backend, backend_params, core):
    clip = backend.load("/m/a.safetensors", None)
    prompt = "<|im_start|>user\nHi \\(x\\) embedding:e<|im_end|>\n"
    text = backend.generate(clip, prompt, None, params(backend_params))
    (_, used, tokens, kw), decode = core.events[-2:]
    assert used is clip and text == "AB"
    assert tokens == {"qwen35_9b": [[(t, 1.0) for t in _HF()(prompt)["input_ids"]]]}, "the raw prompt, unaltered"
    assert kw == dict(do_sample=True, max_length=32768, temperature=0.7, top_k=20, top_p=0.8, min_p=0.0,
                      repetition_penalty=1.0, seed=123, presence_penalty=1.5, mtp=True)
    assert decode == ("decode", [65, 66, IM_END], {"skip_special_tokens": True})


@pytest.mark.parametrize("mtp, expected", [("off", False), ("4", 4)])
def test_generate_greedy_still_gets_the_seed(backend, backend_params, core, mtp, expected):
    clip = backend.load("/m/a.safetensors", None)
    backend.generate(clip, "<|im_start|>user\nx<|im_end|>\n", None, params(backend_params, do_sample=False, seed=0, mtp=mtp))
    kw = core.events[-2][3]
    assert kw["do_sample"] is False and kw["seed"] == 0 and type(kw["seed"]) is int and kw["mtp"] == expected


def test_generate_with_images(backend, backend_params, core):
    clip = backend.load("/m/a.safetensors", None)
    images = torch.rand((2, 32, 32, 3))
    prompt = "<|im_start|>user\n<|image_pad|><|image_pad|>Hi<|im_end|>\n"
    backend.generate(clip, prompt, images, params(backend_params))
    (row,) = core.events[-2][2]["qwen35_9b"]
    entries = [t for t, _ in row if isinstance(t, dict)]
    assert len(row) == len(_HF()(prompt)["input_ids"]) and [w for _, w in row] == [1.0] * len(row)
    assert [e["type"] for e in entries] == ["image", "image"] and [e["original_type"] for e in entries] == ["image", "image"]
    assert all(e["data"].shape == (1, 32, 32, 3) and e["data"].data_ptr() == images[i:i + 1].data_ptr()
               for i, e in enumerate(entries))


CUDNN_ERROR = "cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built."


@pytest.fixture
def sdpa(cb, core, monkeypatch):
    """core's SDPA priority list on the comfy.ops stand-in, as comfy/ops.py sets it with CUDA; the process-wide
    flag cleared."""
    from torch.nn.attention import SDPBackend
    priority = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    monkeypatch.setattr(sys.modules["comfy.ops"], "SDPA_BACKEND_PRIORITY", priority, raising=False)
    monkeypatch.setattr(cb, "_cudnn_attention_off", False)
    return types.SimpleNamespace(priority=list(priority), original=priority, cudnn=SDPBackend.CUDNN_ATTENTION,
                                 ops=sys.modules["comfy.ops"])


class _Attention(_Clip):
    """A CLIP whose attention fails in cuDNN while core's SDPA priority holds it (`always`: without it too), or
    raises `error`; it records the priority list each generate saw."""

    def __init__(self, rec, always=False, error=None):
        super().__init__(rec, "/m/vl.safetensors")
        self.always, self.error, self.seen = always, error, []

    def generate(self, tokens, **kw):
        from torch.nn.attention import SDPBackend
        priority = getattr(sys.modules["comfy.ops"], "SDPA_BACKEND_PRIORITY", [])
        self.seen.append(list(priority))
        if self.error is not None:
            raise self.error
        if self.always or SDPBackend.CUDNN_ATTENTION in priority:
            self.rec.events.append(("generate failed", self, tokens, kw))
            raise RuntimeError(CUDNN_ERROR)
        return super().generate(tokens, **kw)


def test_cudnn_attention_failure_retries_once_without_it(backend, backend_params, core, sdpa, caplog, cb):
    clip = _Attention(core)
    prompt = "<|im_start|>user\nDescribe the scene in detail.<|im_end|>\n"
    with caplog.at_level("WARNING"):
        assert backend.generate(clip, prompt, None, params(backend_params)) == "AB"
    without = [b for b in sdpa.priority if b != sdpa.cudnn]
    assert clip.seen == [sdpa.priority, without]
    assert sdpa.ops.SDPA_BACKEND_PRIORITY is sdpa.original and sdpa.original == sdpa.priority
    failed, cleanup, reset, retried = core.events[-5:-1]
    # the executor's own cleanup between the attempts; the retry gets exactly what the first attempt got
    assert kinds(core)[-5:] == ["generate failed", "cleanup_prefetch_queues", "reset_cast_buffers", "generate", "decode"]
    assert retried[2] == failed[2] and retried[3] == failed[3] and retried[3]["seed"] == 123
    (warning,) = [r for r in caplog.records if "cuDNN" in r.getMessage()]
    assert warning.levelname == "WARNING" and "LM generation runs without cuDNN attention in this process" in warning.getMessage()
    assert CUDNN_ERROR in warning.getMessage() and cb._cudnn_attention_off is True
    # the next generate starts without cuDNN attention: no failing first try, no second warning
    caplog.clear()
    clip.seen.clear()
    with caplog.at_level("WARNING"):
        assert backend.generate(clip, prompt, None, params(backend_params)) == "AB"
    assert clip.seen == [without] and sdpa.ops.SDPA_BACKEND_PRIORITY is sdpa.original
    assert not [r for r in caplog.records if "cuDNN" in r.getMessage()] and "generate failed" not in kinds(core)[-2:]


def test_the_original_priority_list_is_back_after_a_failure(backend, backend_params, core, sdpa, cb, monkeypatch):
    monkeypatch.setattr(cb, "_cudnn_attention_off", True)
    oom = RuntimeError("CUDA out of memory")
    clip = _Attention(core, error=oom)
    with pytest.raises(RuntimeError) as e:
        backend.generate(clip, "<|im_start|>user\nx<|im_end|>\n", None, params(backend_params))
    assert e.value is oom and clip.seen == [[b for b in sdpa.priority if b != sdpa.cudnn]]
    assert sdpa.ops.SDPA_BACKEND_PRIORITY is sdpa.original and sdpa.original == sdpa.priority


@pytest.mark.parametrize("error", [RuntimeError("CUDA error: an illegal memory access was encountered"),
                                   type("InterruptProcessingException", (Exception,), {})()])
def test_other_errors_pass_through(backend, backend_params, core, sdpa, cb, caplog, error):
    clip = _Attention(core, error=error)
    with caplog.at_level("WARNING"), pytest.raises(type(error)) as e:
        backend.generate(clip, "<|im_start|>user\nx<|im_end|>\n", None, params(backend_params))
    assert e.value is error and clip.seen == [sdpa.priority] and cb._cudnn_attention_off is False
    assert sdpa.ops.SDPA_BACKEND_PRIORITY is sdpa.original and not caplog.records


def test_cudnn_failing_again_says_what_to_do(backend, backend_params, core, sdpa):
    clip = _Attention(core, always=True)
    with pytest.raises(RuntimeError) as e:
        backend.generate(clip, "<|im_start|>user\nx<|im_end|>\n", None, params(backend_params))
    assert "start ComfyUI with --use-split-cross-attention" in str(e.value) and CUDNN_ERROR in str(e.value)
    assert isinstance(e.value.__cause__, RuntimeError) and str(e.value.__cause__) == CUDNN_ERROR
    assert len(clip.seen) == 2 and sdpa.ops.SDPA_BACKEND_PRIORITY is sdpa.original


def test_no_fallback_without_cudnn_in_cores_priority(backend, backend_params, core, cb, monkeypatch):
    # no SDPA_BACKEND_PRIORITY (no CUDA, an older torch): the error is core's, as it is
    monkeypatch.setattr(cb, "_cudnn_attention_off", False)
    clip = _Attention(core, error=RuntimeError(CUDNN_ERROR))
    with pytest.raises(RuntimeError, match="^cuDNN Frontend error"):
        backend.generate(clip, "<|im_start|>user\nx<|im_end|>\n", None, params(backend_params))
    assert clip.seen == [[]] and cb._cudnn_attention_off is False


def test_token_structure(cb):
    frames = [torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, 3)]
    got = cb.token_structure("qwen3vl_8b", [1, PAD, 2, PAD, 3], PAD, frames)
    (row,) = got["qwen3vl_8b"]
    assert list(got) == ["qwen3vl_8b"] and [w for _, w in row] == [1.0] * 5
    assert [t for t, _ in row if not isinstance(t, dict)] == [1, 2, 3]
    assert row[1][0] == {"type": "image", "data": frames[0], "original_type": "image"} and row[1][0]["data"] is frames[0]
    assert row[3][0]["data"] is frames[1]
    assert cb.token_structure("k", [1, 2], PAD, []) == {"k": [[(1, 1.0), (2, 1.0)]]}


@pytest.mark.parametrize("ids, n", [([1, PAD, PAD], 1), ([1, PAD], 2), ([1, 2], 1)])
def test_token_structure_one_placeholder_per_frame(cb, ids, n):
    with pytest.raises(ValueError, match=rf"holds {ids.count(PAD)} <\|image_pad\|> token\(s\) for {n} image\(s\): remove"):
        cb.token_structure("k", ids, PAD, [torch.zeros(1, 2, 2, 3)] * n)


def test_tokenize_slices_as_sdtokenizer_does(cb):
    plain = types.SimpleNamespace(tokenizer=_tokenizer("qwen3vl_8b"))
    assert cb.tokenize(plain, "<|im_start|>ab", None) == {"qwen3vl_8b": [[(IM_START, 1.0), (97, 1.0), (98, 1.0)]]}
    framed = types.SimpleNamespace(tokenizer=_tokenizer("t5", tokens_start=1, adds_end=True, start=7, end=8))
    assert cb.tokenize(framed, "abcd", None) == {"t5": [[(7, 1.0), (98, 1.0), (99, 1.0), (8, 1.0)]]}


# transformers' Qwen2Tokenizer splits with Qwen2's regex (\p{L}+), the one core's tokenizer uses
QWEN2_SPLIT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
# the official Qwen3.5 / 3.8 tokenizer.json split, given by core's own qwen35 tokenizer_config as pretokenize_regex
OFFICIAL_SPLIT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
HINDI = "नमस्ते दुनिया"  # combining marks: Qwen2's regex splits the two words into 7 pieces, the official one into 2


def _split(regex):
    from tokenizers import Regex, pre_tokenizers
    return pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(regex), "isolated"),
                                    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


class _FastHF:
    """transformers' fast Qwen2Tokenizer in small: a tokenizers.Tokenizer split with Qwen2's regex as core builds it,
    one id per piece (WordLevel: the 7 pieces of Qwen2's split of HINDI are ids 1-7, the 2 of the official split 8
    and 9), and the init_kwargs a tokenizer_config gives."""

    def __init__(self, init_kwargs):
        from tokenizers import Tokenizer, models
        pieces = [p for regex in (QWEN2_SPLIT, OFFICIAL_SPLIT) for p, _ in _split(regex).pre_tokenize_str(HINDI)]
        self.backend_tokenizer = Tokenizer(models.WordLevel({"<unk>": 0, **{p: i + 1 for i, p in enumerate(pieces)}},
                                                            unk_token="<unk>"))
        self.backend_tokenizer.pre_tokenizer = _split(QWEN2_SPLIT)
        self.init_kwargs = init_kwargs

    def __call__(self, text):
        return {"input_ids": self.backend_tokenizer.encode(text).ids}

    def convert_tokens_to_ids(self, token):
        return SPECIAL[token]


def _ids(cb, hf, text):
    clip = types.SimpleNamespace(tokenizer=_tokenizer(hf=hf))
    return [t for t, _ in cb.tokenize(clip, text, None)["qwen35_9b"][0]]


def test_tokenize_splits_as_the_official_tokenizer_json(cb):
    # core's Qwen3.5 / 3.8 tokenizer carries pretokenize_regex; ours encodes with a copy split by it
    hf = _FastHF({"pretokenize_regex": OFFICIAL_SPLIT})
    before = str(hf.backend_tokenizer.pre_tokenizer)
    assert hf(HINDI)["input_ids"] == [1, 2, 3, 4, 5, 6, 7]  # core's ids
    assert _ids(cb, hf, HINDI) == [8, 9]
    # core's instance is never changed; the copy is built once per tokenizer
    assert str(hf.backend_tokenizer.pre_tokenizer) == before and hf(HINDI)["input_ids"] == [1, 2, 3, 4, 5, 6, 7]
    encoder = cb.official_encoder(hf)
    assert encoder is not hf.backend_tokenizer and cb.official_encoder(hf) is encoder
    assert _ids(cb, hf, HINDI) == [8, 9]


def test_tokenize_without_pretokenize_regex_uses_cores_tokenizer(cb):
    hf = _FastHF({})  # core's Qwen3-VL tokenizer: Qwen2's split is its official one
    assert cb.official_encoder(hf) is None and _ids(cb, hf, HINDI) == [1, 2, 3, 4, 5, 6, 7]


def test_an_official_pretokenizer_that_cannot_be_built_warns_once(cb, caplog):
    class _Slow(_HF):  # transformers 4.x: Qwen2Tokenizer is the slow tokenizer, with no backend_tokenizer
        init_kwargs = {"pretokenize_regex": OFFICIAL_SPLIT}

    hf = _Slow()
    with caplog.at_level("WARNING"):
        assert _ids(cb, hf, "ab") == [97, 98] and _ids(cb, hf, "c") == [99]
    (record,) = [r for r in caplog.records if "pre-tokenizer" in r.getMessage()]
    message = record.getMessage()
    assert record.levelname == "WARNING" and "could not be built (AttributeError:" in message
    assert re.search(r"; transformers \S+, tokenizers \S+\): the prompt is tokenized with core's tokenizer", message)


class _Packed(torch.Tensor):
    """Stands in for core's QuantizedTensor: a wrapper subclass reporting the logical [out, in] shape while
    its data is packed (int4, two per byte: [out, in / 2])."""

    @staticmethod
    def __new__(cls, data, logical):
        t = torch.Tensor._make_wrapper_subclass(cls, logical, dtype=torch.bfloat16, device=data.device)
        t.packed = data
        return t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(func)


def test_weight_shapes(cb):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 4, bias=False), torch.nn.LayerNorm(4)])
    quant = torch.nn.Module()
    quant.register_buffer("weight", _Packed(torch.zeros(8, 3, dtype=torch.uint8), (8, 6)))
    model.q_proj = quant
    model.tied = quant
    model.act = torch.nn.GELU()
    assert cb.weight_shapes(model) == {"layers.0.weight": (4, 3), "layers.1.weight": (4,), "q_proj.weight": (8, 6),
                                       "tied.weight": (8, 6)}
    assert cb.weight_shapes(torch.nn.Linear(2, 5)) == {"weight": (5, 2)}


def test_registered_as_the_core_backend(bcnodes, cb):
    registry = bcnodes["models.common.registry"]
    assert "core" in registry.names(registry.LM_BACKEND)
    assert isinstance(registry.get(registry.LM_BACKEND, "core"), cb.CoreBackend)
    assert cb.CoreBackend.name == "core"

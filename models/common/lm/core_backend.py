"""The LM backend over ComfyUI core (registered as "core"): the model file loaded by core's text-encoder
loader (comfy.sd.load_clip, which tells the architecture from the weights), a LoRA applied by core's
loader, generation and decoding by the CLIP's own generate / decode.

The prompt is a raw chat text built by the family. Core's tokenize would alter it (the SD tokenizer splits
the text at every `embedding:` after whitespace and tokenizes the pieces apart, and turns `\\(` / `\\)` into
`(` / `)`), so the token structure core's Qwen tokenizers return is built here from the HF tokenizer the
CLIP holds, the whole prompt as one piece: one row of (id, 1.0) pairs, each image placeholder id replaced
in order by that frame's image entry. For text with neither, it equals core's tokenize_with_weights output
(tests/test_runtime.py compares them on core's own Qwen3.5 and Qwen3-VL tokenizers), except where core departs
from the official tokenizer: core builds transformers' Qwen2Tokenizer, whose pre-tokenizer splits words with
Qwen2's regex (`\\p{L}+`), ignoring the `pretokenize_regex` its own Qwen3.5 tokenizer_config.json gives. The
official Qwen3.5-9B / Qwen3.8-27B tokenizer.json splits with that regex (`[\\p{L}\\p{M}]+`), so a word with
combining marks (Devanagari, Bengali, Tamil, Thai, Arabic with harakat, ...) gets other ids from core. When the
HF tokenizer's init_kwargs carry `pretokenize_regex`, the prompt is encoded with a copy of its backend
tokenizer (one per tokenizer, cached) whose pre-tokenizer is the official one; core's instance is never
changed. Qwen3-VL's tokenizer carries no such key and is used as it is. A copy that cannot be built (a slow
tokenizer under transformers 4.x, say) is one warning naming the cause and the versions, and core's ids.

cuDNN attention: core's SDPA wrapper (comfy/ops.py scaled_dot_product_attention) runs a large enough
attention under sdpa_kernel(SDPA_BACKEND_PRIORITY), the module global [flash, cuDNN, efficient, math] read
at every call. With an attention mask flash is out, so cuDNN runs, and on some GPU / cuDNN builds it builds
no plan ("cuDNN Frontend error: ... No valid execution plans built.", seen with Qwen3-VL on an RTX 3090).
The first such error in an LM generate turns cuDNN attention off for LM generation in this process (one
warning) and the generate runs again: with comfy.ops.SDPA_BACKEND_PRIORITY swapped for the same list without
cuDNN only for the duration of each LM generate, the original list object put back after it, so core's
other models keep their attention. Core's files are not changed.

One slot: the base CLIP of one file (by path) and at most one LoRA clone of it (by LoraSpec.key). Core
keeps a clone's weights shared with the base and points the loaded entry back at the base when the clone
is dropped, so unload() unloads the base and every clone through core before dropping them.
ComfyUI is imported inside the functions.
"""

import contextlib
import logging
import os
import re
import weakref

import torch

from ....libs.tensor_census import module_bytes
from .backend import GenerateParams, LoraSpec

# The token both of core's Qwen tokenizers (Qwen3.5 / 3.8 and Qwen3-VL) replace by an image, one per frame.
IMAGE_PAD = "<|image_pad|>"
# core's HF tokenizer -> the copy of its backend tokenizer with the official pre-tokenizer, None to use it as it is
_OFFICIAL = weakref.WeakKeyDictionary()
# A RuntimeError of cuDNN attention building no plan (torch's cudnn_frontend SDPA path).
CUDNN_ATTENTION_FAILURE = re.compile(r"cuDNN Frontend error|No valid execution plans")
# Set by the first cuDNN attention failure in an LM generate: every later one runs without cuDNN attention.
_cudnn_attention_off = False


def core_mtp(value):
    """core's generate takes mtp=True (auto: core adapts the draft depth), False (off) or a fixed depth 2..5;
    'auto' given as it is raises there."""
    if value == "auto":
        return True
    if value == "off":
        return False
    if value in ("2", "3", "4", "5"):
        return int(value)
    raise ValueError(f"mtp {value!r}: use auto, off, 2, 3, 4 or 5")


def token_structure(key, ids, pad_id, frames):
    """{key: [[(id, 1.0), ...]]}: what core's Qwen tokenizers return for a raw prompt (tokenize_with_weights,
    no weights), every `pad_id` replaced in order by {"type": "image", "data": frame, "original_type":
    "image"} for the next of `frames`. Exactly one placeholder per frame."""
    pads = ids.count(pad_id)
    if pads != len(frames):
        raise ValueError(f"the prompt holds {pads} {IMAGE_PAD} token(s) for {len(frames)} image(s): "
                         f"remove {IMAGE_PAD} from the system, user and assistant text")
    frames = iter(frames)
    return {key: [[({"type": "image", "data": next(frames), "original_type": "image"} if t == pad_id else t, 1.0)
                   for t in ids]]}


def official_encoder(hf):
    """For core's HF tokenizer `hf`: a copy of its backend tokenizer (tokenizers.Tokenizer) whose pre-tokenizer
    is the official one, Split(`pretokenize_regex` of its init_kwargs, isolated) then ByteLevel without a prefix
    space, as the official Qwen3.5 / Qwen3.8 tokenizer.json has it; None when `hf` carries no such regex (encode
    with `hf`), or when the copy cannot be built (one warning). Cached per `hf`; `hf` is never changed."""
    if hf in _OFFICIAL:
        return _OFFICIAL[hf]
    regex = hf.init_kwargs.get("pretokenize_regex")
    encoder = None
    if regex:
        try:
            import copy

            from tokenizers import Regex, pre_tokenizers

            encoder = copy.deepcopy(hf.backend_tokenizer)
            encoder.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.Split(Regex(regex), "isolated"),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        except Exception as e:  # noqa: BLE001 (any failure here is the documented degrade, said once)
            from importlib.metadata import PackageNotFoundError, version

            def installed(name):
                try:
                    return version(name)
                except PackageNotFoundError:
                    return "not installed"

            logging.warning("[BCNodes] LM: the official Qwen pre-tokenizer could not be built (%s: %s; transformers %s, "
                            "tokenizers %s): the prompt is tokenized with core's tokenizer, which gives words with "
                            "combining marks (Devanagari, Thai, Arabic with harakat, ...) other ids than the official "
                            "tokenizer", type(e).__name__, e, installed("transformers"), installed("tokenizers"))
            encoder = None
    _OFFICIAL[hf] = encoder
    return encoder


def tokenize(clip, prompt, images):
    """The tokens of the raw `prompt` for clip.generate, with `images` ((N, H, W, 3) or None) one frame per
    image placeholder, as core slices them (frame i = images[i:i + 1]). The ids are what core's SDTokenizer
    makes of one piece of text: its HF tokenizer's ids (encoded with the official pre-tokenizer when
    official_encoder gives one) from tokens_start, without the end token that HF tokenizer adds, between the
    SDTokenizer's own start and end tokens (core's Qwen tokenizers have none of them, and no padding)."""
    outer = clip.tokenizer  # core's SD1Tokenizer: tokenize_with_weights returns {outer.clip_name: rows}
    inner = getattr(outer, outer.clip)  # its SDTokenizer, holding the HF tokenizer
    end = -1 if inner.tokenizer_adds_end_token else None
    encoder = official_encoder(inner.tokenizer)
    ids = (inner.tokenizer(prompt)["input_ids"] if encoder is None else encoder.encode(prompt).ids)[inner.tokens_start:end]
    if inner.start_token is not None:
        ids.insert(0, inner.start_token)
    if inner.end_token is not None:
        ids.append(inner.end_token)
    frames = [] if images is None else [images[i:i + 1] for i in range(images.shape[0])]
    return token_structure(outer.clip_name, ids, inner.tokenizer.convert_tokens_to_ids(IMAGE_PAD), frames)


def weight_shapes(module):
    """{state-dict key of every module weight: its logical shape}. A quantized weight (core's
    QuantizedTensor) reports its logical [out, in] shape, while state_dict() holds its stored data
    (W4A8: int4 packed two per byte, [out, in / 2]). Tied modules are listed under every name, as
    state_dict() lists them."""
    return {f"{name}.weight" if name else "weight": tuple(m.weight.shape)
            for name, m in module.named_modules(remove_duplicate=False)
            if isinstance(getattr(m, "weight", None), torch.Tensor)}


@contextlib.contextmanager
def without_cudnn_attention(ops, cudnn):
    """comfy.ops.SDPA_BACKEND_PRIORITY (`ops` is comfy.ops) as the same list without `cudnn` for the block; the
    original list object is always put back."""
    original = ops.SDPA_BACKEND_PRIORITY
    ops.SDPA_BACKEND_PRIORITY = [backend for backend in original if backend != cudnn]
    try:
        yield
    finally:
        ops.SDPA_BACKEND_PRIORITY = original


def generate_ids(handle, tokens, kwargs):
    """handle.generate(tokens, **kwargs) with the cuDNN attention fallback (module docstring), which applies only
    while core's SDPA priority list holds cuDNN (comfy.ops.SDPA_BACKEND_PRIORITY exists only with CUDA and a torch
    whose sdpa_kernel takes set_priority). Any other error, ComfyUI's interrupt included, passes through as it
    is; cuDNN failing again without cuDNN attention is an error saying what to do."""
    global _cudnn_attention_off
    import comfy.ops

    try:
        from torch.nn.attention import SDPBackend
    except ImportError:  # a torch without SDPA backends: core has no priority list either
        return handle.generate(tokens, **kwargs)
    cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    priority = getattr(comfy.ops, "SDPA_BACKEND_PRIORITY", None)
    if cudnn is None or not isinstance(priority, list) or cudnn not in priority:
        return handle.generate(tokens, **kwargs)
    first = None
    if not _cudnn_attention_off:
        try:
            return handle.generate(tokens, **kwargs)
        except RuntimeError as e:
            if not CUDNN_ATTENTION_FAILURE.search(str(e)):
                raise
            first = e
        logging.warning("[BCNodes] LM: cuDNN attention failed (%s); LM generation runs without cuDNN attention in "
                        "this process from now on (core's other models keep it), and this one runs again",
                        str(first).strip())
        _cudnn_attention_off = True
        # What ComfyUI's executor does after each node when comfy-aimdo is on (execution.py): the prefetch queues
        # (pinned blocks, malloc graph) and cast buffers the failed forward left are dropped, so the retry starts
        # as an un-failed run does. The KV cache, the seeded generator and the clip options are new in every
        # generate.
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_prefetch

        if comfy.memory_management.aimdo_enabled:
            comfy.model_prefetch.cleanup_prefetch_queues()
            comfy.model_management.reset_cast_buffers()
    try:
        with without_cudnn_attention(comfy.ops, cudnn):
            return handle.generate(tokens, **kwargs)
    except RuntimeError as e:
        if not CUDNN_ATTENTION_FAILURE.search(str(e)):
            raise
        raise RuntimeError(f"LM generation failed in cuDNN although cuDNN attention is off for it ({str(e).strip()}): "
                           "start ComfyUI with --use-split-cross-attention, so core's attention does not go through "
                           "PyTorch's SDPA") from (first or e)


class CoreBackend:
    name = "core"

    def __init__(self):
        self._path = None
        self._base = None
        self._lora_key = None
        self._lora = None

    def load(self, path: str, lora: "LoraSpec | None"):
        """The CLIP of the file at `path` (with `lora` applied: a clone of it); the base is kept by path, the
        clone by lora.key. Another file unloads everything held first; another LoRA, or none, drops the
        clone."""
        if self._base is None or self._path != path:
            self._load_base(path)
        if lora is None:
            self._lora_key, self._lora = None, None
            return self._base
        if self._lora is None or self._lora_key != lora.key:
            import comfy.lora
            import comfy.sd

            self._lora_key, self._lora = None, None
            model = self._base.cond_stage_model
            tensors = lora.build(comfy.lora.model_lora_keys_clip(model, {}), weight_shapes(model))
            self._lora = comfy.sd.load_lora_for_models(None, self._base, tensors, 0, lora.strength)[1]
            self._lora_key = lora.key
        return self._lora

    def _load_base(self, path):
        import comfy.model_management as mm
        import comfy.sd

        if self.unload():
            mm.soft_empty_cache()
        self._base = comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None,
                                        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION)
        self._path = path

    def generate(self, handle, prompt: str, images, params: GenerateParams) -> str:
        """The decoded new text of `handle` (a CLIP from load) for the raw `prompt`; special tokens such as
        <|im_end|> are stripped, <think> / </think> stay."""
        ids = generate_ids(handle, tokenize(handle, prompt, images),
                           dict(do_sample=params.do_sample, max_length=params.max_new_tokens,
                                temperature=params.temperature, top_k=params.top_k, top_p=params.top_p,
                                min_p=params.min_p, repetition_penalty=params.repetition_penalty, seed=params.seed,
                                presence_penalty=params.presence_penalty, mtp=core_mtp(params.mtp)))
        return handle.decode(ids, skip_special_tokens=True)

    def unload(self) -> dict:
        """Unloads the base CLIP and its clones through core (unload_model_and_clones) and drops them (the
        file loads again on the next run): {file name: bytes of its weights}, {} when none was held."""
        if self._base is None:
            return {}
        import comfy.model_management as mm

        held = {os.path.basename(self._path): module_bytes(self._base.cond_stage_model)}
        mm.unload_model_and_clones(self._base.patcher)
        self._path, self._base, self._lora_key, self._lora = None, None, None, None
        return held

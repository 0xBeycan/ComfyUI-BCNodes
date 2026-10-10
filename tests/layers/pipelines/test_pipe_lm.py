"""pipelines/lm.py on the Qwen LM family's real catalog, templates and image sizing, with a recording
stand-in for the "core" LM backend (no ComfyUI, no weights):

  - run's steps in the brief's order (catalog, validation by the family's prompt builder, file, LoRA,
    sampling, images, load, generate, split, unload); a validation error (a typed <|image_pad|> included)
    stops before any file is looked for; unknown model / precision errors list the valid ones;
  - the model file: each model folder (the file in it, then below it, links followed), then below each
    text_encoders folder, first hit wins; else a download into the first model folder; no url: an error
    naming the file and the folder to put it in;
  - the LoRA: "None" and strength 0 give none; the name resolved in the LoRA folders in order; the spec's
    key (path, mtimes of the weights and the config, size, strength), stamped before the files are read,
    and strength; its build runs check_base before it returns the plan's tensors;
  - sampling: the model's defaults for the mode, the config's fields over them, mtp off without an MTP
    head (a warning naming mtp when the config set it), seed and max_new_tokens -> GenerateParams;
  - the outputs split by the family; keep_model_loaded off unloads in a finally, after an error in any step
    once the model is known too (precision, validation, file, LoRA, sampling, images, generate), a model an
    earlier run kept included;
  - the catalog the widgets and the route show: the user models.yaml of every model folder, in folder
    order; the built-in one with the error when a user models.yaml is broken (a run raises it), precisions
    built-in first, the route's per-family JSON and its TypedDicts.
The LoRA formats, the catalog schema and the templates themselves are tested in their own layers.
"""

import dataclasses
import json
import logging
import os
import re

import pytest
import torch

PROMPT_OFF = ("<|im_start|>user\nDescribe this.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
OFF_9B = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5,
              repetition_penalty=1.0, mtp="auto")
ON_9B = dict(OFF_9B, temperature=1.0, top_p=0.95)
LORA_MODULE = "text_encoders.transformer.model.layers.0.mlp.down_proj"


@pytest.fixture
def pipe(bcnodes):
    return bcnodes["pipelines.lm"]


@pytest.fixture
def registry(bcnodes):
    return bcnodes["models.common.registry"]


class Backend:
    """The "core" LM backend stand-in: records its calls in `events`; generate returns `raw` (or raises
    `error`)."""
    name = "core"

    def __init__(self, events):
        self.events, self.raw, self.error = events, "an answer", None
        self.loaded = self.generated = None

    def load(self, path, lora):
        self.events.append("load")
        self.loaded = (path, lora)
        return "handle"

    def generate(self, handle, prompt, images, params):
        self.events.append("generate")
        self.generated = (handle, prompt, images, params)
        if self.error is not None:
            raise self.error
        return self.raw

    def unload(self):
        self.events.append("unload")
        return {}


@pytest.fixture
def events():
    return []


@pytest.fixture
def backend(registry, monkeypatch, events):
    fake = Backend(events)
    monkeypatch.setitem(registry._FAMILIES[registry.LM_BACKEND], "core", fake)
    return fake


@pytest.fixture
def traced(pipe, bcnodes, registry, monkeypatch, events, backend):
    """Every step of run records its name in `events`: the pipeline's own helpers and the family's hooks."""
    def wrap(name, fn):
        def traced_fn(*args, **kwargs):
            events.append(name)
            return fn(*args, **kwargs)
        return traced_fn

    for name, attr in (("catalog", "_load"), ("file", "model_path"), ("lora", "lora_spec"),
                       ("sampling", "generate_params")):
        monkeypatch.setattr(pipe, attr, wrap(name, getattr(pipe, attr)))
    family = bcnodes["models.qwen_lm"].FAMILY
    monkeypatch.setitem(registry._FAMILIES[registry.LM_FAMILY], "qwen_lm", dataclasses.replace(
        family, build_prompt=wrap("prompt", family.build_prompt), prepare_images=wrap("images", family.prepare_images),
        split_output=wrap("split", family.split_output)))
    return events


@pytest.fixture
def folders(tmp_path):
    """A model folder holding the default file (Qwen3.5-9B INT8 ConvRot), a second empty one, and a
    text_encoders folder."""
    a, b, te = tmp_path / "Qwen-LM", tmp_path / "Qwen-LM-2", tmp_path / "text_encoders"
    for folder in (a, b, te):
        folder.mkdir()
    (a / "Qwen3.5-9B_int8_convrot.safetensors").write_bytes(b"")
    return a, b, te


def request(pipe, folders, **changes):
    a, b, te = folders
    fields = dict(family="qwen_lm", node="Qwen LM", model="Qwen3.5-9B", precision="INT8 ConvRot", lora="None",
                  lora_strength=1.0, thinking=False, max_new_tokens=64, seed=7, keep_model_loaded=True, system="",
                  user="Describe this.", assistant="", image=None, config={}, model_folders=(str(a), str(b)),
                  text_encoder_folders=(str(te),))
    return pipe.LMRequest(**{**fields, **changes})


def save_lora(path, rank=2, out=4, inp=3, alpha=4.0, module=LORA_MODULE):
    """A LoRA of one module (ComfyUI-named by default), with its alpha tensor (none when `alpha` is None)."""
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tensors = {f"{module}.lora_A.weight": torch.randn(rank, inp, generator=g),
               f"{module}.lora_B.weight": torch.randn(out, rank, generator=g)}
    if alpha is not None:
        tensors[f"{module}.alpha"] = torch.tensor(alpha)
    save_file(tensors, path)
    return path


# --- run: order, validation, outputs, unload -----------------------------------------------------------------


def test_steps_run_in_order(pipe, traced, backend, folders):
    result = pipe.run(request(pipe, folders, keep_model_loaded=False))
    assert traced == ["catalog", "prompt", "file", "lora", "sampling", "images", "load", "generate", "split", "unload"]
    assert result == pipe.LMResult(text="an answer", thinking="")


def test_what_the_backend_gets(pipe, backend, folders):
    pipe.run(request(pipe, folders))
    path, lora = backend.loaded
    assert path == os.path.join(folders[0], "Qwen3.5-9B_int8_convrot.safetensors") and lora is None
    handle, prompt, images, params = backend.generated
    assert handle == "handle" and prompt == PROMPT_OFF and images is None
    assert params == pipe.GenerateParams(seed=7, max_new_tokens=64, **{k: v for k, v in OFF_9B.items()})


@pytest.mark.parametrize("changes, match", [
    (dict(user="  "), "user is empty"),
    (dict(model="Qwen3-VL-8B Instruct", thinking=True), "has no thinking mode"),
    (dict(thinking=True, assistant="{"), "cannot be used with thinking on"),
    # a typed <|image_pad|> would leave a placeholder without its frame: refused here, not after the download
    (dict(user="Compare with <|image_pad|>", image=torch.rand(1, 64, 64, 3)), re.escape("<|image_pad|>")),
    (dict(system="<|image_pad|>"), re.escape("<|image_pad|>")),
    (dict(assistant="<|image_pad|>"), re.escape("<|image_pad|>")),
])
def test_validation_stops_before_the_file(pipe, traced, backend, folders, changes, match):
    with pytest.raises(ValueError, match=match):
        pipe.run(request(pipe, folders, model_folders=(), **changes))  # no folder: a file search would fail
    assert traced == ["catalog", "prompt"]


def test_an_image_alone_is_a_prompt(pipe, backend, folders):
    image = torch.rand(2, 128, 128, 3)
    pipe.run(request(pipe, folders, user="", image=image))
    _, prompt, images, _ = backend.generated
    assert prompt.count("<|vision_start|><|image_pad|><|vision_end|>") == 2
    assert images.dtype == torch.float32 and images.shape == (2, 256, 256, 3)  # the official sizing: 128 -> 256
    assert backend.loaded[0].endswith("Qwen3.5-9B_int8_convrot.safetensors")


def test_unknown_model_lists_the_models(pipe, backend, folders, bcnodes):
    with pytest.raises(ValueError) as e:
        pipe.run(request(pipe, folders, model="Qwen9"))
    assert "'Qwen9' is not in the Qwen-LM catalog" in str(e.value)
    assert ("the models are: Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-9B Abliterated, Qwen3-VL-8B Instruct, "
            "Qwen3-VL-8B Instruct Abliterated, Qwen3.8-27B") in str(e.value)


def test_unknown_precision_lists_the_models_precisions(pipe, backend, folders):
    with pytest.raises(ValueError, match="Qwen3.5-9B has no precision 'W4A8'; its precisions are: BF16, INT8 ConvRot$"):
        pipe.run(request(pipe, folders, precision="W4A8"))


def test_thinking_output_split(pipe, backend, folders):
    backend.raw = "Let me look.</think>\n\nA cat."
    assert pipe.run(request(pipe, folders, thinking=True)) == pipe.LMResult(text="A cat.", thinking="Let me look.")
    assert backend.generated[1].endswith("<|im_start|>assistant\n<think>\n")


def test_prefill_starts_the_text(pipe, backend, folders):
    backend.raw = ' "a": 1}\n'
    assert pipe.run(request(pipe, folders, assistant=" {")) == pipe.LMResult(text='{ "a": 1}', thinking="")
    assert backend.generated[1].endswith("<think>\n\n</think>\n\n{")


def test_unload_after_the_run_unless_kept(pipe, backend, events, folders):
    pipe.run(request(pipe, folders, keep_model_loaded=True))
    assert events == ["load", "generate"]
    events.clear()
    pipe.run(request(pipe, folders, keep_model_loaded=False))
    assert events == ["load", "generate", "unload"]


def test_unload_after_an_error(pipe, backend, events, folders):
    backend.error = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        pipe.run(request(pipe, folders, keep_model_loaded=False))
    assert events == ["load", "generate", "unload"]
    events.clear()
    with pytest.raises(RuntimeError, match="out of memory"):
        pipe.run(request(pipe, folders, keep_model_loaded=True))
    assert events == ["load", "generate"]


@pytest.mark.parametrize("changes, match", [
    (dict(precision="W4A8"), "has no precision 'W4A8'"),
    (dict(user="  "), "user is empty"),  # validation
    (dict(thinking=True, assistant="{"), "cannot be used with thinking on"),
    (dict(model_folders=()), "no model folder is registered"),  # the file
    (dict(lora="gone.safetensors"), "LoRA gone.safetensors is not in"),
    (dict(lora="no_alpha.safetensors"), "no LoRA alpha"),  # refused while it is planned
    (dict(config={"mtp": "9"}), "mtp is '9'"),  # sampling
    (dict(image=torch.rand(1, 0, 64, 3)), "has no pixels"),  # the images
])
def test_unload_after_an_error_before_loading(pipe, backend, events, folders, changes, match):
    """An error before the load, once the model is known: keep_model_loaded off unloads what an earlier run
    kept; on, the model stays held."""
    # PEFT-named: a ComfyUI-named file without alpha takes core's convention (alpha = rank)
    save_lora(os.path.join(folders[0], "loras", "no_alpha.safetensors"), alpha=None,
              module="base_model.model.model.layers.0.mlp.down_proj")
    pipe.run(request(pipe, folders, keep_model_loaded=True))
    assert events == ["load", "generate"]
    for keep, after in ((True, []), (False, ["unload"])):
        events.clear()
        with pytest.raises((ValueError, FileNotFoundError), match=match):  # LoraError is a ValueError
            pipe.run(request(pipe, folders, keep_model_loaded=keep, **changes))
        assert events == after


# --- the model file ------------------------------------------------------------------------------------------


@pytest.fixture
def precision(bcnodes):
    return bcnodes["models.common.lm.catalog"].Precision("BF16", "m.safetensors", "https://example.org/m.safetensors")


def put(folder, *parts):
    path = os.path.join(folder, *parts, "m.safetensors")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()
    return path


def test_file_in_the_folder_before_below_it(pipe, precision, tmp_path):
    a, te = str(tmp_path / "a"), str(tmp_path / "te")
    below = put(a, "sub")
    assert pipe.model_path(precision, (a,), (te,)) == below
    direct = put(a)
    assert pipe.model_path(precision, (a,), (te,)) == direct


def test_each_model_folder_in_turn_then_text_encoders(pipe, precision, tmp_path):
    a, b, te = str(tmp_path / "a"), str(tmp_path / "b"), str(tmp_path / "te")
    in_te = put(te, "x", "y")
    assert pipe.model_path(precision, (a, b), (te,)) == in_te
    in_b = put(b, "deep", "er")
    assert pipe.model_path(precision, (a, b), (te,)) == in_b
    in_a = put(a, "z")
    put(b)  # a later folder's direct file does not win over an earlier folder's subfolder
    assert pipe.model_path(precision, (a, b), (te,)) == in_a


def test_links_are_followed_and_a_loop_ends(pipe, precision, tmp_path):
    a, elsewhere = tmp_path / "a", tmp_path / "elsewhere"
    a.mkdir()
    target = put(str(elsewhere), "inner")
    os.symlink(elsewhere, a / "linked")
    os.symlink(a, a / "linked" / "back")  # a loop back to the top
    assert pipe.model_path(precision, (str(a),), ()) == os.path.join(a, "linked", "inner", "m.safetensors")
    assert os.path.samefile(pipe.model_path(precision, (str(a),), ()), target)


def test_missing_file_downloads_into_the_first_model_folder(pipe, precision, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipe, "fetch_with_progress", lambda url, path, label: calls.append((url, path, label)) or path)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert pipe.model_path(precision, (a, b), ()) == os.path.join(a, "m.safetensors")
    assert calls == [("https://example.org/m.safetensors", os.path.join(a, "m.safetensors"), "m.safetensors")]


def test_missing_file_without_url_says_where_to_put_it(pipe, precision, tmp_path, monkeypatch):
    monkeypatch.setattr(pipe, "fetch_with_progress", lambda *a: pytest.fail("no download without a url"))
    a, te = str(tmp_path / "a"), str(tmp_path / "te")
    with pytest.raises(FileNotFoundError) as e:
        pipe.model_path(dataclasses.replace(precision, url=None), (a,), (te,))
    assert str(e.value).startswith(f"m.safetensors is not in {a} or under {te}") and str(e.value).endswith(f"put the file in {a}")


def test_run_downloads_the_catalog_url(pipe, backend, folders, monkeypatch):
    calls = []
    monkeypatch.setattr(pipe, "fetch_with_progress", lambda url, path, label: calls.append((url, path)) or path)
    pipe.run(request(pipe, folders, precision="BF16"))
    target = os.path.join(folders[0], "Qwen3.5-9B_bf16.safetensors")
    assert calls == [("https://huggingface.co/beycanai/Qwen-LM/resolve/main/Qwen3.5-9B_bf16.safetensors", target)]
    assert backend.loaded[0] == target


# --- the LoRA ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name, strength", [("None", 1.0), ("x.safetensors", 0.0)])
def test_no_lora(pipe, name, strength, tmp_path):
    assert pipe.lora_spec(name, strength, [str(tmp_path)], "Qwen LM") is None  # strength 0 reads nothing


def test_lora_spec_from_the_first_folder_holding_it(pipe, bcnodes, folders, tmp_path, caplog):
    a, b, _ = folders
    path = save_lora(os.path.join(b, "loras", "style", "x.safetensors"))
    roots = [os.path.join(a, "loras"), os.path.join(b, "loras")]
    with caplog.at_level(logging.INFO):
        spec = pipe.lora_spec(os.path.join("style", "x.safetensors"), 0.5, roots, "Qwen LM")
    st = os.stat(path)
    assert spec.key == (path, (st.st_mtime_ns, None), st.st_size, 0.5) and spec.strength == 0.5
    lm_lora = bcnodes["libs.lm_lora"]
    notes = lm_lora.plan_lora(lm_lora.read_lora(path)).notes
    assert [r.getMessage() for r in caplog.records] == [
        f"[BCNodes] Qwen LM: LoRA {os.path.join('style', 'x.safetensors')}: {note}" for note in notes]
    tensors = spec.build({LORA_MODULE: "model.layers.0.mlp.down_proj.weight"}, {"model.layers.0.mlp.down_proj.weight": (4, 3)})
    assert sorted(tensors) == sorted(f"{LORA_MODULE}.{s}" for s in ("alpha", "lora_A.weight", "lora_B.weight"))
    lora_error = bcnodes["libs.lm_lora"].LoraError
    with pytest.raises(lora_error, match="another shape"):  # check_base runs first
        spec.build({LORA_MODULE: "model.layers.0.mlp.down_proj.weight"}, {"model.layers.0.mlp.down_proj.weight": (8, 3)})
    with pytest.raises(lora_error, match="not in this model"):
        spec.build({}, {})
    first = save_lora(os.path.join(a, "loras", "style", "x.safetensors"))  # the same name in the first folder wins
    assert pipe.lora_spec(os.path.join("style", "x.safetensors"), 0.5, roots, "Qwen LM").key[0] == first


def test_peft_folder_key_follows_its_config(pipe, tmp_path):
    from safetensors.torch import save_file
    folder = tmp_path / "loras" / "adapter"
    folder.mkdir(parents=True)
    module = "base_model.model.model.layers.0.mlp.down_proj"
    save_file({f"{module}.lora_A.weight": torch.ones(2, 3), f"{module}.lora_B.weight": torch.ones(4, 2)},
              str(folder / "adapter_model.safetensors"))
    config = folder / "adapter_config.json"
    config.write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    first = pipe.lora_spec("adapter", 1.0, [str(tmp_path / "loras")], "Qwen LM")
    weights = os.stat(folder / "adapter_model.safetensors")
    assert first.key == (str(folder), (weights.st_mtime_ns, os.stat(config).st_mtime_ns), weights.st_size, 1.0)
    os.utime(config, ns=(1, 1))  # an edited config re-plans the LoRA
    assert pipe.lora_spec("adapter", 1.0, [str(tmp_path / "loras")], "Qwen LM").key[1] == (weights.st_mtime_ns, 1)


@pytest.mark.parametrize("name, weights, config", [
    ("adapter", os.path.join("adapter", "adapter_model.safetensors"), os.path.join("adapter", "adapter_config.json")),
    ("x.safetensors", "x.safetensors", "x.json"),  # a single file and its <stem>.json
])
def test_lora_stamped_before_the_read(pipe, tmp_path, monkeypatch, name, weights, config):
    """A LoRA saved again while it is read (a trainer writing into the loras folder): the key holds the stamps
    of the files read, so the next run's key differs and the backend re-plans, instead of the old tensors
    kept under the new files' stamps."""
    from safetensors.torch import save_file
    root = str(tmp_path / "loras")
    weights, config = os.path.join(root, weights), os.path.join(root, config)
    os.makedirs(os.path.dirname(weights))
    module = "base_model.model.model.layers.0.mlp.down_proj"

    def save(rank, alpha, ns):
        save_file({f"{module}.lora_A.weight": torch.ones(rank, 3), f"{module}.lora_B.weight": torch.ones(4, rank)},
                  weights)
        with open(config, "w") as f:
            json.dump({"r": rank, "lora_alpha": alpha}, f)
        for path in (weights, config):
            os.utime(path, ns=(ns, ns))
        return os.stat(weights), os.stat(config)

    read = save(2, 4, 10 ** 18)
    read_lora = pipe.read_lora

    def read_then_saved_again(path):
        source = read_lora(path)
        save(3, 8, 2 * 10 ** 18)
        return source

    monkeypatch.setattr(pipe, "read_lora", read_then_saved_again)
    spec = pipe.lora_spec(name, 1.0, [root], "Qwen LM")
    path = os.path.join(root, name)
    assert spec.key == (path, (read[0].st_mtime_ns, read[1].st_mtime_ns), read[0].st_size, 1.0)
    tensors = spec.build({LORA_MODULE: "model.layers.0.mlp.down_proj.weight"}, {"model.layers.0.mlp.down_proj.weight": (4, 3)})
    assert tensors[f"{LORA_MODULE}.alpha"].item() == 4 and tensors[f"{LORA_MODULE}.lora_A.weight"].shape == (2, 3)
    monkeypatch.setattr(pipe, "read_lora", read_lora)
    saved = os.stat(weights), os.stat(config)
    assert pipe.lora_spec(name, 1.0, [root], "Qwen LM").key == (path, (saved[0].st_mtime_ns, saved[1].st_mtime_ns),
                                                                 saved[0].st_size, 1.0) != spec.key


def test_missing_lora_names_the_folders(pipe, tmp_path):
    with pytest.raises(FileNotFoundError, match=f"LoRA gone.safetensors is not in {tmp_path}"):
        pipe.lora_spec("gone.safetensors", 1.0, [str(tmp_path)], "Qwen LM")


def test_run_hands_the_lora_to_the_backend(pipe, backend, folders):
    save_lora(os.path.join(folders[0], "loras", "x.safetensors"))
    pipe.run(request(pipe, folders, lora="x.safetensors", lora_strength=0.8))
    lora = backend.loaded[1]
    assert lora.key[0] == os.path.join(folders[0], "loras", "x.safetensors") and lora.strength == 0.8


def test_lora_choices_merge_the_folders(pipe, folders):
    a, b, _ = folders
    save_lora(os.path.join(a, "loras", "z.safetensors"))
    save_lora(os.path.join(b, "loras", "a.safetensors"))
    save_lora(os.path.join(b, "loras", "z.safetensors"))
    assert pipe.lora_choices("qwen_lm", (str(a), str(b))) == ["None", "a.safetensors", "z.safetensors"]
    assert pipe.lora_choices("qwen_lm", ()) == ["None"]


# --- sampling ------------------------------------------------------------------------------------------------


@pytest.fixture
def models(bcnodes):
    family = bcnodes["models.qwen_lm"].FAMILY
    return bcnodes["models.common.lm.catalog"].load_catalog(family.catalog_path, [], family.templates,
                                                           default_backend=family.default_backend)


def test_defaults_per_mode(pipe, models):
    assert pipe.generate_params(models["Qwen3.5-9B"], False, {}, 3, 10, "Qwen LM") == pipe.GenerateParams(
        seed=3, max_new_tokens=10, **OFF_9B)
    assert pipe.generate_params(models["Qwen3.5-9B"], True, {}, 3, 10, "Qwen LM") == pipe.GenerateParams(
        seed=3, max_new_tokens=10, **ON_9B)
    assert pipe.generate_params(models["Qwen3.8-27B"], True, {}, 3, 10, "Qwen LM").presence_penalty == 0.0


def test_edited_fields_over_the_defaults(pipe, models):
    config = {"temperature": 0.3, "top_k": 0, "do_sample": False, "mtp": "4"}
    params = pipe.generate_params(models["Qwen3.5-9B"], True, config, 3, 10, "Qwen LM")
    assert params == pipe.GenerateParams(seed=3, max_new_tokens=10, **dict(ON_9B, **config))


def test_mtp_off_without_an_mtp_head(pipe, models, caplog):
    vl, abliterated = models["Qwen3-VL-8B Instruct"], models["Qwen3.5-9B Abliterated"]
    with caplog.at_level(logging.WARNING):
        assert pipe.generate_params(vl, False, {}, 0, 1, "Qwen LM").mtp == "off"  # default auto: no warning
        assert pipe.generate_params(abliterated, False, {"mtp": "off"}, 0, 1, "Qwen LM").mtp == "off"
    assert caplog.records == []
    with caplog.at_level(logging.WARNING):
        assert pipe.generate_params(abliterated, False, {"mtp": "3"}, 0, 1, "Qwen LM").mtp == "off"
    assert [r.getMessage() for r in caplog.records] == [
        "[BCNodes] Qwen LM: mtp is 3 in LM Config, but Qwen3.5-9B Abliterated has no MTP head: it runs without MTP"]


@pytest.mark.parametrize("config, match", [
    ({"temprature": 0.5}, "unknown field\\(s\\) temprature; the fields are do_sample, temperature, top_p, top_k"),
    ({"mtp": "auto!"}, "mtp is 'auto!'; use one of auto, off, 2, 3, 4, 5"),
])
def test_config_errors(pipe, models, config, match):
    with pytest.raises(ValueError, match=match):
        pipe.generate_params(models["Qwen3.5-9B"], False, config, 0, 1, "Qwen LM")


def test_run_overlays_the_config(pipe, backend, folders):
    pipe.run(request(pipe, folders, thinking=True, config={"presence_penalty": 0.5}, seed=99, max_new_tokens=5))
    assert backend.generated[3] == pipe.GenerateParams(seed=99, max_new_tokens=5, **dict(ON_9B, presence_penalty=0.5))


# --- the catalog the node and the route show --------------------------------------------------------------


def test_catalog_with_user_precisions(pipe, folders):
    (folders[1] / "models.yaml").write_text(
        "models:\n  Qwen3.5-4B:\n    precisions:\n      FP8: {file: q4_fp8.safetensors}\n")
    view = pipe.catalog("qwen_lm", tuple(map(str, folders[:2])))
    assert view.error is None and view.precisions == ("BF16", "INT8 ConvRot", "W4A8", "FP8")
    assert list(view.models["Qwen3.5-4B"].precisions) == ["BF16", "INT8 ConvRot", "FP8"]


def test_user_catalogs_apply_in_folder_order(pipe, backend, folders, caplog):
    """The models.yaml of every model folder, in folder_paths order: a later folder's over an earlier one's,
    one console line per override in that order; swapped folders, the other one wins."""
    a, b = folders[0], folders[1]
    for folder, temperature in ((a, 0.5), (b, 0.4)):
        (folder / "models.yaml").write_text(
            f"models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {{temperature: {temperature}}}\n")
    for order, temperature in (((a, b), 0.4), ((b, a), 0.5)):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            view = pipe.catalog("qwen_lm", tuple(map(str, order)))
        assert view.error is None and view.models["Qwen3.5-9B"].defaults["thinking_off"].temperature == temperature
        lines = [r.getMessage() for r in caplog.records if " overrides " in r.getMessage()]
        assert [line.partition(" overrides ")[0] for line in lines] == [
            f"[BCNodes] LM catalog: {folder / 'models.yaml'}" for folder in order]
        assert all(line.endswith(" overrides Qwen3.5-9B: defaults.thinking_off.temperature") for line in lines)
        pipe.run(request(pipe, folders, model_folders=tuple(map(str, order))))
        assert backend.generated[3].temperature == temperature


def test_broken_user_catalog(pipe, backend, folders):
    path = folders[0] / "models.yaml"
    path.write_text("models:\n  Qwen3.5-9B:\n    tempo: 3\n")
    view = pipe.catalog("qwen_lm", (str(folders[0]),))
    assert view.error.startswith(f"{path}: models.Qwen3.5-9B: unknown key(s) tempo")
    assert list(view.models) == list(pipe.catalog("qwen_lm", ()).models)  # the built-in catalog
    with pytest.raises(ValueError, match="unknown key\\(s\\) tempo") as e:
        pipe.run(request(pipe, folders))
    assert type(e.value).__name__ == "CatalogError"


def test_catalog_read_again_only_when_a_file_changes(pipe, folders, caplog):
    path = folders[0] / "models.yaml"
    path.write_text("models:\n  Qwen3.5-9B:\n    mtp: false\n")
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            assert pipe.catalog("qwen_lm", (str(folders[0]),)).models["Qwen3.5-9B"].mtp is False
        lines = [r.getMessage() for r in caplog.records if "overrides" in r.getMessage()]
        assert len(lines) == 1  # one console line per override, not one per widget refresh
        path.write_text("models:\n  Qwen3.5-9B:\n    mtp: true \n")  # another size: read again
        assert pipe.catalog("qwen_lm", (str(folders[0]),)).models["Qwen3.5-9B"].mtp is True
        assert len([r for r in caplog.records if "overrides" in r.getMessage()]) == 2


def test_catalog_json(pipe, folders):
    out = pipe.catalog_json({"BC_QwenLM": ("qwen_lm", (str(folders[0]),))})
    json.dumps(out)  # the route serialises it as it is
    family = out["qwen_lm"]
    assert family["node"] == "BC_QwenLM" and family["error"] is None
    assert list(family) == list(pipe.FamilyInfo.__annotations__)  # the TypedDicts, keys in their order
    assert all(list(m) == list(pipe.ModelInfo.__annotations__) for m in family["models"].values())
    assert set(OFF_9B) == set(pipe.LMConfigValues.__annotations__) and pipe.LMConfigValues.__total__ is False
    assert list(family["models"]) == ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen3.5-9B Abliterated", "Qwen3-VL-8B Instruct",
                                      "Qwen3-VL-8B Instruct Abliterated", "Qwen3.8-27B"]
    assert family["models"]["Qwen3.5-9B"] == {"thinking": True, "mtp": True, "precisions": ["BF16", "INT8 ConvRot"],
                                              "defaults": {"thinking_off": OFF_9B, "thinking_on": ON_9B}}
    assert list(family["models"]["Qwen3.5-9B"]["defaults"]["thinking_off"]) == list(OFF_9B)
    vl = family["models"]["Qwen3-VL-8B Instruct"]
    assert (vl["thinking"], vl["mtp"], list(vl["defaults"])) == (False, False, ["thinking_off"])
    assert family["models"]["Qwen3.8-27B"]["precisions"] == ["BF16", "INT8 ConvRot", "W4A8"]

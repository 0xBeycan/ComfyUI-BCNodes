"""models/common/lm/catalog.py on the Qwen LM family's built-in models.yaml and user files:

  - the built-in catalog is the build spec's model table (section 4): the six models in order, their
    template, thinking and MTP flags, the core backend, every precision's file and URL in the order
    BF16, INT8 ConvRot, W4A8, and the official sampling defaults per mode (thinking_on only on a
    thinking model), each defaults block under its source comment;
  - a user models.yaml overrides field by field (defaults per mode and sampling field) and precisions
    by name (replaced in place or added at the end), adds new models after the built-in ones, a later
    file over an earlier one, with one console line per overridden model naming the file and what
    changed (none for an empty entry); thinking: false on a thinking model drops its thinking_on
    defaults;
  - every error names the file and the entry: unknown keys, missing required keys, an unknown template
    or backend, thinking_on defaults on a model without thinking, values of the wrong type; a new model
    is checked complete in the file that adds it (a later file that only adds a precision is not named),
    and an error after the merge names every user file that wrote the entry (not the built-in file, nor
    a file with an empty entry); a file that is not UTF-8, is nested too deeply or holds a value YAML
    reads as a date or a number but cannot convert is an error naming it, as is a number too large for
    a float.
"""

import logging
import re

import pytest

URL = "https://huggingface.co/beycanai/Qwen-LM/resolve/main/"
OFF = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, mtp="auto")
ON = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0, mtp="auto")
ON_27B = {**ON, "presence_penalty": 0.0}
# the build spec's table: display name, template, thinking, mtp head, (BF16, INT8 ConvRot, W4A8) files, defaults
SPEC = [
    ("Qwen3.5-4B", "qwen3.5", True, True, ("Qwen3.5-4B_bf16.safetensors", "Qwen3.5-4B_int8_convrot.safetensors", None), (OFF, ON)),
    ("Qwen3.5-9B", "qwen3.5", True, True, ("Qwen3.5-9B_bf16.safetensors", "Qwen3.5-9B_int8_convrot.safetensors", None), (OFF, ON)),
    ("Qwen3.5-9B Abliterated", "qwen3.5", True, False,
     ("Qwen3.5-9B-abliterated_bf16.safetensors", "Qwen3.5-9B-abliterated_int8_convrot.safetensors", None), (OFF, ON)),
    ("Qwen3-VL-8B Instruct", "qwen3-vl", False, False,
     ("Qwen3-VL-8B-Instruct_bf16.safetensors", "Qwen3-VL-8B-Instruct_int8_convrot.safetensors", None), (OFF, None)),
    ("Qwen3-VL-8B Instruct Abliterated", "qwen3-vl", False, False,
     ("Qwen3-VL-8B-Instruct-abliterated_bf16.safetensors", "Qwen3-VL-8B-Instruct-abliterated_int8_convrot.safetensors", None), (OFF, None)),
    ("Qwen3.8-27B", "qwen3.8", True, True,
     ("Qwen3.8-27B_bf16.safetensors", "Qwen3.8-27B_int8_convrot.safetensors", "Qwen3.8-27B_w4a8.safetensors"), (OFF, ON_27B)),
]
SOURCES = {
    "Qwen3.5-4B": "# source: Qwen/Qwen3.5-9B README lines 833-836 (general tasks; the 4B card is identical)",
    "Qwen3.5-9B": "# source: Qwen/Qwen3.5-9B README lines 833-836 (general tasks)",
    "Qwen3.5-9B Abliterated": "# source: the base model's, Qwen/Qwen3.5-9B README lines 833-836 (general tasks)",
    "Qwen3-VL-8B Instruct": "# source: Qwen/Qwen3-VL-8B-Instruct README lines 134-153 (min_p not given: 0.0 = disabled)",
    "Qwen3-VL-8B Instruct Abliterated": "# source: the base model's, Qwen/Qwen3-VL-8B-Instruct README lines 134-153 (min_p not given: 0.0 = disabled)",
    "Qwen3.8-27B": "# source: Qwen/Qwen3.8-27B README lines 246-264",
}


@pytest.fixture(scope="module")
def cat(bcnodes):
    return bcnodes["models.common.lm.catalog"]


@pytest.fixture(scope="module")
def family(bcnodes):
    return bcnodes["models.qwen_lm"].FAMILY


@pytest.fixture
def load(cat, family):
    def run(*user_files, builtin=None):
        return cat.load_catalog(builtin or family.catalog_path, [str(f) for f in user_files], family.templates,
                                default_backend=family.default_backend)
    return run


@pytest.fixture
def user(tmp_path):
    """Writes a user models.yaml (in its own folder, as each Qwen-LM folder holds one) and returns its path."""
    count = [0]

    def write(text):
        count[0] += 1
        folder = tmp_path / f"Qwen-LM-{count[0]}"
        folder.mkdir()
        path = folder / "models.yaml"
        path.write_text(text, encoding="utf-8")
        return path
    return write


def sampling(model, mode):
    s = model.defaults[mode]
    return {f: getattr(s, f) for f in ("do_sample", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "mtp")}


def test_builtin_catalog_is_the_spec_table(load):
    catalog = load()
    assert list(catalog) == [row[0] for row in SPEC]
    for name, template, thinking, mtp, files, (off, on) in SPEC:
        model = catalog[name]
        assert (model.name, model.template, model.thinking, model.mtp, model.backend) == (name, template, thinking, mtp, "core")
        expected = [(p, f) for p, f in zip(("BF16", "INT8 ConvRot", "W4A8"), files) if f]
        assert [(p.name, p.file, p.url) for p in model.precisions.values()] == [(p, f, URL + f) for p, f in expected]
        assert list(model.precisions) == [p for p, _ in expected]
        assert list(model.defaults) == (["thinking_off", "thinking_on"] if on else ["thinking_off"])
        assert sampling(model, "thinking_off") == off
        if on:
            assert sampling(model, "thinking_on") == on


def test_builtin_values_have_their_types(load):
    for model in load().values():
        for s in model.defaults.values():
            assert type(s.do_sample) is bool and type(s.top_k) is int and type(s.mtp) is str
            assert all(type(getattr(s, f)) is float for f in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"))


def test_each_defaults_block_names_its_source(family):
    lines = open(family.catalog_path, encoding="utf-8").read().splitlines()
    starts = {line.strip()[:-1]: i for i, line in enumerate(lines) if re.match(r"^  \S.*:$", line)}
    assert list(starts) == list(SOURCES)
    for name, comment in SOURCES.items():
        i = starts[name]
        block = lines[i:i + 8]
        defaults = block.index("    defaults:")
        assert block[defaults + 1].strip() == comment, name


def test_the_family_entry(family, bcnodes):
    registry = bcnodes["models.common.registry"]
    assert registry.get(registry.LM_FAMILY, "qwen_lm") is family
    assert (family.key, family.folder, family.lora_subfolder, family.default_backend) == ("qwen_lm", "Qwen-LM", "loras", "core")
    assert family.templates == ("qwen3.5", "qwen3.8", "qwen3-vl") and family.catalog_path.endswith("models.yaml")


def test_override_field_by_field(load, user, caplog):
    path = user("""
models:
  Qwen3.5-9B:
    mtp: false
    defaults:
      thinking_off: {temperature: 0.5, mtp: off}
      thinking_on: {top_k: 40}
""")
    with caplog.at_level(logging.INFO):
        model = load(path)["Qwen3.5-9B"]
    assert model.mtp is False and model.template == "qwen3.5" and model.thinking is True
    assert sampling(model, "thinking_off") == {**OFF, "temperature": 0.5, "mtp": "off"}
    assert sampling(model, "thinking_on") == {**ON, "top_k": 40}
    assert [p.file for p in model.precisions.values()] == ["Qwen3.5-9B_bf16.safetensors", "Qwen3.5-9B_int8_convrot.safetensors"]
    lines = [r.getMessage() for r in caplog.records if "LM catalog" in r.getMessage()]
    assert lines == [f"[BCNodes] LM catalog: {path} overrides Qwen3.5-9B: mtp, defaults.thinking_off.temperature, "
                     "defaults.thinking_off.mtp, defaults.thinking_on.top_k"]


def test_override_precisions_by_name(load, user, caplog):
    path = user("""
models:
  Qwen3.8-27B:
    precisions:
      INT8 ConvRot: {file: my_int8.safetensors}
      FP8: {file: Qwen3.8-27B_fp8.safetensors, url: "https://example.com/Qwen3.8-27B_fp8.safetensors"}
""")
    with caplog.at_level(logging.INFO):
        model = load(path)["Qwen3.8-27B"]
    assert [(p.name, p.file, p.url) for p in model.precisions.values()] == [
        ("BF16", "Qwen3.8-27B_bf16.safetensors", URL + "Qwen3.8-27B_bf16.safetensors"),
        ("INT8 ConvRot", "my_int8.safetensors", None),
        ("W4A8", "Qwen3.8-27B_w4a8.safetensors", URL + "Qwen3.8-27B_w4a8.safetensors"),
        ("FP8", "Qwen3.8-27B_fp8.safetensors", "https://example.com/Qwen3.8-27B_fp8.safetensors"),
    ]
    assert sampling(model, "thinking_on") == ON_27B
    assert "overrides Qwen3.8-27B: precisions.INT8 ConvRot (replaced), precisions.FP8 (added)" in caplog.text


def test_new_model_and_file_order(load, user, caplog):
    first = user("""
models:
  My Qwen:
    template: qwen3.5
    thinking: false
    mtp: false
    defaults:
      thinking_off: {do_sample: false, temperature: 1, top_p: 1, top_k: 0, min_p: 0, presence_penalty: 0, repetition_penalty: 1.1, mtp: 2}
    precisions:
      BF16: {file: my_qwen.safetensors}
  Qwen3.5-4B:
    defaults:
      thinking_off: {temperature: 0.6}
""")
    second = user("""
models:
  Qwen3.5-4B:
    defaults:
      thinking_off: {temperature: 0.4}
  My Qwen:
    backend: core
""")
    with caplog.at_level(logging.INFO):
        catalog = load(first, second)
    assert list(catalog) == [row[0] for row in SPEC] + ["My Qwen"]
    mine = catalog["My Qwen"]
    assert (mine.template, mine.thinking, mine.mtp, mine.backend, list(mine.defaults)) == ("qwen3.5", False, False, "core", ["thinking_off"])
    assert sampling(mine, "thinking_off") == dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
                                                  presence_penalty=0.0, repetition_penalty=1.1, mtp="2")
    assert [(p.name, p.file, p.url) for p in mine.precisions.values()] == [("BF16", "my_qwen.safetensors", None)]
    assert catalog["Qwen3.5-4B"].defaults["thinking_off"].temperature == 0.4
    lines = [r.getMessage() for r in caplog.records if "LM catalog" in r.getMessage()]
    assert lines == [f"[BCNodes] LM catalog: {first} overrides Qwen3.5-4B: defaults.thinking_off.temperature",
                     f"[BCNodes] LM catalog: {second} overrides Qwen3.5-4B: defaults.thinking_off.temperature",
                     f"[BCNodes] LM catalog: {second} overrides My Qwen: backend"]


def test_thinking_false_drops_thinking_on(load, user, caplog):
    path = user("models:\n  Qwen3.5-9B:\n    thinking: false\n")
    with caplog.at_level(logging.INFO):
        model = load(path)["Qwen3.5-9B"]
    assert model.thinking is False and list(model.defaults) == ["thinking_off"]
    assert [r.getMessage() for r in caplog.records if "LM catalog" in r.getMessage()] == [
        f"[BCNodes] LM catalog: {path} overrides Qwen3.5-9B: defaults.thinking_on (dropped: thinking false), thinking"]


def test_an_empty_entry_overrides_nothing(load, user, caplog):
    with caplog.at_level(logging.INFO):
        catalog = load(user("models:\n  Qwen3.5-9B: {}\n"))
    assert catalog == load() and "LM catalog" not in caplog.text


@pytest.mark.parametrize("text", ["", "# all commented out\n", "models:\n", "models: {}\n"])
def test_an_empty_user_file_changes_nothing(load, user, text):
    assert load(user(text)) == load()


def error(load, path, entry, match, builtin=None):
    with pytest.raises(ValueError, match=re.escape(f"{path}: {entry}") + ".*" + match) as info:
        load(*([] if builtin else [path]), builtin=builtin)
    assert type(info.value).__name__ == "CatalogError"
    return str(info.value)


GOOD = ("template: qwen3.5\n    thinking: true\n    mtp: false\n    defaults:\n"
        "      thinking_off: {do_sample: true, temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: auto}\n"
        "      thinking_on: {do_sample: true, temperature: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 1.5, repetition_penalty: 1.0, mtp: auto}\n"
        "    precisions:\n      BF16: {file: x.safetensors}\n")


def new_model(body):
    return "models:\n  New:\n    " + body


@pytest.mark.parametrize("text, entry, match", [
    # unknown keys
    ("models:\n  Qwen3.5-9B:\n    temprature: 0.5\n", "models.Qwen3.5-9B", r"unknown key\(s\) temprature; the keys are template, thinking"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_maybe: {}\n", "models.Qwen3.5-9B.defaults", r"unknown key\(s\) thinking_maybe"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_on: {topk: 3}\n", "models.Qwen3.5-9B.defaults.thinking_on", r"unknown key\(s\) topk"),
    ("models:\n  Qwen3.5-9B:\n    precisions:\n      BF16: {file: a.safetensors, sha256: abc}\n", "models.Qwen3.5-9B.precisions.BF16", r"unknown key\(s\) sha256"),
    # missing required keys
    (new_model("template: qwen3.5\n    thinking: false\n"), "models.New", "a new model needs mtp, defaults, precisions"),
    (new_model(GOOD.replace("      thinking_on:", "      #")), "models.New", "defaults.thinking_on is missing"),
    (new_model(GOOD.replace(" top_k: 20, min_p: 0.0,", "", 1)), "models.New", "defaults.thinking_off misses top_k, min_p"),
    (new_model(GOOD.replace("BF16: {file: x.safetensors}", "BF16: {url: \"https://e.com/x\"}")), "models.New.precisions.BF16", "file must be a plain file name"),
    # template, backend, thinking_on without thinking
    (new_model(GOOD.replace("qwen3.5", "qwen9")), "models.New", "template 'qwen9' is not one of this family's: qwen3.5, qwen3.8, qwen3-vl"),
    ("models:\n  Qwen3.5-9B:\n    backend: vllm\n", "models.Qwen3.5-9B", "backend 'vllm' is not one of: core"),
    (new_model(GOOD.replace("thinking: true", "thinking: false")), "models.New", "thinking_on defaults on a model with thinking: false"),
    ("models:\n  Qwen3-VL-8B Instruct:\n    defaults:\n      thinking_on: {top_k: 5}\n", "models.Qwen3-VL-8B Instruct",
     "thinking_on defaults on a model with thinking: false"),
    # values
    ("models:\n  Qwen3.5-9B:\n    thinking: \"yes\"\n", "models.Qwen3.5-9B", "thinking must be true or false"),
    ("models:\n  Qwen3.5-9B:\n    mtp: 1\n", "models.Qwen3.5-9B", "mtp must be true or false"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {do_sample: \"true\"}\n", "models.Qwen3.5-9B.defaults.thinking_off", "do_sample must be true or false"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {top_k: 20.5}\n", "models.Qwen3.5-9B.defaults.thinking_off", "top_k must be a whole number"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {temperature: hot}\n", "models.Qwen3.5-9B.defaults.thinking_off", "temperature must be a number"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {top_p: .nan}\n", "models.Qwen3.5-9B.defaults.thinking_off", "top_p must be a number"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {temperature: 1" + "0" * 400 + "}\n",  # too large for a float
     "models.Qwen3.5-9B.defaults.thinking_off", "temperature must be a number"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {mtp: 7}\n", "models.Qwen3.5-9B.defaults.thinking_off", "mtp must be one of auto, off, 2, 3, 4, 5"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: {mtp: on}\n", "models.Qwen3.5-9B.defaults.thinking_off", "mtp must be one of auto, off"),
    ("models:\n  Qwen3.5-9B:\n    defaults: [1]\n", "models.Qwen3.5-9B", "defaults must map thinking_off and thinking_on"),
    ("models:\n  Qwen3.5-9B:\n    defaults:\n      thinking_off: 3\n", "models.Qwen3.5-9B.defaults.thinking_off", "must be a mapping of do_sample"),
    ("models:\n  Qwen3.5-9B:\n    precisions: {}\n", "models.Qwen3.5-9B", "precisions must map each precision name"),
    ("models:\n  Qwen3.5-9B:\n    precisions:\n      BF16: x.safetensors\n", "models.Qwen3.5-9B.precisions.BF16", "must be a mapping"),
    ("models:\n  Qwen3.5-9B:\n    precisions:\n      BF16: {file: sub/x.safetensors}\n", "models.Qwen3.5-9B.precisions.BF16", "file must be a plain file name"),
    ("models:\n  Qwen3.5-9B:\n    precisions:\n      BF16: {file: x.safetensors, url: \"ftp://e.com/x\"}\n", "models.Qwen3.5-9B.precisions.BF16", "url must be an http"),
    ("models:\n  Qwen3.5-9B: 3\n", "models.Qwen3.5-9B", "the entry must be a mapping"),
])
def test_errors_name_the_file_and_the_entry(load, user, text, entry, match):
    error(load, user(text), entry, match)


@pytest.mark.parametrize("text, match", [
    ("Qwen3.5-9B:\n  thinking: true\n", "the file must hold one key, models:"),
    ("models: [a]\n", "the file must hold one key, models:"),
    ("models: {}\nextra: 1\n", "the file must hold one key, models:"),
    ("models:\n  \"\": {}\n", "model name '' must be a non-empty text"),
    ("models:\n  Qwen3.5-9B: {thinking: [}\n", "is not valid YAML"),
])
def test_file_errors_name_the_file(load, user, text, match):
    path = user(text)
    with pytest.raises(ValueError, match=re.escape(str(path)) + ".*" + match):
        load(path)


def test_an_unreadable_user_file(load, tmp_path):
    with pytest.raises(ValueError, match=re.escape(str(tmp_path / "missing.yaml")) + ": cannot be read"):
        load(tmp_path / "missing.yaml")


def merge_error(load, *paths):
    with pytest.raises(ValueError) as info:
        load(*paths)
    assert type(info.value).__name__ == "CatalogError"
    return str(info.value)


@pytest.mark.parametrize("text, match", [
    (new_model(GOOD.replace("thinking: true", "thinking: false")), "thinking_on defaults on a model with thinking: false"),
    (new_model(GOOD.replace(" top_k: 20, min_p: 0.0,", "", 1)), "defaults.thinking_off misses top_k, min_p"),
    (new_model(GOOD.replace("      thinking_on:", "      #")), "defaults.thinking_on is missing"),
], ids=["thinking_on without thinking", "a sampling field missing", "a mode missing"])
def test_a_new_model_is_checked_in_the_file_that_adds_it(load, user, text, match):
    # a later file that only adds a precision to it holds nothing to fix and is not named
    first, second = user(text), user("models:\n  New:\n    precisions:\n      INT8: {file: new_int8.safetensors}\n")
    assert merge_error(load, first, second).startswith(f"{first}: models.New: {match}")


def test_an_error_after_the_merge_names_every_user_file_that_wrote_the_entry(load, user):
    new = user(new_model(GOOD.replace("thinking: true", "thinking: false").replace("      thinking_on:", "      #")))
    on = user("models:\n  New:\n    thinking: true\n")
    assert merge_error(load, new, on) == f"{new}, {on}: models.New: defaults.thinking_on is missing"
    # a built-in model: its built-in entry, complete on its own, is not named, nor a file with an empty entry
    on = user("models:\n  Qwen3-VL-8B Instruct:\n    thinking: true\n")
    empty = user("models:\n  Qwen3-VL-8B Instruct: {}\n")
    added = user("models:\n  Qwen3-VL-8B Instruct:\n    precisions:\n      FP8: {file: vl_fp8.safetensors}\n")
    assert merge_error(load, on, empty, added) == f"{on}, {added}: models.Qwen3-VL-8B Instruct: defaults.thinking_on is missing"


def test_a_user_file_that_is_not_utf8(load, user):
    path = user("")
    path.write_bytes("# modèle perso\nmodels:\n".encode("cp1252"))  # as an older Windows editor saves it
    message = merge_error(load, path)
    assert message.startswith(f"{path}: is not UTF-8 text (") and message.endswith("); save it as UTF-8")


def test_a_user_file_nested_too_deeply(load, user):
    path = user("models:\n  Qwen3.5-9B: " + "[" * 1000 + "]" * 1000 + "\n")
    assert merge_error(load, path).startswith(f"{path}: is nested too deeply to read; write the entries at most "
                                              "four levels below models:")


@pytest.mark.parametrize("value", ["2026-13-01", "2026-01-01 00:00:00+25:00", "0x_"])
def test_a_value_yaml_cannot_convert(load, user, value):
    # YAML reads these as a date or a number and fails to build it with a plain ValueError
    path = user(f"models:\n  Qwen3.5-9B:\n    template: {value}\n")
    message = merge_error(load, path)
    assert message.startswith(f"{path}: holds a value YAML reads as a date or a number but cannot convert (")
    assert message.endswith("); put that value in quotes to make it text")


def test_builtin_errors(load, tmp_path, family):
    text = open(family.catalog_path, encoding="utf-8").read()
    path = tmp_path / "models.yaml"
    path.write_text(text.replace(', url: "https://huggingface.co/beycanai/Qwen-LM/resolve/main/Qwen3.5-4B_bf16.safetensors"', "", 1))
    error(load, path, "models.Qwen3.5-4B.precisions.BF16", "a built-in precision needs its url", builtin=path)
    path.write_text(text.replace("    template: qwen3.8\n", "", 1))
    error(load, path, "models.Qwen3.8-27B", "missing template", builtin=path)
    path.write_text("")
    with pytest.raises(ValueError, match="the file must hold one key, models:"):
        load(builtin=path)


def test_catalog_error_is_a_value_error(cat):
    assert issubclass(cat.CatalogError, ValueError)

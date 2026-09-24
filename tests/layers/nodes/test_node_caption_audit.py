"""Golden: Caption Audit end to end, through CaptionAudit().audit / IS_CHANGED, on the
caption_audit package of the ENV.

- audit() on a tmp dataset of five placeholder caption files (plus images, an orphan image, a dot
  file and a subfolder): an explicit trigger with class words and a fuse term; an inferred
  trigger with the stopword filter off (only the trigger token sits at >= 90% of the captions,
  so caption_audit's inference has no tie, which it would break by set order, i.e. by the hash
  seed); recursive with a separate images_dir, a declared class word missing from the captions
  and other thresholds. Recorded: report_text and report_json with the dataset dir replaced by
  /path/to/dataset (and the tmp root by /path/to), (critical, warning), the ui payload with the
  random file name masked, and the card digest with the "dir" row blanked (rows 119-140: the
  only place the card draws the tmp path; the file names in the card are caption stems). The
  saved preview PNG decodes to the returned IMAGE.
- The error paths, on placeholder paths that do not exist, so the whole result is pinned: a
  missing folder, thresholds out of order (warning > critical, info > warning), a folder outside
  the roots. Exact result tuples, the error card digest, the ui payload.
- IS_CHANGED: the exact value for a folder that does not exist (the path itself) and the
  "rejected:" text for one outside the roots; for the real dataset the value is the sha256 of
  "path|mtime_ns|size;" over the sorted .txt files (stated independently here; the path is the tmp
  dir, so there is no literal to record), and it follows caption edits only.
- save_temp_preview: the file name pattern, the payload, the PNG bytes, and the folder_paths-absent
  fallback (system temp dir, empty payload).

Seams: folder_paths attributes set to placeholder dirs under /path/to/comfyui and a per-test temp
dir; BC_CAPTION_ROOTS via monkeypatch.setenv. The card uses the fonts found on this machine
(test_pipe_caption_card pins which font file that is).
"""

import hashlib
import os
import re
import sys
import tempfile

import numpy as np
import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
    'caption_audit': '1.3.0',
    'caption_audit_py_md5': 'a7efde33511593897d021eece74d2c57',
}

GOLDEN = {
    'audit/explicit_trigger': 'cca9d4e8e11620cf47fcabdcb390f83e',
    'audit/inferred_trigger': 'cc7d217d7c5514e12a2eb482ac3b2827',
    'audit/recursive_images_dir': '5bbec76bcf7f030f5c68beca9ffb4875',
    'audit_error/missing_folder': "({'images': [{'filename': '<name>', 'subfolder': '', 'type': 'temp'}]}, '359cfcfdaa45d50984abb4cb6450b994', 'caption-audit: not a directory: /path/to/dataset', '{}', 1, 0)",
    'audit_error/warning_above_critical': "({'images': [{'filename': '<name>', 'subfolder': '', 'type': 'temp'}]}, '788c1dacd4861c439236b4fdc855c1d0', 'caption-audit: need 0 < warn_threshold <= critical_threshold <= 1.0', '{}', 1, 0)",
    'audit_error/info_above_warning': "({'images': [{'filename': '<name>', 'subfolder': '', 'type': 'temp'}]}, 'a5ab2ed8508a8e6123eb764d9f779e9e', 'caption-audit: need 0 < info_threshold <= warn_threshold', '{}', 1, 0)",
    'audit_error/outside_the_roots': '({\'images\': [{\'filename\': \'<name>\', \'subfolder\': \'\', \'type\': \'temp\'}]}, \'8c7114aed0e63ae46772a6e116f73c26\', "caption-audit: /path/elsewhere/dataset is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (\':\'-separated) and restart ComfyUI.", \'{}\', 1, 0)',
    'is_changed/missing_folder/recursive_False': '/path/to/dataset',
    'is_changed/outside_the_roots/recursive_False': "rejected:/path/elsewhere/dataset is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (':'-separated) and restart ComfyUI.",
    'is_changed/missing_folder/recursive_True': '/path/to/dataset',
    'is_changed/outside_the_roots/recursive_True': "rejected:/path/elsewhere/dataset is outside the folders this node may read (/path/to/comfyui, /path/to/comfyui/output, /path/to/comfyui/user, /path/to). Put the dataset under one of them, or name its root in the BC_CAPTION_ROOTS environment variable (':'-separated) and restart ComfyUI.",
    'save_temp_preview/None': 'f2f03a20fa0e6bf935910750230a2309',
    'save_temp_preview/x_prefix': 'f2f03a20fa0e6bf935910750230a2309',
    'save_temp_preview/no_folder_paths': "({'images': []}, 'f2f03a20fa0e6bf935910750230a2309')",
}

WHERE = Where({
    "save_temp_preview": "nodes.caption_audit:save_temp_preview",
})

ROOTS_ENV = "BC_CAPTION_ROOTS"
BG = (18, 20, 26)  # the card background
DIR_ROWS = slice(119, 141)  # the card's "dir" row
NAME = re.compile(r"^caption_audit_[0-9a-f]{12}\.png$")

CAPTIONS = {
    "img_01.txt": "<trigger>, a person, red scarf, studio light, term_a",
    "img_02.txt": "<trigger>, a person, red scarf, outdoors, term_a",
    "img_03.txt": "<trigger>, a person, red scarf, term_b",
    "img_04.txt": "<trigger>, a person wearing term_c, red scarf",
    "img_05.txt": "<trigger>, term_a, studio light",
    ".hidden.txt": "<trigger>, ignored",
    "sub/img_07.txt": "<trigger>, a person, term_d",
}
IMAGES = ["img_01.png", "img_02.png", "img_03.png", "img_04.png", "img_06.png"]
IMAGES_DIR = ["img_01.png", "img_05.png", "img_07.png"]

WIDGETS = dict(trigger="", class_words="", fuse="", critical_threshold=0.85, warn_threshold=0.60, info_threshold=0.35,
               ngram_max=3, no_stopwords=False, recursive=False, table_rows=12)

AUDIT_CASES = {
    "explicit_trigger": dict(trigger="<trigger>", class_words="person", fuse="red scarf"),
    "inferred_trigger": dict(no_stopwords=True, table_rows=5),
    "recursive_images_dir": dict(trigger="<trigger>", class_words="person, term_z", recursive=True, images_dir="images",
                                 critical_threshold=0.9, warn_threshold=0.5, info_threshold=0.3, ngram_max=2, table_rows=1),
}

ERROR_CASES = {
    "missing_folder": dict(directory="/path/to/dataset"),
    "warning_above_critical": dict(directory="/path/to/dataset", warn_threshold=0.9),
    "info_above_warning": dict(directory="/path/to/dataset", info_threshold=0.7),
    "outside_the_roots": dict(directory="/path/elsewhere/dataset"),
}


def _png(path, seed):
    from PIL import Image

    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (4, 4, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def env(bcnodes, monkeypatch, tmp_path):
    """folder_paths with placeholder roots and tmp_path/temp as its temp dir; the dataset under
    tmp_path/dataset, images under tmp_path/images. Returns (node module, dataset, temp dir)."""
    fp = sys.modules["folder_paths"]
    temp = tmp_path / "temp"
    monkeypatch.setattr(fp, "models_dir", "/path/to/comfyui/models", raising=True)
    monkeypatch.setattr(fp, "get_output_directory", lambda: "/path/to/comfyui/output", raising=True)
    monkeypatch.setattr(fp, "get_user_directory", lambda: "/path/to/comfyui/user", raising=True)
    monkeypatch.setattr(fp, "get_temp_directory", lambda: str(temp), raising=True)
    dataset = tmp_path / "dataset"
    for rel, text in CAPTIONS.items():
        (dataset / rel).parent.mkdir(parents=True, exist_ok=True)
        (dataset / rel).write_text(text, encoding="utf-8")
    for i, name in enumerate(IMAGES):
        _png(dataset / name, i)
    (tmp_path / "images").mkdir()
    for i, name in enumerate(IMAGES_DIR):
        _png(tmp_path / "images" / name, 10 + i)
    return bcnodes["caption_audit"], dataset, temp


def _ui(ui):
    """The ui payload with each file name checked against the pattern and masked."""
    images = ui["images"]
    assert all(NAME.match(item["filename"]) for item in images), images
    return {"images": [dict(item, filename="<name>") for item in images]}


def _preview(temp, image):
    """The one PNG in the temp dir, decoded, equals the returned IMAGE."""
    from PIL import Image

    (name,) = os.listdir(temp)
    with Image.open(temp / name) as im:
        pixels = np.asarray(im.convert("RGB"))
    assert np.array_equal(pixels, np.round(image[0].numpy() * 255).astype(np.uint8))


@pytest.mark.parametrize("name", list(AUDIT_CASES))
def test_audit(name, env, monkeypatch, tmp_path):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    m, dataset, temp = env
    monkeypatch.setenv(ROOTS_ENV, str(tmp_path))
    kwargs = dict(WIDGETS, **AUDIT_CASES[name])
    if "images_dir" in kwargs:
        kwargs["images_dir"] = str(tmp_path / kwargs["images_dir"])
    out = m.CaptionAudit().audit(str(dataset), **kwargs)
    image, text, report_json, critical, warning = out["result"]
    _preview(temp, image)
    real_dataset, real_root = os.path.realpath(dataset), os.path.realpath(tmp_path)

    def scrub(s):
        return s.replace(real_dataset, "/path/to/dataset").replace(real_root, "/path/to")

    card = image.clone()
    card[:, DIR_ROWS] = torch.tensor(BG, dtype=torch.float32) / 255.0
    check(GOLDEN, f"audit/{name}", digest({
        "ui": _ui(out["ui"]), "text": scrub(text), "json": scrub(report_json), "critical": critical,
        "warning": warning, "card": digest(card)}))


@pytest.mark.parametrize("name", list(ERROR_CASES))
def test_audit_error(name, env, monkeypatch):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    m, _dataset, temp = env
    monkeypatch.setenv(ROOTS_ENV, "/path/to")
    kwargs = dict(WIDGETS, **ERROR_CASES[name])
    out = m.CaptionAudit().audit(**kwargs)
    image, *rest = out["result"]
    _preview(temp, image)
    check(GOLDEN, f"audit_error/{name}", repr((_ui(out["ui"]), digest(image), *rest)))


@pytest.mark.parametrize("name", ["missing_folder", "outside_the_roots"])
@pytest.mark.parametrize("recursive", [False, True])
def test_is_changed_placeholder(name, recursive, env, monkeypatch):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    m, _dataset, _temp = env
    monkeypatch.setenv(ROOTS_ENV, "/path/to")
    value = m.CaptionAudit.IS_CHANGED(**dict(WIDGETS, recursive=recursive, directory=ERROR_CASES[name]["directory"]))
    check(GOLDEN, f"is_changed/{name}/recursive_{recursive}", value)


def _fingerprint(directory, recursive):
    """IS_CHANGED's value for a real folder: sha256 over "path|mtime_ns|size;" of the sorted
    (dir, name) pairs whose name ends in .txt (any case) and does not start with a dot."""
    if recursive:
        pairs = [(d, f) for d, _, names in os.walk(directory) for f in names]
    else:
        pairs = [(directory, f) for f in os.listdir(directory)]
    h = hashlib.sha256()
    for d, f in sorted(pairs):
        if f.lower().endswith(".txt") and not f.startswith("."):
            st = os.stat(os.path.join(d, f))
            h.update(f"{os.path.join(d, f)}|{st.st_mtime_ns}|{st.st_size};".encode())
    return h.hexdigest()


@pytest.mark.parametrize("recursive", [False, True])
def test_is_changed_dataset(recursive, env, monkeypatch, tmp_path):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    m, dataset, _temp = env
    monkeypatch.setenv(ROOTS_ENV, str(tmp_path))
    real = os.path.realpath(dataset)

    def is_changed(directory=str(dataset)):
        return m.CaptionAudit.IS_CHANGED(**dict(WIDGETS, recursive=recursive, directory=directory))

    first = is_changed()
    assert first == _fingerprint(real, recursive)
    monkeypatch.chdir(dataset)
    assert is_changed("") == first, "'' is the cwd"
    _png(dataset / "img_01.png", 99)
    (dataset / ".hidden.txt").write_text("<trigger>, still ignored, longer", encoding="utf-8")
    assert is_changed() == first, "only non-dot .txt files count"
    (dataset / "sub" / "img_07.txt").write_text("<trigger>, a person, term_d, term_e", encoding="utf-8")
    assert (is_changed() != first) == recursive, "a caption in a subfolder counts only when recursive"
    (dataset / "img_02.txt").write_text("<trigger>, a person, red scarf", encoding="utf-8")
    assert is_changed() != first and is_changed() == _fingerprint(real, recursive)


@pytest.mark.parametrize("prefix", [None, "x_prefix"])
def test_save_temp_preview(prefix, env):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    _m, _dataset, temp = env
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(5).integers(0, 256, (24, 32, 3), dtype=np.uint8))
    ui, path = WHERE["save_temp_preview"](img) if prefix is None else WHERE["save_temp_preview"](img, prefix)
    name = os.path.basename(path)
    assert re.match(rf"^{prefix or 'caption_audit'}_[0-9a-f]{{12}}\.png$", name) and os.path.dirname(path) == str(temp)
    assert ui == {"images": [{"filename": name, "subfolder": "", "type": "temp"}]}
    with open(path, "rb") as f:
        check(GOLDEN, f"save_temp_preview/{prefix}", hashlib.md5(f.read()).hexdigest())


def test_save_temp_preview_without_folder_paths(env, monkeypatch):
    check_env(ENV, "torch", "numpy", "Pillow", "caption_audit")
    from PIL import Image

    monkeypatch.setitem(sys.modules, "folder_paths", None)
    img = Image.fromarray(np.random.default_rng(5).integers(0, 256, (24, 32, 3), dtype=np.uint8))
    ui, path = WHERE["save_temp_preview"](img)
    try:
        assert os.path.dirname(path) == tempfile.gettempdir() and NAME.match(os.path.basename(path))
        with open(path, "rb") as f:
            check(GOLDEN, "save_temp_preview/no_folder_paths", repr((ui, hashlib.md5(f.read()).hexdigest())))
    finally:
        os.remove(path)

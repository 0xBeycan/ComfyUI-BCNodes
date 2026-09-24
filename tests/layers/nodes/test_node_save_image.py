"""Golden: Save Image end to end, through SaveImage().save_images.

- Every base format (the node's output_ext combo with the AVIF / JXL plugins reported absent)
  x quality {0, 50, 90, 100} x save_metadata {True, False}, on the widget defaults with a
  placeholder prompt: every file written (sorted relative path, raw bytes md5 | decoded pixel
  md5) and the ui payload. Quality 0 is mapped to DEFAULT_QUALITY by libs/image_write.py:
  write_image.
- The other save flows of the same method (pipelines/save_image.py:save_batch): jobs.json and one
  JSON per image, counter first / named keys / a multi-character delimiter, counter_digits 0, no
  name at all, image_preview off, a second run continuing the counter, no prompt, None and empty
  batches.

Seams: `datetime` of nodes/save_image.py (a module global read at call time; it stays in the
node) is patched to a fixed now(); importlib.util.find_spec reports pillow_avif / pillow_jxl
absent; folder_paths.get_output_directory points at the test's own tmp dir.
"""

import importlib.util
import os
import pathlib
import sys
from datetime import datetime

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
}

GOLDEN = {
    '.webp/q0/metadata_True': '0fc755dee84d7f8be4a8c4c17c8cbab2',
    '.webp/q0/metadata_False': 'c1c5f548b452c5e91f5e860ea941dfd3',
    '.webp/q50/metadata_True': '2ac59d141a85ceab827480699b0a6fde',
    '.webp/q50/metadata_False': '1db5862a88a81c45ba4e81119b8c147b',
    '.webp/q90/metadata_True': '0fc755dee84d7f8be4a8c4c17c8cbab2',
    '.webp/q90/metadata_False': 'c1c5f548b452c5e91f5e860ea941dfd3',
    '.webp/q100/metadata_True': 'ed50fa1de4b48db33680bdfca2824886',
    '.webp/q100/metadata_False': 'f1c880ffa2f7f41451c49b4d76a4fa92',
    '.png/q0/metadata_True': 'dd78574ed6d675223016b97cf796f164',
    '.png/q0/metadata_False': '96f282610a420d83fb08c5f0fca7d92d',
    '.png/q50/metadata_True': 'b4180002644eb96efd01f6864eda9740',
    '.png/q50/metadata_False': '5dbba02a725fec8568c13e25378a6596',
    '.png/q90/metadata_True': 'dd78574ed6d675223016b97cf796f164',
    '.png/q90/metadata_False': '96f282610a420d83fb08c5f0fca7d92d',
    '.png/q100/metadata_True': 'dd78574ed6d675223016b97cf796f164',
    '.png/q100/metadata_False': '96f282610a420d83fb08c5f0fca7d92d',
    '.jpg/q0/metadata_True': 'b4e1da160eb8fb57955d9d9962638139',
    '.jpg/q0/metadata_False': '6c83e13efe4bd341553b1f34c79b33b8',
    '.jpg/q50/metadata_True': 'f4de6b94b9bb6f9be333100cd294e45c',
    '.jpg/q50/metadata_False': '064e95bc14c625fe0ac1a1bf4c547170',
    '.jpg/q90/metadata_True': 'b4e1da160eb8fb57955d9d9962638139',
    '.jpg/q90/metadata_False': '6c83e13efe4bd341553b1f34c79b33b8',
    '.jpg/q100/metadata_True': '7c3bcb5ad53aa7d270f3eb0b68ebf920',
    '.jpg/q100/metadata_False': 'ba7db4da8cf2853e15557a1317c7ab34',
    '.jpeg/q0/metadata_True': '3141a26170e21b38d730a7a807ee651c',
    '.jpeg/q0/metadata_False': '2c0a48b9b673e32af6c141c57fb43332',
    '.jpeg/q50/metadata_True': 'a6dff5055e453f3c7c41e81c89938d24',
    '.jpeg/q50/metadata_False': '1c16183d4af8c468eda94a5322a89a08',
    '.jpeg/q90/metadata_True': '3141a26170e21b38d730a7a807ee651c',
    '.jpeg/q90/metadata_False': '2c0a48b9b673e32af6c141c57fb43332',
    '.jpeg/q100/metadata_True': '5d9d0089f9aa20bad5c1569ba676f48d',
    '.jpeg/q100/metadata_False': '270e3ad66ad8e3bc228e47fac1dcde2f',
    '.j2k/q0/metadata_True': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q0/metadata_False': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q50/metadata_True': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q50/metadata_False': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q90/metadata_True': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q90/metadata_False': 'd5ce38c4bc88114f4fd796a65df89c16',
    '.j2k/q100/metadata_True': '35d88c4b800e5d6b2fa02a1cf106faf2',
    '.j2k/q100/metadata_False': '35d88c4b800e5d6b2fa02a1cf106faf2',
    '.jp2/q0/metadata_True': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q0/metadata_False': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q50/metadata_True': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q50/metadata_False': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q90/metadata_True': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q90/metadata_False': '447cfdc6463380a0a4307472fe44b4d3',
    '.jp2/q100/metadata_True': '7bf17fd06216db912f26e111a54777c3',
    '.jp2/q100/metadata_False': '7bf17fd06216db912f26e111a54777c3',
    '.gif/q0/metadata_True': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q0/metadata_False': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q50/metadata_True': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q50/metadata_False': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q90/metadata_True': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q90/metadata_False': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q100/metadata_True': 'ad96c274915de7f0836b5d385f8fedf6',
    '.gif/q100/metadata_False': 'ad96c274915de7f0836b5d385f8fedf6',
    '.tiff/q0/metadata_True': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q0/metadata_False': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q50/metadata_True': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q50/metadata_False': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q90/metadata_True': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q90/metadata_False': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q100/metadata_True': 'ddb5c930c5e6c9271e70872b75be2640',
    '.tiff/q100/metadata_False': 'ddb5c930c5e6c9271e70872b75be2640',
    '.bmp/q0/metadata_True': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q0/metadata_False': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q50/metadata_True': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q50/metadata_False': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q90/metadata_True': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q90/metadata_False': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q100/metadata_True': '4af5b5079598faf33d58775841fb6a5e',
    '.bmp/q100/metadata_False': '4af5b5079598faf33d58775841fb6a5e',
    'flow/jobs_json': '5fb7a8150c1d149d50d131a2cee9db1f',
    'flow/job_data_per_image': '4d0238e211839b1313b0401c38851b5e',
    'flow/job_data_text_inputs': '8a8660133fc5b3b2ede02302348a927f',
    'flow/counter_first_named_keys': '50eaa661c89d1a54c619df2fc6a6e28a',
    'flow/counter_digits_0': '8649f22f6b76d0b76e571084bfd191c4',
    'flow/no_name': '0d51665b4c2d49afac17def7e0f29b27',
    'flow/no_name_first': 'dd6947cdaebf313846c27ec171102191',
    'flow/image_preview_off': '49f78357cff0a58d45f234f663f7415c',
    'flow/second_run_continues': '3f08576347b0a0e0ea65abff73e06093',
    'flow/no_prompt': 'd75ebe8f6e8bf414fb29632deb007b05',
    'flow/subfolder_keys': 'ac1cb89b4da14bc9f95d607fbbe3c875',
    'flow/none_batch': 'e8ebc0b85bb29cfddaaa97c453904ef4',
    'flow/empty_batch': 'e8ebc0b85bb29cfddaaa97c453904ef4',
}

WHERE = Where({
    "datetime": "nodes.save_image:datetime",
})

FORMATS = [".webp", ".png", ".jpg", ".jpeg", ".j2k", ".jp2", ".gif", ".tiff", ".bmp"]
QUALITIES = [0, 50, 90, 100]

PROMPT = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 123456, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        "positive": ["6", 0], "negative": ["7", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "family_a/model_a.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "<trigger>, a person, studio light", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, lowres", "clip": ["4", 1]}},
    "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "lora_a.safetensors", "strength_model": 0.8}},
}
EXTRA_PNGINFO = {"workflow": {"nodes": [{"id": 3, "type": "KSampler"}, {"id": 4, "type": "CheckpointLoaderSimple"}], "links": []}}

DEFAULTS = dict(
    filename_prefix="ComfyUI", filename_keys="sampler_name, cfg, steps, %F %H-%M-%S", foldername_prefix="",
    foldername_keys="ckpt_name", delimiter="-", save_job_data="disabled", job_data_per_image=False, job_custom_text="",
    save_metadata=True, counter_digits=4, counter_position="last", one_counter_per_folder=True, image_preview=True,
    output_ext=".webp", quality=90, named_keys=False,
)

JOB = dict(save_job_data="basic, models, sampler, prompt", job_custom_text="a note", output_ext=".png")

# name -> list of runs (overrides of DEFAULTS; "images": "batch" | "one" | "none" | "empty"; "prompt": False drops it)
FLOWS = {
    "jobs_json": [dict(JOB)],
    "job_data_per_image": [dict(JOB, job_data_per_image=True)],
    "job_data_text_inputs": [dict(JOB, positive_text_opt="wired <trigger> text", negative_text_opt="wired negative")],
    "counter_first_named_keys": [dict(counter_position="first", named_keys=True, delimiter="_x", output_ext=".png")],
    "counter_digits_0": [dict(counter_digits=0, output_ext=".png"), dict(counter_digits=0, output_ext=".png", images="one")],
    "no_name": [dict(filename_prefix="", filename_keys="", foldername_keys="", counter_digits=3, output_ext=".jpg")],
    "no_name_first": [dict(filename_prefix="", filename_keys="", foldername_keys="", counter_position="first", output_ext=".jpg")],
    "image_preview_off": [dict(image_preview=False, output_ext=".png")],
    "second_run_continues": [dict(output_ext=".png"), dict(output_ext=".png", images="one", save_job_data="prompt")],
    "no_prompt": [dict(output_ext=".png", prompt=False, save_job_data="basic, models, sampler, prompt")],
    "subfolder_keys": [dict(foldername_prefix="%Y", foldername_keys="/sampler_name, ../steps", filename_keys="'x', seed", output_ext=".png")],
    "none_batch": [dict(images="none")],
    "empty_batch": [dict(images="empty")],
}


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2024, 5, 22, 9, 13, 58)


def _images(kind):
    batch = torch.rand((2, 8, 12, 3), generator=torch.Generator().manual_seed(1))
    return {"batch": batch, "one": batch[:1], "none": None, "empty": batch[:0]}[kind]


def _files(root):
    out = []
    for d, _, names in os.walk(root):
        for name in names:
            path = pathlib.Path(d, name)
            out.append((path.relative_to(root).as_posix(), digest(path)))
    return sorted(out)


@pytest.fixture
def save(bcnodes, monkeypatch, tmp_path):
    """save(**overrides) -> the node's return value; output under tmp_path/output. save.root is
    tmp_path, so a `../` key that climbs out of output/ (by design, 12 B-18) is seen too."""
    output = str(tmp_path / "output")
    monkeypatch.setattr(sys.modules["folder_paths"], "get_output_directory", lambda: output, raising=True)
    WHERE.patch(monkeypatch, "datetime", _FixedDatetime)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, package=None: None if name in ("pillow_avif", "pillow_jxl") else real_find_spec(name, package),
                        raising=True)
    node = bcnodes["save_image"].SaveImage()
    assert node.INPUT_TYPES()["required"]["output_ext"][0] == FORMATS

    def run(**overrides):
        kwargs = dict(DEFAULTS, **overrides)
        images = _images(kwargs.pop("images", "batch"))
        prompt = PROMPT if kwargs.pop("prompt", True) else None
        return node.save_images(images, prompt=prompt, extra_pnginfo=EXTRA_PNGINFO, **kwargs)

    run.root = str(tmp_path)
    return run


@pytest.mark.parametrize("save_metadata", [True, False])
@pytest.mark.parametrize("quality", QUALITIES)
@pytest.mark.parametrize("ext", FORMATS)
def test_format(ext, quality, save_metadata, save):
    check_env(ENV, "torch", "numpy", "Pillow")
    ui = save(output_ext=ext, quality=quality, save_metadata=save_metadata)
    check(GOLDEN, f"{ext}/q{quality}/metadata_{save_metadata}", digest({"ui": ui, "files": _files(save.root)}))


@pytest.mark.parametrize("name", list(FLOWS))
def test_flow(name, save, capsys):
    check_env(ENV, "torch", "numpy", "Pillow")
    uis = [save(**run) for run in FLOWS[name]]
    stdout = capsys.readouterr().out
    check(GOLDEN, f"flow/{name}", digest({"ui": uis, "files": _files(save.root), "stdout": stdout}))

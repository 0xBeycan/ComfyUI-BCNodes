"""Golden: Social Media Export end to end, through SocialMediaExport().export.

- Every file written (sorted relative path, raw bytes md5 | decoded pixel md5), the
  folder_paths.get_save_image_path calls and the report, for resize_mode {crop, pad} x
  allow_upscale {False, True} with all 11 platforms of nodes/social_specs.json ticked, on a
  Generator(0) batch of two (2, 90, 160, 3) masters and on its first frame transposed (a
  portrait master). The images output is the input object itself.
- _tensor_to_pil for (8, 12), (8, 12, 1), (8, 12, 3), (8, 12, 4) tensors -> mode + pixels (the
  4-channel one composites onto white through the engine's flatten_to_rgb).
- INPUT_TYPES falls back to no platform checkbox when the specs JSON is missing or unreadable.

Seams: folder_paths.get_output_directory and get_save_image_path (a stand-in that counts the
files already written under the prefix, as ComfyUI does, so batch items do not overwrite each
other); PIL.ImageFile.MAXBLOCK is reset to 65536 before every case (process-global, 12 B-20);
the node's _SPECS_PATH (a module global read at call time) for the fallback. The embedded sRGB
profile carries its creation time (ICC header bytes 24-35), so ImageCmsProfile.tobytes is wrapped
to write a fixed time and the engine's _srgb_icc cache (reached through the node's `core`
import) is emptied before and after every case.
"""

import os
import pathlib
import struct
import sys

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
    'batch_2/crop/upscale_False': '94fef9fe39e45c566750171235ba5289',
    'batch_2/crop/upscale_True': 'fe80e0d0e793d4a9bb7df4fa142740ac',
    'batch_2/pad/upscale_False': '70fae8919aeb5c84102618f64b71feaa',
    'batch_2/pad/upscale_True': 'f825085f529190aa6e4f2f45c0feb86c',
    'transposed/crop/upscale_False': 'c1f881231a15552fdcb403bfbb5254c7',
    'transposed/crop/upscale_True': 'd473ff9355cf34ec849d67120c7b12a3',
    'transposed/pad/upscale_False': '0403a9529aa431bf351a307389463fad',
    'transposed/pad/upscale_True': '133a1e57fe3b3b451934c7e6a6cdd0fd',
    'tensor_to_pil/(8, 12)': "('RGB', (12, 8), '4ac25a610bb8f108b9f1b53532172d15')",
    'tensor_to_pil/(8, 12, 1)': "('RGB', (12, 8), '4ac25a610bb8f108b9f1b53532172d15')",
    'tensor_to_pil/(8, 12, 3)': "('RGB', (12, 8), '94072549afdb2457baf9c0bba67e0250')",
    'tensor_to_pil/(8, 12, 4)': "('RGB', (12, 8), '12707f321e9a29680d1c14052549f87f')",
    'input_types/missing': 'f568e3a8807ec02011225038635e0788',
    'input_types/unreadable': 'f568e3a8807ec02011225038635e0788',
}

WHERE = Where({
    "tensor_to_pil": "nodes.social_media_export:_tensor_to_pil",
    "SPECS_PATH": "nodes.social_media_export:_SPECS_PATH",
    "core": "nodes.social_media_export:core",
})

PLATFORMS = ["instagram_feed", "instagram_story", "x_timeline", "tiktok_photo", "tiktok_vertical", "pinterest_pin",
             "facebook_feed", "facebook_story", "threads_feed", "bluesky", "reddit"]


def _masters():
    batch = torch.rand((2, 90, 160, 3), generator=torch.Generator().manual_seed(0))
    return {"batch_2": batch, "transposed": batch[0].transpose(0, 1).unsqueeze(0).contiguous()}


def _files(root):
    out = []
    for d, _, names in os.walk(root):
        for name in names:
            path = pathlib.Path(d, name)
            out.append((path.relative_to(root).as_posix(), digest(path)))
    return sorted(out)


ICC_TIME = struct.pack(">6H", 2024, 1, 1, 0, 0, 0)  # the ICC header's dateTimeNumber


@pytest.fixture(autouse=True)
def maxblock(monkeypatch):
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile, "MAXBLOCK", 65536, raising=True)


@pytest.fixture(autouse=True)
def fixed_icc_time(bcnodes, monkeypatch):
    from PIL import ImageCms

    real_tobytes = ImageCms.ImageCmsProfile.tobytes

    def tobytes(self):
        data = real_tobytes(self)
        return data[:24] + ICC_TIME + data[36:]

    monkeypatch.setattr(ImageCms.ImageCmsProfile, "tobytes", tobytes, raising=True)
    WHERE["core"]._srgb_icc.cache_clear()
    yield
    WHERE["core"]._srgb_icc.cache_clear()


@pytest.fixture
def output(monkeypatch, tmp_path):
    """The output dir, and the get_save_image_path calls (prefix, output dir relative to it,
    width, height)."""
    root = str(tmp_path / "output")
    calls = []

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        calls.append((filename_prefix, os.path.relpath(output_dir, root), image_width, image_height))
        subfolder, filename = os.path.split(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(output_dir, subfolder)
        taken = os.listdir(full_output_folder) if os.path.isdir(full_output_folder) else []
        counter = 1 + sum(1 for f in taken if f.startswith(filename + "_"))
        return full_output_folder, filename, counter, subfolder, filename_prefix

    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_output_directory", lambda: root, raising=True)
    monkeypatch.setattr(fp, "get_save_image_path", get_save_image_path, raising=True)
    return root, calls


@pytest.mark.parametrize("allow_upscale", [False, True])
@pytest.mark.parametrize("resize_mode", ["crop", "pad"])
@pytest.mark.parametrize("master", ["batch_2", "transposed"])
def test_export(master, resize_mode, allow_upscale, bcnodes, output):
    check_env(ENV, "torch", "numpy", "Pillow")
    root, calls = output
    images = _masters()[master]
    result = bcnodes["social_media_export"].SocialMediaExport().export(
        images, resize_mode, 92, allow_upscale, "social/export", **{name: True for name in PLATFORMS})
    assert list(result) == ["result"] and result["result"][0] is images
    check(GOLDEN, f"{master}/{resize_mode}/upscale_{allow_upscale}",
          digest({"report": result["result"][1], "calls": calls, "files": _files(root)}))


@pytest.mark.parametrize("shape", [(8, 12), (8, 12, 1), (8, 12, 3), (8, 12, 4)])
def test_tensor_to_pil(shape, bcnodes):
    check_env(ENV, "torch", "numpy", "Pillow")
    image = torch.rand(shape, generator=torch.Generator().manual_seed(0))
    pil = WHERE["tensor_to_pil"](image)
    check(GOLDEN, f"tensor_to_pil/{shape}", repr((pil.mode, pil.size, digest(pil))))


@pytest.mark.parametrize("specs", ["missing", "unreadable"])
def test_input_types_fallback(specs, bcnodes, monkeypatch, tmp_path):
    check_env(ENV, "torch", "numpy", "Pillow")
    path = tmp_path / "social_specs.json"
    if specs == "unreadable":
        path.write_text("{ not json", encoding="utf-8")
    WHERE.patch(monkeypatch, "SPECS_PATH", str(path))
    check(GOLDEN, f"input_types/{specs}", digest(bcnodes["social_media_export"].SocialMediaExport.INPUT_TYPES()))

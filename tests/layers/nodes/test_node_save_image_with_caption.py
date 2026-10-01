"""Save Image With Caption (nodes/save_image_with_caption.py; its flow at the end of
pipelines/save_image.py), through the node method under the stub folder_paths:

- the surface: the inputs, defaults, outputs and category the owner asked for;
- each image saved as <prefix>_<counter>_.png with the caption next to it under the same name,
  the caption byte for byte, the pixels as ComfyUI's Save Image converts them (clipped, truncated),
  prompt and workflow in the PNG unless --disable-metadata;
- no caption connected: images only (the extension is then not read);
- no file is ever overwritten: an existing name is passed over (the stub's counter always starts
  at 1, as ComfyUI's does when %batch_num% is in the prefix);
- output_folder: `output`, a folder inside it, an absolute path, a path out of it refused before
  anything is written; a prefix `sub/` folder stays inside the target;
- the caption extension: the dot added, plain-text formats only, checked before anything is written;
- an empty batch writes nothing.

ComfyUI's own get_save_image_path (its counter continued from the folder, %year% ...) is used in
tests/test_runtime.py.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

CAPTION = "a photo of subject_a, smiling\nsecond line"


@pytest.fixture
def module(bcnodes):
    return bcnodes["save_image_with_caption"]


@pytest.fixture
def node(module):
    return module.SaveImageWithCaption()


@pytest.fixture
def calls():
    """The get_save_image_path calls (filename_prefix, output_dir, image_width, image_height)."""
    return []


@pytest.fixture
def output(tmp_path, monkeypatch, comfy_stubs, calls):
    """ComfyUI's output directory, tmp_path/output. The stub get_save_image_path is wrapped to
    create the folder, as ComfyUI's does when it is missing, and to record its calls."""
    fp = sys.modules["folder_paths"]
    out = tmp_path / "output"
    stub = fp.get_save_image_path

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        calls.append((filename_prefix, output_dir, image_width, image_height))
        result = stub(filename_prefix, output_dir, image_width, image_height)
        os.makedirs(result[0], exist_ok=True)
        return result

    monkeypatch.setattr(fp, "get_output_directory", lambda: str(out))
    monkeypatch.setattr(fp, "get_save_image_path", get_save_image_path)
    return out


def files(folder):
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


def test_the_surface(module):
    cls = module.SaveImageWithCaption
    types = cls.INPUT_TYPES()
    assert list(types["required"]) == ["images", "filename_prefix", "output_folder"]
    assert [spec[0] for spec in types["required"].values()] == ["IMAGE", "STRING", "STRING"]
    assert (types["required"]["filename_prefix"][1]["default"], types["required"]["output_folder"][1]["default"]) == ("ComfyUI", "output")
    assert list(types["optional"]) == ["caption_file_extension", "caption"]
    assert types["optional"]["caption_file_extension"][:1] == ("STRING",) and types["optional"]["caption_file_extension"][1]["default"] == ".txt"
    assert types["optional"]["caption"][0] == "STRING" and types["optional"]["caption"][1]["forceInput"] is True
    assert types["hidden"] == {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.OUTPUT_NODE, cls.CATEGORY) == (("STRING",), ("filename",), True, "BCNodes/image")
    assert module.NODE_CLASS_MAPPINGS == {"BC_SaveImageWithCaption": cls}
    assert module.NODE_DISPLAY_NAME_MAPPINGS == {"BC_SaveImageWithCaption": "Save Image With Caption"}


def test_each_image_gets_the_caption_under_its_name(node, output, calls):
    from PIL import Image

    images = torch.rand(2, 6, 5, 3) * 1.2 - 0.1  # out of range on both sides: clipped
    prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 5}}}
    workflow = {"id": "wf", "nodes": []}
    out = node.save_images(images, "set/shot", "output", ".txt", caption=CAPTION, prompt=prompt, extra_pnginfo={"workflow": workflow})
    assert out == ("shot_00002_.png",)
    assert calls == [("set/shot", str(output), 5, 6)]  # width, height of the first image
    folder = output / "set"
    assert files(folder) == ["shot_00001_.png", "shot_00001_.txt", "shot_00002_.png", "shot_00002_.txt"]
    for i in range(2):
        assert (folder / f"shot_0000{i + 1}_.txt").read_bytes() == CAPTION.encode("utf-8")
        with Image.open(folder / f"shot_0000{i + 1}_.png") as img:
            assert np.array_equal(np.array(img), np.clip(255.0 * images[i].numpy(), 0, 255).astype(np.uint8))
            assert img.text == {"prompt": json.dumps(prompt), "workflow": json.dumps(workflow)}


def test_no_caption_saves_the_images_only(node, output):
    assert node.save_images(torch.rand(1, 4, 4, 3), "shot", "output", "not read") == ("shot_00001_.png",)
    assert files(output) == ["shot_00001_.png"]


def test_disable_metadata_leaves_the_prompt_out(node, output, monkeypatch):
    from PIL import Image

    monkeypatch.setattr(sys.modules["comfy.cli_args"].args, "disable_metadata", True)
    node.save_images(torch.rand(1, 4, 4, 3), "shot", "output", prompt={"1": {}}, extra_pnginfo={"workflow": {}})
    with Image.open(output / "shot_00001_.png") as img:
        assert img.text == {}


def test_an_existing_file_is_never_overwritten(node, output):
    first = torch.zeros(2, 4, 4, 3)
    node.save_images(first, "shot", "output", caption="first")
    before = {name: (output / name).read_bytes() for name in files(output)}
    (output / "shot_00003_.txt").write_text("a caption of the owner's own")  # a caption without its image
    assert node.save_images(torch.ones(1, 4, 4, 3), "shot", "output", caption="second") == ("shot_00004_.png",)
    assert files(output) == ["shot_00001_.png", "shot_00001_.txt", "shot_00002_.png", "shot_00002_.txt", "shot_00003_.txt",
                             "shot_00004_.png", "shot_00004_.txt"]
    assert all((output / name).read_bytes() == data for name, data in before.items())
    assert (output / "shot_00003_.txt").read_text() == "a caption of the owner's own"
    assert (output / "shot_00004_.txt").read_text() == "second"


def test_batch_num_is_the_index_and_overwrites_nothing(node, output):
    node.save_images(torch.zeros(2, 4, 4, 3), "img%batch_num%", "output", caption="x")
    assert files(output) == ["img0_00001_.png", "img0_00001_.txt", "img1_00002_.png", "img1_00002_.txt"]
    node.save_images(torch.ones(2, 4, 4, 3), "img%batch_num%", "output", caption="y")
    assert files(output) == ["img0_00001_.png", "img0_00001_.txt", "img0_00002_.png", "img0_00002_.txt",
                             "img1_00002_.png", "img1_00002_.txt", "img1_00003_.png", "img1_00003_.txt"]
    assert (output / "img1_00002_.txt").read_text() == "x"


@pytest.mark.parametrize("value, sub", [("output", ""), ("", ""), ("./output", ""), ("output/set", "set"), ("set", "set"),
                                        ("output/set/../other", "other")])
def test_a_relative_output_folder_is_inside_the_output_directory(node, output, value, sub):
    node.save_images(torch.rand(1, 4, 4, 3), "shot", value)
    assert files(os.path.join(output, sub)) == ["shot_00001_.png"]


def test_an_absolute_output_folder_is_used_as_it_is_with_the_prefix_folder_inside(node, output, tmp_path):
    target = tmp_path / "datasets" / "set_a"  # does not exist yet
    assert node.save_images(torch.rand(1, 4, 4, 3), "sub/shot", str(target), caption="c") == ("shot_00001_.png",)
    assert files(target / "sub") == ["shot_00001_.png", "shot_00001_.txt"]
    assert files(output) == []


@pytest.mark.parametrize("value", ["..", "../elsewhere", "output/../../elsewhere"])
def test_a_relative_output_folder_out_of_the_output_directory_is_refused(node, output, value):
    with pytest.raises(ValueError, match="leads out of ComfyUI's output directory .*give an absolute path"):
        node.save_images(torch.rand(1, 4, 4, 3), "shot", value)
    assert not os.path.exists(output.parent / "elsewhere") and files(output) == []


@pytest.mark.parametrize("value, written", [("caption", ".caption"), (".txt", ".txt"), (".TXT", ".TXT"), ("json", ".json")])
def test_the_caption_extension_gets_its_dot(node, output, value, written):
    node.save_images(torch.rand(1, 4, 4, 3), "shot", "output", value, caption="c")
    assert set(files(output)) == {"shot_00001_.png", f"shot_00001_{written}"}


@pytest.mark.parametrize("value", [".exe", "../x.txt", "sub/.txt", "", ".py", " .txt"])
def test_a_caption_extension_that_is_not_plain_text_is_refused_before_any_write(node, output, value):
    with pytest.raises(ValueError, match="is not a plain-text format; use one of .txt, .caption"):
        node.save_images(torch.rand(1, 4, 4, 3), "shot", "output", value, caption="c")
    assert files(output) == []


def test_an_empty_batch_writes_nothing(node, output, calls):
    assert node.save_images(torch.zeros(0, 4, 4, 3), "shot", "output", caption="c") == ("",)
    assert node.save_images(None, "shot", "output") == ("",)
    assert files(output) == [] and calls == []

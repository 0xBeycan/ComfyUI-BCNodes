"""libs/image_write.py write_image's PNG compression level: quality 0-60 gives level 0-6 (quality / 10),
anything above gives 6; quality 0 is DEFAULT_QUALITY (90), so 6; Save Image With Caption's
PNG_QUALITY (40) stays level 4. A TIFF written with save_metadata reads back with the prompt in EXIF
`Make` and the workflow in `ImageDescription`, a workflow larger than JPEG's 64 KB too. JPEG 2000 does
not use the value: every quality below 100 writes the same lossy file, 100 a lossless one."""

import json

import pytest


@pytest.fixture
def image_write(bcnodes):
    return bcnodes["libs.image_write"]


class RecordingImage:
    """Stands in for a PIL image: keeps the keyword arguments save() was called with."""

    def save(self, path, **kwargs):
        self.kwargs = kwargs


def png_level(image_write, quality):
    img = RecordingImage()
    image_write.write_image("out.png", img, None, False, None, quality)
    return img.kwargs["compress_level"]


@pytest.mark.parametrize("quality, level", [(1, 0), (10, 1), (40, 4), (60, 6), (61, 6), (90, 6), (100, 6), (0, 6)])
def test_png_level_follows_quality_up_to_six(image_write, quality, level):
    assert png_level(image_write, quality) == level


def test_caption_saves_keep_level_four(bcnodes, image_write):
    assert png_level(image_write, bcnodes["pipelines.save_image"].PNG_QUALITY) == 4


def test_tiff_keeps_prompt_and_workflow(image_write, tmp_path):
    from PIL import Image

    prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 7}}}
    workflow = {"nodes": [{"id": i, "widgets_values": ["word " * 200]} for i in range(100)]}  # ~100 KB
    path = tmp_path / "out.tiff"
    image_write.write_image(str(path), Image.new("RGB", (8, 8), (10, 20, 30)), prompt, True, {"workflow": workflow}, 90)
    exif = Image.open(path).getexif()
    assert exif[0x010F] == "Prompt: " + json.dumps(prompt)
    assert exif[0x010E] == "Workflow: " + json.dumps(workflow)


def test_jpeg2000_lossy_below_100_lossless_at_100(image_write, tmp_path):
    import numpy as np
    from PIL import Image, features

    if not features.check("jpg_2000"):
        pytest.skip("Pillow built without OpenJPEG")
    pixels = np.random.default_rng(0).integers(0, 256, (24, 32, 3), dtype=np.uint8)
    for quality in (50, 95, 100):
        image_write.write_image(str(tmp_path / f"{quality}.jp2"), Image.fromarray(pixels), None, False, None, quality)
    assert (tmp_path / "50.jp2").read_bytes() == (tmp_path / "95.jp2").read_bytes()
    assert not np.array_equal(np.array(Image.open(tmp_path / "95.jp2")), pixels)
    assert np.array_equal(np.array(Image.open(tmp_path / "100.jp2")), pixels)

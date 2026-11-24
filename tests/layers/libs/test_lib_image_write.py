"""libs/image_write.py write_image's PNG compression level: quality 0-60 gives level 0-6 (quality / 10),
anything above gives 6; quality 0 is DEFAULT_QUALITY (90), so 6; Save Image With Caption's
PNG_QUALITY (40) stays level 4."""

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

"""Golden: the Social Media Export engine (pipelines/social_export.py).

- plan / crop_fraction / pad_box for every platform of nodes/social_specs.json and the inline
  FEED / STORY specs, x 7 master sizes x allow_upscale False/True (default anchor 0.4, the value
  the node passes).
- encode -> (md5 of the data, final quality, ext, exceeded) for a default_rng(0) RGB image and its
  RGBA / LA / P-with-transparency forms x {jpg, webp, png, jpg with max_bytes 4000, png with
  max_bytes 10} x quality {92, 50} x chroma_444 {True, False}.
- PIL.ImageFile.MAXBLOCK after a 200x200 jpg encode (3 * 200 * 200 = 120000 > 65536, so the
  process-global raise of _encode_once shows, 12 B-20) and after a small one.
- flatten_to_rgb per image mode (mode, pixels, and whether an RGB image comes back as the same
  object).
- REPORT_HEADER, format_report_line for the four strategies and the byte-cap notes, human_bytes.

PIL.ImageFile.MAXBLOCK is reset to 65536 before every case. The embedded sRGB profile
(ImageCms.createProfile, cached by _srgb_icc) carries its creation time in its header (bytes
24-35), so the encoded bytes differ between processes: ImageCmsProfile.tobytes (looked up at call
time) is wrapped to write a fixed time there, and the _srgb_icc cache is emptied before and after
every case. The engine is reached through the WHERE entry "core" (the module), flatten_to_rgb
through "to_rgb".
"""

import hashlib
import json
import os
import struct

import numpy as np
import pytest

from _golden import Where, check, check_env, digest
from _harness import PKG_DIR

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'Pillow': '12.3.0',
    'numpy': '2.5.3',
}

GOLDEN = {
    'plan/instagram_feed': '09093d895222e55e51582b924d1a2142',
    'plan/instagram_story': 'c0f6423bb43e9e62a07b075dd5009a06',
    'plan/x_timeline': '37077732a70dccbfeca12e5d70aaaf3f',
    'plan/tiktok_photo': 'ab3171cf66e31f33ccb513bb084bb565',
    'plan/tiktok_vertical': 'c0f6423bb43e9e62a07b075dd5009a06',
    'plan/pinterest_pin': '2e82785504a5decd9de7ac5f5d6c48ae',
    'plan/facebook_feed': '09093d895222e55e51582b924d1a2142',
    'plan/facebook_story': 'c0f6423bb43e9e62a07b075dd5009a06',
    'plan/threads_feed': '5316fc2c6fff48b20291c6e879e7b61e',
    'plan/bluesky': 'e55fd1a2d7131d68c4b66f187f68b9f4',
    'plan/reddit': '0148d2d912ba8d3cc7f5ef5f1703b3ae',
    'plan/inline_FEED': '09093d895222e55e51582b924d1a2142',
    'plan/inline_STORY': '6de18a3e1eca8cccfd71bb4ba57a15e6',
    'encode/RGB/jpg/q92/444_True': "('03ec41c59738eccda506359720733f9f', 92, 'jpg', False)",
    'encode/RGB/jpg/q92/444_False': "('e7f91803422b0b327d6069fcd7f5a1a2', 92, 'jpg', False)",
    'encode/RGB/jpg/q50/444_True': "('0ac6fcab246526071f4329fd4a72653d', 50, 'jpg', False)",
    'encode/RGB/jpg/q50/444_False': "('e93efd71a91069a3b7f12b8d0ed8b9c0', 50, 'jpg', False)",
    'encode/RGB/webp/q92/444_True': "('ce514119e7908d82d193350a977607b8', 92, 'webp', False)",
    'encode/RGB/webp/q92/444_False': "('ce514119e7908d82d193350a977607b8', 92, 'webp', False)",
    'encode/RGB/webp/q50/444_True': "('88dd1fbccf841447499d7f5e43ee7ddd', 50, 'webp', False)",
    'encode/RGB/webp/q50/444_False': "('88dd1fbccf841447499d7f5e43ee7ddd', 50, 'webp', False)",
    'encode/RGB/png/q92/444_True': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', False)",
    'encode/RGB/png/q92/444_False': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', False)",
    'encode/RGB/png/q50/444_True': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', False)",
    'encode/RGB/png/q50/444_False': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', False)",
    'encode/RGB/jpg_max_bytes_4000/q92/444_True': "('03ec41c59738eccda506359720733f9f', 92, 'jpg', False)",
    'encode/RGB/jpg_max_bytes_4000/q92/444_False': "('e7f91803422b0b327d6069fcd7f5a1a2', 92, 'jpg', False)",
    'encode/RGB/jpg_max_bytes_4000/q50/444_True': "('0ac6fcab246526071f4329fd4a72653d', 50, 'jpg', False)",
    'encode/RGB/jpg_max_bytes_4000/q50/444_False': "('e93efd71a91069a3b7f12b8d0ed8b9c0', 50, 'jpg', False)",
    'encode/RGB/png_max_bytes_10/q92/444_True': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', True)",
    'encode/RGB/png_max_bytes_10/q92/444_False': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', True)",
    'encode/RGB/png_max_bytes_10/q50/444_True': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', True)",
    'encode/RGB/png_max_bytes_10/q50/444_False': "('62feb476f206acd6ea09a2d8d947134f', 100, 'png', True)",
    'encode/RGBA/jpg/q92/444_True': "('acb5a7588f7e67a975f0664c58464000', 92, 'jpg', False)",
    'encode/RGBA/jpg/q92/444_False': "('74d828fc1adda869fab3c1d3e08f80b7', 92, 'jpg', False)",
    'encode/RGBA/jpg/q50/444_True': "('5dd2321284742f4c381215981efc374e', 50, 'jpg', False)",
    'encode/RGBA/jpg/q50/444_False': "('37a3a446ee988cf1fc66e02c4d849488', 50, 'jpg', False)",
    'encode/RGBA/webp/q92/444_True': "('01036a951ef46eb114edc81a4e7f2da7', 92, 'webp', False)",
    'encode/RGBA/webp/q92/444_False': "('01036a951ef46eb114edc81a4e7f2da7', 92, 'webp', False)",
    'encode/RGBA/webp/q50/444_True': "('45229cba1a78ec35fc52fa9b449eec5f', 50, 'webp', False)",
    'encode/RGBA/webp/q50/444_False': "('45229cba1a78ec35fc52fa9b449eec5f', 50, 'webp', False)",
    'encode/RGBA/png/q92/444_True': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', False)",
    'encode/RGBA/png/q92/444_False': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', False)",
    'encode/RGBA/png/q50/444_True': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', False)",
    'encode/RGBA/png/q50/444_False': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', False)",
    'encode/RGBA/jpg_max_bytes_4000/q92/444_True': "('acb5a7588f7e67a975f0664c58464000', 92, 'jpg', False)",
    'encode/RGBA/jpg_max_bytes_4000/q92/444_False': "('74d828fc1adda869fab3c1d3e08f80b7', 92, 'jpg', False)",
    'encode/RGBA/jpg_max_bytes_4000/q50/444_True': "('5dd2321284742f4c381215981efc374e', 50, 'jpg', False)",
    'encode/RGBA/jpg_max_bytes_4000/q50/444_False': "('37a3a446ee988cf1fc66e02c4d849488', 50, 'jpg', False)",
    'encode/RGBA/png_max_bytes_10/q92/444_True': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', True)",
    'encode/RGBA/png_max_bytes_10/q92/444_False': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', True)",
    'encode/RGBA/png_max_bytes_10/q50/444_True': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', True)",
    'encode/RGBA/png_max_bytes_10/q50/444_False': "('d3b993a5c931bd860ec4df63e6a7ba12', 100, 'png', True)",
    'encode/LA/jpg/q92/444_True': "('8142f4261b81d3a7da27944172124a85', 92, 'jpg', False)",
    'encode/LA/jpg/q92/444_False': "('7ee7cb8e9881dc8f44b3f37992792e99', 92, 'jpg', False)",
    'encode/LA/jpg/q50/444_True': "('0c9cfd22ebeae60b2171cd57055f9107', 50, 'jpg', False)",
    'encode/LA/jpg/q50/444_False': "('7b2c71791c2b144a9bfd6ec9ee47a97f', 50, 'jpg', False)",
    'encode/LA/webp/q92/444_True': "('1c625ee632dcbcefec6a8744dfd7a22f', 92, 'webp', False)",
    'encode/LA/webp/q92/444_False': "('1c625ee632dcbcefec6a8744dfd7a22f', 92, 'webp', False)",
    'encode/LA/webp/q50/444_True': "('251c4a7fa33eea26ec50d3fd01becfbe', 50, 'webp', False)",
    'encode/LA/webp/q50/444_False': "('251c4a7fa33eea26ec50d3fd01becfbe', 50, 'webp', False)",
    'encode/LA/png/q92/444_True': "('6ed90562b9e64072c08568b086edba82', 100, 'png', False)",
    'encode/LA/png/q92/444_False': "('6ed90562b9e64072c08568b086edba82', 100, 'png', False)",
    'encode/LA/png/q50/444_True': "('6ed90562b9e64072c08568b086edba82', 100, 'png', False)",
    'encode/LA/png/q50/444_False': "('6ed90562b9e64072c08568b086edba82', 100, 'png', False)",
    'encode/LA/jpg_max_bytes_4000/q92/444_True': "('8142f4261b81d3a7da27944172124a85', 92, 'jpg', False)",
    'encode/LA/jpg_max_bytes_4000/q92/444_False': "('7ee7cb8e9881dc8f44b3f37992792e99', 92, 'jpg', False)",
    'encode/LA/jpg_max_bytes_4000/q50/444_True': "('0c9cfd22ebeae60b2171cd57055f9107', 50, 'jpg', False)",
    'encode/LA/jpg_max_bytes_4000/q50/444_False': "('7b2c71791c2b144a9bfd6ec9ee47a97f', 50, 'jpg', False)",
    'encode/LA/png_max_bytes_10/q92/444_True': "('6ed90562b9e64072c08568b086edba82', 100, 'png', True)",
    'encode/LA/png_max_bytes_10/q92/444_False': "('6ed90562b9e64072c08568b086edba82', 100, 'png', True)",
    'encode/LA/png_max_bytes_10/q50/444_True': "('6ed90562b9e64072c08568b086edba82', 100, 'png', True)",
    'encode/LA/png_max_bytes_10/q50/444_False': "('6ed90562b9e64072c08568b086edba82', 100, 'png', True)",
    'encode/P_transparency/jpg/q92/444_True': "('f413a23e8520d32736204a1760e7e962', 92, 'jpg', False)",
    'encode/P_transparency/jpg/q92/444_False': "('87b6b2199b73f5ec940e648469a96769', 92, 'jpg', False)",
    'encode/P_transparency/jpg/q50/444_True': "('34d1df59efdc01806c3900f57ee90c25', 50, 'jpg', False)",
    'encode/P_transparency/jpg/q50/444_False': "('4402f232e9d151f4c72a0d023c0e999d', 50, 'jpg', False)",
    'encode/P_transparency/webp/q92/444_True': "('d5407cb0a0e7a49e360c6b3401bcd640', 92, 'webp', False)",
    'encode/P_transparency/webp/q92/444_False': "('d5407cb0a0e7a49e360c6b3401bcd640', 92, 'webp', False)",
    'encode/P_transparency/webp/q50/444_True': "('7a4127d50d44ea7b1ae766bc245a4288', 50, 'webp', False)",
    'encode/P_transparency/webp/q50/444_False': "('7a4127d50d44ea7b1ae766bc245a4288', 50, 'webp', False)",
    'encode/P_transparency/png/q92/444_True': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', False)",
    'encode/P_transparency/png/q92/444_False': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', False)",
    'encode/P_transparency/png/q50/444_True': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', False)",
    'encode/P_transparency/png/q50/444_False': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', False)",
    'encode/P_transparency/jpg_max_bytes_4000/q92/444_True': "('f413a23e8520d32736204a1760e7e962', 92, 'jpg', False)",
    'encode/P_transparency/jpg_max_bytes_4000/q92/444_False': "('87b6b2199b73f5ec940e648469a96769', 92, 'jpg', False)",
    'encode/P_transparency/jpg_max_bytes_4000/q50/444_True': "('34d1df59efdc01806c3900f57ee90c25', 50, 'jpg', False)",
    'encode/P_transparency/jpg_max_bytes_4000/q50/444_False': "('4402f232e9d151f4c72a0d023c0e999d', 50, 'jpg', False)",
    'encode/P_transparency/png_max_bytes_10/q92/444_True': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', True)",
    'encode/P_transparency/png_max_bytes_10/q92/444_False': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', True)",
    'encode/P_transparency/png_max_bytes_10/q50/444_True': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', True)",
    'encode/P_transparency/png_max_bytes_10/q50/444_False': "('5aa6f0d03b66411f087c1dae1f9d4175', 100, 'png', True)",
    'maxblock/200x200': '120000',
    'maxblock/48x48': '65536',
    'to_rgb/RGB': "('RGB', 'RGB', (48, 32), '8795e3ca7acaf7d876a7b8bc72a4f5b9', True)",
    'to_rgb/RGBA': "('RGBA', 'RGB', (48, 32), '9d312e9b26d0ea412a9f798dc48aa8ff', False)",
    'to_rgb/LA': "('LA', 'RGB', (48, 32), '7d9028d5ebce169a6ea6b7f646b5c258', False)",
    'to_rgb/PA': "('PA', 'RGB', (48, 32), '28ae1d20f5918cef791f7c75606e7dd9', False)",
    'to_rgb/P': "('P', 'RGB', (48, 32), '4039c730c9efc8e4dc517cd80a07f89d', False)",
    'to_rgb/P_transparency': "('P', 'RGB', (48, 32), 'cff45499e6b663eb98b05977c43fc7eb', False)",
    'to_rgb/L': "('L', 'RGB', (48, 32), '13574e52da0aa25bd972f8c978a7e134', False)",
    'to_rgb/1': "('1', 'RGB', (48, 32), '9a863f0cf080cc344fe2d624310244ec', False)",
    'to_rgb/CMYK': "('CMYK', 'RGB', (48, 32), '8795e3ca7acaf7d876a7b8bc72a4f5b9', False)",
    'to_rgb/I': "('I', 'RGB', (48, 32), '13574e52da0aa25bd972f8c978a7e134', False)",
    'to_rgb/F': "('F', 'RGB', (48, 32), 'ad1268c3ff0d7c6f15d27cac10f5fb60', False)",
    'report_line/scale': '  0  instagram_story    1920x1080 ->   1920x1080 ->   1080x1920    0.0%  scale q92  jpg    120.6KB',
    'report_line/crop': '  1  instagram_story    1920x1080 ->    608x1080 ->   1080x1920   68.3%  crop  q92  jpg      1023B',
    'report_line/crop_forced': ' 12  instagram_story    1920x1080 ->    608x1080 ->   1080x1920   68.3%  crop! q87  webp    1.00MB  OVER-BUDGET crop (forced)',
    'report_line/pad': '  3  instagram_story    1920x1080 ->   1920x1080 ->   1080x1920    0.0%  pad   q100 png    20.00MB  letterboxed (blurred bg), no crop',
    'report_line/exceeded': '  0  instagram_story    1920x1080 ->    608x1080 ->   1080x1920   68.3%  crop  q70  jpg     1.00MB  >max_bytes (976.6KB) even at q70',
    'report_line/exceeded_no_cap': '  0  instagram_story    1920x1080 ->    608x1080 ->   1080x1920   68.3%  crop  q70  jpg      4.9KB',
    'report_line/pad_exceeded': '123  a_long_platform_name_x   1920x1080 ->    608x1080 ->   1080x1920   68.3%  pad   q70  png         9B  letterboxed (blurred bg), no crop; >max_bytes (10B) even at q70',
    'report_header': '  #  platform               input ->        crop ->      output   crop%  mode  qual fmt       size  notes',
    'human_bytes': "['0B', '1023B', '1.0KB', '1024.0KB', '1.00MB']",
}

WHERE = Where({
    "core": "pipelines.social_export",
    "to_rgb": "pipelines.social_export:flatten_to_rgb",
})

# The inline specs of test_pipe_social_export.py.
FEED = {"ar_min": 0.8, "ar_max": 1.91, "max_w": 1440, "max_h": 1800, "min_w": 1080, "format": "jpg"}
STORY = {"ar_min": 0.5625, "ar_max": 0.5625, "max_w": 1080, "max_h": 1920, "format": "jpg"}

SIZES = [(160, 90), (90, 160), (64, 64), (200, 200), (4000, 1000), (1000, 4000), (1920, 1080)]

ENCODE_SPECS = {
    "jpg": {"format": "jpg"},
    "webp": {"format": "webp"},
    "png": {"format": "png"},
    "jpg_max_bytes_4000": {"format": "jpg", "max_bytes": 4000},
    "png_max_bytes_10": {"format": "png", "max_bytes": 10},
}


def _shipped_specs():
    with open(os.path.join(PKG_DIR, "nodes", "social_specs.json"), encoding="utf-8") as f:
        specs = json.load(f)
    return {k: v for k, v in specs.items() if not k.startswith("_")}


SPECS = {**_shipped_specs(), "inline_FEED": FEED, "inline_STORY": STORY}


ICC_TIME = struct.pack(">6H", 2024, 1, 1, 0, 0, 0)  # the ICC header's dateTimeNumber


@pytest.fixture(autouse=True)
def maxblock(monkeypatch):
    from PIL import ImageFile

    monkeypatch.setattr(ImageFile, "MAXBLOCK", 65536, raising=True)
    return ImageFile


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


def _rgb(w=48, h=32):
    from PIL import Image

    return Image.fromarray(np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def _alpha(w=48, h=32):
    from PIL import Image

    return Image.fromarray(np.random.default_rng(1).integers(0, 256, (h, w), dtype=np.uint8), "L")


def _image(mode):
    from PIL import Image

    rgb = _rgb()
    if mode == "RGB":
        return rgb
    if mode == "RGBA":
        img = rgb.copy()
        img.putalpha(_alpha())
        return img
    if mode == "LA":
        return Image.merge("LA", (rgb.convert("L"), _alpha()))
    if mode == "PA":
        img = rgb.copy()
        img.putalpha(_alpha())
        return img.convert("PA")
    if mode == "P_transparency":
        img = rgb.quantize(16)
        img.info["transparency"] = 0
        return img
    return rgb.convert(mode)


@pytest.mark.parametrize("platform", list(SPECS))
def test_plan_table(platform, bcnodes):
    check_env(ENV, "Pillow", "numpy")
    core, spec = WHERE["core"], SPECS[platform]
    rows = []
    for w, h in SIZES:
        for up in (False, True):
            (ox, oy, cw, ch), out = core.plan(w, h, spec, allow_upscale=up)
            rows.append(((w, h, up), (ox, oy, cw, ch), out, core.crop_fraction(w, h, cw, ch), core.pad_box(w, h, spec, up)))
    check(GOLDEN, f"plan/{platform}", digest(rows))


@pytest.mark.parametrize("chroma_444", [True, False])
@pytest.mark.parametrize("quality", [92, 50])
@pytest.mark.parametrize("fmt", list(ENCODE_SPECS))
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "P_transparency"])
def test_encode(mode, fmt, quality, chroma_444, bcnodes):
    check_env(ENV, "Pillow", "numpy")
    data, q, ext, exceeded = WHERE["core"].encode(_image(mode), ENCODE_SPECS[fmt], quality=quality, chroma_444=chroma_444)
    check(GOLDEN, f"encode/{mode}/{fmt}/q{quality}/444_{chroma_444}", repr((hashlib.md5(data).hexdigest(), q, ext, exceeded)))


@pytest.mark.parametrize("size", [200, 48])
def test_maxblock_after_jpg_encode(size, bcnodes, maxblock):
    check_env(ENV, "Pillow", "numpy")
    WHERE["core"].encode(_rgb(size, size), {"format": "jpg"})
    check(GOLDEN, f"maxblock/{size}x{size}", repr(maxblock.MAXBLOCK))


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA", "PA", "P", "P_transparency", "L", "1", "CMYK", "I", "F"])
def test_to_rgb(mode, bcnodes):
    check_env(ENV, "Pillow", "numpy")
    img = _image(mode)
    out = WHERE["to_rgb"](img)
    check(GOLDEN, f"to_rgb/{mode}", repr((img.mode, out.mode, out.size, digest(out), out is img)))


def _meta(strategy, over=False):
    return {"platform": "instagram_story", "input": (1920, 1080), "crop": (608, 1080), "out": (1080, 1920),
            "crop_pct": 68.33333333333333, "strategy": strategy, "over": over}


# name -> (index, meta, quality, size_bytes, ext, exceeded, max_bytes)
REPORT_LINES = {
    "scale": (0, dict(_meta("scale"), crop=(1920, 1080), crop_pct=0.0), 92, 123456, "jpg", False, None),
    "crop": (1, _meta("crop"), 92, 1023, "jpg", False, 8388608),
    "crop_forced": (12, _meta("crop!", True), 87, 1048576, "webp", False, None),
    "pad": (3, dict(_meta("pad", True), crop=(1920, 1080), crop_pct=0.0), 100, 20971520, "png", False, None),
    "exceeded": (0, _meta("crop"), 70, 1048577, "jpg", True, 1000000),
    "exceeded_no_cap": (0, _meta("crop"), 70, 5000, "jpg", True, None),
    "pad_exceeded": (123, dict(_meta("pad", True), platform="a_long_platform_name_x"), 70, 9, "png", True, 10),
}


@pytest.mark.parametrize("name", list(REPORT_LINES))
def test_format_report_line(name, bcnodes):
    check_env(ENV, "Pillow", "numpy")
    check(GOLDEN, f"report_line/{name}", WHERE["core"].format_report_line(*REPORT_LINES[name]))


def test_report_header_and_human_bytes(bcnodes):
    check_env(ENV, "Pillow", "numpy")
    core = WHERE["core"]
    check(GOLDEN, "report_header", core.REPORT_HEADER)
    check(GOLDEN, "human_bytes", repr([core.human_bytes(n) for n in (0, 1023, 1024, 1048575, 1048576)]))

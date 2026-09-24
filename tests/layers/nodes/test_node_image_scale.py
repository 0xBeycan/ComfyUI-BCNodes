"""Golden: Image Scale By Aspect Ratio, through the node method.

The full widget grid RATIOS x FITS x METHODS x SIDES x scale_to_length {64, 50} x
round_to_multiple {"8", "None", "64"} runs on a seeded (1, 48, 96, 3) image with a seeded mask,
and again on both transposed (so "original" is once wider, once taller). Each case digests the
five outputs (image, mask, original_size box, width, height); the cases of one orientation and
aspect ratio are folded into one digest, so a failure names the pair. proportional_width /
proportional_height keep their widget defaults (1, 1) in the grid; the extras run a custom 5:3.

Extras: the widget defaults, mask only, the (64, 64) placeholder mask (alone and in a batch), an
fp16 image, a 4-channel image, a red letterbox, a custom 5:3 ratio, an image batch with one
mask, and the two ValueErrors with their texts.
"""

import hashlib

import pytest
import torch

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
    'Pillow': '12.3.0',
}

GOLDEN = {
    'grid/wide/original': '74f6f6428ed79d1b28ae0f1c8e6fb48d',
    'grid/tall/original': 'f5f75e43f85226adfa9fff1a6596caf5',
    'grid/wide/custom': 'c5ef9c691d68de15c98614f6ab1dc6fb',
    'grid/tall/custom': '171856e275bb2c67a1f9fd4138d48e75',
    'grid/wide/1:1': 'c5ef9c691d68de15c98614f6ab1dc6fb',
    'grid/tall/1:1': '171856e275bb2c67a1f9fd4138d48e75',
    'grid/wide/3:2': '5d426b3f19eee9da8bd217eef9f2407b',
    'grid/tall/3:2': '0e9069f5aa45df611ca1fb63f3e68987',
    'grid/wide/4:3': 'cff9dceec7793d2961279a9d52abb270',
    'grid/tall/4:3': '1389a2e257c74f9c1dbaafcc6d9ee99e',
    'grid/wide/16:9': '4c089f543173795bc960573ca3dabb87',
    'grid/tall/16:9': '1efb57aadff14ae1a2bac8d933ecdacd',
    'grid/wide/2:3': 'cb2d5361ad8bbfa59ff42b6c1870bb6f',
    'grid/tall/2:3': 'a675c02d00985f3830160a7281195202',
    'grid/wide/3:4': '767ee01b12596c4db2636bfb1d943976',
    'grid/tall/3:4': 'f308c4f56b4b0eb5bd8794cb0e3e77af',
    'grid/wide/9:16': '642b241e9da8ce52d81b54050438a8dc',
    'grid/tall/9:16': '1e42373174687d6f7a8b24a9192b38e0',
    'extras/defaults': 'b927a050bed49bac7ad42619ddfe5d05',
    'extras/defaults_image_only': '9235879a9ed206b32a8b4d8a15d3f486',
    'extras/mask_only_crop': '2523ef88850369bce9e71a14be92109c',
    'extras/mask_only_letterbox': 'a62f9a6211cac694b004872f82497114',
    'extras/placeholder_mask': '0f40f4d0bf47af1d4946e42ad693b4f8',
    'extras/placeholder_in_batch': '9aff0ea979990e867bddd73a41cfdfa8',
    'extras/placeholder_mask_only': 'fff6fb1e22112165ff4d3c5b410513a7',
    'extras/fp16_image': 'c6d7aecce3b779e1c1224562d1ee1221',
    'extras/rgba_image': 'cb5c8f137269229c901065165cee9246',
    'extras/red_letterbox': '9d24f66555c6e075ddaf40df5a9995cf',
    'extras/custom_5_3': '55e1bc4b9cea4dceb1f2b4996def92af',
    'extras/custom_3_5_total_pixels': 'de3d0e45a67d229d042670abba19d987',
    'extras/batch_2_one_mask': 'c8427abbbec80290894a4801db713315',
    'extras/empty_image_batch_with_mask': 'b4e8146d7f2592634674a2e6a72633a9',
    'extras/error_size_mismatch': ('ValueError', 'BC_ImageScaleByAspectRatio: mask is 12x10 but image is 96x48; they must match'),
    'extras/error_nothing_connected': ('ValueError', 'BC_ImageScaleByAspectRatio: connect an image or a mask'),
    'extras/error_placeholder_only': ('ValueError', 'BC_ImageScaleByAspectRatio: connect an image or a mask'),
}

RATIOS = ["original", "custom", "1:1", "3:2", "4:3", "16:9", "2:3", "3:4", "9:16"]
FITS = ["letterbox", "crop", "fill"]
METHODS = ["lanczos", "bicubic", "hamming", "bilinear", "box", "nearest"]
SIDES = ["None", "longest", "shortest", "width", "height", "total_pixel(kilo pixel)"]
LENGTHS = [64, 50]
MULTIPLES = ["8", "None", "64"]


def _image():
    return torch.rand((1, 48, 96, 3), generator=torch.Generator().manual_seed(11))


def _mask():
    return torch.rand((1, 48, 96), generator=torch.Generator().manual_seed(12))


ORIENTATIONS = {
    "wide": lambda: (_image(), _mask()),
    "tall": lambda: (_image().transpose(1, 2), _mask().transpose(1, 2)),
}


def _kw(**overrides):
    kw = dict(aspect_ratio="original", proportional_width=1, proportional_height=1, fit="letterbox", method="lanczos",
              round_to_multiple="8", scale_to_side="None", scale_to_length=1024, background_color="#000000")
    kw.update(overrides)
    return kw


def test_widget_lists_match_the_node(bcnodes):
    node = bcnodes["image_scale"]
    assert (node.RATIOS, node.FITS, node.METHODS, node.SIDES) == (RATIOS, FITS, METHODS, SIDES)


@pytest.mark.parametrize("orientation", list(ORIENTATIONS))
@pytest.mark.parametrize("ratio", RATIOS)
def test_grid(ratio, orientation, bcnodes):
    check_env(ENV, "torch", "numpy", "Pillow")
    image, mask = ORIENTATIONS[orientation]()
    node = bcnodes["image_scale"].ImageScaleByAspectRatio()
    folded = hashlib.md5()
    for fit in FITS:
        for method in METHODS:
            for side in SIDES:
                for length in LENGTHS:
                    for multiple in MULTIPLES:
                        out = node.scale(**_kw(aspect_ratio=ratio, fit=fit, method=method, scale_to_side=side,
                                               scale_to_length=length, round_to_multiple=multiple),
                                         image=image, mask=mask)
                        folded.update(f"{fit}|{method}|{side}|{length}|{multiple}|{digest(out)}\n".encode())
    check(GOLDEN, f"grid/{orientation}/{ratio}", folded.hexdigest())


def _placeholder_batch():
    mask = torch.zeros((2, 64, 64))
    mask[1, 10:40, 20:50] = 1.0
    return mask


EXTRAS = {
    "defaults": lambda: dict(image=_image(), mask=_mask()),
    "defaults_image_only": lambda: dict(image=_image()),
    "mask_only_crop": lambda: dict(mask=_mask(), **_kw(aspect_ratio="1:1", fit="crop", scale_to_side="longest", scale_to_length=64)),
    "mask_only_letterbox": lambda: dict(mask=_mask(), **_kw(aspect_ratio="9:16", scale_to_side="height", scale_to_length=50, round_to_multiple="None")),
    "placeholder_mask": lambda: dict(image=_image(), mask=torch.zeros((1, 64, 64)), **_kw(aspect_ratio="4:3", fit="crop", scale_to_side="width", scale_to_length=64)),
    "placeholder_in_batch": lambda: dict(image=torch.rand((1, 64, 64, 3), generator=torch.Generator().manual_seed(11)),
                                         mask=_placeholder_batch(), **_kw(aspect_ratio="3:2", scale_to_side="longest", scale_to_length=50)),
    "placeholder_mask_only": lambda: dict(mask=_placeholder_batch(), **_kw(scale_to_side="shortest", scale_to_length=32)),
    "fp16_image": lambda: dict(image=_image().half(), **_kw(aspect_ratio="16:9", fit="fill", method="bicubic", scale_to_side="longest", scale_to_length=64)),
    "rgba_image": lambda: dict(image=torch.rand((1, 48, 96, 4), generator=torch.Generator().manual_seed(11)),
                               **_kw(aspect_ratio="1:1", fit="letterbox", scale_to_side="longest", scale_to_length=64)),
    "red_letterbox": lambda: dict(image=_image(), mask=_mask(), **_kw(aspect_ratio="1:1", scale_to_side="longest", scale_to_length=64,
                                                                      background_color="#ff0000")),
    "custom_5_3": lambda: dict(image=_image(), mask=_mask(), **_kw(aspect_ratio="custom", proportional_width=5, proportional_height=3,
                                                                   fit="crop", scale_to_side="shortest", scale_to_length=50)),
    "custom_3_5_total_pixels": lambda: dict(image=_image(), **_kw(aspect_ratio="custom", proportional_width=3, proportional_height=5,
                                                                  fit="letterbox", scale_to_side="total_pixel(kilo pixel)", scale_to_length=4)),
    "batch_2_one_mask": lambda: dict(image=torch.rand((2, 48, 96, 3), generator=torch.Generator().manual_seed(11)), mask=_mask(),
                                     **_kw(aspect_ratio="2:3", fit="crop", method="nearest", scale_to_side="height", scale_to_length=64)),
    "empty_image_batch_with_mask": lambda: dict(image=torch.zeros((0, 48, 96, 3)), mask=_mask(), **_kw(scale_to_side="width", scale_to_length=64)),
    "error_size_mismatch": lambda: dict(image=_image(), mask=torch.ones((1, 10, 12))),
    "error_nothing_connected": lambda: dict(),
    "error_placeholder_only": lambda: dict(mask=torch.zeros((1, 64, 64))),
}


@pytest.mark.parametrize("name", list(EXTRAS))
def test_extras(name, bcnodes):
    check_env(ENV, "torch", "numpy", "Pillow")
    kwargs = EXTRAS[name]()
    kwargs = {**_kw(), **kwargs}
    try:
        value = digest(bcnodes["image_scale"].ImageScaleByAspectRatio().scale(**kwargs))
    except ValueError as e:
        value = ("ValueError", str(e))
    check(GOLDEN, f"extras/{name}", value)

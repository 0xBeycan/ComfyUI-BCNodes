"""Frequency Merge, the surface: base and detail IMAGE inputs, split_sigma and detail_strength with
their defaults and ranges, the optional device, one IMAGE output in BCNodes/image; the node hands
its widgets to libs/frequency.py (detail_strength 0 gives base's low-pass, detail = base gives
base), and anything but two image batches is an error that says what to do."""

import pytest
import torch

CPU = torch.device("cpu")


@pytest.fixture
def node(bcnodes):
    return bcnodes["frequency_merge"].FrequencyMerge


def test_surface(bcnodes, node):
    inputs = node.INPUT_TYPES()
    required = inputs["required"]
    assert list(required) == ["base", "detail", "split_sigma", "detail_strength"] and list(inputs["optional"]) == ["device"]
    assert required["base"][0] == required["detail"][0] == "IMAGE"
    sigma, strength = required["split_sigma"][1], required["detail_strength"][1]
    assert (sigma["default"], sigma["min"], sigma["max"]) == (3.0, 0.5, 64.0)
    assert (strength["default"], strength["min"], strength["max"]) == (1.0, 0.0, 2.0)
    assert inputs["optional"]["device"][1]["default"] == "cpu"
    assert (node.RETURN_TYPES, node.RETURN_NAMES, node.CATEGORY) == (("IMAGE",), ("image",), "BCNodes/image")
    assert not hasattr(node, "HEAVY_OUTPUTS") and set(inputs) == {"required", "optional"}
    assert bcnodes["frequency_merge"].NODE_DISPLAY_NAME_MAPPINGS == {"BC_FrequencyMerge": "Frequency Merge"}


def test_merge_hands_its_widgets_to_the_lib(bcnodes, node):
    lib = bcnodes["libs.frequency"].frequency_merge
    g = torch.Generator().manual_seed(0)
    base, detail = torch.rand(2, 32, 40, 3, generator=g), torch.rand(2, 32, 40, 3, generator=g)
    assert torch.equal(node().merge(base, detail, 2.5, 0.7)[0], lib(base, detail, 2.5, 0.7, CPU))
    assert torch.allclose(node().merge(base, base, 2.5, 1.0)[0], base, atol=1e-6)
    assert torch.equal(node().merge(base, detail, 2.5, 0.0)[0], node().merge(base, torch.zeros_like(detail), 2.5, 0.0)[0])


@pytest.mark.parametrize("base, detail", [(None, torch.zeros(1, 8, 8, 3)), (torch.zeros(1, 8, 8, 3), None),
                                          (torch.zeros(8, 8, 3), torch.zeros(8, 8, 3))])
def test_not_two_image_batches_says_what_to_do(node, base, detail):
    with pytest.raises(ValueError, match="Frequency Merge: connect two image batches"):
        node().merge(base, detail, 2.0, 1.0)

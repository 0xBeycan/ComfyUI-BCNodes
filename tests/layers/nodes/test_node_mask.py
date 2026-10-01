"""Repeat Mask Batch through its node method: the whole batch repeated in order, as core's Repeat
Image Batch does for images (`x.repeat((amount, 1, 1, 1))`, written out for (B, H, W)); a (H, W)
mask counts as one frame; amount 1 hands the input on without a copy.
"""

import torch


def test_repeat_is_the_batch_in_order(bcnodes):
    mask = torch.rand(2, 5, 7)
    (out,) = bcnodes["mask"].RepeatMaskBatch().repeat(mask, 3)
    assert torch.equal(out, mask.repeat((3, 1, 1))) and torch.equal(out[2], mask[0])


def test_repeat_of_a_single_2d_mask(bcnodes):
    mask = torch.rand(5, 7)
    (out,) = bcnodes["mask"].RepeatMaskBatch().repeat(mask, 4)
    assert out.shape == (4, 5, 7) and torch.equal(out[3], mask)


def test_repeat_once_is_the_input(bcnodes):
    mask = torch.rand(2, 5, 7)
    (out,) = bcnodes["mask"].RepeatMaskBatch().repeat(mask, 1)
    assert out.data_ptr() == mask.data_ptr() and torch.equal(out, mask)

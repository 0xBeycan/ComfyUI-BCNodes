"""Golden: the vendored BiRefNet architecture (plan 8.2 row 17).

models/birefnet/arch/ is the vendored BiRefNet code, byte for byte; these pins make sure the
package the loader imports is still that code:
  - the state-dict layout of BiRefNet("swin_v1_t") and BiRefNet("swin_v1_l"):
    repr(sorted((key, shape, dtype))), so every released checkpoint still loads strict;
  - the logits of BiRefNet("swin_v1_t").eval() on a seeded (1, 3, 64, 64) input, weights from
    random init after torch.manual_seed(0).

The global RNG is forked, so the seeding does not leak into other tests. The forward runs with
one intra-op thread: on this machine the logits bytes differ between 1 and 5 threads (float
reduction order), and the thread count is not what the golden pins. swin_v1_l needs ~0.9 GB RAM.

Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import pytest
import torch

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'torchvision': '0.29.0',
}

GOLDEN = {
    'layout/swin_v1_t': '0ce737a1acd413e0c3366da674575828',
    'layout/swin_v1_l': '278165c4f067a9e29a247948ce45d4b5',
    'logits/swin_v1_t': 'fe0874f32c1535ddc32b4327693836e7',
}

WHERE = Where({
    "arch": "models.birefnet.arch",
})


@pytest.mark.parametrize("backbone", ["swin_v1_t", "swin_v1_l"])
def test_state_dict_layout(backbone, bcnodes):
    check_env(ENV, "torch", "torchvision")
    with torch.random.fork_rng(devices=[]):
        net = WHERE["arch"].BiRefNet(backbone)
    layout = repr(sorted((k, tuple(v.shape), str(v.dtype)) for k, v in net.state_dict().items()))
    check(GOLDEN, f"layout/{backbone}", digest(layout))


def test_logits_swin_v1_t(bcnodes):
    check_env(ENV, "torch", "torchvision")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            net = WHERE["arch"].BiRefNet("swin_v1_t").eval()
        x = torch.rand((1, 3, 64, 64), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            logits = net(x)
    finally:
        torch.set_num_threads(threads)
    check(GOLDEN, "logits/swin_v1_t", digest(logits))

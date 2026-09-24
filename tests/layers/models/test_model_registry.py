"""Golden: the BiRefNet checkpoint table the matting registry is built from (plan 8.2 row 18, 6.1).

The table is CHECKPOINTS, the typed BiRefNetCheckpoint rows of models/birefnet/checkpoints.py
registered under MATTING. Two WHERE adapters read it:
  - config_dump: json.dumps({c.name: {"repo": c.repo, "file": c.file, "backbone": c.backbone,
                                      "res": c.res} for c in CHECKPOINTS})
  - names: registry.names(MATTING)
json.dumps writes 512, 512.0 and "512" differently, so the digest also proves every `res` is an
int, and key order is pinned with the values.

Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE.
"""

import json

from _golden import Where, at, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'config_dump': '2a6b35b00fc25e4b4a942f24a4f6f062',
    'names': 'd5d98e7a2dab59adf678923e44a3b0ee',
}

WHERE = Where({
    "config_dump": lambda: json.dumps({c.name: {"repo": c.repo, "file": c.file, "backbone": c.backbone, "res": c.res}
                                       for c in at("models.birefnet.checkpoints:CHECKPOINTS")}),
    "names": lambda: at("models.common.registry:names")(at("models.common.registry:MATTING")),
})


def test_config_dump(bcnodes):
    check_env(ENV)
    check(GOLDEN, "config_dump", digest(WHERE["config_dump"]()))


def test_names(bcnodes):
    check_env(ENV)
    check(GOLDEN, "names", digest(WHERE["names"]()))

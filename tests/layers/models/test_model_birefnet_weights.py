"""BiRefNet weights (no download): a checkpoint missing from models/background_removal is fetched
from beycanai/BiRefNet as <name>.safetensors, for every registered matting name."""

import sys


def test_every_checkpoint_is_fetched_from_the_pack_repo(bcnodes, monkeypatch, tmp_path):
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"), raising=True)
    monkeypatch.setattr(fp, "folder_names_and_paths", {}, raising=True)
    fetched = []
    monkeypatch.setattr(bcnodes["models.common.download"], "fetch_with_progress",
                        lambda url, path, label: fetched.append((url, path)) or path, raising=True)
    registry = bcnodes["models.common.registry"]
    names = registry.names(registry.MATTING)
    for name in names:
        bcnodes["models.birefnet.weights"].weights_path(name)
    folder = tmp_path / "models" / "background_removal"
    assert len(names) == 11
    assert fetched == [(f"https://huggingface.co/beycanai/BiRefNet/resolve/main/{n}.safetensors",
                        str(folder / f"{n}.safetensors")) for n in names]

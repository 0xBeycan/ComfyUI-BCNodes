"""SeedVR2 Chunk Size, the surface: a LATENT and the safety_margin widget (default the DiT law's 0.64) in,
one INT out under the name Split SeedVR2 Latent's sub-input has (frames_per_chunk), the SeedVR2 category,
and both inputs handed to the flow under their own names. That the INT reaches Split SeedVR2 Latent's
manual frames_per_chunk through ComfyUI's validation and executor is in tests/test_runtime.py."""

import torch


def test_surface(bcnodes):
    mod = bcnodes["seedvr2"]
    cls = mod.SeedVR2ChunkSize
    inputs = cls.INPUT_TYPES()
    required = inputs["required"]
    assert set(inputs) == {"required"}
    assert list(required) == ["latent", "safety_margin"]
    assert required["latent"][0] == "LATENT"
    margin = required["safety_margin"]
    assert margin[0] == "FLOAT" and (margin[1]["default"], margin[1]["min"], margin[1]["max"]) == (0.64, 0.0, 4.0)
    assert margin[1]["default"] == bcnodes["models.seedvr2.dit"].SAFETY_MARGIN
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY) == (("INT",), ("frames_per_chunk",), "size", "BCNodes/seedvr2")
    assert not getattr(cls, "OUTPUT_NODE", False) and not hasattr(cls, "HEAVY_OUTPUTS")
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["BC_SeedVR2ChunkSize"] == "SeedVR2 Chunk Size"


def test_inputs_reach_the_flow_by_name(bcnodes, monkeypatch):
    seen = {}

    def frames_per_chunk(latent, safety_margin):
        seen.update(latent=latent, safety_margin=safety_margin)
        return (41,)

    monkeypatch.setattr(bcnodes["pipelines.seedvr2.chunk_size"], "frames_per_chunk", frames_per_chunk)
    latent = {"samples": torch.zeros(1, 16, 3, 2, 2)}
    assert bcnodes["seedvr2"].SeedVR2ChunkSize().size(latent, 0.5) == (41,)
    assert seen["latent"] is latent and seen["safety_margin"] == 0.5

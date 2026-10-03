"""SeedVR2 Preprocess (Compact) and SeedVR2 PostProcess (Compact), the surface: each takes the widgets of
the nodes it replaces with their names, specs and order (the temporal tile widgets, which the SeedVR2
VAE ignores, left out), returns what the brief fixes (a LATENT and the SEEDVR2_PLAN; IMAGE), and sits
in the SeedVR2 category's Compact subcategory. Their tile_size adds 0 (auto, the default) below the range of
VAE Encode/Decode's, which is otherwise the same, so a saved value runs as before. The four nodes they
replace keep their own surface (tests/test_nodes.py, the unused-outputs table)."""


def assert_auto_tile(spec, replaced):
    """tile_size: VAE Encode/Decode's spec with the range opened down to 0 (auto), the new default."""
    assert spec[0] == replaced[0] == "INT"
    assert (spec[1]["default"], spec[1]["min"]) == (0, 0) and (replaced[1]["default"], replaced[1]["min"]) == (1024, 64)
    assert all(spec[1][k] == replaced[1][k] for k in ("max", "step", "advanced"))
    assert "auto" in spec[1]["tooltip"]

COMPACT = "BCNodes/seedvr2/compact"


def test_preprocess_takes_resize_and_encode_widgets(bcnodes):
    mod = bcnodes["seedvr2"]
    cls = mod.SeedVR2PreprocessCompact
    inputs = cls.INPUT_TYPES()
    required, resize, encode = inputs["required"], mod.SeedVR2Resize.INPUT_TYPES()["required"], mod.SeedVR2VAEEncode.INPUT_TYPES()["required"]
    assert set(inputs) == {"required"}  # no heavy output, so no hidden link inputs
    assert list(required) == ["image", "vae", "upscale_factor", "downscale_factor", "max_resolution", "emulate_bf16", "tile_size", "overlap"]
    assert required["image"][0] == "IMAGE" and required["vae"] == encode["vae"]
    assert all(required[name] == resize[name] for name in ("upscale_factor", "downscale_factor", "max_resolution", "emulate_bf16"))
    assert required["overlap"] == encode["overlap"]
    assert_auto_tile(required["tile_size"], encode["tile_size"])
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY) == (("LATENT", "SEEDVR2_PLAN"), ("latent", "plan"), "preprocess", COMPACT)
    assert len(cls.OUTPUT_TOOLTIPS) == 2 and not getattr(cls, "OUTPUT_NODE", False)


def test_postprocess_takes_decode_and_postprocess_widgets(bcnodes):
    mod = bcnodes["seedvr2"]
    cls = mod.SeedVR2PostProcessCompact
    inputs = cls.INPUT_TYPES()
    required, decode, post = inputs["required"], mod.SeedVR2VAEDecode.INPUT_TYPES()["required"], mod.SeedVR2PostProcess.INPUT_TYPES()["required"]
    assert set(inputs) == {"required"}
    assert list(required) == ["samples", "vae", "image", "plan", "color_correction_method", "tile_size", "overlap"]
    assert required["samples"][0] == "LATENT" and required["vae"] == decode["vae"]
    assert required["image"][0] == "IMAGE" and required["plan"][0] == "SEEDVR2_PLAN"
    assert required["color_correction_method"] == post["color_correction_method"]
    assert required["overlap"] == decode["overlap"]
    assert_auto_tile(required["tile_size"], decode["tile_size"])
    assert (cls.RETURN_TYPES, cls.RETURN_NAMES, cls.FUNCTION, cls.CATEGORY) == (("IMAGE",), ("images",), "process", COMPACT)
    assert not getattr(cls, "OUTPUT_NODE", False)


def test_registered_next_to_the_nodes_they_replace(bcnodes):
    mod = bcnodes["seedvr2"]
    assert list(mod.NODE_CLASS_MAPPINGS) == ["BC_SeedVR2FramingDownscale", "BC_SeedVR2Resize", "BC_SeedVR2VAEEncode", "BC_SeedVR2ChunkSize",
                                             "BC_SeedVR2VAEDecode", "BC_SeedVR2PostProcess", "BC_SeedVR2PreprocessCompact",
                                             "BC_SeedVR2PostProcessCompact"]
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["BC_SeedVR2PreprocessCompact"] == "SeedVR2 Preprocess (Compact)"
    assert mod.NODE_DISPLAY_NAME_MAPPINGS["BC_SeedVR2PostProcessCompact"] == "SeedVR2 PostProcess (Compact)"
    assert all(mod.NODE_CLASS_MAPPINGS[key].CATEGORY == "BCNodes/seedvr2"
               for key in ("BC_SeedVR2Resize", "BC_SeedVR2VAEEncode", "BC_SeedVR2VAEDecode", "BC_SeedVR2PostProcess"))

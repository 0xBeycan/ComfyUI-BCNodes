"""Claude and Claude Config (nodes/claude.py) through the node classes, the flow (pipelines/claude/messages.run) replaced by a
recorder:

  - the surface: Claude's widgets in order with their defaults, ranges, multiline text and the seed's `fixed`
    control, a tooltip on every input, config and image_1 declared, any other input name answered as IMAGE (the
    slots web/js/claude.js adds), outputs, category, display names; Claude Config's widgets and defaults (those of
    pipelines/claude/messages.ClaudeConfigValues);
  - Claude hands the widgets to the flow unchanged, every frame of image_1, image_2, ..., image_10 in slot order
    (not name order) as an 8-bit image, a half-precision frame read as its float32 source, and Claude Config's
    defaults when no config is connected;
  - Claude Config: model_id stripped, stop sequences one per line with blank lines dropped, the other fields as
    given; cache_system_prompt optional (added after the release), on when missing.
"""

import numpy as np
import pytest
import torch

REQUIRED = ["prompt", "model", "max_tokens", "effort", "seed", "system_prompt"]
CONFIG = ["model_id", "stop_sequences", "on_refusal", "speed", "image_format", "timeout", "max_retries"]


@pytest.fixture
def node(bcnodes):
    return bcnodes["claude"]


@pytest.fixture
def pipe(bcnodes):
    return bcnodes["pipelines.claude.messages"]


def test_claude_surface(node, pipe):
    inputs = node.Claude.INPUT_TYPES()
    required = inputs["required"]
    assert list(required) == REQUIRED
    assert required["model"][0] == ["Opus 5.5", "Sonnet 5.5", "Haiku 5.5", "Fable 5.2"]
    assert required["model"][1]["default"] == "Opus 5.5"
    assert required["effort"][0] == ["low", "medium", "high", "xhigh", "max"] and required["effort"][1]["default"] == "high"
    assert required["max_tokens"][1] | {"tooltip": None} == {"default": 32768, "min": 1, "max": 128000, "tooltip": None}
    assert required["seed"][1]["control_after_generate"] == "fixed" and required["seed"][1]["max"] == 2 ** 64 - 1
    assert required["prompt"][1]["multiline"] and required["system_prompt"][1]["multiline"]
    assert required["prompt"][1]["default"] == "" and required["system_prompt"][1]["default"] == ""
    optional = inputs["optional"]
    assert list(optional) == ["config", "image_1"]
    assert optional["config"][0] == "BC_CLAUDE_CONFIG" and optional["image_1"][0] == "IMAGE"
    assert "image_7" in optional and optional["image_7"] == ("IMAGE",)
    for spec in list(required.values()) + [optional["config"], optional["image_1"]]:
        assert spec[1].get("tooltip"), spec
    assert node.Claude.RETURN_TYPES == ("STRING", "STRING") and node.Claude.RETURN_NAMES == ("text", "thinking")
    assert node.Claude.CATEGORY == node.ClaudeConfig.CATEGORY == "BCNodes/lm"
    assert node.NODE_DISPLAY_NAME_MAPPINGS == {"BC_Claude": "Claude", "BC_ClaudeConfig": "Claude Config"}
    assert set(node.NODE_CLASS_MAPPINGS) == {"BC_Claude", "BC_ClaudeConfig"}


def test_config_surface(node, pipe):
    required = node.ClaudeConfig.INPUT_TYPES()["required"]
    assert list(required) == CONFIG
    defaults = pipe.ClaudeConfigValues()
    assert {name: spec[1]["default"] for name, spec in required.items()} == {
        "model_id": "", "stop_sequences": "", "on_refusal": defaults.on_refusal, "speed": defaults.speed,
        "image_format": defaults.image_format, "timeout": 600, "max_retries": 2}
    assert (defaults.on_refusal, defaults.speed, defaults.image_format) == ("fallback", "standard", "png")
    assert required["on_refusal"][0] == ["fallback", "error"] and required["speed"][0] == ["standard", "fast"]
    assert required["image_format"][0] == ["png", "jpeg"]
    assert all(spec[1].get("tooltip") for spec in required.values())
    # added after the release: optional and last, so saved workflows and API prompts keep working
    optional = node.ClaudeConfig.INPUT_TYPES()["optional"]
    assert list(optional) == ["cache_system_prompt"] and optional["cache_system_prompt"][0] == "BOOLEAN"
    assert optional["cache_system_prompt"][1]["default"] is True and optional["cache_system_prompt"][1]["tooltip"]
    assert node.ClaudeConfig.RETURN_TYPES == ("BC_CLAUDE_CONFIG",)


@pytest.fixture
def recorded(pipe, monkeypatch):
    """pipelines/claude/messages.run replaced: the request it got, its images read into a list."""
    seen = {}

    def run(request):
        seen["request"] = request
        seen["images"] = [np.array(image) for image in request.images]
        return pipe.ClaudeResult(text="A red cube.", thinking="Looked.")

    monkeypatch.setattr(pipe, "run", run)
    return seen


def frames(*levels, dtype=torch.float32):
    """One frame per level, 4 x 6 RGB, every pixel at that 8-bit level."""
    return torch.stack([torch.full((4, 6, 3), level / 255) for level in levels]).to(dtype)


def test_claude_hands_widgets_and_frames_in_slot_order(node, pipe, recorded):
    out = node.Claude().generate("Describe.", "Sonnet 5.5", 1000, "max", 7, "Be brief.",
                                 image_10=frames(10), image_2=frames(2, 3), image_1=frames(1))
    assert out == ("A red cube.", "Looked.")
    req = recorded["request"]
    assert (req.model, req.max_tokens, req.effort, req.prompt, req.system_prompt) == (
        "Sonnet 5.5", 1000, "max", "Describe.", "Be brief.")
    assert req.config == pipe.ClaudeConfigValues()
    assert [int(image[0, 0, 0]) for image in recorded["images"]] == [1, 2, 3, 10]
    assert all(image.dtype == np.uint8 and image.shape == (4, 6, 3) for image in recorded["images"])


def test_claude_reads_half_frames_as_their_source(node, recorded):
    node.Claude().generate("x", "Opus 5.5", 10, "low", 0, "", image_1=frames(1, 128, 255, dtype=torch.float16))
    assert [int(image[0, 0, 0]) for image in recorded["images"]] == [1, 128, 255]


def test_claude_passes_the_config(node, pipe, recorded):
    config = pipe.ClaudeConfigValues(model_id="claude-sonnet-5")
    node.Claude().generate("x", "Opus 5.5", 10, "low", 0, "", config=config)
    assert recorded["request"].config is config and recorded["images"] == []


def test_config_build(node, pipe):
    (values,) = node.ClaudeConfig().build("  claude-sonnet-5 \n", "</caption>\n\n  \nEND", "error", "fast", "jpeg", 90, 5)
    assert values == pipe.ClaudeConfigValues(model_id="claude-sonnet-5", stop_sequences=("</caption>", "END"),
                                             on_refusal="error", speed="fast", image_format="jpeg", timeout=90,
                                             max_retries=5, cache_system_prompt=True)
    (values,) = node.ClaudeConfig().build("", "", "fallback", "standard", "png", 600, 2, cache_system_prompt=False)
    assert values.cache_system_prompt is False

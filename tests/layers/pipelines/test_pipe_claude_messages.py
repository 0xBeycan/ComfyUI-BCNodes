"""The Claude node's flow (pipelines/claude/messages.py) with a stand-in SDK client, no network:

  - resized_size: the official reference implementation's sizes (the Vision pages' examples on both tiers, one
    tie where the page's table and its code differ, a tall image as the wide one transposed, an image that fits
    unchanged);
  - encode_image: an image within the limits sent at its own size, a larger one at resized_size (decoded back),
    PNG or JPEG, an encoding over the per-image limit an error naming the image and image_format;
  - the request: the system prompt as written (with a cache breakpoint unless Claude Config turns it off) and none
    when empty, never tools or sampling fields, adaptive
    thinking with the summary, the effort, the model's ID or Claude Config's model_id, one image without a label
    and two with "Image N:" labels, an empty prompt left out (an error with no image), stop sequences, fast
    mode, the server-side fallback (its beta) except on Haiku 5.5 and with on_refusal error;
  - the answer: text blocks joined, thinking summaries joined, a refusal and an answer cut at max_tokens errors, a
    fallback model's answer logged;
  - the client: no key is an error naming the settings, a stored key builds the SDK client with the config's
    timeout and retries; SDK errors become messages that say what to do; ComfyUI's interrupt ends the stream.
"""

import base64
import io
import logging
import sys
import types

import pytest

anthropic = pytest.importorskip("anthropic")
import httpx2  # noqa: E402  (the SDK's HTTP library, installed with it)


@pytest.fixture
def claude(bcnodes):
    return bcnodes["pipelines.claude.messages"]


class InterruptProcessingException(BaseException):
    pass


@pytest.fixture
def interrupt(comfy_stubs, monkeypatch):
    """core's interrupt flag on the stub comfy.model_management: set state["on"] to interrupt."""
    mm = sys.modules["comfy.model_management"]
    state = {"on": False}

    def throw():
        if state["on"]:
            state["on"] = False
            raise InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", throw, raising=False)
    return state


def block(kind, value=""):
    if kind == "text":
        return types.SimpleNamespace(type="text", text=value)
    if kind == "thinking":
        return types.SimpleNamespace(type="thinking", thinking=value, signature="sig")
    return types.SimpleNamespace(type=kind)


def message(content, model="claude-opus-5-5", stop_reason="end_turn", stop_details=None, output_tokens=7):
    return types.SimpleNamespace(content=content, model=model, stop_reason=stop_reason, stop_details=stop_details,
                                 usage=types.SimpleNamespace(input_tokens=11, output_tokens=output_tokens,
                                                       cache_read_input_tokens=0, cache_creation_input_tokens=None))


class StandIn:
    """client.beta.messages.stream as the SDK's context manager: `events` events, then `final`; `enter_error`
    is raised on entering (an API error before the first event)."""

    def __init__(self, final=None, events=3, enter_error=None, on_event=None):
        self.final = final or message([block("text", "A red cube.")])
        self.events, self.enter_error, self.on_event = events, enter_error, on_event
        self.params, self.closed, self.seen = None, False, 0
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self.stream))

    def stream(self, **params):
        self.params = params
        return self

    def __enter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        for i in range(self.events):
            self.seen += 1
            if self.on_event:
                self.on_event(i)
            yield types.SimpleNamespace(type="ping")

    def get_final_message(self):
        return self.final

    request_id = "req_test"


def request(claude, prompt="Describe this.", images=(), model="Opus 5.5", system_prompt="", **config):
    return claude.ClaudeRequest(model=model, max_tokens=32768, effort="high", prompt=prompt, system_prompt=system_prompt,
                                images=iter(images), config=claude.ClaudeConfigValues(**config))


def run_with(claude, monkeypatch, req, client=None):
    client = client or StandIn()
    monkeypatch.setattr(claude, "make_client", lambda config: client)
    return claude.run(req), client


def pil(width, height, color=(200, 30, 30)):
    from PIL import Image

    return Image.new("RGB", (width, height), color)


# --- resized_size -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("size, high, standard", [
    ((200, 200), (200, 200), (200, 200)),
    ((1000, 1000), (1000, 1000), (1000, 1000)),
    ((1920, 1080), (1920, 1080), (1456, 819)),
    # The Vision page's table says 1269x952; its reference implementation gives 1270x952: 1270 x 0.75 = 952.5 is a
    # tie, which the code rounds half to even (952, fits) as its comments say the live API does, and half up (953,
    # does not fit) gives the table's 1269. The reference implementation is the one called exact.
    ((2000, 1500), (2000, 1500), (1270, 952)),
    ((3840, 2160), (2576, 1449), (1456, 819)),
    ((1075, 1520), (1075, 1520), (924, 1307)),
])
def test_resized_size_matches_the_official_examples(claude, size, high, standard):
    assert claude.resized_size(*size) == high
    assert claude.resized_size(*size, max_edge=1568, max_tokens=1568) == standard


def test_resized_size_square_and_tall(claude):
    # under the edge limit but over the token limit: 72 x 72 = 5184 patches > 4784
    assert claude.resized_size(2000, 2000) == (1932, 1932)
    assert claude.resized_size(2160, 3840) == (1449, 2576)


# --- encode_image -------------------------------------------------------------------------------------------


def decode(source):
    from PIL import Image

    return Image.open(io.BytesIO(base64.standard_b64decode(source["data"])))


def test_encode_image_keeps_a_fitting_image_and_downscales_a_larger_one(claude):
    small = claude.encode_image(pil(1920, 1080), "png", 1)
    assert small["type"] == "base64" and small["media_type"] == "image/png"
    assert decode(small).size == (1920, 1080) and decode(small).format == "PNG"
    big = claude.encode_image(pil(3840, 2160), "jpeg", 2)
    assert big["media_type"] == "image/jpeg"
    assert decode(big).size == (2576, 1449) and decode(big).format == "JPEG"


def test_encode_image_png_is_lossless(claude):
    import numpy as np
    from PIL import Image

    pixels = np.random.default_rng(0).integers(0, 256, (64, 48, 3), dtype=np.uint8)
    assert np.array_equal(np.array(decode(claude.encode_image(Image.fromarray(pixels), "png", 1))), pixels)


def test_encode_image_over_the_limit_names_the_image_and_the_way_out(claude, monkeypatch):
    import numpy as np
    from PIL import Image

    monkeypatch.setattr(claude, "MAX_IMAGE_BYTES", 1000)
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"image 3 is .* PNG at 64x64.*image_format to jpeg"):
        claude.encode_image(noise, "png", 3)


# --- the request --------------------------------------------------------------------------------------------


def test_request_holds_only_what_the_node_gives(claude, monkeypatch, interrupt):
    _, client = run_with(claude, monkeypatch, request(claude))
    p = client.params
    assert p["model"] == "claude-opus-5-5" and p["max_tokens"] == 32768
    assert p["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Describe this."}]}]
    assert p["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert p["output_config"] == {"effort": "high"}
    assert "system" not in p  # an empty system prompt sends none
    for absent in ("tools", "tool_choice", "temperature", "top_p", "top_k", "stop_sequences", "speed"):
        assert absent not in p
    # the server-side fallback, on by default
    assert p["betas"] == [claude.FALLBACK_BETA] and p["fallbacks"] == "default"


def test_system_prompt_as_written_cached_whitespace_only_none(claude, monkeypatch, interrupt):
    # cached by default: the breakpoint on the system prompt, the one part that repeats between runs
    _, client = run_with(claude, monkeypatch, request(claude, system_prompt="  Be brief.\n"))
    assert client.params["system"] == [{"type": "text", "text": "  Be brief.\n", "cache_control": {"type": "ephemeral"}}]
    assert "cache_control" not in client.params  # no top-level breakpoint: it would sit on the changing prompt
    _, client = run_with(claude, monkeypatch, request(claude, system_prompt="  Be brief.\n", cache_system_prompt=False))
    assert client.params["system"] == "  Be brief.\n"
    _, client = run_with(claude, monkeypatch, request(claude, system_prompt=" \n "))
    assert "system" not in client.params


def test_images_before_the_prompt_labelled_from_two(claude, monkeypatch, interrupt):
    _, client = run_with(claude, monkeypatch, request(claude, images=[pil(8, 8)]))
    content = client.params["messages"][0]["content"]
    assert [b["type"] for b in content] == ["image", "text"]
    _, client = run_with(claude, monkeypatch, request(claude, images=[pil(8, 8), pil(9, 9), pil(10, 10)]))
    content = client.params["messages"][0]["content"]
    assert [b.get("text") for b in content if b["type"] == "text"] == ["Image 1:", "Image 2:", "Image 3:", "Describe this."]
    assert [b["type"] for b in content] == ["text", "image"] * 3 + ["text"]
    assert [decode(b["source"]).size for b in content if b["type"] == "image"] == [(8, 8), (9, 9), (10, 10)]


def test_empty_prompt_left_out_and_an_error_without_images(claude, monkeypatch, interrupt):
    _, client = run_with(claude, monkeypatch, request(claude, prompt="  ", images=[pil(8, 8)]))
    assert [b["type"] for b in client.params["messages"][0]["content"]] == ["image"]
    with pytest.raises(ValueError, match="prompt is empty and no image is connected"):
        run_with(claude, monkeypatch, request(claude, prompt=""))


def test_config_fields_reach_the_request(claude, monkeypatch, interrupt):
    _, client = run_with(claude, monkeypatch, request(claude, model="Sonnet 5.5", model_id="claude-sonnet-5",
                                                      stop_sequences=("</caption>", "END"), speed="fast",
                                                      on_refusal="error"))
    p = client.params
    assert p["model"] == "claude-sonnet-5" and p["stop_sequences"] == ["</caption>", "END"] and p["speed"] == "fast"
    assert "fallbacks" not in p and "betas" not in p
    _, client = run_with(claude, monkeypatch, request(claude, model="Haiku 5.5"))
    assert client.params["model"] == "claude-haiku-5-5" and "fallbacks" not in client.params
    _, client = run_with(claude, monkeypatch, request(claude, model="Fable 5.2", image_format="jpeg", images=[pil(8, 8)]))
    assert client.params["model"] == "claude-fable-5-2" and client.params["fallbacks"] == "default"
    assert client.params["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


# --- the answer ---------------------------------------------------------------------------------------------


def test_text_and_thinking_joined(claude, monkeypatch, interrupt):
    final = message([block("thinking", "Look at the shape."), block("text", "A red "), block("thinking", ""),
                     block("text", "cube.")])
    result, client = run_with(claude, monkeypatch, request(claude), StandIn(final))
    assert (result.text, result.thinking) == ("A red cube.", "Look at the shape.")
    assert client.closed and client.seen == 3


def test_refusal_is_an_error_with_its_category(claude, monkeypatch, interrupt):
    details = types.SimpleNamespace(category="cyber", explanation="Declined by policy.")
    final = message([block("text", "partial")], stop_reason="refusal", stop_details=details)
    with pytest.raises(RuntimeError, match=r"claude-opus-5-5 declined the request \(cyber\): Declined by policy\."):
        run_with(claude, monkeypatch, request(claude), StandIn(final))
    with pytest.raises(RuntimeError, match=r"declined the request\. Rephrase"):
        run_with(claude, monkeypatch, request(claude), StandIn(message([], stop_reason="refusal")))


def test_answer_cut_at_max_tokens_is_an_error(claude, monkeypatch, interrupt):
    final = message([block("text", "A red")], stop_reason="max_tokens", output_tokens=32768)
    with pytest.raises(RuntimeError, match=r"max_tokens \(32768 tokens.*Raise max_tokens or lower effort"):
        run_with(claude, monkeypatch, request(claude), StandIn(final))


def test_stop_sequence_is_an_answer(claude, monkeypatch, interrupt):
    final = message([block("text", "A red cube.")], stop_reason="stop_sequence")
    result, _ = run_with(claude, monkeypatch, request(claude), StandIn(final))
    assert result.text == "A red cube."


def test_fallback_answer_is_logged(claude, monkeypatch, interrupt, caplog):
    final = message([block("fallback"), block("text", "A red cube.")], model="claude-opus-4-8")
    with caplog.at_level(logging.WARNING):
        result, _ = run_with(claude, monkeypatch, request(claude), StandIn(final))
    assert result.text == "A red cube."
    assert "claude-opus-5-5 declined the request and claude-opus-4-8 answered it" in caplog.text


# --- the client, errors, interrupt --------------------------------------------------------------------------


@pytest.fixture
def user_dir(comfy_stubs, monkeypatch, tmp_path):
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path / "user"))
    return tmp_path / "user"


def test_no_key_is_an_error_naming_the_settings(claude, user_dir):
    with pytest.raises(ValueError, match=r"no Anthropic API key is set.*Settings -> BCNodes -> Claude"):
        claude.make_client(claude.ClaudeConfigValues())


def test_stored_key_builds_the_sdk_client(claude, bcnodes, user_dir, monkeypatch):
    bcnodes["libs.download"].save_tokens({"anthropic": " sk-ant-test "})
    made = {}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: made.update(kw) or "client")
    assert claude.make_client(claude.ClaudeConfigValues(timeout=90, max_retries=5)) == "client"
    assert made == {"api_key": "sk-ant-test", "timeout": 90, "max_retries": 5}


def status_error(cls, status, text):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    res = httpx2.Response(status, request=req, headers={"request-id": "req_42"},
                          json={"type": "error", "error": {"type": "x", "message": text}})
    return cls(text, response=res, body={"type": "error", "error": {"type": "x", "message": text}})


def test_sdk_errors_say_what_to_do(claude, monkeypatch, interrupt):
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    cases = [
        (status_error(anthropic.AuthenticationError, 401, "invalid x-api-key"),
         r"API key was not accepted \(401\) \(request req_42\)\. Enter a valid key"),
        (status_error(anthropic.BadRequestError, 400, "model: unknown"),
         r"the API answered 400 \(request req_42\): model: unknown"),
        (anthropic.APITimeoutError(request=req), r"no answer within 600 s\. Raise timeout"),
        (anthropic.APIConnectionError(request=req), r"cannot reach the Anthropic API"),
    ]
    for error, pattern in cases:
        with pytest.raises(RuntimeError, match=pattern):
            run_with(claude, monkeypatch, request(claude), StandIn(enter_error=error))


def test_interrupt_ends_the_stream(claude, monkeypatch, interrupt):
    def cancel(i):
        if i == 1:
            interrupt["on"] = True

    client = StandIn(events=10, on_event=cancel)
    with pytest.raises(InterruptProcessingException):
        run_with(claude, monkeypatch, request(claude), client)
    assert client.closed and client.seen == 2

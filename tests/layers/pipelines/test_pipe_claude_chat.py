"""Claude Chat (pipelines/claude/chat.py) with a stand-in SDK client, no network, sessions in a temporary user
directory:

  - sessions: created with the defaults (Opus 5.5, high, 64000, no system prompt) or the given settings, listed
    newest first, loaded, changed, deleted; an unknown setting, a value outside its range and an id that is not a
    32-hex token are errors (no path from an id);
  - a turn: the events streamed (answer and thinking pieces), the request (the history plus the new message, the
    top-level cache_control, adaptive thinking with the summary, the effort and max_tokens, the system prompt only
    when set, the fallback except on Haiku 5.5, no tools), the turn saved (the user message, the assistant's blocks
    as returned, thinking signatures included, the usage, the title from the first line);
  - the next turn sends the saved history unchanged before its new message;
  - images: attachments sized and encoded as the node's, labelled from two, JPEG kept, an unknown type an error;
  - not saved: a refusal, an API error, a cancel (the stream closed), an empty message (raises);
  - after a fallback the thinking before the last fallback block is left out of the saved blocks.
"""

import base64
import io
import sys
import types

import pytest

anthropic = pytest.importorskip("anthropic")
import anthropic.types.beta as bt  # noqa: E402
import httpx2  # noqa: E402


@pytest.fixture
def chat(bcnodes, comfy_stubs, monkeypatch, tmp_path):
    monkeypatch.setattr(sys.modules["folder_paths"], "get_user_directory", lambda: str(tmp_path / "user"))
    return bcnodes["pipelines.claude.chat"]


@pytest.fixture
def messages(bcnodes):
    return bcnodes["pipelines.claude.messages"]


def delta(kind, text):
    field = "text" if kind == "text_delta" else "thinking"
    return types.SimpleNamespace(type="content_block_delta", delta=types.SimpleNamespace(type=kind, **{field: text}))


def final(blocks, stop_reason="end_turn", model="claude-opus-5-5", stop_details=None):
    usage = types.SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=100,
                                  cache_creation_input_tokens=None)
    return types.SimpleNamespace(content=blocks, stop_reason=stop_reason, model=model, stop_details=stop_details,
                                 usage=usage)


def text_block(text):
    return bt.BetaTextBlock(type="text", text=text)


def thinking_block(text, signature="sig-1"):
    return bt.BetaThinkingBlock(type="thinking", thinking=text, signature=signature)


class StandIn:
    def __init__(self, events=None, message=None, enter_error=None):
        self.events = events if events is not None else [delta("thinking_delta", "Hmm."), delta("text_delta", "Hi "),
                                                         delta("text_delta", "there.")]
        self.message = message or final([thinking_block("Hmm."), text_block("Hi there.")])
        self.enter_error, self.params, self.closed = enter_error, [], False
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(stream=self.stream))

    def stream(self, **params):
        self.params.append(params)
        return self

    def __enter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return self.message


@pytest.fixture
def client(messages, monkeypatch):
    stand_in = StandIn()
    monkeypatch.setattr(messages, "make_client", lambda config: stand_in)
    return stand_in


def run(chat, session_id, text="Hello", images=(), cancelled=lambda: False):
    return list(chat.send(session_id, text, list(images), cancelled))


# --- sessions -----------------------------------------------------------------------------------------------


def test_sessions_lifecycle(chat):
    a = chat.create()
    assert (a["model"], a["effort"], a["max_tokens"], a["system"], a["title"], a["messages"]) == (
        "Opus 5.5", "high", 64000, "", "", [])
    b = chat.create({"model": "Haiku 5.5", "system": "Be brief."})
    assert (b["model"], b["system"]) == ("Haiku 5.5", "Be brief.")
    chat.update(a["id"], {"effort": "low", "max_tokens": 1000, "title": "Mine"})
    assert [s["id"] for s in chat.list_sessions()] == [a["id"], b["id"]]  # a changed last
    assert chat.load(a["id"])["effort"] == "low" and chat.load(a["id"])["title"] == "Mine"
    chat.delete(b["id"])
    assert [s["id"] for s in chat.list_sessions()] == [a["id"]]
    with pytest.raises(ValueError, match="no session"):
        chat.load(b["id"])


@pytest.mark.parametrize("changes, message", [
    ({"colour": "red"}, "unknown setting"), ({"model": "Opus 9"}, "no model"), ({"effort": "huge"}, "no effort"),
    ({"max_tokens": 0}, "max_tokens must be"), ({"max_tokens": 128001}, "max_tokens must be"),
    ({"system": 3}, "system must be text"),
])
def test_bad_settings_are_errors(chat, changes, message):
    with pytest.raises(ValueError, match=message):
        chat.create(changes)


@pytest.mark.parametrize("bad", ["../../etc/passwd", "ABC", "0" * 31, None])
def test_session_ids_are_plain_tokens(chat, bad):
    with pytest.raises(ValueError, match="no session"):
        chat.load(bad)


# --- a turn -------------------------------------------------------------------------------------------------


def test_turn_streams_requests_and_saves(chat, client):
    session = chat.create({"system": "Be brief."})
    events = run(chat, session["id"], "Hello\nsecond line")
    assert events[:3] == [{"type": "thinking", "text": "Hmm."}, {"type": "text", "text": "Hi "},
                          {"type": "text", "text": "there."}]
    done = events[-1]
    assert done["type"] == "done" and done["session"]["title"] == "Hello"
    assert done["turn"] == {"model": "claude-opus-5-5", "stop_reason": "end_turn",
                            "usage": {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 100,
                                      "cache_creation_input_tokens": 0}}
    p = client.params[0]
    assert p["model"] == "claude-opus-5-5" and p["max_tokens"] == 64000 and p["system"] == "Be brief."
    assert p["cache_control"] == {"type": "ephemeral"}
    assert p["thinking"] == {"type": "adaptive", "display": "summarized"} and p["output_config"] == {"effort": "high"}
    assert p["fallbacks"] == "default" and p["betas"] == [messages_beta(chat)]
    assert p["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Hello\nsecond line"}]}]
    assert "tools" not in p and "temperature" not in p
    saved = chat.load(session["id"])
    assert saved["messages"][1] == {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "Hmm.", "signature": "sig-1"}, {"type": "text", "text": "Hi there."}]}
    assert saved["turns"] == [done["turn"]] and client.closed


def messages_beta(chat):
    return chat.messages.FALLBACK_BETA


def test_next_turn_sends_the_history_unchanged(chat, client):
    session = chat.create()
    run(chat, session["id"], "First")
    history = chat.load(session["id"])["messages"]
    run(chat, session["id"], "Second")
    assert client.params[1]["messages"] == history + [{"role": "user", "content": [{"type": "text", "text": "Second"}]}]
    assert len(chat.load(session["id"])["messages"]) == 4 and chat.load(session["id"])["title"] == "First"


def test_no_system_and_no_fallback_on_haiku(chat, client):
    session = chat.create({"model": "Haiku 5.5", "system": "  "})
    run(chat, session["id"])
    p = client.params[0]
    assert p["model"] == "claude-haiku-5-5" and "system" not in p and "fallbacks" not in p and "betas" not in p


def png_b64(width, height):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def jpeg_b64(width, height):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def test_images_sized_labelled_and_jpeg_kept(chat, client):
    session = chat.create()
    run(chat, session["id"], "Compare.", [{"media_type": "image/png", "data": png_b64(3840, 2160)},
                                          {"media_type": "image/jpeg", "data": jpeg_b64(16, 16)}])
    content = client.params[0]["messages"][0]["content"]
    assert [b.get("text") for b in content if b["type"] == "text"] == ["Image 1:", "Image 2:", "Compare."]
    images = [b["source"] for b in content if b["type"] == "image"]
    assert [s["media_type"] for s in images] == ["image/png", "image/jpeg"]
    from PIL import Image

    assert Image.open(io.BytesIO(base64.standard_b64decode(images[0]["data"]))).size == (2576, 1449)
    with pytest.raises(ValueError, match="image 1 is 'image/bmp'"):
        run(chat, session["id"], "x", [{"media_type": "image/bmp", "data": ""}])


# --- not saved ----------------------------------------------------------------------------------------------


def test_refusal_error_cancel_and_empty_save_nothing(chat, client, messages, monkeypatch):
    session = chat.create()
    client.message = final([], stop_reason="refusal", stop_details=types.SimpleNamespace(category="cyber"))
    assert run(chat, session["id"])[-1] == {"type": "error", "error": "Claude: claude-opus-5-5 declined the message "
                                            "(cyber). Nothing was saved; rephrase it or pick another model."}
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    res = httpx2.Response(529, request=req, json={"type": "error", "error": {"type": "overloaded_error",
                                                                              "message": "Overloaded"}})
    failing = StandIn(enter_error=anthropic.InternalServerError("Overloaded", response=res, body=None))
    monkeypatch.setattr(messages, "make_client", lambda config: failing)
    error = run(chat, session["id"])[-1]
    assert error["type"] == "error" and "the API answered 529" in error["error"]
    cancelling = StandIn()
    monkeypatch.setattr(messages, "make_client", lambda config: cancelling)
    assert run(chat, session["id"], cancelled=lambda: True) == [{"type": "cancelled"}] and cancelling.closed
    with pytest.raises(ValueError, match="the message is empty"):
        run(chat, session["id"], "  ")
    saved = chat.load(session["id"])
    assert saved["messages"] == [] and saved["turns"] == [] and saved["title"] == ""


def test_thinking_before_the_last_fallback_is_left_out(chat, client):
    fallback = bt.BetaFallbackBlock.model_construct(type="fallback", from_=types.SimpleNamespace(model="a"),
                                                    to=types.SimpleNamespace(model="b"))
    fallback.to_dict = lambda exclude_none=True: {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}}
    client.message = final([thinking_block("before"), text_block("partial "), fallback, thinking_block("after", "s2"),
                            text_block("rest")], model="claude-opus-4-8")
    session = chat.create()
    run(chat, session["id"])
    assert [b["type"] for b in chat.load(session["id"])["messages"][1]["content"]] == [
        "text", "fallback", "thinking", "text"]

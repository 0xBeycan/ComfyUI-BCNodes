"""Claude Chat: the conversations of the Claude sidebar tab (web/js/claude_chat.js), kept on the ComfyUI server, one
JSON file per session in user/BCNodes/claude_chat/ (a session id is a plain 32-hex token, so it names a file).

A session holds its settings (model, effort, max_tokens, system prompt), its title (the first message's first line)
and its messages as the API takes them: every turn sends the whole history plus the new user message. The history
only grows at its end, and the assistant's content blocks are stored as the API returned them (thinking blocks with
their signatures), so the next turn passes them back unchanged, as the current models require. Prompt caching is on
(the top-level cache_control: the breakpoint follows the last block, so each turn reads the history before it from
the cache); it changes the price, never the answer. A turn is saved only when it ends with an answer: an error, a
refusal or a cancel leaves the session as it was, and the message can be sent again.

Each request is built like the Claude node's (pipelines/claude/messages.py: the models, the client and its key, the
image sizing and encoding, the Image N: labels, adaptive thinking with its summary, the server-side refusal
fallback), with no tools.
"""

import base64
import io
import json
import os
import re
import time
import uuid

from . import messages

DEFAULT_MAX_TOKENS = 64000  # the streaming default the official SDK guide gives
_ID = re.compile(r"^[0-9a-f]{32}$")
SETTINGS = ("title", "model", "effort", "max_tokens", "system")
IMAGE_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "png", "image/gif": "png"}


def chat_dir():
    from ...libs.download import user_file

    return user_file("claude_chat")


def _path(session_id):
    if not isinstance(session_id, str) or not _ID.match(session_id):
        raise ValueError(f"Claude Chat: no session {session_id!r}")
    return os.path.join(chat_dir(), session_id + ".json")


def load(session_id):
    try:
        with open(_path(session_id), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Claude Chat: no session {session_id!r}") from None


def save(session):
    """Writes the session through a temporary file, so a crash never leaves half a file."""
    path = _path(session["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(session, f)
    os.replace(tmp, path)


def summary(session):
    """A session's entry in the session list."""
    return {key: session[key] for key in ("id", "title", "model", "updated")}


def options():
    """The settings' choices the panel shows: the models, the efforts, the max_tokens range and default."""
    return {"models": list(messages.MODELS), "efforts": messages.EFFORTS, "max_tokens": messages.MAX_TOKENS,
            "default_max_tokens": DEFAULT_MAX_TOKENS}


def list_sessions():
    """Every session's summary, the most recently updated first."""
    folder = chat_dir()
    names = [n[:-5] for n in os.listdir(folder) if n.endswith(".json")] if os.path.isdir(folder) else []
    return sorted((summary(load(n)) for n in names if _ID.match(n)), key=lambda s: s["updated"], reverse=True)


def _settings(changes):
    """`changes` checked against SETTINGS: an unknown name or a value outside its widget's range is an error."""
    unknown = sorted(set(changes) - set(SETTINGS))
    if unknown:
        raise ValueError(f"Claude Chat: unknown setting(s) {', '.join(unknown)}; the settings are {', '.join(SETTINGS)}")
    if "model" in changes and changes["model"] not in messages.MODELS:
        raise ValueError(f"Claude Chat: no model {changes['model']!r}; the models are {', '.join(messages.MODELS)}")
    if "effort" in changes and changes["effort"] not in messages.EFFORTS:
        raise ValueError(f"Claude Chat: no effort {changes['effort']!r}; the efforts are {', '.join(messages.EFFORTS)}")
    if "max_tokens" in changes and not (isinstance(changes["max_tokens"], int)
                                        and 1 <= changes["max_tokens"] <= messages.MAX_TOKENS):
        raise ValueError(f"Claude Chat: max_tokens must be an integer from 1 to {messages.MAX_TOKENS}")
    for key in ("title", "system"):
        if key in changes and not isinstance(changes[key], str):
            raise ValueError(f"Claude Chat: {key} must be text")
    return changes


def create(changes=None):
    now = time.time()
    session = {"id": uuid.uuid4().hex, "title": "", "model": messages.DEFAULT_MODEL, "effort": messages.DEFAULT_EFFORT,
               "max_tokens": DEFAULT_MAX_TOKENS, "system": "", "created": now, "updated": now, "messages": [],
               "turns": []}
    session.update(_settings(changes or {}))
    save(session)
    return session


def update(session_id, changes):
    session = load(session_id)
    session.update(_settings(changes))
    session["updated"] = time.time()
    save(session)
    return session


def delete(session_id):
    try:
        os.remove(_path(session_id))
    except FileNotFoundError:
        raise ValueError(f"Claude Chat: no session {session_id!r}") from None


def _sources(images):
    """The browser's attachments ({"media_type", "data": base64}) as image block sources: decoded, sized and encoded
    as the Claude node's images (JPEG stays JPEG, any other type becomes PNG)."""
    from PIL import Image

    sources = []
    for number, image in enumerate(images, 1):
        image_format = IMAGE_TYPES.get(image.get("media_type"))
        if image_format is None:
            raise ValueError(f"Claude Chat: image {number} is {image.get('media_type')!r}; the API takes "
                             f"{', '.join(IMAGE_TYPES)}")
        pil = Image.open(io.BytesIO(base64.standard_b64decode(image["data"])))
        pil.load()
        if pil.mode not in ("RGB", "RGBA", "L"):
            pil = pil.convert("RGBA" if "A" in pil.getbands() else "RGB")
        try:
            sources.append(messages.encode_image(pil, image_format, number))
        except ValueError:
            raise ValueError(f"Claude Chat: image {number} is over the API's {messages.MAX_IMAGE_BYTES // 2 ** 20} MB "
                             "per image as PNG; attach it as a JPEG.") from None
    return sources


def turn_params(session, content):
    """The keyword arguments of the SDK's client.beta.messages.stream for the next turn: the history plus a user
    message of `content`."""
    model = messages.MODELS[session["model"]]
    params = {"model": model, "max_tokens": session["max_tokens"],
              "messages": session["messages"] + [{"role": "user", "content": content}],
              "thinking": {"type": "adaptive", "display": "summarized"},
              "output_config": {"effort": session["effort"]},
              "cache_control": {"type": "ephemeral"}}
    if session["system"].strip():
        params["system"] = session["system"]
    if model not in messages.NO_FALLBACK:
        params["betas"] = [messages.FALLBACK_BETA]
        params["fallbacks"] = "default"
    return params


def assistant_content(blocks):
    """The final message's content blocks as the next turn sends them back: as the API returned them, except that
    after a fallback the thinking, redacted thinking and tool-use blocks before the last fallback block are left out
    (the official rule for echoing a fallback turn); the fallback block stays where it was."""
    blocks = [block.to_dict(exclude_none=True) for block in blocks]
    last = max((i for i, block in enumerate(blocks) if block["type"] == "fallback"), default=-1)
    return [block for i, block in enumerate(blocks)
            if i >= last or block["type"] not in ("thinking", "redacted_thinking", "tool_use")]


def send(session_id, text, images, cancelled):
    """One turn of the session: yields the events the browser shows, a dict each. {"type": "text" | "thinking",
    "text"}: a piece of the answer or of the thinking summary; {"type": "fallback", "from", "to"}: another model
    took over after a refusal; {"type": "done", "session", "turn"}: the turn is saved (the session's summary, the
    turn's model, usage and stop reason); {"type": "error", "error"}: nothing was saved; {"type": "cancelled"}:
    `cancelled()` turned true (the browser left), the stream was closed and nothing was saved. Errors before the
    request (no key, an empty message, an unusable image) raise."""
    import anthropic

    session = load(session_id)
    if not text.strip() and not images:
        raise ValueError("Claude Chat: the message is empty.")
    client = messages.make_client(messages.ClaudeConfigValues())
    content = messages.user_content(text, _sources(images))
    params = turn_params(session, content)
    try:
        with client.beta.messages.stream(**params) as stream:
            for event in stream:
                if cancelled():
                    yield {"type": "cancelled"}
                    return
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield {"type": "text", "text": event.delta.text}
                elif event.type == "content_block_delta" and event.delta.type == "thinking_delta":
                    yield {"type": "thinking", "text": event.delta.thinking}
                elif event.type == "content_block_start" and event.content_block.type == "fallback":
                    yield {"type": "fallback", "from": event.content_block.from_.model,
                           "to": event.content_block.to.model}
            message = stream.get_final_message()
    except anthropic.APIError as error:
        yield {"type": "error", "error": messages.api_error(error, messages.ClaudeConfigValues().timeout)}
        return
    if message.stop_reason == "refusal":
        details = message.stop_details
        category = f" ({details.category})" if details and details.category else ""
        yield {"type": "error", "error": f"Claude: {message.model} declined the message{category}. Nothing was saved; "
                                         "rephrase it or pick another model."}
        return
    usage = message.usage
    turn = {"model": message.model, "stop_reason": message.stop_reason,
            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                      "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
                      "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0}}
    session["messages"] += [{"role": "user", "content": content},
                            {"role": "assistant", "content": assistant_content(message.content)}]
    session["turns"].append(turn)
    if not session["title"]:
        first = text.strip().splitlines()[0] if text.strip() else "Images"
        session["title"] = first[:60]
    session["updated"] = time.time()
    save(session)
    yield {"type": "done", "session": summary(session), "turn": turn}

"""Claude nodes: Anthropic's Claude models through the Messages API with the user's own API key (no Comfy.org
service in between).

    BC_Claude        (Claude)         prompt, model, max_tokens, effort, seed, system_prompt; image_1, image_2, ...;
                                      config -> text, thinking
    BC_ClaudeConfig  (Claude Config)  the request's other settings -> BC_CLAUDE_CONFIG (a
                                      pipelines/claude/messages.ClaudeConfigValues)

The flow is pipelines/claude/messages.py. The API key is entered in ComfyUI's settings (web/js/claude.js), kept
server side by libs/download's token store and never written into a workflow. web/js/claude.js also grows the image
inputs: one empty image_N after the last connected one; they arrive here through **images (the optional mapping
answers for any name, nodes/common.FlexibleOptionalInputType).

Also here, not a node: the routes of Claude Chat, the Claude sidebar tab (web/js/claude_chat.js), registered by
register_routes() (called from the root __init__) only inside a running ComfyUI server; their flow is
pipelines/claude/chat.py. server and aiohttp are imported inside the function.
"""

import re

from .common import FlexibleOptionalInputType, slot_index

_SLOT = re.compile(r"^image_(\d+)$")


class Claude:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.claude.messages import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, EFFORTS, MAX_TOKENS, MODELS

        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "The user message, sent as written after the images. Empty only when "
                                                 "an image is connected."}),
                "model": (list(MODELS), {"default": DEFAULT_MODEL,
                                         "tooltip": "The Claude model: Opus 5.5 ($4 / $20 per million input / output "
                                                    "tokens), Sonnet 5.5 ($2 / $10), Haiku 5.5 ($0.10 / $0.50), Fable "
                                                    "5.2 ($8 / $40). model_id on a Claude Config node runs any other "
                                                    "model."}),
                "max_tokens": ("INT", {"default": DEFAULT_MAX_TOKENS, "min": 1, "max": MAX_TOKENS,
                                       "tooltip": "The most tokens the model writes, thinking and answer together. "
                                                  "Only the tokens written are billed. An answer cut here is an error, "
                                                  "never a partial text."}),
                "effort": (EFFORTS, {"default": DEFAULT_EFFORT,
                                     "tooltip": "How much the model thinks and how many tokens it spends. The API's "
                                                "own default is medium on Opus 5.5 and Haiku 5.5, high on the "
                                                "others. Thinking is always on with these models; effort is its "
                                                "control."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": "fixed",
                                 "tooltip": "Not sent: the API has no seed. It only decides whether ComfyUI runs the "
                                            "node again. Its control starts at fixed, so queueing again gives the "
                                            "cached answer and spends nothing; randomize asks again on every "
                                            "queue."}),
                "system_prompt": ("STRING", {"default": "", "multiline": True,
                                             "tooltip": "The system prompt, sent as written; it is the whole system "
                                                        "prompt (the node sends no tools, so the API adds none). "
                                                        "Empty: no system prompt."}),
            },
            "optional": FlexibleOptionalInputType("IMAGE", known={
                "config": ("BC_CLAUDE_CONFIG", {"tooltip": "A Claude Config node: model_id, stop sequences, refusal "
                                                           "fallback, speed, image format, timeout, retries and "
                                                           "system prompt caching. Not connected: its defaults."}),
                "image_1": ("IMAGE", {"tooltip": "Images the model sees, before the prompt: every frame of the batch "
                                                 "is one image. Connecting the last image input adds the next one. "
                                                 "With two or more images each is labelled Image 1:, Image 2:, ... "
                                                 "in input order. An image over 2576 px or 4784 visual tokens is "
                                                 "downscaled to the size the API itself would use."}),
            }),
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "thinking")
    OUTPUT_TOOLTIPS = ("The answer.", "A summary of the model's thinking; empty when it did not think.")
    FUNCTION = "generate"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "claude", "anthropic", "opus", "sonnet", "haiku", "llm", "vlm", "api", "caption",
                      "prompt enhancer", "chat"]
    DESCRIPTION = ("Asks a Claude model through Anthropic's Messages API with your own API key (ComfyUI's Settings -> "
                   "BCNodes -> Claude; no Comfy.org service): the system prompt and the prompt as written, the "
                   "images from image_1, image_2, ... before the prompt. Each run is one independent request: no "
                   "history, no tools, nothing added but the Image N: labels of two or more images. text is the "
                   "answer, thinking a summary of the model's thinking. A Claude Config node sets the rest.")

    def generate(self, prompt, model, max_tokens, effort, seed, system_prompt, config=None, **images):
        """`seed` only decides whether ComfyUI runs the node again. Every frame of image_1, image_2, ... in slot order
        is one image, converted to 8-bit as the pipeline reads it."""
        from ..libs.image import tensor_to_pil_u8
        from ..pipelines.claude.messages import ClaudeConfigValues, ClaudeRequest, run

        tensors = [images[name] for name in sorted(images, key=lambda name: slot_index(_SLOT, name))]
        result = run(ClaudeRequest(model=model, max_tokens=max_tokens, effort=effort, prompt=prompt,
                                   system_prompt=system_prompt,
                                   images=(tensor_to_pil_u8(frame) for tensor in tensors for frame in tensor),
                                   config=config or ClaudeConfigValues()))
        return (result.text, result.thinking)


class ClaudeConfig:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.claude.messages import IMAGE_FORMATS, ON_REFUSAL, SPEEDS, ClaudeConfigValues

        defaults = ClaudeConfigValues()
        return {
            "required": {
                "model_id": ("STRING", {"default": defaults.model_id,
                                        "tooltip": "A model ID (e.g. claude-sonnet-5) that replaces the Claude node's "
                                                   "model: one the list does not have, or a newer one. Empty: the "
                                                   "node's model."}),
                "stop_sequences": ("STRING", {"default": "", "multiline": True,
                                              "tooltip": "One per line; the answer ends where one appears (it is not "
                                                         "included). Blank lines are ignored. Empty: none."}),
                "on_refusal": (ON_REFUSAL, {"default": defaults.on_refusal,
                                            "tooltip": "When the model's safety classifiers decline the request. "
                                                       "fallback: the API runs it again on the model Anthropic "
                                                       "recommends for that refusal category, billed at that model's "
                                                       "rates, and the console names the model that answered (not on "
                                                       "Haiku 5.5, which has no fallback). error: the run fails. A "
                                                       "refusal of the whole chain fails the run either way."}),
                "speed": (SPEEDS, {"default": defaults.speed,
                                   "tooltip": "fast: the model's fast mode, up to 2.5x faster output at a higher "
                                              "price (Opus 5.5: $8 / $40 per million tokens; Fable 5.2: $16 / $80). "
                                              "Another model rejects it."}),
                "image_format": (IMAGE_FORMATS, {"default": defaults.image_format,
                                                 "tooltip": "How images are uploaded. png: lossless. jpeg: quality 95, "
                                                            "much smaller; for an image whose PNG exceeds the API's "
                                                            "10 MB per image."}),
                "timeout": ("INT", {"default": defaults.timeout, "min": 1, "max": 3600,
                                    "tooltip": "Seconds the request may wait for the API (each read of the stream). "
                                               "600 is the SDK's default."}),
                "max_retries": ("INT", {"default": defaults.max_retries, "min": 0, "max": 10,
                                        "tooltip": "How often the SDK retries a connection error, a rate limit (429) "
                                                   "or a server error, with backoff. 2 is the SDK's default."}),
            },
            # Added after the node's release, optional so an API-format prompt saved without it still validates.
            "optional": {
                "cache_system_prompt": ("BOOLEAN", {"default": defaults.cache_system_prompt,
                                                    "tooltip": "Prompt caching of the system prompt: a run within 5 "
                                                               "minutes of the last reads it at about a tenth of the "
                                                               "input price (the first writes it at 1.25x); the answer "
                                                               "is the same. Pays off when many runs share one long "
                                                               "system prompt (captioning a set); below the model's "
                                                               "minimum (512 tokens on Opus 5.5) nothing is cached "
                                                               "and nothing extra is charged."}),
            },
        }

    RETURN_TYPES = ("BC_CLAUDE_CONFIG",)
    RETURN_NAMES = ("config",)
    OUTPUT_TOOLTIPS = ("For a Claude node's config input.",)
    FUNCTION = "build"
    CATEGORY = "BCNodes/lm"
    SEARCH_ALIASES = ["BCNodes", "claude config", "anthropic", "stop sequences", "fallback", "fast mode"]
    DESCRIPTION = ("The Claude node's other settings: a model ID that replaces its model, stop sequences, what a "
                   "refusal does, fast mode, the image upload format, the timeout, the retries and prompt caching of "
                   "the system prompt.")

    def build(self, model_id, stop_sequences, on_refusal, speed, image_format, timeout, max_retries,
              cache_system_prompt=True):
        """cache_system_prompt was added after the node's release: a saved workflow without it gets the default."""
        from ..pipelines.claude.messages import ClaudeConfigValues

        return (ClaudeConfigValues(model_id=model_id.strip(),
                                   stop_sequences=tuple(line for line in stop_sequences.splitlines() if line.strip()),
                                   on_refusal=on_refusal, speed=speed, image_format=image_format, timeout=timeout,
                                   max_retries=max_retries, cache_system_prompt=cache_system_prompt),)


def register_routes():
    """Claude Chat's routes, JSON in and out (an error is {"error"} with status 400):

        GET  /bcnodes/claude/chats              the session list and the settings' choices (chat.options)
        POST /bcnodes/claude/chats              a new session ({settings} -> the session)
        GET  /bcnodes/claude/chats/{id}         one session
        POST /bcnodes/claude/chats/{id}         change its settings ({settings} -> the session)
        POST /bcnodes/claude/chats/{id}/delete  delete it
        POST /bcnodes/claude/chats/{id}/send    one turn ({"text", "images"}): the turn's events as NDJSON, one per
                                                line, as they come (pipelines/claude/chat.send)

    A turn's request runs on a worker thread; when the browser goes away (Stop, a closed tab) the next write fails
    and the turn is cancelled at its next event. Only inside a running ComfyUI server; returns at once without one."""
    try:
        from server import PromptServer
    except ImportError:  # not under a ComfyUI server (tests, the import gate)
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    import asyncio
    import json
    import threading

    from aiohttp import web

    routes = server.routes

    async def answer(work, *args):
        from ..pipelines.claude import chat

        try:
            return web.json_response(await asyncio.get_running_loop().run_in_executor(None, work, chat, *args))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)

    @routes.get("/bcnodes/claude/chats")
    async def _list(request):
        return await answer(lambda chat: {"sessions": chat.list_sessions(), **chat.options()})

    @routes.post("/bcnodes/claude/chats")
    async def _create(request):
        return await answer(lambda chat, changes: chat.create(changes), await request.json())

    @routes.get("/bcnodes/claude/chats/{id}")
    async def _get(request):
        return await answer(lambda chat, sid: chat.load(sid), request.match_info["id"])

    @routes.post("/bcnodes/claude/chats/{id}")
    async def _update(request):
        return await answer(lambda chat, sid, changes: chat.update(sid, changes), request.match_info["id"],
                            await request.json())

    @routes.post("/bcnodes/claude/chats/{id}/delete")
    async def _delete(request):
        return await answer(lambda chat, sid: chat.delete(sid) or {}, request.match_info["id"])

    @routes.post("/bcnodes/claude/chats/{id}/send")
    async def _send(request):
        from ..pipelines.claude import chat

        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "application/x-ndjson"})
        await response.prepare(request)
        loop = asyncio.get_running_loop()
        events = asyncio.Queue()
        cancelled = threading.Event()

        def work():
            try:
                for event in chat.send(request.match_info["id"], body.get("text", ""), body.get("images", []),
                                       cancelled.is_set):
                    loop.call_soon_threadsafe(events.put_nowait, event)
            except Exception as error:  # an error before the request (no key, an empty message): shown, not raised
                loop.call_soon_threadsafe(events.put_nowait, {"type": "error", "error": str(error)})
            finally:
                loop.call_soon_threadsafe(events.put_nowait, None)

        worker = loop.run_in_executor(None, work)
        try:
            while (event := await events.get()) is not None:
                await response.write((json.dumps(event) + "\n").encode())
        except (ConnectionResetError, asyncio.CancelledError):
            cancelled.set()
            raise
        await worker
        await response.write_eof()
        return response


NODE_CLASS_MAPPINGS = {
    "BC_Claude": Claude,
    "BC_ClaudeConfig": ClaudeConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_Claude": "Claude",
    "BC_ClaudeConfig": "Claude Config",
}

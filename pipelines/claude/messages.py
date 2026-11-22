"""Claude: one Messages API request per run of the Claude node (nodes/claude.py), through Anthropic's official
Python SDK (`anthropic`, imported inside the functions), with the user's own API key.

The request holds only what the node gives it: the system prompt as written (no system field when it is empty),
one user turn of the images and the prompt, the model, max_tokens, effort, adaptive thinking with its summary
returned, and the Claude Config fields. It sends no tools, so the API builds no tool-use system prompt around the
user's one, and no sampling fields (every current model rejects them). The API is stateless: a run sees only its
own request. With two or more images each one is introduced by a text label ("Image 1:", "Image 2:", ...), as the
official vision guide advises; that label is the only text the node adds.

The answer streams (max_tokens up to 128000 needs it) and ComfyUI's interrupt is checked between the stream's
events: a cancel closes the connection. The API key is the one entered in ComfyUI's settings (web/js/claude.js),
kept by libs/download's token store under TOKEN_SERVICE.
"""

import base64
import dataclasses
import io
import logging
import math

# The node's model widget: the name shown -> the official model ID.
MODELS = {"Opus 5.5": "claude-opus-5-5", "Sonnet 5.5": "claude-sonnet-5-5", "Haiku 5.5": "claude-haiku-5-5",
          "Fable 5.2": "claude-fable-5-2"}
DEFAULT_MODEL = "Opus 5.5"
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
DEFAULT_EFFORT = "high"
MAX_TOKENS = 128000  # the output cap of every model in MODELS
DEFAULT_MAX_TOKENS = 32768
# Claude Haiku 5.5 has no server-side fallback; the official advice is to send it no `fallbacks`.
NO_FALLBACK = {"claude-haiku-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # the beta `fallbacks: "default"` needs
# The high-resolution tier (Claude 4.7 and later, every model in MODELS): the API downscales an image whose padded
# edges exceed 2576 px or whose 28 x 28 px patches exceed 4784; encode_image downscales it to that size first.
MAX_EDGE = 2576
MAX_VISUAL_TOKENS = 4784
PATCH = 28
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # one image's base64 data on the Claude API: at most 10 MB
TOKEN_SERVICE = "anthropic"  # the key's name in libs/download's token store
ON_REFUSAL = ["fallback", "error"]
SPEEDS = ["standard", "fast"]
IMAGE_FORMATS = ["png", "jpeg"]
JPEG_QUALITY = 95


@dataclasses.dataclass(frozen=True)
class ClaudeConfigValues:
    """Claude Config's fields, the BC_CLAUDE_CONFIG value. The defaults are a run with no Claude Config node."""
    model_id: str = ""            # a model ID that replaces the node's model; "" keeps the node's
    stop_sequences: tuple = ()    # strings that end the answer where they appear
    on_refusal: str = "fallback"  # fallback: the API's server-side fallback answers a declined request; error: the run fails
    speed: str = "standard"       # fast: the model's fast mode (Opus 5.5 / Fable 5.2, at a higher price)
    image_format: str = "png"     # png or jpeg (quality JPEG_QUALITY)
    timeout: int = 600            # seconds; the SDK's default
    max_retries: int = 2          # the SDK's default
    cache_system_prompt: bool = True  # a cache breakpoint on the system prompt (prompt caching)


@dataclasses.dataclass(frozen=True)
class ClaudeRequest:
    """One Claude node run. `images`: 8-bit PIL images in order, read once (the node hands a generator, so one frame
    is converted at a time)."""
    model: str          # a MODELS name
    max_tokens: int
    effort: str         # one of EFFORTS
    prompt: str
    system_prompt: str
    images: object
    config: ClaudeConfigValues


@dataclasses.dataclass(frozen=True)
class ClaudeResult:
    text: str      # the answer's text blocks, joined
    thinking: str  # the thinking summaries, joined by a blank line; empty when the model did not think


def resized_size(width, height, max_edge=MAX_EDGE, max_tokens=MAX_VISUAL_TOKENS):
    """The size the API downscales a `width` x `height` image to before padding: the largest aspect-preserving size
    whose edges, padded to multiples of 28, stay within `max_edge` and whose 28 x 28 patches stay within
    `max_tokens`. An image that fits comes back unchanged. The official reference implementation (Vision
    coordinates, "Resize your image before uploading"); round() rounds half to even, as the live API does."""
    def fits(w, h):
        return (math.ceil(w / PATCH) * PATCH <= max_edge and math.ceil(h / PATCH) * PATCH <= max_edge
                and math.ceil(w / PATCH) * math.ceil(h / PATCH) <= max_tokens)

    if fits(width, height):
        return (width, height)
    if height > width:
        resized_h, resized_w = resized_size(height, width, max_edge, max_tokens)
        return (resized_w, resized_h)
    # Binary search along the long edge for the largest aspect-preserving size that fits.
    aspect_ratio = width / height
    lo, hi = 1, width  # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect_ratio), 1))


def encode_image(image, image_format, number):
    """`image` (an 8-bit PIL image) as an image block's base64 source. An image the API would downscale is
    downscaled here first, to the same size (resized_size; Lanczos, as the API names no filter), so the upload
    carries no pixel the model never sees; then PNG, or JPEG at quality JPEG_QUALITY (no alpha). An encoding over the
    API's MAX_IMAGE_BYTES is an error naming `number`, the image's 1-based place."""
    from PIL import Image

    size = resized_size(*image.size)
    if size != image.size:
        image = image.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    if image_format == "jpeg":
        image.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    else:
        image.save(buf, format="PNG")
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Claude: image {number} is {len(data) / 2 ** 20:.1f} MB as base64 {image_format.upper()} at "
                         f"{size[0]}x{size[1]}; the API takes at most {MAX_IMAGE_BYTES // 2 ** 20} MB per image. Set "
                         "image_format to jpeg on a Claude Config node.")
    return {"type": "base64", "media_type": f"image/{image_format}", "data": data}


def user_content(prompt, sources):
    """The user turn's content: each image source (with an "Image N:" label before it when there are two or more),
    then the prompt as written. A whitespace-only prompt is left out (the API rejects an empty text block), which is
    an error with no image."""
    blocks = []
    for number, source in enumerate(sources, 1):
        if len(sources) > 1:
            blocks.append({"type": "text", "text": f"Image {number}:"})
        blocks.append({"type": "image", "source": source})
    if prompt.strip():
        blocks.append({"type": "text", "text": prompt})
    if not blocks:
        raise ValueError("Claude: the prompt is empty and no image is connected; write a prompt or connect an image.")
    return blocks


def request_params(request, sources):
    """The keyword arguments of the SDK's client.beta.messages.stream for `request`, its images encoded as
    `sources`. Adaptive thinking is the only thinking setting every model in MODELS takes; its summary is returned
    (display "summarized", billed as the default "omitted")."""
    config = request.config
    model = config.model_id or MODELS[request.model]
    params = {"model": model, "max_tokens": request.max_tokens,
              "messages": [{"role": "user", "content": user_content(request.prompt, sources)}],
              "thinking": {"type": "adaptive", "display": "summarized"},
              "output_config": {"effort": request.effort}}
    if request.system_prompt.strip() and config.cache_system_prompt:
        # The system prompt is the request's only part that repeats between runs (images and prompt change), so
        # the breakpoint goes on it: a later run within the cache's 5 minutes reads it at about a tenth of the
        # input price (the first writes it at 1.25x). Below the model's minimum cacheable length (512 tokens on
        # Opus 5.5) the API caches nothing and charges nothing extra.
        params["system"] = [{"type": "text", "text": request.system_prompt, "cache_control": {"type": "ephemeral"}}]
    elif request.system_prompt.strip():
        params["system"] = request.system_prompt
    if config.stop_sequences:
        params["stop_sequences"] = list(config.stop_sequences)
    if config.speed == "fast":
        params["speed"] = "fast"
    if config.on_refusal == "fallback" and model not in NO_FALLBACK:
        params["betas"] = [FALLBACK_BETA]
        params["fallbacks"] = "default"
    return params


def make_client(config):
    """The SDK client for one run: the API key from the token store (entered in ComfyUI's settings), the config's
    timeout and retries. The tests replace this function."""
    import anthropic

    from ...libs.download import read_tokens

    key = read_tokens().get(TOKEN_SERVICE)
    if not key:
        raise ValueError("Claude: no Anthropic API key is set. Enter one (a Claude Console API key) in ComfyUI's "
                         "Settings -> BCNodes -> Claude.")
    return anthropic.Anthropic(api_key=key, timeout=config.timeout, max_retries=config.max_retries)


def stream_message(client, params):
    """Streams the request; ComfyUI's interrupt between events ends the stream (its connection closed) and the run.
    -> (the final message, the request id)."""
    import comfy.model_management

    with client.beta.messages.stream(**params) as stream:
        for _event in stream:
            comfy.model_management.throw_exception_if_processing_interrupted()
        return stream.get_final_message(), stream.request_id


def api_error(error, timeout):
    """The run's error message for the SDK exception `error`: what failed and what to do."""
    import anthropic

    request = f" (request {error.request_id})" if getattr(error, "request_id", None) else ""
    if isinstance(error, anthropic.AuthenticationError):
        return (f"Claude: the API key was not accepted (401){request}. Enter a valid key in ComfyUI's Settings -> "
                "BCNodes -> Claude.")
    if isinstance(error, anthropic.APITimeoutError):
        return f"Claude: no answer within {timeout} s. Raise timeout on a Claude Config node."
    if isinstance(error, anthropic.APIConnectionError):
        return f"Claude: cannot reach the Anthropic API ({error}). Check the network."
    if isinstance(error, anthropic.APIStatusError):
        return f"Claude: the API answered {error.status_code}{request}: {error.api_message or error.message}"
    return f"Claude: {error}"


def read_answer(message, model):
    """The final `message` of a request for `model` as a ClaudeResult. A refusal (the whole fallback chain's, with
    fallbacks on) and an answer cut at max_tokens are errors: neither is a usable answer. An answer from another
    model (the server-side fallback) is logged."""
    if message.stop_reason == "refusal":
        details = message.stop_details
        category = f" ({details.category})" if details and details.category else ""
        explanation = f": {details.explanation}" if details and details.explanation else ""
        raise RuntimeError(f"Claude: {message.model} declined the request{category}{explanation}. Rephrase the "
                           "prompt or pick another model.")
    if message.stop_reason == "max_tokens":
        raise RuntimeError(f"Claude: the answer stopped at max_tokens ({message.usage.output_tokens} tokens, thinking "
                           "included) before it ended. Raise max_tokens or lower effort.")
    if message.model != model:
        logging.warning("[BCNodes] Claude: %s declined the request and %s answered it (the server-side fallback, "
                        "billed at its rates).", model, message.model)
    return ClaudeResult(text="".join(block.text for block in message.content if block.type == "text"),
                        thinking="\n\n".join(block.thinking for block in message.content
                                             if block.type == "thinking" and block.thinking))


def run(request):
    """One Claude run: the client (the key checked first), the images encoded, the request streamed, the answer
    read. -> ClaudeResult."""
    import anthropic

    client = make_client(request.config)
    sources = [encode_image(image, request.config.image_format, number)
               for number, image in enumerate(request.images, 1)]
    params = request_params(request, sources)
    try:
        message, request_id = stream_message(client, params)
    except anthropic.APIError as error:
        raise RuntimeError(api_error(error, request.config.timeout)) from error
    usage = message.usage
    logging.info("[BCNodes] Claude: %s, %d input (cache read %d, written %d) + %d output tokens, request %s",
                 message.model, usage.input_tokens, usage.cache_read_input_tokens or 0,
                 usage.cache_creation_input_tokens or 0, usage.output_tokens, request_id)
    return read_answer(message, params["model"])

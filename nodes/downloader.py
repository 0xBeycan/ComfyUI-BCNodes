"""Model downloader.

    BC_AutoModelDownloader (Auto Model Downloader)

Each line on the node is a model URL, a directory under ComfyUI/models and two
flags: "HF token needed" (gated Hugging Face repos) and "Civitai token needed".
Tokens are entered in the UI and kept server side
(user/BCNodes/downloader_tokens.json), never in the workflow. The node has two
halves:

  * the node itself — when the prompt runs it fetches whatever is missing, so
    a workflow queued through the API (no browser) still gets its models;
  * HTTP routes the frontend (web/js/auto_model_downloader.js) uses to check
    what is missing, start a download job, follow it and remember that the
    first-open prompt was already answered.

`server` and `aiohttp` are loaded long before custom nodes, so importing them
here costs nothing; the import is guarded so the module also loads outside
ComfyUI (tests), where the routes are simply not registered. `folder_paths`
is imported where it is used.
"""

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    from server import PromptServer  # first: missing outside ComfyUI, so aiohttp is never touched there
    from aiohttp import web
except ImportError:
    PromptServer = None
    web = None

EVENT = "bcnodes.downloader"
CHUNK = 1 << 20
USER_AGENT = "ComfyUI-BCNodes/2.0"
MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".pkl", ".msgpack", ".yaml", ".json")
# The only hosts a line may download from, and the service whose token they get.
# A shared workflow carries its URLs, so anything else - a lookalike domain, a
# machine on the local network - is refused.
HOSTS = {"huggingface.co": "huggingface", "civitai.com": "civitai"}


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------

def parse_entries(text):
    """The node stores its lines as a JSON list of {"url", "dir"}."""
    if not text or not str(text).strip():
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("entries must be a JSON list")
    return [e for e in data if isinstance(e, dict) and ((e.get("url") or "").strip() or (e.get("dir") or "").strip())]


def normalize_url(url):
    url = url.strip()
    # A Hugging Face "blob" link is the web page; "resolve" serves the file.
    return re.sub(r"^(https?://huggingface\.co/[^?#]+?)/blob/", r"\1/resolve/", url)


def filename_from_url(url):
    path = urllib.parse.urlparse(url).path
    return urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])


def resolve_entry(entry):
    """-> {"url", "dir", "filename", "path"} or raises ValueError with a message
    the user can act on."""
    url = normalize_url(entry.get("url") or "")
    target = (entry.get("dir") or "").strip().replace("\\", "/").strip("/")
    if not url:
        raise ValueError("model URL is empty")
    if not target:
        raise ValueError(f"directory is empty for {url}")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url}")
    service = service_of(url)
    if service is None:
        raise ValueError(f"only Hugging Face and Civitai URLs are allowed: {url}")

    import folder_paths

    # "dir/name.safetensors" renames the file; "dir" keeps the name from the URL.
    if target.lower().endswith(MODEL_EXTS):
        rel_dir, filename = os.path.split(target)
    else:
        rel_dir, filename = target, filename_from_url(url)
    if not filename.lower().endswith(MODEL_EXTS):
        raise ValueError(f"cannot tell the file name from {url}; write it after the directory, e.g. {target}/model.safetensors")
    # The URL name is decoded after the split, so "%2F" or "%5C" can smuggle a separator in.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"invalid file name {filename!r} from {url}")

    models_dir = os.path.abspath(folder_paths.models_dir)
    dest_dir = os.path.abspath(os.path.join(models_dir, rel_dir))
    if os.path.commonpath([dest_dir, models_dir]) != models_dir:
        raise ValueError(f"directory must stay under models/: {target}")
    # The final file must land in dest_dir itself. Compared against dest_dir's
    # realpath rather than models/, so a models/ subfolder symlinked to another volume
    # (common on pods) still works.
    real_dir = os.path.realpath(dest_dir)
    if os.path.commonpath([os.path.realpath(os.path.join(dest_dir, filename)), real_dir]) != real_dir:
        raise ValueError(f"file must stay under models/{rel_dir}: {filename}")
    legacy = bool(entry.get("token"))  # older lines carried one flag; the host said which token
    return {
        "url": url, "dir": rel_dir.replace(os.sep, "/"), "filename": filename, "path": os.path.join(dest_dir, filename),
        "service": service,
        "hf": bool(entry.get("hf")) or (legacy and service == "huggingface"),
        "civitai": bool(entry.get("civitai")) or (legacy and service == "civitai"),
    }


def service_of(url):
    """The HOSTS service for url's exact host or a subdomain of it, else None."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    for domain, service in HOSTS.items():
        if host == domain or host.endswith("." + domain):
            return service
    return None


def entries_key(items):
    pairs = sorted((i["url"], i["dir"], i["filename"]) for i in items)
    return hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:16]


def describe(entries):
    """Resolves every entry; bad ones come back with an "error" instead of a path."""
    items = []
    for entry in entries:
        try:
            item = resolve_entry(entry)
            item["exists"] = os.path.isfile(item["path"])
            item["size"] = os.path.getsize(item["path"]) if item["exists"] else 0
        except ValueError as e:
            item = {"url": (entry.get("url") or "").strip(), "dir": (entry.get("dir") or "").strip(), "filename": "", "path": "",
                    "exists": False, "size": 0, "service": None, "hf": bool(entry.get("hf")), "civitai": bool(entry.get("civitai")), "error": str(e)}
        items.append(item)
    return items


def missing_tokens(items, tokens=None):
    """{service: [filename, ...]} for files still to download whose line is
    marked as needing a token that is not stored."""
    tokens = _read_tokens() if tokens is None else tokens
    out = {}
    for i in items:
        if i.get("error") or i["exists"]:
            continue
        for service, flag in (("huggingface", "hf"), ("civitai", "civitai")):
            if i.get(flag) and not tokens.get(service):
                out.setdefault(service, []).append(i["filename"])
    return out


SERVICE_NAMES = {"huggingface": "Hugging Face", "civitai": "Civitai"}


def missing_tokens_message(missing):
    return "; ".join(f"{SERVICE_NAMES[s]} token required for {', '.join(files)}" for s, files in missing.items())


# ---------------------------------------------------------------------------
# "Already asked" marker — kept server side so it survives a changing URL
# (RunPod proxies) and is shared by every browser that opens this ComfyUI.
# ---------------------------------------------------------------------------

def _user_file(name):
    import folder_paths

    return os.path.join(folder_paths.get_user_directory(), "BCNodes", name)


def _seen_file():
    return _user_file("downloader_seen.json")


def _tokens_file():
    return _user_file("downloader_tokens.json")


def _read_tokens():
    try:
        with open(_tokens_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in SERVICE_NAMES and isinstance(v, str) and v} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_tokens(update):
    """Merges {service: token}; an empty string removes that token."""
    tokens = _read_tokens()
    for service, value in update.items():
        if service not in SERVICE_NAMES:
            continue
        value = (value or "").strip()
        if value:
            tokens[service] = value
        else:
            tokens.pop(service, None)
    path = _tokens_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokens, f)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return tokens


def tokens_present():
    tokens = _read_tokens()
    return {s: bool(tokens.get(s)) for s in SERVICE_NAMES}


def _read_seen():
    try:
        with open(_seen_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def mark_seen(key):
    seen = _read_seen()
    seen.add(key)
    path = _seen_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _auth(url):
    headers = {"User-Agent": USER_AGENT}
    tokens = _read_tokens()
    service = service_of(url)
    token = tokens.get(service) if service else None
    if token and service == "huggingface":
        headers["Authorization"] = f"Bearer {token}"
    if token and service == "civitai" and "token=" not in url:
        url += ("&" if "?" in url else "?") + "token=" + urllib.parse.quote(token)
    return url, headers


def download_file(url, path, on_progress=None):
    """Streams url into path via path + ".part", resuming a previous partial
    download when the server supports ranges. on_progress(downloaded, total)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    url, headers = _auth(url)

    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 416 and have:  # the .part is already complete
            os.replace(part, path)
            return
        hint = ""
        if e.code in (401, 403):
            service = service_of(url)
            hint = f" ({SERVICE_NAMES[service]} token missing or not accepted)" if service else ""
        raise RuntimeError(f"HTTP {e.code} for {url}{hint}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"cannot reach {url}: {e.reason}") from e

    with resp:
        if have and resp.status != 206:
            have = 0  # server ignored the range; start over
        ctype = resp.headers.get("Content-Type", "")
        if ctype.startswith("text/html"):
            raise RuntimeError(f"{url} returned a web page instead of a file (login or consent required?)")
        length = resp.headers.get("Content-Length")
        total = (int(length) + have) if length else 0
        downloaded = have
        last = 0.0
        with open(part, "ab" if have else "wb") as f:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if on_progress and (now - last >= 0.5 or downloaded == total):
                    last = now
                    on_progress(downloaded, total)
    if total and downloaded < total:
        raise RuntimeError(f"connection closed early for {url}: {downloaded}/{total} bytes (run again to resume)")
    os.replace(part, path)
    if on_progress:
        on_progress(downloaded, total or downloaded)


# ---------------------------------------------------------------------------
# Background job driven from the frontend
# ---------------------------------------------------------------------------

class _Job:
    lock = threading.Lock()
    current = None  # {"id", "items", "index", "running", "errors", "progress"}

    @classmethod
    def snapshot(cls):
        job = cls.current
        if job is None:
            return {"running": False}
        return {
            "running": job["running"], "id": job["id"], "index": job["index"], "count": len(job["items"]),
            "files": [i["filename"] for i in job["items"]], "errors": job["errors"], "progress": job["progress"],
        }


def _send(data):
    instance = getattr(PromptServer, "instance", None)
    if instance is not None:
        instance.send_sync(EVENT, data)


def _run_job(job):
    items = job["items"]
    for index, item in enumerate(items):
        job["index"] = index
        base = {"job": job["id"], "index": index, "count": len(items), "filename": item["filename"], "dir": item["dir"]}

        def on_progress(downloaded, total, base=base):
            job["progress"] = {"downloaded": downloaded, "total": total, **base}
            _send({"event": "progress", "downloaded": downloaded, "total": total, **base})

        try:
            print(f"[BCNodes] downloading {item['url']} -> models/{item['dir']}/{item['filename']}")
            download_file(item["url"], item["path"], on_progress)
            _send({"event": "file_done", **base})
        except Exception as e:  # keep going with the next file, report at the end
            job["errors"].append({"filename": item["filename"], "error": str(e)})
            print(f"[BCNodes] failed {item['filename']}: {e}")
            _send({"event": "file_error", "error": str(e), **base})
    job["running"] = False
    _send({"event": "done", "job": job["id"], "errors": job["errors"], "count": len(items)})


def start_job(items):
    with _Job.lock:
        if _Job.current is not None and _Job.current["running"]:
            return _Job.current, False
        job = {"id": hashlib.sha1(f"{time.time()}".encode()).hexdigest()[:8], "items": items, "index": 0,
               "running": True, "errors": [], "progress": None}
        _Job.current = job
    threading.Thread(target=_run_job, args=(job,), name="bcnodes-downloader", daemon=True).start()
    return job, True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

if getattr(PromptServer, "instance", None) is not None:
    routes = PromptServer.instance.routes

    @routes.post("/bcnodes/downloader/check")
    async def _check(request):
        body = await request.json()
        items = describe(body.get("entries") or [])
        valid = [i for i in items if not i.get("error")]
        key = entries_key(valid)
        return web.json_response({
            "items": items, "key": key, "seen": key in _read_seen(),
            "missing": sum(1 for i in valid if not i["exists"]), "errors": sum(1 for i in items if i.get("error")),
            "tokens": tokens_present(), "missing_tokens": missing_tokens(valid),
            "job": _Job.snapshot(),
        })

    @routes.get("/bcnodes/downloader/tokens")
    async def _tokens_get(request):
        return web.json_response({"tokens": tokens_present()})

    @routes.post("/bcnodes/downloader/tokens")
    async def _tokens_set(request):
        body = await request.json()
        save_tokens({k: body.get(k) for k in SERVICE_NAMES if k in body})
        return web.json_response({"tokens": tokens_present()})

    @routes.post("/bcnodes/downloader/dismiss")
    async def _dismiss(request):
        body = await request.json()
        key = body.get("key")
        if key:
            mark_seen(key)
        return web.json_response({"ok": True})

    @routes.post("/bcnodes/downloader/start")
    async def _start(request):
        body = await request.json()
        valid = [i for i in describe(body.get("entries") or []) if not i.get("error")]
        missing = [i for i in valid if not i["exists"]]
        if not missing:
            return web.json_response({"started": False, "reason": "nothing to download", "job": _Job.snapshot()})
        needed = missing_tokens(valid)
        if needed:
            return web.json_response({"started": False, "error": missing_tokens_message(needed), "missing_tokens": needed}, status=400)
        mark_seen(entries_key(valid))  # the user chose to download; no need to ask again
        job, started = start_job(missing)
        return web.json_response({"started": started, "job": _Job.snapshot()})

    @routes.get("/bcnodes/downloader/status")
    async def _status(request):
        return web.json_response({"job": _Job.snapshot()})


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class AutoModelDownloader:
    """Lists the models a workflow needs; when queued it downloads the ones that
    are missing. The lines are edited on the node (web/js/auto_model_downloader.js)
    and stored in `entries` as JSON."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"entries": ("STRING", {"default": "[]", "multiline": False})}}

    RETURN_TYPES = ()
    FUNCTION = "download"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/loaders"
    SEARCH_ALIASES = ['BCNodes', 'auto model downloader', 'download models', 'model downloader']

    @classmethod
    def IS_CHANGED(cls, entries):
        return float("nan")  # always check: a file may have been deleted meanwhile

    def download(self, entries):
        from comfy.utils import ProgressBar

        items = describe(parse_entries(entries))
        bad = [i for i in items if i.get("error")]
        if bad:
            raise ValueError("; ".join(i["error"] for i in bad))
        missing = [i for i in items if not i["exists"]]
        needed = missing_tokens(items)
        if needed:
            raise ValueError(missing_tokens_message(needed) + " — enter it on the node")
        if items:
            mark_seen(entries_key(items))

        pbar = ProgressBar(max(len(missing), 1))
        for item in missing:
            state = {"pct": -1}

            def on_progress(downloaded, total, item=item, state=state):
                pct = int(100 * downloaded / total) if total else 0
                if pct // 10 != state["pct"] // 10:
                    state["pct"] = pct
                    print(f"[BCNodes] {item['filename']} {pct}% ({downloaded / 1e6:.0f}/{total / 1e6:.0f} MB)")

            print(f"[BCNodes] downloading {item['url']} -> models/{item['dir']}/{item['filename']}")
            download_file(item["url"], item["path"], on_progress)
            pbar.update(1)
        if not missing:
            pbar.update(1)

        text = f"{len(items) - len(missing)}/{len(items)} present, {len(missing)} downloaded" if items else "no models listed"
        return {"ui": {"text": [text]}}


NODE_CLASS_MAPPINGS = {
    "BC_AutoModelDownloader": AutoModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_AutoModelDownloader": "Auto Model Downloader",
}

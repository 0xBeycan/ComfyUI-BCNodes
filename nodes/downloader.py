"""Model downloader.

    BC_AutoModelDownloader (Auto Model Downloader)

Each line on the node is a model URL, a directory under ComfyUI/models and two
flags: "HF token needed" (gated Hugging Face repos) and "Civitai token needed".
The node has two halves:

  * the node itself — when the prompt runs it fetches whatever is missing, so
    a workflow queued through the API (no browser) still gets its models;
  * HTTP routes the frontend (web/js/auto_model_downloader.js) uses to check
    what is missing, start a download job, follow it and remember that the
    first-open prompt was already answered.

`server` and `aiohttp` are loaded long before custom nodes, so importing them
here costs nothing; the import is guarded so the module also loads outside
ComfyUI (tests), where the routes are simply not registered. `folder_paths`
is imported inside libs/download.py and pipelines/model_download.py, where it
is used.
"""

import hashlib
import threading
import time

try:
    from server import PromptServer  # first: missing outside ComfyUI, so aiohttp is never touched there
    from aiohttp import web
except ImportError:
    PromptServer = None
    web = None

from ..libs.download import SERVICE_NAMES, download_file, save_tokens, tokens_present
from ..pipelines.model_download import (
    describe, entries_key, mark_seen, missing_tokens, missing_tokens_message, parse_entries, read_seen,
)

EVENT = "bcnodes.downloader"


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


def _download_item(item, on_progress):
    """One resolved item, announced on the console first (the job runner and the node)."""
    print(f"[BCNodes] downloading {item['url']} -> models/{item['dir']}/{item['filename']}")
    download_file(item["url"], item["path"], on_progress)


def _run_job(job):
    items = job["items"]
    for index, item in enumerate(items):
        job["index"] = index
        base = {"job": job["id"], "index": index, "count": len(items), "filename": item["filename"], "dir": item["dir"]}

        def on_progress(downloaded, total, base=base):
            job["progress"] = {"downloaded": downloaded, "total": total, **base}
            _send({"event": "progress", "downloaded": downloaded, "total": total, **base})

        try:
            _download_item(item, on_progress)
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
            "items": items, "key": key, "seen": key in read_seen(),
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

            _download_item(item, on_progress)
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

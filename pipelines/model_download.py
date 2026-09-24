"""The Auto Model Downloader's entries: each line resolved to one file under
ComfyUI/models, the token gate and the "already asked" marker. The transport and the
token store are libs/download.py."""

import hashlib
import json
import os
import re
import urllib.parse
from typing import List, TypedDict, Union

from ..libs.download import SERVICE_NAMES, read_tokens, service_of, user_file

MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx", ".pkl", ".msgpack", ".yaml", ".json")


class ResolvedEntry(TypedDict):
    """resolve_entry's dict, keys in the order it builds them. A dict, not a dataclass:
    /bcnodes/downloader/check serialises it as it is and tests index it."""
    url: str
    dir: str
    filename: str
    path: str
    service: str
    hf: bool
    civitai: bool


class DescribedItem(ResolvedEntry):
    """describe's item for an entry that resolved: ResolvedEntry, then these two keys."""
    exists: bool
    size: int


class DescribedError(TypedDict):
    """describe's item for an entry that did not resolve, keys in the order it builds them."""
    url: str
    dir: str
    filename: str
    path: str
    exists: bool
    size: int
    service: None
    hf: bool
    civitai: bool
    error: str


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


def resolve_entry(entry) -> ResolvedEntry:
    """-> ResolvedEntry (url, dir, filename, path, service, hf, civitai), or raises
    ValueError with a message the user can act on."""
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


def entries_key(items):
    pairs = sorted((i["url"], i["dir"], i["filename"]) for i in items)
    return hashlib.sha1(json.dumps(pairs).encode()).hexdigest()[:16]


def describe(entries) -> List[Union[DescribedItem, DescribedError]]:
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
    tokens = read_tokens() if tokens is None else tokens
    out = {}
    for i in items:
        if i.get("error") or i["exists"]:
            continue
        for service, flag in (("huggingface", "hf"), ("civitai", "civitai")):
            if i.get(flag) and not tokens.get(service):
                out.setdefault(service, []).append(i["filename"])
    return out


def missing_tokens_message(missing):
    return "; ".join(f"{SERVICE_NAMES[s]} token required for {', '.join(files)}" for s, files in missing.items())


# ---------------------------------------------------------------------------
# "Already asked" marker — kept server side so it survives a changing URL
# (RunPod proxies) and is shared by every browser that opens this ComfyUI.
# ---------------------------------------------------------------------------

def seen_file():
    return user_file("downloader_seen.json")


def read_seen():
    try:
        with open(seen_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def mark_seen(key):
    seen = read_seen()
    seen.add(key)
    path = seen_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)

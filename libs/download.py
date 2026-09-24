"""HTTP download with resume, the allowed hosts and the token store.

Tokens are entered in the UI and kept server side
(user/BCNodes/downloader_tokens.json), never in the workflow.
"""

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 1 << 20
USER_AGENT = "ComfyUI-BCNodes/2.0"
# The only hosts a line may download from, and the service whose token they get.
# A shared workflow carries its URLs, so anything else - a lookalike domain, a
# machine on the local network - is refused.
HOSTS = {"huggingface.co": "huggingface", "civitai.com": "civitai"}


def service_of(url):
    """The HOSTS service for url's exact host or a subdomain of it, else None."""
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    for domain, service in HOSTS.items():
        if host == domain or host.endswith("." + domain):
            return service
    return None


SERVICE_NAMES = {"huggingface": "Hugging Face", "civitai": "Civitai"}


def user_file(name):
    import folder_paths

    return os.path.join(folder_paths.get_user_directory(), "BCNodes", name)


def tokens_file():
    return user_file("downloader_tokens.json")


def read_tokens():
    try:
        with open(tokens_file(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in SERVICE_NAMES and isinstance(v, str) and v} if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_tokens(update):
    """Merges {service: token}; an empty string removes that token."""
    tokens = read_tokens()
    for service, value in update.items():
        if service not in SERVICE_NAMES:
            continue
        value = (value or "").strip()
        if value:
            tokens[service] = value
        else:
            tokens.pop(service, None)
    path = tokens_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tokens, f)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return tokens


def tokens_present():
    tokens = read_tokens()
    return {s: bool(tokens.get(s)) for s in SERVICE_NAMES}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _auth(url):
    headers = {"User-Agent": USER_AGENT}
    tokens = read_tokens()
    service = service_of(url)
    token = tokens.get(service) if service else None
    if token and (service == "huggingface" or (service == "civitai" and "token=" not in url)) and url[:7].lower() == "http://":
        url = "https://" + url[7:]  # a stored token is sent over TLS only: same host, path and query
    if token and service == "huggingface":
        headers["Authorization"] = f"Bearer {token}"
    if token and service == "civitai" and "token=" not in url:
        url += ("&" if "?" in url else "?") + "token=" + urllib.parse.quote(token)
    return url, headers


class _StripAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """urllib copies every request header to a redirect target, whatever its host.
    The token stays with the host it was sent to; huggingface.co hands file
    bodies to a CDN on another host, which does not need it (signed URL)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).hostname != urllib.parse.urlparse(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def download_file(url, path, on_progress=None):
    """Streams url into path via path + ".part", resuming a previous partial
    download when the server supports ranges. on_progress(downloaded, total)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    request_url, headers = _auth(url)  # url itself stays token-free: it goes into every message below

    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have:
        headers["Range"] = f"bytes={have}-"

    req = urllib.request.Request(request_url, headers=headers)
    try:
        resp = urllib.request.build_opener(_StripAuthOnHostChange).open(req, timeout=60)
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

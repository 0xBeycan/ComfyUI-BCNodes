"""PreFlight's feedback store: an append-only JSONL log of predictions and outcomes.

Standard library only: other tools may read and write this store directly, and the store format lives in
exactly one place, here.

Two record types share an ``id`` and are joined at read time:

    {"type": "prediction", "id": "a3f9c2d1", "ts": ..., "engine_version": ...,
     "schema_version": ..., "observations": {...}, "caption_excerpt": ...,
     "caption_flags": [...], "fired_rules": [...], "verdicts": {...}}
    {"type": "outcome", "id": "a3f9c2d1", "ts": ..., "platform": "instagram",
     "result": "demoted"}

The store is append-only and never updated in place: outcomes arrive days later, possibly from a different
process, so there is no locking and no read-modify-write. Conflicts (several outcomes for one
``(id, platform)``) are resolved at read time: the last write wins.

Every function takes the store's ``path``; the node layer resolves it
(<ComfyUI user dir>/BCNodes/preflight/feedback.jsonl) and passes it in.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

PLATFORMS = ("instagram", "tiktok", "x")
RESULTS = ("clean", "demoted", "removed")


def _now():
    """ISO-8601 UTC timestamp, second precision, e.g. 2026-07-24T06:02:11Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append(path, record):
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_prediction(report, caption="", *, path):
    """Append one prediction record built from a ``rules.judge`` report.

    Generates and returns the 8-char record id (the join key). The report's own ``record_id`` field is set
    separately by the Report flow after this returns.
    """
    record_id = uuid.uuid4().hex[:8]
    observations = report.get("observations", {}) or {}
    schema_version = ""
    meta = observations.get("meta") if isinstance(observations, dict) else None
    if isinstance(meta, dict):
        schema_version = meta.get("schema_version", "")

    _append(path, {
        "type": "prediction",
        "id": record_id,
        "ts": _now(),
        "engine_version": report.get("engine_version", ""),
        "schema_version": schema_version,
        "observations": observations,
        "caption_excerpt": (caption or "")[:80],
        "caption_flags": report.get("caption_flags", []),
        "fired_rules": report.get("fired_rules", []),
        "verdicts": report.get("verdicts", {}),
    })
    return record_id


def log_outcome(record_id, platform, result, *, path):
    """Append one outcome record. Raises ValueError on an unknown platform / result so bad data never
    enters the store; the Outcome flow reports the error as its status string rather than raising into the
    graph."""
    if platform not in PLATFORMS:
        raise ValueError("platform must be one of %s, got %r" % (PLATFORMS, platform))
    if result not in RESULTS:
        raise ValueError("result must be one of %s, got %r" % (RESULTS, result))
    if not record_id:
        raise ValueError("record_id is required")
    _append(path, {
        "type": "outcome",
        "id": record_id,
        "ts": _now(),
        "platform": platform,
        "result": result,
    })


def _read_records(path):
    """Yield parsed records, skipping blank or corrupt lines (the store may be appended to concurrently and
    must never fail a whole read on one bad line)."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                continue


def load_joined(path):
    """Return predictions (in first-seen order) with their outcomes attached.

    Each returned dict is the prediction record plus ``"outcomes"``: a ``{platform: result}`` mapping
    resolved last-write-wins.
    """
    predictions = {}
    order = []
    outcomes = {}  # id -> {platform: result}, last write wins
    for rec in _read_records(path):
        if not isinstance(rec, dict):
            continue
        rtype = rec.get("type")
        rid = rec.get("id")
        if not rid:
            continue
        if rtype == "prediction":
            if rid not in predictions:
                order.append(rid)
            predictions[rid] = rec
        elif rtype == "outcome":
            plat = rec.get("platform")
            result = rec.get("result")
            if plat in PLATFORMS and result in RESULTS:
                outcomes.setdefault(rid, {})[plat] = result  # later line overwrites

    joined = []
    for rid in order:
        pred = dict(predictions[rid])
        pred["outcomes"] = outcomes.get(rid, {})
        joined.append(pred)
    return joined


def recent_predictions(path, limit=50):
    """Predictions newest-first, capped at ``limit``: the Outcome node's record list."""
    joined = load_joined(path)
    joined.reverse()
    return joined[:limit]


def store_stamp(path):
    """The store's change stamp, "<mtime_ns>:<size>"; "" while there is no store. Every append changes it."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return ""
    return f"{st.st_mtime_ns}:{st.st_size}"

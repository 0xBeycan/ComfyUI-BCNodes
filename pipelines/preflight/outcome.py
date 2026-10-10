"""PreFlight Outcome's flow and its record list.

The record list: one label per prediction, newest first, built entirely from the record itself (no file
name convention is needed to find a prediction later); the record id is the label's first token. The node
lists it in its `record` combo, and the records route and event hand it to the frontend.

log(record, platform, result, record_id_override, store): one outcome line appended to the store for the
id of `record_id_override` when set, else of the `record` label; the status string says what was logged,
or why nothing was. Errors come back in the string, never raised into the graph.
"""

import logging
from typing import TypedDict

from . import feedback

NO_RECORDS = "no records yet"  # the list's only entry while the store holds no prediction
RECORD_LIMIT = 50


class RecordsPayload(TypedDict):
    """GET /bcnodes/preflight/records' body and the bcnodes.preflight.records event's data, read by
    web/js/preflight.js."""
    records: list[str]  # the record labels, newest first; [NO_RECORDS] when the store holds none


def format_record_label(rec):
    """One combo line per prediction: id · MM-DD HH:MM (UTC) · worst verdict per platform · garment/setting ·
    the caption's start."""
    rid = rec.get("id", "????????")
    ts = rec.get("ts", "")
    when = ts[5:16].replace("T", " ") if len(ts) >= 16 else ts  # MM-DD HH:MM
    verdicts = rec.get("verdicts", {})
    worst = " ".join("%s:%s" % (short, verdicts.get(p, {}).get("worst", "?"))
                     for p, short in (("instagram", "IG"), ("tiktok", "TT"), ("x", "X")))
    obs = rec.get("observations", {}) or {}
    what = "%s/%s" % (obs.get("garment", "?"), obs.get("setting", "?"))
    excerpt = (rec.get("caption_excerpt", "") or "").replace("\n", " ")
    if len(excerpt) > 18:
        excerpt = excerpt[:17] + "…"
    return "%s · %s · %s · %s · \"%s\"" % (rid, when, worst, what, excerpt)


def id_from_label(label):
    """The record id (first token) of a combo label; "" for NO_RECORDS or an empty label."""
    if not label or label == NO_RECORDS:
        return ""
    return label.split(" ", 1)[0].strip()


def record_labels(store, limit=RECORD_LIMIT):
    """The record list: the newest `limit` predictions of the store as labels, newest first; [NO_RECORDS]
    when there is none, or when the store cannot be read (logged)."""
    try:
        labels = [format_record_label(r) for r in feedback.recent_predictions(store, limit=limit)]
    except Exception as exc:  # noqa: BLE001  the node definition must load whatever the store holds
        logging.warning("[BCNodes] PreFlight Outcome: the feedback store %s cannot be read (%s); the record list "
                        "is empty", store, exc)
        labels = []
    return labels or [NO_RECORDS]


def log(record, platform, result, record_id_override, store):
    """The status string of one outcome logged (module docstring)."""
    record_id = record_id_override.strip() or id_from_label(record)
    if not record_id:
        return "no record selected — pick one from the list or set record_id_override"
    try:
        feedback.log_outcome(record_id, platform, result, path=store)
        return "logged: %s %s=%s" % (record_id, platform, result)
    except Exception as exc:  # noqa: BLE001  never raise into the graph
        return "error: %s" % exc

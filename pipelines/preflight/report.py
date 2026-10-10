"""PreFlight Report's flow: an observation (PreFlight Observe's JSON) and the caption through the rules engine,
the prediction logged to the feedback store.

report(observations_json, caption, log_prediction, store):
  1. the JSON parsed; text that is not JSON, or JSON that is not an object, becomes an {"error": ...}
     observation, which the rules fail closed on (UNKNOWN);
  2. motion rules count only for video: meta.frames_analyzed above 1;
  3. rules.judge over the observation and the caption (scanned together with the in-image text);
  4. log_prediction on: one prediction line appended to the store, its id written into the report before
     the summary is built; a store that cannot be written logs a warning and leaves the id empty, the
     verdicts still come out.
"""

import json
import logging

from . import feedback, rules


def report(observations_json, caption, log_prediction, store):
    """(report_json, summary, record_id); record_id is "" when logging is off or failed."""
    try:
        observations = json.loads(observations_json)
    except (ValueError, TypeError):
        observations = {"error": "observations_json was not valid JSON"}
    if not isinstance(observations, dict):
        observations = {"error": "observations_json was not a JSON object"}

    meta = observations.get("meta")
    frames = meta.get("frames_analyzed", 1) if isinstance(meta, dict) else 1
    is_video = isinstance(frames, int) and frames > 1

    result = rules.judge(observations, caption or "", is_video=is_video)

    record_id = ""
    if log_prediction:
        try:
            record_id = feedback.log_prediction(result, caption or "", path=store)
            result["record_id"] = record_id  # injected before the summary is built
        except Exception as exc:  # noqa: BLE001  a store failure must not break the graph
            logging.warning("[BCNodes] PreFlight Report: the feedback store write failed (%s); this prediction is "
                            "not logged", exc)
            record_id = ""

    return json.dumps(result, ensure_ascii=False, indent=2), rules.summary_text(result), record_id

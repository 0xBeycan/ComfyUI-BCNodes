"""PreFlight's append-only feedback store (pipelines/preflight/feedback.py) and the calibration tables
(pipelines/preflight/calibrate.py), no ComfyUI. Each test writes to a fresh ``tmp_path`` store.
"""

import json
import types

import pytest


@pytest.fixture
def pf(bcnodes):
    return types.SimpleNamespace(**{name: bcnodes[f"pipelines.preflight.{name}"] for name in ("calibrate", "feedback", "rules")})


def _obs(**kw):
    base = dict(
        subject_appears_under_18=False, garment="regular", setting="studio",
        framing="full_body", pose="neutral", see_through_or_wet=False,
        exposure="none", nudity_or_sexual_act=False, motion_flags="none",
        visible_text="", confidence="high", meta={"schema_version": "1"},
    )
    base.update(kw)
    return base


def _report(pf, **kw):
    return pf.rules.judge(_obs(**kw))


# --- round trip ---------------------------------------------------------------

def test_log_prediction_then_outcome_joins(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = pf.feedback.log_prediction(_report(pf, garment="bikini", setting="beach_pool",
                                          exposure="mild"),
                                  caption="summer drop", path=store)
    assert len(rid) == 8
    pf.feedback.log_outcome(rid, "instagram", "demoted", path=store)

    joined = pf.feedback.load_joined(store)
    assert len(joined) == 1
    assert joined[0]["id"] == rid
    assert joined[0]["outcomes"] == {"instagram": "demoted"}
    assert joined[0]["caption_excerpt"] == "summer drop"
    assert "base.bikini" in joined[0]["fired_rules"]


def test_prediction_record_shape(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    pf.feedback.log_prediction(_report(pf, garment="lingerie", exposure="significant"),
                            caption="x", path=store)
    line = store.read_text().strip()
    rec = json.loads(line)
    assert rec["type"] == "prediction"
    assert rec["engine_version"] == pf.rules.ENGINE_VERSION
    assert rec["schema_version"] == "1"
    assert rec["ts"].endswith("Z")
    assert set(rec) >= {"type", "id", "ts", "engine_version", "schema_version",
                        "observations", "caption_excerpt", "caption_flags",
                        "fired_rules", "verdicts"}


def test_caption_excerpt_truncated_to_80(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    long_caption = "z" * 200
    pf.feedback.log_prediction(_report(pf), caption=long_caption, path=store)
    rec = json.loads(store.read_text().strip())
    assert rec["caption_excerpt"] == "z" * 80


# --- append-only semantics ----------------------------------------------------

def test_last_write_wins_per_id_platform(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = pf.feedback.log_prediction(_report(pf), path=store)
    pf.feedback.log_outcome(rid, "instagram", "clean", path=store)
    pf.feedback.log_outcome(rid, "instagram", "removed", path=store)   # later wins
    joined = pf.feedback.load_joined(store)
    assert joined[0]["outcomes"]["instagram"] == "removed"
    # every write is a distinct line (nothing updated in place)
    assert len(store.read_text().strip().splitlines()) == 3


def test_multiple_platforms_attach_independently(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = pf.feedback.log_prediction(_report(pf), path=store)
    pf.feedback.log_outcome(rid, "instagram", "demoted", path=store)
    pf.feedback.log_outcome(rid, "tiktok", "removed", path=store)
    outcomes = pf.feedback.load_joined(store)[0]["outcomes"]
    assert outcomes == {"instagram": "demoted", "tiktok": "removed"}


def test_predictions_preserve_first_seen_order(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    ids = [pf.feedback.log_prediction(_report(pf), path=store) for _ in range(3)]
    assert [p["id"] for p in pf.feedback.load_joined(store)] == ids


# --- recent_predictions (Outcome combo source) --------------------------------

def test_recent_predictions_newest_first_and_limited(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    ids = [pf.feedback.log_prediction(_report(pf), path=store) for _ in range(5)]
    recent = pf.feedback.recent_predictions(store, limit=3)
    assert [p["id"] for p in recent] == list(reversed(ids))[:3]


# --- robustness ---------------------------------------------------------------

def test_missing_store_reads_empty(pf, tmp_path):
    assert pf.feedback.load_joined(tmp_path / "nope.jsonl") == []
    assert pf.feedback.recent_predictions(tmp_path / "nope.jsonl") == []


def test_corrupt_line_is_skipped(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = pf.feedback.log_prediction(_report(pf), path=store)
    with store.open("a") as fh:
        fh.write("this is not json\n")
    assert len(pf.feedback.load_joined(store)) == 1
    assert pf.feedback.load_joined(store)[0]["id"] == rid


@pytest.mark.parametrize("platform,result", [
    ("myspace", "clean"),
    ("instagram", "shadowbanned"),
])
def test_log_outcome_rejects_bad_values(pf, tmp_path, platform, result):
    store = tmp_path / "fb.jsonl"
    with pytest.raises(ValueError):
        pf.feedback.log_outcome("abcd1234", platform, result, path=store)


# --- calibration report -------------------------------------------------------

def test_calibrate_report_counts_and_rules(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    # Bikini-at-beach predicted RISK on IG; three came back clean -> over-predicted.
    for _ in range(3):
        rid = pf.feedback.log_prediction(
            _report(pf, garment="bikini", setting="beach_pool", exposure="mild"), path=store)
        pf.feedback.log_outcome(rid, "instagram", "clean", path=store)
    # A prediction with no outcome must be ignored by calibration (costs nothing).
    pf.feedback.log_prediction(_report(pf, nudity_or_sexual_act=True, exposure="significant"),
                            path=store)

    report = pf.calibrate.build_report(pf.feedback.load_joined(store))
    assert "Per-platform calibration" in report
    assert "Per-rule outcomes" in report
    assert "base.bikini" in report
    assert "predictions: 4" in report and "with at least one outcome: 3" in report


def test_calibrate_empty_store(pf, tmp_path):
    report = pf.calibrate.build_report(pf.feedback.load_joined(tmp_path / "empty.jsonl"))
    assert "Store is empty" in report


# --- the change stamp (PreFlight Calibrate's IS_CHANGED) ------------------------------------------------------


def test_store_stamp_changes_with_every_append(pf, tmp_path):
    store = tmp_path / "fb.jsonl"
    assert pf.feedback.store_stamp(store) == ""
    rid = pf.feedback.log_prediction(_report(pf), path=store)
    first = pf.feedback.store_stamp(store)
    assert first and first == pf.feedback.store_stamp(store)       # unchanged store: the same stamp
    pf.feedback.log_outcome(rid, "x", "clean", path=store)
    assert pf.feedback.store_stamp(store) not in ("", first)


def test_every_store_function_takes_the_path(pf):
    """No default store: a call without the path is an error, never a file written in the working directory."""
    with pytest.raises(TypeError):
        pf.feedback.log_prediction(_report(pf))
    with pytest.raises(TypeError):
        pf.feedback.log_outcome("abcd1234", "x", "clean")
    with pytest.raises(TypeError):
        pf.feedback.load_joined()


def test_calibrate_tables_exactly(pf, tmp_path):
    """The two tables, column for column: one prediction per row kind (scored inside, above and below its
    range; a fail-closed one, skipped by the platform table but counted per rule; one with no outcome)."""
    store = tmp_path / "fb.jsonl"
    log = pf.feedback.log_prediction
    r1 = log(_report(pf, garment="bikini", setting="beach_pool", exposure="mild"), path=store)
    pf.feedback.log_outcome(r1, "instagram", "clean", path=store)
    pf.feedback.log_outcome(r1, "tiktok", "removed", path=store)
    r2 = log(_report(pf, garment="lingerie", exposure="significant"), path=store)
    pf.feedback.log_outcome(r2, "x", "demoted", path=store)
    r3 = log(pf.rules.judge({"error": "x"}), path=store)
    pf.feedback.log_outcome(r3, "instagram", "clean", path=store)
    log(_report(pf), path=store)
    assert pf.calibrate.build_report(pf.feedback.load_joined(store)) == "\n".join([
        "PreFlight calibration report",
        "============================",
        "predictions: 4   with at least one outcome: 3",
        f"engine_versions seen: {pf.rules.ENGINE_VERSION} (4)",
        "",
        "== Per-platform calibration ==",
        "(outcome vs predicted [best,worst] range; UNKNOWN verdicts skipped)",
        "",
        "platform     scored  consistent  under-predicted over-predicted",
        "---------------------------------------------------------------",
        "IG                1           0                0              1",
        "TT                1           0                1              0",
        "X                 1           1                0              0",
        "",
        "== Per-rule outcomes ==",
        "(clean/demoted/removed counts among scored posts where the rule fired)",
        "",
        "rule                    fires   IG c/d/r    TT c/d/r    X c/d/r     ",
        "--------------------------------------------------------------------",
        "base.bikini                 1   1/0/0       0/0/1       0/0/0       ",
        "base.default                1   0/0/0       0/0/0       0/0/0       ",
        "base.minimal                1   0/0/0       0/0/0       0/1/0       ",
        "guard.fail_closed           1   1/0/0       0/0/0       0/0/0       ",
    ])

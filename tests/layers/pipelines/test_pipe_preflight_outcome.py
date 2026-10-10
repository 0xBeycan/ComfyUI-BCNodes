"""PreFlight Outcome's flow and record list (pipelines/preflight/outcome.py), no ComfyUI:

  - the label: id · MM-DD HH:MM · worst verdict per platform · garment/setting · caption start, its id back;
  - the record list: newest first, at most RECORD_LIMIT, [NO_RECORDS] for an empty or unreadable store
    (logged), the RecordsPayload shape;
  - log: record_id_override over the label's id, nothing selected, an unknown platform as an "error:" status.
"""

import json
import logging

import pytest


@pytest.fixture
def oc(bcnodes):
    return bcnodes["pipelines.preflight.outcome"]


@pytest.fixture
def feedback(bcnodes):
    return bcnodes["pipelines.preflight.feedback"]


def judged(bcnodes, **kw):
    obs = dict(subject_appears_under_18=False, garment="regular", setting="studio", framing="full_body",
               pose="neutral", see_through_or_wet=False, exposure="none", nudity_or_sexual_act=False,
               motion_flags="none", visible_text="", confidence="high", meta={"schema_version": "1"})
    obs.update(kw)
    return bcnodes["pipelines.preflight.rules"].judge(obs)


def test_record_label_roundtrips_to_id(oc):
    rec = {"id": "a3f9c2d1", "ts": "2026-07-24T14:02:11Z",
           "verdicts": {"instagram": {"worst": "RISK"}, "tiktok": {"worst": "BLOCK"},
                        "x": {"worst": "OK"}},
           "observations": {"garment": "bikini", "setting": "beach_pool"},
           "caption_excerpt": "summer drop 🌞 limited"}
    label = oc.format_record_label(rec)
    assert label == 'a3f9c2d1 · 07-24 14:02 · IG:RISK TT:BLOCK X:OK · bikini/beach_pool · "summer drop 🌞 lim…"'
    assert oc.id_from_label(label) == "a3f9c2d1"


def test_label_of_a_fail_closed_record(oc):
    rec = {"id": "0badc0de", "ts": "", "verdicts": {}, "observations": {"error": "x"}, "caption_excerpt": "a\nb"}
    assert oc.format_record_label(rec) == '0badc0de ·  · IG:? TT:? X:? · ?/? · "a b"'


@pytest.mark.parametrize("label", ["", None, "no records yet"])
def test_no_id_from_an_empty_label(oc, label):
    assert oc.id_from_label(label) == ""


def test_record_labels_newest_first_and_limited(oc, feedback, bcnodes, tmp_path):
    store = tmp_path / "fb.jsonl"
    assert oc.record_labels(store) == [oc.NO_RECORDS] == ["no records yet"]
    ids = [feedback.log_prediction(judged(bcnodes), path=store) for _ in range(5)]
    labels = oc.record_labels(store, limit=3)
    assert [oc.id_from_label(label) for label in labels] == list(reversed(ids))[:3]
    assert oc.RECORD_LIMIT == 50 and len(oc.record_labels(store)) == 5


def test_record_labels_of_an_unreadable_store(oc, tmp_path, caplog):
    store = tmp_path / "fb.jsonl"
    store.write_text(json.dumps({"type": "prediction", "id": "abcd1234", "verdicts": ["not", "a", "dict"]}) + "\n")
    with caplog.at_level(logging.WARNING):
        assert oc.record_labels(store) == [oc.NO_RECORDS]
    assert "PreFlight Outcome" in caplog.text and "cannot be read" in caplog.text


def test_records_payload_shape(oc):
    assert oc.RecordsPayload.__annotations__ == {"records": list[str]}


def test_log_via_override_and_via_label(oc, feedback, bcnodes, tmp_path):
    store = tmp_path / "fb.jsonl"
    rid = feedback.log_prediction(judged(bcnodes), path=store)
    assert oc.log(oc.NO_RECORDS, "tiktok", "removed", rid, store) == "logged: %s tiktok=removed" % rid
    [label] = oc.record_labels(store)
    assert oc.log(label, "instagram", "clean", "  ", store) == "logged: %s instagram=clean" % rid
    assert oc.log("ffff0000 · stale label", "x", "demoted", rid, store) == "logged: %s x=demoted" % rid  # override wins
    assert feedback.load_joined(store)[0]["outcomes"] == {"tiktok": "removed", "instagram": "clean", "x": "demoted"}


def test_log_nothing_selected(oc, tmp_path):
    status = oc.log(oc.NO_RECORDS, "instagram", "clean", "", tmp_path / "fb.jsonl")
    assert "no record selected" in status and not (tmp_path / "fb.jsonl").exists()


def test_log_bad_value_reported_not_raised(oc, tmp_path):
    status = oc.log("badlabel", "myspace", "clean", "x", tmp_path / "fb.jsonl")
    assert status.startswith("error:") and "myspace" in status

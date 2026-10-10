"""PreFlight's rules engine (pipelines/preflight/rules.py), no ComfyUI.

Every case asserts the per-platform verdict range AND the exact ``fired_rules`` content and order, since a
real-world miss has to be attributable to a specific rule (the Calibrate node's per-rule table).
"""

import itertools

import pytest


@pytest.fixture
def rules(bcnodes):
    return bcnodes["pipelines.preflight.rules"]


_RANK = {"OK": 0, "RISK": 1, "BLOCK": 2}


def obs(**kw):
    """A fully-benign, schema-valid observation; override fields via kwargs."""
    base = dict(
        subject_appears_under_18=False, garment="regular", setting="studio",
        framing="full_body", pose="neutral", see_through_or_wet=False,
        exposure="none", nudity_or_sexual_act=False, motion_flags="none",
        visible_text="", confidence="high", meta={"schema_version": "1"},
    )
    base.update(kw)
    return base


def V(report, platform):
    v = report["verdicts"][platform]
    return v["best"], v["worst"]


# --- 1. base cases: verdict range + exact fired_rules -------------------------

def test_bikini_beach(rules):
    r = rules.judge(obs(garment="bikini", setting="beach_pool", exposure="mild"))
    assert V(r, "instagram") == ("RISK", "RISK")
    assert V(r, "tiktok") == ("RISK", "RISK")       # beach/pool collapses TT to RISK
    assert V(r, "x") == ("OK", "RISK")
    assert r["verdicts"]["x"]["label_required"] is False
    assert r["fired_rules"] == ["base.bikini"]   # mild exposure alone no longer fires
    assert not any(r["verdicts"][p]["hard"] for p in rules.PLATFORMS)


def test_bikini_bedroom_kills_tiktok(rules):
    r = rules.judge(obs(garment="bikini", setting="bedroom", exposure="mild"))
    assert V(r, "instagram") == ("RISK", "RISK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")     # indoors -> FYF ineligible
    assert r["fired_rules"] == ["base.bikini"]


def test_lingerie(rules):
    r = rules.judge(obs(garment="lingerie", exposure="significant"))
    assert V(r, "instagram") == ("RISK", "RISK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert V(r, "x") == ("RISK", "RISK")
    assert r["verdicts"]["x"]["label_required"] is True
    assert r["fired_rules"] == ["base.minimal"]


def test_see_through(rules):
    r = rules.judge(obs(garment="dress", see_through_or_wet=True))
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert V(r, "x") == ("RISK", "RISK")
    assert r["fired_rules"] == ["base.see_through"]


def test_nudity(rules):
    r = rules.judge(obs(nudity_or_sexual_act=True, exposure="significant"))
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert V(r, "x") == ("RISK", "RISK")            # X only labels, never blocks nudity
    assert r["verdicts"]["x"]["label_required"] is True
    assert r["fired_rules"] == ["base.nudity", "base.minimal"]


def test_clean_default(rules):
    r = rules.judge(obs())
    for p in rules.PLATFORMS:
        assert V(r, p) == ("OK", "OK")
    assert r["verdicts"]["x"]["label_required"] is False
    assert r["fired_rules"] == ["base.default"]
    assert r["unknown"] is False


# --- 2. apparent-minor override (gated on a suggestive signal) ----------------

def test_sexualized_minor_blocks_all_and_stops(rules):
    # minor + bikini + moderate exposure = a suggestive signal -> hard block.
    r = rules.judge(obs(subject_appears_under_18=True, garment="bikini",
                        exposure="moderate", confidence="low"))
    for p in rules.PLATFORMS:
        assert V(r, p) == ("BLOCK", "BLOCK")
        assert r["verdicts"][p]["hard"] is True
    assert any("apparent minor" in x for x in r["verdicts"]["instagram"]["reasons"])
    # Override skips every later stage — including confidence widening.
    assert r["fired_rules"] == ["override.minor"]


def test_benign_apparent_minor_is_not_blocked(rules):
    # A clothed, neutral apparent minor is ordinary content — NOT a violation.
    # This is the false-positive that made the tool unusable for adult creators.
    r = rules.judge(obs(subject_appears_under_18=True, garment="regular",
                        setting="outdoor", framing="face_centric", pose="neutral",
                        exposure="none"))
    for p in rules.PLATFORMS:
        assert V(r, p) == ("OK", "OK")
    assert r["fired_rules"] == ["base.default"]
    assert "override.minor" not in r["fired_rules"]


@pytest.mark.parametrize("kw", [
    dict(exposure="moderate"),
    dict(see_through_or_wet=True),
    dict(pose="suggestive"),
    dict(garment="lingerie"),
    dict(framing="chest_focus"),
])
def test_minor_gate_fires_on_each_suggestive_signal(rules, kw):
    r = rules.judge(obs(subject_appears_under_18=True, **kw))
    assert r["fired_rules"] == ["override.minor"]
    assert V(r, "instagram") == ("BLOCK", "BLOCK")


# --- 3. hard collapses --------------------------------------------------------

def test_solicitation_combo(rules):
    r = rules.judge(obs(garment="bikini", setting="beach_pool", exposure="moderate"),
                    caption_text="DM for content 🔥")
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert r["verdicts"]["instagram"]["hard"] is True
    assert "price_or_subscription_cta" in r["caption_flags"]
    assert r["fired_rules"] == ["base.bikini", "base.exposure_mod", "hard.solicitation"]


def test_adult_link_caption(rules):
    r = rules.judge(obs(garment="regular"), caption_text="all my content → fanvue.com/example")
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert r["verdicts"]["instagram"]["hard"] is True
    assert r["verdicts"]["tiktok"]["hard"] is True
    assert "adult_link" in r["caption_flags"]
    # Regular clothing + no suggestive pose => solicitation rule does NOT fire;
    # only the visual-independent adult_link collapse does.
    assert r["fired_rules"] == ["base.default", "hard.adult_link"]


def test_in_image_fanvue_watermark_triggers_adult_link(rules):
    # No caption at all — the link is only in visible_text (burned-in watermark).
    r = rules.judge(obs(garment="regular", visible_text="fanvue.com/example ♡"))
    assert "adult_link" in r["caption_flags"]
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert r["fired_rules"] == ["base.default", "hard.adult_link"]


# --- 4. confidence widening (and its interaction with hard) -------------------

def test_low_confidence_widens_range(rules):
    r = rules.judge(obs(garment="bikini", setting="beach_pool", exposure="mild",
                        confidence="low"))
    assert V(r, "instagram") == ("OK", "BLOCK")     # RISK/RISK widened both ways
    assert V(r, "tiktok") == ("OK", "BLOCK")
    assert r["fired_rules"] == ["base.bikini", "mod.confidence_low"]
    assert any("range widened" in x for x in r["verdicts"]["instagram"]["reasons"])


def test_low_confidence_never_reopens_hard_solicitation(rules):
    r = rules.judge(obs(garment="lingerie", exposure="significant", confidence="low"),
                    caption_text="onlyfans — subscribe for more")
    ig = r["verdicts"]["instagram"]
    assert (ig["best"], ig["worst"]) == ("BLOCK", "BLOCK")   # hard, NOT widened open
    assert ig["hard"] is True
    tt = r["verdicts"]["tiktok"]                             # not hard -> widened
    assert (tt["best"], tt["worst"]) == ("RISK", "BLOCK")
    assert r["fired_rules"] == ["base.minimal", "hard.solicitation", "mod.confidence_low"]


# --- 5. fail-closed -----------------------------------------------------------

@pytest.mark.parametrize("bad", [
    None,
    {"error": "vision failed"},
    {"garment": "bikini", "meta": {"schema_version": "9"}},   # unknown schema
    {"garment": "bikini"},                                    # meta missing entirely
])
def test_fail_closed_returns_unknown_never_ok(rules, bad):
    r = rules.judge(bad)
    assert r["unknown"] is True
    assert r["fired_rules"] == ["guard.fail_closed"]
    for p in rules.PLATFORMS:
        assert V(r, p) == ("UNKNOWN", "UNKNOWN")
        assert r["verdicts"][p]["best"] != "OK"     # never emits OK in this state


# --- 6. modifiers, motion, order ----------------------------------------------

def test_twerk_motion_floors_but_is_not_hard(rules):
    r = rules.judge(obs(garment="shorts", motion_flags="twerk_grind_striptease"),
                    is_video=True)
    assert V(r, "instagram") == ("BLOCK", "BLOCK")
    assert V(r, "tiktok") == ("BLOCK", "BLOCK")
    assert V(r, "x") == ("RISK", "RISK")
    assert r["verdicts"]["instagram"]["hard"] is False       # modifier, not a hard rule
    assert r["fired_rules"] == ["base.default", "mod.motion_twerk"]


def test_motion_flag_ignored_for_single_image(rules):
    # Same flag but is_video=False -> motion suppressed (guards a stray still flag).
    r = rules.judge(obs(garment="shorts", motion_flags="twerk_grind_striptease"),
                    is_video=False)
    assert "mod.motion_twerk" not in r["fired_rules"]
    assert r["fired_rules"] == ["base.default"]


def test_fired_rules_full_order(rules):
    # base(2) -> modifiers(framing, pose, text) -> widening, in strict order.
    r = rules.judge(
        obs(garment="bikini", setting="indoor_other", exposure="moderate",
            framing="chest_focus", pose="suggestive", confidence="low"),
        caption_text="spicy pics 18+ 🍑")
    assert r["fired_rules"] == [
        "base.bikini", "base.exposure_mod",
        "mod.framing", "mod.pose_suggestive", "mod.text_age",
        "mod.confidence_low",
    ]


# --- 7. caption flag precision ------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("DM for collabs", []),                                  # must NOT match cta
    ("DM for content", ["price_or_subscription_cta"]),
    ("subscribe now", ["price_or_subscription_cta"]),
    ("18+ only", ["age_marker"]),
    ("spicy pics in bio", ["age_marker"]),
    ("check my OnlyFans", ["adult_platform_mention"]),
    ("fansly.com/x", ["adult_platform_mention", "adult_link"]),
    ("hot 🍑", ["flagged_emoji"]),
    ("MORNING SEX OR LATE-NIGHT SEX?", ["sexual_text"]),   # overt sexual overlay
    ("nudes in bio", ["sexual_text"]),
    ("sexy summer vibes", []),                             # "sexy" is not "sex"
    ("just a normal caption", []),
])
def test_caption_flags_precision(rules, text, expected):
    assert rules.caption_flags(text) == expected


# --- 8. X label requirement ---------------------------------------------------

@pytest.mark.parametrize("kw,expected", [
    (dict(exposure="moderate"), False),
    (dict(exposure="significant"), True),
    (dict(see_through_or_wet=True), True),
    (dict(garment="underwear_only"), True),
    (dict(pose="suggestive"), True),
    (dict(exposure="mild"), False),
    (dict(garment="regular"), False),
])
def test_x_label_required(rules, kw, expected):
    r = rules.judge(obs(**kw))
    assert r["verdicts"]["x"]["label_required"] is expected


# --- 8b. calibration fixes (v1.1.0): fewer false positives, tighter ranges ----

def test_mild_exposure_on_ordinary_garment_is_ok(rules):
    # bare arms/legs/ordinary cleavage on a normal outfit is no longer a signal.
    r = rules.judge(obs(garment="regular", exposure="mild"))
    for p in rules.PLATFORMS:
        assert V(r, p) == ("OK", "OK")
    assert r["fired_rules"] == ["base.default"]


def test_revealing_casualwear_alone_is_ordinary_content(rules):
    # A cropped sweater, full-body, neutral pose, in a cafe: the garment class on
    # its own is not a demotion signal (the false positive from the field).
    r = rules.judge(obs(garment="croptop", exposure="mild"))
    for p in rules.PLATFORMS:
        assert V(r, p) == ("OK", "OK")
    assert r["fired_rules"] == ["base.default"]


def test_revealing_casualwear_with_pose_is_risk_not_removal(rules):
    r = rules.judge(obs(garment="croptop", exposure="mild", pose="mildly_suggestive"))
    assert V(r, "instagram") == ("OK", "RISK")
    assert V(r, "tiktok") == ("OK", "RISK")
    assert r["fired_rules"] == ["base.exposure_mild", "mod.pose_mild"]


def test_soft_modifiers_never_reach_block(rules):
    # croptop + mild pose + sexualized framing must STAY RISK — soft modifiers
    # must not manufacture a BLOCK (removal). This is the OK->BLOCK bug from the field.
    r = rules.judge(obs(garment="croptop", exposure="mild", pose="mildly_suggestive",
                        framing="chest_focus"))
    assert V(r, "instagram") == ("OK", "RISK")
    assert V(r, "tiktok") == ("OK", "RISK")
    assert r["fired_rules"] == ["base.exposure_mild", "mod.framing", "mod.pose_mild"]


def test_framing_alone_on_clothed_neutral_subject_is_ok(rules):
    # The sensor reads an ordinary 3/4 portrait as chest_focus; two soft signals
    # (croptop + framing) must not certify each other. This is the field case.
    r = rules.judge(obs(garment="croptop", exposure="mild", framing="chest_focus"))
    for p in rules.PLATFORMS:
        assert V(r, p) == ("OK", "OK")
    assert r["fired_rules"] == ["base.default"]
    r = rules.judge(obs(garment="regular", exposure="none", framing="butt_focus"))
    assert V(r, "instagram") == ("OK", "OK")
    assert r["fired_rules"] == ["base.default"]


def test_framing_counts_with_a_load_bearing_signal(rules):
    r = rules.judge(obs(garment="regular", exposure="moderate", framing="butt_focus"))
    assert "mod.framing" in r["fired_rules"]
    assert V(r, "instagram") == ("OK", "RISK")
    assert r["verdicts"]["instagram"]["worst"] != "BLOCK"


def test_moderate_exposure_demotes_never_removes(rules):
    # Deep cleavage in a covered crop top lands here; it must read as "probably
    # fine, at worst demoted" on every platform and need no X label.
    r = rules.judge(obs(garment="croptop", exposure="moderate", pose="mildly_suggestive"))
    assert V(r, "instagram") == ("OK", "RISK")
    assert V(r, "tiktok") == ("OK", "RISK")
    assert V(r, "x") == ("OK", "OK")
    assert r["verdicts"]["x"]["label_required"] is False
    assert r["fired_rules"] == ["base.exposure_mod", "base.exposure_mild", "mod.pose_mild"]


def test_sexual_text_overlay_flagged_and_capped(rules):
    r = rules.judge(obs(garment="croptop", framing="chest_focus", exposure="mild",
                        pose="mildly_suggestive",
                        visible_text="MORNING SEX OR LATE-NIGHT SEX?"))
    assert "sexual_text" in r["caption_flags"]
    assert V(r, "instagram") == ("OK", "RISK")          # demotion, not removal
    assert "mod.sexual_text" in r["fired_rules"]


def test_every_spread_has_a_range_driver(rules):
    # A best!=worst verdict must never be left unexplained.
    r = rules.judge(obs(garment="croptop", exposure="mild", pose="mildly_suggestive"))
    assert V(r, "instagram") == ("OK", "RISK")
    assert r["range_drivers"], "OK/RISK spread must carry a range_driver"


# --- 9. invariants ------------------------------------------------------------

def test_best_never_exceeds_worst_across_grid(rules):
    grid = itertools.product(
        ["bikini", "lingerie", "regular", "dress"],
        ["beach_pool", "bedroom", "indoor_other", "outdoor"],
        ["none", "mild", "moderate", "significant"],
        ["neutral", "mildly_suggestive", "suggestive"],
        ["high", "low"],
    )
    for garment, setting, exposure, pose, conf in grid:
        for cap in ("", "onlyfans subscribe", "fanvue.com/x", "18+ 🍑"):
            r = rules.judge(obs(garment=garment, setting=setting, exposure=exposure,
                                pose=pose, confidence=conf), caption_text=cap)
            for p in rules.PLATFORMS:
                best, worst = V(r, p)
                assert _RANK[best] <= _RANK[worst], (garment, setting, exposure, pose, conf, cap, p)


def test_report_carries_engine_version(rules):
    assert rules.judge(obs())["engine_version"] == rules.ENGINE_VERSION


# --- 10. summary_text ---------------------------------------------------------

def test_summary_single_status_and_arrow(rules):
    r = rules.judge(obs(garment="bikini", setting="beach_pool", exposure="mild",
                        confidence="low"))
    s = rules.summary_text(r)
    assert "Instagram:" in s and "TikTok:" in s and "X:" in s
    assert "→" in s                                 # widened range shows an arrow
    assert "⚡ Range driver:" in s
    assert "record:" not in s                       # no record_id set yet


def test_summary_shows_record_when_present(rules):
    r = rules.judge(obs())
    r["record_id"] = "a3f9c2d1"
    assert "record: a3f9c2d1" in rules.summary_text(r)

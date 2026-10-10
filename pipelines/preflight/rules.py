"""PreFlight's rules engine: maps sensor observations to per-platform verdict ranges.

Standard library only (re). The single home of the rule set: the node layer only calls judge /
summary_text and never re-implements a rule.

The model is a pure observation sensor; this engine is the only place that "judges". Every verdict is a
RANGE (best..worst) because the sensor cannot resolve everything with certainty and because some rules
depend on factors outside the frame (setting, region, account signals). ``range_drivers`` names the
unknown so the user can collapse the range themselves.

Every rule has a stable ID (e.g. ``base.bikini``); whenever a rule changes a verdict its ID is appended to
``fired_rules`` in application order: the hook the Calibrate node uses to attribute real-world outcomes to
individual rules.
"""

import re

ENGINE_VERSION = "1.4.0"

# Observation schema versions this engine knows how to interpret. Stage 0 fails
# closed on anything else, so an observation produced by a newer/older schema is
# never silently judged against rules written for a different shape.
KNOWN_SCHEMA_VERSIONS = {"1"}

# --- severity ----------------------------------------------------------------
# Internal severity is an int so bumps/merges are plain saturating arithmetic.
OK, RISK, BLOCK = 0, 1, 2
_STATUS = {OK: "OK", RISK: "RISK", BLOCK: "BLOCK"}
_ICON = {"OK": "✅", "RISK": "⚠️", "BLOCK": "❌", "UNKNOWN": "❓"}

PLATFORMS = ("instagram", "tiktok", "x")

DISCLAIMER = (
    "This is a prediction, not a platform approval. Account history, bio "
    "signals, posting cadence and cluster patterns are not considered. Final "
    "judgment is yours."
)
_FAIL_CLOSED_REASON = "vision sensor unavailable or incompatible — manual review required"

# Ordered so ``exposure >= moderate`` is a simple integer comparison.
_EXPOSURE_ORDER = {"none": 0, "mild": 1, "moderate": 2, "significant": 3}
_MINIMAL_GARMENTS = {"lingerie", "underwear_only"}
_MILD_GARMENTS = {"miniskirt", "croptop", "shorts", "fitness_wear"}
_FRAMING_SEXUALIZED = {"chest_focus", "butt_focus", "lowangle_body"}
_FLAGGED_EMOJI = {"🍑", "🍆", "💦", "👅", "🥵"}


def _clamp(n):
    """Saturate a severity to the valid range [OK, BLOCK]."""
    return max(OK, min(BLOCK, n))


# ---------------------------------------------------------------------------
# Caption / in-image text flags
# ---------------------------------------------------------------------------

# Ordered (name, compiled pattern). Patterns that can feed a hard rule must be
# high-precision — a bare "dm for" must NOT match ("DM for collabs"); only the
# content-qualified forms count.
_FLAG_PATTERNS = [
    ("adult_platform_mention", re.compile(r"(?i)(onlyfans|only\s*fans|fanvue|fansly)")),
    ("price_or_subscription_cta", re.compile(
        r"(?i)(dm\s*for\s*(more|content|pics?|vids?|access|fun)|pay\s*to|"
        r"subscribe|sub\s*for|ppv|exclusive\s*content)")),
    ("age_marker", re.compile(r"(?i)(18\s*\+|\bnsfw\b|\bspicy\s+(content|pics?|vids?|link)\b)")),
    ("adult_link", re.compile(r"(?i)(fanvue\.com|onlyfans\.com|fansly\.com)")),
    # Overt sexual wording in the caption or burned into the image. A demotion
    # signal (not solicitation): platforms down-rank / age-gate sexually explicit
    # text overlays even over clothed visuals.
    ("sexual_text", re.compile(
        r"(?i)(\bsex\b|\bsexting\b|\bnudes?\b|\bnaked\b|\bporn\b|\bxxx\b|"
        r"\bhookup\b|\bhorny\b|\bcum\b|blow\s?job|hand\s?job|\banal\b|"
        r"\bdeepthroat\b|\bthirst\s*trap\b)")),
]


def caption_flags(text):
    """Deterministic flags present in a single text blob (pure function).

    Returns flag names in a stable order. ``flagged_emoji`` is membership-based;
    the rest are regex. No estimation happens here — this is exact matching.
    """
    if not isinstance(text, str) or not text:
        return []
    flags = [name for name, pat in _FLAG_PATTERNS if pat.search(text)]
    if any(ch in _FLAGGED_EMOJI for ch in text):
        flags.append("flagged_emoji")
    return flags


# ---------------------------------------------------------------------------
# Verdict helpers — a verdict is {"best":int,"worst":int,"hard":bool,
# "reasons":[str]} (+ "label_required":bool for X).
# ---------------------------------------------------------------------------

def _new_verdict():
    return {"best": OK, "worst": OK, "hard": False, "reasons": []}


def _add_reason(v, reason):
    if reason and reason not in v["reasons"]:
        v["reasons"].append(reason)


def _normalize(v):
    """Clamp both bounds and re-enforce the best <= worst invariant.

    Called after every stage. If a stage raised ``best`` above ``worst`` (e.g.
    raising IG best to RISK on a formerly OK/OK verdict), the range collapses up
    to a point rather than becoming the impossible [RISK, OK].
    """
    v["best"] = _clamp(v["best"])
    v["worst"] = _clamp(max(v["worst"], v["best"]))


def _merge(v, best, worst, reason=None):
    """Max-merge a (best, worst) contribution into a verdict (never lowers)."""
    v["best"] = max(v["best"], best)
    v["worst"] = max(v["worst"], worst)
    _add_reason(v, reason)


def _bump_worst(v, n=1):
    v["worst"] = _clamp(v["worst"] + n)


def _bump_worst_capped(v, cap=RISK):
    """Raise ``worst`` by one, but never above ``cap`` via this bump alone.

    Used by soft signals (framing, mild pose, suggestive text): they can push a
    verdict up to RISK (reach demotion) but must NOT by themselves manufacture a
    BLOCK (removal) — that is reserved for genuine policy violations. A verdict
    already at BLOCK from a base rule keeps it (max), it is never lowered.
    """
    v["worst"] = max(v["worst"], min(cap, v["worst"] + 1))


def _raise_best(v, floor):
    v["best"] = _clamp(max(v["best"], floor))


# ---------------------------------------------------------------------------
# Stage 2 — base rules
# ---------------------------------------------------------------------------

def _apply_base_rules(V, obs, fired):
    ig, tt, x = V["instagram"], V["tiktok"], V["x"]
    garment = obs.get("garment", "regular")
    exposure = obs.get("exposure", "none")
    exposure_rank = _EXPOSURE_ORDER.get(exposure, 0)
    setting = obs.get("setting", "indoor_other")
    pose = obs.get("pose", "neutral")
    matched = []

    if obs.get("nudity_or_sexual_act") is True:
        matched.append("base.nudity")
        _merge(ig, BLOCK, BLOCK, "nudity (violates community standards)")
        _merge(tt, BLOCK, BLOCK, "nudity")
        _merge(x, RISK, RISK, "adult label + sensitive media settings required")

    if garment in _MINIMAL_GARMENTS or exposure == "significant":
        matched.append("base.minimal")
        _merge(ig, RISK, RISK, "minimal coverage — suggestive demotion risk")
        _merge(tt, BLOCK, BLOCK, "significant exposure — FYF ineligible")
        _merge(x, RISK, RISK, "adult label recommended")

    if garment == "bikini":
        matched.append("base.bikini")
        _merge(ig, RISK, RISK, "bikini — demotion possible")
        # TikTok collapses on setting: FYF-killed indoors, RISK at beach/pool,
        # otherwise the RISK..BLOCK range stands (setting is the unknown).
        if setting in ("bedroom", "indoor_other"):
            _merge(tt, BLOCK, BLOCK, "bikini indoors — FYF ineligible")
        elif setting == "beach_pool":
            _merge(tt, RISK, RISK, "bikini at beach/pool — demotion possible")
        else:
            _merge(tt, RISK, BLOCK, "bikini — RISK at beach/pool, FYF kill indoors")
        # X: OK/RISK, label only when exposure is at least moderate.
        _merge(x, OK, RISK, "bikini")

    if obs.get("see_through_or_wet") is True:
        matched.append("base.see_through")
        _merge(ig, BLOCK, BLOCK, "see-through — explicit demotion example in IG guidelines")
        _merge(tt, BLOCK, BLOCK, "implied nudity / age-restricted")
        _merge(x, RISK, RISK, "see-through / wet — adult label recommended")

    if exposure == "moderate":
        # Soft signal. The sensor files deep cleavage under "moderate" alongside
        # sideboob/underboob/partial buttock whatever the prompt says, and the
        # former is ordinary feed content — so this tier can only demote, never
        # remove, and X needs no label for it.
        matched.append("base.exposure_mod")
        _merge(ig, OK, RISK, "moderate exposure — demotion possible")
        _merge(tt, OK, RISK, "moderate exposure — demotion possible")
        _merge(x, OK, OK)

    if garment in _MILD_GARMENTS and pose != "neutral":
        # Revealing casualwear (croptop/miniskirt/shorts/fitness_wear) is a
        # demotion signal only in combination with a non-neutral pose. A cropped
        # sweater in a cafe, neutral, is everyday content — the garment class
        # alone used to flag it as RISK. Framing is deliberately NOT a companion
        # here: it is the sensor's noisiest field (see _apply_modifiers) and two
        # soft signals must not certify each other. Plain "mild" exposure on a
        # normal garment likewise does not fire.
        matched.append("base.exposure_mild")
        _merge(ig, OK, RISK, "revealing casualwear — mild demotion possible")
        _merge(tt, OK, RISK, "revealing casualwear — mild demotion possible")
        _merge(x, OK, OK)

    if not matched:
        matched.append("base.default")

    fired.extend(matched)


# ---------------------------------------------------------------------------
# Stage 3 — modifiers (saturating bumps, applied in listed order)
# ---------------------------------------------------------------------------

def _has_load_bearing_signal(obs):
    """Pose and exposure are the axes the sensor reads reliably; framing and
    garment class are amplifiers on top of them. A sexualized framing on a
    clothed, neutral subject is the field's most common false positive (an
    ordinary 3/4 portrait read as chest_focus), so it only counts when one of
    these is present too.
    """
    return (
        _EXPOSURE_ORDER.get(obs.get("exposure", "none"), 0) >= _EXPOSURE_ORDER["moderate"]
        or obs.get("pose", "neutral") != "neutral"
        or obs.get("garment") in ({"bikini"} | _MINIMAL_GARMENTS)
        or obs.get("see_through_or_wet") is True
    )


def _apply_modifiers(V, obs, flags, motion, fired):
    ig, tt = V["instagram"], V["tiktok"]

    if obs.get("framing") in _FRAMING_SEXUALIZED and _has_load_bearing_signal(obs):
        fired.append("mod.framing")
        _bump_worst_capped(ig)
        _bump_worst_capped(tt)
        note = "sexualized framing (demotes reach even with ordinary clothing)"
        _add_reason(ig, note)
        _add_reason(tt, note)

    pose = obs.get("pose", "neutral")
    if pose == "suggestive":
        fired.append("mod.pose_suggestive")
        _bump_worst_capped(ig)
        _bump_worst_capped(tt)
        _raise_best(ig, RISK)
        _add_reason(ig, "IG demotes suggestive poses directly")
        _add_reason(tt, "suggestive pose")
    elif pose == "mildly_suggestive":
        fired.append("mod.pose_mild")
        _bump_worst_capped(tt)
        _add_reason(tt, "mildly suggestive pose")

    if motion == "twerk_grind_striptease":
        fired.append("mod.motion_twerk")
        # Explicit sexual motion IS a genuine violation -> floor at BLOCK
        # (max-merge, not assignment) so an already-worse verdict is kept.
        _merge(ig, BLOCK, BLOCK, "twerk/grind/striptease motion")
        _merge(tt, BLOCK, BLOCK, "twerk/grind/striptease motion — age-restricted")
        _merge(V["x"], RISK, RISK, "sexualized motion — adult label recommended")
    elif motion == "intimate_kissing":
        fired.append("mod.motion_kiss")
        _bump_worst_capped(tt)
        _add_reason(tt, "intimate kissing")

    if "age_marker" in flags or "flagged_emoji" in flags:
        fired.append("mod.text_age")
        _bump_worst_capped(ig)
        _bump_worst_capped(tt)
        note = "adult/age-coded text or emoji in caption or image"
        _add_reason(ig, note)
        _add_reason(tt, note)

    if "sexual_text" in flags:
        fired.append("mod.sexual_text")
        _bump_worst_capped(ig)
        _bump_worst_capped(tt)
        note = "sexually explicit text in caption or image overlay"
        _add_reason(ig, note)
        _add_reason(tt, note)


# ---------------------------------------------------------------------------
# Stage 4 — hard collapses (immune to Stage 5 widening)
# ---------------------------------------------------------------------------

def _apply_hard_rules(V, obs, flags, fired):
    ig, tt = V["instagram"], V["tiktok"]
    garment = obs.get("garment", "regular")
    exposure = obs.get("exposure", "none")
    pose = obs.get("pose", "neutral")

    solicit = any(f in flags for f in
                  ("adult_platform_mention", "price_or_subscription_cta", "adult_link"))
    suggestive_visual = (exposure != "none" or pose != "neutral"
                         or garment in ({"bikini"} | _MINIMAL_GARMENTS))
    if solicit and suggestive_visual:
        fired.append("hard.solicitation")
        _merge(ig, BLOCK, BLOCK,
               "sexual solicitation: suggestive visual + caption offer = removal")
        ig["hard"] = True

    if "adult_link" in flags:
        fired.append("hard.adult_link")
        note = ("adult platform link (caption or in-image watermark); "
                "applies independently of the visual")
        _merge(ig, BLOCK, BLOCK, note)
        _merge(tt, BLOCK, BLOCK, note)
        ig["hard"] = True
        tt["hard"] = True


# ---------------------------------------------------------------------------
# Stage 5 — confidence widening (never reopens a hard collapse)
# ---------------------------------------------------------------------------

def _apply_confidence(V, obs, fired):
    if obs.get("confidence") != "low":
        return
    widened = False
    for v in V.values():
        if v["hard"]:
            continue  # widening must never reopen a hard verdict
        v["best"] = _clamp(v["best"] - 1)
        v["worst"] = _clamp(v["worst"] + 1)
        _add_reason(v, "low sensor confidence — range widened")
        widened = True
    if widened:
        fired.append("mod.confidence_low")


# ---------------------------------------------------------------------------
# Stage 6 — X label requirement
# ---------------------------------------------------------------------------

def _has_suggestive_signal(obs, flags, motion):
    """Any signal that turns an apparent-minor depiction into a platform
    violation. A clothed, neutral apparent minor has NONE of these and is
    ordinary content — it must not be blocked. Everyday youth clothing (a plain
    croptop/shorts, full-body, neutral) is deliberately NOT enough on its own;
    it takes real exposure, a swim/intimate garment, a suggestive pose/framing,
    sexual motion, or sexual/adult text.
    """
    if obs.get("exposure") in ("moderate", "significant"):
        return True
    if obs.get("see_through_or_wet") is True or obs.get("nudity_or_sexual_act") is True:
        return True
    if obs.get("pose") in ("mildly_suggestive", "suggestive"):
        return True
    if obs.get("garment") in ({"bikini", "swimsuit_onepiece"} | _MINIMAL_GARMENTS):
        return True
    if obs.get("framing") in _FRAMING_SEXUALIZED:
        return True
    if motion != "none":
        return True
    return any(f in flags for f in ("sexual_text", "age_marker", "adult_platform_mention",
                                    "price_or_subscription_cta", "adult_link", "flagged_emoji"))


def _apply_x_label(V, obs, motion):
    exposure = obs.get("exposure", "none")
    required = (
        obs.get("nudity_or_sexual_act") is True
        or exposure == "significant"
        or obs.get("see_through_or_wet") is True
        or obs.get("garment") in _MINIMAL_GARMENTS
        or obs.get("pose") == "suggestive"
        or motion == "twerk_grind_striptease"
    )
    x = V["x"]
    x["label_required"] = bool(required)
    if required:
        _add_reason(x, "never use as profile photo, header, banner, or live thumbnail")


# ---------------------------------------------------------------------------
# range_drivers — human-readable explanations of what the range hinges on
# ---------------------------------------------------------------------------

def _range_drivers(obs, fired, widened_low_conf, V):
    drivers = []
    explained = set()
    if "base.bikini" in fired:
        setting = obs.get("setting", "indoor_other")
        if setting == "beach_pool":
            drivers.append(
                "TikTok: bikini reported at 'beach_pool' — FYF demotion (RISK). If it "
                "were indoors (bedroom/indoor_other) TikTok would be FYF-ineligible (BLOCK).")
        elif setting in ("bedroom", "indoor_other"):
            drivers.append(
                "TikTok: bikini reported indoors ('%s') — FYF-ineligible (BLOCK). If "
                "this is actually a beach or pool, the verdict drops to RISK." % setting)
        else:
            drivers.append(
                "TikTok: bikini reported in '%s' — verdict spans RISK (beach/pool) to "
                "BLOCK (indoors); confirm the setting to collapse it." % setting)
        explained.add("tiktok")
    if widened_low_conf:
        drivers.append(
            "All non-final verdicts were widened because the sensor reported LOW "
            "confidence — a clearer, less-occluded image would tighten these ranges.")
        explained.update(PLATFORMS)
    # Generic fallback: never leave a best!=worst spread unexplained, so the user
    # always knows what the range means.
    labels = {"instagram": "Instagram", "tiktok": "TikTok"}
    spread = [labels[p] for p in ("instagram", "tiktok")
              if p not in explained and V[p]["best"] < V[p]["worst"]]
    if spread:
        drivers.append(
            "%s: the best-worst spread is soft demotion risk — the low end assumes a "
            "clean account and neutral context, the high end an unlucky classifier "
            "pass. It means possible reach loss, not removal." % "/".join(spread))
    return drivers


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _serialize(v, is_x=False):
    out = {
        "best": _STATUS[v["best"]],
        "worst": _STATUS[v["worst"]],
        "hard": bool(v["hard"]),
        "reasons": list(v["reasons"]),
    }
    if is_x:
        out["label_required"] = bool(v.get("label_required", False))
    return out


def _build_report(observations, flags, fired, verdicts, drivers, unknown):
    return {
        "engine_version": ENGINE_VERSION,
        "record_id": "",  # injected by the Report flow after logging
        "observations": observations if isinstance(observations, dict) else {},
        "caption_flags": flags,
        "fired_rules": fired,
        "verdicts": {
            "instagram": _serialize(verdicts["instagram"]),
            "tiktok": _serialize(verdicts["tiktok"]),
            "x": _serialize(verdicts["x"], is_x=True),
        },
        "range_drivers": drivers,
        "unknown": unknown,
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge(observations, caption_text="", is_video=False):
    """Apply the full pipeline and return a structured report.

    Stages run in strict order and must not be reordered: 0 fail-closed,
    1 minor override, 2 base rules, 3 modifiers, 4 hard collapses, 5 confidence
    widening, 6 X label.
    """
    # Flags are computed over the UNION of the caption and the in-image text, so
    # a fanvue.com watermark burned into the frame fires the same rules a caption
    # link would. caption_flags() itself stays a pure single-text function.
    visible_text = ""
    if isinstance(observations, dict):
        vt = observations.get("visible_text", "")
        visible_text = vt if isinstance(vt, str) else ""
    flags = caption_flags((caption_text or "") + "\n" + visible_text)

    # --- Stage 0: fail closed -------------------------------------------------
    schema_version = None
    if isinstance(observations, dict):
        meta = observations.get("meta")
        if isinstance(meta, dict):
            schema_version = meta.get("schema_version")
    if (observations is None or not isinstance(observations, dict)
            or "error" in observations
            or schema_version not in KNOWN_SCHEMA_VERSIONS):
        unknown_v = {"best": "UNKNOWN", "worst": "UNKNOWN", "hard": False,
                     "reasons": [_FAIL_CLOSED_REASON]}
        verdicts = {
            "instagram": dict(unknown_v),
            "tiktok": dict(unknown_v),
            "x": {**unknown_v, "label_required": False},
        }
        # Rebuild reasons as independent lists (dict() shares the list object).
        for name in PLATFORMS:
            verdicts[name]["reasons"] = [_FAIL_CLOSED_REASON]
        return {
            "engine_version": ENGINE_VERSION,
            "record_id": "",
            "observations": observations if isinstance(observations, dict) else {},
            "caption_flags": flags,
            "fired_rules": ["guard.fail_closed"],
            "verdicts": verdicts,
            "range_drivers": [],
            "unknown": True,
            "disclaimer": DISCLAIMER,
        }

    # motion flags only count for genuine multi-frame (video) input; guard a
    # stray flag on a single still. Computed before Stage 1 so the minor gate
    # can consider sexual motion.
    motion = observations.get("motion_flags", "none") if is_video else "none"

    # --- Stage 1: apparent-minor override, GATED on a suggestive signal ------
    # A clothed, neutral depiction of an apparent minor is ordinary content and
    # is NOT blocked — platforms allow minors to appear fully clothed. Only a
    # *sexualized* depiction is a removal-and-report risk. This replaces the old
    # unconditional block-everything, which nuked every youthful-looking adult
    # (the tool's entire audience is adult creators).
    if (observations.get("subject_appears_under_18") is True
            and _has_suggestive_signal(observations, flags, motion)):
        verdicts = {}
        for name in PLATFORMS:
            verdicts[name] = {"best": "BLOCK", "worst": "BLOCK", "hard": True,
                              "reasons": ["sexualized depiction of an apparent minor "
                                          "— removal and report risk"]}
        verdicts["x"]["label_required"] = False
        return {
            "engine_version": ENGINE_VERSION,
            "record_id": "",
            "observations": observations,
            "caption_flags": flags,
            "fired_rules": ["override.minor"],
            "verdicts": verdicts,
            "range_drivers": [],
            "unknown": False,
            "disclaimer": DISCLAIMER,
        }

    V = {name: _new_verdict() for name in PLATFORMS}
    fired = []

    _apply_base_rules(V, observations, fired)
    for v in V.values():
        _normalize(v)

    _apply_modifiers(V, observations, flags, motion, fired)
    for v in V.values():
        _normalize(v)

    _apply_hard_rules(V, observations, flags, fired)
    for v in V.values():
        _normalize(v)

    _low_conf = observations.get("confidence") == "low"
    _apply_confidence(V, observations, fired)
    widened = _low_conf and "mod.confidence_low" in fired
    for v in V.values():
        _normalize(v)

    _apply_x_label(V, observations, motion)

    drivers = _range_drivers(observations, fired, widened, V)
    return _build_report(observations, flags, fired, V, drivers, unknown=False)


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

def _status_part(best, worst):
    if best == worst:
        return "%s %s" % (_ICON[best], best)
    return "%s %s → %s %s" % (_ICON[best], best, _ICON[worst], worst)


def summary_text(report):
    """One-screen preview of a report."""
    verdicts = report.get("verdicts", {})
    labels = [("Instagram", "instagram"), ("TikTok", "tiktok"), ("X", "x")]
    lines = []
    for label, key in labels:
        v = verdicts.get(key, {})
        best = v.get("best", "UNKNOWN")
        worst = v.get("worst", "UNKNOWN")
        status = _status_part(best, worst)
        if key == "x" and v.get("label_required"):
            reason = "adult label recommended"
        else:
            reason = "; ".join(v.get("reasons", []))
        line = "%-11s%s" % (label + ":", status)
        if reason:
            line += " — " + reason
        lines.append(line)

    for driver in report.get("range_drivers", []):
        lines.append("")
        lines.append("⚡ Range driver: " + driver)

    record_id = report.get("record_id", "")
    if record_id:
        lines.append("")
        lines.append("record: " + record_id)

    return "\n".join(lines)

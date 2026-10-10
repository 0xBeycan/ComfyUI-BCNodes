"""PreFlight's observation prompt and schema.

The single source of truth for (a) the verbatim prompt the sensor runs (sent as the user text, after the
images; observe.py) and (b) the JSON schema that prompt is expected to emit. The two travel together on
purpose: PROMPT_VERSION bumps on any wording change, SCHEMA_VERSION on any change to the JSON shape.
PreFlight Observe stamps both into every observation's meta, so old records stay interpretable when the
rules are run again later (docs/preflight.md, "Determinism and re-evaluating old content").

Standard library only: the schema and the validate_observations coercion are tested without ComfyUI.
"""

# Bump SCHEMA_VERSION on any change to the JSON keys/enums below; bump
# PROMPT_VERSION on any change to OBSERVATION_PROMPT wording. rules.py declares
# which SCHEMA_VERSIONs it can still interpret (rules.KNOWN_SCHEMA_VERSIONS).
SCHEMA_VERSION = "1"
PROMPT_VERSION = "5"

# The observation engine's prompt. Kept verbatim; the model is a pure
# "what is visually present" sensor and never judges acceptability.
OBSERVATION_PROMPT = """You are a visual observation engine for pre-publication content checks.
You do NOT make decisions, give ratings, or judge acceptability. You only report what is visually present.

Look at the image(s). If multiple frames are given, they are sampled from one video: infer motion from the sequence, but base all appearance fields on the MOST REVEALING frame.

Return ONLY a JSON object. No prose, no explanation, no markdown fences.

{
  "subject_appears_under_18": boolean,
  "garment": "bikini" | "lingerie" | "underwear_only" | "miniskirt" | "croptop" | "shorts" | "dress" | "fitness_wear" | "swimsuit_onepiece" | "regular" | "other",
  "setting": "beach_pool" | "bedroom" | "indoor_other" | "outdoor" | "gym" | "studio",
  "framing": "face_centric" | "full_body" | "chest_focus" | "butt_focus" | "lowangle_body",
  "pose": "neutral" | "mildly_suggestive" | "suggestive",
  "see_through_or_wet": boolean,
  "exposure": "none" | "mild" | "moderate" | "significant",
  "nudity_or_sexual_act": boolean,
  "motion_flags": "none" | "twerk_grind_striptease" | "intimate_kissing",
  "visible_text": string,
  "confidence": "high" | "medium" | "low"
}

Field definitions:
- subject_appears_under_18: true ONLY when there are concrete visual indicators the person is a minor — child or early-adolescent facial structure and body proportions, or an unmistakable school-age context. A youthful-looking adult is NOT a minor. If the person is plausibly an adult, answer false. Do not answer true merely because the face looks young or the styling is casual.
- garment: the most revealing garment worn. "bikini" = two-piece swimwear. "lingerie" = intimate apparel not intended as swimwear. "underwear_only" = plain underwear with no outer layer.
- setting: the physical environment. Use "studio" only for plain/seamless backdrops with no environmental cues.
- framing: where the composition places emphasis. "face_centric" = head-and-shoulders portrait. "full_body" = an ordinary portrait or lifestyle shot where the face and most of the torso or body are visible — this is the default for everyday photos, even when the person is centered or the chest happens to sit mid-frame. "chest_focus" / "butt_focus" = the frame is deliberately composed around that body region: it fills most of the frame and the face is cropped out, cut off, or pushed to the edge, regardless of how much clothing is worn. "lowangle_body" = camera clearly below waist height angled up at the body.
- pose: "neutral" = standing, sitting, walking, ordinary lifestyle posture. "mildly_suggestive" = arched back, hand on hip with body emphasis, over-shoulder glance. "suggestive" = poses whose primary purpose is sexual appeal — spread legs, on all fours, hands on intimate areas, bent over toward camera.
- see_through_or_wet: true if fabric is transparent, mesh, wet-clinging, or if nipple outline is visible through clothing.
- exposure: "none" = fully covered. "mild" = bare arms, bare legs, bare midriff, and a low neckline showing the upper chest and the inner curve of the breasts — no matter how deep. "moderate" = the side of the breast, the underside of the breast, or part of the buttocks is uncovered. "significant" = underwear-only, nipple covers only, implied nudity.
- nudity_or_sexual_act: true only for exposed genitals, exposed female nipples, exposed full buttocks, or a depicted sexual act.
- motion_flags: only from multi-frame input; use "none" for a single image.
- visible_text: transcribe any readable text visible in the image — watermarks, overlaid captions, usernames, URLs, stickers. Verbatim, up to ~200 characters. Use "" if there is none.
- confidence: your own certainty about these observations. Use "low" for heavy occlusion, extreme crops, poor lighting, or ambiguous framing.

Report only what is visible. Do not infer intent, do not speculate about where the content will be posted, and do not soften observations."""

# Optional few-shot examples appended after the schema during calibration.
# Leave empty by default; fill with "<image description>\n<expected JSON>" blocks
# if empirical accuracy is poor. Appended verbatim by build_prompt().
FEW_SHOT_EXAMPLES = ""


def build_prompt():
    """The full prompt sent to the model (schema + any few-shot block)."""
    if FEW_SHOT_EXAMPLES.strip():
        return OBSERVATION_PROMPT + "\n\nExamples:\n" + FEW_SHOT_EXAMPLES.strip()
    return OBSERVATION_PROMPT


# --- schema as data ----------------------------------------------------------
# Kept as plain structures so validation is table-driven and lives in one place.

# Enum fields -> the set of values the model is allowed to emit.
ENUMS = {
    "garment": {
        "bikini", "lingerie", "underwear_only", "miniskirt", "croptop",
        "shorts", "dress", "fitness_wear", "swimsuit_onepiece", "regular", "other",
    },
    "setting": {"beach_pool", "bedroom", "indoor_other", "outdoor", "gym", "studio"},
    "framing": {"face_centric", "full_body", "chest_focus", "butt_focus", "lowangle_body"},
    "pose": {"neutral", "mildly_suggestive", "suggestive"},
    "exposure": {"none", "mild", "moderate", "significant"},
    "motion_flags": {"none", "twerk_grind_striptease", "intimate_kissing"},
    "confidence": {"high", "medium", "low"},
}
BOOL_KEYS = ("subject_appears_under_18", "see_through_or_wet", "nudity_or_sexual_act")
STRING_KEYS = ("visible_text",)

# Every key the model must return (meta is injected by the observe flow, not the model).
REQUIRED_KEYS = frozenset(ENUMS) | frozenset(BOOL_KEYS) | frozenset(STRING_KEYS)

# Coercion targets for missing/invalid values. The guiding rule is "never
# silently default to the permissive end":
#   * Primary risk axes lean cautious: exposure->moderate, pose->mildly_suggestive,
#     garment->croptop (mild-coverage tier), confidence->low (widens the range).
#   * Descriptive/modifier axes default neutral (framing->full_body, setting->
#     indoor_other, motion_flags->none) so a single dropped field does not
#     manufacture a bump; the primary axes above still carry the caution.
#   * The high-stakes booleans (nudity, see_through, subject_appears_under_18)
#     default False: a minor / nudity / see-through verdict must rest on a
#     POSITIVE observation, not on a dropped field. Defaulting the age flag True
#     used to nuke every observation with a parse hiccup to BLOCK; the minor rule
#     now also requires a suggestive visual (see rules.py), so this is doubly safe.
SAFE_DEFAULTS = {
    "subject_appears_under_18": False,
    "garment": "croptop",
    "setting": "indoor_other",
    "framing": "full_body",
    "pose": "mildly_suggestive",
    "see_through_or_wet": False,
    "exposure": "moderate",
    "nudity_or_sexual_act": False,
    "motion_flags": "none",
    "visible_text": "",
    "confidence": "low",
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f"}


def _coerce_bool(value, default):
    """Best-effort bool from model output; unrecognized/missing -> default.

    bool is checked before int because ``isinstance(True, int)`` is True.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
    return default


def has_observation_keys(obj):
    """True if obj is a dict that shares at least one key with the schema.

    The observe flow uses this to tell a partial-but-real observation (coerce and
    proceed) from junk like ``{}`` or ``{"foo": 1}`` (treat as a parse failure
    and retry / fail closed). A completely unrecognized object is never silently
    turned into a full cautious observation.
    """
    return isinstance(obj, dict) and bool(REQUIRED_KEYS & set(obj.keys()))


def validate_observations(obj):
    """Return a schema-clean copy of ``obj``.

    Every enum value not in its allowed set (or missing) is coerced to the
    cautious default; booleans are coerced from common truthy/falsy forms;
    ``visible_text`` becomes "" if it is not a string. The result always has
    exactly the schema keys (no ``meta`` — the observe flow injects that afterwards).
    """
    cleaned = {}
    for key, allowed in ENUMS.items():
        val = obj.get(key)
        cleaned[key] = val if val in allowed else SAFE_DEFAULTS[key]
    for key in BOOL_KEYS:
        cleaned[key] = _coerce_bool(obj.get(key), SAFE_DEFAULTS[key])
    text = obj.get("visible_text", "")
    cleaned["visible_text"] = text if isinstance(text, str) else ""
    return cleaned

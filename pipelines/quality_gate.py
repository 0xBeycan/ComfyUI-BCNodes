"""Image Quality Gate flow: shot-type profiles scale the thresholds, the five
libs/image_metrics scores are checked three-tier (QualityCheck), folded into a verdict
(PASS / SO-SO / FAIL), a text report and a badge (PIL image). cv2 and PIL are imported
inside the functions that use them.
"""

from dataclasses import dataclass

import numpy as np

from ..libs.image_metrics import blur_detection, clipping_analysis, entropy_analysis, noise_estimation, sharpness_hybrid

# Shot type presets apply multipliers to base thresholds.
# "custom" uses raw slider values with no scaling.
SHOT_PROFILES = {
    "custom": {
        "blur": 1.0,
        "sharpness": 1.0,
        "noise": 1.0,
        "clipping": 1.0,
        "entropy": 1.0,
        "block_size": 24,
    },
    "close-up": {
        "blur": 0.6,
        "sharpness": 1.5,
        "noise": 0.8,
        "clipping": 0.8,
        "entropy": 1.1,
        "block_size": 16,
    },
    "medium": {
        "blur": 1.0,
        "sharpness": 1.0,
        "noise": 1.0,
        "clipping": 1.0,
        "entropy": 1.0,
        "block_size": 24,
    },
    "wide / full-body": {
        "blur": 1.6,
        "sharpness": 0.6,
        "noise": 1.3,
        "clipping": 1.2,
        "entropy": 0.85,
        "block_size": 32,
    },
}

# Margin zone multiplier for SO-SO detection.
# Upper-bound metrics (blur, noise, clip): score <= threshold*MF → marginal
# Lower-bound metrics (sharp, entropy): score >= threshold/MF → marginal
MARGIN_FACTOR = 1.4


@dataclass
class QualityCheck:
    """One metric scored against its effective threshold. score is what the metric returned
    (float for blur, np.float64 for the others), unconverted; threshold and margin are kept
    although nothing reads them."""

    name: str
    score: float
    threshold: float
    condition: str
    margin: str
    state: str  # "pass" | "marginal" | "fail"


def analyze(image, shot_type, blur_mode, blur_threshold, blur_var_threshold,
            sharpness_threshold, noise_threshold, clipping_threshold, entropy_threshold):
    """Frame 0 of an IMAGE batch -> (badge PIL image, verdict int, report, blur, sharpness,
    noise, clipping and entropy scores)."""
    import cv2

    profile = SHOT_PROFILES[shot_type]

    # Effective thresholds (custom = 1.0 multiplier = raw slider value)
    eff_blur_t = blur_threshold * profile["blur"]
    eff_sharp_t = sharpness_threshold * profile["sharpness"]
    eff_noise_t = noise_threshold * profile["noise"]
    eff_clip_t = clipping_threshold * profile["clipping"]
    eff_entropy_t = entropy_threshold * profile["entropy"]
    eff_block = profile["block_size"]

    # Tensor to numpy
    img_np = np.clip(image[0].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- Run all 5 analyses ---
    center_weighted = (blur_mode == "center-weighted")
    blur_score = blur_detection(gray, eff_block, center_weighted, blur_var_threshold)
    sharpness_score = sharpness_hybrid(gray)
    noise_score = noise_estimation(gray)
    clipping_score = clipping_analysis(gray)
    entropy_score = entropy_analysis(gray)

    # --- Three-tier evaluation per metric ---
    mf = MARGIN_FACTOR

    checks = [
        evaluate_upper(f"Blur [{blur_mode}]", blur_score, eff_blur_t, mf),
        evaluate_lower("Sharpness", sharpness_score, eff_sharp_t, mf),
        evaluate_upper("Noise", noise_score, eff_noise_t, mf),
        evaluate_upper("Clipping", clipping_score, eff_clip_t, mf),
        evaluate_lower("Entropy", entropy_score, eff_entropy_t, mf),
    ]

    # --- Overall verdict ---
    # verdict int: 0=FAIL, 1=SO-SO, 2=PASS (for Switch node routing)
    states = [c.state for c in checks]
    if "fail" in states:
        verdict_label = "FAIL"
        verdict_int = 0
    elif "marginal" in states:
        verdict_label = "SO-SO"
        verdict_int = 1
    else:
        verdict_label = "PASS"
        verdict_int = 2

    # --- Text report ---
    report = build_report(verdict_label, shot_type, blur_mode, blur_var_threshold, checks)

    # --- Visual badge ---
    badge_image = render_badge(verdict_label, checks, shot_type, blur_mode, blur_var_threshold)

    return (badge_image, verdict_int, report, blur_score, sharpness_score,
            noise_score, clipping_score, entropy_score)


# ----------------------------------------------------------------
# Three-tier evaluation
# ----------------------------------------------------------------

def evaluate_upper(name, score, threshold, margin_factor):
    """Evaluate metric where lower is better (blur, noise, clipping)."""
    margin_limit = threshold * margin_factor
    if score <= threshold:
        state = "pass"
    elif score <= margin_limit:
        state = "marginal"
    else:
        state = "fail"
    return QualityCheck(
        name=name,
        score=score,
        threshold=threshold,
        condition=f"< {threshold:.3f}",
        margin=f"< {margin_limit:.3f}",
        state=state,
    )


def evaluate_lower(name, score, threshold, margin_factor):
    """Evaluate metric where higher is better (sharpness, entropy)."""
    margin_limit = threshold / margin_factor
    if score >= threshold:
        state = "pass"
    elif score >= margin_limit:
        state = "marginal"
    else:
        state = "fail"
    return QualityCheck(
        name=name,
        score=score,
        threshold=threshold,
        condition=f"> {threshold:.1f}",
        margin=f"> {margin_limit:.1f}",
        state=state,
    )


# ----------------------------------------------------------------
# Report
# ----------------------------------------------------------------

def build_report(verdict_label, shot_type, blur_mode, var_threshold, checks):
    lines = []
    if verdict_label == "PASS":
        lines.append("PASS — Suitable for dataset")
    elif verdict_label == "SO-SO":
        lines.append("SO-SO — Marginal quality, review recommended")
    else:
        lines.append("FAIL — Rejected")
    lines.append("-" * 44)
    lines.append(f"Shot: {shot_type} | Blur: {blur_mode} | Var_t: {var_threshold:.0f}")
    lines.append("")
    for c in checks:
        if c.state == "pass":
            tag = "[OK]  "
        elif c.state == "marginal":
            tag = "[~]   "
        else:
            tag = "[FAIL]"
        lines.append(f"  {tag} {c.name}: {c.score:.4f}  (need {c.condition})")
    return "\n".join(lines)


# ----------------------------------------------------------------
# Badge renderer
# ----------------------------------------------------------------

def render_badge(verdict_label, checks, shot_type, blur_mode, var_threshold):
    """Renders a green/yellow/red visual badge with metric breakdown."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 420, 310
    if verdict_label == "PASS":
        bg = (34, 120, 34)
    elif verdict_label == "SO-SO":
        bg = (180, 140, 20)
    else:
        bg = (170, 30, 30)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_meta = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_meta = ImageFont.load_default()

    # Verdict header
    icons = {"PASS": "\u2713", "SO-SO": "~", "FAIL": "\u2717"}
    header = f"{icons.get(verdict_label, '')}  {verdict_label}"
    draw.text((width // 2 - 60, 10), header, fill="white", font=font_large)

    # Shot info
    meta = f"[{shot_type}]"
    draw.text((width // 2 - 30, 50), meta, fill=(220, 220, 220), font=font_meta)

    # Separator
    draw.line([(15, 72), (width - 15, 72)], fill=(255, 255, 255, 120), width=1)

    # Metric rows
    y = 82
    for c in checks:
        state = c.state
        if state == "pass":
            icon = "\u2713"
            color = (180, 255, 180)
            icon_color = (80, 200, 80)
        elif state == "marginal":
            icon = "~"
            color = (255, 240, 160)
            icon_color = (220, 200, 60)
        else:
            icon = "\u2717"
            color = (255, 170, 170)
            icon_color = (255, 80, 80)

        draw.text((20, y), icon, fill=icon_color, font=font_small)
        draw.text((42, y), f"{c.name}:", fill=color, font=font_small)
        draw.text((210, y), f"{c.score:.3f}", fill="white", font=font_small)
        draw.text((290, y), f"({c.condition})", fill=(170, 170, 170), font=font_small)
        y += 40

    # Footer
    draw.text((width // 2 - 48, height - 20), "ImageQualityGate", fill=(200, 200, 200), font=font_meta)

    return img

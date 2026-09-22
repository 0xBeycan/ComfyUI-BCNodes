"""Image Quality Gate — blur / sharpness / noise / clipping / entropy verdict.

    BC_ImageQualityGate (Image Quality Gate)

Single-node quality control for AI-generated images, built for LoRA dataset
filtering: five metrics, each scored against a threshold scaled by a shot-type
preset, folded into a three-tier verdict (PASS / SO-SO / FAIL) with a badge
image and a text report. cv2 and PIL are imported inside the methods that use
them.
"""

import numpy as np
import torch


class ImageQualityGate:
    """Analyzes Blur, Sharpness, Noise, Clipping and Entropy; PASS (green),
    SO-SO (yellow) or FAIL (red)."""

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "shot_type": (["custom", "close-up", "medium", "wide / full-body"],),
                "blur_mode": (["full image", "center-weighted"],),
                "blur_threshold": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Max blur ratio. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "blur_var_threshold": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 5.0,
                        "max": 500.0,
                        "step": 5.0,
                        "tooltip": "Laplacian variance per block. AI-generated: 20-50, photos: 80-150.",
                    },
                ),
                "sharpness_threshold": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "min": 0.0,
                        "max": 500.0,
                        "step": 0.5,
                        "tooltip": "Min sharpness. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "noise_threshold": (
                    "FLOAT",
                    {
                        "default": 25.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Max noise level. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "clipping_threshold": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": "Max clipping ratio. Presets scale this; 'custom' uses as-is.",
                    },
                ),
                "entropy_threshold": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Min entropy in bits. Presets scale this; 'custom' uses as-is.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("badge", "verdict", "report", "blur_score", "sharpness_score", "noise_score", "clipping_score", "entropy_score")
    FUNCTION = "analyze"
    CATEGORY = "BCNodes/analysis"
    SEARCH_ALIASES = ["BCNodes", "image quality", "blur", "sharpness", "dataset filter"]

    def analyze(self, image, shot_type, blur_mode, blur_threshold, blur_var_threshold,
                sharpness_threshold, noise_threshold, clipping_threshold, entropy_threshold):
        import cv2

        profile = self.SHOT_PROFILES[shot_type]

        # Effective thresholds (custom = 1.0 multiplier = raw slider value)
        eff_blur_t = blur_threshold * profile["blur"]
        eff_sharp_t = sharpness_threshold * profile["sharpness"]
        eff_noise_t = noise_threshold * profile["noise"]
        eff_clip_t = clipping_threshold * profile["clipping"]
        eff_entropy_t = entropy_threshold * profile["entropy"]
        eff_block = profile["block_size"]

        # Tensor to numpy
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # --- Run all 5 analyses ---
        center_weighted = (blur_mode == "center-weighted")
        blur_score = self._blur_detection(gray, eff_block, center_weighted, blur_var_threshold)
        sharpness_score = self._sharpness_hybrid(gray)
        noise_score = self._noise_estimation(gray)
        clipping_score = self._clipping_analysis(gray)
        entropy_score = self._entropy_analysis(gray)

        # --- Three-tier evaluation per metric ---
        mf = self.MARGIN_FACTOR

        checks = [
            self._evaluate_upper(f"Blur [{blur_mode}]", blur_score, eff_blur_t, mf),
            self._evaluate_lower("Sharpness", sharpness_score, eff_sharp_t, mf),
            self._evaluate_upper("Noise", noise_score, eff_noise_t, mf),
            self._evaluate_upper("Clipping", clipping_score, eff_clip_t, mf),
            self._evaluate_lower("Entropy", entropy_score, eff_entropy_t, mf),
        ]

        # --- Overall verdict ---
        # verdict int: 0=FAIL, 1=SO-SO, 2=PASS (for Switch node routing)
        states = [c["state"] for c in checks]
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
        report = self._build_report(verdict_label, shot_type, blur_mode, blur_var_threshold, checks)

        # --- Visual badge ---
        badge_tensor = self._render_badge(verdict_label, checks, shot_type, blur_mode, blur_var_threshold)

        return (badge_tensor, verdict_int, report, blur_score, sharpness_score,
                noise_score, clipping_score, entropy_score)

    # ----------------------------------------------------------------
    # Three-tier evaluation
    # ----------------------------------------------------------------

    def _evaluate_upper(self, name, score, threshold, margin_factor):
        """Evaluate metric where lower is better (blur, noise, clipping)."""
        margin_limit = threshold * margin_factor
        if score <= threshold:
            state = "pass"
        elif score <= margin_limit:
            state = "marginal"
        else:
            state = "fail"
        return {
            "name": name,
            "score": score,
            "threshold": threshold,
            "condition": f"< {threshold:.3f}",
            "margin": f"< {margin_limit:.3f}",
            "state": state,
        }

    def _evaluate_lower(self, name, score, threshold, margin_factor):
        """Evaluate metric where higher is better (sharpness, entropy)."""
        margin_limit = threshold / margin_factor
        if score >= threshold:
            state = "pass"
        elif score >= margin_limit:
            state = "marginal"
        else:
            state = "fail"
        return {
            "name": name,
            "score": score,
            "threshold": threshold,
            "condition": f"> {threshold:.1f}",
            "margin": f"> {margin_limit:.1f}",
            "state": state,
        }

    # ----------------------------------------------------------------
    # Analysis methods
    # ----------------------------------------------------------------

    def _blur_detection(self, gray, block_size, center_weighted=False, var_threshold=30.0):
        """
        Localized Laplacian variance blur detection.

        In center-weighted mode, blocks are weighted by distance from center:
        - Center 30%: weight 3.0x (subject area)
        - Mid ring 30-60%: weight 1.0x
        - Outer 60%+: weight 0.3x (background)
        """
        import cv2

        h, w = gray.shape
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)

        weighted_blur = 0.0
        total_weight = 0.0

        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                block = gray[y:y + block_size, x:x + block_size]
                lap_var = cv2.Laplacian(block, cv2.CV_64F).var()
                is_blurry = 1.0 if lap_var < var_threshold else 0.0

                if center_weighted:
                    bx = x + block_size / 2.0
                    by = y + block_size / 2.0
                    dist = np.sqrt((bx - cx) ** 2 + (by - cy) ** 2) / max_dist
                    if dist < 0.3:
                        weight = 3.0
                    elif dist < 0.6:
                        weight = 1.0
                    else:
                        weight = 0.3
                else:
                    weight = 1.0

                weighted_blur += is_blurry * weight
                total_weight += weight

        return weighted_blur / max(total_weight, 1.0)

    def _sharpness_hybrid(self, gray):
        """Hybrid sharpness: Laplacian variance + Tenengrad."""
        import cv2

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_score = lap.var()
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        ten_score = np.mean(gx ** 2 + gy ** 2)
        return (lap_score + ten_score / 1000.0) / 2.0

    def _noise_estimation(self, gray):
        """Gaussian difference noise estimation."""
        import cv2

        blurred = cv2.GaussianBlur(gray.astype(np.float64), (5, 5), 0)
        diff = np.abs(gray.astype(np.float64) - blurred)
        return np.mean(diff)

    def _clipping_analysis(self, gray, threshold=5):
        """Highlight/shadow clipping ratio."""
        total = gray.size
        shadows = np.sum(gray <= threshold)
        highlights = np.sum(gray >= 255 - threshold)
        return (shadows + highlights) / total

    def _entropy_analysis(self, gray):
        """Shannon entropy in bits (0-8)."""
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------

    def _build_report(self, verdict_label, shot_type, blur_mode, var_threshold, checks):
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
            if c["state"] == "pass":
                tag = "[OK]  "
            elif c["state"] == "marginal":
                tag = "[~]   "
            else:
                tag = "[FAIL]"
            lines.append(f"  {tag} {c['name']}: {c['score']:.4f}  (need {c['condition']})")
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Badge renderer
    # ----------------------------------------------------------------

    def _render_badge(self, verdict_label, checks, shot_type, blur_mode, var_threshold):
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
            state = c["state"]
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
            draw.text((42, y), f"{c['name']}:", fill=color, font=font_small)
            draw.text((210, y), f"{c['score']:.3f}", fill="white", font=font_small)
            draw.text((290, y), f"({c['condition']})", fill=(170, 170, 170), font=font_small)
            y += 40

        # Footer
        draw.text((width // 2 - 48, height - 20), "ImageQualityGate", fill=(200, 200, 200), font=font_meta)

        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "BC_ImageQualityGate": ImageQualityGate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ImageQualityGate": "Image Quality Gate",
}

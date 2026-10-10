"""PreFlight Calibrate's tables: the feedback store (predictions joined with real outcomes) as two text tables.

It only reads: it modifies nothing and tunes nothing. Rule edits are manual and versioned through
``rules.ENGINE_VERSION``; since every prediction record carries the engine_version that produced it,
cross-version comparison is free.

Outcomes with no matching prediction, and predictions with no outcome, are skipped: an unscored prediction
costs nothing. The join itself is feedback.load_joined.
"""

from collections import Counter

# Map outcomes and verdicts onto the same 0/1/2 severity scale so they compare.
_RESULT_RANK = {"clean": 0, "demoted": 1, "removed": 2}
_STATUS_RANK = {"OK": 0, "RISK": 1, "BLOCK": 2}
_PLATFORMS = ("instagram", "tiktok", "x")
_PLAT_SHORT = {"instagram": "IG", "tiktok": "TT", "x": "X"}


def _platform_table(joined):
    """For each platform: how often the real outcome fell inside the predicted
    [best, worst] range (consistent), above worst (under-predicted — the
    dangerous kind), or below best (over-predicted — lost reach on safe content).
    """
    lines = ["", "== Per-platform calibration ==",
             "(outcome vs predicted [best,worst] range; UNKNOWN verdicts skipped)",
             ""]
    header = "%-10s %8s %11s %16s %14s" % (
        "platform", "scored", "consistent", "under-predicted", "over-predicted")
    lines.append(header)
    lines.append("-" * len(header))
    for platform in _PLATFORMS:
        consistent = under = over = 0
        for pred in joined:
            result = pred.get("outcomes", {}).get(platform)
            if result not in _RESULT_RANK:
                continue
            verdict = pred.get("verdicts", {}).get(platform, {})
            best = _STATUS_RANK.get(verdict.get("best"))
            worst = _STATUS_RANK.get(verdict.get("worst"))
            if best is None or worst is None:
                continue  # UNKNOWN / malformed — not comparable
            rank = _RESULT_RANK[result]
            if rank > worst:
                under += 1
            elif rank < best:
                over += 1
            else:
                consistent += 1
        scored = consistent + under + over
        lines.append("%-10s %8d %11d %16d %14d" % (
            _PLAT_SHORT[platform], scored, consistent, under, over))
    return lines


def _rule_table(joined):
    """For each rule ID that ever fired: how many posts it fired on, and the
    outcome distribution (clean/demoted/removed) of those posts, per platform.
    This is the table that answers 'which rule is miscalibrated'.
    """
    fires = Counter()
    # rule -> platform -> Counter(result)
    dist = {}
    for pred in joined:
        outcomes = pred.get("outcomes", {})
        for rule in dict.fromkeys(pred.get("fired_rules", [])):  # de-dupe per post
            fires[rule] += 1
            per_plat = dist.setdefault(rule, {p: Counter() for p in _PLATFORMS})
            for platform, result in outcomes.items():
                if platform in _PLATFORMS and result in _RESULT_RANK:
                    per_plat[platform][result] += 1

    lines = ["", "== Per-rule outcomes ==",
             "(clean/demoted/removed counts among scored posts where the rule fired)",
             ""]
    header = "%-22s %6s   %s" % ("rule", "fires",
                                 "".join("%-12s" % ("%s c/d/r" % _PLAT_SHORT[p])
                                         for p in _PLATFORMS))
    lines.append(header)
    lines.append("-" * len(header))
    for rule, count in sorted(fires.items(), key=lambda kv: (-kv[1], kv[0])):
        cells = ""
        for platform in _PLATFORMS:
            c = dist[rule][platform]
            cells += "%-12s" % ("%d/%d/%d" % (c["clean"], c["demoted"], c["removed"]))
        lines.append("%-22s %6d   %s" % (rule, count, cells))
    return lines


def build_report(joined):
    """The calibration report of feedback.load_joined's `joined`: a heading, then the two tables."""
    total = len(joined)
    scored = sum(1 for p in joined if p.get("outcomes"))
    versions = Counter(p.get("engine_version", "?") for p in joined)
    lines = [
        "PreFlight calibration report",
        "============================",
        "predictions: %d   with at least one outcome: %d" % (total, scored),
        "engine_versions seen: %s" % (", ".join(
            "%s (%d)" % (v, n) for v, n in sorted(versions.items())) or "none"),
    ]
    if total == 0:
        lines.append("")
        lines.append("Store is empty — log some predictions and outcomes first.")
        return "\n".join(lines)
    lines += _platform_table(joined)
    lines += _rule_table(joined)
    return "\n".join(lines)

"""The Caption Audit card: a report dict drawn onto a fixed-size PIL canvas (render_card,
render_error_card), the data shaping behind it, fonts, geometry and palette.
"""

import textwrap

from .audit import package_version


# Severity labels as they appear in the report dict (caption_audit.constants.SEV_*).
SEV_CRITICAL = "CRITICAL"
SEV_WARNING = "WARNING"
SEV_INFO = "INFO"
SEV_INTENDED = "INTENDED"
SEV_EXPECTED = "EXPECTED"

# Severity order used when picking what to show first in the fixed-height table.
SEV_RANK = {SEV_CRITICAL: 0, SEV_WARNING: 1, SEV_INFO: 2, SEV_INTENDED: 3, SEV_EXPECTED: 4}


# --- geometry (px) --------------------------------------------------------

W = 1280
PAD = 28
ROW_H = 26
TABLE_TOP = 358          # y of the table header, everything above is fixed
TABLE_HEADER_H = 28
FOOTER_H = 202           # interpretation box + the 'reads captions only' note
EM = "\u2014"            # em dash: the placeholder for every missing value

# --- palette --------------------------------------------------------------

BG = (18, 20, 26)
PANEL = (26, 29, 37)
PANEL_HI = (33, 37, 47)
LINE = (48, 53, 66)
TEXT = (228, 232, 240)
DIM = (134, 143, 163)
FAINT = (92, 100, 118)

RED = (255, 96, 96)
AMBER = (255, 181, 84)
CYAN = (86, 196, 245)
BLUE = (128, 160, 255)
GREEN = (94, 214, 148)
MAGENTA = (214, 146, 255)

SEV_COLOR = {
    SEV_CRITICAL: RED,
    SEV_WARNING: AMBER,
    SEV_INFO: CYAN,
    SEV_INTENDED: BLUE,
    SEV_EXPECTED: FAINT,
}

MONO_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Monaco.ttf",
    "/Library/Fonts/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\Consolas.ttf",
    "C:\\Windows\\Fonts\\lucon.ttf",
)

_FONT_CACHE = {}


def font(size):
    """A monospace face at `size`, or Pillow's built-in bitmap font.

    Monospace matters here: the table is laid out in character cells, so the
    columns line up without measuring every string.
    """
    from PIL import ImageFont

    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    f = None
    for path in MONO_CANDIDATES:
        try:
            f = ImageFont.truetype(path, size)
            break
        except (OSError, ValueError):
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size=size)   # Pillow >= 10.1
        except TypeError:
            f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def char_w(f):
    """Advance width of one cell in a monospace face."""
    try:
        return f.getlength("M")
    except AttributeError:
        return f.getsize("M")[0]


def card_size(table_rows):
    """Canvas size for a given `table_rows`. The only thing that moves it."""
    rows = max(1, int(table_rows))
    return W, TABLE_TOP + TABLE_HEADER_H + rows * ROW_H + FOOTER_H


# --- small drawing helpers ------------------------------------------------

def _text(d, xy, s, f, fill=TEXT, bold=False):
    d.text(xy, s, font=f, fill=fill)
    if bold:  # no bold face is guaranteed to exist; a 0.6px smear is enough
        d.text((xy[0] + 0.6, xy[1]), s, font=f, fill=fill)


def _panel(d, box, fill=PANEL, outline=LINE, radius=8):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _clip(s, n):
    """Truncate to n cells with a trailing ellipsis. Never returns ''."""
    s = "" if s is None else str(s).replace("\n", " ")
    if not s:
        return EM
    return s if len(s) <= n else s[:max(1, n - 1)] + "\u2026"


def _wrap_fixed(s, width, lines):
    """Wrap to exactly `lines` lines of `width` cells, padding with ''.

    Fixed line count is what keeps the footer from changing the card height.
    """
    out = textwrap.wrap(s or "", width=width) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = _clip(out[-1] + " \u2026", width)
    return out + [""] * (lines - len(out))


def _pct(x):
    return "%.0f%%" % (100 * x)


def _class_words(rep):
    """'woman 100%, young man 0%' — each declared class word with its coverage.

    A class word is only a useful declaration if it is actually in the captions,
    so the card states the measurement next to the claim rather than echoing
    back what was typed into the widget.
    """
    cov = rep["trigger"].get("class_word_coverage")
    if cov:
        return ", ".join("%s %s" % (c["word"], _pct(c["coverage"])) for c in cov)
    return ", ".join(rep["trigger"]["class_words"]) or EM


def _look_at(files, cells):
    """'look at: a, b, c (+27 more)' fitted into one line of `cells`.

    The '+N more' count is the point of the line, so names are dropped until
    the whole thing fits rather than letting the tail get truncated away.
    """
    if not files:
        return ""
    for k in range(len(files), 0, -1):
        rest = len(files) - k
        s = ("look at: " + ", ".join(files[:k])
             + (" (+%d more)" % rest if rest else ""))
        if len(s) <= cells:
            return s
    return _clip("look at: %d caption file(s)" % len(files), cells)


# --- data shaping ---------------------------------------------------------

def table_rows_for(rep, limit):
    """The rows the card shows: worst first, flagged/declared only.

    Same selection the terminal report makes, then cut to the widget's fixed
    row count so the table height is a layout constant, not a data property.

    A term you declared is never cut. EXPECTED and INTENDED sort last and a
    declared term sitting at 0% sorts last among those, so the fixed row count
    would otherwise drop exactly the row that says a --class-word or --fuse you
    typed is nowhere in the captions.
    """
    def rank(r):
        return (SEV_RANK.get(r["severity"], 9), -r["df_ratio"])

    rows = sorted((r for r in rep["df"]["rows"]
                   if r["flagged"] or r["severity"] in (SEV_EXPECTED,
                                                        SEV_INTENDED)),
                  key=rank)
    # the card's height is fixed, so declared rows claim their slots rather
    # than extending the table - anything traded away is counted in the tail
    declared = [r for r in rows if r.get("declared")][:limit]
    rest = [r for r in rows if not r.get("declared")]
    shown = sorted(declared + rest[:max(0, limit - len(declared))], key=rank)
    return shown, max(0, len(rows) - len(shown))


def top_finding(rep):
    """The worst flagged term and its interpretation: (row, interp), or None.

    `analyze()` already stored one interpretation per CRITICAL/WARNING term,
    keyed 'view\\tterm'; `df.primary` is sorted worst-first.
    """
    flagged = [r for r in rep["df"]["primary"]
               if r["severity"] in (SEV_CRITICAL, SEV_WARNING)]
    if not flagged:
        return None
    r = flagged[0]
    return r, rep["interpretation"].get(r["view"] + "\t" + r["term"], {})


def relational_note(rep):
    """'not flagged: wearing 70% - relational, the bias is in what follows'.

    A term the function-word filter kept out of the table is still a real
    number, and 'nothing flagged' next to a hidden 70% would read as a
    contradiction to anyone who has looked at the captions.
    """
    rel = [r for r in rep["df"].get("suppressed", [])
           if r.get("suppressed_reason") == "relational word" and r["size"] == 1]
    if not rel:
        return ""
    return ("not flagged: %s — relational, so the bias would be in what "
            "follows, not in the word itself"
            % ", ".join("%s %s" % (r["term"], _pct(r["df_ratio"]))
                        for r in rel[:4]))


def verdict_of(rep):
    """(label, color, detail) — the same three-way verdict the CLI prints."""
    crit = rep["summary"]["critical"]
    warn = rep["summary"]["warning"]
    if crit:
        return "FAIL", RED, "%d CRITICAL, %d WARNING" % (crit, warn)
    if warn:
        return "PASS WITH WARNINGS", AMBER, "0 CRITICAL, %d WARNING" % warn
    return "PASS", GREEN, "no critical findings"


def _tiles(rep):
    """The ten fixed stat tiles: (label, value, sub, color).

    Every tile is always drawn; a value that does not exist shows an em dash
    rather than removing the tile and reflowing the row.
    """
    t = rep["trigger"]
    d = rep["distribution"]
    p = rep["pairing"]
    red = rep["redundancy"]

    cov = t["coverage"]
    variants = len(t["variants"])
    exact = len(red["exact"])
    near = len(red["near"])
    # a bare-trigger set is identical captions and a near-zero ratio on purpose;
    # the tiles say so instead of colouring the style as a defect
    bare = bool(rep["meta"].get("bare_trigger"))
    crit_terms = sum(1 for r in rep["df"]["primary"] if r["severity"] == SEV_CRITICAL)
    warn_terms = sum(1 for r in rep["df"]["primary"] if r["severity"] == SEV_WARNING)

    return [
        ("captions / images",
         "%d / %d" % (p["captions"], p["images"]),
         "bare-trigger set" if bare else
         ("caption-only mode" if p["caption_only_mode"] else "paired by basename"),
         TEXT if p["images"] else AMBER),
        ("trigger coverage", _pct(cov), "%d / %d captions" % (t["present"], t["total"]),
         GREEN if cov >= 1.0 else RED),
        ("trigger first token", _pct(t["first_token_pct"]),
         "position idx 0", GREEN if t["first_token_pct"] >= 1.0 else AMBER),
        ("spelling variants", "%d" % variants,
         "edit distance 1-2" if variants else "no near-misses",
         RED if variants else GREEN),
        ("flagged terms", "%d / %d" % (crit_terms, warn_terms),
         "critical / warning",
         RED if crit_terms else (AMBER if warn_terms else GREEN)),
        ("exact duplicates", "%d" % exact,
         "identical by design" if bare else "byte-identical groups",
         CYAN if bare and exact else (RED if exact else GREEN)),
        ("near duplicates", "%d" % near,
         "jaccard >= %.2f" % red["threshold"], AMBER if near else GREEN),
        ("caption length", "%d/%.0f/%d" % (d["min"], d["median"], d["max"]),
         "min / median / max", TEXT),
        ("type-token ratio", "%.3f" % d["type_token_ratio"],
         "the trigger repeated" if bare else
         "%d unique / %d tokens" % (d["unique_tokens"], d["total_tokens"]),
         TEXT if bare else (AMBER if 0 < d["type_token_ratio"] < 0.25 else TEXT)),
        ("hapax", "%d" % d["hapax"], "%s of vocabulary" % _pct(d["hapax_ratio"]), TEXT),
    ]


# --- the card -------------------------------------------------------------

def _canvas(table_rows):
    """The blank card for `table_rows`: (rows_n, width, height, img, d)."""
    from PIL import Image, ImageDraw

    rows_n = max(1, int(table_rows))
    width, height = card_size(rows_n)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    return rows_n, width, height, img, d


def _title(d, title):
    """The title in the top-left corner."""
    _text(d, (PAD, PAD - 4), title, font(25), TEXT, bold=True)


def _verdict_bar(d, y, right, label, color, detail):
    """The verdict panel at `y`: colour stripe, `label` in `color`, then `detail`."""
    _panel(d, (PAD, y, right, y + 46), fill=PANEL_HI, outline=color)
    d.rounded_rectangle((PAD, y, PAD + 6, y + 46), radius=3, fill=color)
    _text(d, (PAD + 20, y + 8), label, font(21), color, bold=True)
    _text(d, (PAD + 20 + char_w(font(21)) * (len(label) + 2), y + 14), detail,
          font(15), TEXT)


def render_card(rep, args, table_rows=12, title="CAPTION AUDIT"):
    """Draw the whole report onto one fixed-size canvas. Returns a PIL image."""
    rows_n, width, height, img, d = _canvas(table_rows)

    f_body = font(15)
    f_small = font(13)
    f_tile = font(19)
    f_micro = font(11)
    cw = char_w(f_body)
    cw_s = char_w(f_small)
    right = width - PAD

    # 1. title row
    _title(d, title)
    tool = "caption-audit %s" % package_version()
    _text(d, (right - char_w(f_small) * len(tool), PAD + 5), tool, f_small, FAINT)

    # 2. verdict bar
    label, color, detail = verdict_of(rep)
    y = PAD + 34
    _verdict_bar(d, y, right, label, color, detail)
    gate = "exit code %d" % (1 if rep["summary"]["critical"] else 0)
    _text(d, (right - cw_s * len(gate) - 16, y + 16), gate, f_small, DIM)

    # 3. meta lines
    m = rep["meta"]
    t = rep["trigger"]
    y += 60
    meta = [
        ("dir", _clip(m["dir"], 118)),
        ("trigger", "%s%s     class words: %s     fused (--fuse): %s"
         % (t["trigger"],
            "  [INFERRED]" if m["trigger_inferred"] else "",
            _class_words(rep),
            ", ".join(m["fuse"]) or EM)),
        ("thresholds", "critical >= %s   warning >= %s   info >= %s   "
                       "ngram-max %d   stopword filter %s"
         % (_pct(args.critical_threshold), _pct(args.warn_threshold),
            _pct(args.info_threshold), args.ngram_max,
            "on" if not args.no_stopwords else "off")),
    ]
    for key, val in meta:
        _text(d, (PAD, y), "%-11s" % key, f_small, FAINT)
        _text(d, (PAD + cw_s * 12, y), val, f_small,
              AMBER if "[INFERRED]" in val else DIM)
        y += 19

    # 4. stat tiles — two fixed rows of five
    tiles = _tiles(rep)
    cols, gap, tile_h = 5, 10, 62
    tile_w = (width - 2 * PAD - gap * (cols - 1)) / cols
    ty = 194
    for i, (tlabel, value, sub, tcolor) in enumerate(tiles):
        col, row = i % cols, i // cols
        x0 = PAD + col * (tile_w + gap)
        y0 = ty + row * (tile_h + gap)
        _panel(d, (x0, y0, x0 + tile_w, y0 + tile_h))
        _text(d, (x0 + 11, y0 + 8), _clip(tlabel.upper(), 22), f_micro, FAINT)
        _text(d, (x0 + 11, y0 + 23), _clip(value, 15), f_tile, tcolor, bold=True)
        _text(d, (x0 + 11, y0 + 45), _clip(sub, 24), f_micro, DIM)

    # 5. the flagged-term table
    rows, hidden = table_rows_for(rep, rows_n)
    y = TABLE_TOP
    _text(d, (PAD, y - 22), "FLAGGED TERMS  \u00b7  document frequency = captions "
          "containing the term at least once, not raw occurrences",
          f_micro, FAINT)

    # column cells, measured from the right so the numbers always line up
    x_sev = PAD + 12
    x_term = x_sev + int(10 * cw)
    x_p = right - int(9 * cw)
    x_dfp = x_p - int(8 * cw)
    x_df = x_dfp - int(6 * cw)
    x_view = x_df - int(17 * cw)
    term_cells = max(8, int((x_view - x_term - cw) / cw))

    d.rectangle((PAD, y, right, y + TABLE_HEADER_H), fill=PANEL)
    hy = y + 7
    _text(d, (x_sev, hy), "SEVERITY", f_small, DIM, bold=True)
    _text(d, (x_term, hy), "TERM", f_small, DIM, bold=True)
    _text(d, (x_view, hy), "VIEW", f_small, DIM, bold=True)
    _text(d, (x_df, hy), "%5s" % "DF", f_small, DIM, bold=True)
    _text(d, (x_dfp, hy), "%7s" % "DF%", f_small, DIM, bold=True)
    _text(d, (x_p, hy), "%8s" % "P(t|trg)", f_small, DIM, bold=True)
    y += TABLE_HEADER_H

    for i in range(rows_n):
        r = rows[i] if i < len(rows) else None
        if i % 2 == 0:
            d.rectangle((PAD, y, right, y + ROW_H), fill=(22, 25, 32))
        ry = y + 5
        if r is None:                       # keep the empty row, do not shrink
            for x, w in ((x_sev, 0), (x_term, 0), (x_view, 0),
                         (x_df, 5), (x_dfp, 7), (x_p, 8)):
                _text(d, (x, ry), "%*s" % (w, EM) if w else EM, f_body, (58, 63, 78))
            y += ROW_H
            continue

        sev = r["severity"]
        note = ""
        if not r["flagged"] and sev != SEV_EXPECTED:
            note = "  (%s)" % (r["suppressed_reason"] or "suppressed")
        elif r["subsumed_by"]:
            note = "  (part of '%s')" % _clip(r["subsumed_by"], 26)

        _text(d, (x_sev, ry), sev, f_body, SEV_COLOR.get(sev, TEXT),
              bold=sev == SEV_CRITICAL)
        _text(d, (x_term, ry), _clip(r["term"] + note, term_cells), f_body, TEXT)
        _text(d, (x_view, ry), _clip(r["view"], 16), f_body, DIM)
        _text(d, (x_df, ry), "%5d" % r["df"], f_body, TEXT)
        _text(d, (x_dfp, ry), "%6.1f%%" % (100 * r["df_ratio"]), f_body,
              SEV_COLOR.get(sev, TEXT))
        pg = r["p_given_trigger"]
        _text(d, (x_p, ry), "%8.2f" % pg, f_body, RED if pg >= 0.999 else TEXT)
        y += ROW_H

    tail = ("+%d more row(s) — raise table_rows" % hidden if hidden
            else "no further rows above the info threshold")
    _text(d, (x_sev, y + 6), tail, f_micro, FAINT)

    # 6. the question this tool refuses to answer
    y += 26
    _panel(d, (PAD, y, right, y + 126), fill=PANEL, outline=(70, 52, 92))
    inner = PAD + 16
    body_cells = int((right - inner - 20) / cw_s)
    found = top_finding(rep)
    short = [c for c in rep["trigger"].get("class_word_coverage", [])
             if c["coverage"] < 1.0]
    if found is None and short:
        # nothing is fused, but a word was declared that the captions do not
        # contain - saying 'nothing flagged' here would bury the one finding
        c = short[0]
        _text(d, (inner, y + 12),
              "DECLARED CLASS WORD MISSING FROM THE CAPTIONS  ·  %s  ·  %s"
              % (_clip(c["word"], 38), _pct(c["coverage"])),
              f_micro, AMBER)
        q_lines = _wrap_fixed(
            "'%s' appears in %d of %d captions. Either write it into the "
            "captions or drop it from 'class_words' - declared but absent, it "
            "claims a word this caption set does not actually contain."
            % (c["word"], c["present"], c["total"]), body_cells, 2)
        legend = ("A class word is what the trigger is a kind of. Declaring one "
                  "is what stops it being read as accidental fusion.")
        # at 0% every caption is 'missing' it, which names nothing worth opening
        look = _look_at(c["missing_files"], body_cells) if c["present"] else ""
        qcolor = AMBER
    elif found is None and rep["meta"].get("bare_trigger"):
        # 'nothing flagged' would read as a clean spread caption set, which this
        # is the opposite of: there is no caption text to spread in the first place
        _text(d, (inner, y + 12), "BARE-TRIGGER SET", f_micro, CYAN)
        q_lines = _wrap_fixed(
            "Every caption is the trigger and nothing else, so the captions are "
            "identical on purpose and the model reads every attribute off the "
            "pixels. Nothing in these images is separately promptable - that is "
            "the trade, not a defect.", body_cells, 2)
        legend = ("Duplicate and diversity findings are reported as INFO here. "
                  "One descriptive caption in the set ends the exemption.")
        look, qcolor = "", CYAN
    elif found is None:
        _text(d, (inner, y + 12), "NOTHING FLAGGED", f_micro, GREEN)
        q_lines = _wrap_fixed("No term sits at CRITICAL or WARNING. The caption "
                              "distribution is spread enough that nothing is "
                              "welded to the trigger.", body_cells, 2)
        legend, look, qcolor = relational_note(rep), "", DIM
    else:
        r, interp = found
        sev = r["severity"]
        _text(d, (inner, y + 12),
              "THE QUESTION THIS TOOL REFUSES TO ANSWER  \u00b7  %s  \u00b7  %s"
              "  \u00b7  df %s  \u00b7  P(t|trigger) %.2f"
              % (sev, _clip(r["term"], 38), _pct(r["df_ratio"]),
                 r["p_given_trigger"]),
              f_micro, SEV_COLOR.get(sev, MAGENTA))
        q_lines = _wrap_fixed(interp.get("question", ""), body_cells, 2)
        legend = ("yes \u2192 dataset problem (b): collect contrast data.     "
                  "no \u2192 caption problem (a): delete the word where it is "
                  "not the point.")
        look = _look_at(r["files"], body_cells)
        qcolor = MAGENTA

    ly = y + 34
    for line in q_lines:
        _text(d, (inner, ly), line, f_small, qcolor)
        ly += 20
    _text(d, (inner, ly), _clip(legend, body_cells) if legend else "",
          f_small, DIM)
    _text(d, (inner, ly + 20), _clip(look, body_cells) if look else "",
          f_small, FAINT)

    _text(d, (PAD, height - PAD - 14),
          "caption-audit reads .txt captions only \u2014 it never opens your "
          "images, so the answer above has to come from you.", f_micro, FAINT)
    return img


def render_error_card(message, directory="", table_rows=12):
    """Same canvas, same size, for a run that never produced a report."""
    _, width, height, img, d = _canvas(table_rows)
    f_small = font(13)
    cw_s = char_w(f_small)

    _title(d, "CAPTION AUDIT")
    y = PAD + 34
    _verdict_bar(d, y, width - PAD, "ERROR", RED, "the audit did not run")

    y += 60
    _panel(d, (PAD, y, width - PAD, height - PAD))
    inner = PAD + 16
    cells = int((width - 2 * inner - 8) / cw_s)
    y += 16
    _text(d, (inner, y), "dir", f_small, FAINT)
    _text(d, (inner + cw_s * 12, y), _clip(directory, cells - 12), f_small, DIM)
    y += 26
    for line in _wrap_fixed(message, cells, 4):
        _text(d, (inner, y), line, f_small, AMBER)
        y += 20
    y += 14
    for line in _wrap_fixed(
            "Point 'directory' at a folder holding .txt caption sidecars named "
            "after the images. If the images live elsewhere, set 'images_dir'; "
            "with no images anywhere the audit still runs in caption-only mode. "
            "If no trigger word can be inferred from the captions, set 'trigger' "
            "explicitly. Nothing was audited, so 'critical' is 1 — a caption set "
            "that could not be checked has not been cleared for training.",
            cells, 6):
        _text(d, (inner, y), line, f_small, DIM)
        y += 20
    return img

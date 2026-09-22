"""Caption Audit — run caption-audit over a dataset folder and show the card.

    BC_CaptionAudit (Caption Audit)

A caption set is a folder, not a string: images plus same-named `.txt`
sidecars. So the node takes a path, runs the audit in-process (no subprocess,
no temp caption files) and draws the result as a preview image inside the
node. `caption_audit` is driven entirely through its public surface:

    caption_audit.cli.analyze(directory, args)  -> the whole report dict
    caption_audit.report.render_term_report / to_json

`analyze` and the renderers read their settings off an argparse namespace, so
`build_args()` produces a plain object carrying exactly the fields they touch.

The card is a report dict drawn onto a fixed-size PIL canvas. Fixed size is
the point: the node re-renders on every queue, and a canvas that grew with the
number of findings would resize the node in the graph each run. So the
geometry depends on ONE input — `table_rows` — and on nothing in the data.

`caption_audit` and PIL are imported inside the functions that use them: the
package is a pip dependency, and a missing install must not take the rest of
the pack down with it.
"""

import os
import textwrap


def _ca():
    try:
        import caption_audit
        import caption_audit.cli  # noqa: F401  (submodule, not re-exported)
    except ImportError as exc:
        raise ImportError("Caption Audit needs the `caption-audit` package: pip install 'caption-audit>=1.3'") from exc
    return caption_audit


def _version():
    return getattr(_ca(), "__version__", "")


# Severity labels as they appear in the report dict (caption_audit.constants.SEV_*).
SEV_CRITICAL = "CRITICAL"
SEV_WARNING = "WARNING"
SEV_INFO = "INFO"
SEV_INTENDED = "INTENDED"
SEV_EXPECTED = "EXPECTED"

# Severity order used when picking what to show first in the fixed-height table.
SEV_RANK = {SEV_CRITICAL: 0, SEV_WARNING: 1, SEV_INFO: 2, SEV_INTENDED: 3, SEV_EXPECTED: 4}


# --- audit plumbing ---------------------------------------------------------

class AuditArgs:
    """An argparse.Namespace stand-in.

    `analyze()` and the renderers only ever *read* attributes, so a plain
    object with the same field names is a complete substitute — no argv
    round-trip, no subprocess, no temp files.
    """

    __slots__ = ("dir", "images_dir", "recursive", "trigger", "class_word",
                 "fuse", "critical_threshold", "warn_threshold",
                 "info_threshold", "ngram_max", "jaccard", "no_stopwords",
                 "list_files", "max_rows", "max_interpret", "no_color",
                 "format", "out", "compare", "positional_dir")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))

    def __repr__(self):
        return "AuditArgs(%s)" % ", ".join(
            "%s=%r" % (k, getattr(self, k)) for k in self.__slots__)


def split_terms(text):
    """'red scarf, studio lighting' -> ['red scarf', 'studio lighting'].

    Comma-separated because a term is itself allowed to contain spaces.
    """
    if not text:
        return []
    return [t.strip() for t in str(text).split(",") if t.strip()]


def resolve_dir(path):
    """Expand ~ and relative paths; '' means the ComfyUI process cwd."""
    return os.path.abspath(os.path.expanduser(str(path or "").strip() or os.getcwd()))


def build_args(directory, trigger="", images_dir="", class_words="", fuse="",
               critical_threshold=0.85, warn_threshold=0.60,
               info_threshold=0.35, ngram_max=3, jaccard=0.9,
               no_stopwords=False, recursive=False, max_rows=200):
    """Every field `analyze()` and the renderers read, and nothing else."""
    imgs = str(images_dir or "").strip()
    return AuditArgs(
        dir=directory,
        positional_dir=None,
        images_dir=resolve_dir(imgs) if imgs else None,
        recursive=bool(recursive),
        trigger=str(trigger or "").strip() or None,
        class_word=split_terms(class_words),
        fuse=split_terms(fuse),
        critical_threshold=float(critical_threshold),
        warn_threshold=float(warn_threshold),
        info_threshold=float(info_threshold),
        ngram_max=int(ngram_max),
        jaccard=float(jaccard),
        no_stopwords=bool(no_stopwords),
        # report-only knobs: generous, since the text/JSON outputs are the
        # complete record and the card does its own fixed-height trimming
        list_files=10,
        max_rows=int(max_rows),
        max_interpret=12,
        no_color=True,
        format="term",
        out=None,
        compare=None,
    )


def run_audit(directory, args):
    """analyze() with the thresholds validated the way the CLI validates them.

    Returns the report dict, or raises ValueError with a message meant to be
    read on the card.
    """
    if not (0 < args.warn_threshold <= args.critical_threshold <= 1.0):
        raise ValueError("need 0 < warn_threshold <= critical_threshold <= 1.0")
    if not (0 < args.info_threshold <= args.warn_threshold):
        raise ValueError("need 0 < info_threshold <= warn_threshold")
    if not os.path.isdir(directory):
        raise ValueError("not a directory: %s" % directory)
    if args.images_dir and not os.path.isdir(args.images_dir):
        raise ValueError("not a directory: %s" % args.images_dir)

    rep = _ca().cli.analyze(directory, args)
    if rep is None:
        raise ValueError("no .txt caption files in %s (or no trigger could be "
                         "inferred - set 'trigger')" % directory)
    return rep


def report_text(rep, args):
    """The plain-text terminal report, ANSI disabled."""
    from caption_audit.report import Palette, render_term_report
    return render_term_report(rep, Palette(False), args)


def report_json(rep):
    from caption_audit.report import to_json
    return to_json(rep)


def dir_fingerprint(directory, recursive=False):
    """(path, mtime, size) digest of the caption files under `directory`.

    Used by IS_CHANGED so editing captions re-runs the audit, while an
    untouched dataset still hits ComfyUI's execution cache.
    """
    import hashlib

    h = hashlib.sha256()
    directory = str(directory or "")
    if not os.path.isdir(directory):
        return directory
    try:
        if recursive:
            walker = ((dp, fn) for dp, _dn, fns in os.walk(directory) for fn in fns)
        else:
            walker = ((directory, fn) for fn in os.listdir(directory))
        for dirpath, fn in sorted(walker):
            if not fn.lower().endswith(".txt") or fn.startswith("."):
                continue
            p = os.path.join(dirpath, fn)
            st = os.stat(p)
            h.update(("%s|%d|%d;" % (p, st.st_mtime_ns, st.st_size)).encode("utf-8"))
    except OSError as exc:
        return "%s|error:%s" % (directory, exc)
    return h.hexdigest()


# --- PIL <-> ComfyUI IMAGE bridge -----------------------------------------

def pil_to_image(img):
    """PIL RGB image -> ComfyUI IMAGE tensor (1, H, W, 3), float32, [0,1]."""
    import numpy as np
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def save_temp_preview(img, prefix="caption_audit"):
    """Write the card into ComfyUI's temp dir and describe it for the UI.

    Returns the ``{"images": [...]}`` payload PreviewImage uses, so the card
    shows inside the node with no frontend extension of our own. Outside
    ComfyUI (folder_paths missing) the file still lands in the system temp
    directory and an empty payload is returned.
    """
    import tempfile
    import uuid

    try:
        import folder_paths
        temp_dir = folder_paths.get_temp_directory()
        in_comfy = True
    except Exception:
        temp_dir = tempfile.gettempdir()
        in_comfy = False

    os.makedirs(temp_dir, exist_ok=True)
    filename = "%s_%s.png" % (prefix, uuid.uuid4().hex[:12])
    path = os.path.join(temp_dir, filename)
    img.save(path, compress_level=4)
    if not in_comfy:
        return {"images": []}, path
    return {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}, path


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

def render_card(rep, args, table_rows=12, title="CAPTION AUDIT"):
    """Draw the whole report onto one fixed-size canvas. Returns a PIL image."""
    from PIL import Image, ImageDraw

    rows_n = max(1, int(table_rows))
    width, height = card_size(rows_n)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)

    f_title = font(25)
    f_body = font(15)
    f_small = font(13)
    f_tile = font(19)
    f_micro = font(11)
    cw = char_w(f_body)
    cw_s = char_w(f_small)
    right = width - PAD

    # 1. title row
    _text(d, (PAD, PAD - 4), title, f_title, TEXT, bold=True)
    tool = "caption-audit %s" % _version()
    _text(d, (right - char_w(f_small) * len(tool), PAD + 5), tool, f_small, FAINT)

    # 2. verdict bar
    label, color, detail = verdict_of(rep)
    y = PAD + 34
    _panel(d, (PAD, y, right, y + 46), fill=PANEL_HI, outline=color)
    d.rounded_rectangle((PAD, y, PAD + 6, y + 46), radius=3, fill=color)
    _text(d, (PAD + 20, y + 8), label, font(21), color, bold=True)
    _text(d, (PAD + 20 + char_w(font(21)) * (len(label) + 2), y + 14), detail,
          f_body, TEXT)
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
    from PIL import Image, ImageDraw

    rows_n = max(1, int(table_rows))
    width, height = card_size(rows_n)
    img = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(img)
    f_small = font(13)
    cw_s = char_w(f_small)

    _text(d, (PAD, PAD - 4), "CAPTION AUDIT", font(25), TEXT, bold=True)
    y = PAD + 34
    _panel(d, (PAD, y, width - PAD, y + 46), fill=PANEL_HI, outline=RED)
    d.rounded_rectangle((PAD, y, PAD + 6, y + 46), radius=3, fill=RED)
    _text(d, (PAD + 20, y + 8), "ERROR", font(21), RED, bold=True)
    _text(d, (PAD + 20 + char_w(font(21)) * 7, y + 14),
          "the audit did not run", font(15), TEXT)

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


# --- the node --------------------------------------------------------------

class CaptionAudit:
    CATEGORY = "BCNodes/analysis"
    SEARCH_ALIASES = ["BCNodes", "caption audit", "lora captions", "trigger word", "dataset"]
    FUNCTION = "audit"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("report_image", "report_text", "report_json",
                    "critical", "warning")
    DESCRIPTION = (
        "Audit a LoRA caption set for tokens fused with the trigger word. Point "
        "'directory' at a folder of images + same-named .txt captions; the audit "
        "runs there and the report card renders inside the node.\n\n"
        "The primary measure is document frequency — in how many captions a term "
        "appears at least once, across words, phrases and whole comma segments. A "
        "term near 100% cannot be prompted in or out at inference: the model "
        "cannot tell it apart from the trigger.\n\n"
        "This reads .txt files only, never the images. Every flagged term has two "
        "possible causes needing opposite fixes, so the card ends with the "
        "question you have to answer by looking at the pictures. Wire 'critical' "
        "into a gate to stop a training workflow before it starts.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Folder holding the caption set: images plus "
                               ".txt sidecars sharing each image's basename. "
                               "Empty = ComfyUI's working directory."}),
                "trigger": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "The trigger word this LoRA is supposed to own. "
                               "Leave empty to have it inferred from the "
                               "captions — the card marks that [INFERRED], and "
                               "an inferred guess is worth checking."}),
                "class_words": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated class words, e.g. 'woman, car'. "
                               "Shown as EXPECTED and never flagged: they are "
                               "supposed to be everywhere. Coverage is measured "
                               "and shown per word — one you declare but never "
                               "wrote into the captions is a WARNING, not a "
                               "silent pass."}),
                "fuse": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated attributes you WANT welded to "
                               "the trigger, e.g. 'red scarf'. Covers the "
                               "phrase and its fragments, shown as INTENDED, "
                               "excluded from the critical count. Keeps its row "
                               "even at 0%, so a --fuse that no longer matches "
                               "your captions is visible rather than inert."}),
                "critical_threshold": ("FLOAT", {"default": 0.85, "min": 0.05,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Document frequency at or above which a term "
                               "counts as fused with the trigger."}),
                "warn_threshold": ("FLOAT", {"default": 0.60, "min": 0.05,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Strong bias: will bleed into unrelated prompts."}),
                "info_threshold": ("FLOAT", {"default": 0.35, "min": 0.01,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Below this a term is not reported at all."}),
                "ngram_max": ("INT", {"default": 3, "min": 1, "max": 5,
                    "tooltip": "Longest phrase analysed. Phrases never cross a "
                               "comma, so tag lists produce no phantom n-grams."}),
                "no_stopwords": ("BOOLEAN", {"default": False,
                    "tooltip": "Off (default) hides function words like 'a' / "
                               "'with' from the flag list. On shows everything."}),
                "recursive": ("BOOLEAN", {"default": False,
                    "tooltip": "Descend into subdirectories."}),
                "table_rows": ("INT", {"default": 12, "min": 1, "max": 60,
                    "tooltip": "Rows in the card's term table. This is the only "
                               "input that changes the card's size — the canvas "
                               "is otherwise fixed, so the node does not resize "
                               "between runs. Unused rows show an em dash. Terms "
                               "you declared always keep a row here."}),
            },
            "optional": {
                "images_dir": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Where the images live, when captions are kept "
                               "in a separate folder. Empty = alongside the "
                               "captions; with no images anywhere the audit "
                               "runs in caption-only mode."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, directory="", recursive=False, **kwargs):
        # Captions are edited outside the graph, so widget values alone do not
        # say whether a re-run is needed. Fingerprint the .txt files instead:
        # an untouched dataset still hits the execution cache.
        return dir_fingerprint(resolve_dir(directory), recursive)

    def audit(self, directory, trigger, class_words, fuse, critical_threshold,
              warn_threshold, info_threshold, ngram_max, no_stopwords,
              recursive, table_rows, images_dir=""):
        resolved = resolve_dir(directory)
        args = build_args(
            resolved, trigger=trigger, images_dir=images_dir,
            class_words=class_words, fuse=fuse,
            critical_threshold=critical_threshold,
            warn_threshold=warn_threshold, info_threshold=info_threshold,
            ngram_max=ngram_max, no_stopwords=no_stopwords, recursive=recursive)

        try:
            rep = run_audit(resolved, args)
        except Exception as exc:  # never raise into the graph; show the failure
            img = render_error_card("%s: %s" % (type(exc).__name__, exc),
                                         resolved, table_rows)
            ui, _path = save_temp_preview(img)
            text = "caption-audit: %s" % exc
            # critical=1 so a gate wired to it stops the workflow: a caption set
            # that could not be audited has not been cleared for training.
            return {"ui": ui,
                    "result": (pil_to_image(img), text, "{}", 1, 0)}

        img = render_card(rep, args, table_rows=table_rows)
        ui, _path = save_temp_preview(img)
        return {
            "ui": ui,
            "result": (
                pil_to_image(img),
                report_text(rep, args),
                report_json(rep),
                int(rep["summary"]["critical"]),
                int(rep["summary"]["warning"]),
            ),
        }


NODE_CLASS_MAPPINGS = {"BC_CaptionAudit": CaptionAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_CaptionAudit": "Caption Audit"}

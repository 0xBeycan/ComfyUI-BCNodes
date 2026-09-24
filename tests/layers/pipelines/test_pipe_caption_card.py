"""Golden: the Caption Audit card (nodes/caption_audit.py, target pipelines/caption_audit/card.py).

- render_card pixels for six report fixtures x table_rows 0 / 1 / 12 / 60 (0 exercises the
  max(1, ...) clamp of the canvas block, C71), each in two font modes: MONO_CANDIDATES as found
  on this machine, and () (forces ImageFont.load_default(size)). The fixtures cover the four
  question-box branches (a flagged term, a declared class word missing from the captions with
  and without a "look at" line, a bare-trigger set, nothing flagged with the relational note),
  filler rows, the hidden-rows tail, [INFERRED], every tile colour rule, clipped terms and notes.
  One more card with a custom title (C60 _title).
- render_error_card for table_rows 0 / 1 / 12 / 60 with a message that wraps past four lines,
  and one with an empty directory, in both font modes.
- The data shaping as digests: table_rows_for, top_finding, relational_note, verdict_of, _tiles,
  _class_words per fixture; _look_at, _clip, _wrap_fixed, card_size, _pct over small tables.
- The font file the "found" mode uses (basename and md5), so a system font update names itself
  instead of showing up as changed card pixels.

The fixtures start from a real caption-audit 1.3.0 report of a five-caption placeholder set and
are shaped by hand for each branch. sys.modules["caption_audit"] is a stub (__version__ "1.3.0",
cli) so the version on the card is fixed; the font cache is emptied per case (WHERE _FONT_CACHE).
"""

import copy
import hashlib
import os
import sys
import types

import pytest

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'Pillow': '12.3.0',
}

GOLDEN = {
    'card/flagged/0/found': "('RGB', (1280, 614), '1706fb9c868658212f08a89f1b36f915')",
    'card/flagged/0/default': "('RGB', (1280, 614), 'c71810eff64481244e1a75f768a869e3')",
    'card/flagged/1/found': "('RGB', (1280, 614), '1706fb9c868658212f08a89f1b36f915')",
    'card/flagged/1/default': "('RGB', (1280, 614), 'c71810eff64481244e1a75f768a869e3')",
    'card/flagged/12/found': "('RGB', (1280, 900), 'cfc6a2964689c98cb994bd99e45a86b6')",
    'card/flagged/12/default': "('RGB', (1280, 900), 'e922cc4ed2414855568b4198fcf7a65b')",
    'card/flagged/60/found': "('RGB', (1280, 2148), '65f171021aae647fa4985309c7afef67')",
    'card/flagged/60/default': "('RGB', (1280, 2148), '7ac7bb1722167898c225f58170a0fa65')",
    'card/declared_missing/0/found': "('RGB', (1280, 614), '211d90944f1b98eb70b2a72692e1011e')",
    'card/declared_missing/0/default': "('RGB', (1280, 614), 'c0e5bffd869517757fcdf689589f681d')",
    'card/declared_missing/1/found': "('RGB', (1280, 614), '211d90944f1b98eb70b2a72692e1011e')",
    'card/declared_missing/1/default': "('RGB', (1280, 614), 'c0e5bffd869517757fcdf689589f681d')",
    'card/declared_missing/12/found': "('RGB', (1280, 900), 'c465dac3b88070cd71ea1ec5f355cddc')",
    'card/declared_missing/12/default': "('RGB', (1280, 900), 'ad29caa2a1aad06ecf0fe02c75998739')",
    'card/declared_missing/60/found': "('RGB', (1280, 2148), 'aa63c9f123c3fcbf3e095533707b20f3')",
    'card/declared_missing/60/default': "('RGB', (1280, 2148), '238cd12e9628f564e7b41d2da292bf0b')",
    'card/declared_missing_everywhere/0/found': "('RGB', (1280, 614), '2746566b2a8fd72a91b46069fd8be5fa')",
    'card/declared_missing_everywhere/0/default': "('RGB', (1280, 614), 'a1d4833309f40119d3f78613d845c0f6')",
    'card/declared_missing_everywhere/1/found': "('RGB', (1280, 614), '2746566b2a8fd72a91b46069fd8be5fa')",
    'card/declared_missing_everywhere/1/default': "('RGB', (1280, 614), 'a1d4833309f40119d3f78613d845c0f6')",
    'card/declared_missing_everywhere/12/found': "('RGB', (1280, 900), 'd1f2892367f0697b7ab4c3cb25df7811')",
    'card/declared_missing_everywhere/12/default': "('RGB', (1280, 900), 'c4b82b9372d5a5eba9c125c47cf1f50b')",
    'card/declared_missing_everywhere/60/found': "('RGB', (1280, 2148), '24a177a05bac71fa9a77e27e313b37a5')",
    'card/declared_missing_everywhere/60/default': "('RGB', (1280, 2148), '28f7c2b88677fcc03e5a20d5d02552e3')",
    'card/bare_trigger/0/found': "('RGB', (1280, 614), '8f6673fd4a83b5ff74d9d1b55556f600')",
    'card/bare_trigger/0/default': "('RGB', (1280, 614), 'fb6d3f21ceb1951529aa021efb62afbb')",
    'card/bare_trigger/1/found': "('RGB', (1280, 614), '8f6673fd4a83b5ff74d9d1b55556f600')",
    'card/bare_trigger/1/default': "('RGB', (1280, 614), 'fb6d3f21ceb1951529aa021efb62afbb')",
    'card/bare_trigger/12/found': "('RGB', (1280, 900), '1ed058d11ab19ad9c385eac07d981429')",
    'card/bare_trigger/12/default': "('RGB', (1280, 900), 'a731979d8ccba11fe67ed937321af0bd')",
    'card/bare_trigger/60/found': "('RGB', (1280, 2148), '286899966df1f26b769cfe3b5a836882')",
    'card/bare_trigger/60/default': "('RGB', (1280, 2148), 'bc2eac081242f71f7942c11f6f405751')",
    'card/nothing_flagged/0/found': "('RGB', (1280, 614), 'e8607f06834717b123e1ed907798bb55')",
    'card/nothing_flagged/0/default': "('RGB', (1280, 614), 'bdf7d6f50427478adfb697ef73241d82')",
    'card/nothing_flagged/1/found': "('RGB', (1280, 614), 'e8607f06834717b123e1ed907798bb55')",
    'card/nothing_flagged/1/default': "('RGB', (1280, 614), 'bdf7d6f50427478adfb697ef73241d82')",
    'card/nothing_flagged/12/found': "('RGB', (1280, 900), 'f7cc0ed29f26b96cd3b9a2127b19ff97')",
    'card/nothing_flagged/12/default': "('RGB', (1280, 900), '71241c13c2ec61fcff395b747d153d64')",
    'card/nothing_flagged/60/found': "('RGB', (1280, 2148), '8cc3d65e2928d990edafec9bba98fde7')",
    'card/nothing_flagged/60/default': "('RGB', (1280, 2148), '4e6ff76ee82f496dfc3990c209b7154f')",
    'card/inferred_long/0/found': "('RGB', (1280, 614), '5a263dff6a770bd57f68c7179b146975')",
    'card/inferred_long/0/default': "('RGB', (1280, 614), '360204c03d4734e906616a21a136adc9')",
    'card/inferred_long/1/found': "('RGB', (1280, 614), '5a263dff6a770bd57f68c7179b146975')",
    'card/inferred_long/1/default': "('RGB', (1280, 614), '360204c03d4734e906616a21a136adc9')",
    'card/inferred_long/12/found': "('RGB', (1280, 900), 'd69c1f7f09002d7e89b0196408d2d6d3')",
    'card/inferred_long/12/default': "('RGB', (1280, 900), '9b1973827f61fa87d525ceffccd869a8')",
    'card/inferred_long/60/found': "('RGB', (1280, 2148), 'a8fa3462a29fb69886d8806ffa333ece')",
    'card/inferred_long/60/default': "('RGB', (1280, 2148), 'b378f2d5c658096851703a10cae73e85')",
    'card/custom_title/found': "('RGB', (1280, 900), '5d2f70a6d4054b60588e02d530b05e53')",
    'card/custom_title/default': "('RGB', (1280, 900), '8a505425ac89d1038c687c121f77c052')",
    'error_card/0/found': "('RGB', (1280, 614), '19d08bb1e1218becbef28ca9fd3ab06b')",
    'error_card/0/default': "('RGB', (1280, 614), '13a50f8fbea290054da2044632ed022c')",
    'error_card/1/found': "('RGB', (1280, 614), '19d08bb1e1218becbef28ca9fd3ab06b')",
    'error_card/1/default': "('RGB', (1280, 614), '13a50f8fbea290054da2044632ed022c')",
    'error_card/12/found': "('RGB', (1280, 900), 'd92f8110860ff1a5945f3a4a56c1bb8d')",
    'error_card/12/default': "('RGB', (1280, 900), '7de5c049f05af77620952bf82e86fc48')",
    'error_card/60/found': "('RGB', (1280, 2148), '83a456d0a733c949c864f90f47460135')",
    'error_card/60/default': "('RGB', (1280, 2148), '45e428ff4e330f34301c075b888fe91c')",
    'error_card/no_directory/found': "('RGB', (1280, 900), '50404025863b1fb20c27b1d8fb1e7e0d')",
    'error_card/no_directory/default': "('RGB', (1280, 900), '4b190528058551a790801bf2aa18ddb1')",
    'shaping/flagged': '5b4351b504e7f17253aaef12a1ab26c5',
    'shaping/declared_missing': '9a0bf71f07c39295961ffd162aaf27af',
    'shaping/declared_missing_everywhere': '20a0ff6b1658f9b863a65b47714d4070',
    'shaping/bare_trigger': '37d835d69949f9bf34db9adae675023e',
    'shaping/nothing_flagged': '7e017a37298217dcb0355400cbf6bb11',
    'shaping/inferred_long': 'dcbe31fa93ad63f77b1c76ec798542d9',
    'helpers': '20b82ce872063792a59b6850febc1fae',
}

FONT = {
    'mono_font': ('Menlo.ttc', 'ec05dfca6821d85440da0899493002c3'),
}

WHERE = Where({
    "MONO_CANDIDATES": "pipelines.caption_audit.card:MONO_CANDIDATES",
    "_FONT_CACHE": "pipelines.caption_audit.card:_FONT_CACHE",
    "render_card": "pipelines.caption_audit.card:render_card",
    "render_error_card": "pipelines.caption_audit.card:render_error_card",
    "table_rows_for": "pipelines.caption_audit.card:table_rows_for",
    "top_finding": "pipelines.caption_audit.card:top_finding",
    "relational_note": "pipelines.caption_audit.card:relational_note",
    "verdict_of": "pipelines.caption_audit.card:verdict_of",
    "_tiles": "pipelines.caption_audit.card:_tiles",
    "_class_words": "pipelines.caption_audit.card:_class_words",
    "_look_at": "pipelines.caption_audit.card:_look_at",
    "_clip": "pipelines.caption_audit.card:_clip",
    "_wrap_fixed": "pipelines.caption_audit.card:_wrap_fixed",
    "card_size": "pipelines.caption_audit.card:card_size",
    "_pct": "pipelines.caption_audit.card:_pct",
})

TABLE_ROWS = [0, 1, 12, 60]
FONT_MODES = ["found", "default"]
FILES = ["img_01", "img_02", "img_03", "img_04", "img_05"]
QUESTION = ("Is '%s' actually visually present in all %d of those images? If yes -> dataset problem (b). "
            "If no -> caption problem (a).")


def _row(term, view, size, df, severity, flagged, reason=None, subsumed=None, declared=False, n=5, p=None, files=None):
    return {"term": term, "view": view, "size": size, "df": df, "df_ratio": df / n,
            "p_given_trigger": df / n if p is None else p, "severity": severity, "flagged": flagged,
            "suppressed_reason": reason, "subsumed_by": subsumed, "declared": declared,
            "files": FILES[:df] if files is None else files}


def _real():
    """The trimmed caption-audit 1.3.0 report of the placeholder set (trigger '<trigger>', class
    word 'person'), the keys the card reads."""
    rows = [
        _row("red scarf", "segment+bigram", 2, 5, "CRITICAL", True),
        _row("<trigger>", "segment", 1, 5, "CRITICAL", True),
        _row("red", "unigram", 1, 5, "CRITICAL", True, subsumed="red scarf"),
        _row("scarf", "unigram", 1, 5, "CRITICAL", True, subsumed="red scarf"),
        _row("term_a", "segment+unigram", 1, 3, "WARNING", True, files=["img_01", "img_02", "img_05"]),
        _row("a", "unigram", 1, 4, "WARNING", False, reason="function word", subsumed="a person"),
        _row("person", "unigram", 1, 5, "EXPECTED", False, reason="declared trigger/class word", declared=True),
        _row("trigger", "unigram", 1, 5, "EXPECTED", False, reason="declared trigger/class word", declared=True),
        _row("a person", "bigram", 2, 4, "EXPECTED", False, reason="declared trigger/class word"),
        _row("a person", "segment", 2, 3, "EXPECTED", False, reason="declared trigger/class word"),
        _row("person", "segment", 1, 1, "EXPECTED", False, reason="declared trigger/class word", declared=True, files=["img_05"]),
    ]
    return {
        "meta": {"dir": "/path/to/dataset", "trigger_inferred": False, "bare_trigger": False, "fuse": []},
        "pairing": {"captions": 5, "images": 0, "caption_only_mode": True},
        "trigger": {"trigger": "<trigger>", "present": 5, "total": 5, "coverage": 1.0, "first_token_pct": 1.0,
                    "variants": [], "class_words": ["person"],
                    "class_word_coverage": [{"word": "person", "tokens": ["person"], "present": 5, "total": 5,
                                             "coverage": 1.0, "missing_files": []}]},
        "df": {"rows": rows, "primary": [rows[0], rows[1], rows[4]], "suppressed": []},
        "redundancy": {"exact": [], "near": [], "threshold": 0.9},
        "distribution": {"min": 5, "median": 7, "max": 8, "type_token_ratio": 0.36363636363636365, "unique_tokens": 12,
                         "total_tokens": 33, "hapax": 6, "hapax_ratio": 0.5},
        "interpretation": {"segment+bigram\tred scarf": {"question": QUESTION % ("red scarf", 5)},
                           "segment\t<trigger>": {"question": QUESTION % ("<trigger>", 5)},
                           "segment+unigram\tterm_a": {"question": QUESTION % ("term_a", 3)}},
        "summary": {"critical": 2, "warning": 1},
    }


def _not_flagged(rep):
    """rep with no CRITICAL / WARNING term left: flagged rows become INFO or EXPECTED ones."""
    rows = [r for r in rep["df"]["rows"] if r["severity"] == "EXPECTED"]
    rows.append(_row("studio light", "segment", 2, 2, "INFO", True))
    rep["df"] = {"rows": rows, "primary": [rows[-1]], "suppressed": rep["df"]["suppressed"]}
    rep["interpretation"] = {}
    rep["summary"] = {"critical": 0, "warning": 0}
    return rep


def _declared_missing(present):
    rep = _not_flagged(_real())
    missing = FILES[present:]
    rep["trigger"]["class_words"] = ["person", "term_z"]
    rep["trigger"]["class_word_coverage"].append(
        {"word": "term_z", "tokens": ["term_z"], "present": present, "total": 5, "coverage": present / 5,
         "missing_files": missing})
    rep["df"]["rows"].append(_row("term_z", "unigram", 1, present, "EXPECTED", False,
                                  reason="declared trigger/class word", declared=True))
    rep["pairing"] = {"captions": 5, "images": 5, "caption_only_mode": False}
    rep["summary"] = {"critical": 0, "warning": 1}
    return rep


def _bare_trigger():
    rep = _not_flagged(_real())
    rep["meta"]["bare_trigger"] = True
    rep["trigger"]["class_words"] = []
    rep["trigger"]["class_word_coverage"] = []
    rep["df"]["rows"] = [_row("<trigger>", "segment", 1, 5, "EXPECTED", False, reason="declared trigger/class word", declared=True)]
    rep["df"]["primary"] = []
    rep["pairing"] = {"captions": 5, "images": 5, "caption_only_mode": False}
    rep["redundancy"] = {"exact": [{"caption": "<trigger>", "files": FILES}], "near": [], "threshold": 0.9}
    rep["distribution"] = {"min": 1, "median": 1, "max": 1, "type_token_ratio": 0.2, "unique_tokens": 1,
                           "total_tokens": 5, "hapax": 0, "hapax_ratio": 0.0}
    return rep


def _nothing_flagged():
    rep = _not_flagged(_real())
    rep["pairing"] = {"captions": 5, "images": 3, "caption_only_mode": False}
    rep["df"]["suppressed"] = [
        {"term": term, "size": size, "df_ratio": ratio, "suppressed_reason": reason}
        for term, size, ratio, reason in [("wearing", 1, 0.7, "relational word"), ("with", 1, 0.6, "relational word"),
                                          ("in front of", 3, 0.6, "relational word"), ("the", 1, 0.9, "function word"),
                                          ("holding", 1, 0.4, "relational word"), ("near", 1, 0.35, "relational word"),
                                          ("under", 1, 0.35, "relational word")]]
    return rep


def _inferred_long():
    """Every tile rule on its warning side, [INFERRED], a WARNING as the top finding with P(t|trg)
    past 0.999, many rows (hidden tail), long terms, notes and file lists."""
    rep = _real()
    n = 40
    names = [f"img_{i:02d}" for i in range(1, n + 1)]
    rows = [
        _row("an extraordinarily long placeholder phrase term_a term_b term_c term_d", "segment", 9, 38, "WARNING",
             True, n=n, p=0.9995, files=names[:38]),
        _row("red scarf", "segment+bigram", 2, 36, "CRITICAL", True, n=n, files=names[:36]),
        _row("term_q", "unigram", 1, 36, "CRITICAL", True, subsumed="a very long subsuming phrase for term_q and more",
             n=n, files=names[:36]),
    ]
    for i in range(60):
        severity = ["WARNING", "INFO", "CRITICAL"][i % 3]
        rows.append(_row(f"term_{i:02d}", "unigram" if i % 2 else "segment", 1, 30 - i // 3, severity, i % 5 != 0,
                         reason=None if i % 5 else "function word", n=n, files=names[:30 - i // 3]))
    rows += [
        _row("person", "unigram", 1, 0, "EXPECTED", False, reason="declared trigger/class word", declared=True, n=n, files=[]),
        _row("term_b", "segment", 1, 0, "INTENDED", False, reason="declared --fuse", declared=True, n=n, files=[]),
        _row("red scarf", "segment", 2, 40, "INTENDED", False, reason="declared --fuse", declared=True, n=n, files=names),
    ]
    rep["df"] = {"rows": rows, "primary": [rows[0], rows[1], rows[2]], "suppressed": []}
    rep["interpretation"] = {"segment\t" + rows[0]["term"]: {"question": " ".join(["Is this placeholder phrase really in every image?"] * 6)}}
    rep["meta"].update(trigger_inferred=True, fuse=["red scarf", "term_b"], dir="/path/to/dataset/" + "sub/" * 40)
    rep["trigger"].update(trigger="<trigger>", present=32, total=n, coverage=0.8, first_token_pct=0.6,
                          variants=[{"token": "<trigger>s", "distance": 1, "df": 2, "files": names[:2]}],
                          class_words=["person"],
                          class_word_coverage=[{"word": "person", "tokens": ["person"], "present": 0, "total": n,
                                                "coverage": 0.0, "missing_files": names}])
    rep["pairing"] = {"captions": n, "images": 0, "caption_only_mode": True}
    rep["redundancy"] = {"exact": [{"caption": "x", "files": names[:2]}, {"caption": "y", "files": names[2:4]}],
                         "near": [{"files": names[4:6]}], "threshold": 0.85}
    rep["distribution"] = {"min": 2, "median": 11.5, "max": 64, "type_token_ratio": 0.2, "unique_tokens": 90,
                           "total_tokens": 450, "hapax": 30, "hapax_ratio": 0.3333333333333333}
    rep["summary"] = {"critical": 3, "warning": 5}
    return rep


FIXTURES = {
    "flagged": _real,
    "declared_missing": lambda: _declared_missing(2),
    "declared_missing_everywhere": lambda: _declared_missing(0),
    "bare_trigger": _bare_trigger,
    "nothing_flagged": _nothing_flagged,
    "inferred_long": _inferred_long,
}

ARGS = {
    "defaults": types.SimpleNamespace(critical_threshold=0.85, warn_threshold=0.6, info_threshold=0.35, ngram_max=3, no_stopwords=False),
    "no_stopwords": types.SimpleNamespace(critical_threshold=0.9, warn_threshold=0.555, info_threshold=0.2, ngram_max=5, no_stopwords=True),
}

ERROR_MESSAGE = ("ValueError: " + "the audit could not read the placeholder folder /path/to/dataset because of a reason "
                 "that is long enough to wrap onto more than four lines of the card, " * 4).strip()


@pytest.fixture
def card(bcnodes, monkeypatch):
    """Sets the font mode: card(mode) empties the font cache and, for "default", the candidates."""
    ca = types.ModuleType("caption_audit")
    ca.__version__ = "1.3.0"
    ca.cli = types.ModuleType("caption_audit.cli")
    monkeypatch.setitem(sys.modules, "caption_audit", ca)
    monkeypatch.setitem(sys.modules, "caption_audit.cli", ca.cli)

    def mode(font_mode):
        WHERE.patch(monkeypatch, "_FONT_CACHE", {})
        if font_mode == "default":
            WHERE.patch(monkeypatch, "MONO_CANDIDATES", ())

    return mode


def test_font_file(bcnodes):
    """The first MONO_CANDIDATES entry that loads: what the "found" cards are drawn with."""
    check_env(ENV, "Pillow")
    from PIL import ImageFont

    found = None
    for path in WHERE["MONO_CANDIDATES"]:
        try:
            ImageFont.truetype(path, 13)
        except (OSError, ValueError):
            continue
        with open(path, "rb") as f:
            found = (os.path.basename(path), hashlib.md5(f.read()).hexdigest())
        break
    check(FONT, "mono_font", found)


@pytest.mark.parametrize("font_mode", FONT_MODES)
@pytest.mark.parametrize("table_rows", TABLE_ROWS)
@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_render_card(fixture, table_rows, font_mode, card):
    check_env(ENV, "Pillow")
    card(font_mode)
    rep = FIXTURES[fixture]()
    before = copy.deepcopy(rep)
    args = ARGS["no_stopwords" if fixture == "inferred_long" else "defaults"]
    img = WHERE["render_card"](rep, args, table_rows=table_rows)
    assert rep == before, "render_card changed the report"
    check(GOLDEN, f"card/{fixture}/{table_rows}/{font_mode}", repr((img.mode, img.size, digest(img))))


@pytest.mark.parametrize("font_mode", FONT_MODES)
def test_render_card_title(font_mode, card):
    check_env(ENV, "Pillow")
    card(font_mode)
    img = WHERE["render_card"](_real(), ARGS["defaults"], table_rows=12, title="PLACEHOLDER TITLE")
    check(GOLDEN, f"card/custom_title/{font_mode}", repr((img.mode, img.size, digest(img))))


@pytest.mark.parametrize("font_mode", FONT_MODES)
@pytest.mark.parametrize("table_rows", TABLE_ROWS)
def test_render_error_card(table_rows, font_mode, card):
    check_env(ENV, "Pillow")
    card(font_mode)
    img = WHERE["render_error_card"](ERROR_MESSAGE, "/path/to/dataset", table_rows)
    check(GOLDEN, f"error_card/{table_rows}/{font_mode}", repr((img.mode, img.size, digest(img))))


@pytest.mark.parametrize("font_mode", FONT_MODES)
def test_render_error_card_no_directory(font_mode, card):
    check_env(ENV, "Pillow")
    card(font_mode)
    img = WHERE["render_error_card"]("ValueError: need 0 < warn_threshold <= critical_threshold <= 1.0")
    check(GOLDEN, f"error_card/no_directory/{font_mode}", repr((img.mode, img.size, digest(img))))


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_data_shaping(fixture, card):
    check_env(ENV, "Pillow")
    rep = FIXTURES[fixture]()
    shaped = {
        "table_rows_for": [WHERE["table_rows_for"](rep, n) for n in TABLE_ROWS],
        "top_finding": WHERE["top_finding"](rep),
        "relational_note": WHERE["relational_note"](rep),
        "verdict_of": WHERE["verdict_of"](rep),
        "tiles": WHERE["_tiles"](rep),
        "class_words": WHERE["_class_words"](rep),
    }
    check(GOLDEN, f"shaping/{fixture}", digest(shaped))


def test_helpers(card):
    check_env(ENV, "Pillow")
    names = [f"img_{i:02d}" for i in range(1, 31)]
    helpers = {
        "look_at": [WHERE["_look_at"](files, cells) for files, cells in
                    [([], 50), (["img_01"], 50), (names, 60), (names, 20), (["x" * 100], 30), (names[:3], 36)]],
        "clip": [WHERE["_clip"](s, n) for s, n in
                 [(None, 5), ("", 5), ("abc", 5), ("abcde", 5), ("abcdef", 5), ("a\nb", 5), ("abcdef", 1), ("abcdef", 0),
                  (123456, 4)]],
        "wrap_fixed": [WHERE["_wrap_fixed"](s, w, n) for s, w, n in
                       [("", 10, 2), (None, 10, 2), ("short", 10, 2), ("word " * 12, 20, 2), ("word " * 12, 20, 4),
                        ("averyveryverylongsingleword" * 2, 15, 3)]],
        "card_size": [WHERE["card_size"](n) for n in (0, 1, 12, 60, -5, 2.9, "7")],
        "pct": [WHERE["_pct"](x) for x in (0, 0.005, 0.015, 0.5, 0.995, 1.0, 1.2345)],
    }
    check(GOLDEN, "helpers", digest(helpers))

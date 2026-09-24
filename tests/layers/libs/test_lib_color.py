"""Golden: the BiRefNet background colour parser (plan 8.2 row 15).

libs/color.py:parse_hex_color turns the BiRefNet node's Color background widget into
(r, g, b, a) in 0..1. Every value of the table is pinned: the
returned list (its type and exact floats) or the exception type and message.

Recorded with BCNODES_GOLDEN_RECORD=1 on FIXED_BASE; a move edits only WHERE.
"""

import pytest

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    "'#222222'": 'd9d19404211f62dfeb700f12b7d517fd',
    "'#11223380'": 'bbd0195d4ea8f14dc210a46ea6971dfa',
    "'#00000000'": '51206cef7d0a2446b29fd022760dccf3',
    "'#ffffff'": '2c85bdbff9bc7c4cf1f4423f66b4fce4',
    "'#FfFfFf'": '2c85bdbff9bc7c4cf1f4423f66b4fce4',
    "'#fff'": '2c85bdbff9bc7c4cf1f4423f66b4fce4',
    "'#0f0'": '176751c8caac2aa3f55404247581bb62',
    "'abc'": '0e763c84808a50e7bf74160ef6e9923b',
    "'a1b2c3'": '69769d1c3d24c467c0649c3c965df016',
    "'  #a1b2c3  '": '69769d1c3d24c467c0649c3c965df016',
    "'#a1b2c3d4'": 'b2da51fc3e0113a729f97cfb4559a7a6',
    "'##112233'": 'bbb4668535fe617b5ddd2650f05ac2be',
    '123456': 'cc66fca80ffc396c1eef180a1b885d53',
    "'#12'": 'cf3acbd7175521c3c70cee33ca958643',
    "'#1234'": 'ec00c980c898427fcd077ea2e9902d1b',
    "'#12345'": 'e21d45293a6e4d7d2e3f31f6e4fc3cd0',
    "'#1234567'": 'c667102cacdf29e73539697faafb83b7',
    "'#1122334455'": '56de964f7c8a714c8756a3c60761df16',
    "'red'": 'a49b29d14ea5e332e2875c7866b5e472',
    "'#gggggg'": 'b9ecf2a210d5d212937dcc15e621f815',
    "'#11 2233'": '134fb4886330001e56ad9c2483bd09dc',
    "'#-12345'": '7549744fd6d9961514b8acdd02ea25c2',
    "'#\\uff11\\uff12\\uff13\\uff14\\uff15\\uff16'": '9579d7eaca4af64efb506e0725872d10',
    "''": 'a16cad2042963ec8d61ea6b30133ee5f',
    "' '": 'ccb4dcf3be734b07709282c2cdebdde6',
    "'#'": 'e97e0f67cf70aa6ed5a2a0c6182feeb3',
    'None': '31470e9e31edde5b9d3a8174db491d2b',
    '0': 'ac5a60a69d28ef2ce4d520fb12fd7597',
    'False': '1ad0a3148d47b9890ee76a26f6722008',
}

WHERE = Where({
    "parse_hex_color": "libs.color:parse_hex_color",
})

VALUES = [
    "#222222", "#11223380", "#00000000", "#ffffff", "#FfFfFf", "#fff", "#0f0", "abc", "a1b2c3",
    "  #a1b2c3  ", "#a1b2c3d4", "##112233", 123456, "#12", "#1234", "#12345", "#1234567",
    "#1122334455", "red", "#gggggg", "#11 2233", "#-12345", "#\uff11\uff12\uff13\uff14\uff15\uff16",
    "", " ", "#", None, 0, False,
]


def _call(fn, value):
    try:
        out = fn(value)
        return {"value": out, "type": type(out).__name__}
    except Exception as e:
        return {"exception": (type(e).__name__, str(e))}


@pytest.mark.parametrize("value", VALUES, ids=[ascii(v) for v in VALUES])
def test_parse_hex_color(value, bcnodes):
    check_env(ENV)
    check(GOLDEN, ascii(value), digest(_call(WHERE["parse_hex_color"], value)))

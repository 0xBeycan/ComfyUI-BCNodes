"""Golden: Image Quality Gate evaluators and report (target pipelines/quality_gate.py).

- evaluate_upper (lower is better: blur, noise, clipping) and evaluate_lower (higher is better:
  sharpness, entropy) with the margin factor 1.4, for a score at the threshold, just past it, at
  the margin limit (computed as the code does: threshold * 1.4 / threshold / 1.4), just past
  that, and far on either side; the score as a Python float and as np.float64. Each returned
  check, a QualityCheck dataclass, is recorded with _golden.digest, which turns it into
  dataclasses.asdict, so the digest pins its field order, values and value types.
- build_report per verdict (PASS / SO-SO / FAIL), shot type, blur mode and var threshold (incl.
  the round-half-even of :.0f), its checks always produced by the WHERE evaluators.

The evaluators and the report are plain functions of pipelines/quality_gate.py, reached through
WHERE.
"""

import math

import numpy as np
import pytest

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'numpy': '2.5.3',
}

GOLDEN = {
    'evaluate_upper/Blur [full image]/0.4/far_better/float': 'bf92cf2269137d09ac899165e93f9028',
    'evaluate_upper/Blur [full image]/0.4/far_better/np.float64': '6887b1302c3f407b78b50f7eb1af4f35',
    'evaluate_upper/Blur [full image]/0.4/at_threshold/float': 'ad6f5a76950ce69207882d357c8d479c',
    'evaluate_upper/Blur [full image]/0.4/at_threshold/np.float64': '323c8ac01fff1b636913a713f9465faa',
    'evaluate_upper/Blur [full image]/0.4/past_threshold/float': 'c9d50e5cf07bdef68e1b556bbacc0fd0',
    'evaluate_upper/Blur [full image]/0.4/past_threshold/np.float64': '3b2c5a1a801045b4dd9f38409d729bd8',
    'evaluate_upper/Blur [full image]/0.4/at_margin/float': 'f8ff3fe1165b1de3034cb3fbd843d328',
    'evaluate_upper/Blur [full image]/0.4/at_margin/np.float64': '12c1e50683aa30fd98f320c735dd699e',
    'evaluate_upper/Blur [full image]/0.4/past_margin/float': 'd6717a57ab6700b8e6494dc20417a609',
    'evaluate_upper/Blur [full image]/0.4/past_margin/np.float64': 'b6997203fb2a7237d0ea989ae2cc2607',
    'evaluate_upper/Blur [full image]/0.4/far_worse/float': '559d94a51ddd9ad9f2e564b4f77dfec9',
    'evaluate_upper/Blur [full image]/0.4/far_worse/np.float64': '3dfc39f1386983664984ee08f78ba307',
    'evaluate_upper/Blur [center-weighted]/0.24/far_better/float': '5cc8fab41c97c1ef79fd71c92551bdb0',
    'evaluate_upper/Blur [center-weighted]/0.24/far_better/np.float64': '826959a53d241db95812925c840647e1',
    'evaluate_upper/Blur [center-weighted]/0.24/at_threshold/float': 'd4d4d858391832bcfda8788af7669de3',
    'evaluate_upper/Blur [center-weighted]/0.24/at_threshold/np.float64': 'ed3aef9c280fe46a6b0f035430b42e78',
    'evaluate_upper/Blur [center-weighted]/0.24/past_threshold/float': 'd8fdaec7f5f3f85469d17a8f2a039b5e',
    'evaluate_upper/Blur [center-weighted]/0.24/past_threshold/np.float64': '1201c7100569fc9b293a033ad07155b7',
    'evaluate_upper/Blur [center-weighted]/0.24/at_margin/float': '707a764c8d757f7f58b527f018322ac1',
    'evaluate_upper/Blur [center-weighted]/0.24/at_margin/np.float64': 'd446a2ac91146af1ceb9e704026ae76f',
    'evaluate_upper/Blur [center-weighted]/0.24/past_margin/float': 'e17cef6657c1a2590e63b332658d43e0',
    'evaluate_upper/Blur [center-weighted]/0.24/past_margin/np.float64': 'dbdac3d62e248c9d854f4386694655ab',
    'evaluate_upper/Blur [center-weighted]/0.24/far_worse/float': '81c1be58aff149fa277b06bc0a4b1d76',
    'evaluate_upper/Blur [center-weighted]/0.24/far_worse/np.float64': 'f34ee276e4bc7139bfccdde2deac4a5e',
    'evaluate_upper/Noise/25.0/far_better/float': '220d54effbe5f6c9f1f5b9af5f8dd644',
    'evaluate_upper/Noise/25.0/far_better/np.float64': 'e8a90fba2dcbbea1c120aba5d15baf12',
    'evaluate_upper/Noise/25.0/at_threshold/float': '112aaaa7e0462636e20d48667345a012',
    'evaluate_upper/Noise/25.0/at_threshold/np.float64': 'f9c3ae0426e23aacf971bdccda81afc2',
    'evaluate_upper/Noise/25.0/past_threshold/float': '198a26cb8a2813035e24c0fa29c47485',
    'evaluate_upper/Noise/25.0/past_threshold/np.float64': 'ed91715fcda22d64d98ca66e06aae830',
    'evaluate_upper/Noise/25.0/at_margin/float': 'b4880b756906be8b510914d895695fab',
    'evaluate_upper/Noise/25.0/at_margin/np.float64': '403a21aa0d99b58b260dc1487b9ff1c3',
    'evaluate_upper/Noise/25.0/past_margin/float': '937f8709c2b5b89738e62822cb49c7de',
    'evaluate_upper/Noise/25.0/past_margin/np.float64': 'c5b54af16a6d5720a104383105357aef',
    'evaluate_upper/Noise/25.0/far_worse/float': 'd04e844fb2e930b0bfa6e0402897842e',
    'evaluate_upper/Noise/25.0/far_worse/np.float64': 'b07b7ef9064a644a3cdd60c3ce43faeb',
    'evaluate_upper/Noise/32.5/far_better/float': 'ad148200d794ba0d9e55e5486582fa0d',
    'evaluate_upper/Noise/32.5/far_better/np.float64': 'b0d38b49ea64477f61196d0985f0c731',
    'evaluate_upper/Noise/32.5/at_threshold/float': '12d13915fca977c36922c0a536560dc0',
    'evaluate_upper/Noise/32.5/at_threshold/np.float64': '6289e4a1f91756ccd2b443f6df9a3b70',
    'evaluate_upper/Noise/32.5/past_threshold/float': '4cea97f223990ec383800c7b1e7992f6',
    'evaluate_upper/Noise/32.5/past_threshold/np.float64': '74c739001a9fb16313b23b68ffde33d0',
    'evaluate_upper/Noise/32.5/at_margin/float': 'f34cec762d9902196cca17b2b749ca60',
    'evaluate_upper/Noise/32.5/at_margin/np.float64': 'b65ace2b8ce17e69bc0920ca532bd6d2',
    'evaluate_upper/Noise/32.5/past_margin/float': '2f298c1f9b302b90c585760d7d5680e9',
    'evaluate_upper/Noise/32.5/past_margin/np.float64': 'e64ed960dbfd2e1ad8ac9eaa045a3ffa',
    'evaluate_upper/Noise/32.5/far_worse/float': 'f6affdab71391d5fb5c2df4fdeecbf07',
    'evaluate_upper/Noise/32.5/far_worse/np.float64': 'a9a1a908ef8651982b3c6a1cdde915ae',
    'evaluate_upper/Clipping/0.02/far_better/float': 'db6c50f200f703a43a98674950b7f1a4',
    'evaluate_upper/Clipping/0.02/far_better/np.float64': 'f41b2b42c3c6a5bad6170b706f70221d',
    'evaluate_upper/Clipping/0.02/at_threshold/float': '7a07809328f9c015f5ea7270b7b9cc13',
    'evaluate_upper/Clipping/0.02/at_threshold/np.float64': 'a32ac3a71d90eac1912d23a298ea4ecb',
    'evaluate_upper/Clipping/0.02/past_threshold/float': '952477618600d044a0d1beb4e165e2b8',
    'evaluate_upper/Clipping/0.02/past_threshold/np.float64': 'a8a5e6772fc87d0452379ca5ca68e107',
    'evaluate_upper/Clipping/0.02/at_margin/float': '3c29b820c6830c545fbe884f27ca054f',
    'evaluate_upper/Clipping/0.02/at_margin/np.float64': '2ff27721a191b509cd9371f2dbce5b5e',
    'evaluate_upper/Clipping/0.02/past_margin/float': 'd9a4d0c703a739f104fab9b23d37a808',
    'evaluate_upper/Clipping/0.02/past_margin/np.float64': 'a3e4097ccca82d27401d8ca7556dcf79',
    'evaluate_upper/Clipping/0.02/far_worse/float': 'b27c0e5d7440ba4117f43d3e66f6a887',
    'evaluate_upper/Clipping/0.02/far_worse/np.float64': 'ebfbfc764825d37f948471fda776552f',
    'evaluate_upper/Clipping/0.024/far_better/float': '207a4efe503fcd1fbf457973d8299449',
    'evaluate_upper/Clipping/0.024/far_better/np.float64': 'bd1042109f2c4e475d2bf2f7e0245727',
    'evaluate_upper/Clipping/0.024/at_threshold/float': 'bee43315a4bb28d2d8072a04fd294d7c',
    'evaluate_upper/Clipping/0.024/at_threshold/np.float64': 'c828211a1ed65a420f2f89745cc10f2c',
    'evaluate_upper/Clipping/0.024/past_threshold/float': '038eec9d3a97676edec5b7a2833b78d7',
    'evaluate_upper/Clipping/0.024/past_threshold/np.float64': '11f49132b150ba7cc0c8a2944ab486f1',
    'evaluate_upper/Clipping/0.024/at_margin/float': 'a769ed4b9fb9abfcf8e042285d7ef0fb',
    'evaluate_upper/Clipping/0.024/at_margin/np.float64': '6bf54ea831cfe55ee51e02cdebb99967',
    'evaluate_upper/Clipping/0.024/past_margin/float': '68caad26581f13de7b1e936b3c6e6e2c',
    'evaluate_upper/Clipping/0.024/past_margin/np.float64': '053b59159c227b6260206ec07c3e0040',
    'evaluate_upper/Clipping/0.024/far_worse/float': '1903778721efc3ec0b35e015e1e0bded',
    'evaluate_upper/Clipping/0.024/far_worse/np.float64': 'b67b8a1f8d4ee6d3af28f97b6731e7be',
    'evaluate_lower/Sharpness/15.0/far_better/float': '8baea038984fb77926d9dadf9b07f8fd',
    'evaluate_lower/Sharpness/15.0/far_better/np.float64': '9ee0c0d57cedfde202e1d48beb864651',
    'evaluate_lower/Sharpness/15.0/at_threshold/float': '302d88f66d983d8840be1c2d1f577454',
    'evaluate_lower/Sharpness/15.0/at_threshold/np.float64': 'a7d3e8d29cfeabd1ff258be4b2e8a1a1',
    'evaluate_lower/Sharpness/15.0/past_threshold/float': 'd2c8f2243c340339ab1fcbd12da14bc8',
    'evaluate_lower/Sharpness/15.0/past_threshold/np.float64': 'ce77f82bb857772144f842b96b64a40e',
    'evaluate_lower/Sharpness/15.0/at_margin/float': '130b3bec4660ada98d0d1d6774ddf429',
    'evaluate_lower/Sharpness/15.0/at_margin/np.float64': '2720a3772d6910d756f139f911eca8a9',
    'evaluate_lower/Sharpness/15.0/past_margin/float': '27baabbfc89f73b4b849c8f8a1410de8',
    'evaluate_lower/Sharpness/15.0/past_margin/np.float64': '2bee6df50649b2dc87428ab3feba9f1a',
    'evaluate_lower/Sharpness/15.0/far_worse/float': '0040f05f5962ed1018d06db9416d4dc2',
    'evaluate_lower/Sharpness/15.0/far_worse/np.float64': '178463e1b9ae12c1f84ef4529bb161b2',
    'evaluate_lower/Sharpness/22.5/far_better/float': 'd38129a0003129188006937aa3ec47df',
    'evaluate_lower/Sharpness/22.5/far_better/np.float64': '1b9d60bd51255b9fc7975617e3162f1b',
    'evaluate_lower/Sharpness/22.5/at_threshold/float': '36d0eb1ed663da9ac6fd2894fdd56292',
    'evaluate_lower/Sharpness/22.5/at_threshold/np.float64': 'a0d0a5d44f36ba3c16725c92dd09970f',
    'evaluate_lower/Sharpness/22.5/past_threshold/float': '65d271ac9989b2de143100247ee1d56a',
    'evaluate_lower/Sharpness/22.5/past_threshold/np.float64': '14fc37bae26e9e1cb0e259830e4edbc5',
    'evaluate_lower/Sharpness/22.5/at_margin/float': 'b32b701c23e82a5f43c2ca13f774539e',
    'evaluate_lower/Sharpness/22.5/at_margin/np.float64': 'fccdb7436c362a52861c0e8b231dfc9c',
    'evaluate_lower/Sharpness/22.5/past_margin/float': '8ce60cb4be781ed39425c7f2b3534b28',
    'evaluate_lower/Sharpness/22.5/past_margin/np.float64': '103f1b3e91299b997f1c69c276815b73',
    'evaluate_lower/Sharpness/22.5/far_worse/float': '402371e5036912511459fcc44fceeaee',
    'evaluate_lower/Sharpness/22.5/far_worse/np.float64': 'b46b6a1d5aec38e8a075116fe498da62',
    'evaluate_lower/Entropy/5.0/far_better/float': '723cdfd72660a105a60d63f1207a9b97',
    'evaluate_lower/Entropy/5.0/far_better/np.float64': '28688430fff63a4dec310fbe1b7f2abb',
    'evaluate_lower/Entropy/5.0/at_threshold/float': '3e699c8e820f7543c4777015fd0efebc',
    'evaluate_lower/Entropy/5.0/at_threshold/np.float64': '06a1cd1dc58526baafacd2ae7eb5ec1e',
    'evaluate_lower/Entropy/5.0/past_threshold/float': '10b97125c408343ee6b206af65a8e629',
    'evaluate_lower/Entropy/5.0/past_threshold/np.float64': '8c949178c04460c374a511ee79c6c75b',
    'evaluate_lower/Entropy/5.0/at_margin/float': '4785e77d16e3e76226c8410689c2c807',
    'evaluate_lower/Entropy/5.0/at_margin/np.float64': '9a48afe9746a648757da241dc82ce426',
    'evaluate_lower/Entropy/5.0/past_margin/float': '310e1e8db428d3cec802de2c0bfd459d',
    'evaluate_lower/Entropy/5.0/past_margin/np.float64': '68c65681f9e1537b7f91b4dd95d0536d',
    'evaluate_lower/Entropy/5.0/far_worse/float': 'ddf2f1fde4bba21e40cb58b1246d616b',
    'evaluate_lower/Entropy/5.0/far_worse/np.float64': '7fc64fb91f417c2a28d82422375bde73',
    'evaluate_lower/Entropy/4.25/far_better/float': '39215a3caecffcb24823aa3bad1c81dc',
    'evaluate_lower/Entropy/4.25/far_better/np.float64': '9d7a291f58a3bf9db5c6c39a541d138e',
    'evaluate_lower/Entropy/4.25/at_threshold/float': '7cf7ab3d8ed80a17f79d3f6a6fb29572',
    'evaluate_lower/Entropy/4.25/at_threshold/np.float64': '40902fc9abd432d1abb6e31d34f712dc',
    'evaluate_lower/Entropy/4.25/past_threshold/float': '1b97b8d6aaf0ab8306811d9e59467616',
    'evaluate_lower/Entropy/4.25/past_threshold/np.float64': '9e1f8ffb2c7de1b7bb436d4daf5863c0',
    'evaluate_lower/Entropy/4.25/at_margin/float': '47dc71c055d6ab53520fee5a00c8a8bd',
    'evaluate_lower/Entropy/4.25/at_margin/np.float64': '5cf31bf21ed1ad088fc5e733f8bf3c62',
    'evaluate_lower/Entropy/4.25/past_margin/float': '2b1dbd45cb0100a84ffba39b291b69d7',
    'evaluate_lower/Entropy/4.25/past_margin/np.float64': 'c0b6d001786c3d0c326d4f42e4afd995',
    'evaluate_lower/Entropy/4.25/far_worse/float': '1e039b80b70b24c4e72e6c9cb3c722ad',
    'evaluate_lower/Entropy/4.25/far_worse/np.float64': '85551f2fbead52d5e16769b66c892689',
    'report/pass': 'PASS — Suitable for dataset\n--------------------------------------------\nShot: custom | Blur: full image | Var_t: 30\n\n  [OK]   Blur [full image]: 0.2500  (need < 0.400)\n  [OK]   Sharpness: 120.6218  (need > 15.0)\n  [OK]   Noise: 2.2330  (need < 25.000)\n  [OK]   Clipping: 0.0000  (need < 0.020)\n  [OK]   Entropy: 7.2887  (need > 5.0)',
    'report/so_so': 'SO-SO — Marginal quality, review recommended\n--------------------------------------------\nShot: wide / full-body | Blur: center-weighted | Var_t: 80\n\n  [OK]   Blur [center-weighted]: 0.0000  (need < 0.400)\n  [OK]   Sharpness: 24073.2903  (need > 15.0)\n  [FAIL] Noise: 36.3246  (need < 25.000)\n  [OK]   Clipping: 0.0002  (need < 0.020)\n  [OK]   Entropy: 7.5918  (need > 5.0)',
    'report/so_so_lower': 'SO-SO — Marginal quality, review recommended\n--------------------------------------------\nShot: close-up | Blur: full image | Var_t: 28\n\n  [OK]   Blur [full image]: 0.1000  (need < 0.400)\n  [~]    Sharpness: 12.0000  (need > 15.0)\n  [OK]   Noise: 3.0000  (need < 25.000)\n  [OK]   Clipping: 0.0010  (need < 0.020)\n  [~]    Entropy: 4.0000  (need > 5.0)',
    'report/fail': 'FAIL — Rejected\n--------------------------------------------\nShot: medium | Blur: full image | Var_t: 32\n\n  [FAIL] Blur [full image]: 1.0000  (need < 0.400)\n  [FAIL] Sharpness: 0.0000  (need > 15.0)\n  [OK]   Noise: 0.0000  (need < 25.000)\n  [OK]   Clipping: 0.0000  (need < 0.020)\n  [FAIL] Entropy: -0.0000  (need > 5.0)',
    'report/fail_mixed': 'FAIL — Rejected\n--------------------------------------------\nShot: custom | Blur: center-weighted | Var_t: 500\n\n  [~]    Blur [center-weighted]: 0.5000  (need < 0.400)\n  [~]    Sharpness: 11.0000  (need > 15.0)\n  [~]    Noise: 30.0000  (need < 25.000)\n  [FAIL] Clipping: 0.5000  (need < 0.020)\n  [OK]   Entropy: 6.0000  (need > 5.0)',
}

WHERE = Where({
    "evaluate_upper": "pipelines.quality_gate:evaluate_upper",
    "evaluate_lower": "pipelines.quality_gate:evaluate_lower",
    "build_report": "pipelines.quality_gate:build_report",
})

MF = 1.4  # MARGIN_FACTOR, the value analyze() passes

# evaluator -> (check name, threshold) pairs: the widget defaults and some preset-scaled values
THRESHOLDS = {
    "evaluate_upper": [("Blur [full image]", 0.4), ("Blur [center-weighted]", 0.4 * 0.6), ("Noise", 25.0),
                       ("Noise", 25.0 * 1.3), ("Clipping", 0.02), ("Clipping", 0.02 * 1.2)],
    "evaluate_lower": [("Sharpness", 15.0), ("Sharpness", 15.0 * 1.5), ("Entropy", 5.0), ("Entropy", 5.0 * 0.85)],
}


def _scores(evaluator, threshold):
    """Boundary scores, in the direction a score gets worse for that evaluator."""
    worse = math.inf if evaluator == "evaluate_upper" else -math.inf
    margin = threshold * MF if evaluator == "evaluate_upper" else threshold / MF
    return {
        "far_better": threshold / 10 if evaluator == "evaluate_upper" else threshold * 10,
        "at_threshold": threshold,
        "past_threshold": math.nextafter(threshold, worse),
        "at_margin": margin,
        "past_margin": math.nextafter(margin, worse),
        "far_worse": threshold * 10 if evaluator == "evaluate_upper" else threshold / 10,
    }


def _evaluator_cases():
    for evaluator, pairs in THRESHOLDS.items():
        for name, threshold in pairs:
            for label in _scores(evaluator, threshold):
                for kind in ("float", "np.float64"):
                    yield evaluator, name, threshold, label, kind


@pytest.mark.parametrize("evaluator,name,threshold,label,kind", list(_evaluator_cases()))
def test_evaluator(evaluator, name, threshold, label, kind, bcnodes):
    check_env(ENV, "numpy")
    score = _scores(evaluator, threshold)[label]
    score = np.float64(score) if kind == "np.float64" else float(score)
    check(GOLDEN, f"{evaluator}/{name}/{threshold!r}/{label}/{kind}", digest(WHERE[evaluator](name, score, threshold, MF)))


def _checks(blur_mode, blur, sharp, noise, clip, entropy):
    """The five checks in analyze()'s order, through the WHERE evaluators, on the default
    thresholds; blur is a Python float as blur_detection returns it, the rest np.float64."""
    up, low = WHERE["evaluate_upper"], WHERE["evaluate_lower"]
    return [
        up(f"Blur [{blur_mode}]", float(blur), 0.4, MF),
        low("Sharpness", np.float64(sharp), 15.0, MF),
        up("Noise", np.float64(noise), 25.0, MF),
        up("Clipping", np.float64(clip), 0.02, MF),
        low("Entropy", np.float64(entropy), 5.0, MF),
    ]


# name -> (verdict label, shot type, blur mode, var threshold, (blur, sharp, noise, clip, entropy))
REPORTS = {
    "pass": ("PASS", "custom", "full image", 30.0, (0.25, 120.6218, 2.233, 0.0, 7.2887)),
    "so_so": ("SO-SO", "wide / full-body", "center-weighted", 80.0, (0.0, 24073.2903, 36.3246, 0.0002, 7.5918)),
    "so_so_lower": ("SO-SO", "close-up", "full image", 27.5, (0.1, 12.0, 3.0, 0.001, 4.0)),
    "fail": ("FAIL", "medium", "full image", 32.5, (1.0, 0.0, 0.0, 0.0, -0.0)),
    "fail_mixed": ("FAIL", "custom", "center-weighted", 500.0, (0.5, 11.0, 30.0, 0.5, 6.0)),
}


@pytest.mark.parametrize("name", list(REPORTS))
def test_build_report(name, bcnodes):
    check_env(ENV, "numpy")
    verdict, shot_type, blur_mode, var_threshold, scores = REPORTS[name]
    checks = _checks(blur_mode, *scores)
    check(GOLDEN, f"report/{name}", WHERE["build_report"](verdict, shot_type, blur_mode, var_threshold, checks))

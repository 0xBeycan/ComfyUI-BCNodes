"""Emulate's profiles of Detailer and Face Detailer (pipelines/process_monitor/profiles.py), worked out
by hand from pipelines/detailer; the graph helpers are the emulate test's.

  - Detailer: the image and mask outputs at the image's size and dtype, each only when linked; with the
    image linked, a float32 working frame, each mask's region (one byte per pixel of a crop smaller than
    resolution^2) and a region's work at most eight float32 images of resolution^2; with it unlinked,
    nothing sampled: the regions and one mask frame read; an unlinked mask made and dropped;
  - Face Detailer: both outputs made, an unlinked one dropped at return (transient); the working frame,
    the character mask's read when it is linked, then the larger of core SAM3_Detect's 1008 x 1008 copy
    and a region's work; the faces' masks not counted;
  - a resolution from a link: not counted.
"""

import pytest

import test_pipe_process_monitor_emulate as emulate
from test_pipe_process_monitor_emulate import N, loader, rows_of

H, W = 480, 640  # ref.png
FRAME, MASK, PIXELS = H * W * 3 * 4, H * W * 4, H * W
REGION = 8 * 3 * 4 * 1024 * 1024  # at resolution 1024
SAM3_COPY = 1008 * 1008 * 3 * 4


class Env(emulate.Env):
    TYPES = {**emulate.Env.TYPES, "BC_Detailer": ["IMAGE", "MASK"], "BC_FaceDetailer": ["IMAGE", "MASK"]}


SAMPLING = dict(model=["8", 0], vae=["7", 0], positive=["6", 0], negative=["6", 0], seed=0, steps=20, cfg=4.0, sampler_name="euler",
                scheduler="simple", denoise=0.5, refine_denoise=0.2, resolution=1024, context=3.0)


@pytest.fixture
def em(bcnodes):
    return bcnodes["pipelines.process_monitor.emulate"]


def prompt(node, linked):
    """`node` as "2" fed by ref.png ("9"), a consumer per linked output slot."""
    p = {"9": N("LoadImage", image="ref.png"), "2": node}
    for slot in linked:
        p[f"c{slot}"] = N("SomeCustomNode", x=["2", slot])
    return p


def row(em, node, linked):
    return rows_of(em, prompt(node, linked), Env(videos={"v.mp4": emulate.PORTRAIT}))["2"]


def detailer(**widgets):
    return N("BC_Detailer", image=["9", 0], mask=["9", 1], **dict(SAMPLING, **widgets))


def face_detailer(**inputs):
    return N("BC_FaceDetailer", image=["9", 0], **dict(SAMPLING, sam3_model="sam3.safetensors", threshold=0.5, min_face_height=64,
                                                       max_face_height=16384, face=True, mouth=False, eyes=True, **inputs))


def test_detailer(em):
    r = row(em, detailer(), (0, 1))
    assert r["status"] == "counted" and [o["shape"] for o in r["outputs"]] == [[1, H, W, 3], [1, H, W]]
    assert r["output_bytes"] == FRAME + MASK and r["transient"] == PIXELS + FRAME + REGION
    assert "upper bound" in r["note"] and "VAE" in r["note"]
    r = row(em, detailer(), (0,))  # the mask made whole and dropped
    assert r["output_bytes"] == FRAME and r["transient"] == PIXELS + MASK + FRAME + REGION
    r = row(em, detailer(), (1,))  # nothing sampled: the regions and one mask frame read
    assert r["output_bytes"] == MASK and r["transient"] == PIXELS + MASK + PIXELS and "nothing sampled" in r["note"]
    r = row(em, detailer(resolution=2048), (0, 1))
    assert r["transient"] == PIXELS + FRAME + 4 * REGION


def test_detailer_half_image(em):
    p = {"9": loader(frame_count="5"), "5": N("LoadImage", image="ref.png"),
         "2": N("BC_Detailer", image=["9", 0], mask=["5", 1], **SAMPLING), "c0": N("SomeCustomNode", x=["2", 0])}
    r = rows_of(em, p, Env(videos={"v.mp4": emulate.PORTRAIT}))["2"]
    h, w = 1280, 720
    assert r["outputs"][0]["shape"] == [5, h, w, 3] and r["output_bytes"] == 5 * h * w * 3 * 2  # float16, as the frames
    assert r["transient"] == 1 * min(h * w, 1024 * 1024) + 5 * h * w * 2 + h * w * 3 * 4 + REGION  # one mask, the dropped float16 mask


def test_face_detailer(em):
    r = row(em, face_detailer(), (0, 1))
    assert r["status"] == "counted" and r["output_bytes"] == FRAME + MASK and r["transient"] == FRAME + REGION
    assert "faces' masks" in r["note"] and "SAM 3" in r["note"]
    r = row(em, face_detailer(mask=["9", 1]), (0,))  # the mask output made and dropped; the character's mask read
    assert r["output_bytes"] == FRAME and r["transient"] == MASK + FRAME + MASK + PIXELS + REGION
    r = row(em, face_detailer(), (1,))  # the image made all the same
    assert r["output_bytes"] == MASK and r["transient"] == FRAME + FRAME + REGION
    r = row(em, face_detailer(resolution=256), (0, 1))  # a small region: SAM 3's copy is the larger
    assert r["transient"] == FRAME + SAM3_COPY


@pytest.mark.parametrize("node", [detailer(resolution=["9", 0]), face_detailer(resolution=["9", 0])])
def test_resolution_from_a_link_is_not_counted(em, node):
    r = row(em, node, (0, 1))
    assert r["status"] == "not counted" and "resolution comes from a link" in r["note"]

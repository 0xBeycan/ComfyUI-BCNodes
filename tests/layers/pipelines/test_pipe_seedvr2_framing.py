"""SeedVR2 Framing Downscale's flow (pipelines/seedvr2/framing.py) with a stand-in SAM 3 detector
that returns given face boxes:

  - the tallest face box / frame height picks the band: at least close_up_min_face -> close-up
    factor, at least medium_min_face -> medium factor, below -> far factor (the bounds belong to the
    upper band); no face -> no_face_factor, face fraction 0, a warning in the log;
  - the fraction is that of the tallest box over every face and every frame of the batch, a box cut
    to the frame first;
  - the detector gets the checkpoint and threshold given, the face prompt, no mask refinement, and
    at most FRAMES_PER_CHUNK frames per call;
  - medium_min_face not below close_up_min_face, and no image batch, are errors that say what to do.
"""

import logging

import pytest
import torch

DEFAULTS = dict(close_up_min_face=0.25, close_up_factor=0.5, medium_min_face=0.15, medium_factor=0.75, far_factor=1.0,
                no_face_factor=0.9)


class Detector:
    """Stands in for models/sam3 load + detect: frame i of the batch gets one box per (y, height) of
    faces[i], in pixels; records every call."""

    def __init__(self, faces):
        self.faces, self.calls, self.loaded, self.encoded = faces, [], [], []
        self.seen = 0

    def load(self, name):
        self.loaded.append(name)
        return "model", "clip"

    def text_condition(self, clip, text):
        self.encoded.append((clip, text))
        return f"{clip}:{text}"

    def detect(self, model, cond, image, threshold, refine_iterations=2):
        self.calls.append(dict(model=model, cond=cond, frames=image.shape[0], threshold=threshold,
                               refine_iterations=refine_iterations))
        boxes = [[dict(x=10.0, y=float(y), width=50.0, height=float(h), score=0.9) for y, h in self.faces[self.seen + i]]
                 for i in range(image.shape[0])]
        self.seen += image.shape[0]
        return torch.zeros(image.shape[0], image.shape[1], image.shape[2]), boxes


@pytest.fixture
def framing(bcnodes):
    return bcnodes["pipelines.seedvr2.framing"]


def run(framing, monkeypatch, faces, height=1000, **widgets):
    det = Detector(faces)
    monkeypatch.setattr(framing, "load", det.load)
    monkeypatch.setattr(framing, "detect", det.detect)
    monkeypatch.setattr(framing, "text_condition", det.text_condition)
    image = torch.zeros(len(faces), height, 600, 3)
    out = framing.downscale_factor(image, "sam3.safetensors", 0.4, **dict(DEFAULTS, **widgets))
    return out, det


@pytest.mark.parametrize("face_height, factor", [
    (400, 0.5),   # 0.40: close-up
    (250, 0.5),   # 0.25: close-up (the bound belongs to the close-up band)
    (249, 0.75),  # medium
    (150, 0.75),  # 0.15: medium (the bound belongs to the medium band)
    (149, 1.0),   # far
    (40, 1.0),    # far
])
def test_the_tallest_face_picks_the_band(framing, monkeypatch, face_height, factor):
    (got, fraction), _ = run(framing, monkeypatch, [[(100, face_height)]])
    assert got == factor and fraction == pytest.approx(face_height / 1000)


def test_the_factors_and_bounds_are_the_widgets(framing, monkeypatch):
    widgets = dict(close_up_min_face=0.5, close_up_factor=0.3, medium_min_face=0.4, medium_factor=0.6, far_factor=0.8)
    assert run(framing, monkeypatch, [[(0, 550)]], **widgets)[0] == (0.3, 0.55)
    assert run(framing, monkeypatch, [[(0, 450)]], **widgets)[0] == (0.6, 0.45)
    assert run(framing, monkeypatch, [[(0, 300)]], **widgets)[0] == (0.8, 0.3)


def test_no_face_gives_no_face_factor_and_a_log_line(framing, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        (got, fraction), _ = run(framing, monkeypatch, [[], []])
    assert (got, fraction) == (0.9, 0.0)
    assert "no face found in 2 frame(s), downscale_factor 0.9" in caplog.text


def test_the_band_is_logged(framing, monkeypatch, caplog):
    with caplog.at_level(logging.INFO):
        run(framing, monkeypatch, [[(100, 200)]])
    assert "tallest face 0.200 of the frame height (medium), downscale_factor 0.75" in caplog.text


def test_the_tallest_face_of_the_whole_batch(framing, monkeypatch):
    # two faces in frame 0, the tallest in frame 5 (the second detector call), one frame without a face
    faces = [[(100, 120), (500, 90)], [(0, 100)], [], [(10, 140)], [(10, 130)], [(300, 260)]]
    assert run(framing, monkeypatch, faces)[0] == (0.5, pytest.approx(0.26))


def test_a_box_is_cut_to_the_frame(framing, monkeypatch):
    # 300 tall, 100 of it above the frame; 280 tall, 80 of it below the frame
    assert run(framing, monkeypatch, [[(-100, 300)]])[0] == (0.75, pytest.approx(0.2))
    assert run(framing, monkeypatch, [[(800, 280)]])[0] == (0.75, pytest.approx(0.2))


def test_what_the_detector_is_asked(framing, monkeypatch):
    chunk = framing.FRAMES_PER_CHUNK
    _, det = run(framing, monkeypatch, [[(0, 100)]] * (2 * chunk + 1))
    assert det.loaded == ["sam3.safetensors"]
    assert [c["frames"] for c in det.calls] == [chunk, chunk, 1]
    assert all(c == dict(model="model", cond="clip:face:4", frames=c["frames"], threshold=0.4, refine_iterations=0)
               for c in det.calls)
    assert det.encoded == [("clip", "face:4")]  # the prompt encoded once


@pytest.mark.parametrize("medium", [0.25, 0.3])
def test_medium_bound_not_below_close_up_bound_says_what_to_do(framing, monkeypatch, medium):
    with pytest.raises(ValueError, match="medium_min_face .* must be below close_up_min_face"):
        run(framing, monkeypatch, [[(0, 100)]], medium_min_face=medium)


@pytest.mark.parametrize("image", [None, torch.zeros(0, 64, 64, 3), torch.zeros(64, 64, 3)])
def test_no_image_batch_says_what_to_do(framing, image):
    with pytest.raises(ValueError, match="SeedVR2 Framing Downscale: connect an image batch"):
        framing.downscale_factor(image, "sam3.safetensors", 0.5, **DEFAULTS)

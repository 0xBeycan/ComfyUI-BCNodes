"""Face Detailer's flow (pipelines/detailer/face.py), with a stand-in SAM 3 and a stand-in core pass (it
records each region and the model, and paints the region so a later detection can see it):

  - the fixed prompts; the checkpoint and the threshold handed to SAM 3;
  - the character: the largest face without a mask, the face the mask overlaps most with one, none when
    the mask overlaps no face; detailed first, with the main model; the others with model_no_lora, and
    not at all without it;
  - the size gate: a face under 64 px gets no pass and no eye detection (63 px none, 64 px both); a face
    one px over max_face_height skips the face pass while its eyes still run, at exactly it the pass runs;
    a face whose longer side is face_size or more skips the face pass, its eyes still run; the eyes skip
    their pass when their box's longer side is eye_size or more;
  - each pass its own settings: the face pass face_size, face_crop_size and denoise, the eye pass eye_size,
    eye_crop_size and eye_denoise (the crops' sampling sizes worked out by hand from region.plan);
  - the eyes: found on the face's box of the frame the face pass left, cut to the face's mask (a part
    outside it dropped), both eyes one region on one crop; one eye alone when only one is found;
  - the passes in order face, eyes; the mask output the union of the regions detailed.
"""

import importlib

import pytest
import torch

H = W = 200
FACES = {  # name: (rows, cols) of its mask
    "a": (slice(20, 120), slice(20, 100)),   # 100 px tall, the largest
    "b": (slice(30, 100), slice(130, 180)),  # 70 px tall
    "c": (slice(150, 170), slice(10, 30)),   # 20 px tall
    "d": (slice(130, 193), slice(120, 160)),  # 63 px tall, one under MIN_FACE_HEIGHT
    "e": (slice(130, 194), slice(120, 160)),  # 64 px tall, at it
}
B = (130, 30, 180, 100)  # b's box


def face_mask(name):
    m = torch.zeros(H, W)
    m[FACES[name]] = 1.0
    if name == "a":
        m[20:40, 20:40] = 0.0  # a corner of a's box outside its mask
    return m


class World:
    """The stand-ins: SAM 3 (faces by name, parts as masks in the face box's coordinates) and the core
    pass. `calls` lists ("detect", prompt, image shape, the image's largest value) and ("detail", model,
    region bounds in the frame); `sampled` (size, crop size, denoise, sampling width, sampling height) and
    `pixels` (the region's pixel count) of each detail."""

    def __init__(self, faces=("a", "b", "c"), parts=None):
        self.faces, self.parts, self.calls, self.sampled, self.pixels = faces, parts or {}, [], [], []

    def detect(self, sam, cond, image, threshold, refine_iterations=2, individual_masks=False):
        assert (sam, threshold, individual_masks) == ("sam", 0.4, True)
        self.calls.append(("detect", cond, tuple(image.shape), float(image.max())))
        if cond == "face:200":
            return torch.stack([face_mask(n) for n in self.faces]) if self.faces else torch.zeros(0, H, W), None
        make = self.parts.get(cond)
        return (make(image.shape[1], image.shape[2]) if make else torch.zeros(0, image.shape[1], image.shape[2])), None

    def detail(self, frame, located, sampler, s):
        crop, region = located
        rows, cols = torch.where(region)
        self.calls.append(("detail", sampler, (crop.x0 + int(cols.min()), crop.y0 + int(rows.min()),
                                               crop.x0 + int(cols.max()) + 1, crop.y0 + int(rows.max()) + 1)))
        self.sampled.append((s.size, s.crop_size, s.denoise, crop.width, crop.height))
        self.pixels.append(int(region.sum()))
        frame[crop.y0:crop.y1, crop.x0:crop.x1][region] = 0.9


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.detailer.face"]


def run(flow, monkeypatch, world, model_no_lora="other", select=None, max_h=1024, size=1024, crop_size=2048, eye_size=1024,
        eye_crop_size=2048, eye_denoise=0.35, face=True, eyes=False):
    settings = importlib.import_module(flow.__package__ + ".core").Settings
    face_s = settings(None, [], [], 0, 1, 1.0, "euler", "simple", 0.5, size, crop_size)
    eye_s = settings(None, [], [], 0, 1, 1.0, "euler", "simple", eye_denoise, eye_size, eye_crop_size)
    monkeypatch.setattr(flow, "load", lambda name: ("sam", "clip") if name == "sam3.safetensors" else None)
    monkeypatch.setattr(flow, "text_condition", lambda clip, text: text)
    monkeypatch.setattr(flow, "detect", world.detect)
    monkeypatch.setattr(flow, "detail", world.detail)
    monkeypatch.setattr(flow, "differential", lambda model: ("dd", model))
    image = torch.zeros(1, H, W, 3)
    return flow.run(image, "main", model_no_lora, face_s, eye_s, "sam3.safetensors", 0.4, select, max_h, face, eyes)


def details(world):
    return [call[1:] for call in world.calls if call[0] == "detail"]


def test_the_largest_face_is_the_character_and_goes_first(flow, monkeypatch):
    world = World(faces=("b", "a", "c"))
    out, changed = run(flow, monkeypatch, world)
    assert world.calls[0][:2] == ("detect", "face:200")
    assert details(world) == [(("dd", "main"), (20, 20, 100, 120)), (("dd", "other"), B)]  # c is under 64 px
    assert torch.equal(changed[0], (face_mask("a") + face_mask("b")).clamp(0, 1))
    assert torch.equal(out[0, ..., 0], 0.9 * changed[0])


def test_the_mask_picks_the_character(flow, monkeypatch):
    select = torch.zeros(1, H, W)
    select[0, 50:60, 140:150] = 1.0  # on b
    world = World()
    run(flow, monkeypatch, world, select=select)
    assert details(world) == [(("dd", "main"), B), (("dd", "other"), (20, 20, 100, 120))]
    world = World()
    run(flow, monkeypatch, world, select=torch.zeros(1, H, W))  # overlaps no face: no character
    assert details(world) == [(("dd", "other"), (20, 20, 100, 120)), (("dd", "other"), B)]


def test_other_faces_untouched_without_model_no_lora(flow, monkeypatch):
    world = World()
    run(flow, monkeypatch, world, model_no_lora=None, eyes=True)
    assert all(call[0] == ("dd", "main") for call in details(world))
    assert [call for call in world.calls if call[0] == "detect"][1][2] == (1, 100, 80, 3)  # only a's box searched for parts


def eye_masks(h, w):
    m = torch.zeros(2, h, w)
    m[0, 30:40, w // 8:3 * w // 8] = 1.0
    m[1, 30:40, 5 * w // 8:7 * w // 8] = 1.0
    return m


def eyes_with_a_stray(h, w):
    m = torch.zeros(3, h, w)
    m[:2] = eye_masks(h, w)
    m[2, 2:10, 2:10] = 1.0  # in the box's corner outside a's mask: dropped
    return m


def one_eye(h, w):
    return eye_masks(h, w)[:1]


A, A_EYES = (20, 20, 100, 120), (30, 50, 90, 60)  # a's box; the box of its two eyes (20 x 10 px each, 20 px apart): 60 x 10


def test_both_eyes_on_one_crop(flow, monkeypatch):
    world = World(faces=("a",), parts={"eyes:2": eyes_with_a_stray})
    run(flow, monkeypatch, world, eyes=True)
    detects = [call for call in world.calls if call[0] == "detect"]
    assert [call[1] for call in detects] == ["face:200", "eyes:2"]
    # on a's box (100 x 80) of the frame the face pass painted
    assert detects[1][2] == (1, 100, 80, 3) and detects[1][3] == pytest.approx(0.9)
    # one eye pass: both eyes' pixels (2 x 200) as one region, the stray outside a's mask dropped
    assert details(world) == [(("dd", "main"), A), (("dd", "main"), A_EYES)]
    assert world.pixels[1] == 400


def test_one_eye_found_is_detailed_alone(flow, monkeypatch):
    world = World(faces=("a",), parts={"eyes:2": one_eye})
    run(flow, monkeypatch, world, eyes=True)
    assert details(world) == [(("dd", "main"), A), (("dd", "main"), (30, 50, 50, 60))] and world.pixels[1] == 200


def test_each_pass_its_own_settings(flow, monkeypatch):
    world = World(faces=("a",), parts={"eyes:2": eye_masks})
    run(flow, monkeypatch, world, size=256, crop_size=384, eye_size=192, eye_crop_size=576, eye_denoise=0.35, eyes=True)
    assert [call[1] for call in details(world)] == [A, A_EYES]
    # the face (80 x 100) at 256 in a 384 crop: a 120 x 150 crop at 256 / 100 -> 307.2 x 384 -> 320 x 384, at denoise
    # the eyes (60 x 10) at 192 in a 576 crop: a 180 x 30 crop at 192 / 60 -> 576 x 96 -> 576 x 128, at eye_denoise (the
    # face's pair: 384 x 64)
    assert world.sampled == [(256, 384, 0.5, 320, 384), (192, 576, 0.35, 576, 128)]


def test_max_face_height(flow, monkeypatch):
    world = World(faces=("a", "b"), parts={"eyes:2": eye_masks})
    run(flow, monkeypatch, world, max_h=80, eyes=True)
    # a (100 px, over 80) skips its face pass but its eyes run; b (70 px) gets both passes
    assert details(world) == [(("dd", "main"), A_EYES), (("dd", "other"), B), (("dd", "other"), (136, 60, 173, 70))]
    for max_h, face_pass in ((99, False), (100, True)):  # a (100 px) one px over max_face_height: skipped; at exactly it: runs
        world = World(faces=("a",), parts={"eyes:2": eye_masks})
        run(flow, monkeypatch, world, max_h=max_h, eyes=True)
        assert [call[1] for call in details(world)] == [A] * face_pass + [A_EYES]  # the eyes run either way


def test_face_size_or_larger_skips_the_face_pass(flow, monkeypatch):
    for size, face_pass in ((100, False), (101, True)):  # a's longer side (100 px) at face_size skips; one px under it runs
        world = World(faces=("a",), parts={"eyes:2": eye_masks})
        run(flow, monkeypatch, world, size=size, eyes=True)  # under max_face_height (1024) either way
        assert [call[1] for call in details(world)] == [A] * face_pass + [A_EYES]  # the eyes run either way


def test_eye_size_or_larger_skips_the_eye_pass(flow, monkeypatch):
    for eye_size, eye_pass in ((60, False), (61, True)):  # the eyes' box (60 px) at eye_size skips; one px under it runs
        world = World(faces=("a",), parts={"eyes:2": eye_masks})
        run(flow, monkeypatch, world, eye_size=eye_size, eyes=True)
        assert [call[1] for call in details(world)] == [A] + [A_EYES] * eye_pass


def test_faces_under_64_px_get_no_pass(flow, monkeypatch):
    world = World(faces=("a", "c"), parts={"eyes:2": eye_masks})
    run(flow, monkeypatch, world, eyes=True)
    # c (20 px): no pass and no eye search on its box
    assert [call[1] for call in details(world)] == [A, A_EYES]
    assert len([call for call in world.calls if call[0] == "detect"]) == 2
    for face, passes in (("d", 0), ("e", 2)):  # 63 px: none; 64 px: the face and its eyes
        world = World(faces=(face,), parts={"eyes:2": eye_masks})
        run(flow, monkeypatch, world, eyes=True)
        assert len(details(world)) == passes


def test_no_face_leaves_the_image(flow, monkeypatch):
    world = World(faces=())
    out, changed = run(flow, monkeypatch, world, eyes=True)
    assert details(world) == [] and not bool(changed.any()) and not bool(out.any())

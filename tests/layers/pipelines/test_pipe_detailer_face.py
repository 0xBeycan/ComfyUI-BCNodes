"""Face Detailer's flow (pipelines/detailer/face.py), with a stand-in SAM 3 and a stand-in core pass (it
records each region and the model, and paints the region so a later detection can see it):

  - the fixed prompts; the checkpoint and the threshold handed to SAM 3;
  - the character: the largest face without a mask, the face the mask overlaps most with one, none when
    the mask overlaps no face; detailed first, with the main model; the others with model_no_lora, and
    not at all without it;
  - the size gate: a face under min_face_height gets no pass and no part detection; a face over
    max_face_height skips the face pass while its eyes still run;
  - the parts: found on the face's box of the frame the face pass left, cut to the face's mask (a part
    outside it dropped); the mouth's detections one region, each eye a region of its own;
  - the passes in order face, mouth, eyes; the mask output the union of the regions detailed.
"""

import importlib

import pytest
import torch

H = W = 200
FACES = {  # name: (rows, cols) of its mask
    "a": (slice(20, 120), slice(20, 100)),   # 100 px tall, the largest
    "b": (slice(30, 90), slice(130, 180)),   # 60 px tall
    "c": (slice(150, 170), slice(10, 30)),   # 20 px tall
}


def face_mask(name):
    m = torch.zeros(H, W)
    m[FACES[name]] = 1.0
    if name == "a":
        m[20:40, 20:40] = 0.0  # a corner of a's box outside its mask
    return m


class World:
    """The stand-ins: SAM 3 (faces by name, parts as masks in the face box's coordinates) and the core
    pass. `calls` lists ("detect", prompt, image shape, the image's largest value) and ("detail", model,
    region bounds in the frame)."""

    def __init__(self, faces=("a", "b", "c"), parts=None):
        self.faces, self.parts, self.calls = faces, parts or {}, []

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
        frame[crop.y0:crop.y1, crop.x0:crop.x1][region] = 0.9


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.detailer.face"]


def run(flow, monkeypatch, world, model_no_lora="other", select=None, min_h=32, max_h=16384, face=True, mouth=False, eyes=False):
    settings = importlib.import_module(flow.__package__ + ".core").Settings(None, [], [], 0, 1, 1.0, "euler", "simple", 0.5, 0.2,
                                                                            1024, 2.0)
    monkeypatch.setattr(flow, "load", lambda name: ("sam", "clip") if name == "sam3.safetensors" else None)
    monkeypatch.setattr(flow, "text_condition", lambda clip, text: text)
    monkeypatch.setattr(flow, "detect", world.detect)
    monkeypatch.setattr(flow, "detail", world.detail)
    monkeypatch.setattr(flow, "differential", lambda model: ("dd", model))
    image = torch.zeros(1, H, W, 3)
    return flow.run(image, "main", model_no_lora, settings, "sam3.safetensors", 0.4, select, min_h, max_h, face, mouth, eyes)


def details(world):
    return [call[1:] for call in world.calls if call[0] == "detail"]


def test_the_largest_face_is_the_character_and_goes_first(flow, monkeypatch):
    world = World(faces=("b", "a", "c"))
    out, changed = run(flow, monkeypatch, world)
    assert world.calls[0][:2] == ("detect", "face:200")
    assert details(world) == [(("dd", "main"), (20, 20, 100, 120)), (("dd", "other"), (130, 30, 180, 90))]  # c is under 32 px
    assert torch.equal(changed[0], (face_mask("a") + face_mask("b")).clamp(0, 1))
    assert torch.equal(out[0, ..., 0], 0.9 * changed[0])


def test_the_mask_picks_the_character(flow, monkeypatch):
    select = torch.zeros(1, H, W)
    select[0, 50:60, 140:150] = 1.0  # on b
    world = World()
    run(flow, monkeypatch, world, select=select)
    assert details(world) == [(("dd", "main"), (130, 30, 180, 90)), (("dd", "other"), (20, 20, 100, 120))]
    world = World()
    run(flow, monkeypatch, world, select=torch.zeros(1, H, W))  # overlaps no face: no character
    assert details(world) == [(("dd", "other"), (20, 20, 100, 120)), (("dd", "other"), (130, 30, 180, 90))]


def test_other_faces_untouched_without_model_no_lora(flow, monkeypatch):
    world = World()
    run(flow, monkeypatch, world, model_no_lora=None, mouth=True, eyes=True)
    assert all(call[0] == ("dd", "main") for call in details(world))
    assert [call for call in world.calls if call[0] == "detect"][1][2] == (1, 100, 80, 3)  # only a's box searched for parts


def eye_masks(h, w):
    m = torch.zeros(2, h, w)
    m[0, 30:40, w // 8:3 * w // 8] = 1.0
    m[1, 30:40, 5 * w // 8:7 * w // 8] = 1.0
    return m


def mouth_masks(h, w):
    m = torch.zeros(3, h, w)
    m[0, 70:76, 25:55] = 1.0   # lips
    m[1, 74:80, 30:50] = 1.0   # teeth
    m[2, 2:10, 2:10] = 1.0     # in the box's corner outside a's mask: dropped
    return m


def test_parts_found_on_the_detailed_face_and_cut_to_its_mask(flow, monkeypatch):
    world = World(faces=("a",), parts={"teeth, lips": mouth_masks, "eyes:2": eye_masks})
    run(flow, monkeypatch, world, mouth=True, eyes=True)
    detects = [call for call in world.calls if call[0] == "detect"]
    assert [call[1] for call in detects] == ["face:200", "teeth, lips", "eyes:2"]
    # on a's box (100 x 80) of the frame the face pass painted
    assert all(call[2] == (1, 100, 80, 3) and call[3] == pytest.approx(0.9) for call in detects[1:])
    assert details(world) == [
        (("dd", "main"), (20, 20, 100, 120)),   # face
        (("dd", "main"), (45, 90, 75, 100)),    # the mouth: lips and teeth as one region
        (("dd", "main"), (30, 50, 50, 60)),     # each eye on its own
        (("dd", "main"), (70, 50, 90, 60)),
    ]


def test_size_gate(flow, monkeypatch):
    world = World(faces=("a", "b"), parts={"eyes:2": eye_masks})
    run(flow, monkeypatch, world, max_h=80, eyes=True)
    # a (100 px) skips its face pass but its eyes run; b (60 px) gets both
    assert [call[0] for call in details(world)] == [("dd", "main")] * 2 + [("dd", "other")] * 3
    assert details(world)[0][1] == (30, 50, 50, 60) and details(world)[2][1] == (130, 30, 180, 90)
    world = World(faces=("a", "b"), parts={"eyes:2": eye_masks})
    run(flow, monkeypatch, world, min_h=61, eyes=True)
    # b (60 px) is under the minimum: no pass and no part search on its box
    assert all(call[0] == ("dd", "main") for call in details(world))
    assert len([call for call in world.calls if call[0] == "detect"]) == 2


def test_no_face_leaves_the_image(flow, monkeypatch):
    world = World(faces=())
    out, changed = run(flow, monkeypatch, world, eyes=True)
    assert details(world) == [] and not bool(changed.any()) and not bool(out.any())

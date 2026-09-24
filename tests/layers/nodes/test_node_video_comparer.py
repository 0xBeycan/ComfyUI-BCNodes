"""Golden: Video Comparer's _encode and compare(), with a recording fake PyAV.

The fake `av` module records every call _encode makes: the container (path, mode, format), each
stream (codec, rate and its type, layout) and every attribute set on it (width, height, pix_fmt,
options), every rgb24 frame handed to VideoFrame.from_ndarray (dtype, shape, md5) and its
reformat, the audio frame (format, layout, dtype, shape, md5, sample_rate, pts), every encode
and mux in order, and the close. The returned dict is pinned with the random file name masked;
the name itself must match ^bcnodes\\.compare\\.[0-9a-f]{8}\\.mp4$ and be the file the
container was opened on, inside folder_paths' temp directory (created by _encode).

Inputs: A = 6 frames of 33x65, B = 4 frames of 32x64 (seeded); fps 12.0 and 29.97; audio with
1, 2, 6 and 3 channels, a short and a float64 waveform. compare(): the fps fallback, no sides,
zero-length audio, one side only. _encode stays on the node module, so no WHERE.
"""

import hashlib
import os
import re
import sys
import types

import pytest
import torch

from _golden import check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
    'torch': '2.14.0',
    'numpy': '2.5.3',
}

GOLDEN = {
    '_encode/fps_12': 'fc1fedb1e7460a56c99009aa7ae50b01',
    '_encode/fps_29.97': 'aa71c095ce0067e43f58f14db6f35656',
    '_encode/fps_29.97_stereo': '772af58d8407ee773ea630c7da28a563',
    '_encode/audio_mono': '19efad1641af781ce44d44c118f44325',
    '_encode/audio_stereo': '51a1f460f3130e9dd08afb4e4d170d00',
    '_encode/audio_5.1': '69249e52d44e5fe62a8938f29cbe1380',
    '_encode/audio_3_channels': 'af9d701ca8f21f790cc5f081907c2917',
    '_encode/audio_shorter_than_video': '8d6efb402e4c56487b06ec8816897d7e',
    '_encode/audio_float64': 'aa12ea40b817ced87d27035e7205c6e1',
    '_encode/b_first': '3f0cfa800c5cdb6b79eff8be85b16217',
    '_encode/one_side': '01eea7bc116428092004fb094506e79c',
    '_encode/rgba_side': '0b99846c1202df99f61418a14d53abd6',
    '_encode/b_larger': '5e597353674a5ec9e02963edeef09efc',
    'compare/fps_fallback_zero': 'dc0651e423ee2d82f7a4238996d0440e',
    'compare/fps_fallback_none': '7b9b49c08533080dcf64537a760c4cc5',
    'compare/fps_int': '091ece124677bb886c112fc357443e6a',
    'compare/zero_length_audio': '12df866bdb5a8c84175b94f7fe56e460',
    'compare/with_audio': 'f9d40a3815e8701277896b3d574bd081',
    'compare/empty_b': '951cce3932a00557ab5e485d331eba97',
    'compare/nothing_wired': "{'ui': {'bc_video': []}}",
    'compare/fps_none': "{'ui': {'bc_video': []}}",
    'compare/empty_batches': "{'ui': {'bc_video': []}}",
    'compare/audio_only': "{'ui': {'bc_video': []}}",
}

NAME = re.compile(r"^bcnodes\.compare\.[0-9a-f]{8}\.mp4$")


def _frames(n, h, w, seed, c=3):
    return torch.rand((n, h, w, c), generator=torch.Generator().manual_seed(seed))


def _a():
    return _frames(6, 33, 65, 1)


def _b():
    return _frames(4, 32, 64, 2)


def _audio(channels, samples=4000, sample_rate=8000, dtype=torch.float32):
    wave = torch.rand((1, channels, samples), generator=torch.Generator().manual_seed(3), dtype=dtype) * 2 - 1
    return {"waveform": wave, "sample_rate": sample_rate}


def _nd(arr):
    return (str(arr.dtype), tuple(arr.shape), hashlib.md5(arr.tobytes()).hexdigest())


class _Recorder:
    def __init__(self):
        self.log = []
        self.paths = []


def _fake_av(rec):
    av = types.ModuleType("av")

    class Frame:
        def __init__(self, kind, n):
            object.__setattr__(self, "tag", f"{kind}#{n}")

        def __setattr__(self, name, value):
            rec.log.append(("set", self.tag, name, repr(value)))
            object.__setattr__(self, name, value)

        def reformat(self, format=None, **kw):
            rec.log.append(("reformat", self.tag, format, kw))
            return Frame("reformatted", len(rec.log))

    class VideoFrame:
        @staticmethod
        def from_ndarray(arr, format=None, **kw):
            rec.log.append(("VideoFrame.from_ndarray", format, kw, _nd(arr)))
            return Frame("video", len(rec.log))

    class AudioFrame:
        @staticmethod
        def from_ndarray(arr, format=None, layout=None, **kw):
            rec.log.append(("AudioFrame.from_ndarray", format, layout, kw, _nd(arr), arr.flags["C_CONTIGUOUS"]))
            return Frame("audio", len(rec.log))

    class Stream:
        def __init__(self, n):
            object.__setattr__(self, "tag", f"stream#{n}")

        def __setattr__(self, name, value):
            rec.log.append(("set", self.tag, name, repr(value)))
            object.__setattr__(self, name, value)

        def encode(self, frame=None):
            rec.log.append(("encode", self.tag, None if frame is None else frame.tag))
            return [f"packet:{self.tag}:{len(rec.log)}"]

    class Container:
        streams = 0

        def add_stream(self, codec, rate=None, **kw):
            rec.log.append(("add_stream", codec, repr(rate), type(rate).__name__, kw))
            Container.streams += 1
            return Stream(Container.streams)

        def mux(self, packets):
            rec.log.append(("mux", packets))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            rec.log.append(("close", exc[0].__name__ if exc[0] else None))
            return False

    def open_(path, mode="r", format=None, **kw):
        rec.paths.append(path)
        rec.log.append(("open", "<path>", mode, format, kw))
        return Container()

    av.open = open_
    av.VideoFrame = VideoFrame
    av.AudioFrame = AudioFrame
    return av


@pytest.fixture
def fake_av(monkeypatch, tmp_path):
    """A recording `av` in sys.modules and folder_paths' temp dir under tmp_path (not created yet)."""
    rec = _Recorder()
    monkeypatch.setitem(sys.modules, "av", _fake_av(rec))
    temp = str(tmp_path / "comfy_temp")
    monkeypatch.setattr(sys.modules["folder_paths"], "get_temp_directory", lambda: temp, raising=True)
    rec.temp = temp
    return rec


def _masked(info, rec):
    """The /view entry with its file name checked and masked."""
    assert NAME.match(info["filename"]), info["filename"]
    assert rec.paths == [os.path.join(rec.temp, info["filename"])]
    assert os.path.isdir(rec.temp)
    return {**info, "filename": "<filename>"}


ENCODE = {
    "fps_12": lambda: ([("A", _a()), ("B", _b())], 12.0, None),
    "fps_29.97": lambda: ([("A", _a()), ("B", _b())], 29.97, None),
    "fps_29.97_stereo": lambda: ([("A", _a()), ("B", _b())], 29.97, _audio(2)),
    "audio_mono": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(1)),
    "audio_stereo": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(2)),
    "audio_5.1": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(6)),
    "audio_3_channels": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(3)),
    "audio_shorter_than_video": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(2, samples=1000)),
    "audio_float64": lambda: ([("A", _a()), ("B", _b())], 12.0, _audio(2, dtype=torch.float64)),
    "b_first": lambda: ([("B", _b()), ("A", _a())], 12.0, None),
    "one_side": lambda: ([("A", _a())], 12.0, None),
    "rgba_side": lambda: ([("A", _frames(3, 33, 65, 1, c=4)), ("B", _b())], 12.0, None),
    "b_larger": lambda: ([("A", _frames(2, 20, 30, 1)), ("B", _b())], 12.0, None),
}


@pytest.mark.parametrize("name", list(ENCODE))
def test_encode(name, bcnodes, fake_av):
    check_env(ENV, "torch", "numpy")
    sides, fps, audio = ENCODE[name]()
    info = bcnodes["video_comparer"]._encode(sides, fps, audio)
    check(GOLDEN, f"_encode/{name}", digest({"info": _masked(info, fake_av), "calls": fake_av.log}))


COMPARE = {
    "fps_fallback_zero": lambda: dict(fps=0, video_a=_a(), video_b=_b()),
    "fps_fallback_none": lambda: dict(fps=None, video_a=_a()),
    "fps_int": lambda: dict(fps=30, video_b=_b()),
    "zero_length_audio": lambda: dict(fps=12.0, video_a=_a(), video_b=_b(), audio={"waveform": torch.zeros((1, 2, 0)), "sample_rate": 8000}),
    "with_audio": lambda: dict(fps=12.0, video_a=_a(), video_b=_b(), audio=_audio(1)),
    "empty_b": lambda: dict(fps=12.0, video_a=_a(), video_b=torch.zeros((0, 32, 64, 3))),
}


@pytest.mark.parametrize("name", list(COMPARE))
def test_compare(name, bcnodes, fake_av):
    result = bcnodes["video_comparer"].VideoComparer().compare(**COMPARE[name]())
    entries = [_masked(info, fake_av) for info in result["ui"]["bc_video"]]
    check(GOLDEN, f"compare/{name}", digest({"result": {**result, "ui": {"bc_video": entries}}, "calls": fake_av.log}))


NOTHING = {
    "nothing_wired": lambda: dict(),
    "fps_none": lambda: dict(fps=None),
    "empty_batches": lambda: dict(fps=24.0, video_a=torch.zeros((0, 8, 8, 3)), video_b=torch.zeros((0, 8, 8, 3))),
    "audio_only": lambda: dict(fps=24.0, audio=_audio(2)),
}


@pytest.mark.parametrize("name", list(NOTHING))
def test_compare_without_sides(name, bcnodes, fake_av):
    result = bcnodes["video_comparer"].VideoComparer().compare(**NOTHING[name]())
    assert fake_av.log == [] and not os.path.exists(fake_av.temp)
    check(GOLDEN, f"compare/{name}", repr(result))

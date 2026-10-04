"""The black box (pipelines/process_monitor/blackbox.py) and the settings file: the writer and its
reader, rotation, crash detection from a log without an end record, the oom_kill confirmation, the
macOS "fell into swap" note, and the Last run report.
"""

import json
import os

import pytest

GIB = 1 << 30


@pytest.fixture
def bb(bcnodes):
    return bcnodes["pipelines.process_monitor.blackbox"]


def start(session="old", oom_kills=0, platform="linux", loaded=(), armed=False):
    return {"type": "start", "v": 1, "session": session, "prompt_id": "p-1", "workflow_id": "wf", "t": 100.0, "pid": 7,
            "platform": platform, "ram_source": "cgroup v2 at /sys/fs/cgroup", "ram_limit": 2 * GIB, "oom_kills": oom_kills,
            "models_loaded": list(loaded), "armed": armed, "threshold": 0.85}


def sample(t, ram, node="5", line="/x/custom_nodes/pack/nodes.py:120 load", **extra):
    return {"type": "sample", "t": t, "ram": ram, "ram_limit": 2 * GIB, "node": node, "class_type": "LoadThing", "line": line, **extra}


def write(bb, path, records, cut_last=False):
    log = bb.RunLog(str(path))
    for r in records:
        log.write(r)
    log.close()
    if cut_last:
        with open(path, "a") as f:
            f.write('{"type":"sample","t":10')  # the kill cut this line


def test_writer_and_reader(bb, tmp_path):
    path = tmp_path / "run.jsonl"
    recs = [start(), sample(100.1, GIB), {"type": "end", "t": 101.0, "status": "success", "seconds": 1.0, "monitor": {}}]
    write(bb, path, recs)
    assert bb.read_records(str(path)) == recs and bb.ended(recs)


def test_lines_reach_the_file_before_close(bb, tmp_path):
    """Line-buffered: a record is in the file the moment write() returns (what a SIGKILL leaves)."""
    path = tmp_path / "run.jsonl"
    log = bb.RunLog(str(path))
    log.write(start())
    with open(path) as f:
        assert json.loads(f.read())["type"] == "start"
    log.close()
    log.write(sample(1.0, 1))  # after close: dropped, no error
    assert len(bb.read_records(str(path))) == 1


def test_a_line_cut_by_the_kill_is_dropped_but_a_bad_middle_line_raises(bb, tmp_path):
    path = tmp_path / "run.jsonl"
    write(bb, path, [start(), sample(100.1, GIB)], cut_last=True)
    assert [r["type"] for r in bb.read_records(str(path))] == ["start", "sample"]
    with open(path, "w") as f:
        f.write('{"type":"start"}\nnot json\n{"type":"end"}\n')
    with pytest.raises(ValueError):
        bb.read_records(str(path))


def test_rotation_keeps_the_newest(bb, tmp_path):
    for i in range(6):
        (tmp_path / f"20260101-00000{i}-000-p{i}.jsonl").write_text("{}\n")
    bb.rotate(str(tmp_path), keep=3)  # room for the run about to start: 2 kept
    assert sorted(os.listdir(tmp_path)) == ["20260101-000004-000-p4.jsonl", "20260101-000005-000-p5.jsonl"]
    assert bb.run_files(str(tmp_path / "missing")) == []


def test_run_path_sorts_by_time(bb, tmp_path):
    a = bb.run_path(str(tmp_path), 1_000_000.25, "abcdefgh-1234")
    b = bb.run_path(str(tmp_path), 1_000_000.75, "00000000-1234")
    assert os.path.basename(a).endswith("-250-abcdefgh.jsonl") and sorted([b, a]) == [a, b]


def test_crash_detection_reads_head_and_tail_only(bb, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    long_run = [start()] + [sample(100 + i * 0.1, GIB, line="x" * 200) for i in range(2000)]  # ~0.5 MB
    write(bb, runs / "20260101-000001-000-p.jsonl", long_run, cut_last=True)
    assert bb.latest_crash(str(runs), session="new") is not None
    write(bb, runs / "20260101-000002-000-p.jsonl", long_run + [{"type": "end", "t": 1, "status": "success", "seconds": 1,
                                                                   "monitor": {}}])
    assert bb.latest_crash(str(runs), session="new") is None  # the end record is seen from the tail
    (runs / "20260101-000003-000-q.jsonl").write_text('{"type":"sta')  # killed inside the start line
    assert bb.latest_crash(str(runs), session="new") is None


def test_crash_detected_only_for_another_session_without_end(bb, tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    write(bb, runs / "20260101-000001-000-p.jsonl", [start(session="old"), sample(100.1, GIB)])
    found = bb.latest_crash(str(runs), session="new")
    assert found is not None and bb.read_records(found)[0]["prompt_id"] == "p-1"
    assert bb.latest_crash(str(runs), session="old") is None  # this process's own running run
    # a newer run that ended: the crash is history
    write(bb, runs / "20260101-000002-000-q.jsonl", [start(session="new"), {"type": "end", "t": 1, "status": "success",
                                                                            "seconds": 1, "monitor": {}}])
    assert bb.latest_crash(str(runs), session="newer") is None
    assert bb.latest_finished(str(runs))[0]["session"] == "new"


def crashed_run(bb, oom_kills=0, platform="linux", swap=None):
    recs = [start(oom_kills=oom_kills, platform=platform)]
    recs.append({"type": "node", "t": 100.5, "node": "5", "display": "5", "class_type": "LoadThing", "ram": 300 << 20,
                 "cache": 0, "inputs": []})
    for i in range(30):
        extra = {} if swap is None else {"swap": swap + i * (100 << 20)}
        recs.append(sample(101.0 + i * 0.1, (400 << 20) + i * (50 << 20), **extra))
    recs.append({"type": "snapshot", "t": 103.5, "ram": int(1.75 * GIB), "ram_limit": 2 * GIB, "threshold": 0.85, "node": "5",
                 "class_type": "LoadThing", "line": "/x/custom_nodes/pack/nodes.py:120 load",
                 "census": {"groups": [{"count": 3, "shape": [609, 1280, 720, 3], "dtype": "float32", "device": "cpu",
                                        "bytes_each": 6735052800, "bytes": 3 * 6735052800}], "small_bytes": 0, "total_bytes": 0},
                 "stack": ["  File x, line 1\n"], "seconds": 0.4, "stopped": False})
    return recs


def test_crash_report_names_node_line_tensors_and_confirms_the_oom_kill(bb):
    r = bb.crash_report(crashed_run(bb, oom_kills=2), oom_kills_now=3, platform="linux")
    assert r["cause"]["kind"] == "oom_kill" and "2 to 3" in r["cause"]["text"]
    assert (r["node"], r["class_type"], r["line"]) == ("5", "LoadThing", "/x/custom_nodes/pack/nodes.py:120 load")
    assert r["node_start"]["ram"] == 300 << 20 and r["cache"] == 0
    assert r["snapshot"]["census"]["groups"][0]["count"] == 3
    assert r["ran_s"] == pytest.approx(3.5) and r["curve"][0] == [1.0, 400 << 20]
    assert r["growth_per_s"] == pytest.approx(500 << 20, rel=1e-6)  # 50 MiB per 100 ms sample


@pytest.mark.parametrize("before,now,kind", [(1, 1, "not_oom"), (5, 0, "unknown"), (None, 1, "unknown"), (1, None, "unknown")])
def test_crash_cause_without_a_confirmed_oom_kill(bb, before, now, kind):
    assert bb.crash_report(crashed_run(bb, oom_kills=before), now, "linux")["cause"]["kind"] == kind


def test_macos_says_fell_into_swap(bb):
    r = bb.crash_report(crashed_run(bb, platform="darwin", swap=GIB), None, "darwin")
    assert r["cause"]["kind"] == "no_oom_kill"
    assert r["swap"]["text"].startswith("fell into swap") and r["swap"]["node"] == "5"
    assert bb.crash_report(crashed_run(bb, platform="darwin", swap=None), None, "darwin")["swap"] is None


def test_run_report(bb):
    recs = [start(loaded=[{"name": "WanModel", "bytes": 10 * GIB}], armed=True), sample(100.1, GIB, vram=3 * GIB, vram_device=9 * GIB),
            sample(100.2, int(1.5 * GIB), vram=4 * GIB, vram_device=8 * GIB),
            {"type": "cached", "t": 100.12, "nodes": [{"node": "4", "class_type": "W", "output_bytes": 8},
                                                      {"node": "6", "class_type": "Y", "output_bytes": 9}]},
            {"type": "node", "t": 100.15, "node": "5", "display": "5", "class_type": "X", "ram": GIB, "cache": 0, "inputs": []},
            {"type": "node_end", "t": 100.3, "node": "5", "display": "5", "class_type": "X", "state": "executed", "seconds": 0.15},
            {"type": "node_end", "t": 100.31, "node": "6", "display": "6", "class_type": "Y", "state": "cached"},
            {"type": "end", "t": 102.0, "status": "success", "seconds": 2.0,
             "monitor": {"hook_s": 0.001, "sampler_s": 0.002, "snapshot_s": 0.0, "total_s": 0.003}}]
    r = bb.run_report(recs)
    assert r["status"] == "success" and r["armed"] and r["overhead_pct"] == pytest.approx(0.15)
    assert r["ram_peak"] == int(1.5 * GIB) and r["vram_peak"] == 4 * GIB and r["vram_device_peak"] == 9 * GIB
    # node 4 never ran (in the cache at the start), node 6 was staged and found cached: one row each
    assert [(n["node"], n["state"]) for n in r["nodes"]] == [("4", "cached"), ("5", "executed"), ("6", "cached")]
    assert r["profile"].startswith("repeat run: 1 model(s)") and "WanModel" in r["profile"]
    # the last node runs until the run ends
    assert r["timeline"] == [{"node": "5", "display": "5", "class_type": "X", "ram": GIB, "cache": 0, "at": 0.15, "seconds": 1.85}]
    assert bb.run_report([start(), recs[-1]])["profile"].startswith("first run")
    assert bb.run_report([start(), recs[-1]])["vram_device_peak"] is None


def test_run_report_timeline_seconds(bb):
    node = lambda t, n: {"type": "node", "t": t, "node": n, "display": n, "class_type": "X", "ram": GIB, "cache": 0, "inputs": []}
    end = {"type": "end", "t": 110.0, "status": "success", "seconds": 10.0, "monitor": {}}
    r = bb.run_report([start(), node(100.5, "1"), node(101.0, "2"), node(106.0, "3"), end])
    # each node until the next one starts
    assert [(t["node"], t["at"], t["seconds"]) for t in r["timeline"]] == [("1", 0.5, 0.5), ("2", 1.0, 5.0), ("3", 6.0, 4.0)]
    # a run cut off before its end record: the last node's time is unknown, never a guess
    assert bb.run_report([start(), node(100.5, "1"), node(101.0, "2")])["timeline"][-1]["seconds"] is None


def test_settings_validate_load_and_save(bcnodes, tmp_path):
    st = bcnodes["pipelines.process_monitor.settings"]
    s = st.load(str(tmp_path / "none.json"))
    # no file: the defaults, the monitor on
    assert (s.enabled, s.black_box, s.threshold, s.stop_at_threshold, s.keep_runs) == (True, True, 0.85, False, 20)
    s = st.apply(s, {"threshold": "0.9", "keep_runs": 5, "enabled": False})
    path = str(tmp_path / "pm" / "settings.json")
    st.save(path, s)
    assert st.load(path) == s
    for bad, match in (({"threshold": 1.5}, "between 0.5 and 0.99"), ({"keep_runs": "x"}, "must be a number"),
                       ({"enabled": "yes"}, "true or false"), ({"colour": 1}, "unknown monitor setting")):
        with pytest.raises(ValueError, match=match):
            st.apply(s, bad)

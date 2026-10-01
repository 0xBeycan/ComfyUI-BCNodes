"""The black box: one JSON line per record, written as things happen, read back after the fact.

A RAM OOM is a SIGKILL: no except, finally or atexit runs, so nothing is written at the end. Every
record goes to the kernel the moment it exists (line-buffered, no fsync: data the kernel holds
survives a process kill; only a machine crash loses it). One file per run, named by its start time
so a name sort is a time sort; the last `keep` runs are kept.

A run whose file has no end record and was written by another process (another `session`) did
not end: crash_report explains it, with the cgroup's oom_kill counter as the confirmation.
"""

import json
import os
import threading
import time
from typing import TypedDict

FORMAT = 1
_SWAP_STEP = 512 * 1024 * 1024  # host swap growth that counts as "fell into swap"
_CURVE_POINTS = 150


class RunStart(TypedDict):
    type: str  # "start"
    v: int
    session: str
    prompt_id: str
    workflow_id: str
    t: float
    pid: int
    platform: str
    ram_source: str
    ram_limit: int
    oom_kills: int  # None outside a cgroup v2 / v1 with the counter
    models_loaded: list
    armed: bool
    threshold: float


class Sample(TypedDict, total=False):
    type: str  # "sample"
    t: float
    ram: int
    ram_raw: int
    ram_limit: int
    swap: int
    vram: int
    vram_reserved: int
    vram_device: int
    vram_total: int
    gpu_util: int
    node: str
    class_type: str
    line: str  # the innermost frame outside the stdlib and site-packages: the node's own line
    top: str  # the innermost frame, when it is another one


class NodeStart(TypedDict):
    type: str  # "node"
    t: float
    node: str
    display: str
    class_type: str
    ram: int
    cache: int  # bytes held by ComfyUI's output cache when the node starts
    inputs: list  # [{"name", "bytes", "count", "tensors": [tensor_census.describe(t) ...]}]


class NodeEnd(TypedDict, total=False):
    type: str  # "node_end"
    t: float
    node: str
    display: str
    class_type: str
    state: str  # "executed", "cached" (not measured), "failed"
    seconds: float
    ram_start: int
    ram_end: int
    ram_peak: int
    peak_source: str
    vram_peak: int
    vram_reserved_peak: int
    outputs: list
    output_bytes: int
    cache: int
    models_loaded: list
    models_unloaded: list


class Snapshot(TypedDict):
    type: str  # "snapshot", twice: scope "execution thread locals" at once, then "whole process"
    scope: str
    t: float
    ram: int
    ram_limit: int
    threshold: float
    node: str
    class_type: str
    line: str
    census: dict  # tensor_census.census()
    stack: list
    seconds: float
    stopped: bool


class CachedNodes(TypedDict):
    type: str  # "cached", once per armed run, before its first node record
    t: float
    nodes: list  # [{"node", "class_type", "output_bytes"}] in the output cache when the run started


class RunEnd(TypedDict):
    type: str  # "end"
    t: float
    status: str
    seconds: float
    monitor: dict  # {"hook_s", "sampler_s", "snapshot_s", "total_s"}


class RunLog:
    """Appends records to one run file. Two threads write (the sampler and the hook on the execution
    thread), so a line is written under a lock."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, record):
        """Writes one line; a line that arrives after close (the other thread ended the run between
        its read and its write) is dropped."""
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        with self._lock:
            if not self._file.closed:
                self._file.write(line)

    def close(self):
        with self._lock:
            self._file.close()


def run_path(runs_dir, t, prompt_id):
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}"
    return os.path.join(runs_dir, f"{stamp}-{prompt_id[:8]}.jsonl")


def run_files(runs_dir):
    """Run files, newest first."""
    try:
        names = [n for n in os.listdir(runs_dir) if n.endswith(".jsonl")]
    except FileNotFoundError:
        return []
    return [os.path.join(runs_dir, n) for n in sorted(names, reverse=True)]


def rotate(runs_dir, keep):
    """Deletes all but the newest keep - 1 runs, making room for the one about to start."""
    for path in run_files(runs_dir)[max(keep - 1, 0):]:
        os.remove(path)


def read_records(path):
    """The file's records. A last line cut by the kill is dropped; any other bad line raises."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    records = []
    for i, line in enumerate(lines):
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            if i < len(lines) - 2:
                raise
    return records


def ended(records):
    return bool(records) and records[-1].get("type") == "end"


def _head_and_tail(path):
    """(start record or None, whether the last line is an end record), reading only the file's first
    line and its last 64 KB: a long run's log is megabytes."""
    with open(path, "rb") as f:
        first = f.readline()
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 65536))
        tail = [line for line in f.read().split(b"\n") if line.strip()]
    try:
        start = json.loads(first)
    except ValueError:  # killed before the start line was complete
        start = None
    try:
        last_is_end = bool(tail) and json.loads(tail[-1]).get("type") == "end"
    except ValueError:  # the last line was cut by the kill
        last_is_end = False
    return start, last_is_end


def latest_crash(runs_dir, session):
    """Path of the newest run when it has no end record and another process (another session) wrote
    it, else None. Only the newest run counts: a later run that ended means the crash is history."""
    files = run_files(runs_dir)
    if not files:
        return None
    start, last_is_end = _head_and_tail(files[0])
    if start is None or last_is_end or start.get("session") == session:
        return None
    return files[0]


def latest_finished(runs_dir):
    """Records of the newest run with an end record, or None."""
    for path in run_files(runs_dir):
        if _head_and_tail(path)[1]:
            return read_records(path)
    return None


def _of(records, kind):
    return [r for r in records if r.get("type") == kind]


def _curve(samples, t0):
    step = max(1, len(samples) // _CURVE_POINTS)
    return [[round(s["t"] - t0, 2), s.get("ram")] for s in samples[::step]]


def _swap_note(samples):
    """'fell into swap' on a host without an OOM kill: where host swap first grew by _SWAP_STEP."""
    swaps = [s for s in samples if s.get("swap") is not None]
    if not swaps:
        return None
    base = swaps[0]["swap"]
    for s in swaps:
        if s["swap"] - base >= _SWAP_STEP:
            grown = max(x["swap"] for x in swaps) - base
            return {"text": f"fell into swap: host swap grew by {grown / 2**30:.1f} GiB during the run, first at node "
                            f"{s.get('node')} ({s.get('class_type')}), line {s.get('line')}",
                    "t": s["t"], "node": s.get("node"), "line": s.get("line"), "grown": grown}
    return None


def _cause(start, oom_kills_now, platform):
    before = start.get("oom_kills")
    if platform == "darwin":
        return {"kind": "no_oom_kill", "text": "macOS does not OOM-kill: the process ended for another reason "
                "(crash, force quit, or a kill from outside). See the swap note for memory pressure."}
    if before is None or oom_kills_now is None:
        return {"kind": "unknown", "text": "no cgroup oom_kill counter to confirm an OOM kill (not in a memory-limited "
                "container, or the kernel does not expose it)"}
    if oom_kills_now > before:
        return {"kind": "oom_kill", "text": f"OOM kill confirmed: the cgroup's oom_kill counter went from {before} to "
                f"{oom_kills_now}"}
    if oom_kills_now == before:
        return {"kind": "not_oom", "text": f"not an OOM kill of this cgroup (oom_kill unchanged at {before}): another "
                "crash, a kill from outside, or the host's own OOM killer"}
    return {"kind": "unknown", "text": f"cannot confirm: the oom_kill counter is lower than at the run's start "
            f"({oom_kills_now} < {before}), so the container was recreated"}


def crash_report(records, oom_kills_now, platform):
    """Why the run of `records` (no end record) died: cause, node, line, RAM at node start, the growth
    curve, the tensors alive at the threshold snapshot and the cache total."""
    start = records[0]
    samples, nodes, snaps = _of(records, "sample"), _of(records, "node"), _of(records, "snapshot")
    last = samples[-1] if samples else {}
    node = nodes[-1] if nodes else None
    t_last = records[-1].get("t", start["t"])
    recent = [s for s in samples if s["t"] >= t_last - 3.0 and s.get("ram") is not None]
    growth = None
    if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
        growth = (recent[-1]["ram"] - recent[0]["ram"]) / (recent[-1]["t"] - recent[0]["t"])
    return {
        "prompt_id": start["prompt_id"], "workflow_id": start.get("workflow_id"), "started": start["t"],
        "ran_s": round(t_last - start["t"], 2), "cause": _cause(start, oom_kills_now, platform),
        "ram_source": start.get("ram_source"), "ram_limit": start.get("ram_limit"),
        "node": last.get("node") or (node or {}).get("node"), "class_type": last.get("class_type") or (node or {}).get("class_type"),
        "line": last.get("line"), "top": last.get("top"), "ram_last": last.get("ram"),
        "node_start": node, "cache": (node or {}).get("cache"), "growth_per_s": growth,
        "curve": _curve(samples, start["t"]), "snapshot": snaps[-1] if snaps else None,
        "swap": _swap_note(samples), "samples": len(samples),
    }


def _node_rows(records):
    """The armed run's node_end records, preceded by the nodes that were in the cache when it started
    and never ran (ComfyUI does not call execute for them): "from cache, not measured", never 0."""
    ends = _of(records, "node_end")
    seen = {r["node"] for r in ends}
    listed = [n for r in _of(records, "cached") for n in r["nodes"] if n["node"] not in seen]
    return [{"type": "node_end", "node": n["node"], "display": n["node"], "class_type": n["class_type"], "state": "cached",
             "output_bytes": n["output_bytes"], "outputs": []} for n in listed] + ends


def run_report(records):
    """The Last run tab: status, the monitor's own time, peaks, and the per-node table of an armed run
    (or the node timeline the hook recorded)."""
    start, end = records[0], records[-1]
    samples = _of(records, "sample")
    loaded = start.get("models_loaded") or []
    profile = (f"repeat run: {len(loaded)} model(s) were already loaded at the start "
               f"({', '.join(m['name'] for m in loaded)}); a first run loads them and differs"
               if loaded else "first run: no model was loaded at the start")
    seconds = end.get("seconds") or 0.0
    monitor = end.get("monitor") or {}
    snaps = _of(records, "snapshot")
    return {
        "prompt_id": start["prompt_id"], "workflow_id": start.get("workflow_id"), "started": start["t"],
        "status": end.get("status"), "seconds": seconds, "monitor": monitor,
        "overhead_pct": 100.0 * monitor.get("total_s", 0.0) / seconds if seconds else None,
        "armed": start.get("armed", False), "profile": profile,
        "ram_source": start.get("ram_source"), "ram_limit": start.get("ram_limit"),
        "ram_peak": max((s["ram"] for s in samples if s.get("ram") is not None), default=None),
        "vram_peak": max((s["vram"] for s in samples if s.get("vram") is not None), default=None),
        "nodes": _node_rows(records),
        "timeline": [{k: r.get(k) for k in ("node", "display", "class_type", "ram", "cache")} | {"at": round(r["t"] - start["t"], 2)}
                     for r in _of(records, "node")],
        "snapshot": snaps[-1] if snaps else None,
        "swap": _swap_note(samples) if start.get("platform") == "darwin" else None,
    }

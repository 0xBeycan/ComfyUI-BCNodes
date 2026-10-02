"""RAM / VRAM readers (libs/memory_sources.py) against fake cgroup trees and mocked torch counters.

This Mac has no cgroup and no CUDA: the cgroup v2 / v1 readers, the per-reader memory.peak reset,
the CUDA counters and the NVML path run here against fixtures only (files written by the test,
torch.cuda and pynvml replaced by fakes). The process reader (psutil) runs for real.
"""

import os
import stat
import sys
import types

import pytest
import torch

GIB = 1 << 30


@pytest.fixture
def ms(bcnodes):
    return bcnodes["libs.memory_sources"]


def tree(tmp_path, proc_line, files):
    """A fake /sys/fs/cgroup at tmp_path/cg and a /proc/self/cgroup holding proc_line."""
    root = tmp_path / "cg"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    proc = tmp_path / "proc_self_cgroup"
    proc.write_text(proc_line + "\n")
    return str(root), str(proc)


V2_FILES = {
    "memory.max": f"{2 * GIB}\n",
    "memory.current": "1500000000\n",
    "memory.stat": "anon 1300000000\nfile 200000000\nactive_file 100000000\ninactive_file 100000000\n",
    "memory.peak": "1600000000\n",
    "memory.events": "low 0\nhigh 0\nmax 4\noom 2\noom_kill 1\noom_group_kill 0\n",
}


def test_v2_container_root(ms, tmp_path):
    root, proc = tree(tmp_path, "0::/", V2_FILES)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert (cg.version, cg.limit, cg.directory) == (2, 2 * GIB, root)
    # working set = usage - inactive_file, as docker stats and the kernel's reclaim see it
    assert cg.read() == {"ram": 1400000000, "ram_raw": 1500000000, "ram_limit": 2 * GIB}
    assert cg.oom_kills() == 1


def test_v2_nested_path_takes_the_tightest_limit_of_the_lineage(ms, tmp_path):
    files = {f"docker/abc/{k}": v for k, v in V2_FILES.items()}
    files["docker/abc/memory.max"] = "max\n"  # no limit of its own
    files["docker/memory.max"] = f"{3 * GIB}\n"
    files["memory.max"] = f"{8 * GIB}\n"
    root, proc = tree(tmp_path, "0::/docker/abc", files)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert cg.limit == 3 * GIB and cg.directory == os.path.join(root, "docker")


def test_no_limit_below_host_ram_is_no_container(ms, tmp_path):
    files = dict(V2_FILES, **{"memory.max": "max\n"})
    root, proc = tree(tmp_path, "0::/", files)
    assert ms.find_cgroup(16 * GIB, root, proc) is None
    files = dict(V2_FILES, **{"memory.max": f"{32 * GIB}\n"})
    root, proc = tree(tmp_path / "b", "0::/", files)
    assert ms.find_cgroup(16 * GIB, root, proc) is None


def test_v1_memory_controller(ms, tmp_path):
    files = {
        "memory/memory.limit_in_bytes": f"{GIB}\n",
        "memory/memory.usage_in_bytes": "900000000\n",
        "memory/memory.stat": "cache 300000000\ntotal_inactive_file 50000000\n",
        "memory/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 3\n",
    }
    root, proc = tree(tmp_path, "12:cpu,cpuacct:/docker/x\n4:memory:/docker/x", files)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert cg.version == 1 and cg.limit == GIB
    assert cg.read() == {"ram": 850000000, "ram_raw": 900000000, "ram_limit": GIB}
    assert cg.oom_kills() == 3
    assert cg.open_peak_window() is None  # v1 has no per-reader reset


def test_v1_unlimited_is_ignored(ms, tmp_path):
    files = {"memory/memory.limit_in_bytes": "9223372036854771712\n", "memory/memory.usage_in_bytes": "1\n"}
    root, proc = tree(tmp_path, "4:memory:/", files)
    assert ms.find_cgroup(16 * GIB, root, proc) is None


def test_no_proc_self_cgroup_means_no_cgroup(ms, tmp_path):
    assert ms.find_cgroup(16 * GIB, str(tmp_path), str(tmp_path / "missing")) is None


def test_peak_window_reads_through_its_descriptor(ms, tmp_path):
    path = tmp_path / "peak"
    path.write_text("1234567\n")
    window = ms.PeakWindow(os.open(path, os.O_RDWR))
    assert window.close() == 1234567


def test_peak_window_reset_is_refused_on_a_read_only_file(ms, tmp_path):
    root, proc = tree(tmp_path, "0::/", V2_FILES)
    os.chmod(os.path.join(root, "memory.peak"), stat.S_IRUSR)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert cg.open_peak_window() is None  # docker mounts cgroupfs read-only: the sampler max stands in


def test_peak_window_reset_writes_to_memory_peak(ms, tmp_path):
    root, proc = tree(tmp_path, "0::/", V2_FILES)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    window = cg.open_peak_window()
    assert window is not None
    window.close()
    with open(os.path.join(root, "memory.peak")) as f:
        assert f.read().startswith("reset")  # a plain file keeps the write; the kernel resets instead


def test_process_reader_runs_for_real(ms):
    source = ms.ProcessMemory()
    r = source.read()
    assert 0 < r["ram"] < r["ram_limit"] and r["swap"] >= 0
    assert source.oom_kills() is None and source.open_peak_window() is None


class FakeCuda:
    def __init__(self):
        self.reset = 0

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def current_device(self):
        return 0

    def memory_allocated(self, d):
        return 3 * GIB

    def memory_reserved(self, d):
        return 4 * GIB

    def mem_get_info(self, d):
        return (18 * GIB, 24 * GIB)

    def max_memory_allocated(self, d):
        return 5 * GIB

    def max_memory_reserved(self, d):
        return 6 * GIB

    def reset_peak_memory_stats(self, d):
        self.reset += 1

    def get_device_name(self, d):
        return "Fake GPU"

    def get_device_properties(self, d):
        return types.SimpleNamespace(uuid="1234")


def test_cuda_counters_without_nvml(ms, monkeypatch):
    fake = FakeCuda()
    monkeypatch.setattr(ms.torch, "cuda", fake)
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import pynvml -> ImportError
    gpu = ms.gpu_source()
    assert isinstance(gpu, ms.CudaMemory) and gpu.nvml is None
    assert gpu.read() == {"vram": 3 * GIB, "vram_reserved": 4 * GIB, "vram_device": 6 * GIB, "vram_total": 24 * GIB}
    gpu.reset_peak()
    assert fake.reset == 1 and gpu.peak() == {"vram_peak": 5 * GIB, "vram_reserved_peak": 6 * GIB}
    assert "NVML not installed" in gpu.describe()


def test_cuda_with_nvml(ms, monkeypatch):
    monkeypatch.setattr(ms.torch, "cuda", FakeCuda())
    seen = {}
    nvml = types.ModuleType("pynvml")
    nvml.nvmlInit = lambda: None
    nvml.nvmlDeviceGetHandleByUUID = lambda uuid: seen.setdefault("uuid", uuid)
    nvml.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=7 * GIB, total=32 * GIB)
    nvml.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=87)
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    gpu = ms.gpu_source()
    assert seen["uuid"] == "GPU-1234"
    r = gpu.read()
    assert r["vram_device"] == 7 * GIB and r["vram_total"] == 32 * GIB and r["gpu_util"] == 87 and r["vram"] == 3 * GIB


def test_cuda_not_initialised_is_not_read(ms, monkeypatch):
    fake = FakeCuda()
    fake.is_initialized = lambda: False
    monkeypatch.setattr(ms.torch, "cuda", fake)
    monkeypatch.setattr(ms.torch.backends.mps, "is_available", lambda: False)
    assert ms.gpu_source() is None


def test_mps_counters(ms, monkeypatch):
    monkeypatch.setattr(ms.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ms.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(ms.torch.mps, "current_allocated_memory", lambda: 11)
    monkeypatch.setattr(ms.torch.mps, "driver_allocated_memory", lambda: 22)
    monkeypatch.setattr(ms.torch.mps, "recommended_max_memory", lambda: 33, raising=False)
    gpu = ms.gpu_source()
    assert isinstance(gpu, ms.MpsMemory)
    assert gpu.read() == {"vram": 11, "vram_reserved": 22, "vram_total": 33}
    assert gpu.peak() == {}


# --- what the full clear reads: the RAM's parts, glibc's free blocks, pinned host memory -------------

def test_v2_breakdown(ms, tmp_path):
    root, proc = tree(tmp_path, "0::/", dict(V2_FILES, **{"memory.stat": V2_FILES["memory.stat"] + "shmem 4096\n"}))
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert cg.breakdown() == {"anon": 1300000000, "file": 200000000, "active_file": 100000000,
                              "inactive_file": 100000000, "shmem": 4096}


def test_v1_breakdown_reads_the_total_keys(ms, tmp_path):
    files = {
        "memory/memory.limit_in_bytes": f"{GIB}\n",
        "memory/memory.usage_in_bytes": "900000000\n",
        "memory/memory.stat": "rss 1\ncache 2\ntotal_rss 700000000\ntotal_cache 200000000\ntotal_active_file 150000000\n"
                              "total_inactive_file 50000000\n",
    }
    root, proc = tree(tmp_path, "4:memory:/docker/x", files)
    cg = ms.find_cgroup(16 * GIB, root, proc)
    assert cg.breakdown() == {"anon": 700000000, "file": 200000000, "active_file": 150000000,
                              "inactive_file": 50000000, "shmem": None}  # a key the kernel does not write


def test_process_memory_splits_the_rss_from_proc_status(ms, tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t  3000 kB\nRssAnon:\t  2000 kB\nRssFile:\t   900 kB\nRssShmem:\t 100 kB\n")
    assert ms.process_memory(str(status)) == {"rss": 3000 * 1024, "rss_anon": 2000 * 1024, "rss_file": 900 * 1024,
                                              "rss_shmem": 100 * 1024, "uss": None}


def test_process_memory_without_proc_status_runs_for_real(ms, tmp_path):
    r = ms.process_memory(str(tmp_path / "missing"))  # macOS: psutil's RSS and USS
    assert r["rss"] > 0 and r["uss"] is not None and 0 < r["uss"] <= r["rss"] and r["rss_anon"] is None


class FakeLibc:
    """glibc's malloc_trim and mallinfo2 as ctypes functions look to Glibc (attributes settable)."""

    def __init__(self, mallinfo2_type, fordblks):
        self.trims = []

        def malloc_trim(pad):
            self.trims.append(pad)
            return 1

        def mallinfo2():
            return mallinfo2_type(arena=10 * GIB, fordblks=fordblks)

        self.malloc_trim, self.mallinfo2 = malloc_trim, mallinfo2


def test_glibc_counts_its_free_blocks_and_trims(ms):
    lib = FakeLibc(ms.Mallinfo2, fordblks=3 * GIB)
    g = ms.Glibc(lib)
    assert g.free_bytes() == 3 * GIB
    assert g.trim() is True and lib.trims == [0]
    del lib.mallinfo2  # glibc before 2.33
    assert ms.Glibc(lib).free_bytes() is None


def test_no_glibc_here_or_on_musl(ms, monkeypatch):
    def cdll(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ms.ctypes, "CDLL", cdll)
    assert ms.glibc() is None
    monkeypatch.setattr(ms.ctypes, "CDLL", lambda name: types.SimpleNamespace())  # a libc without malloc_trim
    assert ms.glibc() is None


def test_cuda_pinned_host_cache(ms, monkeypatch):
    fake = FakeCuda()
    fake.host_memory_stats = lambda: {"allocated_bytes.current": 2 * GIB, "active_bytes.current": GIB}
    monkeypatch.setattr(ms.torch, "cuda", fake)
    assert ms.CudaMemory(None).pinned_cache() == 2 * GIB
    del fake.host_memory_stats
    assert ms.CudaMemory(None).pinned_cache() is None
    assert ms.MpsMemory().pinned_cache() is None

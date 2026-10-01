"""RAM and VRAM readers for the Process Monitor.

RAM comes from the container's cgroup when one limits this process below the host's RAM (v2
`memory.current` / `memory.max` / `memory.peak` / `memory.events`, or the v1 files of the same
counters), otherwise from the process RSS through psutil. Host RAM is never used inside a
container: it hides the limit the kernel kills at.

VRAM comes from torch's allocator counters (CUDA, or MPS on Apple silicon); NVML adds the
device-wide use (every process) and the GPU load when a binding (`pynvml`) is importable.

Every read is a counter: a few bytes from a cgroup file or an allocator statistic, never a tensor.
psutil and pynvml are imported inside the functions that use them. The cgroup root and
/proc/self/cgroup are arguments, so tests run against fake trees.
"""

import os

import torch

CGROUP_ROOT = "/sys/fs/cgroup"
PROC_SELF_CGROUP = "/proc/self/cgroup"

# The same counters under cgroup v2 and v1. v1's oom_kill line lives in memory.oom_control
# (kernel 4.13+).
_FILES = {
    2: {"usage": "memory.current", "limit": "memory.max", "peak": "memory.peak", "stat": "memory.stat",
        "inactive": "inactive_file", "events": "memory.events"},
    1: {"usage": "memory.usage_in_bytes", "limit": "memory.limit_in_bytes", "peak": "memory.max_usage_in_bytes",
        "stat": "memory.stat", "inactive": "total_inactive_file", "events": "memory.oom_control"},
}


def _read(path):
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except (OSError, ValueError):
        return None


def _int(path):
    """The file's integer, or None when it is missing or not a number ("max" = no limit)."""
    raw = _read(path)
    try:
        return int(raw.strip()) if raw is not None else None
    except ValueError:
        return None


def _keyed(text, key):
    """The value of `key value` lines (memory.stat, memory.events, memory.oom_control)."""
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            return int(parts[1])
    return None


def _candidates(root, proc_self_cgroup):
    """(directory, version) of this process's cgroups, innermost first, then their parents up to the
    mount root. Inside a container the mount root is the container's own cgroup."""
    found = []
    for line in (_read(proc_self_cgroup) or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        if parts[1] == "":
            base, version = root, 2
        elif "memory" in parts[1].split(","):
            base, version = os.path.join(root, "memory"), 1
        else:
            continue
        rel = [p for p in parts[2].split("/") if p]
        for depth in range(len(rel), -1, -1):
            entry = (os.path.join(base, *rel[:depth]), version)
            if entry not in found:
                found.append(entry)
    return found


class CgroupMemory:
    """The tightest memory-limited cgroup of this process."""

    kind = "cgroup"

    def __init__(self, directory, version, limit):
        self.directory, self.version, self.limit = directory, version, limit
        self._files = {k: os.path.join(directory, v) if k != "inactive" else v for k, v in _FILES[version].items()}

    def describe(self):
        return f"cgroup v{self.version} at {self.directory}"

    def read(self):
        """{"ram": working set (usage minus inactive file cache, what the kill is decided on and what
        docker stats shows), "ram_raw": usage with every cache, "ram_limit"}."""
        usage = _int(self._files["usage"])
        inactive = _keyed(_read(self._files["stat"]), self._files["inactive"]) or 0
        return {"ram": None if usage is None else max(0, usage - inactive), "ram_raw": usage, "ram_limit": self.limit}

    def oom_kills(self):
        return _keyed(_read(self._files["events"]), "oom_kill")

    def open_peak_window(self):
        """A PeakWindow over memory.peak reset to the current usage, or None when the kernel does not
        allow a per-reader reset (cgroup v1, kernels before 6.12, a read-only cgroup mount)."""
        if self.version != 2:
            return None
        try:
            fd = os.open(self._files["peak"], os.O_RDWR)
        except OSError:
            return None
        try:
            os.write(fd, b"reset\n")
        except OSError:
            os.close(fd)
            return None
        return PeakWindow(fd)


class PeakWindow:
    """memory.peak as seen through one file descriptor: the peak since that descriptor's reset."""

    def __init__(self, fd):
        self.fd = fd

    def close(self):
        """The peak in bytes since the reset; closes the descriptor."""
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            return int(os.read(self.fd, 64).strip())
        except (OSError, ValueError):
            return None
        finally:
            os.close(self.fd)


def find_cgroup(host_total, root=CGROUP_ROOT, proc_self_cgroup=PROC_SELF_CGROUP):
    """The cgroup whose memory limit is the tightest one below host_total, or None (no container, no
    limit, or not Linux when the default paths are used)."""
    best = None
    for directory, version in _candidates(root, proc_self_cgroup):
        limit = _int(os.path.join(directory, _FILES[version]["limit"]))
        if limit is not None and 0 < limit < host_total and (best is None or limit < best.limit):
            best = CgroupMemory(directory, version, limit)
    return best


class ProcessMemory:
    """The process RSS against the host's RAM (no container limit applies)."""

    kind = "process"
    version = None

    def __init__(self):
        import psutil

        self._psutil = psutil
        self._process = psutil.Process()
        self.limit = psutil.virtual_memory().total

    def describe(self):
        return "process RSS (psutil), no container memory limit found"

    def read(self):
        """{"ram": RSS, "ram_limit": host RAM, "swap": host swap in use}. macOS has no OOM kill: a run
        past the RAM limit falls into swap, so swap is part of the reading."""
        return {"ram": self._process.memory_info().rss, "ram_limit": self.limit, "swap": self._psutil.swap_memory().used}

    def oom_kills(self):
        return None

    def open_peak_window(self):
        return None


def ram_source(root=CGROUP_ROOT, proc_self_cgroup=PROC_SELF_CGROUP):
    """CgroupMemory inside a memory-limited container, else ProcessMemory (macOS has no
    /proc/self/cgroup, so it always gets the process reading)."""
    process = ProcessMemory()
    return find_cgroup(process.limit, root, proc_self_cgroup) or process


class CudaMemory:
    """torch's CUDA allocator counters of the current device, plus NVML when importable."""

    kind = "cuda"

    def __init__(self, nvml=None):
        self.device = torch.cuda.current_device()
        self.nvml = nvml

    def describe(self):
        name = torch.cuda.get_device_name(self.device)
        extra = "NVML for device use and GPU load" if self.nvml else "NVML not installed (pip package nvidia-ml-py): no GPU load, device use from cudaMemGetInfo"
        return f"CUDA {name}: torch allocator counters; {extra}"

    def read(self):
        out = {"vram": torch.cuda.memory_allocated(self.device), "vram_reserved": torch.cuda.memory_reserved(self.device)}
        if self.nvml is not None:
            out.update(self.nvml.read())
        else:
            free, total = torch.cuda.mem_get_info(self.device)
            out.update({"vram_device": total - free, "vram_total": total})
        return out

    def reset_peak(self):
        torch.cuda.reset_peak_memory_stats(self.device)

    def peak(self):
        return {"vram_peak": torch.cuda.max_memory_allocated(self.device),
                "vram_reserved_peak": torch.cuda.max_memory_reserved(self.device)}


class MpsMemory:
    """torch's MPS counters (Apple silicon: the GPU shares the RAM). MPS keeps no peak; the sampler's
    maximum stands in for it."""

    kind = "mps"

    def describe(self):
        return "MPS (unified memory): torch.mps allocator counters, no device-wide use or GPU load"

    def read(self):
        out = {"vram": torch.mps.current_allocated_memory(), "vram_reserved": torch.mps.driver_allocated_memory()}
        if hasattr(torch.mps, "recommended_max_memory"):
            out["vram_total"] = torch.mps.recommended_max_memory()
        return out

    def reset_peak(self):
        pass

    def peak(self):
        return {}


class Nvml:
    """Device-wide memory and load of one GPU through pynvml."""

    def __init__(self, pynvml, handle):
        self._nvml, self._handle = pynvml, handle

    def read(self):
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        return {"vram_device": mem.used, "vram_total": mem.total, "gpu_util": util.gpu}


def open_nvml(device):
    """Nvml for torch's CUDA device, or None when no binding is importable or NVML fails to start."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}") if uuid is not None else pynvml.nvmlDeviceGetHandleByIndex(device)
        return Nvml(pynvml, handle)
    except Exception:  # pynvml raises its own NVMLError family; no NVML means the torch counters only
        return None


def gpu_source():
    """CudaMemory, MpsMemory or None. CUDA is read only once something initialised it (ComfyUI does
    at startup): a read before that would create a CUDA context of its own."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return CudaMemory(open_nvml(torch.cuda.current_device()))
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return MpsMemory()
    return None

"""The monitor's settings, kept in the user directory (never in a workflow).

`enabled` mirrors the ComfyUI setting "BCNodes.ProcessMonitor.Enabled": the frontend posts every
change, and this copy lets the monitor start with ComfyUI when no browser is open (a pod queued
through the API). On by default, like the ComfyUI setting. The rest are the modal's Settings tab.
`drop_stale_outputs` (drop_stale.py) applies with the monitor on or off.
"""

import json
import os
from dataclasses import asdict, dataclass, fields


@dataclass
class MonitorSettings:
    enabled: bool = True
    black_box: bool = True
    threshold: float = 0.85  # fraction of the RAM limit that triggers the snapshot (and the stop)
    stop_at_threshold: bool = False  # experimental: interrupt the prompt at the threshold
    keep_runs: int = 20  # run logs kept (pod disks are small)
    drop_stale_outputs: bool = False  # each prompt frees the cached outputs it does not use (classic cache)


_RANGES = {"threshold": (0.5, 0.99), "keep_runs": (1, 500)}


def apply(settings, changes):
    """A copy of settings with `changes` applied; ValueError naming the field and its valid values."""
    current = asdict(settings)
    for name, value in changes.items():
        if name not in current:
            raise ValueError(f"unknown monitor setting {name!r}; known: {', '.join(current)}")
        kind = type(current[name])
        if kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true or false, got {value!r}")
        else:
            try:
                value = kind(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a number, got {value!r}") from None
            low, high = _RANGES[name]
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}, got {value}")
        current[name] = value
    return MonitorSettings(**current)


def load(path):
    """The saved settings, or the defaults when there is no file. A file that cannot be parsed raises:
    it was written by this module, so a broken one is worth seeing."""
    if not os.path.isfile(path):
        return MonitorSettings()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    known = {f.name for f in fields(MonitorSettings)}
    return apply(MonitorSettings(), {k: v for k, v in data.items() if k in known})


def save(path, settings):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=1)
    os.replace(tmp, path)

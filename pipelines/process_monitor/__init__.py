"""Process Monitor: what a workflow uses (RAM, VRAM, time), live, as an estimate before a run,
measured per node during an armed run, and why a killed run died; and the full clear, which brings
RAM and VRAM back to the reading taken when ComfyUI started.

    settings   the persisted settings (the on/off toggle, black box, threshold, stop, rotation)
    blackbox   the run log (one JSON line per record), rotation, the Last run and Crash reports
    hook       the execution.py hook point: detection, install, the per-node wrapper
    monitor    the controller: sampler thread, runs, per-node records, threshold snapshot
    emulate    the estimate before a run, from the prompt, safetensors headers and cost profiles
    clear      the full clear: its steps, each measured, against the baseline read at startup

Nothing here imports ComfyUI at module level; the node layer (nodes/process_monitor.py) hands in
the server, the folders and the node classes.
"""

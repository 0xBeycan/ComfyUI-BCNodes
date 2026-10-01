# Process Monitor

Not a node: server code (`nodes/process_monitor.py` over `pipelines/process_monitor/`) and `web/js/process_monitor.js`. It shows what a workflow uses — RAM, VRAM, time — live, as an estimate before a run, measured per node during a run, and explains a run that was killed.

**On / off.** One ComfyUI setting, *Settings → BCNodes → Process Monitor*, on by default, applied live, no restart. Off means no thread, no hook into the executor and no file writes. The server keeps a copy of the setting, so a ComfyUI started without a browser (a pod queued through the API) still runs the monitor when it was left on, or never set.

**Top bar.** RAM against its limit, VRAM and the GPU load, once a second, while the monitor is on; and a `PM` button that opens the modal. The button is there with the monitor off too (Emulate and the crash report do not need it on) and turns red when the last run was killed.

| Tab | What it shows |
| --- | --- |
| Live | The current values in detail, and the sources: the RAM source (the container's cgroup, or the process RSS outside a container), the VRAM source, and whether per-node measurement is available |
| Emulate | An estimate of the current workflow, labelled *rough estimate; a real measurement exists only after the workflow has run once* (below) |
| Last run | The last finished run: status, run time, the monitor's own time (`run 2.9 s, monitor 4.9 ms (0.17%)`), RAM / VRAM peaks, and the per-node table of an armed run. A row click selects and centres the node; a node inside a subgraph centres its subgraph node |
| Crash | The report of a run that ended without an end record (below) |
| Settings | Black box on / off, the snapshot threshold, the experimental stop, how many run logs are kept |

**Sources.**

- RAM: inside a memory-limited container, the cgroup (v2 `memory.current` / `memory.max` / `memory.peak` / `memory.events`, or the v1 files of the same counters) — its working set, usage minus the inactive file cache, as `docker stats` shows it. The host's RAM is never used there: it hides the limit the kernel kills at. Outside a container (macOS), the process RSS and the host's swap.
- VRAM: torch's allocator counters (CUDA, or MPS on Apple silicon). NVML adds the device-wide use (every process) and the GPU load when a binding is installed (`pip install nvidia-ml-py`); it is optional, and without it the Live tab says so and the bar uses torch's counters.

**Black box.** While a prompt runs, one JSON line every 100 ms: RAM and its limit, VRAM, the prompt, the executing node and the Python line the execution thread is on. With per-node measurement available, a line at every node start too: its inputs (shape, dtype, device, bytes), RAM, and the output cache total. Lines are written as they happen (no `fsync`: what the kernel holds survives a process kill), one file per run, in `user/BCNodes/process_monitor/runs/`; the last 20 runs are kept (a setting).

**Threshold snapshot.** Once per run, when RAM crosses 85% of its limit (a setting): first the execution thread's stack and the tensors its locals hold, written at once, then every live tensor of the process grouped by shape / dtype / device with their bytes (`73 × (720, 1280, 3) float32, 11.1 MB each`). Limit: when RAM jumps from below the threshold to the kill within one 100 ms sample, the report names the node and the line but no tensors.

**After a kill.** A RAM OOM is a `SIGKILL`: nothing runs at the end. After the restart the newest run log without an end record is the crash: the report names the node, the line, RAM at the node's start, the growth curve, the tensors alive at the snapshot and the output cache total. The cgroup's `oom_kill` counter confirms an OOM kill (it rose since the run started), or says it was not one. macOS has no OOM kill: there the report says when the run *fell into swap* instead. A VRAM OOM is an ordinary exception: ComfyUI survives it and the run log ends normally.

**Per-node measurement.** Armed from the Last run tab (*Measure next run*), for one run. Per node: time, RAM peak (`memory.peak` reset per node where the kernel allows it, otherwise the 100 ms sampler's maximum; the table says which), VRAM peak, outputs (shape, dtype, device, bytes), the output cache total, models loaded and offloaded. A node served from the cache reads *from cache, not measured*, never 0; the report says whether models were already loaded at the start (first-run and repeat-run profiles differ). It hooks ComfyUI's executor and needs ComfyUI **0.17.0** or newer; on an older ComfyUI it is off and the modal says so, while the bars, Emulate and the black box keep working. An error inside the monitor turns the measurement off with a message; the workflow is never affected.

**Emulate.** Built per component from the workflow as the frontend sends it: model weights from the safetensors headers (no model load), tensors exact from their sizes, every output kept in RAM until the prompt ends (ComfyUI-BCVideoNodes' WanAnimate Preprocess `final_mask` and `bg_images` only when something is connected to them: the node makes them only then), and each node's own transients (copies, lists before a stack or concat) from a cost profile worked out from that node's code. A node without a profile is listed as *not counted*, never guessed; bypassed and muted nodes are listed, not added; subgraphs are expanded. A video workflow also gets a table of resolution (480p / 720p / 1080p) by frame count. The fit check sets the estimate against a 24 GB and a 32 GB GPU and the RAM limit. After an armed run, that workflow's measured nodes replace the formulas, scaled to other sizes; the measurements live in `user/BCNodes/process_monitor/measurements/`, never in the workflow.

**Stop at the threshold** (experimental, off by default) interrupts the prompt when the snapshot is taken. It only works inside nodes that check ComfyUI's interrupt (between sampler steps, for example); a running `torch.stack` or `np.fromiter` cannot be stopped.

**Its own cost.** The monitor counts its own time (the hook on the execution thread, the sampler's CPU time, the snapshot) and reports it with every run. Measured on a 12-node CPU workflow of 2.9 s: 4.2 ms with the black box, 4.9 ms armed (0.15–0.17%).

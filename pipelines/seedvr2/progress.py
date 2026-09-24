"""The progress bar and the timed log lines of the SeedVR2 VAE slice loops."""

import logging
import time

import torch

# Seconds between progress log lines of the VAE slice loops.
LOG_EVERY_SECONDS = 10


class Progress:
    """ComfyUI's progress bar plus a log line at the start, every LOG_EVERY_SECONDS and at
    the end: slices done, the tile and the frame it has reached, driver-reported VRAM in
    use, elapsed time and the time left at the running rate (edge tiles are smaller than
    interior ones, so the estimate is rough on a tiled run)."""

    def __init__(self, label, n_tiles, n_slices, t_frames, device):
        from comfy.utils import ProgressBar

        self.label, self.n_tiles, self.n_slices, self.t_frames, self.device = label, n_tiles, n_slices, t_frames, device
        self.total = n_tiles * n_slices
        self.bar = ProgressBar(self.total)
        self.done = self.tile = 0
        self.started = self.last_log = time.monotonic()
        logging.info("%s: %d frames, %d tile(s) x %d slice(s)", label, t_frames, n_tiles, n_slices)

    def begin_tile(self):
        self.tile += 1

    def step(self, frame_end):
        """One slice done; `frame_end` is the frame the current tile has reached."""
        self.done += 1
        self.bar.update(1)
        now = time.monotonic()
        if now - self.last_log < LOG_EVERY_SECONDS and self.done < self.total:
            return
        self.last_log = now
        elapsed = now - self.started
        vram = ""
        if self.device.type == "cuda":
            free, total = torch.cuda.mem_get_info(self.device)
            vram = f", VRAM {(total - free) / 2 ** 30:.1f} GiB used"
        logging.info("%s: slice %d/%d, tile %d/%d, frame %d/%d%s, %.0f s elapsed, ~%.0f s left", self.label, self.done, self.total,
                     self.tile, self.n_tiles, frame_end, self.t_frames, vram, elapsed, elapsed / self.done * (self.total - self.done))

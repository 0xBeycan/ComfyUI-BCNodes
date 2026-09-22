"""SeedVR2 nodes for ComfyUI's native SeedVR2 graph.

    BC_SeedVR2Resize      (SeedVR2 Resize)       original image -> the frame the VAE encodes + colour reference
    BC_SeedVR2VAEEncode   (SeedVR2 VAE Encode)   VAE Encode (Tiled) with the frames streamed from RAM
    BC_SeedVR2VAEDecode   (SeedVR2 VAE Decode)   VAE Decode (Tiled) with the decoded frames streamed to RAM
    BC_SeedVR2PostProcess (SeedVR2 PostProcess)  Post-Process SeedVR2 Output, one frame at a time

Resize is the whole input stage of the SeedVR2 upscale graph in one node.
The encode, decode and post-process nodes follow ComfyUI's own
comfy/ldm/seedvr/vae.py and comfy_extras/nodes_seedvr.py (GPL-3.0) step for
step; what they change is where the frames live, not what is computed. Encode
and decode unload every other model first when the free VRAM is below the
VAE's working set (ComfyUI will not evict one dynamic model for another).

Resize — from the original image to the frame the VAE encodes:

  1. downscale by `downscale_factor`, the way ComfyUI's ImageScaleBy(lanczos)
     does it: `round(side * factor)`, PIL LANCZOS on 8-bit (comfy/utils.py
     `lanczos`); 1.0 skips it;
  2. `resolution = min(width, height) * upscale_factor` from the ORIGINAL
     image;
  3. shortest-edge resize plus clamp:
     `TVF.resize` to `resolution` on the shortest edge (torchvision floors the
     long edge), BICUBIC with antialias, a second resize to
     `round(edge * max_resolution / longest)` when the longest edge exceeds
     `max_resolution`, then clamp to [0, 1];
  4. zero-pad bottom / right to a multiple of 16 and, for a frame batch,
     repeat the last frame up to a 4n+1 frame count (the same arithmetic as
     comfy_extras/nodes_seedvr.py:75-90), so `image` goes straight to
     VAE Encode.

Step 3 runs twice: on the VAE device in bfloat16 for the frame that gets
encoded, and on the CPU in float32 for the colour-correction reference, which
is then cropped to the even output size. `image` is the first (padded),
`reference` the second, with the original frame count — the post-process node drops the
repeated frames against it. Frames are processed a few at a time so a long
video batch never has to fit on the GPU at once. Both outputs are stored as
float16: VAE Encode casts `image` to the VAE's float16 anyway (comfy/sd.py
process_input), and `reference` only feeds the colour transfer, where a
float16 rounding of the reference moves the result by ~0.1/255. PIL and
torchvision are imported inside the function.

VAE Encode — ComfyUI's VAE Encode (Tiled) moves the whole clip to the GPU
first (comfy/sd.py _encode_tiled_owned: `process_input(...).to(dtype).to(device)`,
after a full float32 copy in RAM), then encodes it slice by slice
(vae.py slicing_encode). VRAM grows with the frame count on top of the
encoder's fixed working set. This node runs the same slice loop with the
same causal memory cache and the same `x * 2 - 1`, but builds each input
slice on the GPU only when its turn comes. The posterior mode, the crop to
the latent size and the `* scaling_factor` follow the native path. One
spatial tile.

VAE Decode — ComfyUI's VAE Decode (Tiled) on the SeedVR2 VAE decodes the
whole clip in one call and keeps every decoded frame on the GPU until the end
(vae.py slicing_decode: the slices are collected and torch.cat'ed), so VRAM
grows with the frame count — a 5090 tops out near 190 frames at 1080p. This
node runs the same slice loop with the same causal memory cache, but moves
each decoded slice to RAM as soon as it exists, after the same
`/scaling_factor`, even crop and `(x + 1) / 2` clamp. VRAM is then the
decoder's fixed working set, whatever the length. The frames are stored as
float16, which is what the float16 VAE produced; the native node only upcasts
them. One spatial tile: the whole frame goes through the decoder at once, as
the native node does for a frame smaller than its tile.

PostProcess — Post-Process SeedVR2 Output builds five full-size float32
copies of the clip on the way through (raw range conversion, the flattening
reshape, the result buffer and the add/div/clamp chain), which is what runs a
30-second 1080p clip out of RAM. This node does the same operations in the
same order — crop to the reference, `x * 2 - 1`, the colour transfer from
comfy/ldm/seedvr/color_fix.py on the VAE device, `(x + 1) / 2` clamp, alpha
from the reference, even crop — one frame at a time into a single preallocated
float16 output. The colour maths itself stays float32 per frame, exactly as in
the native node, which also processes lab frame by frame.
"""

import logging
import time

import numpy as np
import torch

PAD_MULTIPLE = 16
FRAMES_PER_CHUNK = 4
LATENT_CHANNELS = 16
# Working set of the SeedVR2 VAE per tile pixel, in bytes, independent of the
# frame count. Measured on a 5090 at 1088x1920 (float16, one tile): the
# encoder's first slice needs more than 28.5 GB (13.6 KB/px; it ran out of
# memory there in a process holding nothing else), the decoder is in the same
# class and not measured. Biased 20% high: an under-estimate kills the run,
# an over-estimate costs a DiT reload.
DECODER_BYTES_PER_PIXEL = 17_000
ENCODER_BYTES_PER_PIXEL = 16_300
# Slices between allocator-pool trims in the VAE slice loops: fragmentation built up over
# ~100 slices in the run that showed it; a trim costs a free/re-alloc of the working set.
TRIM_EVERY = 16
# Seconds between progress log lines of the VAE slice loops.
LOG_EVERY_SECONDS = 10


def _torch_device():
    import comfy.model_management

    return comfy.model_management.get_torch_device()


def lanczos_scale_by(image_bhwc, factor):
    """ImageScaleBy(lanczos, factor): comfy.utils.lanczos on 8-bit PIL frames."""
    from PIL import Image

    height, width = image_bhwc.shape[1], image_bhwc.shape[2]
    size = (round(width * factor), round(height * factor))
    frames = []
    for frame in image_bhwc:
        pil = Image.fromarray(np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8))
        frames.append(torch.from_numpy(np.array(pil.resize(size, resample=Image.Resampling.LANCZOS)).astype(np.float32) / 255.0))
    return torch.stack(frames).to(image_bhwc.device, image_bhwc.dtype)


def side_resize(frames, resolution, max_resolution):
    """Shortest-edge resize on a (B, C, H, W) tensor, followed by the clamp."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF

    resized = TVF.resize(frames, resolution, InterpolationMode.BICUBIC, antialias=True)
    if max_resolution > 0:
        h, w = resized.shape[-2:]
        if max(h, w) > max_resolution:
            scale = max_resolution / max(h, w)
            resized = TVF.resize(resized, (round(h * scale), round(w * scale)), InterpolationMode.BICUBIC, antialias=True)
    return torch.clamp(resized, 0.0, 1.0)


def divisible_pad(frames, multiple=PAD_MULTIPLE):
    """Zeros on the bottom / right up to a multiple."""
    h, w = frames.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return frames
    return torch.nn.functional.pad(frames, (0, pad_w, 0, pad_h), mode="constant", value=0.0)


def frames_to_4n1(t):
    """Frames to append so a batch of t frames is 4n+1 (Pre-Process SeedVR2 Input `cut_videos`)."""
    if t == 1:
        return 0
    if t <= 4:
        return 4 - t + 1
    return (4 - (t - 1) % 4) % 4


class SeedVR2Resize:
    """Original image -> the padded frame the VAE encodes, and its colour reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_factor": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 16.0, "step": 0.01,
                                             "tooltip": "Shortest edge of the output = shortest edge of the input × this "
                                                        "(the resize resolution, computed from the original image)."}),
                "downscale_factor": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                               "tooltip": "Lanczos downscale applied first, like ImageScaleBy(lanczos). 1 = none."}),
                "max_resolution": ("INT", {"default": 4096, "min": 0, "max": 16384, "step": 2,
                                           "tooltip": "Cap on the longest edge, 0 = none."}),
                "emulate_bf16": ("BOOLEAN", {"default": True,
                                             "tooltip": "Resize `image` in bfloat16 on the GPU when CUDA is available. "
                                                        "Without CUDA the resize runs in float32."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "reference")
    OUTPUT_TOOLTIPS = (
        "The frames the VAE encodes: downscaled, resized, clamped, padded to a multiple of 16 and to 4n+1 frames, float16. Wire to VAE Encode.",
        "The colour-correction reference: float32 resize stored as float16, cropped to even, not padded. Wire to SeedVR2 PostProcess.",
    )
    FUNCTION = "resize"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "resize", "shortest edge", "upscale", "downscale", "pad"]

    def resize(self, image, upscale_factor, downscale_factor, max_resolution, emulate_bf16):
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] == 0:
            raise ValueError("BC_SeedVR2Resize: connect an image batch (B, H, W, C)")
        image = image[..., :3]
        resolution = int(round(min(image.shape[1], image.shape[2]) * upscale_factor))
        if downscale_factor != 1.0:
            image = lanczos_scale_by(image, downscale_factor)
        frames = image.permute(0, 3, 1, 2)

        device = torch.device("cpu")
        vae_dtype = torch.float32
        if emulate_bf16:
            device = _torch_device()
            if device.type == "cuda":
                vae_dtype = torch.bfloat16
            else:
                logging.warning("BC_SeedVR2Resize: emulate_bf16 needs CUDA (device is %s), resizing in float32", device)
                device = torch.device("cpu")

        t = frames.shape[0]
        extra = frames_to_4n1(t)
        out_image = out_reference = None
        for start in range(0, t, FRAMES_PER_CHUNK):
            chunk = frames[start:start + FRAMES_PER_CHUNK]
            resized = divisible_pad(side_resize(chunk.to(device=device, dtype=vae_dtype), resolution, max_resolution))
            resized = resized.to(device="cpu", dtype=torch.float16).permute(0, 2, 3, 1)
            reference = side_resize(chunk.to(device="cpu", dtype=torch.float32), resolution, max_resolution)
            h, w = reference.shape[-2:]
            reference = reference[:, :, :(h // 2) * 2, :(w // 2) * 2].permute(0, 2, 3, 1).to(torch.float16)
            if out_image is None:
                out_image = torch.empty((t + extra,) + tuple(resized.shape[1:]), dtype=torch.float16)
                out_reference = torch.empty((t,) + tuple(reference.shape[1:]), dtype=torch.float16)
            out_image[start:start + resized.shape[0]] = resized
            out_reference[start:start + reference.shape[0]] = reference
        if extra:
            out_image[t:] = out_image[t - 1]
        return (out_image, out_reference)


def _seedvr2_vae_model(vae):
    model = getattr(vae, "first_stage_model", None)
    if type(model).__name__ != "VideoAutoencoderKLWrapper":
        raise ValueError(f"SeedVR2 VAE Encode/Decode: needs the SeedVR2 VAE, got {type(model).__name__}")
    return model


def _tile_plan(h, w, tile, overlap, device):
    """The spatial tiles of comfy/ldm/seedvr/vae.py tiled_vae (lines 137-146) on a grid of
    `h` x `w` cells, and the cosine ramp used for their blend weights."""
    stride = max(1, tile - overlap)
    ranges = []
    for y in range(0, h, stride):
        y_end = min(y + tile, h)
        if y > 0 and (y_end - y) <= overlap:
            continue
        for x in range(0, w, stride):
            x_end = min(x + tile, w)
            if x > 0 and (x_end - x) <= overlap:
                continue
            ranges.append((y, y_end, x, x_end))

    ramps = {}

    def ramp(steps):
        if steps not in ramps:
            t = torch.linspace(0, 1, steps=steps, device=device, dtype=torch.float32)
            ramps[steps] = 0.5 - 0.5 * torch.cos(t * torch.pi)
        return ramps[steps]

    return ranges, ramp


def _tile_weight(y, y_end, x, x_end, h, w, th, tw, edge_h, edge_w, ramp, device):
    """tiled_vae lines 182-207: cosine fades on interior edges only, separable."""
    cur_ov_h = max(0, min(edge_h, th // 2))
    cur_ov_w = max(0, min(edge_w, tw // 2))
    w_h = torch.ones((th,), device=device)
    w_w = torch.ones((tw,), device=device)
    if cur_ov_h > 0:
        r = ramp(cur_ov_h)
        if y > 0:
            w_h[:cur_ov_h] = r
        if y_end < h:
            w_h[-cur_ov_h:] = 1.0 - r
    if cur_ov_w > 0:
        r = ramp(cur_ov_w)
        if x > 0:
            w_w[:cur_ov_w] = r
        if x_end < w:
            w_w[-cur_ov_w:] = 1.0 - r
    return w_h.view(1, 1, 1, -1, 1) * w_w.view(1, 1, 1, 1, -1)


TILED_INPUTS = {
    "tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32, "advanced": True,
                          "tooltip": "Spatial tile in pixels, as VAE Encode/Decode (Tiled). A tile that covers the frame means no tiling."}),
    "overlap": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 32, "advanced": True}),
    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "advanced": True,
                              "tooltip": "Ignored, as it is by the SeedVR2 VAE in VAE Encode/Decode (Tiled): the causal VAE slices time itself."}),
    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "advanced": True,
                                 "tooltip": "Ignored, as it is by the SeedVR2 VAE in VAE Encode/Decode (Tiled)."}),
}


def _free_vram(device):
    """Free VRAM in bytes as the driver reports it, after handing cached blocks back.

    ComfyUI's get_free_memory adds `reserved - active` from torch.cuda.memory_stats,
    and under the cudaMallocAsync backend (ComfyUI's default on CUDA) `active` stays
    0, so everything the pool holds — live model weights included — reads as free.
    After a trim, mem_get_info is honest under both allocators."""
    import comfy.model_management as mm

    mm.soft_empty_cache()
    if device.type == "cuda":
        return torch.cuda.mem_get_info(device)[0]
    return mm.get_free_memory(device)


class _Progress:
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


def _make_room_for_vae(vae, needed):
    """Load the VAE with `needed` bytes of VRAM free for its working set.

    ComfyUI's load_models_gpu will not evict one "dynamic" model (the DiT) for
    another (the VAE): free_memory(for_dynamic=True) skips them
    (comfy/model_management.py). At 1080p the decoder's working set plus a
    resident DiT does not fit a 32 GB card, so when the free VRAM is short
    everything is unloaded first; the DiT is staged in RAM and comes back on
    demand, the VAE is reloaded right here."""
    import comfy.model_management as mm

    free = _free_vram(vae.device)
    if free < needed:
        mm.unload_all_models()
        after = _free_vram(vae.device)
        logging.info("SeedVR2 VAE: %.1f GiB free, working set needs %.1f GiB -> unloaded all models, %.1f GiB free now",
                     free / 2 ** 30, needed / 2 ** 30, after / 2 ** 30)
    else:
        logging.info("SeedVR2 VAE: %.1f GiB free, working set needs %.1f GiB -> models stay", free / 2 ** 30, needed / 2 ** 30)
    mm.load_models_gpu([vae.patcher], memory_required=needed, force_full_load=vae.disable_offload)


class SeedVR2VAEEncode:
    """VAE Encode (Tiled) for the SeedVR2 VAE, streaming frames from RAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE", {"tooltip": "Frames from SeedVR2 Resize `image` (padded to /16 and 4n+1 frames)."}),
                "vae": ("VAE",),
                **TILED_INPUTS,
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "vae encode", "video", "streaming", "tiled"]

    def encode(self, pixels, vae, tile_size, overlap, temporal_size=64, temporal_overlap=8):
        import comfy.model_management as mm
        from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
        from comfy.ldm.seedvr.vae import MemoryState

        model = _seedvr2_vae_model(vae)
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[0] == 0:
            raise ValueError("BC_SeedVR2VAEEncode: pixels must be an image batch (B, H, W, C)")
        pixels = pixels[..., :3]  # comfy/sd.py vae_encode_crop_pixels: output_channels = 3
        n, height, width = pixels.shape[0], pixels.shape[1], pixels.shape[2]
        target_t, target_h, target_w = (n + 3) // 4, (height + 7) // 8, (width + 7) // 8  # vae.py tiled_vae encode targets
        overlap = min(overlap, max(0, tile_size - 8))  # vae.py encode_tiled
        single_tile = height <= tile_size and width <= tile_size
        working_set = ENCODER_BYTES_PER_PIXEL * (height * width if single_tile else min(height, tile_size) * min(width, tile_size))
        _make_room_for_vae(vae, working_set)
        device = vae.device

        def gpu_slice(start, end, y0=0, y1=height, x0=0, x1=width):
            """Frames [start, end) of the tile as (1, 3, t, th, tw) on the GPU, through VAE.process_input."""
            chunk = pixels[start:end, y0:y1, x0:x1].movedim(-1, 1).permute(1, 0, 2, 3).unsqueeze(0)
            return (chunk * 2.0 - 1.0).to(vae.vae_dtype).to(device)  # comfy/sd.py process_input, then the dtype/device cast

        # vae.py slicing_encode: a first slice of 1 + split frames, then `split` frames each, a runt merged into the previous.
        sliced = model.use_slicing and (n - 1) > model.slicing_sample_min_size
        if sliced:
            split = max(model.slicing_sample_min_size, getattr(model, "temporal_downsample_factor", 1))
            bounds = list(range(1, n, split)) + [n]
            slices = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
            if len(slices) > 1 and slices[-1][1] - slices[-1][0] < getattr(model, "temporal_downsample_factor", 1):
                slices[-2] = (slices[-2][0], slices[-1][1])
                slices.pop()
            slices[0] = (0, slices[0][1])
        else:
            slices = [(0, n)]

        def encode_tile(y0, y1, x0, x1, sink, progress):
            """The slice loop of slicing_encode over one spatial tile; every latent slice goes to `sink` at once."""
            memory_cache = {}
            for i, (start, end) in enumerate(slices):
                if not sliced:
                    h = model._encode(gpu_slice(start, end, y0, y1, x0, x1))
                else:
                    if i > 0 and i % TRIM_EVERY == 0:
                        mm.soft_empty_cache()  # the slice loop fragments the allocator pool; hand freed blocks back now and then
                    state = MemoryState.INITIALIZING if i == 0 else MemoryState.ACTIVE
                    h = model._encode(gpu_slice(start, end, y0, y1, x0, x1), memory_state=state, memory_cache=memory_cache)
                sink(i, h)
                progress.step(end)

        with mm.cuda_device_context(device):
            model.device = device  # vae.py _encode_with_raw_latent
            if single_tile:
                parts = []
                progress = _Progress("SeedVR2 VAE Encode", 1, len(slices), n, device)
                progress.begin_tile()
                encode_tile(0, height, 0, width, lambda i, h: parts.append(torch.chunk(h, 2, dim=1)[0].to("cpu")), progress)
                z = torch.cat(parts, dim=2)
            else:
                ranges, ramp = _tile_plan(height, width, tile_size, overlap, device)
                edge_h = edge_w = overlap // 8  # tiled_vae encode: fades `overlap // 8` latent cells wide
                result = count = None
                progress = _Progress("SeedVR2 VAE Encode", len(ranges), len(slices), n, device)
                for y0, y1, x0, x1 in ranges:
                    progress.begin_tile()
                    ys, xs = y0 // 8, x0 // 8
                    weight = None
                    offsets = []

                    def sink(i, h, ys=ys, xs=xs, y0=y0, y1=y1, x0=x0, x1=x1):
                        nonlocal result, count, weight
                        tile = torch.chunk(h, 2, dim=1)[0]  # vae.py DiagonalGaussianDistribution.mode
                        th, tw = tile.shape[3], tile.shape[4]
                        if weight is None:
                            weight = _tile_weight(y0, y1, x0, x1, height, width, th, tw, edge_h, edge_w, ramp, device)
                        if result is None:
                            result = torch.zeros((1, LATENT_CHANNELS, target_t, target_h, target_w), dtype=torch.float32)
                            count = torch.zeros((1, 1, 1, target_h, target_w), dtype=torch.float32)
                        t0 = sum(offsets)
                        tile = tile[:, :, : max(0, target_t - t0)]
                        tile.mul_(weight)  # tiled_vae: the fade is applied in the tile's own dtype
                        result[:, :, t0:t0 + tile.shape[2], ys:ys + th, xs:xs + tw] += tile.to("cpu", torch.float32)
                        if t0 == 0:
                            count[:, :, :, ys:ys + th, xs:xs + tw] += weight.to("cpu")
                        offsets.append(tile.shape[2])

                    encode_tile(y0, y1, x0, x1, sink, progress)
                z = result.div_(count.clamp(min=1e-6)).to(vae.vae_dtype)  # tiled_vae: normalised, returned in the input dtype
        z = z[:, :, :target_t, :target_h, :target_w].to(torch.float32).contiguous() * BYTEDANCE_VAE_SCALING_FACTOR  # crop, VAE output dtype, comfy_format_encoded
        return ({"samples": z},)


class SeedVR2VAEDecode:
    """VAE Decode (Tiled) for the SeedVR2 VAE, streaming decoded frames to RAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                **TILED_INPUTS,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Decoded frames (B*T, H, W, 3), float16, values in [0, 1].",)
    FUNCTION = "decode"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "vae decode", "video", "streaming", "tiled"]

    def decode(self, samples, vae, tile_size, overlap, temporal_size=64, temporal_overlap=8):
        import comfy.model_management as mm
        from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR, BYTEDANCE_VAE_SHIFTING_FACTOR
        from comfy.ldm.seedvr.vae import MemoryState

        model = _seedvr2_vae_model(vae)
        z = samples["samples"]
        if z.ndim != 5 or z.shape[1] != LATENT_CHANNELS:
            raise ValueError(f"BC_SeedVR2VAEDecode: expected a SeedVR2 latent (B, {LATENT_CHANNELS}, T, H, W), got {tuple(z.shape)}")
        b, _, t_latent, h, w = z.shape
        t_pixel = max(1, t_latent * 4 - 3)  # comfy/sd.py upscale_ratio for this VAE
        if tile_size < overlap * 4:  # nodes.py VAEDecodeTiled
            overlap = tile_size // 4
        tile_lat = max(1, tile_size // 8)  # vae.py decode_tiled -> tiled_vae(encode=False): latent tile and overlap
        ov_lat = min(max(0, (min(overlap, max(0, tile_size - 8))) // 8), tile_lat - 1)
        single_tile = h <= tile_lat and w <= tile_lat
        tile_px_h, tile_px_w = (h * 8, w * 8) if single_tile else (min(h, tile_lat) * 8, min(w, tile_lat) * 8)
        _make_room_for_vae(vae, DECODER_BYTES_PER_PIXEL * tile_px_h * tile_px_w)
        device = vae.device
        out = None

        def finish(frames, t0):
            """A run of decoded frames (B, 3, t, H, W) in [-1, 1] on the GPU -> rows [t0, t0 + t) of `out` on the CPU."""
            nonlocal out
            frames = vae.process_output(frames.float())  # (x + 1) / 2, clamp, as VAE.decode does
            height, width = frames.shape[-2:]
            height, width = height - height % 2, width - width % 2  # vae.py wrapper.decode even crop
            frames = frames[:, :, :, :height, :width].to(torch.float16).movedim(1, -1)
            if out is None:
                out = torch.empty((b, t_pixel, height, width, frames.shape[-1]), dtype=torch.float16)
            if t0 + frames.shape[1] > t_pixel:
                raise RuntimeError(f"BC_SeedVR2VAEDecode: decoder produced more than {t_pixel} frames")
            out[:, t0:t0 + frames.shape[1]] = frames.to("cpu")

        # slicing_decode: latent frame 0 rides with the first slice, then one slice per remaining frame.
        n_slices = t_latent - 1 if model.use_slicing and (t_latent - 1) > model.slicing_latent_min_size else 1

        def decode_tile(latent, sink, progress):
            """The slice loop of slicing_decode over one latent tile; every decoded slice goes to `sink` at
            once, and `sink` returns the frame the tile has reached."""
            if n_slices > 1:
                memory_cache = {}
                z_slices = latent[:, :, 1:].split(split_size=model.slicing_latent_min_size, dim=2)
                progress.step(sink(model._decode(torch.cat((latent[:, :, :1], z_slices[0]), dim=2),
                                                 memory_state=MemoryState.INITIALIZING, memory_cache=memory_cache)))
                for i in range(1, len(z_slices)):
                    if i % TRIM_EVERY == 0:
                        mm.soft_empty_cache()  # see SeedVR2VAEEncode
                    progress.step(sink(model._decode(z_slices[i], memory_state=MemoryState.ACTIVE, memory_cache=memory_cache)))
            else:
                progress.step(sink(model._decode(latent)))

        with mm.cuda_device_context(device):
            latent = z.to(vae.vae_dtype).to(device)
            latent = latent / BYTEDANCE_VAE_SCALING_FACTOR + BYTEDANCE_VAE_SHIFTING_FACTOR  # vae.py wrapper.decode
            model.device = device
            if single_tile:
                filled = [0]

                def sink(decoded):
                    finish(decoded, filled[0])
                    filled[0] += decoded.shape[2]
                    return filled[0]

                progress = _Progress("SeedVR2 VAE Decode", 1, n_slices, t_pixel, device)
                progress.begin_tile()
                decode_tile(latent, sink, progress)
                filled = filled[0]
            else:
                # tiled_vae(encode=False): tiles on the latent grid, blended on the pixel grid with
                # fades `ov_lat * 8` wide, normalised by the summed weights, cast to the VAE dtype.
                # The sum is kept in float16 here (native: float32 on the GPU): it holds the whole clip
                # in RAM, and float32 was 22 GB for 30 s at 1080p. Interior pixels get one term with
                # weight 1, so they are exact; on the overlap bands the float16 sum of up to four
                # already-float16 terms is within ~1.5e-3, a fifth of an 8-bit step after (x + 1) / 2.
                ranges, ramp = _tile_plan(h, w, tile_lat, ov_lat, device)
                edge_h = edge_w = ov_lat * 8  # tiled_vae decode: fades `ov_lat * 8` pixels wide
                result = torch.zeros((b, 3, t_pixel, h * 8, w * 8), dtype=torch.float16)
                count = torch.zeros((1, 1, 1, h * 8, w * 8), dtype=torch.float32)
                filled = 0
                progress = _Progress("SeedVR2 VAE Decode", len(ranges), n_slices, t_pixel, device)
                for y0, y1, x0, x1 in ranges:
                    progress.begin_tile()
                    ys, xs = y0 * 8, x0 * 8
                    weight = None
                    offsets = []

                    def sink(decoded, ys=ys, xs=xs, y0=y0, y1=y1, x0=x0, x1=x1, offsets=offsets):
                        nonlocal weight
                        th, tw = decoded.shape[3], decoded.shape[4]
                        if weight is None:
                            weight = _tile_weight(y0, y1, x0, x1, h, w, th, tw, edge_h, edge_w, ramp, device)
                        t0 = sum(offsets)
                        decoded = decoded[:, :, : max(0, t_pixel - t0)]
                        decoded.mul_(weight)  # tiled_vae: the fade is applied in the tile's own dtype
                        result[:, :, t0:t0 + decoded.shape[2], ys:ys + th, xs:xs + tw] += decoded.to("cpu")
                        if t0 == 0:
                            count[:, :, :, ys:ys + th, xs:xs + tw] += weight.to("cpu")
                        offsets.append(decoded.shape[2])
                        return t0 + decoded.shape[2]

                    decode_tile(latent[:, :, :, y0:y1, x0:x1], sink, progress)
                    filled = sum(offsets)
                logging.info("SeedVR2 VAE Decode: %d tiles decoded, normalising %d frames", len(ranges), t_pixel)
                count = count.clamp(min=1e-6).to(device)
                # Normalised in float32, then through the VAE dtype (tiled_vae returns `result.to(x.dtype)`) before the output range op.
                for t0 in range(0, t_pixel, FRAMES_PER_CHUNK):
                    chunk = result[:, :, t0:t0 + FRAMES_PER_CHUNK].to(device, torch.float32).div_(count)
                    finish(chunk.to(vae.vae_dtype), t0)
                del result, count
        if filled != t_pixel:
            raise RuntimeError(f"BC_SeedVR2VAEDecode: decoder produced {filled} frames, expected {t_pixel}")
        return (out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1]),)


class SeedVR2PostProcess:
    """Post-Process SeedVR2 Output, one frame at a time into one float16 output."""

    METHODS = ["lab", "wavelet", "adain", "none"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The generated frames."}),
                "original_resized_images": ("IMAGE", {"tooltip": "The reference frames (SeedVR2 Resize `reference`)."}),
                "color_correction_method": (cls.METHODS, {"default": "lab"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Aligned, colour-corrected frames, float16.",)
    FUNCTION = "process"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "color correction", "postprocess", "lab", "video"]

    def process(self, images, original_resized_images, color_correction_method):
        import comfy.model_management as mm
        from comfy.ldm.seedvr.color_fix import adain_color_transfer, lab_color_transfer, wavelet_color_transfer

        if color_correction_method not in self.METHODS:
            raise ValueError(f"BC_SeedVR2PostProcess: unknown color_correction_method {color_correction_method!r}")
        transfer = {"lab": lab_color_transfer, "wavelet": wavelet_color_transfer, "adain": adain_color_transfer}.get(color_correction_method)
        for name, x in (("images", images), ("original_resized_images", original_resized_images)):
            if not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[0] == 0:
                raise ValueError(f"BC_SeedVR2PostProcess: {name} must be an image batch (B, H, W, C)")

        # The same alignment as Post-Process SeedVR2 Output on 4-D inputs: alpha from the
        # reference, common frame count, crop to the common size, even output size.
        alpha = original_resized_images[..., 3:4] if original_resized_images.shape[-1] == 4 else None
        reference = original_resized_images[..., :3]
        t = min(images.shape[0], reference.shape[0])
        target_h = min(images.shape[1], reference.shape[1])
        target_w = min(images.shape[2], reference.shape[2])
        out_h, out_w = target_h - target_h % 2, target_w - target_w % 2
        out = torch.empty((t, out_h, out_w, 3 + (1 if alpha is not None else 0)), dtype=torch.float16)
        device = mm.vae_device()
        from comfy.utils import ProgressBar

        progress = ProgressBar(t)
        for i in range(t):
            decoded = images[i, :target_h, :target_w, :3]
            if transfer is None:
                frame = decoded
            else:
                ref = reference[i]
                if ref.shape[0] != target_h or ref.shape[1] != target_w:
                    ref = _resize_reference(ref, target_h, target_w)
                decoded_raw = decoded.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
                reference_raw = ref.to(device=device, dtype=torch.float32).permute(2, 0, 1)[None].mul(2.0).sub(1.0)
                frame = transfer(decoded_raw, reference_raw)[0].permute(1, 2, 0).add(1.0).div(2.0).clamp(0.0, 1.0)
            out[i, :, :, :3] = frame[:out_h, :out_w].to(device="cpu", dtype=torch.float16)
            if alpha is not None:
                out[i, :, :, 3] = alpha[i, :out_h, :out_w, 0].to(torch.float16)
            progress.update(1)
        return (out,)


def _resize_reference(frame_hwc, height, width):
    """nodes_seedvr.py _resize_reference for one frame: bicubic, antialiased except on MPS."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TVF

    chw = frame_hwc.to(torch.float32).permute(2, 0, 1)[None]
    resized = TVF.resize(chw, size=(height, width), interpolation=InterpolationMode.BICUBIC,
                         antialias=not chw.device.type == "mps")
    return resized[0].permute(1, 2, 0)


NODE_CLASS_MAPPINGS = {
    "BC_SeedVR2Resize": SeedVR2Resize,
    "BC_SeedVR2VAEEncode": SeedVR2VAEEncode,
    "BC_SeedVR2VAEDecode": SeedVR2VAEDecode,
    "BC_SeedVR2PostProcess": SeedVR2PostProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_SeedVR2Resize": "SeedVR2 Resize",
    "BC_SeedVR2VAEEncode": "SeedVR2 VAE Encode",
    "BC_SeedVR2VAEDecode": "SeedVR2 VAE Decode",
    "BC_SeedVR2PostProcess": "SeedVR2 PostProcess",
}

"""VAE Decode — ComfyUI's VAE Decode (Tiled) on the SeedVR2 VAE decodes the
whole clip in one call and keeps every decoded frame on the GPU until the end
(vae.py slicing_decode: the slices are collected and torch.cat'ed), so VRAM
grows with the frame count — a 5090 tops out near 190 frames at 1080p. This
flow runs the same slice loop with the same causal memory cache, but moves
each decoded slice to RAM as soon as it exists, after the same
`/scaling_factor`, even crop and `(x + 1) / 2` clamp. VRAM is then the
decoder's fixed working set, whatever the length. The frames are stored as
float16, which is what the float16 VAE produced; the native node only upcasts
them. A latent that fits in one tile goes through the decoder at once, as the
native node does; a larger one is split into the spatial tiles of tiled_vae
and blended (models/seedvr2/tiling.py).
"""

import logging

import torch

from ...models.seedvr2.tiling import tile_plan, tile_weight
from ...models.seedvr2.vae import DECODER_BYTES_PER_PIXEL, LATENT_CHANNELS, make_room_for_vae, vae_model
from . import FRAMES_PER_CHUNK, TRIM_EVERY
from .progress import Progress


def decode(samples, vae, tile_size, overlap):
    import comfy.model_management as mm
    from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR, BYTEDANCE_VAE_SHIFTING_FACTOR
    from comfy.ldm.seedvr.vae import MemoryState

    model = vae_model(vae)
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
    make_room_for_vae(vae, DECODER_BYTES_PER_PIXEL * tile_px_h * tile_px_w)
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
                    mm.soft_empty_cache()  # see TRIM_EVERY in pipelines/seedvr2/__init__.py
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

            progress = Progress("SeedVR2 VAE Decode", 1, n_slices, t_pixel, device)
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
            ranges, ramp = tile_plan(h, w, tile_lat, ov_lat, device)
            edge_h = edge_w = ov_lat * 8  # tiled_vae decode: fades `ov_lat * 8` pixels wide
            result = torch.zeros((b, 3, t_pixel, h * 8, w * 8), dtype=torch.float16)
            count = torch.zeros((1, 1, 1, h * 8, w * 8), dtype=torch.float32)
            filled = 0
            progress = Progress("SeedVR2 VAE Decode", len(ranges), n_slices, t_pixel, device)
            for y0, y1, x0, x1 in ranges:
                progress.begin_tile()
                ys, xs = y0 * 8, x0 * 8
                weight = None
                offsets = []

                def sink(decoded, ys=ys, xs=xs, y0=y0, y1=y1, x0=x0, x1=x1, offsets=offsets):
                    nonlocal weight
                    th, tw = decoded.shape[3], decoded.shape[4]
                    if weight is None:
                        weight = tile_weight(y0, y1, x0, x1, h, w, th, tw, edge_h, edge_w, ramp, device)
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

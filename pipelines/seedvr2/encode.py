"""VAE Encode — ComfyUI's VAE Encode (Tiled) moves the whole clip to the GPU
first (comfy/sd.py _encode_tiled_owned: `process_input(...).to(dtype).to(device)`,
after a full float32 copy in RAM), then encodes it slice by slice
(vae.py slicing_encode). VRAM grows with the frame count on top of the
encoder's fixed working set. This flow runs the same slice loop with the
same causal memory cache and the same `x * 2 - 1`, but builds each input
slice on the GPU only when its turn comes. The posterior mode, the crop to
the latent size and the `* scaling_factor` follow the native path. A frame
that fits in one tile is encoded whole; a larger one is split into the
spatial tiles of tiled_vae and blended (models/seedvr2/tiling.py).
"""

import torch

from ...models.seedvr2.tiling import tile_plan, tile_weight
from ...models.seedvr2.vae import ENCODER_BYTES_PER_PIXEL, LATENT_CHANNELS, make_room_for_vae, vae_model
from . import TRIM_EVERY, require_image_batch
from .progress import Progress


def encode(pixels, vae, tile_size, overlap):
    import comfy.model_management as mm
    from comfy.ldm.seedvr.constants import BYTEDANCE_VAE_SCALING_FACTOR
    from comfy.ldm.seedvr.vae import MemoryState

    model = vae_model(vae)
    require_image_batch(pixels, "BC_SeedVR2VAEEncode: pixels must be an image batch (B, H, W, C)")
    pixels = pixels[..., :3]  # comfy/sd.py vae_encode_crop_pixels: output_channels = 3
    n, height, width = pixels.shape[0], pixels.shape[1], pixels.shape[2]
    target_t, target_h, target_w = (n + 3) // 4, (height + 7) // 8, (width + 7) // 8  # vae.py tiled_vae encode targets
    overlap = min(overlap, max(0, tile_size - 8))  # vae.py encode_tiled
    single_tile = height <= tile_size and width <= tile_size
    working_set = ENCODER_BYTES_PER_PIXEL * (height * width if single_tile else min(height, tile_size) * min(width, tile_size))
    make_room_for_vae(vae, working_set)
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
            progress = Progress("SeedVR2 VAE Encode", 1, len(slices), n, device)
            progress.begin_tile()
            encode_tile(0, height, 0, width, lambda i, h: parts.append(torch.chunk(h, 2, dim=1)[0].to("cpu")), progress)
            z = torch.cat(parts, dim=2)
        else:
            ranges, ramp = tile_plan(height, width, tile_size, overlap, device)
            edge_h = edge_w = overlap // 8  # tiled_vae encode: fades `overlap // 8` latent cells wide
            result = count = None
            progress = Progress("SeedVR2 VAE Encode", len(ranges), len(slices), n, device)
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
                        weight = tile_weight(y0, y1, x0, x1, height, width, th, tw, edge_h, edge_w, ramp, device)
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

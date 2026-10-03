"""VAE Encode — ComfyUI's VAE Encode (Tiled) moves the whole clip to the GPU
first (comfy/sd.py _encode_tiled_owned: `process_input(...).to(dtype).to(device)`,
after a full float32 copy in RAM), then encodes it slice by slice
(vae.py slicing_encode). VRAM grows with the frame count on top of the
encoder's fixed working set. This flow runs the same slice loop with the
same causal memory cache and the same `x * 2 - 1`, but builds each input
slice on the GPU only when its turn comes. The posterior mode, the crop to
the latent size and the `* scaling_factor` follow the native path. A frame
that fits in one tile is encoded whole; a larger one is split into the
spatial tiles of tiled_vae and blended (models/seedvr2/tiling.py). Either
way every latent slice goes straight into the float32 latent the node
returns; the blend's round trip through the VAE dtype (a few latent frames at
a time) and the scaling run on it in place.
"""

import torch

from ...models.seedvr2.tiling import tile_plan, tile_weight
from ...models.seedvr2.vae import ENCODER_BYTES_PER_PIXEL, ENCODER_FIXED_BYTES, LATENT_CHANNELS, make_room_for_vae, tile_for, vae_model
from . import FRAMES_PER_CHUNK, TRIM_EVERY, require_image_batch
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
    tile_size = tile_for(tile_size, height, width, ENCODER_FIXED_BYTES, ENCODER_BYTES_PER_PIXEL, vae.device, "SeedVR2 VAE Encode")
    overlap = min(overlap, max(0, tile_size - 8))  # vae.py encode_tiled
    single_tile = height <= tile_size and width <= tile_size
    tile_pixels = height * width if single_tile else min(height, tile_size) * min(width, tile_size)
    working_set = ENCODER_FIXED_BYTES + ENCODER_BYTES_PER_PIXEL * tile_pixels
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
        result = None
        if single_tile:
            # tiled_vae's one tile: the mode cropped to the target size. Each latent slice goes straight
            # into one float32 latent, the dtype the last line casts to (the upcast is exact).
            offsets = []

            def sink(i, h):
                nonlocal result
                t0 = sum(offsets)
                mean = torch.chunk(h, 2, dim=1)[0][:, :, : max(0, target_t - t0), :target_h, :target_w]
                if result is None:
                    result = torch.empty(tuple(mean.shape[:2]) + (target_t,) + tuple(mean.shape[3:]), dtype=torch.float32)
                result[:, :, t0:t0 + mean.shape[2]] = mean.to("cpu", torch.float32)
                offsets.append(mean.shape[2])

            progress = Progress("SeedVR2 VAE Encode", 1, len(slices), n, device)
            progress.begin_tile()
            encode_tile(0, height, 0, width, sink, progress)
            z = result[:, :, :sum(offsets)]
        else:
            ranges, ramp = tile_plan(height, width, tile_size, overlap, device)
            edge_h = edge_w = overlap // 8  # tiled_vae encode: fades `overlap // 8` latent cells wide
            count = None
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
            z = result.div_(count.clamp(min=1e-6))
            for t0 in range(0, z.shape[2], FRAMES_PER_CHUNK):  # tiled_vae: normalised, returned in the input dtype
                chunk = z[:, :, t0:t0 + FRAMES_PER_CHUNK]
                chunk.copy_(chunk.to(vae.vae_dtype))  # rounded through it in place, a chunk at a time
    z = z[:, :, :target_t, :target_h, :target_w].to(torch.float32).contiguous().mul_(BYTEDANCE_VAE_SCALING_FACTOR)  # crop, VAE output dtype, comfy_format_encoded
    return ({"samples": z},)

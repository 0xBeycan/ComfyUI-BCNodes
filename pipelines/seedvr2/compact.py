"""Compact — the SeedVR2 upscale chain in two nodes, with no clip-sized link between them.

Today's chain (Resize -> VAE Encode -> sampler -> VAE Decode -> PostProcess) leaves four
clip-sized tensors in ComfyUI's output cache until the prompt ends: Resize's padded `image` and
its `reference`, VAE Decode's frames and PostProcess's result. The compact pair leaves one, the
result:

  - Preprocess = Resize (`image` only) + VAE Encode: the padded frames exist inside the node while
    they are encoded and go when it returns. It returns the latent and the plan (SEEDVR2_PLAN, a
    few numbers): what PostProcess needs to rebuild the colour reference and to cut the padding;
  - PostProcess = VAE Decode + PostProcess: the decode stores only the frames, rows and columns
    the output keeps (decode `keep`), and each frame is colour-corrected in place in that buffer.
    Its reference is rebuilt from the original frames by the code Resize builds `reference` with,
    in the same four-frame chunks, so it is the same reference; each chunk is built on a worker
    thread while the frames before it are decoded or colour-corrected.

Both run the four nodes' flows, so the output equals today's chain bit for bit. With a
downscale_factor below 1 the lanczos downscale runs twice per frame (once in each node) instead of
once; the worker thread takes the second pass off the critical path.
"""

from typing import TypedDict

import torch

from ...models.seedvr2.vae import vae_model
from . import FRAMES_PER_CHUNK, require_image_batch
from . import decode as decode_flow, encode as encode_flow, resize as resize_flow
from .postprocess import color_transfer, corrected

PREPROCESS = "BC_SeedVR2PreprocessCompact"
POSTPROCESS = "BC_SeedVR2PostProcessCompact"


class SeedVR2Plan(TypedDict):
    """SEEDVR2_PLAN: Preprocess (Compact) -> PostProcess (Compact)."""

    frames: int  # the original frames' count, height and width: PostProcess checks its `image` against them
    height: int
    width: int
    downscale_factor: float  # Resize's step 1
    resolution: int  # step 2, from the original frames' shortest edge and upscale_factor
    max_resolution: int  # step 3's cap
    out_height: int  # the colour reference's size (resized, cropped to even): the most the output keeps
    out_width: int


def preprocess(image, vae, upscale_factor, downscale_factor, max_resolution, emulate_bf16, tile_size, overlap):
    """-> (latent, plan): SeedVR2 Resize's `image` encoded by SeedVR2 VAE Encode, and the plan."""
    vae_model(vae)  # a VAE that is not SeedVR2's stops here, not after the whole resize
    require_image_batch(image, f"{PREPROCESS}: connect an image batch (B, H, W, C)")
    pixels, reference = resize_flow.resize(image, upscale_factor, downscale_factor, max_resolution, emulate_bf16, want_reference=False)
    latent = encode_flow.encode(pixels, vae, tile_size, overlap)[0]
    del pixels  # the padded clip: the only clip-sized tensor here, gone before the node returns
    plan = SeedVR2Plan(frames=image.shape[0], height=image.shape[1], width=image.shape[2], downscale_factor=downscale_factor,
                       resolution=resize_flow.target_resolution(image, upscale_factor), max_resolution=max_resolution,
                       out_height=reference.shape[1], out_width=reference.shape[2])  # `reference` has 0 frames: its size only
    return (latent, plan)


def postprocess(samples, vae, image, plan, tile_size, overlap, color_correction_method):
    """-> (images,): SeedVR2 VAE Decode then SeedVR2 PostProcess, in one float16 buffer."""
    import comfy.model_management as mm
    from comfy.utils import ProgressBar

    transfer = color_transfer(color_correction_method, POSTPROCESS)
    require_image_batch(image, f"{POSTPROCESS}: connect the original frames (B, H, W, C), the batch SeedVR2 Preprocess (Compact) got")
    planned = (plan["frames"], plan["height"], plan["width"])
    if tuple(image.shape[:3]) != planned:
        raise ValueError(f"{POSTPROCESS}: image is {tuple(image.shape[:3])} (frames, height, width), the plan was made from "
                         f"{planned}: connect the frames that went into SeedVR2 Preprocess (Compact)")
    z = samples["samples"]
    if z.ndim == 5 and z.shape[0] != 1:
        raise ValueError(f"{POSTPROCESS}: samples holds {z.shape[0]} videos; connect the latent of one video "
                         "(the sampler's output for SeedVR2 Preprocess (Compact)'s latent)")
    keep = (plan["frames"], plan["out_height"], plan["out_width"])
    if transfer is None:
        return decode_flow.decode(samples, vae, tile_size, overlap, keep=keep)
    from concurrent.futures import ThreadPoolExecutor

    image = image[..., :3]

    def references(start):
        """Resize's `reference` for frames [start, start + FRAMES_PER_CHUNK): its chunk, its code, float16 (B, H, W, C)."""
        chunk = resize_flow._downscaled(image[start:start + FRAMES_PER_CHUNK], plan["downscale_factor"])
        return resize_flow._reference(chunk, plan["resolution"], plan["max_resolution"])

    # The references are CPU work (lanczos, bicubic): each chunk's is built on a worker thread while the decode (the
    # first chunk's) or the colour transfer of the chunk before runs, so the second lanczos pass costs no wall time.
    with ThreadPoolExecutor(max_workers=1) as worker:
        pending = worker.submit(references, 0)
        out = decode_flow.decode(samples, vae, tile_size, overlap, keep=keep)[0]
        device = mm.vae_device()
        progress = ProgressBar(out.shape[0])
        for start in range(0, out.shape[0], FRAMES_PER_CHUNK):
            chunk = pending.result()
            if start + FRAMES_PER_CHUNK < out.shape[0]:
                pending = worker.submit(references, start + FRAMES_PER_CHUNK)
            for i in range(start, min(start + FRAMES_PER_CHUNK, out.shape[0])):
                # contiguous, as a frame of Resize's `reference` output is
                out[i] = corrected(out[i], chunk[i - start].contiguous(), transfer, device).to(device="cpu", dtype=torch.float16)
                progress.update(1)
    return (out,)

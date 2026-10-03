"""SeedVR2 nodes for ComfyUI's native SeedVR2 graph.

    BC_SeedVR2Resize      (SeedVR2 Resize)       original image -> the frame the VAE encodes + colour reference
    BC_SeedVR2VAEEncode   (SeedVR2 VAE Encode)   VAE Encode (Tiled) with the frames streamed from RAM
    BC_SeedVR2VAEDecode   (SeedVR2 VAE Decode)   VAE Decode (Tiled) with the decoded frames streamed to RAM
    BC_SeedVR2PostProcess (SeedVR2 PostProcess)  Post-Process SeedVR2 Output, one frame at a time

    BC_SeedVR2PreprocessCompact  (SeedVR2 Preprocess (Compact))   Resize + VAE Encode, returns the latent and a plan
    BC_SeedVR2PostProcessCompact (SeedVR2 PostProcess (Compact))  VAE Decode + PostProcess into one buffer

    BC_SeedVR2FramingDownscale (SeedVR2 Framing Downscale)  Resize's downscale_factor from the face size (SAM 3)
    BC_SeedVR2ChunkSize        (SeedVR2 Chunk Size)         the frames per chunk the card holds, for Split SeedVR2 Latent

Resize is the whole input stage of the SeedVR2 upscale graph in one node. The compact
pair is the same chain with no clip between the input stage and the end (the output
cache keeps only the result); it gives the same frames.

The flows are in pipelines/seedvr2/ (resize, encode, decode, postprocess, compact, framing, chunk_size),
the SeedVR2 VAE adapter, its tiling, the frame-shape rules and the DiT's VRAM law in models/seedvr2/.
Each node imports its flow inside the method that runs it, and a widget list or default inside
INPUT_TYPES.
"""

from .common import LINK_INPUTS, drop_unwanted, heavy_wanted, wants

RESIZE_INPUTS = {
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
}


class SeedVR2Resize:
    """Original image -> the padded frame the VAE encodes, and its colour reference."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                **RESIZE_INPUTS,
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "reference")
    # each is a resize of its own, not computed when nothing links it (nodes/common.py)
    HEAVY_OUTPUTS = ("image", "reference")
    OUTPUT_TOOLTIPS = (
        "The frames the VAE encodes: downscaled, resized, clamped, padded to a multiple of 16 and to 4n+1 frames, float16. Wire to VAE Encode.",
        "The colour-correction reference: float32 resize stored as float16, cropped to even, not padded. Wire to SeedVR2 PostProcess.",
    )
    FUNCTION = "resize"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "resize", "shortest edge", "upscale", "downscale", "pad"]

    def resize(self, image, upscale_factor, downscale_factor, max_resolution, emulate_bf16, prompt_graph=None, unique_id=None):
        from ..pipelines.seedvr2 import resize as resize_flow

        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), resize_flow.resize(
            image, upscale_factor, downscale_factor, max_resolution, emulate_bf16, want_image=wants(wanted, "image"),
            want_reference=wants(wanted, "reference")), wanted)


TILE_INPUTS = {
    "tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32, "advanced": True,
                          "tooltip": "Spatial tile in pixels, as VAE Encode/Decode (Tiled). A tile that covers the frame means no tiling."}),
    "overlap": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 32, "advanced": True}),
}
COMPACT_TILE_INPUTS = {
    "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 32, "advanced": True,
                          "tooltip": "0 = auto: the largest tile whose working set fits this card (no tiling when the "
                                     "whole frame fits). Otherwise the spatial tile in pixels, as VAE Encode/Decode "
                                     "(Tiled); a tile that covers the frame means no tiling."}),
    "overlap": TILE_INPUTS["overlap"],
}
TILED_INPUTS = {
    **TILE_INPUTS,
    "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "advanced": True,
                              "tooltip": "Ignored, as it is by the SeedVR2 VAE in VAE Encode/Decode (Tiled): the causal VAE slices time itself."}),
    "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "advanced": True,
                                 "tooltip": "Ignored, as it is by the SeedVR2 VAE in VAE Encode/Decode (Tiled)."}),
}


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
        from ..pipelines.seedvr2 import encode as encode_flow

        return encode_flow.encode(pixels, vae, tile_size, overlap)


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
        from ..pipelines.seedvr2 import decode as decode_flow

        return decode_flow.decode(samples, vae, tile_size, overlap)


class SeedVR2PostProcess:
    """Post-Process SeedVR2 Output, one frame at a time into one float16 output."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.seedvr2.postprocess import METHODS

        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The generated frames."}),
                "original_resized_images": ("IMAGE", {"tooltip": "The reference frames (SeedVR2 Resize `reference`)."}),
                "color_correction_method": (METHODS, {"default": "lab"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Aligned, colour-corrected frames, float16.",)
    FUNCTION = "process"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "color correction", "postprocess", "lab", "video"]

    def process(self, images, original_resized_images, color_correction_method):
        from ..pipelines.seedvr2 import postprocess as postprocess_flow

        return postprocess_flow.process(images, original_resized_images, color_correction_method)


COMPACT_CATEGORY = "BCNodes/seedvr2/compact"


class SeedVR2PreprocessCompact:
    """SeedVR2 Resize + SeedVR2 VAE Encode: the latent and the plan, no clip-sized output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The original frames. Connect the same batch to SeedVR2 PostProcess (Compact) `image`."}),
                "vae": ("VAE",),
                **RESIZE_INPUTS,
                **COMPACT_TILE_INPUTS,
            },
        }

    RETURN_TYPES = ("LATENT", "SEEDVR2_PLAN")
    RETURN_NAMES = ("latent", "plan")
    OUTPUT_TOOLTIPS = (
        "The latent SeedVR2 VAE Encode gives for SeedVR2 Resize `image`, float32. Wire to the sampler.",
        "The resize settings and the output size (a few numbers). Wire to SeedVR2 PostProcess (Compact) `plan`.",
    )
    FUNCTION = "preprocess"
    CATEGORY = COMPACT_CATEGORY
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "compact", "resize", "vae encode", "upscale", "low ram"]

    def preprocess(self, image, vae, upscale_factor, downscale_factor, max_resolution, emulate_bf16, tile_size, overlap):
        from ..pipelines.seedvr2 import compact as compact_flow

        return compact_flow.preprocess(image, vae, upscale_factor, downscale_factor, max_resolution, emulate_bf16, tile_size, overlap)


class SeedVR2PostProcessCompact:
    """SeedVR2 VAE Decode + SeedVR2 PostProcess: decoded and colour-corrected in one float16 buffer."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.seedvr2.postprocess import METHODS

        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The sampler's latent."}),
                "vae": ("VAE",),
                "image": ("IMAGE", {"tooltip": "The original frames: the batch SeedVR2 Preprocess (Compact) got. "
                                               "The colour reference is rebuilt from them, a few frames at a time."}),
                "plan": ("SEEDVR2_PLAN", {"tooltip": "SeedVR2 Preprocess (Compact) `plan`."}),
                "color_correction_method": (METHODS, {"default": "lab"}),
                **COMPACT_TILE_INPUTS,
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Decoded, colour-corrected frames, float16, cut to the original frame count and the resized size.",)
    FUNCTION = "process"
    CATEGORY = COMPACT_CATEGORY
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "compact", "vae decode", "color correction", "postprocess", "low ram"]

    def process(self, samples, vae, image, plan, color_correction_method, tile_size, overlap):
        from ..pipelines.seedvr2 import compact as compact_flow

        return compact_flow.postprocess(samples, vae, image, plan, tile_size, overlap, color_correction_method)


UNCALIBRATED = "Uncalibrated default (an estimate, not yet measured against SeedVR2 output)."


class SeedVR2FramingDownscale:
    """SeedVR2 Resize's downscale_factor from the tallest face in the frame (SAM 3)."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..models.sam3.checkpoint import DEFAULT_SAM3, choices

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The original image(s), the batch SeedVR2 Resize gets. One factor for the whole "
                                               "batch, from its tallest face (Resize takes one factor)."}),
                "sam3_model": (choices(), {"default": DEFAULT_SAM3,
                                          "tooltip": "SAM 3 checkpoint under models/checkpoints; the default is downloaded when missing."}),
                "close_up_min_face": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 1.0, "step": 0.01,
                                                "tooltip": "Face height / image height from which the shot is a close-up. " + UNCALIBRATED}),
                "close_up_factor": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01,
                                              "tooltip": "downscale_factor for a close-up: a large face keeps its shape at a strong "
                                                         "downscale and gains the most skin texture."}),
                "medium_min_face": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01,
                                              "tooltip": "Face height / image height from which the shot is medium; below it, far. "
                                                         "Must be below close_up_min_face. " + UNCALIBRATED}),
                "medium_factor": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 1.0, "step": 0.01,
                                            "tooltip": "downscale_factor for a medium shot."}),
                "far_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
                                         "tooltip": "downscale_factor for a far shot. 1 = no downscale, so SeedVR2 does not "
                                                    "change a small face."}),
                "no_face_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
                                             "tooltip": "downscale_factor when no face is found (logged)."}),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                                  "tooltip": "SAM 3 detection threshold for the face."}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("downscale_factor", "face_fraction")
    OUTPUT_TOOLTIPS = (
        "Wire to SeedVR2 Resize (or SeedVR2 Preprocess (Compact)) downscale_factor.",
        "Height of the tallest face box / image height, over the whole batch; 0 when no face is found.",
    )
    FUNCTION = "choose"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "downscale", "framing", "face size", "close-up", "sam3"]
    DESCRIPTION = ("Measures the tallest face (SAM 3) as a fraction of the image height and picks SeedVR2 Resize's "
                   "downscale_factor: close-up, medium or far, each with its own factor. A lower factor gives more skin "
                   "texture but changes a small face, so a far shot is not downscaled. The thresholds are uncalibrated "
                   "estimates.")

    def choose(self, image, sam3_model, close_up_min_face, close_up_factor, medium_min_face, medium_factor, far_factor,
               no_face_factor, detection_threshold):
        from ..pipelines.seedvr2 import framing as framing_flow

        return framing_flow.downscale_factor(image, sam3_model, detection_threshold, close_up_min_face, close_up_factor,
                                             medium_min_face, medium_factor, far_factor, no_face_factor)


class SeedVR2ChunkSize:
    """The largest 4n+1 frames per chunk whose DiT working set fits the card, for Split SeedVR2 Latent."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..models.seedvr2.dit import SAFETY_MARGIN

        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The encoded clip (SeedVR2 Preprocess (Compact) or SeedVR2 VAE Encode `latent`): "
                                                 "its length and frame size."}),
                "safety_margin": ("FLOAT", {"default": SAFETY_MARGIN, "min": 0.0, "max": 4.0, "step": 0.01,
                                            "tooltip": "Each frame's share of the DiT's working set is budgeted at (1 + this) x the "
                                                       "measured line. 0.64: a long clip needed up to 1.44x it on a 32 GB and a "
                                                       "96 GB card; this keeps 14% over that. Raise it if a chunk runs out of "
                                                       "memory, lower it to try longer chunks."}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames_per_chunk",)
    OUTPUT_TOOLTIPS = ("Pixel frames per chunk (4n+1), at most the clip's. Wire to Split SeedVR2 Latent's frames_per_chunk "
                       "with its chunking_mode on manual.",)
    FUNCTION = "size"
    CATEGORY = "BCNodes/seedvr2"
    SEARCH_ALIASES = ["BCNodes", "seedvr2", "seedvr", "chunk", "frames per chunk", "temporal", "vram", "split"]
    DESCRIPTION = ("The longest chunk the SeedVR2 sampler can take on this card: the DiT's working set (a fixed part plus a "
                   "part per frame and megapixel, measured on the 7B) against the card's total memory, not the memory free "
                   "right now. Set Split SeedVR2 Latent's chunking_mode to manual and wire this to its frames_per_chunk; its "
                   "own auto ran out of memory on both cards measured.")

    def size(self, latent, safety_margin):
        from ..pipelines.seedvr2 import chunk_size as chunk_size_flow

        return chunk_size_flow.frames_per_chunk(latent, safety_margin)


NODE_CLASS_MAPPINGS = {
    "BC_SeedVR2FramingDownscale": SeedVR2FramingDownscale,
    "BC_SeedVR2Resize": SeedVR2Resize,
    "BC_SeedVR2VAEEncode": SeedVR2VAEEncode,
    "BC_SeedVR2ChunkSize": SeedVR2ChunkSize,
    "BC_SeedVR2VAEDecode": SeedVR2VAEDecode,
    "BC_SeedVR2PostProcess": SeedVR2PostProcess,
    "BC_SeedVR2PreprocessCompact": SeedVR2PreprocessCompact,
    "BC_SeedVR2PostProcessCompact": SeedVR2PostProcessCompact,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_SeedVR2FramingDownscale": "SeedVR2 Framing Downscale",
    "BC_SeedVR2Resize": "SeedVR2 Resize",
    "BC_SeedVR2VAEEncode": "SeedVR2 VAE Encode",
    "BC_SeedVR2ChunkSize": "SeedVR2 Chunk Size",
    "BC_SeedVR2VAEDecode": "SeedVR2 VAE Decode",
    "BC_SeedVR2PostProcess": "SeedVR2 PostProcess",
    "BC_SeedVR2PreprocessCompact": "SeedVR2 Preprocess (Compact)",
    "BC_SeedVR2PostProcessCompact": "SeedVR2 PostProcess (Compact)",
}

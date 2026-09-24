"""SeedVR2 nodes for ComfyUI's native SeedVR2 graph.

    BC_SeedVR2Resize      (SeedVR2 Resize)       original image -> the frame the VAE encodes + colour reference
    BC_SeedVR2VAEEncode   (SeedVR2 VAE Encode)   VAE Encode (Tiled) with the frames streamed from RAM
    BC_SeedVR2VAEDecode   (SeedVR2 VAE Decode)   VAE Decode (Tiled) with the decoded frames streamed to RAM
    BC_SeedVR2PostProcess (SeedVR2 PostProcess)  Post-Process SeedVR2 Output, one frame at a time

Resize is the whole input stage of the SeedVR2 upscale graph in one node.

The flows are in pipelines/seedvr2/ (resize, encode, decode, postprocess), the
SeedVR2 VAE adapter, its tiling and the frame-shape rules in models/seedvr2/.
"""

from ..pipelines.seedvr2 import (
    decode as decode_flow, encode as encode_flow, postprocess as postprocess_flow, resize as resize_flow,
)


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
        return resize_flow.resize(image, upscale_factor, downscale_factor, max_resolution, emulate_bf16)


TILED_INPUTS = {
    "tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 32, "advanced": True,
                          "tooltip": "Spatial tile in pixels, as VAE Encode/Decode (Tiled). A tile that covers the frame means no tiling."}),
    "overlap": ("INT", {"default": 256, "min": 0, "max": 4096, "step": 32, "advanced": True}),
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
        return decode_flow.decode(samples, vae, tile_size, overlap)


class SeedVR2PostProcess:
    """Post-Process SeedVR2 Output, one frame at a time into one float16 output."""

    METHODS = postprocess_flow.METHODS

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
        return postprocess_flow.process(images, original_resized_images, color_correction_method)


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

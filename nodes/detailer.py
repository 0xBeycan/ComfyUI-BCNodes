"""Detailer — a region of an image sampled again at a higher resolution and pasted back.

    BC_Detailer      (Detailer)
    BC_FaceDetailer  (Face Detailer)
"""

import torch

from .common import LINK_INPUTS, drop_unlinked_heavy, drop_unwanted, heavy_wanted, wants


def sampling_inputs():
    """The sampler and detailer widgets both nodes take, in order: core KSampler's (same ranges), then the
    two passes' denoise, the sampling resolution and the crop's context."""
    import comfy.samplers

    return {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                         "tooltip": "The noise seed of the structure pass; the refine pass uses seed + 1. Every region gets the same."}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Steps of each pass."}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                          "tooltip": "Classifier-free guidance scale, as on KSampler."}),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler, as on KSampler."}),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler, as on KSampler."}),
        "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "Denoise of the structure pass, the first one: how far the region may be redrawn. 0 skips the pass."}),
        "refine_denoise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Denoise of the refine pass, run on the structure pass's result: fine detail only. 0 skips the pass."}),
        "resolution": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64,
                               "tooltip": "The long side, in px, each crop is sampled at: the model's native size (SD 1.5 512, SDXL 1024, "
                                          "Qwen-Image 1328). A crop already this large or larger is left as it is: sampling it adds no detail."}),
        "context": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1,
                              "tooltip": "The crop around a region, as a multiple of the region's box: the surroundings the model sees. "
                                         "Larger shows more context but samples the region itself at fewer pixels."}),
    }


class Detailer:
    """Details each mask region of an image: crop with context, sample at the model's size, paste back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "The regions to detail, one per mask of the batch (e.g. SAM 3 Detect with individual_masks "
                                             "on), each applied to every frame. Only the pixels inside a mask change."}),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING", {"tooltip": "A ControlNet applied to it (Apply ControlNet) has its hint cut to each crop."}),
                "negative": ("CONDITIONING",),
                **sampling_inputs(),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    # nodes/common.py: image (the sampling) is not computed when nothing links it; mask never reads it
    HEAVY_OUTPUTS = ("image", "mask")
    FUNCTION = "detail"
    CATEGORY = "BCNodes/detailer"
    SEARCH_ALIASES = ["BCNodes", "detailer", "detail", "inpaint", "crop and stitch", "differential diffusion"]
    DESCRIPTION = ("Details each mask region of the image: the region is cropped with its context, enlarged to the sampling "
                   "resolution, sampled in a structure pass and a refine pass (Differential Diffusion, inpaint conditioning, "
                   "ControlNet hints cut to the crop), shrunk back and pasted through the mask's own shape, colour matched at "
                   "its edge and blended with a Laplacian pyramid. Pixels outside the masks never change. Outputs the image and "
                   "the pixels that changed.")

    def detail(self, image, mask, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise,
               refine_denoise, resolution, context, prompt_graph=None, unique_id=None):
        from ..pipelines.detailer import core

        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        if mask is None:
            mask = torch.zeros((0, *image.shape[1:3]))
        settings = core.Settings(vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, refine_denoise,
                                 resolution, context)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), core.run(image, mask, model, settings, want_image=wants(wanted, "image")), wanted)


class FaceDetailer:
    """Details every face SAM 3 finds: the character's with the main model, the others with model_no_lora."""

    @classmethod
    def INPUT_TYPES(cls):
        from ..models.sam3.checkpoint import DEFAULT_SAM3, choices

        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL", {"tooltip": "The pipeline's main model, with the character LoRA: the character's face is detailed with it."}),
                "vae": ("VAE",),
                "positive": ("CONDITIONING", {"tooltip": "A ControlNet applied to it (Apply ControlNet) has its hint cut to each crop."}),
                "negative": ("CONDITIONING",),
                **sampling_inputs(),
                "sam3_model": (choices(), {"default": DEFAULT_SAM3,
                                          "tooltip": "SAM 3 checkpoint under models/checkpoints; the default is downloaded when missing."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "SAM 3 detection threshold, for the faces, the mouth and the eyes."}),
                "min_face_height": ("INT", {"default": 64, "min": 0, "max": 16384,
                                            "tooltip": "A face shorter than this (px) is left as it is: no face, mouth or eye pass."}),
                "max_face_height": ("INT", {"default": 16384, "min": 0, "max": 16384,
                                            "tooltip": "A face taller than this (px) skips the face pass; its mouth and eyes are still "
                                                       "detailed. A face whose crop is already at the sampling resolution skips it anyway."}),
                "face": ("BOOLEAN", {"default": True, "tooltip": "The face pass."}),
                "mouth": ("BOOLEAN", {"default": False, "tooltip": "A mouth pass (teeth and lips) after the face pass, on a smaller crop."}),
                "eyes": ("BOOLEAN", {"default": True, "tooltip": "An eye pass after the face pass, each eye on a crop of its own."}),
            },
            "optional": {
                "model_no_lora": ("MODEL", {"tooltip": "Connected: every face but the character's is detailed with it. Not connected: "
                                                       "the other faces are left as they are."}),
                "mask": ("MASK", {"tooltip": "Picks the character: the face it overlaps most (none when it overlaps none). Not "
                                             "connected: the largest face."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    # nodes/common.py: both computed (the mouth and eyes are found on the detailed face), the unlinked one dropped at return
    HEAVY_OUTPUTS = ("image", "mask")
    FUNCTION = "detail"
    CATEGORY = "BCNodes/detailer"
    SEARCH_ALIASES = ["BCNodes", "face detailer", "detailer", "face", "eyes", "mouth", "sam3"]
    DESCRIPTION = ("Finds every face with SAM 3 and details it with the Detailer's core: the character's face (the one the mask "
                   "overlaps, or the largest) with the main model, the other faces with model_no_lora when it is connected. A "
                   "mouth pass and an eye pass (each eye on its own crop) follow the face pass. Faces outside the height range "
                   "are skipped; a face too large to gain from sampling keeps its own pixels while its eyes are still detailed. "
                   "Outputs the image and the pixels that changed.")

    def detail(self, image, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, refine_denoise,
               resolution, context, sam3_model, threshold, min_face_height, max_face_height, face, mouth, eyes,
               model_no_lora=None, mask=None, prompt_graph=None, unique_id=None):
        from ..pipelines.detailer import core
        from ..pipelines.detailer import face as face_flow

        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        settings = core.Settings(vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, refine_denoise,
                                 resolution, context)
        return drop_unlinked_heavy(type(self), face_flow.run(image, model, model_no_lora, settings, sam3_model, threshold, mask,
                                                             min_face_height, max_face_height, face, mouth, eyes),
                                   prompt_graph, unique_id)


NODE_CLASS_MAPPINGS = {
    "BC_Detailer": Detailer,
    "BC_FaceDetailer": FaceDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_Detailer": "Detailer",
    "BC_FaceDetailer": "Face Detailer",
}

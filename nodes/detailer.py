"""Detailer — a region of an image sampled again at a larger size and pasted back.

    BC_Detailer      (Detailer)
    BC_FaceDetailer  (Face Detailer)
"""

import torch

from .common import LINK_INPUTS, drop_unlinked_heavy, drop_unwanted, heavy_wanted, wants


def sampling_inputs():
    """The sampler widgets both nodes take, in order: core KSampler's (same ranges), then denoise. Each node's
    size, crop size and largest size follow them, named and described for what it details."""
    import comfy.samplers

    return {
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True,
                         "tooltip": "The noise seed. Every region gets the same."}),
        "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Steps of each region's sampling."}),
        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                          "tooltip": "Classifier-free guidance scale, as on KSampler."}),
        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler, as on KSampler."}),
        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler, as on KSampler."}),
        "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "How far each region may be redrawn (Face Detailer: the face pass; the eyes take "
                                         "eye_denoise). 0 leaves every region as it is: nothing is encoded, sampled or "
                                         "pasted, and the mask output marks none (Face Detailer: no face pass)."}),
    }


class Detailer:
    """Details each mask region of an image: sample a crop at region_crop_size px with the region at region_size px inside
    it, paste back; a region over max_region_size px, or already region_size px, is left as it is."""

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
                "region_size": ("INT", {"default": 640, "min": 64, "max": 8192, "step": 64,
                                        "tooltip": "The size, in px, each region is redrawn at inside its crop: its box's longer side "
                                                   "is enlarged to region_size px. A region whose longer side is already region_size px "
                                                   "or more is left as it is (nothing to enlarge), as is one over max_region_size px."}),
                "region_crop_size": ("INT", {"default": 1280, "min": 256, "max": 8192, "step": 64,
                                             "tooltip": "The model samples at this size: the crop's longer side, in px. The crop holds "
                                                        "the region at region_size px and its surroundings around it (region_size 512, "
                                                        "region_crop_size 1280: the model works at 1280 with the region 512 px inside); "
                                                        "its shorter side follows the region's aspect, rounded up to a multiple of 64. "
                                                        "Under region_size it is region_size (no surroundings). A crop past the image's "
                                                        "edge is moved inside, at the same scale, and cut where the image is too small. "
                                                        "Only for context: pixels outside the mask never change."}),
                "max_region_size": ("INT", {"default": 384, "min": 64, "max": 16384, "step": 1,
                                            "tooltip": "The largest region, in px (its box's longer side), that is detailed: a larger "
                                                       "one is left as it is. Apart from region_size, so mid-size regions can be "
                                                       "skipped without changing how much a small one is enlarged."}),
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
    DESCRIPTION = ("Details each mask region of the image: a crop around the region, enlarged so the region's longer side is "
                   "region_size px and the crop's longer side region_crop_size px, is sampled at that size (Differential "
                   "Diffusion, inpaint conditioning, ControlNet hints cut to the crop), shrunk back and pasted through the mask's "
                   "own shape, colour matched at its edge and blended with a Laplacian pyramid. A region over max_region_size px, or already "
                   "region_size px or larger, is left as it is. Pixels outside the masks never change. Outputs the image and the "
                   "pixels that changed.")

    def detail(self, image, mask, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise,
               region_size, region_crop_size, max_region_size, prompt_graph=None, unique_id=None):
        from ..pipelines.detailer import core

        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        if mask is None:
            mask = torch.zeros((0, *image.shape[1:3]))
        settings = core.Settings(vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, region_size,
                                 region_crop_size)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        return drop_unwanted(type(self), core.run(image, mask, model, settings, max_region_size, want_image=wants(wanted, "image")),
                             wanted)


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
                "face_size": ("INT", {"default": 640, "min": 64, "max": 8192, "step": 64,
                                      "tooltip": "The size, in px, each face is redrawn at inside its crop: its box's longer side (its "
                                                 "height when upright) is enlarged to face_size px. A face whose longer side is already "
                                                 "face_size px or more skips the face pass (nothing to enlarge), as does one taller "
                                                 "than max_face_height; its eyes still run."}),
                "face_crop_size": ("INT", {"default": 1280, "min": 256, "max": 8192, "step": 64,
                                           "tooltip": "The model samples at this size: the face crop's longer side, in px. The crop holds "
                                                      "the face at face_size px and its surroundings around it (face_size 512, "
                                                      "face_crop_size 1280: the model works at 1280 with the face 512 px inside); its "
                                                      "shorter side follows the face's aspect, rounded up to a multiple of 64. Under "
                                                      "face_size it is face_size (no surroundings). A crop past the image's edge is moved "
                                                      "inside, at the same scale, and cut where the image is too small. Only for context: "
                                                      "pixels outside the face's mask never change."}),
                "max_face_height": ("INT", {"default": 384, "min": 64, "max": 16384, "step": 1,
                                            "tooltip": "The tallest face, in px (its box's height), the face pass redraws: a taller face "
                                                       "skips it while its eyes still run. Apart from face_size, so mid-size faces can be "
                                                       "skipped without changing how much a small face is enlarged. A face under 64 px "
                                                       "gets no pass at all."}),
                "eye_size": ("INT", {"default": 320, "min": 64, "max": 8192, "step": 64,
                                     "tooltip": "The size, in px, the eyes are redrawn at inside their crop: both eyes are one region "
                                                "(one crop, so the model sees both and their gazes match; one eye alone when only one is "
                                                "found), whose box's longer side is enlarged to eye_size px. Eyes whose box is already "
                                                "eye_size px or more skip the pass."}),
                "eye_crop_size": ("INT", {"default": 640, "min": 256, "max": 8192, "step": 64,
                                          "tooltip": "The model samples at this size: the eye crop's longer side, in px. The crop holds "
                                                     "the eyes' box at eye_size px and its surroundings around it (eye_size 320, "
                                                     "eye_crop_size 640: the model works at 640 with the eyes 320 px inside); its shorter "
                                                     "side follows the box's aspect, rounded up to a multiple of 64. Under eye_size it is "
                                                     "eye_size (no surroundings). A crop past the image's edge is moved inside, at the "
                                                     "same scale, and cut where the image is too small. Only for context: pixels outside "
                                                     "the eyes' masks never change."}),
                "eye_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "How far the eyes may be redrawn: the eye pass's denoise, in place of denoise. 0 "
                                                     "turns the eye pass off: the eyes are not searched for and keep their "
                                                     "pixels."}),
                "sam3_model": (choices(), {"default": DEFAULT_SAM3,
                                          "tooltip": "SAM 3 checkpoint under models/checkpoints; the default is downloaded when missing."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "SAM 3 detection threshold, for the faces and the eyes."}),
                "face": ("BOOLEAN", {"default": True, "tooltip": "The face pass."}),
                "eyes": ("BOOLEAN", {"default": True, "tooltip": "An eye pass after the face pass: both eyes on one crop."}),
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
    # nodes/common.py: both computed (the eyes are found on the detailed face), the unlinked one dropped at return
    HEAVY_OUTPUTS = ("image", "mask")
    FUNCTION = "detail"
    CATEGORY = "BCNodes/detailer"
    SEARCH_ALIASES = ["BCNodes", "face detailer", "detailer", "face", "eyes", "sam3"]
    DESCRIPTION = ("Finds every face with SAM 3 and details it with the Detailer's core: the character's face (the one the mask "
                   "overlaps, or the largest) with the main model, the other faces with model_no_lora when it is connected. Each "
                   "face is redrawn with its longer side at face_size px inside a crop the model samples at face_crop_size px; "
                   "then both eyes as one region on one crop (so their gazes match) by the same rule, with eye_size, "
                   "eye_crop_size and eye_denoise. A face under 64 px is skipped; a face taller than max_face_height, or already face_size px or "
                   "larger, keeps its own pixels while its eyes are still detailed, and eyes already eye_size px or larger keep "
                   "theirs. Outputs the image and the pixels that changed.")

    def detail(self, image, model, vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, face_size,
               face_crop_size, max_face_height, eye_size, eye_crop_size, eye_denoise, sam3_model, threshold, face, eyes,
               model_no_lora=None, mask=None, prompt_graph=None, unique_id=None):
        from dataclasses import replace

        from ..pipelines.detailer import core
        from ..pipelines.detailer import face as face_flow

        if image is None or image.shape[0] == 0:
            return (torch.zeros((0, 64, 64, 3)), torch.zeros((0, 64, 64)))
        face_settings = core.Settings(vae, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, face_size,
                                      face_crop_size)
        eye_settings = replace(face_settings, denoise=eye_denoise, size=eye_size, crop_size=eye_crop_size)
        return drop_unlinked_heavy(type(self), face_flow.run(image, model, model_no_lora, face_settings, eye_settings, sam3_model,
                                                             threshold, mask, max_face_height, face, eyes),
                                   prompt_graph, unique_id)


NODE_CLASS_MAPPINGS = {
    "BC_Detailer": Detailer,
    "BC_FaceDetailer": FaceDetailer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_Detailer": "Detailer",
    "BC_FaceDetailer": "Face Detailer",
}

"""ComfyUI-BCNodes — utility nodes for ComfyUI.

Every node module only touches torch / numpy / stdlib at import time; scipy,
PIL, cv2, safetensors, torchvision, the BiRefNet architecture and the pip
dependencies (postfx, caption-audit) are imported inside the functions that
use them. tests/test_import_time.py holds the package to that.

Nodes:

    BC_AutoBypass                frontend-only virtual node, lives in web/js
    BC_LogicBoolean              0-1 float -> BOOLEAN / NUMBER / INT / FLOAT
    BC_IsMaskEmpty               MASK -> BOOLEAN
    BC_MaskFillHoles             fill enclosed holes in a mask
    BC_MaskGrow                  grow / shrink + blur a mask
    BC_ImageScaleByAspectRatio   scale image / mask to an aspect ratio and side length
    BC_JoinImageLists            concatenate image lists, unbounded inputs
    BC_BiRefNetRemoveBackground  BiRefNet matting, plain torch
    BC_AutoModelDownloader       fetch a workflow's models into models/
    BC_MathExpression            arithmetic over a, b, c without eval()
    BC_PromptList                one prompt per line, as a list
    BC_AnySwitch                 first connected non-None input, any type
    BC_Seed                      seed widget; -1 = new random seed every run
    BC_ShowText                  show incoming text on the node, pass it on
    BC_ImageComparer             two images, divider comparison on the node
    BC_VideoComparer             two videos (VHS_FILENAMES), synced divider comparison
    BC_PowerLoraLoader           MODEL + LoRA rows -> MODEL (no CLIP)
    BC_AnythingEverywhere        feeds unconnected inputs of a type at prompt time
    BC_FastGroupsBypasser        one bypass toggle per group
    BC_SeedVR2Resize             SeedVR2 resize: shortest edge, antialiased bicubic, bf16 path
    BC_SeedVR2VAEEncode          SeedVR2 VAE encode, frames streamed from RAM
    BC_SeedVR2VAEDecode          SeedVR2 VAE decode, frames streamed to RAM
    BC_SeedVR2PostProcess        SeedVR2 post-process, one frame at a time
    BC_PostFxApply               apply a postfx film-emulation look (+ condition, strength, mask)
    BC_PostFxTheme               built-in postfx theme -> POSTFX_LOOK
    BC_PostFxCustomLook          build / override a look from common controls
    BC_PostFxLut                 3D .cube LUT as a look, standalone or layered
    BC_PostFxSignatureSheet      labeled contact sheet of every theme in a category
    BC_CaptionAudit              caption-audit over a dataset folder, card drawn on the node
    BC_SocialMediaExport         per-platform derivatives of a master image, spec-driven
    BC_ImageQualityGate          blur / sharpness / noise / clipping / entropy -> PASS / SO-SO / FAIL
    BC_SaveImage                 save images, folder / file names from prompt values, gallery-only preview
    BC_SkinTexture               micro-texture on skin inside a SAM 3 mask, before upscale and grain

Frontend-only pieces in web/js: BC_AutoBypass, Join Image Lists' growing
slots, the downloader's node UI and first-open prompt, and the Align buttons
in the selection toolbox.
"""

from .nodes import (
    any_switch, birefnet, caption_audit, downloader, everywhere, image_comparer, image_quality_gate, image_scale, lists,
    logic, mask, math_expression, postfx, power_lora_loader, prompt_list, save_image, seed, seedvr2, show_text, skin_texture, social_media_export,
    video_comparer,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _module in (logic, mask, image_scale, lists, birefnet, downloader, math_expression, prompt_list, any_switch, seed, show_text,
                image_comparer, video_comparer, power_lora_loader, everywhere, seedvr2,
                postfx, caption_audit, social_media_export, image_quality_gate, save_image, skin_texture):
    NODE_CLASS_MAPPINGS.update(_module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_module.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

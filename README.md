# ComfyUI-BCNodes

Utility nodes for ComfyUI, in one small pack.

| Registration key | Display name | What it does |
| --- | --- | --- |
| `BC_AutoBypass` | [Auto Bypass](docs/workflow.md#bc_autobypass--auto-bypass) | Bypasses its targets automatically when a watched source is empty |
| `BC_LogicBoolean` | [Logic Boolean](docs/logic.md#bc_logicboolean--logic-boolean) | 0–1 float → `BOOLEAN` / `NUMBER` / `INT` / `FLOAT` |
| `BC_IsMaskEmpty` | [Is Mask Empty](docs/mask.md#bc_ismaskempty--is-mask-empty) | `MASK` → `BOOLEAN` |
| `BC_MaskFillHoles` | [Mask Fill Holes](docs/mask.md#bc_maskfillholes--mask-fill-holes) | Fills enclosed holes in a mask |
| `BC_MaskGrow` | [MaskGrow](docs/mask.md#bc_maskgrow--maskgrow) | Grows / shrinks a mask, then blurs it |
| `BC_DrawMaskOnImage` | [Draw Mask On Image](docs/mask.md#bc_drawmaskonimage--draw-mask-on-image) | Paints a colour (with opacity) through a mask onto an image batch |
| `BC_BlockifyMask` | [Blockify Mask](docs/mask.md#bc_blockifymask--blockify-mask) | A mask as the blocks of its bounding box that hold any of it |
| `BC_RepeatMaskBatch` | [Repeat Mask Batch](docs/mask.md#bc_repeatmaskbatch--repeat-mask-batch) | A mask batch repeated `amount` times |
| `BC_ImageScaleByAspectRatio` | [Image Scale By Aspect Ratio](docs/image.md#bc_imagescalebyaspectratio--image-scale-by-aspect-ratio) | Scales an image / mask to an aspect ratio and side length; `MASK` output is never `None` |
| `BC_ImageResize` | [Image Resize](docs/image.md#bc_imageresize--image-resize) | Resizes an image batch (and mask) to a width and height: stretch, keep the proportion, pad or crop |
| `BC_JoinImageLists` | [Join Image Lists](docs/image.md#bc_joinimagelists--join-image-lists) | Concatenates image lists, unlimited inputs |
| `BC_MathExpression` | [Math Expression](docs/logic.md#bc_mathexpression--math-expression) | Arithmetic over `a`, `b`, `c` without `eval()` |
| `BC_PromptList` | [Prompt List](docs/text.md#bc_promptlist--prompt-list) | One prompt per line, as a list |
| `BC_AnySwitch` | [Any Switch](docs/logic.md#bc_anyswitch--any-switch) | First connected non-None input, any type, unlimited inputs |
| `BC_SelectSwitch` | [Select Switch](docs/logic.md#bc_selectswitch--select-switch) | Input of the selected named option, any type; only the selected branch runs |
| `BC_Seed` | [Seed](docs/logic.md#bc_seed--seed) | Seed widget; `-1` draws a new random seed on every run |
| `BC_ShowText` | [Show Text](docs/text.md#bc_showtext--show-text) | Shows incoming text on the node, passes it on |
| `BC_ImageComparer` | [Image Comparer](docs/workflow.md#bc_imagecomparer--image-comparer) | Two images on the node, compared with a sliding divider |
| `BC_PowerLoraLoader` | [Power Lora Loader](docs/loaders.md#bc_powerloraloader--power-lora-loader) | `MODEL` + any number of LoRA rows → `MODEL`, no CLIP |
| `BC_AnythingEverywhere` | [Anything Everywhere](docs/workflow.md#bc_anythingeverywhere--anything-everywhere) | Feeds unconnected inputs of a type at prompt time |
| `BC_FastGroupsBypasser` | [Fast Groups Bypasser](docs/workflow.md#bc_fastgroupsbypasser--fast-groups-bypasser) | One bypass toggle per group |
| `BC_BiRefNetRemoveBackground` | [BiRefNet Remove Background](docs/mask.md#bc_birefnetremovebackground--birefnet-remove-background) | Background removal with BiRefNet, plain torch |
| `BC_DepthAnythingV2` | [Depth Anything](docs/image.md#bc_depthanythingv2--depth-anything) | Depth map with Depth Anything V2 Small or Depth Anything 3 (Small, Base, Mono-Large, Metric-Large; all Apache-2.0), near = white, at a ControlNet preprocessor's size or exactly width x height |
| `BC_SeedVR2Resize` | [SeedVR2 Resize](docs/seedvr2.md#bc_seedvr2resize--seedvr2-resize) | Original image → the padded frame SeedVR2 encodes (lanczos downscale, shortest-edge antialiased bicubic, pad 16, 4n+1 frames) plus the colour reference |
| `BC_SeedVR2VAEEncode` | [SeedVR2 VAE Encode](docs/seedvr2.md#bc_seedvr2vaeencode--seedvr2-vae-encode) | SeedVR2 VAE encode with the frames streamed from RAM slice by slice, so VRAM does not grow with the frame count |
| `BC_SeedVR2VAEDecode` | [SeedVR2 VAE Decode](docs/seedvr2.md#bc_seedvr2vaedecode--seedvr2-vae-decode) | SeedVR2 VAE decode with every decoded slice streamed to RAM, so VRAM does not grow with the frame count |
| `BC_SeedVR2PostProcess` | [SeedVR2 PostProcess](docs/seedvr2.md#bc_seedvr2postprocess--seedvr2-postprocess) | Post-Process SeedVR2 Output one frame at a time into one float16 output, no full-size temporaries |
| `BC_AutoModelDownloader` | [Auto Model Downloader](docs/loaders.md#bc_automodeldownloader--auto-model-downloader) | Lists a workflow's models and fetches the missing ones into `models/` |
| `BC_PostFxApply` | [PostFx Apply](docs/postfx.md#bc_postfxapply--postfx-apply-and-the-look-nodes) | Applies a `postfx` film-emulation look (theme + condition + strength) to an image batch, optional look override and mask |
| `BC_PostFxTheme` | [PostFx Theme](docs/postfx.md#bc_postfxapply--postfx-apply-and-the-look-nodes) | Built-in `postfx` theme → `POSTFX_LOOK`, to start a chain from a named look |
| `BC_PostFxCustomLook` | [PostFx Custom Look](docs/postfx.md#bc_postfxapply--postfx-apply-and-the-look-nodes) | Builds a look from common controls, or overrides them on top of an incoming look |
| `BC_PostFxLut` | [PostFx LUT](docs/postfx.md#bc_postfxapply--postfx-apply-and-the-look-nodes) | 3D `.cube` LUT as a look, standalone or layered on a theme |
| `BC_PostFxSignatureSheet` | [PostFx Signature Sheet](docs/postfx.md#bc_postfxapply--postfx-apply-and-the-look-nodes) | Labeled contact sheet of every theme in a category applied to one image |
| `BC_CaptionAudit` | [Caption Audit](docs/analysis.md#bc_captionaudit--caption-audit) | Runs `caption-audit` over a LoRA caption folder and draws the report card on the node |
| `BC_SocialMediaExport` | [Social Media Export](docs/image.md#bc_socialmediaexport--social-media-export) | One platform-ready derivative per ticked platform, minimum crop, spec-driven |
| `BC_ImageQualityGate` | [Image Quality Gate](docs/analysis.md#bc_imagequalitygate--image-quality-gate) | Blur / sharpness / noise / clipping / entropy → `PASS` / `SO-SO` / `FAIL` badge, verdict and scores |
| `BC_SaveImage` | [Save Image](docs/image.md#bc_saveimage--save-image) | Saves images with folder / file names built from prompt widget values, any Pillow format, prompt + workflow embedded; preview only in the gallery, never under the node |
| `BC_SaveImageWithCaption` | [Save Image With Caption](docs/image.md#bc_saveimagewithcaption--save-image-with-caption) | Saves images as PNG and, with a caption connected, the caption next to each image under the same name, for training datasets |
| `BC_SkinTexture` | [Skin Texture](docs/image.md#bc_skintexture--skin-texture) | Micro-texture on skin inside a SAM 3 mask: boosts the image's own detail and multiplies in a synthetic pore field, in linear light |
| — | [Align](docs/workflow.md#align) | Align / distribute buttons in the selection toolbox |
| — | [Process Monitor](docs/process-monitor.md) | Not a node: live RAM / VRAM bars, an estimate before a run, per-node measurement, and the reason a killed run died (see [Process Monitor](docs/process-monitor.md)) |

Registration keys are BCNodes' own, so the packages above can be installed side by side without a clash. Type `BCNodes` in the node library to see them all; in the menu they sit in these groups:

| Category | Nodes |
| --- | --- |
| [`BCNodes/logic`](docs/logic.md) | Logic Boolean, Math Expression, Any Switch, Select Switch, Seed |
| [`BCNodes/mask`](docs/mask.md) | Mask Fill Holes, MaskGrow, Draw Mask On Image, Blockify Mask, Repeat Mask Batch, Is Mask Empty, BiRefNet Remove Background |
| [`BCNodes/image`](docs/image.md) | Image Scale By Aspect Ratio, Image Resize, Join Image Lists, Depth Anything, Social Media Export, Save Image, Save Image With Caption, Skin Texture |
| [`BCNodes/postfx`](docs/postfx.md) | PostFx Apply, Theme, Custom Look, LUT, Signature Sheet |
| [`BCNodes/analysis`](docs/analysis.md) | Image Quality Gate, Caption Audit |
| [`BCNodes/text`](docs/text.md) | Prompt List, Show Text |
| [`BCNodes/loaders`](docs/loaders.md) | Power Lora Loader, Auto Model Downloader |
| [`BCNodes/seedvr2`](docs/seedvr2.md) | SeedVR2 Resize, VAE Encode, VAE Decode, PostProcess |
| [`BCNodes/workflow`](docs/workflow.md) | Image Comparer, Anything Everywhere, Fast Groups Bypasser, Auto Bypass |

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/0xBeycan/ComfyUI-BCNodes
```

```
pip install -r ComfyUI-BCNodes/requirements.txt
```

Restart ComfyUI. `requirements.txt` holds `opencv-python` (the morphology of MaskGrow and the hole filling of Mask Fill Holes) and the two packages that back the PostFx and Caption Audit nodes, [`postfx`](https://github.com/0xBeycan/postfx) and [`caption-audit`](https://github.com/0xBeycan/caption-audit); everything else ships with ComfyUI. Without `postfx` and `caption-audit` the pack still loads and only those nodes are absent. If this repository is still present under its old name `ComfyUI-AutoBypass`, delete that folder — its `AutoBypass` node is this pack's `BC_AutoBypass`.

The Align buttons are not a node; they appear in the toolbox above a multi-selection.

## Documentation

- [Logic](docs/logic.md) — booleans, arithmetic, switches and the seed
- [Mask](docs/mask.md) — mask checks, fill / grow / draw / blockify / repeat, BiRefNet background removal
- [Image](docs/image.md) — scaling and resizing, image lists, depth maps, social media export, saving, skin texture
- [PostFx](docs/postfx.md) — film-emulation looks: themes, custom looks, LUTs, signature sheets
- [Analysis](docs/analysis.md) — the caption set audit and the image quality gate
- [Text](docs/text.md) — prompt lists and showing text
- [Loaders](docs/loaders.md) — LoRA loading and the model downloader
- [SeedVR2](docs/seedvr2.md) — the SeedVR2 input stage, streaming VAE encode / decode, post-processing
- [Workflow](docs/workflow.md) — image comparer, Anything Everywhere, group bypass, Auto Bypass, the Align buttons
- [Unused outputs](docs/unused-outputs.md) — unconnected whole-batch outputs come out empty
- [Measurements](docs/measurements.md) — time, RAM and output size of the mask nodes and Image Resize against the nodes they replace
- [Process Monitor](docs/process-monitor.md) — RAM / VRAM live, an estimate before a run, per-node measurement, the report of a killed run
- [Development](docs/development.md) — the repository layout, the tests and the import rule

## Third-party code

- `models/birefnet/arch/` — the BiRefNet architecture and its Swin v1 backbone, MIT, notices in [`models/birefnet/arch/LICENSE`](models/birefnet/arch/LICENSE).
- `models/depth_anything_v2/arch/` — the Depth Anything V2 DPT head and its DINOv2 ViT-S backbone, Apache-2.0, notices and license text in [`models/depth_anything_v2/arch/LICENSE`](models/depth_anything_v2/arch/LICENSE).

Everything else in this pack is BCNodes' own code.

## License

MIT. The code in `models/birefnet/arch/` keeps its own MIT notice, in `models/birefnet/arch/LICENSE`; the code in `models/depth_anything_v2/arch/` keeps its Apache-2.0 license, in `models/depth_anything_v2/arch/LICENSE`.

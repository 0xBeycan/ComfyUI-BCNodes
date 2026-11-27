# Unused outputs

A whole-batch IMAGE or MASK output that nothing is connected to comes out as an empty (0-frame)
tensor instead of staying in ComfyUI's cache until the prompt ends; where it is a step of its own,
the step does not run at all. Connecting such an output later runs the node again.

| Node | Output | When nothing is connected to it |
| --- | --- | --- |
| Image Resize | `IMAGE`, `mask` | not resized |
| Image Scale By Aspect Ratio | `image`, `mask` | not resized |
| BiRefNet Remove Background | `IMAGE`, `MASK_IMAGE` | not built |
| BiRefNet Remove Background | `MASK` | the matte, made for the other outputs, dropped |
| SeedVR2 Resize | `image`, `reference` | not made: each resize runs only for a connected output |
| Skin Texture | `image` | no texture |
| Skin Texture | `skin_mask` | dropped |
| Detailer | `image` | nothing sampled |
| Detailer | `mask` | dropped |
| Face Detailer | `image`, `mask` | both computed (the eyes are found on the detailed face), the unconnected one dropped |
| Dual Sampler | `second_image` | not decoded |
| Dual Sampler | `first_image` | not decoded, unless `blend_factor` is above 0: the pixel way decodes the first latent, and that image is dropped |

When a prompt is queued, the pack writes which of these outputs are connected into the node's
inputs (`bc_linked_heavy`), which makes the link state part of ComfyUI's cache key.

The limit: another custom node pack can change a queued prompt after this pack has read it (an
`on_prompt` handler that runs after this pack's), and a link it adds could then reach a cached empty
output. Once every custom node has loaded (at server startup), this pack moves its handler (and
ComfyUI-BCVideoNodes') after every other pack's. A handler added later, while ComfyUI runs, still
runs after it: for such a prompt the saving is off, every output comes out full as without this
feature, and the console says "RAM saving of unused outputs is off for this run: <pack> changes the
prompt after it." ComfyUI-BCVideoNodes does the same for its own nodes and is not counted.

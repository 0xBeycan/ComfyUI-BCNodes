# Unused outputs

A whole-batch IMAGE or MASK output that nothing is connected to comes out as an empty (0-frame)
tensor instead of staying in ComfyUI's cache until the prompt ends; where it is a step of its own,
the step does not run at all. That covers Image Resize `IMAGE` and `mask` and Image Scale By Aspect
Ratio `image` and `mask` (not resized), BiRefNet Remove Background `IMAGE` and `MASK_IMAGE` (not
built) and `MASK` (the matte, dropped), SeedVR2 Resize `image` and `reference` (each resize runs
only for a connected output), and Skin Texture `image` (no texture) and `skin_mask` (dropped).
Connecting such an output later runs the node again.

When a prompt is queued, the pack writes which of these outputs are connected into the node's
inputs (`bc_linked_heavy`), which makes the link state part of ComfyUI's cache key.

The limit: another custom node pack can change a queued prompt after this pack has read it (an
`on_prompt` handler that runs after this pack's), and a link it adds could then reach a cached empty
output. Once every custom node has loaded (at server startup), this pack moves its handler (and
ComfyUI-BCVideoNodes') after every other pack's. A handler added later, while ComfyUI runs, still
runs after it: for such a prompt the saving is off, every output comes out full as without this
feature, and the console says "RAM saving of unused outputs is off for this run: <pack> changes the
prompt after it." ComfyUI-BCVideoNodes does the same for its own nodes and is not counted.

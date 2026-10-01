# Measured against KJNodes and VideoHelperSuite

One Wan Animate replacement workflow, run once with KJNodes (d3cfe21) and VideoHelperSuite
(4d907be) and once with this pack (075ad7a) and ComfyUI-BCVideoNodes in their place: RTX PRO 6000
Blackwell (96 GB), ComfyUI 79be670e, a 1080 x 1920, 30 fps clip of 612 frames loaded at 720p (609
frames of 720 x 1280), the mask nodes on the CPU. Per node: the Process Monitor's time, RAM rise (the node's peak minus its start,
the container's working set sampled every 100 ms) and output size (what ComfyUI keeps in its
cache).

| Step, 609 frames | KJNodes | This pack |
| --- | --- | --- |
| Grow by 10, no blur | GrowMaskWithBlur (expand 10): 1.2 s, 6.33 GiB, 4.18 GiB (a second, inverted mask) | MaskGrow (grow 10, blur 0): 0.8 s, 2.09 GiB, 2.09 GiB |
| Blockify, 32 | BlockifyMask: 1.0 s, 4.19 GiB, 2.09 GiB | Blockify Mask: 0.5 s, 2.10 GiB, 2.09 GiB |
| Paint the mask black | DrawMaskOnImage (`0, 0, 0`): 1.3 s, 19.73 GiB, 6.27 GiB | Draw Mask On Image (`0, 0, 0`): 0.5 s, 6.29 GiB, 6.27 GiB |

Image Resize: the earlier workflow resized the 609 frames from 1080 x 1920 with ImageResizeKJv2
(lanczos, crop): 13.2 s, a 12.57 GiB RAM rise, 6.27 GiB of output, kept next to the 14.18 GiB of
full-size frames; the new one loads the video at 720 x 1280 with ComfyUI-BCVideoNodes' Load Video
instead, so no video resize ran. On single images (to 720 x 1280, in a second workflow that
animates one image) ImageResizeKJv2 and Image Resize took 0.02-0.03 s each, either pack.

Repeat Mask Batch: one 720 x 1280 mask to 81 frames, VHS Duplicate Masks and Repeat Mask Batch
both 0.01 s and a 0.28 GiB RAM rise for the 0.28 GiB output: no difference.

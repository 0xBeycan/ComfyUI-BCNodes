"""The SeedVR2 DiT's VRAM law: what one temporal chunk of the sampler needs on the card.

Fitted on an RTX 5090 (32 GB, device limit 31.36 GiB), 7B int8 DiT, 720x1280 -> 1080x1920 (the DiT at
1088x1920, 2.09 Mpx: latent 136 x 240), dynamic VRAM on, cudaMallocAsync, an 81-frame clip: the
KSampler's allocated peak at 21 / 33 / 41 / 49 / 57 frames per chunk (6 / 9 / 11 / 13 / 15 latent
frames) was 16.38 / 19.00 / 22.47 / 25.12 / 27.38 GiB. The least-squares line through them is
8.36 GiB + 1.269 GiB per latent frame, and the DiT's tokens are latent frames x latent area, so the
part per frame scales with the area: 1.269 / 2.089 = 0.6075 GiB per megapixel per latent frame
(core's own law, fitted on the 3B fp16, says 0.55; the 7B is wider). The fixed part holds the
weights and does not scale.

The allocated peak is not what a long clip on a busy card needs. Measured against that line, at the
driver level (the memory the card actually gives out):
  - the 5090 on a 900-frame clip at 49 frames failed in its 11th chunk with NVML at 31.8 GiB, 1.42x
    the line's part per frame (the allocator's reserve crept from 25.7 to 30.4 GiB over the chunks);
  - an RTX PRO 6000 (96 GB, device limit 94.97 GiB) on that clip at 161 frames ran with NVML at
    83.5 GiB, 1.44x the line (its allocated peak, 78.0 GiB, is far above the 60.4 the line gives:
    a big card holds more than the line counts), and failed at 281 frames.
So the part per frame is budgeted at (1 + safety_margin) x the line. SAFETY_MARGIN (0.64) keeps 14%
over the worst of those two (1.64 / 1.44) and picks 41 frames on the 5090 and 161 on the PRO 6000
at 2.09 Mpx: the largest chunk measured to run on each card's long clip or, on the 5090, the next
4n+1 below the 49 that failed. The 3B is narrower than the 7B, so the law over-asks for it (safe).
"""

FIXED_GIB = 8.36
GIB_PER_MPX_LATENT_FRAME = 0.6075
SAFETY_MARGIN = 0.64

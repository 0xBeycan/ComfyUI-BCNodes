"""SAM 3 text detection through ComfyUI core's SAM3_Detect node (imported on the first call)."""


def detect(model, clip, image, text, threshold, refine_iterations=2):
    """SAM 3 text detection through the core node: -> (B, H, W) union mask, per-frame boxes (a list per
    frame of {x, y, width, height, score} in pixels). `refine_iterations` 0 skips the SAM decoder's
    mask refinement (the boxes are the detector's either way)."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    out = SAM3_Detect.execute(model, image, conditioning=cond, threshold=threshold, refine_iterations=refine_iterations,
                              individual_masks=False)
    masks, boxes = out.args
    return masks.float().cpu(), boxes

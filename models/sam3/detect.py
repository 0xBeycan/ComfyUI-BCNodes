"""SAM 3 text detection through ComfyUI core's SAM3_Detect node (imported on the first call)."""


def text_condition(clip, text):
    """The conditioning SAM 3 detects `text` with: encoded once, then handed to every detect call."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def detect(model, cond, image, threshold, refine_iterations=2):
    """SAM 3 text detection through the core node, `cond` from text_condition: -> (B, H, W) union mask,
    per-frame boxes (a list per frame of {x, y, width, height, score} in pixels). `refine_iterations` 0
    skips the SAM decoder's mask refinement (the boxes are the detector's either way)."""
    from comfy_extras.nodes_sam3 import SAM3_Detect

    out = SAM3_Detect.execute(model, image, conditioning=cond, threshold=threshold, refine_iterations=refine_iterations,
                              individual_masks=False)
    masks, boxes = out.args
    return masks.float().cpu(), boxes

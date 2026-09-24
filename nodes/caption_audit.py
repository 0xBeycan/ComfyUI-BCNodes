"""Caption Audit — run caption-audit over a dataset folder and show the card.

    BC_CaptionAudit (Caption Audit)

pipelines/caption_audit/ imports `caption_audit` and PIL inside the functions
that use them: the package is a pip dependency, and a missing install must not
take the rest of the pack down with it.
"""

import os

from ..pipelines.caption_audit.audit import (
    build_args, dir_fingerprint, report_json, report_text, resolve_dir, run_audit,
)
from ..pipelines.caption_audit.card import render_card, render_error_card


# --- PIL <-> ComfyUI IMAGE bridge -----------------------------------------

def pil_to_image(img):
    """PIL RGB image -> ComfyUI IMAGE tensor (1, H, W, 3), float32, [0,1]."""
    import numpy as np
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def save_temp_preview(img, prefix="caption_audit"):
    """Write the card into ComfyUI's temp dir and describe it for the UI.

    Returns the ``{"images": [...]}`` payload PreviewImage uses, so the card
    shows inside the node with no frontend extension of our own. Outside
    ComfyUI (folder_paths missing) the file still lands in the system temp
    directory and an empty payload is returned.
    """
    import tempfile
    import uuid

    try:
        import folder_paths
        temp_dir = folder_paths.get_temp_directory()
        in_comfy = True
    except Exception:
        temp_dir = tempfile.gettempdir()
        in_comfy = False

    os.makedirs(temp_dir, exist_ok=True)
    filename = "%s_%s.png" % (prefix, uuid.uuid4().hex[:12])
    path = os.path.join(temp_dir, filename)
    img.save(path, compress_level=4)
    if not in_comfy:
        return {"images": []}, path
    return {"images": [{"filename": filename, "subfolder": "", "type": "temp"}]}, path


# --- the node --------------------------------------------------------------

class CaptionAudit:
    CATEGORY = "BCNodes/analysis"
    SEARCH_ALIASES = ["BCNodes", "caption audit", "lora captions", "trigger word", "dataset"]
    FUNCTION = "audit"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("report_image", "report_text", "report_json",
                    "critical", "warning")
    DESCRIPTION = (
        "Audit a LoRA caption set for tokens fused with the trigger word. Point "
        "'directory' at a folder of images + same-named .txt captions; the audit "
        "runs there and the report card renders inside the node.\n\n"
        "The primary measure is document frequency — in how many captions a term "
        "appears at least once, across words, phrases and whole comma segments. A "
        "term near 100% cannot be prompted in or out at inference: the model "
        "cannot tell it apart from the trigger.\n\n"
        "This reads .txt files only, never the images. Every flagged term has two "
        "possible causes needing opposite fixes, so the card ends with the "
        "question you have to answer by looking at the pictures. Wire 'critical' "
        "into a gate to stop a training workflow before it starts.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Folder holding the caption set: images plus "
                               ".txt sidecars sharing each image's basename. "
                               "Empty = ComfyUI's working directory."}),
                "trigger": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "The trigger word this LoRA is supposed to own. "
                               "Leave empty to have it inferred from the "
                               "captions — the card marks that [INFERRED], and "
                               "an inferred guess is worth checking."}),
                "class_words": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated class words, e.g. 'woman, car'. "
                               "Shown as EXPECTED and never flagged: they are "
                               "supposed to be everywhere. Coverage is measured "
                               "and shown per word — one you declare but never "
                               "wrote into the captions is a WARNING, not a "
                               "silent pass."}),
                "fuse": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Comma-separated attributes you WANT welded to "
                               "the trigger, e.g. 'red scarf'. Covers the "
                               "phrase and its fragments, shown as INTENDED, "
                               "excluded from the critical count. Keeps its row "
                               "even at 0%, so a --fuse that no longer matches "
                               "your captions is visible rather than inert."}),
                "critical_threshold": ("FLOAT", {"default": 0.85, "min": 0.05,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Document frequency at or above which a term "
                               "counts as fused with the trigger."}),
                "warn_threshold": ("FLOAT", {"default": 0.60, "min": 0.05,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Strong bias: will bleed into unrelated prompts."}),
                "info_threshold": ("FLOAT", {"default": 0.35, "min": 0.01,
                    "max": 1.0, "step": 0.01,
                    "tooltip": "Below this a term is not reported at all."}),
                "ngram_max": ("INT", {"default": 3, "min": 1, "max": 5,
                    "tooltip": "Longest phrase analysed. Phrases never cross a "
                               "comma, so tag lists produce no phantom n-grams."}),
                "no_stopwords": ("BOOLEAN", {"default": False,
                    "tooltip": "Off (default) hides function words like 'a' / "
                               "'with' from the flag list. On shows everything."}),
                "recursive": ("BOOLEAN", {"default": False,
                    "tooltip": "Descend into subdirectories."}),
                "table_rows": ("INT", {"default": 12, "min": 1, "max": 60,
                    "tooltip": "Rows in the card's term table. This is the only "
                               "input that changes the card's size — the canvas "
                               "is otherwise fixed, so the node does not resize "
                               "between runs. Unused rows show an em dash. Terms "
                               "you declared always keep a row here."}),
            },
            "optional": {
                "images_dir": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "Where the images live, when captions are kept "
                               "in a separate folder. Empty = alongside the "
                               "captions; with no images anywhere the audit "
                               "runs in caption-only mode."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, directory="", recursive=False, **kwargs):
        # Captions are edited outside the graph, so widget values alone do not
        # say whether a re-run is needed. Fingerprint the .txt files instead:
        # an untouched dataset still hits the execution cache.
        try:
            resolved = resolve_dir(directory)
        except ValueError as exc:
            # A rejected path is a stable state, not a change: returning the
            # message keeps the node cached on the error card it already drew.
            return "rejected:%s" % exc
        return dir_fingerprint(resolved, recursive)

    def audit(self, directory, trigger, class_words, fuse, critical_threshold,
              warn_threshold, info_threshold, ngram_max, no_stopwords,
              recursive, table_rows, images_dir=""):
        # resolve_dir rejects a path outside the allowed roots, so it belongs
        # inside the same try as the audit: both end on the error card.
        shown = str(directory or "").strip()
        try:
            resolved = resolve_dir(directory)
            shown = resolved
            args = build_args(
                resolved, trigger=trigger, images_dir=images_dir,
                class_words=class_words, fuse=fuse,
                critical_threshold=critical_threshold,
                warn_threshold=warn_threshold, info_threshold=info_threshold,
                ngram_max=ngram_max, no_stopwords=no_stopwords, recursive=recursive)
            rep = run_audit(resolved, args)
        except Exception as exc:  # never raise into the graph; show the failure
            img = render_error_card("%s: %s" % (type(exc).__name__, exc),
                                         shown, table_rows)
            ui, _path = save_temp_preview(img)
            text = "caption-audit: %s" % exc
            # critical=1 so a gate wired to it stops the workflow: a caption set
            # that could not be audited has not been cleared for training.
            return {"ui": ui,
                    "result": (pil_to_image(img), text, "{}", 1, 0)}

        img = render_card(rep, args, table_rows=table_rows)
        ui, _path = save_temp_preview(img)
        return {
            "ui": ui,
            "result": (
                pil_to_image(img),
                report_text(rep, args),
                report_json(rep),
                int(rep["summary"]["critical"]),
                int(rep["summary"]["warning"]),
            ),
        }


NODE_CLASS_MAPPINGS = {"BC_CaptionAudit": CaptionAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_CaptionAudit": "Caption Audit"}

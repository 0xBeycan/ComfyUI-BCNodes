"""PreFlight nodes: how Instagram, TikTok and X would likely treat an image or a video before it is published.

    BC_PreFlightObserve    (PreFlight Observe)    Qwen3.5-9B INT8 ConvRot as a pure observation sensor -> JSON
    BC_PreFlightReport     (PreFlight Report)     the rules engine -> per-platform verdict ranges, summary, record id
    BC_PreFlightOutcome    (PreFlight Outcome)    what a platform actually did to a published post, into the store
    BC_PreFlightCalibrate  (PreFlight Calibrate)  the calibration tables over the feedback store

The flows are pipelines/preflight. Observe runs its fixed model through the LM runtime (pipelines/lm.run on
the Qwen LM family: the model file is found in models/Qwen-LM or models/text_encoders, or downloaded on first
use). This module holds the surface: the widgets, the feedback store's path
(<ComfyUI user dir>/BCNodes/preflight/feedback.jsonl), GET /bcnodes/preflight/records (registered by
register_routes(), called from the root __init__, only inside a running ComfyUI server) and the
bcnodes.preflight.records event Report sends after it logs a prediction: web/js/preflight.js reads both to
keep every Outcome node's record list current. folder_paths, server and aiohttp are imported inside the
functions.
"""

import itertools
import os

from .common import lm_folders

CATEGORY = "BCNodes/preflight"
FAMILY = "qwen_lm"  # the LM family (models/common/registry.py LM_FAMILY) of Observe's model
RECORDS_ROUTE = "/bcnodes/preflight/records"
RECORDS_EVENT = "bcnodes.preflight.records"


def store_path():
    """The feedback store: <ComfyUI user dir>/BCNodes/preflight/feedback.jsonl."""
    from ..libs.download import user_file

    return user_file(os.path.join("preflight", "feedback.jsonl"))


def records_payload():
    """The record list as GET /bcnodes/preflight/records and the records event carry it
    (pipelines/preflight/outcome.RecordsPayload)."""
    from ..pipelines.preflight.outcome import RecordsPayload, record_labels

    return RecordsPayload(records=record_labels(store_path()))


def send_records():
    """Sends the record list to every open browser (RECORDS_EVENT): each Outcome node lists a prediction as
    soon as it is logged. Only inside a running ComfyUI server; nothing without one."""
    try:
        from server import PromptServer
    except ImportError:  # not under a ComfyUI server (tests, the import gate)
        return
    server = getattr(PromptServer, "instance", None)
    if server is not None:
        server.send_sync(RECORDS_EVENT, records_payload())


def observe_generator(frames):
    """generate(extra_instruction) -> the raw answer: one LM run of Observe's model (pipelines/preflight/observe
    MODEL at PRECISION) over `frames`, the images then user_text(extra_instruction) as the user turn, no
    system turn, thinking off, no prefill, greedy (GREEDY), SEED, MAX_NEW_TOKENS. The model stays loaded
    between the runs of one observation; Observe unloads it afterwards when keep_model_loaded is off."""
    from ..models.common import registry
    from ..models.common.registry import LM_FAMILY
    from ..pipelines.lm import NO_LORA, LMRequest, run
    from ..pipelines.preflight import observe as flow

    model_folders, text_encoder_folders = lm_folders(registry.get(LM_FAMILY, FAMILY).folder)

    def generate(extra_instruction):
        return run(LMRequest(
            family=FAMILY, node=NODE_DISPLAY_NAME_MAPPINGS["BC_PreFlightObserve"], model=flow.MODEL,
            precision=flow.PRECISION, lora=NO_LORA, lora_strength=0.0, thinking=False,
            max_new_tokens=flow.MAX_NEW_TOKENS, seed=flow.SEED, keep_model_loaded=True, system="",
            user=flow.user_text(extra_instruction), assistant="", image=frames, config=dict(flow.GREEDY),
            model_folders=model_folders, text_encoder_folders=text_encoder_folders)).text

    return generate


def unload_lm():
    """Unloads every registered LM backend (models/common/registry.py LM_BACKEND): the one slot that holds
    Observe's model, shared with the LM nodes."""
    from ..models.common import registry
    from ..models.common.registry import LM_BACKEND

    for name in registry.names(LM_BACKEND):
        registry.get(LM_BACKEND, name).unload()


class PreFlightObserve:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "A single image or a video frame batch."}),
                "max_frames": ("INT", {"default": 6, "min": 1, "max": 16,
                                       "tooltip": "For a video: how many frames to sample evenly from the batch. "
                                                  "All of them go to the model in one call."}),
                "keep_model_loaded": ("BOOLEAN", {"default": True,
                                                  "tooltip": "On: the model stays loaded for the next run (one LM "
                                                             "model is held at a time, shared with Qwen LM). Off: it "
                                                             "is unloaded when the run ends."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("observations_json", "raw_response")
    FUNCTION = "observe"
    CATEGORY = CATEGORY
    SEARCH_ALIASES = ["BCNodes", "preflight", "observe", "content check", "moderation", "instagram", "tiktok"]
    DESCRIPTION = (
        "Runs Qwen3.5-9B INT8 ConvRot (downloaded on first use) as a pure observation sensor. It describes what "
        "is visually present (garment, exposure, framing, pose, in-image text, ...) as a strict JSON object; it "
        "does NOT judge acceptability. Feed the result into PreFlight Report.\n\n"
        "Deterministic: greedy decoding, so the same image and settings give the same output run to run. Errors "
        "never break the graph: a problem comes back as {\"error\": ...} so Report can fail closed.")

    def observe(self, image, max_frames, keep_model_loaded):
        from ..pipelines.preflight.observe import observe

        try:
            return observe(image, max_frames, observe_generator)
        finally:
            if not keep_model_loaded:
                unload_lm()


class PreFlightReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "observations_json": ("STRING", {"forceInput": True,
                                                 "tooltip": "The observations_json output of PreFlight Observe."}),
                "caption": ("STRING", {"default": "", "multiline": True,
                                       "tooltip": "Optional caption to be published. Scanned for adult solicitation / "
                                                  "links together with the in-image text."}),
                "log_prediction": ("BOOLEAN", {"default": True,
                                               "tooltip": "Append this prediction to the feedback store so you can "
                                                          "later record what the platform actually did (PreFlight "
                                                          "Outcome)."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional pass-through so this node can sit inline in a workflow."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("report_json", "summary", "record_id", "image")
    FUNCTION = "report"
    CATEGORY = CATEGORY
    SEARCH_ALIASES = ["BCNodes", "preflight", "report", "content check", "moderation", "instagram", "tiktok"]
    DESCRIPTION = (
        "Applies the PreFlight rules engine to a sensor observation and predicts, per platform (Instagram / "
        "TikTok / X), whether the content risks removal or reach suppression. Outputs the full JSON report, a "
        "human-readable summary and the feedback record id.\n\n"
        "This is a prediction, not an approval gate. Verdicts are ranges (best..worst); range drivers name the "
        "unknown that would collapse them.")

    def report(self, observations_json, caption="", log_prediction=True, image=None):
        from ..pipelines.preflight.report import report

        report_json, summary, record_id = report(observations_json, caption, log_prediction, store_path())
        if record_id:
            send_records()
        return (report_json, summary, record_id, image)


# A new value on every call, so a repeated queue with identical inputs still writes (ComfyUI's cache would
# otherwise skip logging the second platform); a counter cannot collide the way two same-tick clock reads can.
_IS_CHANGED_COUNTER = itertools.count()


class PreFlightOutcome:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.preflight.feedback import PLATFORMS, RESULTS
        from ..pipelines.preflight.outcome import record_labels

        return {
            "required": {
                "record": (record_labels(store_path()),
                           {"tooltip": "The prediction to attach an outcome to, newest first. The list follows the "
                                       "feedback store: a prediction PreFlight Report logs shows here at once."}),
                "platform": (list(PLATFORMS), {"default": "instagram", "tooltip": "Which platform this outcome is for."}),
                "result": (list(RESULTS), {"default": "clean",
                                           "tooltip": "clean = normal reach; demoted = suppressed / FYF-ineligible / "
                                                      "flagged; removed = taken down. Pick one fixed personal "
                                                      "heuristic for 'demoted' and stick to it."}),
                "record_id_override": ("STRING", {"default": "",
                                                  "tooltip": "If set, this record id is used instead of the list "
                                                             "selection (the id is in the Report summary), e.g. for a "
                                                             "prediction older than the 50 the list shows."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "log"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    SEARCH_ALIASES = ["BCNodes", "preflight", "outcome", "feedback"]
    DESCRIPTION = (
        "Records what a platform actually did to a published post: the feedback half of the loop. Pick the "
        "prediction from the list (newest first), the platform and the result (clean / demoted / removed), then "
        "queue once per platform. No image input: outcomes attach to records, not files.\n\n"
        "The list follows the feedback store: a prediction PreFlight Report logs appears in every Outcome node at "
        "once. record_id_override bypasses the list entirely; the id is shown in the Report summary.")

    @classmethod
    def VALIDATE_INPUTS(cls, record):
        # Any record label: a saved workflow's selection, or one the list no longer shows, still runs (its id
        # is the label's first token; record_id_override wins over it).
        return True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return next(_IS_CHANGED_COUNTER)

    def log(self, record, platform, result, record_id_override=""):
        from ..pipelines.preflight.outcome import log

        return (log(record, platform, result, record_id_override, store_path()),)


class PreFlightCalibrate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "calibrate"
    CATEGORY = CATEGORY
    SEARCH_ALIASES = ["BCNodes", "preflight", "calibrate", "calibration", "feedback"]
    DESCRIPTION = (
        "The calibration report of the PreFlight feedback store, as text (connect it to Show Text): per platform, "
        "how often the real outcome fell inside the predicted range, above it (under-predicted) or below it "
        "(over-predicted); per rule, how many posts it fired on and their outcomes. It reads only: rule changes "
        "are made by hand. It runs again when the store has changed, not on every queue.")

    @classmethod
    def IS_CHANGED(cls):
        from ..pipelines.preflight.feedback import store_stamp

        return store_stamp(store_path())

    def calibrate(self):
        from ..pipelines.preflight.calibrate import build_report
        from ..pipelines.preflight.feedback import load_joined

        return (build_report(load_joined(store_path())),)


def register_routes():
    """GET /bcnodes/preflight/records (records_payload), served off the event loop. Only inside a running
    ComfyUI server; returns at once without one."""
    try:
        from server import PromptServer
    except ImportError:  # not under a ComfyUI server (tests, the import gate)
        return
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    import asyncio

    from aiohttp import web

    @server.routes.get(RECORDS_ROUTE)
    async def _records(request):
        return web.json_response(await asyncio.get_running_loop().run_in_executor(None, records_payload))


NODE_CLASS_MAPPINGS = {
    "BC_PreFlightObserve": PreFlightObserve,
    "BC_PreFlightReport": PreFlightReport,
    "BC_PreFlightOutcome": PreFlightOutcome,
    "BC_PreFlightCalibrate": PreFlightCalibrate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_PreFlightObserve": "PreFlight Observe",
    "BC_PreFlightReport": "PreFlight Report",
    "BC_PreFlightOutcome": "PreFlight Outcome",
    "BC_PreFlightCalibrate": "PreFlight Calibrate",
}

"""The PreFlight nodes (nodes/preflight.py) through the node classes, on the stub folder_paths (the user
directory and models_dir in tmp_path) and a recording stand-in for the "core" LM backend (no weights):

  - the surface: the four nodes' widgets, defaults, ranges and tooltips, outputs, category, display names;
    Observe's description names its fixed model and the download;
  - the feedback store lives in <user dir>/BCNodes/preflight/feedback.jsonl; the old place under the output
    folder is never read;
  - Observe's generate callable: the LMRequest it hands pipelines/lm.run (the fixed model and precision, no
    LoRA, thinking off, greedy config, seed 42, 300 tokens, no system turn, no prefill, the frames as the
    image batch, the prompt as the user text, the retry's instruction appended), and through the real
    pipelines/lm.run the backend's prompt (the images, then the prompt) and GenerateParams; the model kept
    loaded across the retry and unloaded once afterwards when keep_model_loaded is off, after an error and an
    interrupt too; an error comes back as {"error": ...}, which Report fails closed on; ComfyUI's interrupt
    is raised;
  - Report: the prediction logged and its id injected into the JSON and the summary, the image passed
    through, nothing written with logging off, a store failure survived; the record list sent to the
    browsers (bcnodes.preflight.records) only when a prediction was logged;
  - Outcome: the record list from the store, newest first; any record label validates; override, no
    selection and a bad value as status strings; IS_CHANGED differs on every call;
  - Calibrate: the tables of the store; IS_CHANGED is the store's stamp ("" without a store, the same until
    the store changes);
  - GET /bcnodes/preflight/records on a stand-in server, and no route without a running server.
The flows themselves are tests/layers/pipelines/test_pipe_preflight_*.py.
"""

import asyncio
import json
import os
import sys
import types

import pytest
import torch

VALID_JSON = json.dumps({
    "subject_appears_under_18": False, "garment": "bikini", "setting": "beach_pool",
    "framing": "full_body", "pose": "neutral", "see_through_or_wet": False,
    "exposure": "mild", "nudity_or_sexual_act": False, "motion_flags": "none",
    "visible_text": "", "confidence": "high",
})
IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture
def pf(bcnodes):
    return bcnodes["preflight"]


@pytest.fixture
def flow(bcnodes):
    return bcnodes["pipelines.preflight.observe"]


@pytest.fixture
def fp(comfy_stubs, monkeypatch, tmp_path):
    """The stub folder_paths with models_dir, the user directory and the output directory in tmp_path, and one
    text_encoders folder."""
    fp = sys.modules["folder_paths"]
    monkeypatch.setattr(fp, "models_dir", str(tmp_path / "models"))
    monkeypatch.setattr(fp, "get_user_directory", lambda: str(tmp_path / "user"))
    monkeypatch.setattr(fp, "get_output_directory", lambda: str(tmp_path / "output"))
    monkeypatch.setattr(fp, "folder_names_and_paths", {"text_encoders": ([str(tmp_path / "models" / "text_encoders")],
                                                                         {".safetensors"})})
    return fp


@pytest.fixture
def store(fp, tmp_path):
    return str(tmp_path / "user" / "BCNodes" / "preflight" / "feedback.jsonl")


class Backend:
    """The "core" LM backend stand-in: records its calls in `events`; generate returns the next of `answers`
    (the last one again when they run out), or raises `error`."""
    name = "core"

    def __init__(self):
        self.events, self.answers, self.error, self.generated = [], [VALID_JSON], None, []

    def load(self, path, lora):
        self.events.append(("load", path, lora))
        return "handle"

    def generate(self, handle, prompt, images, params):
        self.events.append(("generate",))
        self.generated.append((prompt, images, params))
        if self.error is not None:
            raise self.error
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def unload(self):
        self.events.append(("unload",))
        return {}


@pytest.fixture
def backend(bcnodes, monkeypatch, fp, tmp_path):
    """The stand-in backend as "core", and the model file in models/Qwen-LM."""
    registry = bcnodes["models.common.registry"]
    fake = Backend()
    monkeypatch.setitem(registry._FAMILIES[registry.LM_BACKEND], "core", fake)
    folder = tmp_path / "models" / "Qwen-LM"
    folder.mkdir(parents=True)
    (folder / "Qwen3.5-9B_int8_convrot.safetensors").write_bytes(b"")
    return fake


def obs_json(**kw):
    base = dict(
        subject_appears_under_18=False, garment="regular", setting="studio",
        framing="full_body", pose="neutral", see_through_or_wet=False,
        exposure="none", nudity_or_sexual_act=False, motion_flags="none",
        visible_text="", confidence="high", meta={"schema_version": "1",
                                                  "prompt_version": "1", "model_name": "test", "frames_analyzed": 1})
    base.update(kw)
    return json.dumps(base)


# --- the surface ---------------------------------------------------------------------------------------------


def test_observe_inputs(pf):
    spec = pf.PreFlightObserve.INPUT_TYPES()
    req = spec["required"]
    assert list(spec) == ["required"] and list(req) == ["image", "max_frames", "keep_model_loaded"]
    assert req["image"][0] == "IMAGE"
    assert req["max_frames"][0] == "INT" and {k: req["max_frames"][1][k] for k in ("default", "min", "max")} == {
        "default": 6, "min": 1, "max": 16}
    assert req["keep_model_loaded"][0] == "BOOLEAN" and req["keep_model_loaded"][1]["default"] is True
    assert all(v[1].get("tooltip") for v in req.values())


def test_report_inputs(pf):
    spec = pf.PreFlightReport.INPUT_TYPES()
    req, opt = spec["required"], spec["optional"]
    assert list(req) == ["observations_json", "caption", "log_prediction"] and list(opt) == ["image"]
    assert req["observations_json"][0] == "STRING" and req["observations_json"][1]["forceInput"] is True
    assert req["caption"][0] == "STRING" and req["caption"][1]["default"] == "" and req["caption"][1]["multiline"] is True
    assert req["log_prediction"][0] == "BOOLEAN" and req["log_prediction"][1]["default"] is True
    assert opt["image"][0] == "IMAGE"
    assert all(v[1].get("tooltip") for v in (*req.values(), *opt.values()))


def test_outcome_inputs(pf, store):
    req = pf.PreFlightOutcome.INPUT_TYPES()["required"]
    assert list(req) == ["record", "platform", "result", "record_id_override"]
    assert req["record"][0] == ["no records yet"]
    assert req["platform"][0] == ["instagram", "tiktok", "x"] and req["platform"][1]["default"] == "instagram"
    assert req["result"][0] == ["clean", "demoted", "removed"] and req["result"][1]["default"] == "clean"
    assert req["record_id_override"][0] == "STRING" and req["record_id_override"][1]["default"] == ""
    assert all(v[1].get("tooltip") for v in req.values())
    assert pf.PreFlightCalibrate.INPUT_TYPES() == {"required": {}}


def test_outputs_and_names(pf):
    observe, report, outcome, calibrate = (pf.PreFlightObserve, pf.PreFlightReport, pf.PreFlightOutcome,
                                           pf.PreFlightCalibrate)
    assert (observe.RETURN_TYPES, observe.RETURN_NAMES, observe.FUNCTION) == (
        ("STRING", "STRING"), ("observations_json", "raw_response"), "observe")
    assert (report.RETURN_TYPES, report.RETURN_NAMES, report.FUNCTION) == (
        ("STRING", "STRING", "STRING", "IMAGE"), ("report_json", "summary", "record_id", "image"), "report")
    assert (outcome.RETURN_TYPES, outcome.RETURN_NAMES, outcome.FUNCTION) == (("STRING",), ("status",), "log")
    assert (calibrate.RETURN_TYPES, calibrate.RETURN_NAMES, calibrate.FUNCTION) == (("STRING",), ("report",), "calibrate")
    assert pf.NODE_CLASS_MAPPINGS == {"BC_PreFlightObserve": observe, "BC_PreFlightReport": report,
                                      "BC_PreFlightOutcome": outcome, "BC_PreFlightCalibrate": calibrate}
    assert pf.NODE_DISPLAY_NAME_MAPPINGS == {"BC_PreFlightObserve": "PreFlight Observe", "BC_PreFlightReport": "PreFlight Report",
                                             "BC_PreFlightOutcome": "PreFlight Outcome",
                                             "BC_PreFlightCalibrate": "PreFlight Calibrate"}
    for cls in pf.NODE_CLASS_MAPPINGS.values():
        assert cls.CATEGORY == "BCNodes/preflight" and cls.DESCRIPTION and cls.SEARCH_ALIASES[0] == "BCNodes"
        assert getattr(cls, "OUTPUT_NODE", False) is (cls is outcome)
    assert observe.DESCRIPTION.startswith("Runs Qwen3.5-9B INT8 ConvRot (downloaded on first use)")


# --- the store's place ---------------------------------------------------------------------------------------


def test_store_in_the_user_directory(pf, store, tmp_path):
    assert pf.store_path() == store
    _, _, rid, _ = pf.PreFlightReport().report(obs_json(), "", True)
    assert rid and os.path.isfile(store)
    assert pf.PreFlightOutcome.INPUT_TYPES()["required"]["record"][0][0].startswith(rid + " ")


def test_the_old_store_is_never_read(pf, store, tmp_path, bcnodes):
    old = tmp_path / "output" / "preflight" / "feedback.jsonl"
    feedback = bcnodes["pipelines.preflight.feedback"]
    feedback.log_prediction(bcnodes["pipelines.preflight.rules"].judge(json.loads(obs_json())), path=old)
    assert pf.PreFlightOutcome.INPUT_TYPES()["required"]["record"][0] == ["no records yet"]
    assert "Store is empty" in pf.PreFlightCalibrate().calibrate()[0] and not os.path.exists(store)


# --- Observe -------------------------------------------------------------------------------------------------


def test_generate_builds_the_request(pf, flow, fp, bcnodes, monkeypatch, tmp_path):
    pipe = bcnodes["pipelines.lm"]
    seen = []
    monkeypatch.setattr(pipe, "run", lambda request: seen.append(request) or pipe.LMResult("answer", ""))
    frames = torch.rand(3, 8, 8, 3)
    generate = pf.observe_generator(frames)
    assert generate("") == "answer" and generate(flow.RETRY_INSTRUCTION) == "answer"
    expected = dict(
        family="qwen_lm", node="PreFlight Observe", model="Qwen3.5-9B", precision="INT8 ConvRot", lora="None",
        lora_strength=0.0, thinking=False, max_new_tokens=300, seed=42, keep_model_loaded=True, system="",
        user=flow.prompts.OBSERVATION_PROMPT, assistant="", image=frames,
        config={"do_sample": False, "repetition_penalty": 1.0, "presence_penalty": 0.0},
        model_folders=(str(tmp_path / "models" / "Qwen-LM"),),
        text_encoder_folders=(str(tmp_path / "models" / "text_encoders"),))
    assert seen[0] == pipe.LMRequest(**expected)
    assert seen[1] == pipe.LMRequest(**{**expected, "user": flow.prompts.OBSERVATION_PROMPT + "\n\n" + flow.RETRY_INSTRUCTION})
    assert seen[0].config is not flow.GREEDY  # a copy: the run never shares the module's dict
    assert fp.folder_names_and_paths["Qwen-LM"] == ([str(tmp_path / "models" / "Qwen-LM")], {".safetensors"})


def test_observe_through_the_lm_runtime(pf, flow, backend, tmp_path):
    image = torch.rand(10, 64, 96, 3)
    obs_json_out, raw = pf.PreFlightObserve().observe(image, 6, True)
    obs = json.loads(obs_json_out)
    assert obs["garment"] == "bikini" and raw == VALID_JSON
    assert obs["meta"] == {"schema_version": "1", "prompt_version": "5", "model_name": "Qwen3.5-9B",
                           "precision": "INT8 ConvRot", "frames_analyzed": 6}
    [(prompt, images, params)] = backend.generated
    # the images, then the prompt, in the user turn; no system turn, the thinking-off block, no prefill
    assert prompt == ("<|im_start|>user\n" + IMAGE_BLOCK * 6 + flow.prompts.OBSERVATION_PROMPT
                      + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert images.shape[0] == 6 and images.dtype == torch.float32
    assert (params.do_sample, params.presence_penalty, params.repetition_penalty, params.seed, params.max_new_tokens,
            params.mtp) == (False, 0.0, 1.0, 42, 300, "auto")
    assert backend.events == [("load", str(tmp_path / "models" / "Qwen-LM" / "Qwen3.5-9B_int8_convrot.safetensors"), None),
                              ("generate",)]  # kept loaded


def test_retry_keeps_the_model_and_unloads_once(pf, flow, backend):
    backend.answers = ["Sorry, I cannot help with that.", VALID_JSON]
    obs_json_out, raw = pf.PreFlightObserve().observe(torch.rand(1, 64, 64, 3), 6, False)
    assert json.loads(obs_json_out)["garment"] == "bikini" and "--- retry ---" in raw
    assert [e[0] for e in backend.events] == ["load", "generate", "load", "generate", "unload"]
    assert backend.generated[1][0].endswith(flow.RETRY_INSTRUCTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_observe_error_fails_closed_and_report_gives_unknown(pf, backend, store):
    backend.error = RuntimeError("CUDA out of memory")
    obs_json_out, raw = pf.PreFlightObserve().observe(torch.rand(1, 64, 64, 3), 6, False)
    assert json.loads(obs_json_out) == {"error": "RuntimeError: CUDA out of memory"} and raw == ""
    assert backend.events[-1] == ("unload",)
    report_json, summary, rid, _ = pf.PreFlightReport().report(obs_json_out, "", False)
    assert json.loads(report_json)["unknown"] is True and rid == ""


def test_observe_raises_the_interrupt(pf, backend):
    class InterruptProcessingException(BaseException):  # as comfy.model_management's
        pass

    backend.error = InterruptProcessingException()
    with pytest.raises(InterruptProcessingException):
        pf.PreFlightObserve().observe(torch.rand(1, 64, 64, 3), 6, False)
    assert backend.events[-1] == ("unload",)  # unloaded on the way out


def test_unload_lm_unloads_every_backend(pf, bcnodes, monkeypatch):
    registry = bcnodes["models.common.registry"]
    a, b = Backend(), Backend()
    monkeypatch.setitem(registry._FAMILIES, registry.LM_BACKEND, {"core": a, "other": b})
    pf.unload_lm()
    assert a.events == b.events == [("unload",)]


# --- Report --------------------------------------------------------------------------------------------------


@pytest.fixture
def sent(monkeypatch):
    """A stand-in running server: the events Report sends, as (event, data)."""
    events = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        send_sync=lambda event, data, sid=None: events.append((event, data))))
    monkeypatch.setitem(sys.modules, "server", server)
    return events


def test_report_logs_and_injects_record_id(pf, store, sent):
    report_json, summary, rid, img = pf.PreFlightReport().report(
        obs_json(garment="bikini", setting="bedroom", exposure="mild"), caption="hi", log_prediction=True, image="IMG")
    assert len(rid) == 8
    assert json.loads(report_json)["record_id"] == rid   # injected into JSON
    assert ("record: " + rid) in summary                 # and into the summary
    assert img == "IMG"                                   # pass-through
    with open(store) as f:
        assert len(f.read().strip().splitlines()) == 1   # exactly one line
    [(event, data)] = sent
    assert event == "bcnodes.preflight.records" and data == pf.records_payload() and data["records"][0].startswith(rid + " ")


def test_report_logging_off_writes_nothing(pf, store, sent):
    report_json, summary, rid, _ = pf.PreFlightReport().report(obs_json(), caption="", log_prediction=False)
    assert rid == "" and "record:" not in summary and not os.path.exists(store) and sent == []


def test_report_survives_store_failure(pf, store, sent, bcnodes, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(bcnodes["pipelines.preflight.feedback"], "log_prediction", boom)
    report_json, summary, rid, _ = pf.PreFlightReport().report(obs_json(garment="lingerie", exposure="significant"), "", True)
    assert rid == "" and sent == []
    assert json.loads(report_json)["verdicts"]["tiktok"]["worst"] == "BLOCK"


def test_report_is_video_from_frames_analyzed(pf, store):
    twerk = dict(garment="shorts", motion_flags="twerk_grind_striptease")
    still = json.loads(pf.PreFlightReport().report(obs_json(**twerk), "", False)[0])
    video = json.loads(pf.PreFlightReport().report(
        obs_json(**twerk, meta={"schema_version": "1", "frames_analyzed": 6}), "", False)[0])
    assert "mod.motion_twerk" not in still["fired_rules"] and "mod.motion_twerk" in video["fired_rules"]


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", None])
def test_report_of_unusable_json_is_unknown(pf, store, bad):
    report = json.loads(pf.PreFlightReport().report(bad, "", False)[0])
    assert report["unknown"] is True and report["fired_rules"] == ["guard.fail_closed"]


# --- Outcome -------------------------------------------------------------------------------------------------


def test_outcome_list_follows_the_store(pf, store):
    rids = [pf.PreFlightReport().report(obs_json(), "", True)[2] for _ in range(3)]
    labels = pf.PreFlightOutcome.INPUT_TYPES()["required"]["record"][0]
    assert [label.split(" ", 1)[0] for label in labels] == list(reversed(rids))


def test_outcome_logs_via_override(pf, store, bcnodes):
    rid = pf.PreFlightReport().report(obs_json(), "", True)[2]
    (status,) = pf.PreFlightOutcome().log(record="no records yet", platform="tiktok", result="removed",
                                          record_id_override=rid)
    assert status == "logged: %s tiktok=removed" % rid
    assert bcnodes["pipelines.preflight.feedback"].load_joined(store)[0]["outcomes"] == {"tiktok": "removed"}


def test_outcome_no_selection_and_bad_value(pf, store):
    (status,) = pf.PreFlightOutcome().log(record="no records yet", platform="instagram", result="clean",
                                          record_id_override="")
    assert "no record selected" in status
    (status,) = pf.PreFlightOutcome().log(record="badlabel", platform="myspace", result="clean", record_id_override="x")
    assert status.startswith("error:")


def test_outcome_validates_any_record_label(pf):
    assert pf.PreFlightOutcome.VALIDATE_INPUTS(record="ffff0000 · a label the list no longer shows") is True


def test_outcome_is_changed_always_differs(pf):
    assert pf.PreFlightOutcome.IS_CHANGED() != pf.PreFlightOutcome.IS_CHANGED()


# --- Calibrate -----------------------------------------------------------------------------------------------


def test_calibrate_reruns_only_when_the_store_changes(pf, store, bcnodes):
    calibrate = pf.PreFlightCalibrate
    assert calibrate.IS_CHANGED() == "" == calibrate.IS_CHANGED()
    rid = pf.PreFlightReport().report(obs_json(), "", True)[2]
    first = calibrate.IS_CHANGED()
    assert first != "" and calibrate.IS_CHANGED() == first
    pf.PreFlightOutcome().log("no records yet", "x", "clean", rid)
    assert calibrate.IS_CHANGED() != first
    (text,) = calibrate().calibrate()
    feedback = bcnodes["pipelines.preflight.feedback"]
    assert text == bcnodes["pipelines.preflight.calibrate"].build_report(feedback.load_joined(store))
    assert "predictions: 1   with at least one outcome: 1" in text


# --- the records route ---------------------------------------------------------------------------------------


def test_records_payload(pf, store):
    assert pf.records_payload() == {"records": ["no records yet"]}
    rid = pf.PreFlightReport().report(obs_json(), "", True)[2]
    payload = pf.records_payload()
    assert json.loads(json.dumps(payload)) == payload and list(payload) == ["records"]
    assert len(payload["records"]) == 1 and payload["records"][0].startswith(rid + " ")


def test_routes_need_a_server(pf, monkeypatch):
    monkeypatch.setitem(sys.modules, "server", None)  # `from server import ...` raises ImportError
    assert pf.register_routes() is None and pf.send_records() is None
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace()  # no instance: not a running server
    monkeypatch.setitem(sys.modules, "server", server)
    assert pf.register_routes() is None and pf.send_records() is None


def test_records_route(pf, store, monkeypatch):
    web = pytest.importorskip("aiohttp.web")
    routes = web.RouteTableDef()
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes,
                                                                              send_sync=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "server", server)
    pf.register_routes()
    [route] = list(routes)
    assert (route.method, route.path) == ("GET", "/bcnodes/preflight/records")
    pf.PreFlightReport().report(obs_json(), "", True)
    response = asyncio.run(route.handler(None))
    assert response.status == 200 and json.loads(response.body) == pf.records_payload()

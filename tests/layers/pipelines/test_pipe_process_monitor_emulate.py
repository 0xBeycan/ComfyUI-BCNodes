"""Emulate (pipelines/process_monitor/{emulate,profiles}.py): the tensor formulas written out by
hand, the propagation through the graph, the cache sum, the profiles of both packs' video nodes
(their outputs and transients worked out by hand from each node's code), "not counted" for a node
without a profile, passed-on outputs that add no bytes, bypassed / muted / subgraph handling,
weights from safetensors headers, the video table and the calibration from a measured run.
"""

import pytest
import torch

GIB = 1 << 30
LOADER_TYPES = ["IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO"]


@pytest.fixture
def em(bcnodes):
    return bcnodes["pipelines.process_monitor.emulate"]


@pytest.fixture
def pf(bcnodes):
    return bcnodes["pipelines.process_monitor.profiles"]


class Env:
    TYPES = {
        "VHS_LoadVideo": LOADER_TYPES, "ImageResizeKJv2": ["IMAGE", "INT", "INT", "MASK"], "BC_ImageResize": ["IMAGE", "INT", "INT", "MASK"],
        "DrawMaskOnImage": ["IMAGE"], "BC_DrawMaskOnImage": ["IMAGE"], "VHS_VideoCombine": ["VHS_FILENAMES"],
        "GetImageSize": ["INT", "INT", "INT"], "SomeCustomNode": ["IMAGE"], "BC_MaskGrow": ["MASK"],
        "BC_RepeatMaskBatch": ["MASK"], "VHS_DuplicateMasks": ["MASK", "INT"], "UNETLoader": ["MODEL"],
        "LoraLoaderModelOnly": ["MODEL"], "KSampler": ["LATENT"], "VAEDecode": ["IMAGE"], "VAELoader": ["VAE"],
        "WanAnimateToVideo": ["CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT"],
        "BCVLoadVideo": ["IMAGE", "AUDIO", "BCV_VIDEO_INFO"], "BCVLoadReferenceImage": ["IMAGE", "MASK"],
        "LoadImage": ["IMAGE", "MASK"], "BCVConformVideo": ["IMAGE"],
        "BCVWanAnimateLongVideoSampler": ["IMAGE", "INT", "STRING"], "BCVSCAIL2LongVideoSampler": ["IMAGE", "INT", "STRING"],
        "BCVWanAnimatePreprocess": ["IMAGE", "IMAGE", "MASK", "POSEDATA", "BBOX", "STRING", "BBOX", "MASK", "IMAGE"],
        "BCVSCAIL2Preprocess": ["IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "BOOLEAN"], "BCVSCAIL2ColoredMask": ["IMAGE", "IMAGE"],
        "BCVPoseDetection": ["IMAGE", "POSEDATA", "BBOX", "STRING"], "BCVSAM3VideoTrack": ["MASK"],
        "BCVFaceCrop": ["IMAGE", "BBOX"], "BCVMaskGuard": ["MASK", "STRING", "STRING", "IMAGE"],
        "BCVWanAnimatePreprocessGuard": ["MASK", "POSEDATA", "STRING", "STRING", "IMAGE"],
        "BCVSCAIL2PreprocessGuard": ["IMAGE", "IMAGE", "STRING", "STRING", "IMAGE"], "BCVPoseGuard": ["POSEDATA", "STRING", "STRING", "IMAGE"],
        "BCVGetVideoInfo": ["STRING", "STRING", "STRING", "FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "INT", "FLOAT", "INT", "INT",
                            "AUDIO"],
        "BCVSaveVideo": [], "PreviewImage": ["IMAGE"], "BCVWanAnimate2LongVideoSampler": ["IMAGE", "INT", "STRING"],
        "BC_SeedVR2Resize": ["IMAGE", "IMAGE"], "BC_SeedVR2VAEEncode": ["LATENT"], "BC_SeedVR2VAEDecode": ["IMAGE"],
        "BC_SeedVR2PostProcess": ["IMAGE"], "BC_SeedVR2PreprocessCompact": ["LATENT", "SEEDVR2_PLAN"],
        "BC_SeedVR2PostProcessCompact": ["IMAGE"], "BC_ImageScaleByAspectRatio": ["IMAGE", "MASK", "BOX", "INT", "INT"],
        "BC_BiRefNetRemoveBackground": ["IMAGE", "MASK", "IMAGE"], "BC_DepthAnythingV2": ["IMAGE"], "BC_PostFxApply": ["IMAGE"],
        "BC_AnySwitch": ["*"], "BC_SelectSwitch": ["*"],
        "BC_BlockifyMask": ["MASK"], "BC_MaskFillHoles": ["MASK"], "BlockifyMask": ["MASK"],
    }
    # BCVLoadVideo's bcv_sizes: the resolution "source" is the video's own size (None)
    SIZES = {"Wan": {"480p": [480, 832], "720p": [720, 1280], "source": None},
             "SCAIL": {"512p": [512, 896], "704p": [704, 1280], "source": None},
             "None": {"480p": [480, 854], "720p": [720, 1280], "1080p": [1080, 1920], "source": None}}

    def __init__(self, weights=None, videos=None, models=None):
        self.weights_fn, self.videos, self.models = weights, videos or {}, models or {}

    def return_types(self, class_type):
        return self.TYPES.get(class_type, [])

    def input_types(self, class_type):
        assert class_type == "BCVLoadVideo"
        return {"required": {"resolution": (["480p", "720p", "512p", "704p", "1080p", "source"], {"bcv_sizes": self.SIZES})}}

    def model_path(self, name):
        return self.models.get(name)

    def weights(self, path):
        return self.weights_fn(path)

    def video_meta(self, name):
        return self.videos.get(name)

    def image_size(self, name):
        return (640, 480) if name == "ref.png" else None


def N(class_type, title=None, **inputs):
    node = {"class_type": class_type, "inputs": inputs}
    if title:
        node["_meta"] = {"title": title}
    return node


CLIP = {"width": 1080, "height": 1920, "frames": 609, "fps": 30.0}


def old_chain():
    """The incident's chain: the clip loaded whole at 1080x1920, resized to 720x1280 through a listed
    sub-batch concat, saved frame by frame."""
    return {
        "1": N("VHS_LoadVideo", video="clip.mp4", force_rate=0, custom_width=0, custom_height=0, frame_load_cap=0,
               skip_first_frames=0, select_every_nth=1),
        "2": N("ImageResizeKJv2", image=["1", 0], width=720, height=1280, upscale_method="lanczos", keep_proportion="crop",
               pad_color="0, 0, 0", crop_position="center", divisible_by=16),
        "3": N("VHS_VideoCombine", images=["2", 0], frame_rate=30, filename_prefix="out"),
    }


def test_formulas(pf):
    assert pf.image_bytes(609, 1280, 720) == 609 * 1280 * 720 * 3 * 4 == 6_735_052_800
    assert pf.image_bytes(609, 1280, 720, dtype_bytes=2) == 609 * 1280 * 720 * 3 * 2
    assert pf.mask_bytes(609, 1280, 720) == 609 * 1280 * 720 * 4
    assert [pf.wan_latent_frames(n) for n in (1, 5, 77, 81)] == [1, 2, 20, 21]
    assert pf.wan_latent_bytes(81, 1280, 720) == 21 * 16 * 160 * 90 * 4
    assert pf.wan_latent_bytes(81, 1280, 720, dtype_bytes=2) == 21 * 16 * 160 * 90 * 2


def test_old_chain_peaks_at_the_listed_resize(em):
    env = Env(videos={"clip.mp4": CLIP})
    r = em.estimate(old_chain(), env)
    rows = {row["id"]: row for row in r["rows"]}
    full, small = 609 * 1920 * 1080 * 3 * 4, 609 * 1280 * 720 * 3 * 4
    assert rows["1"]["outputs"][0]["shape"] == [609, 1920, 1080, 3] and rows["1"]["output_bytes"] == full
    assert rows["1"]["transient"] == full // 2  # np.fromiter growth bound
    assert rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3]
    assert rows["2"]["transient"] == small  # sub-batch list + cat
    assert rows["2"]["ram_at_node"] == full + small + small
    assert rows["3"]["status"] == "counted" and rows["3"]["output_bytes"] == 0
    assert (r["ram_peak"], r["ram_peak_node"], r["cache_total"]) == (full + 2 * small, "2", full + small)
    fits = em.fit(r, int(29.6 * GIB), ram_now=4 * GIB)
    assert all(not f["ram_fits"] for f in fits) and fits[0]["ram_need"] == 4 * GIB + full + 2 * small


def test_the_port_has_no_transient(em):
    p = old_chain()
    p["2"]["class_type"] = "BC_ImageResize"
    rows = {row["id"]: row for row in em.estimate(p, Env(videos={"clip.mp4": CLIP}))["rows"]}
    assert rows["2"]["transient"] == 0 and rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3]


def test_loader_frame_and_size_widgets(em):
    p = old_chain()
    p["1"]["inputs"].update(force_rate=15, frame_load_cap=100, custom_width=540, custom_height=0)
    row = em.estimate(p, Env(videos={"clip.mp4": CLIP}))["rows"][0]
    assert row["outputs"][0]["shape"] == [100, 960, 544, 3]  # 609 / 30 * 15 = 304 -> cap 100; 540 x 960 rounded to /8


def test_unknown_node_is_not_counted_and_so_is_what_depends_on_it(em):
    p = {"1": N("SomeCustomNode"), "2": N("BC_MaskGrow", mask=["1", 0]), "3": N("GetImageSize", image=["1", 0])}
    r = em.estimate(p, Env())
    status = {row["id"]: (row["status"], row["note"]) for row in r["rows"]}
    assert status["1"] == ("not counted", "no cost profile for SomeCustomNode (outputs IMAGE)")
    assert status["2"][0] == "not counted" and "'mask' has no estimate" in status["2"][1]
    assert status["3"][0] == "counted"  # numbers only
    assert r["not_counted"] == ["1", "2"] and r["ram_peak"] == 0


def test_masks_and_the_cloning_draw(em):
    p = old_chain()
    del p["3"]
    p["4"] = N("VHS_DuplicateMasks", mask=["2", 3], multiply_by=3)
    p["2"]["inputs"]["mask"] = ["9", 1]
    p["9"] = N("LoadImage", image="ref.png")
    p["5"] = N("DrawMaskOnImage", image=["2", 0], mask=["4", 0], color="0, 0, 0")
    r = em.estimate(p, Env(videos={"clip.mp4": CLIP}))
    rows = {row["id"]: row for row in r["rows"]}
    assert rows["9"]["outputs"][1]["shape"] == [1, 480, 640]
    assert rows["2"]["outputs"][1]["shape"] == [1, 1280, 720]  # the mask follows the image's size
    assert rows["4"]["outputs"][0]["shape"] == [3, 1280, 720]
    image, mask = 609 * 1280 * 720 * 3 * 4, 3 * 1280 * 720 * 4
    assert rows["5"]["transient"] == 2 * image + mask  # clones of image and mask + the frame list


def test_mask_nodes_read_their_masks_input(em):
    # BC_BlockifyMask, BC_MaskFillHoles and KJ's BlockifyMask take their mask as `masks`
    p = {"1": N("VHS_LoadVideo", video="clip.mp4", force_rate=0, custom_width=0, custom_height=0, frame_load_cap=9,
                skip_first_frames=0, select_every_nth=1),
         "2": N("BCVSAM3VideoTrack", images=["1", 0]), "3": N("BC_BlockifyMask", masks=["2", 0], block_size=32),
         "4": N("BC_MaskFillHoles", masks=["3", 0]), "5": N("BlockifyMask", masks=["2", 0], block_size=32)}
    r = em.estimate(p, Env(videos={"clip.mp4": CLIP}))
    rows = {row["id"]: row for row in r["rows"]}
    mask = 9 * 1920 * 1080 * 4
    assert r["not_counted"] == []
    assert [rows[k]["outputs"][0]["shape"] for k in "345"] == [[9, 1920, 1080]] * 3
    assert (rows["3"]["transient"], rows["4"]["transient"], rows["5"]["transient"]) == (0, 0, mask)


def test_bypassed_muted_and_subgraph_nodes(em):
    workflow = {"nodes": [{"id": 1, "type": "VHS_LoadVideo", "mode": 0}, {"id": 7, "type": "BC_MaskGrow", "mode": 4},
                          {"id": 8, "type": "PreviewImage", "mode": 2, "title": "check"}],
                "definitions": {"subgraphs": [{"id": "sg-1", "name": "inner", "nodes": [{"id": 3, "type": "BC_MaskGrow", "mode": 4}]}]}}
    assert em.skipped_nodes(workflow) == [
        {"id": "7", "class_type": "BC_MaskGrow", "title": "BC_MaskGrow", "status": "bypassed"},
        {"id": "8", "class_type": "PreviewImage", "title": "check", "status": "muted"},
        {"id": "3", "class_type": "BC_MaskGrow", "title": "inner: BC_MaskGrow", "status": "bypassed"},
    ]
    # an expanded subgraph arrives as "outer:inner" ids, linked like any other node
    p = {"12:1": N("VHS_LoadVideo", video="clip.mp4"), "12:2": N("BC_ImageResize", image=["12:1", 0], width=720, height=1280,
                                                               keep_proportion="stretch", divisible_by=0)}
    out = em.emulate(p, workflow, Env(videos={"clip.mp4": CLIP}), ram_limit=None, ram_now=0)
    assert [row["id"] for row in out["current"]["rows"]] == ["12:1", "12:2"]
    assert len(out["skipped"]) == 3 and out["label"].startswith("rough estimate")


def test_weights_and_vram_need(em, bcnodes, tmp_path):
    from safetensors.torch import save_file

    unet, lora = tmp_path / "unet.safetensors", tmp_path / "lora.safetensors"
    save_file({"w": torch.zeros(1000, 250, dtype=torch.float16)}, str(unet))
    save_file({"a": torch.zeros(100, dtype=torch.float32)}, str(lora))
    env = Env(weights=bcnodes["libs.safetensors_info"].weights_info,
              models={"unet.safetensors": str(unet), "lora.safetensors": str(lora)})
    p = {
        "1": N("UNETLoader", unet_name="unet.safetensors", weight_dtype="default"),
        "2": N("LoraLoaderModelOnly", model=["1", 0], lora_name="lora.safetensors", strength_model=1.0),
        "3": N("WanAnimateToVideo", width=720, height=1280, length=81, batch_size=1),
        "4": N("KSampler", model=["2", 0], latent_image=["3", 2], seed=0),
        "5": N("VAELoader", vae_name="missing.safetensors"),
        "6": N("VAEDecode", samples=["4", 0], vae=["5", 0]),
    }
    r = em.estimate(p, env)
    rows = {row["id"]: row for row in r["rows"]}
    assert rows["1"]["weights"][0]["bytes"] == 500_000 and rows["1"]["vram_need"] == 0
    assert rows["4"]["vram_need"] == 500_000 + 400 and (r["vram_peak"], r["vram_peak_node"]) == (500_400, "4")
    assert rows["5"]["weights"] == [] and rows["6"]["vram_need"] == 0  # a file that is not found adds nothing
    assert rows["3"]["outputs"] == [{"slot": 2, "type": "LATENT", "shape": [1, 16, 21, 160, 90], "bytes": 21 * 16 * 160 * 90 * 4,
                                     "shared": False}]
    assert rows["4"]["outputs"][0]["shape"] == [1, 16, 21, 160, 90]
    assert rows["6"]["outputs"][0]["shape"] == [81, 1280, 720, 3] and rows["6"]["transient"] is None
    assert r["weights_total"] == 500_400


def test_bcv_loader_reference_and_conform(em):
    env = Env(videos={"v.mp4": {"width": 1920, "height": 1080, "frames": 300, "fps": 30.0}})
    p = {"1": N("BCVLoadVideo", video="v.mp4", model="Wan", resolution="720p", orientation="auto", force_fps="", start_frame=1,
                frame_count=""),
         "2": N("BCVLoadReferenceImage", image="ref.png", video_info=["1", 2]),
         "3": N("BCVConformVideo", images=["1", 0], fit="crop", method="lanczos")}
    rows = {row["id"]: row for row in em.estimate(p, env)["rows"]}
    assert rows["1"]["outputs"][0]["shape"] == [297, 720, 1280, 3]  # landscape source; 300 frames -> 4n+1
    # precision left out (a workflow saved before the widget): its default, fp16
    assert rows["1"]["output_bytes"] == 297 * 720 * 1280 * 3 * 2 and rows["1"]["transient"] == 0
    assert rows["2"]["outputs"][0]["shape"] == [1, 720, 1280, 3]
    assert rows["2"]["output_bytes"] == 720 * 1280 * 3 * 4 + 720 * 1280 * 4  # the reference is float32
    assert rows["3"]["outputs"][0]["shape"] == [297, 720, 1280, 3]
    assert rows["3"]["outputs"][0]["shared"] and rows["3"]["output_bytes"] == 0  # already 720p: passed on untouched
    p["1"]["inputs"]["precision"] = "fp32"
    assert em.estimate(p, env)["rows"][0]["output_bytes"] == 297 * 720 * 1280 * 3 * 4
    del p["1"]["inputs"]["precision"]
    # model None: every frame; resolution source: the video's size, in the other orientation its centred crop
    # to that aspect with the short side kept (1920x1080 -> 608x1080), each side cut to the model's grid
    for model, resolution, orientation, shape in [
            ("None", "720p", "auto", [300, 720, 1280, 3]), ("None", "1080p", "portrait", [300, 1920, 1080, 3]),
            ("Wan", "source", "auto", [297, 1072, 1920, 3]), ("Wan", "source", "portrait", [297, 1072, 608, 3]),
            ("SCAIL", "source", "landscape", [297, 1056, 1920, 3]), ("None", "source", "portrait", [300, 1080, 608, 3]),
            ("None", "source", "landscape", [300, 1080, 1920, 3])]:
        p["1"]["inputs"].update(model=model, resolution=resolution, orientation=orientation)
        assert em.estimate(p, env)["rows"][0]["outputs"][0]["shape"] == shape, (model, resolution, orientation)


def test_bcv_load_video_images_not_linked_and_get_video_info(em):
    # only video_info linked: the clip is loaded all the same (video_info counts its frames) and dropped at return
    env = Env(videos={"v.mp4": {"width": 1920, "height": 1080, "frames": 300, "fps": 30.0}})
    p = {"1": N("BCVLoadVideo", video="v.mp4", model="Wan", resolution="720p", orientation="auto", force_fps="", start_frame=1,
                frame_count="", precision="fp16"),
         "2": N("BCVGetVideoInfo", video_info=["1", 2])}
    rows = {row["id"]: row for row in em.estimate(p, env)["rows"]}
    assert [o["type"] for o in rows["1"]["outputs"]] == ["BCV_VIDEO_INFO"] and rows["1"]["output_bytes"] == 0
    assert rows["1"]["transient"] == 297 * 720 * 1280 * 3 * 2 and rows["1"]["note"].startswith("images not linked")
    # Get Video Info: numbers, and Load Video's audio output itself
    assert rows["2"]["status"] == "counted" and rows["2"]["outputs"] == [] and rows["2"]["output_bytes"] == 0
    # the scenario table takes no frame count from a loader without its frames
    out = em.emulate(p, {}, env, ram_limit=None, ram_now=0)
    assert out["table"]["frames"] == list(em.FRAME_COUNTS)


def test_video_table_and_calibration(em):
    env = Env(videos={"clip.mp4": CLIP})
    out = em.emulate(old_chain(), {}, env, ram_limit=32 * GIB, ram_now=GIB)
    assert out["table"]["frames"] == [81, 161, 321, 609, 641]
    row720 = [row for row in out["table"]["rows"] if row["label"] == "720p"][0]
    small = 81 * 1280 * 720 * 3 * 4
    # 720p scenario: the loader already gives 720x1280, the resize keeps it
    assert row720["cells"][0]["ram_peak"] == max(small + small // 2, small + small + small)
    # an armed run at 81 frames: the measured nodes replace the formulas, scaled by pixel count
    measured = {"nodes": {
        "1": {"state": "executed", "outputs": [{"shape": [81, 1280, 720, 3]}], "output_bytes": small, "ram_start": GIB,
              "ram_peak": GIB + small + 100},
        "2": {"state": "executed", "outputs": [{"shape": [81, 1280, 720, 3]}], "output_bytes": small, "ram_start": GIB,
              "ram_peak": GIB + 2 * small}}}
    r = em.estimate(old_chain(), env, {"frames": 161, "w": 720, "h": 1280}, measured)
    rows = {row["id"]: row for row in r["rows"]}
    ratio = 161 / 81
    assert rows["1"]["status"] == "measured" and rows["1"]["output_bytes"] == int(small * ratio)
    assert rows["1"]["transient"] == int(100 * ratio) and rows["2"]["transient"] == int(small * ratio)
    out = em.emulate(old_chain(), {}, env, ram_limit=None, ram_now=0, measured=measured, node_cost_s=0.0005)
    assert out["calibrated"] and out["measure_cost_s"] == pytest.approx(0.0015)


def test_a_loader_measured_without_frames_scales_nothing(em):
    # an armed run whose loader's images were not linked: measured with 0 frames, so there is no pixel count to
    # scale the scenarios by; they keep their formulas instead of dividing by zero
    env = Env(videos={"clip.mp4": CLIP})
    measured = {"nodes": {"1": {"state": "executed", "outputs": [{"shape": [0, 1280, 720, 3]}], "output_bytes": 0,
                                "ram_start": GIB, "ram_peak": GIB}}}
    r = em.estimate(old_chain(), env, {"frames": 161, "w": 720, "h": 1280}, measured)
    assert all(row["status"] != "measured" for row in r["rows"])
    assert r == em.estimate(old_chain(), env, {"frames": 161, "w": 720, "h": 1280})


# -- the profiles of the video nodes, each worked out by hand from the node's code ----------------

PORTRAIT = {"width": 1080, "height": 1920, "frames": 609, "fps": 30.0}


def rows_of(em, prompt, env=None):
    return {row["id"]: row for row in em.estimate(prompt, env or Env(videos={"v.mp4": PORTRAIT}))["rows"]}


def loader(frame_count="", precision=None):
    """Load Video at 720p: 609 portrait frames of 720x1280; precision left out is its default, fp16."""
    node = N("BCVLoadVideo", video="v.mp4", model="Wan", resolution="720p", orientation="auto", force_fps="",
             start_frame=1, frame_count=frame_count)
    if precision:
        node["inputs"]["precision"] = precision
    return node


FRAME, MASK = 1280 * 720 * 3 * 4, 1280 * 720 * 4  # a 720x1280 frame and mask in float32
FACE = 512 * 512 * 3 * 4


def test_long_video_samplers(em):
    # Wan Animate on Load Video's half clip: pose images half, face crops float32; output linked
    p = {"1": loader(), "2": N("BCVPoseDetection", images=["1", 0]), "3": N("BCVFaceCrop", images=["1", 0]),
         "4": N("BCVWanAnimateLongVideoSampler", pose_video=["2", 0], face_video=["3", 0], width=720, height=1280,
                frames_per_chunk=81, total_frames=0),
         "5": N("PreviewImage", images=["4", 0])}
    rows = rows_of(em, p)
    assert rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3] and rows["2"]["output_bytes"] == 609 * FRAME // 2
    assert rows["3"]["outputs"][0]["shape"] == [609, 512, 512, 3] and rows["3"]["output_bytes"] == 609 * FACE
    sampler = rows["4"]
    # 81 + 7 x 76 = 613 >= 609: the last chunk fitted to the 72 frames left plus the 5 it keeps
    assert "chunk plan 81 + 81 + 81 + 81 + 81 + 81 + 81 + 77" in sampler["note"]
    assert sampler["outputs"][0]["shape"] == [609, 1280, 720, 3] and sampler["output_bytes"] == 609 * FRAME // 2
    # the peak is chunk 2: its pose window spans chunks 1 and 2 (162 frames, requantized to float32; the
    # face crops are float32 and inside their video: a view), its 81 decoded frames, and chunk 1's 81
    # decoded frames kept as its seed (a half output keeps the seed apart, as decoded)
    assert sampler["transient"] == 162 * FRAME + 81 * FRAME + 81 * FRAME
    # one chunk (fit to 53 frames): its pose window requantized, its decoded frames, apart from the half output
    p["4"]["inputs"].update(total_frames=50)
    assert rows_of(em, p)["4"]["transient"] == 53 * FRAME + 53 * FRAME
    # float32: no chunk reads past the videos, so no window; the seed is a view of the output
    p["1"] = loader(precision="fp32")
    p["4"]["inputs"].update(total_frames=0)
    sampler = rows_of(em, p)["4"]
    assert sampler["transient"] == 81 * FRAME and sampler["output_bytes"] == 609 * FRAME
    # one chunk at float32: the output is a view of its 53 decoded frames, the 3 past total add
    p["4"]["inputs"].update(total_frames=50)
    assert rows_of(em, p)["4"]["transient"] == 3 * FRAME
    del p["5"]  # images not linked: generated all the same and dropped at return
    sampler = rows_of(em, p)["4"]
    assert sampler["outputs"] == [] and sampler["transient"] == 53 * FRAME
    # Wan Animate 2 keeps one frame of the chunk before: 81 + 6 x 80 = 561, then 48 + 1 -> 49
    p = {"1": loader(), "4": N("BCVWanAnimate2LongVideoSampler", pose_video=["1", 0], width=720, height=1280,
                               frames_per_chunk=81, total_frames=0), "5": N("PreviewImage", images=["4", 0])}
    assert "chunk plan 81 + 81 + 81 + 81 + 81 + 81 + 81 + 49:" in rows_of(em, p)["4"]["note"]


def test_scail2_sampler(em):
    # SCAIL-2 holds pose_video and pose_video_mask, both half here; every chunk full length (its default);
    # the size is cropped to the VAE's /8
    p = {"1": loader(), "2": N("BCVSCAIL2ColoredMask", driving_mask=["3", 0], replacement_mode=False),
         "3": N("BCVSAM3VideoTrack", images=["1", 0]),
         "4": N("BCVSCAIL2LongVideoSampler", pose_video=["1", 0], pose_video_mask=["2", 0], width=706, height=1284,
                frames_per_chunk=83, total_frames=0),
         "5": N("PreviewImage", images=["4", 0])}
    rows = rows_of(em, p)
    assert rows["3"]["outputs"][0]["shape"] == [609, 1280, 720] and rows["3"]["output_bytes"] == 609 * MASK // 2
    assert rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3] and rows["2"]["outputs"][1]["shape"] == [1, 1280, 720, 3]
    sampler = rows["4"]
    assert "chunk plan 81 + 81 + 81 + 81 + 81 + 81 + 81 + 81:" in sampler["note"]  # 81 + 7 x 76 = 613
    assert sampler["outputs"][0]["shape"] == [609, 1280, 704, 3] and sampler["output_bytes"] == 609 * 1280 * 704 * 3 * 2
    # chunk 2: both windows span chunks 1 and 2 (162 frames each, requantized to float32), its decoded frames and
    # chunk 1's as its seed. The last chunk reads past the videos (613 > 609): gathered windows of 81 frames,
    # less than chunk 2
    decoded = 1280 * 704 * 3 * 4
    assert sampler["transient"] == 2 * 162 * FRAME + 81 * decoded + 81 * decoded


def test_wan_animate_sampler_paints_the_background(em):
    # replacement mode on Load Video's half frames: the background the clip itself, the character mask
    # WanAnimate Preprocess's final mask (half); 161 frames: chunks 81 + 81 + 9
    p = {"1": loader(frame_count="161"),
         "2": N("BCVWanAnimatePreprocess", images=["1", 0], mode="prompt"),
         "3": N("BCVWanAnimateLongVideoSampler", pose_video=["2", 0], face_video=["2", 1], background_video=["1", 0],
                character_mask=["2", 7], width=720, height=1280, frames_per_chunk=81, total_frames=0),
         "4": N("PreviewImage", images=["3", 0])}
    rows = rows_of(em, p)
    assert [o["slot"] for o in rows["2"]["outputs"]] == [0, 1, 7]
    sampler = rows["3"]
    assert "chunk plan 81 + 81 + 9:" in sampler["note"]
    # chunk 2 reads frames 0..161 (its seed, the whole chunk 1, and its own, cut at the plan's reach): the pose and
    # background windows requantized to float32, the character mask's too, the painted half background window
    # (referenced until the next chunk cuts its windows), its decoded frames and chunk 1's as its seed
    assert sampler["transient"] == 161 * FRAME + 161 * FRAME + 161 * MASK + 161 * FRAME // 2 + 81 * FRAME + 81 * FRAME
    # float32: the background painted into a new window every chunk; no requantized copy, no seed apart
    p["1"] = loader(frame_count="161", precision="fp32")
    assert rows_of(em, p)["3"]["transient"] == 161 * FRAME + 81 * FRAME
    # the background alone: nothing painted, a float32 background inside its video is read as a view
    del p["3"]["inputs"]["character_mask"]
    assert rows_of(em, p)["3"]["transient"] == 81 * FRAME


def test_preprocess_wrappers_and_guards(em):
    p = {"1": loader(), "9": N("LoadImage", image="ref.png"),
         "2": N("BCVWanAnimatePreprocess", images=["1", 0], mode="prompt"),
         "3": N("BCVWanAnimatePreprocessGuard", mask=["2", 2], pose_data=["2", 3]),
         "4": N("BCVSCAIL2Preprocess", images=["1", 0], reference_image=["9", 0], replacement_mode=False, mode="box_keypoint",
                black_background=False),
         "5": N("BCVSCAIL2PreprocessGuard", pose_video_mask=["4", 1], reference_image_mask=["4", 2]),
         "6": N("BCVMaskGuard", mask=["4", 3]), "7": N("BCVPoseGuard", pose_data=["2", 3]),
         "8": N("PreviewImage", images=["4", 0]), "10": N("BCVSaveVideo", images=["4", 0])}
    rows = rows_of(em, p)
    # only the raw mask is linked: no pose image drawn, no face cut; the half clip read two float32 frames at a time
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[609, 1280, 720]]
    assert rows["2"]["output_bytes"] == 609 * MASK // 2 and rows["2"]["transient"] == 2 * FRAME
    guard = rows["3"]
    assert guard["output_bytes"] == (128 + 190 * 4) * 1200 * 3 * 4  # the mask is passed on; the timeline is new
    assert guard["transient"] == 5 * 1280 * 720 and guard["outputs"][0]["shared"]
    scail = rows["4"]
    shapes = [o["shape"] for o in scail["outputs"]]
    assert shapes == [[609, 1280, 720, 3], [609, 1280, 720, 3], [1, 480, 640, 3], [609, 1280, 720], [1, 480, 640]]
    assert scail["outputs"][0]["shared"]  # black_background off: the driving video itself
    # the driving masks in the clip's half dtype, the reference ones float32 (the reference image is)
    assert scail["output_bytes"] == 609 * FRAME // 2 + 480 * 640 * 3 * 4 + 609 * MASK // 2 + 480 * 640 * 4
    # box_keypoint runs Pose Detection (two float32 frames; no pose image drawn); the driving mask is colored
    # 16 frames at a time: a float32 copy of the half frames and the boolean cut
    assert scail["transient"] == 16 * 1280 * 720 * (4 + 1)
    assert rows["5"]["transient"] == 609 * 1280 * 720 and rows["5"]["output_bytes"] == (128 + 190 * 2) * 1200 * 3 * 4
    assert rows["6"]["output_bytes"] == (128 + 190 * 3) * 1200 * 3 * 4
    assert rows["7"]["output_bytes"] == (128 + 190 * 2) * 1200 * 3 * 4
    assert rows["8"]["output_bytes"] == 0 and rows["10"]["outputs"] == []
    p["4"]["inputs"].update(mode="prompt", black_background=True)
    scail = rows_of(em, p)["4"]
    assert scail["transient"] == 16 * 1280 * 720 * (4 + 1) and not scail["outputs"][0]["shared"]
    assert scail["outputs"][0]["bytes"] == 609 * FRAME // 2  # the driving video on black, in its dtype
    # the masks not linked: pose_video_mask not rendered, the mask dropped at return (alive until then)
    for node in ("5", "6", "8", "10"):
        del p[node]
    scail = rows_of(em, p)["4"]
    assert [o["slot"] for o in scail["outputs"]] == [2, 4]
    assert scail["transient"] == 609 * MASK // 2 + 480 * 640  # the reference mask's boolean cut, one frame


def test_pose_detection_size_and_model(em):
    p = {"1": loader(), "2": N("BCVPoseDetection", images=["1", 0], width=832, height=480), "3": N("PreviewImage", images=["2", 0])}
    row = rows_of(em, p)["2"]
    # drawn at width x height in the clip's half dtype; the half frames read two float32 frames at a time
    assert row["outputs"][0]["shape"] == [609, 480, 832, 3] and row["output_bytes"] == 609 * 480 * 832 * 3 * 2
    assert row["transient"] == 2 * FRAME
    assert "(ViTPose-H) weights and working set not counted" in row["note"]
    p["2"]["inputs"]["pose_model"] = "Sapiens2 1b bf16"  # the same outputs; the model is named in the note
    row = rows_of(em, p)["2"]
    assert row["outputs"][0]["shape"] == [609, 480, 832, 3] and "(Sapiens2 1b bf16)" in row["note"]
    p["1"] = loader(precision="fp32")  # a float32 clip is read as a view
    row = rows_of(em, p)["2"]
    assert row["transient"] == 0 and row["output_bytes"] == 609 * 480 * 832 * 3 * 4
    p["2"]["inputs"].update(width=["9", 0], height=["9", 1])  # sockets fed by links: known at run time only
    row = rows_of(em, p)["2"]
    assert row["status"] == "not counted" and "width / height come from links: the pose images'" in row["note"]
    del p["3"]  # pose images not linked: not drawn, so their size does not matter
    row = rows_of(em, p)["2"]
    assert row["status"] == "counted" and row["outputs"] == []
    p["3"] = N("PreviewImage", images=["2", 0])
    p["2"]["inputs"] = {"images": ["1", 0], "width": 832}
    row = rows_of(em, p)["2"]
    assert row["status"] == "not counted" and "only one of width / height is set" in row["note"]


def test_wan_animate_preprocess_outputs_only_when_linked(em):
    consumers = {0: N("PreviewImage", images=["2", 0]), 1: N("PreviewImage", images=["2", 1]),
                 2: N("BCVMaskGuard", mask=["2", 2]), 7: N("BCVWanAnimatePreprocessGuard", mask=["2", 7], pose_data=["2", 3]),
                 8: N("PreviewImage", images=["2", 8])}
    for precision, two in (("fp16", 2), ("fp32", 4)):
        base = {"1": loader(precision=precision), "2": N("BCVWanAnimatePreprocess", images=["1", 0], mode="prompt")}
        mask, sizes = 609 * 1280 * 720 * two, {0: 609 * FRAME * two // 4, 1: 609 * FACE, 2: 0, 7: 0, 8: 609 * FRAME * two // 4}
        sizes[2] = sizes[7] = mask
        reading = 2 * FRAME if two == 2 else 0  # Pose Detection's two float32 frames on a half clip
        # linked slots, then the transient: the raw mask made for the final mask and freed once it is made;
        # the final mask made for bg_images and kept until they are painted
        for linked, transient in [((), reading), ((7,), mask), ((8,), 2 * mask), ((7, 8), mask), ((2, 7, 8), reading),
                                  ((0, 1, 2), reading), ((2, 8), mask)]:
            prompt = {**base, **{str(10 + slot): consumers[slot] for slot in linked}}
            row = rows_of(em, prompt)["2"]
            assert [o["slot"] for o in row["outputs"]] == list(linked), (precision, linked)
            assert row["output_bytes"] == sum(sizes[slot] for slot in linked), (precision, linked)
            assert row["transient"] == max(reading, transient), (precision, linked)


def test_scail2_colored_mask(em):
    p = {"1": loader(), "3": N("BCVSAM3VideoTrack", images=["1", 0]),
         "2": N("BCVSCAIL2ColoredMask", driving_mask=["3", 0], replacement_mode=False), "4": N("PreviewImage", images=["2", 0])}
    row = rows_of(em, p)["2"]
    # the driving mask colored in its half dtype, 16 frames at a time (a float32 copy and the boolean cut); without a
    # reference mask the reference is one float32 frame of zeros
    assert row["output_bytes"] == 609 * FRAME // 2 + FRAME and row["transient"] == 16 * 1280 * 720 * (4 + 1)
    del p["4"]  # pose_video_mask not linked: not rendered; the zeros and their cut
    row = rows_of(em, p)["2"]
    assert [o["slot"] for o in row["outputs"]] == [1] and row["transient"] == MASK + 1280 * 720
    p["9"] = N("LoadImage", image="ref.png")
    p["2"]["inputs"]["reference_mask"] = ["9", 1]
    row = rows_of(em, p)["2"]
    assert row["outputs"][0]["shape"] == [1, 480, 640, 3] and row["transient"] == 480 * 640


def test_seedvr2_chain(em):
    p = {"1": loader(frame_count="81"),
         "2": N("BC_SeedVR2Resize", image=["1", 0], upscale_factor=1.5, downscale_factor=1.0, max_resolution=4096, emulate_bf16=True),
         "3": N("BC_SeedVR2VAEEncode", pixels=["2", 0], tile_size=1024, overlap=256),
         "4": N("BC_SeedVR2VAEDecode", samples=["3", 0], tile_size=1024, overlap=256),
         "5": N("BC_SeedVR2PostProcess", images=["4", 0], original_resized_images=["2", 1], color_correction_method="lab")}
    rows = rows_of(em, p)
    # short edge 720 x 1.5 = 1080, the long edge floored: 1920; padded to /16: 1088; 81 frames are 4n+1
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[81, 1920, 1088, 3], [81, 1920, 1080, 3]]
    assert rows["2"]["output_bytes"] == 81 * 1920 * (1088 + 1080) * 3 * 2 and rows["2"]["transient"] == 0
    elements = 16 * 21 * 240 * 136
    assert rows["3"]["outputs"][0]["shape"] == [1, 16, 21, 240, 136] and rows["3"]["output_bytes"] == elements * 4
    # 1920 > tile 1024: the tile sum is the output, rounded through float16 four latent frames at a time
    assert rows["3"]["transient"] == 16 * 4 * 240 * 136 * 2
    assert rows["4"]["outputs"][0]["shape"] == [81, 1920, 1088, 3] and rows["4"]["transient"] == 0  # tiles summed in the output
    assert rows["5"]["outputs"][0]["shape"] == [81, 1920, 1080, 3] and rows["5"]["output_bytes"] == 81 * 1920 * 1080 * 3 * 2
    p["2"]["inputs"]["downscale_factor"] = 0.5  # resolution comes from the original: the same size, downscaled four frames at a time
    resize = rows_of(em, p)["2"]
    assert resize["outputs"][0]["shape"] == [81, 1920, 1088, 3] and resize["transient"] == 0
    p["3"]["inputs"]["tile_size"] = 2048  # one tile: the latent slices go straight into the output
    assert rows_of(em, p)["3"]["transient"] == 0


def test_seedvr2_compact_chain(em):
    p = {"1": loader(frame_count="77"),
         "2": N("BC_SeedVR2PreprocessCompact", image=["1", 0], upscale_factor=1.5, downscale_factor=0.5, max_resolution=4096,
                emulate_bf16=True, tile_size=1024, overlap=256),
         "3": N("BC_SeedVR2PostProcessCompact", samples=["2", 0], plan=["2", 1], image=["1", 0], color_correction_method="lab",
                tile_size=1024, overlap=256)}
    rows = rows_of(em, p)
    # 77 frames of 1280x720: resized 1920x1080, padded 1920x1088; the latent of the padded clip
    elements = 16 * 20 * 240 * 136
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[1, 16, 20, 240, 136], []]
    assert rows["2"]["output_bytes"] == elements * 4  # the plan is a few numbers
    # the padded float16 clip while encoded, the tile sum's cast of four latent frames
    assert rows["2"]["transient"] == 77 * 1920 * 1088 * 3 * 2 + 16 * 4 * 240 * 136 * 2
    # 20 latent frames decode to 77, cut to the plan's 77 frames of 1920x1080, float16
    assert rows["3"]["outputs"][0]["shape"] == [77, 1920, 1080, 3] and rows["3"]["output_bytes"] == 77 * 1920 * 1080 * 3 * 2
    assert rows["3"]["transient"] == 0
    today = {"1": p["1"],
             "2": N("BC_SeedVR2Resize", image=["1", 0], upscale_factor=1.5, downscale_factor=0.5, max_resolution=4096, emulate_bf16=True),
             "3": N("BC_SeedVR2VAEEncode", pixels=["2", 0], tile_size=1024, overlap=256),
             "4": N("BC_SeedVR2VAEDecode", samples=["3", 0], tile_size=1024, overlap=256),
             "5": N("BC_SeedVR2PostProcess", images=["4", 0], original_resized_images=["2", 1], color_correction_method="lab")}
    old, new = em.estimate(today, Env(videos={"v.mp4": PORTRAIT})), em.estimate(p, Env(videos={"v.mp4": PORTRAIT}))
    # what the output cache keeps less: Resize's two clips and Decode's
    assert old["cache_total"] - new["cache_total"] == 77 * 1920 * (1088 + 1080 + 1088) * 3 * 2


def test_bcnodes_image_nodes_and_switches(em):
    one = 480 * 640 * 4
    p = {"9": N("LoadImage", image="ref.png"),
         "1": N("BC_ImageScaleByAspectRatio", image=["9", 0], aspect_ratio="16:9", proportional_width=1, proportional_height=1,
                fit="crop", method="lanczos", round_to_multiple="16", scale_to_side="longest", scale_to_length=1024),
         "2": N("BC_BiRefNetRemoveBackground", image=["9", 0], model="BiRefNet-general"),
         "3": N("BC_DepthAnythingV2", image=["9", 0], resolution=518),
         "4": N("BC_PostFxApply", image=["9", 0], theme="none"),
         "5": N("BC_AnySwitch", any_01=["8", 0], any_02=["9", 0]),
         "6": N("BC_SelectSwitch", selected="b", a=["8", 0], b=["2", 2])}
    rows = rows_of(em, p)
    assert [o["shape"] for o in rows["1"]["outputs"]] == [[1, 576, 1024, 3], [1, 576, 1024]]
    assert rows["1"]["transient"] == 0  # one frame at a time into preallocated outputs
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[1, 480, 640, 4], [1, 480, 640], [1, 480, 640, 3]]
    assert rows["2"]["transient"] == one  # one frame at a time: its raw matte
    # short side 518, 640 * 518 / 480 = 690.67 -> 691; one frame at a time into the preallocated output
    assert rows["3"]["outputs"][0]["shape"] == [1, 518, 691, 3] and rows["3"]["transient"] == 0
    assert rows["4"]["outputs"][0]["shared"] and rows["4"]["output_bytes"] == 0
    assert rows["5"]["outputs"][0]["shared"] and rows["5"]["note"] == "passes on any_02"  # node 8 does not exist
    assert rows["6"]["outputs"][0]["shape"] == [1, 480, 640, 3] and rows["6"]["output_bytes"] == 0
    p["2"]["inputs"].update(background="Color", mask_blur=2)
    p["3"]["inputs"].update(width=832, height=480)
    p["4"]["inputs"]["theme"] = "kodak"
    rows = rows_of(em, p)
    assert rows["3"]["outputs"][0]["shape"] == [1, 480, 832, 3]
    # the raw matte, the blurred one and the "over" blend's background term (the blend made in the output), per frame
    assert rows["2"]["transient"] == 2 * one + 480 * 640 * 3 * 4 and rows["2"]["outputs"][0]["shape"] == [1, 480, 640, 3]
    assert rows["4"]["transient"] == 480 * 640 * 3 * 4  # the processed frame, clipped into the output
    p["3"]["inputs"].update(width=["9", 1], height=["9", 2])  # sizes from links: known at run time only
    row = rows_of(em, p)["3"]
    assert row["status"] == "not counted" and "width / height come from links" in row["note"]


def test_a_passed_on_input_is_counted_once(em):
    p = {"1": loader(), "2": N("BCVSAM3VideoTrack", images=["1", 0]), "3": N("BCVMaskGuard", mask=["2", 0]),
         "4": N("BC_MaskGrow", mask=["3", 0])}
    rows = rows_of(em, p)
    mask = 609 * 1280 * 720 * 4
    assert rows["3"]["cache_after"] == rows["2"]["cache_after"] + rows["3"]["output_bytes"]
    assert rows["4"]["output_bytes"] == mask // 2 and not rows["4"]["outputs"][0]["shared"]  # a new half mask built on it
    assert rows["4"]["transient"] == 2 * MASK  # a frame read as float32 levels, its blurred levels


def test_bcnodes_mask_nodes_keep_the_dtype(em):
    # BCNodes' mask nodes give a half mask (SAM 3.1's on Load Video's half clip) a half output, one frame at a
    # time, each frame read as its float32 levels; Draw Mask On Image's output takes the image's dtype
    p = {"1": loader(), "2": N("BCVSAM3VideoTrack", images=["1", 0]), "3": N("BC_MaskGrow", mask=["2", 0]),
         "4": N("BC_BlockifyMask", masks=["2", 0], block_size=32), "5": N("BC_MaskFillHoles", masks=["2", 0]),
         "6": N("BC_DrawMaskOnImage", image=["1", 0], mask=["2", 0], color="0, 0, 0")}
    rows = rows_of(em, p)
    for node in "345":
        assert rows[node]["outputs"][0]["shape"] == [609, 1280, 720] and rows[node]["output_bytes"] == 609 * MASK // 2, node
    # Grow: the frame's levels and its blurred levels; Blockify compares the half frame itself; Fill Holes: the levels
    assert [rows[node]["transient"] for node in "345"] == [2 * MASK, 0, MASK]
    assert rows["6"]["outputs"][0]["shape"] == [609, 1280, 720, 3] and rows["6"]["output_bytes"] == 609 * FRAME // 2
    assert rows["6"]["transient"] == 2 * FRAME + MASK  # the frame's levels, its float32 blend, the mask's levels
    p["1"] = loader(precision="fp32")  # float32 in, float32 out, each frame made in the output
    rows = rows_of(em, p)
    assert [rows[node]["transient"] for node in "3456"] == [0, 0, 0, 0]
    assert [rows[node]["output_bytes"] for node in "3456"] == [609 * MASK] * 3 + [609 * FRAME]

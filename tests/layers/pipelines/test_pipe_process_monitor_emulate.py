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
        "BCVWanAnimatePreprocess": ["IMAGE", "IMAGE", "MASK", "POSEDATA", "BBOX", "STRING", "BBOX"],
        "BCVSCAIL2Preprocess": ["IMAGE", "IMAGE", "IMAGE", "MASK", "MASK"], "BCVSCAIL2ColoredMask": ["IMAGE", "IMAGE"],
        "BCVPoseDetection": ["IMAGE", "POSEDATA", "BBOX", "STRING"], "BCVSAM3VideoTrack": ["MASK"],
        "BCVFaceCrop": ["IMAGE", "BBOX"], "BCVMaskGuard": ["MASK", "STRING", "STRING", "IMAGE"],
        "BCVWanAnimatePreprocessGuard": ["MASK", "POSEDATA", "STRING", "STRING", "IMAGE"],
        "BCVSCAIL2PreprocessGuard": ["IMAGE", "IMAGE", "STRING", "STRING", "IMAGE"], "BCVPoseGuard": ["POSEDATA", "STRING", "STRING", "IMAGE"],
        "BCVGetVideoInfo": ["STRING", "STRING", "STRING", "FLOAT", "INT"], "BCVSaveVideo": [], "PreviewImage": ["IMAGE"],
        "BC_SeedVR2Resize": ["IMAGE", "IMAGE"], "BC_SeedVR2VAEEncode": ["LATENT"], "BC_SeedVR2VAEDecode": ["IMAGE"],
        "BC_SeedVR2PostProcess": ["IMAGE"], "BC_ImageScaleByAspectRatio": ["IMAGE", "MASK", "BOX", "INT", "INT"],
        "BC_BiRefNetRemoveBackground": ["IMAGE", "MASK", "IMAGE"], "BC_DepthAnythingV2": ["IMAGE"], "BC_PostFxApply": ["IMAGE"],
        "BC_AnySwitch": ["*"], "BC_SelectSwitch": ["*"],
        "BC_BlockifyMask": ["MASK"], "BC_MaskFillHoles": ["MASK"], "BlockifyMask": ["MASK"],
    }
    SIZES = {"Wan": {"480p": [480, 832], "720p": [720, 1280]}, "SCAIL": {"512p": [512, 896], "704p": [704, 1280]}}

    def __init__(self, weights=None, videos=None, models=None):
        self.weights_fn, self.videos, self.models = weights, videos or {}, models or {}

    def return_types(self, class_type):
        return self.TYPES.get(class_type, [])

    def input_types(self, class_type):
        assert class_type == "BCVLoadVideo"
        return {"required": {"resolution": (["480p", "720p", "512p", "704p"], {"bcv_sizes": self.SIZES,
                                                                              "bcv_frames": {"Wan": 4, "SCAIL": 4}})}}

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
    assert rows["2"]["outputs"][0]["shape"] == [1, 720, 1280, 3]
    assert rows["3"]["outputs"][0]["shape"] == [297, 720, 1280, 3]


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


# -- the profiles of the video nodes, each worked out by hand from the node's code ----------------

PORTRAIT = {"width": 1080, "height": 1920, "frames": 609, "fps": 30.0}


def rows_of(em, prompt, env=None):
    return {row["id"]: row for row in em.estimate(prompt, env or Env(videos={"v.mp4": PORTRAIT}))["rows"]}


def loader(frame_count=""):
    return N("BCVLoadVideo", video="v.mp4", model="Wan", resolution="720p", orientation="auto", force_fps="",
             start_frame=1, frame_count=frame_count)


def test_long_video_samplers(em, pf):
    frame = 1280 * 720 * 3 * 4
    p = {"1": loader(), "2": N("BCVPoseDetection", images=["1", 0]), "3": N("BCVFaceCrop", images=["1", 0]),
         "4": N("BCVWanAnimateLongVideoSampler", pose_video=["2", 0], face_video=["3", 0], width=720, height=1280,
                frames_per_chunk=81, total_frames=0)}
    rows = rows_of(em, p)
    assert rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3] and rows["2"]["transient"] == 0
    assert rows["3"]["outputs"][0]["shape"] == [609, 512, 512, 3]
    sampler = rows["4"]
    assert sampler["outputs"][0]["shape"] == [609, 1280, 720, 3]
    # the decoded chunk + the pose and face windows of one chunk (81 frames each)
    assert sampler["transient"] == 81 * frame + 81 * frame + 81 * 512 * 512 * 3 * 4
    # a run shorter than one chunk: the output is a view of the decoded frames, the 31 past total add
    p["4"]["inputs"].update(total_frames=50)
    assert rows_of(em, p)["4"]["transient"] == 31 * frame + 81 * frame + 81 * 512 * 512 * 3 * 4
    # SCAIL-2: holds pose_video and pose_video_mask; the size is cropped to the VAE's /8
    p = {"1": loader(), "2": N("BCVSCAIL2ColoredMask", driving_mask=["3", 0], replacement_mode=False),
         "3": N("BCVSAM3VideoTrack", images=["1", 0]),
         "4": N("BCVSCAIL2LongVideoSampler", pose_video=["1", 0], pose_video_mask=["2", 0], width=706, height=1284,
                frames_per_chunk=83, total_frames=0)}
    rows = rows_of(em, p)
    assert rows["3"]["outputs"][0]["shape"] == [609, 1280, 720]
    assert rows["2"]["outputs"][0]["shape"] == [609, 1280, 720, 3] and rows["2"]["outputs"][1]["shape"] == [1, 1280, 720, 3]
    assert rows["2"]["transient"] == 16 * 1280 * 720  # the boolean cut of 16 frames
    assert rows["4"]["outputs"][0]["shape"] == [609, 1280, 704, 3]
    assert rows["4"]["transient"] == 81 * 1280 * 704 * 3 * 4 + 2 * 81 * frame


def test_preprocess_wrappers_and_guards(em):
    frame, mask = 1280 * 720 * 3 * 4, 1280 * 720 * 4
    p = {"1": loader(), "9": N("LoadImage", image="ref.png"),
         "2": N("BCVWanAnimatePreprocess", images=["1", 0], mode="prompt"),
         "3": N("BCVWanAnimatePreprocessGuard", mask=["2", 2], pose_data=["2", 3]),
         "4": N("BCVSCAIL2Preprocess", images=["1", 0], reference_image=["9", 0], replacement_mode=False, mode="box_keypoint",
                black_background=False),
         "5": N("BCVSCAIL2PreprocessGuard", pose_video_mask=["4", 1], reference_image_mask=["4", 2]),
         "6": N("BCVMaskGuard", mask=["4", 3]), "7": N("BCVPoseGuard", pose_data=["2", 3]),
         "8": N("PreviewImage", images=["4", 0]), "10": N("BCVSaveVideo", images=["4", 0])}
    rows = rows_of(em, p)
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[609, 1280, 720, 3], [609, 512, 512, 3], [609, 1280, 720]]
    guard = rows["3"]
    assert guard["output_bytes"] == (128 + 190 * 4) * 1200 * 3 * 4  # the mask is passed on; the timeline is new
    assert guard["transient"] == 5 * 1280 * 720 and guard["outputs"][0]["shared"]
    scail = rows["4"]
    shapes = [o["shape"] for o in scail["outputs"]]
    assert shapes == [[609, 1280, 720, 3], [609, 1280, 720, 3], [1, 480, 640, 3], [609, 1280, 720], [1, 480, 640]]
    assert scail["outputs"][0]["shared"]  # black_background off: the driving video itself
    assert scail["output_bytes"] == 609 * frame + 480 * 640 * 3 * 4 + 609 * mask + 480 * 640 * 4
    assert scail["transient"] == 609 * frame  # box_keypoint draws pose images it drops
    assert rows["5"]["transient"] == 609 * 1280 * 720 and rows["5"]["output_bytes"] == (128 + 190 * 2) * 1200 * 3 * 4
    assert rows["6"]["output_bytes"] == (128 + 190 * 3) * 1200 * 3 * 4
    assert rows["7"]["output_bytes"] == (128 + 190 * 2) * 1200 * 3 * 4
    assert rows["8"]["output_bytes"] == 0 and rows["10"]["outputs"] == []
    p["4"]["inputs"].update(mode="prompt", black_background=True)
    scail = rows_of(em, p)["4"]
    assert scail["transient"] == 0 and not scail["outputs"][0]["shared"]


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
    assert rows["3"]["transient"] == elements * 6  # 1920 > tile 1024: the float32 tile sum and its cast
    assert rows["4"]["outputs"][0]["shape"] == [81, 1920, 1088, 3] and rows["4"]["transient"] == 81 * 1920 * 1088 * 3 * 2
    assert rows["5"]["outputs"][0]["shape"] == [81, 1920, 1080, 3] and rows["5"]["output_bytes"] == 81 * 1920 * 1080 * 3 * 2
    p["2"]["inputs"]["downscale_factor"] = 0.5  # resolution comes from the original: the same size, plus the listed downscale
    resize = rows_of(em, p)["2"]
    assert resize["outputs"][0]["shape"] == [81, 1920, 1088, 3] and resize["transient"] == 2 * 81 * 640 * 360 * 3 * 4


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
    assert rows["1"]["transient"] == 576 * 1024 * 3 * 4  # the image list; the mask is zeros, not listed
    assert [o["shape"] for o in rows["2"]["outputs"]] == [[1, 480, 640, 4], [1, 480, 640], [1, 480, 640, 3]]
    assert rows["2"]["transient"] == one  # the frame list before torch.stack
    assert rows["3"]["outputs"][0]["shape"] == [1, 480, 640, 3] and rows["3"]["transient"] == 2 * one
    assert rows["4"]["outputs"][0]["shared"] and rows["4"]["output_bytes"] == 0
    assert rows["5"]["outputs"][0]["shared"] and rows["5"]["note"] == "passes on any_02"  # node 8 does not exist
    assert rows["6"]["outputs"][0]["shape"] == [1, 480, 640, 3] and rows["6"]["output_bytes"] == 0
    p["2"]["inputs"].update(background="Color", mask_blur=2)
    p["4"]["inputs"]["theme"] = "kodak"
    rows = rows_of(em, p)
    assert rows["2"]["transient"] == one + one + 3 * 480 * 640 * 3 * 4 and rows["2"]["outputs"][0]["shape"] == [1, 480, 640, 3]
    assert rows["4"]["transient"] == 2 * 480 * 640 * 3 * 4


def test_a_passed_on_input_is_counted_once(em):
    p = {"1": loader(), "2": N("BCVSAM3VideoTrack", images=["1", 0]), "3": N("BCVMaskGuard", mask=["2", 0]),
         "4": N("BC_MaskGrow", mask=["3", 0])}
    rows = rows_of(em, p)
    mask = 609 * 1280 * 720 * 4
    assert rows["3"]["cache_after"] == rows["2"]["cache_after"] + rows["3"]["output_bytes"]
    assert rows["4"]["output_bytes"] == mask and not rows["4"]["outputs"][0]["shared"]  # a new mask built on it

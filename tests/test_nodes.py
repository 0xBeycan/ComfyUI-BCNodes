"""Every node survives None / empty input, and MaskGrow / Image Scale By
Aspect Ratio match their recorded numeric goldens.

Runs without ComfyUI: the few ComfyUI modules the nodes touch lazily are
stubbed. Needs torch and numpy.

    python tests/test_nodes.py
"""

import importlib
import json
import os
import sys
import tempfile
import types

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "bcnodes_under_test"


def stub_comfy(tmp):
    fp = types.ModuleType("folder_paths")
    fp.models_dir = os.path.join(tmp, "models")
    fp.get_user_directory = lambda: os.path.join(tmp, "user")
    fp.folder_names_and_paths = {}

    def add_model_folder_path(name, path, is_default=False):
        fp.folder_names_and_paths.setdefault(name, ([], set()))[0].append(path)

    def get_full_path(name, filename):
        for p in fp.folder_names_and_paths.get(name, ([], set()))[0]:
            f = os.path.join(p, filename)
            if os.path.isfile(f):
                return f
        return None

    fp.add_model_folder_path = add_model_folder_path
    fp.get_full_path = get_full_path
    fp.get_filename_list = lambda name: []
    fp.get_output_directory = lambda: os.path.join(tmp, "output")
    fp.get_temp_directory = lambda: os.path.join(tmp, "temp")

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        subfolder, filename = os.path.split(os.path.normpath(filename_prefix))
        full_output_folder = os.path.join(output_dir, subfolder)
        return full_output_folder, filename, 1, subfolder, filename_prefix

    fp.get_save_image_path = get_save_image_path
    sys.modules["folder_paths"] = fp

    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    cu = types.ModuleType("comfy.utils")
    import torch

    mm.get_torch_device = lambda: torch.device("cpu")
    mm.vae_device = lambda: torch.device("cpu")
    mm.soft_empty_cache = lambda: None
    mm.load_models_gpu = lambda models, **kw: None
    mm.get_free_memory = lambda device=None, torch_free_too=False: 2 ** 40
    mm.unload_all_models = lambda: None

    import contextlib
    mm.cuda_device_context = lambda device: contextlib.nullcontext()

    # comfy.ldm.seedvr: what the SeedVR2 decode / post-process nodes import lazily.
    ldm = types.ModuleType("comfy.ldm")
    seedvr = types.ModuleType("comfy.ldm.seedvr")
    constants = types.ModuleType("comfy.ldm.seedvr.constants")
    constants.BYTEDANCE_VAE_SCALING_FACTOR = 0.9152
    constants.BYTEDANCE_VAE_SHIFTING_FACTOR = 0.0
    vae_mod = types.ModuleType("comfy.ldm.seedvr.vae")

    class MemoryState:
        DISABLED, INITIALIZING, ACTIVE = 0, 1, 2

    vae_mod.MemoryState = MemoryState
    color_fix = types.ModuleType("comfy.ldm.seedvr.color_fix")
    color_fix.lab_color_transfer = lambda content, style: content
    color_fix.wavelet_color_transfer = lambda content, style: content
    color_fix.adain_color_transfer = lambda content, style: style
    for name, mod in (("comfy.ldm", ldm), ("comfy.ldm.seedvr", seedvr), ("comfy.ldm.seedvr.constants", constants),
                      ("comfy.ldm.seedvr.vae", vae_mod), ("comfy.ldm.seedvr.color_fix", color_fix)):
        sys.modules[name] = mod

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, n):
            pass

    cu.ProgressBar = ProgressBar
    comfy.model_management = mm
    comfy.utils = cu
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.utils"] = cu


def load_package():
    pkg = types.ModuleType(PKG_NAME)
    pkg.__path__ = [PKG_DIR]
    sys.modules[PKG_NAME] = pkg
    return {name: importlib.import_module(f"{PKG_NAME}.nodes.{name}")
            for name in ("logic", "mask", "image_scale", "lists", "birefnet", "downloader", "math_expression", "prompt_list", "any_switch", "seed", "show_text",
                         "image_comparer", "video_comparer", "power_lora_loader", "everywhere", "seedvr2",
                         "postfx", "caption_audit", "social_media_export", "social_export_core", "image_quality_gate", "save_image", "skin_texture")}


def main():
    import torch

    tmp = tempfile.mkdtemp()
    stub_comfy(tmp)
    m = load_package()
    failures = []

    def check(name, fn):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print("FAIL", name, "->", repr(e))

    blank = torch.zeros((1, 64, 64))
    empty_mask = torch.zeros((0, 8, 8))

    lb = m["logic"].LogicBoolean()
    check("LogicBoolean 0.0", lambda: lb.return_boolean(0.0) == (False, 0, 0, 0.0) or _fail())
    check("LogicBoolean default", lambda: lb.return_boolean() == (True, 1, 1, 1.0) or _fail())

    ime = m["logic"].IsMaskEmpty()
    check("IsMaskEmpty None", lambda: ime.is_empty(None) == (True,) or _fail())
    check("IsMaskEmpty empty tensor", lambda: ime.is_empty(empty_mask) == (True,) or _fail())

    mfh = m["mask"].MaskFillHoles()
    check("MaskFillHoles None", lambda: torch.equal(mfh.fill_region(None)[0], blank) or _fail())
    check("MaskFillHoles empty tensor", lambda: torch.equal(mfh.fill_region(empty_mask)[0], blank) or _fail())
    check("MaskFillHoles nothing wired", lambda: torch.equal(mfh.fill_region()[0], blank) or _fail())

    mg = m["mask"].MaskGrow()
    check("MaskGrow None", lambda: torch.equal(mg.mask_grow(True, 4, 4, mask=None)[0], blank) or _fail())
    check("MaskGrow empty tensor", lambda: torch.equal(mg.mask_grow(False, -2, 0, mask=empty_mask)[0], blank) or _fail())
    check("MaskGrow nothing wired", lambda: torch.equal(mg.mask_grow(True, 4, 4)[0], blank) or _fail())

    # Numeric goldens on a centred 16x16 square in a 32x32 canvas, keyed by
    # (grow, blur, invert_mask) -> sum of the output mask. The shrink cases are
    # exact by hand: grow=0 is 16*16, grow=-4 is 8*8, grow=-6 is 4*4.
    square = torch.zeros(1, 32, 32)
    square[:, 8:24, 8:24] = 1.0
    mask_grow_golden = {
        ( -6, 0, False): 16.0,
        ( -6, 0,  True): 324.0,
        ( -6, 6, False): 15.7333,
        ( -6, 6,  True): 401.4274,
        ( -4, 0, False): 64.0,
        ( -4, 0,  True): 488.0,
        ( -4, 6, False): 62.9647,
        ( -4, 6,  True): 525.553,
        (  0, 0, False): 256.0,
        (  0, 0,  True): 768.0,
        (  0, 6, False): 245.7255,
        (  0, 6,  True): 778.2745,
        (  4, 0, False): 536.0,
        (  4, 0,  True): 960.0,
        (  4, 6, False): 498.4471,
        (  4, 6,  True): 961.0353,
        ( 10, 0, False): 940.0,
        ( 10, 0,  True): 1024.0,
        ( 10, 6, False): 908.0,
        ( 10, 6,  True): 1024.0,
    }
    for (_g, _b, _i), _total in mask_grow_golden.items():
        check(f"MaskGrow golden grow={_g} blur={_b} invert={_i}",
              lambda g=_g, b=_b, i=_i, t=_total: (lambda r: r.shape == (1, 32, 32) and r.dtype == torch.float32
                                                  and abs(float(r.sum()) - t) < 1e-3)(
                  mg.mask_grow(invert_mask=i, grow=g, blur=b, mask=square.clone())[0]) or _fail())

    isr = m["image_scale"].ImageScaleByAspectRatio()
    scale_kw = dict(aspect_ratio="original", proportional_width=1, proportional_height=1, fit="crop", method="lanczos",
                    round_to_multiple="16", scale_to_side="longest", scale_to_length=64, background_color="#000000")
    image = torch.rand(2, 48, 96, 3)
    check("ImageScale nothing wired -> ValueError", lambda: _raises(ValueError, lambda: isr.scale(**scale_kw)))
    check("ImageScale image only -> zero mask (B, H, W)",
          lambda: torch.equal(isr.scale(image=image, **scale_kw)[1], torch.zeros((2, 32, 64))) or _fail())
    check("ImageScale placeholder mask -> zero mask (B, H, W)",
          lambda: torch.equal(isr.scale(image=image, mask=blank, **scale_kw)[1], torch.zeros((2, 32, 64))) or _fail())
    check("ImageScale outputs: image (B, H, W, 3), box, width, height",
          lambda: (lambda r: r[0].shape == (2, 32, 64, 3) and r[2] == [96, 48] and (r[3], r[4]) == (64, 32))(isr.scale(image=image, **scale_kw)) or _fail())
    check("ImageScale mask size mismatch -> ValueError",
          lambda: _raises(ValueError, lambda: isr.scale(image=image, mask=torch.ones(1, 10, 10), **scale_kw)))
    check("ImageScale mask only -> IMAGE None, mask (1, 32, 64)",
          lambda: (lambda r: r[0] is None and r[1].shape == (1, 32, 64))(isr.scale(mask=torch.ones(1, 48, 96), **scale_kw)) or _fail())

    # Numeric goldens: a fixed seeded image through the widget grid, keyed by
    # (aspect_ratio, fit, method, scale_to_side, scale_to_length, round_to_multiple)
    # -> (image shape, image sum, width, height). box is [96, 48] for all of them.
    _g11 = torch.Generator().manual_seed(11)
    scale_image = torch.rand(1, 48, 96, 3, generator=_g11)
    image_scale_golden = {
        ("1:1", "crop", "lanczos", "longest", 64, "8"): ((1, 64, 64, 3), 6072.6665, 64, 64),
        ("1:1", "letterbox", "lanczos", "longest", 64, "8"): ((1, 64, 64, 3), 3056.4746, 64, 64),
        ("16:9", "fill", "bicubic", "width", 128, "8"): ((1, 72, 128, 3), 13756.1299, 128, 72),
        ("original", "crop", "bilinear", "shortest", 32, "None"): ((1, 32, 64, 3), 3056.6194, 64, 32),
        ("9:16", "letterbox", "nearest", "height", 96, "16"): ((1, 96, 64, 3), 3073.1763, 64, 96),
        ("2:3", "crop", "box", "longest", 100, "None"): ((1, 100, 66, 3), 9859.9561, 66, 100),
    }
    for _key, _want in image_scale_golden.items():
        check(f"ImageScale golden {'/'.join(str(x) for x in _key)}",
              lambda k=_key, w=_want: (lambda r: tuple(r[0].shape) == w[0] and abs(float(r[0].sum()) - w[1]) < 1e-2
                                       and list(r[2]) == [96, 48] and (r[3], r[4]) == (w[2], w[3]))(
                  isr.scale(image=scale_image.clone(), proportional_width=1, proportional_height=1,
                            background_color="#000000", aspect_ratio=k[0], fit=k[1], method=k[2],
                            scale_to_side=k[3], scale_to_length=k[4], round_to_multiple=k[5])) or _fail())

    jil = m["lists"].JoinImageLists()
    check("JoinImageLists no inputs", lambda: jil.join_lists() == ([], []) or _fail())
    check("JoinImageLists None inputs", lambda: jil.join_lists(In1=None, In2=None) == ([], []) or _fail())
    check("JoinImageLists empty lists", lambda: jil.join_lists(In1=[], In2=[]) == ([], [0, 0]) or _fail())

    me = m["math_expression"].MathExpression()
    check("MathExpression empty expression", lambda: me.run("")["result"] == (0, 0.0) or _fail())
    check("MathExpression whitespace expression", lambda: me.run(" \n ")["result"] == (0, 0.0) or _fail())
    check("MathExpression None expression", lambda: me.run(None)["result"] == (0, 0.0) or _fail())
    check("MathExpression unconnected input -> ValueError", lambda: _raises(ValueError, lambda: me.run("a + 1")))

    pl = m["prompt_list"].PromptList()
    check("PromptList empty text", lambda: pl.make_list("") == ([""], [""], m["prompt_list"].HELP) or _fail())
    check("PromptList None text", lambda: pl.make_list(None)[0] == [""] or _fail())
    check("PromptList start beyond end", lambda: pl.make_list("a\nb", start_index=99, max_rows=5)[0] == ["b"] or _fail())

    sw = m["any_switch"].AnySwitch()
    check("AnySwitch no inputs", lambda: sw.switch() == (None,) or _fail())
    check("AnySwitch all None", lambda: sw.switch(any_01=None, any_02=None) == (None,) or _fail())
    check("AnySwitch skips None", lambda: sw.switch(any_01=None, any_02=0, any_03=5) == (0,) or _fail())

    bn = m["birefnet"].BiRefNetRemoveBackground()
    check("BiRefNet empty batch (no model load)", lambda: bn.remove_background(torch.zeros((0, 8, 8, 3)), "BiRefNet_lite")[1].shape == (0, 64, 64) or _fail())
    check("BiRefNet None image", lambda: bn.remove_background(None, "BiRefNet_lite")[0].shape == (0, 64, 64, 4) or _fail())
    check("BiRefNet None image -> 3 outputs, MASK_IMAGE (0,64,64,3)", lambda: bn.remove_background(None, "BiRefNet_lite")[2].shape == (0, 64, 64, 3) or _fail())

    # The matte options run on a synthetic matte: a white image, a 16x16 matte
    # that is 1 in the centre 8x8 square and 0 outside.
    finish = m["birefnet"].finish
    rgb = torch.ones((1, 16, 16, 3))
    matte = torch.zeros((1, 16, 16))
    matte[:, 4:12, 4:12] = 1.0
    check("BiRefNet finish shapes (Alpha)", lambda: tuple(t.shape for t in finish(rgb, matte)) == ((1, 16, 16, 4), (1, 16, 16), (1, 16, 16, 3)) or _fail())
    check("BiRefNet finish alpha = matte", lambda: torch.equal(finish(rgb, matte)[0][..., 3], matte) or _fail())
    check("BiRefNet finish Color: RGB out, bg where matte is 0", lambda: (lambda img: img.shape == (1, 16, 16, 3) and torch.allclose(img[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])) and torch.allclose(img[0, 8, 8], torch.tensor([1.0, 1.0, 1.0])))(finish(rgb, matte, background="Color", background_color="#ff0000")[0]) or _fail())
    check("BiRefNet finish short colour form", lambda: torch.allclose(finish(rgb, matte, background="Color", background_color="#0f0")[0][0, 0, 0], torch.tensor([0.0, 1.0, 0.0])) or _fail())
    check("BiRefNet finish bad colour -> ValueError", lambda: _raises(ValueError, lambda: finish(rgb, matte, background="Color", background_color="red")))
    check("BiRefNet finish invert", lambda: torch.equal(finish(rgb, matte, invert_output=True)[1], 1 - matte) or _fail())
    check("BiRefNet finish offset +1 grows by one pixel", lambda: (lambda mk: mk[0, 3, 3] == 1 and mk[0, 2, 2] == 0)(finish(rgb, matte, mask_offset=1)[1]) or _fail())
    check("BiRefNet finish offset -1 shrinks by one pixel", lambda: (lambda mk: mk[0, 4, 4] == 0 and mk[0, 5, 5] == 1)(finish(rgb, matte, mask_offset=-1)[1]) or _fail())
    check("BiRefNet finish blur softens the edge, keeps range", lambda: (lambda mk: 0 < mk[0, 4, 8] < 1 and mk.min() >= 0 and mk.max() <= 1)(finish(rgb, matte, mask_blur=2)[1]) or _fail())
    check("BiRefNet finish sensitivity 0.5 amplifies a faint matte", lambda: torch.allclose(finish(rgb, matte * 0.4, sensitivity=0.5)[1][0, 8, 8], torch.tensor(0.6)) or _fail())
    check("BiRefNet finish refine keeps a hard matte's interior", lambda: torch.allclose(finish(rgb, matte, refine_foreground=True)[0][0, 8, 8], torch.tensor([1.0, 1.0, 1.0, 1.0])) or _fail())
    check("BiRefNet finish MASK_IMAGE is the matte on 3 channels", lambda: torch.equal(finish(rgb, matte)[2][..., 1], matte) or _fail())

    sd = m["seed"].Seed()
    check("Seed None -> 0", lambda: sd.main(None) == (0,) or _fail())
    check("Seed 0 stays 0", lambda: sd.main(0) == (0,) or _fail())
    check("Seed -1 -> concrete, never -1", lambda: (lambda s: s != -1 and 0 <= s <= 2 ** 53 - 1)(sd.main(-1)[0]) or _fail())
    check("Seed -1 with no prompt/workflow", lambda: sd.main(-1, prompt=None, extra_pnginfo=None, unique_id="5")[0] >= 0 or _fail())

    def seed_metadata():
        prompt = {"5": {"class_type": "BC_Seed", "inputs": {"seed": -1}}}
        info = {"workflow": {"nodes": [{"id": 5, "type": "BC_Seed", "widgets_values": [-1]}]}}
        (s,) = sd.main(-1, prompt=prompt, extra_pnginfo=info, unique_id="5")
        assert prompt["5"]["inputs"]["seed"] == s and info["workflow"]["nodes"][0]["widgets_values"] == [s]
    check("Seed -1 written back into prompt + workflow metadata", seed_metadata)

    st = m["show_text"].ShowText()
    check("ShowText None", lambda: st.show(None) == {"ui": {"text": []}, "result": ([],)} or _fail())
    check("ShowText empty list", lambda: st.show([]) == {"ui": {"text": []}, "result": ([],)} or _fail())
    check("ShowText list with None", lambda: st.show(["a", None])["ui"]["text"] == ["a", ""] or _fail())

    ic = m["image_comparer"].ImageComparer()
    check("ImageComparer nothing wired", lambda: ic.compare() == {"ui": {"a_images": [], "b_images": []}} or _fail())
    check("ImageComparer empty batches", lambda: ic.compare(torch.zeros((0, 8, 8, 3)), torch.zeros((0, 8, 8, 3))) == {"ui": {"a_images": [], "b_images": []}} or _fail())

    vc = m["video_comparer"].VideoComparer()
    check("VideoComparer nothing wired", lambda: vc.compare() == {"ui": {"bc_video": []}} or _fail())
    check("VideoComparer empty batches", lambda: vc.compare(24.0, torch.zeros((0, 8, 8, 3)), torch.zeros((0, 8, 8, 3))) == {"ui": {"bc_video": []}} or _fail())
    check("VideoComparer fps None", lambda: vc.compare(None) == {"ui": {"bc_video": []}} or _fail())
    fit = m["video_comparer"]._fit
    check("VideoComparer fit: larger clip is cropped, not scaled", lambda: (lambda x: torch.equal(fit(x, 64, 32), x[:32, :64, :3]))(torch.rand(33, 65, 4)) or _fail())
    check("VideoComparer fit: smaller clip letterboxed, aspect kept", lambda: (lambda f: f.shape == (720, 1280, 3) and f[:, :16].abs().sum() == 0 and f[:, -16:].abs().sum() == 0 and f[:, 16:-16].abs().sum() > 0)(fit(torch.ones(480, 832, 3), 1280, 720)) or _fail())

    pl2 = m["power_lora_loader"].PowerLoraLoader()
    check("PowerLoraLoader no model", lambda: pl2.load_loras() == (None,) or _fail())
    check("PowerLoraLoader model, no rows", lambda: pl2.load_loras(model="M") == ("M",) or _fail())
    check("PowerLoraLoader rows all off / missing file -> model untouched", lambda: pl2.load_loras(model="M", lora_1={"on": False, "lora": "x", "strength": 1}, lora_2={"on": True, "lora": "missing.safetensors", "strength": 1}) == ("M",) or _fail())
    check("PowerLoraLoader None row ignored", lambda: pl2.load_loras(model="M", lora_1=None) == ("M",) or _fail())

    ae = m["everywhere"].AnythingEverywhere()
    check("AnythingEverywhere noop", lambda: ae.noop(anything=None) == () or _fail())
    check("FastGroupsBypasser noop", lambda: m["everywhere"].FastGroupsBypasser().noop() == () or _fail())

    dl = m["downloader"].AutoModelDownloader()
    check("Downloader empty entries", lambda: dl.download("")["ui"]["text"] == ["no models listed"] or _fail())
    check("Downloader empty list", lambda: dl.download("[]")["ui"]["text"] == ["no models listed"] or _fail())
    check("Downloader None entries", lambda: dl.download(None)["ui"]["text"] == ["no models listed"] or _fail())
    check("Downloader blank line ignored", lambda: m["downloader"].parse_entries(json.dumps([{"url": " ", "dir": ""}])) == [] or _fail())

    sr = m["seedvr2"].SeedVR2Resize()
    check("SeedVR2Resize None -> ValueError", lambda: _raises(ValueError, lambda: sr.resize(None, 2.0, 0.5, 0, False)))
    check("SeedVR2Resize empty batch -> ValueError", lambda: _raises(ValueError, lambda: sr.resize(torch.zeros(0, 8, 8, 3), 2.0, 0.5, 0, False)))
    # 60x90 (h x w) -> 0.5x lanczos 30x45 -> resolution min(60, 90) * 1 = 60 -> 60 x int(60 * 45 / 30) = 90 -> padded 64 x 96.
    check("SeedVR2Resize downscale, shortest edge from the original, floor long edge, pad 16",
          lambda: (lambda r: r[0].shape == (1, 64, 96, 3) and r[1].shape == (1, 60, 90, 3))(sr.resize(torch.rand(1, 60, 90, 3), 1.0, 0.5, 0, False)) or _fail())
    # 60x100 -> 30x50 -> 60x100, cap 96 -> round(60 * 96 / 100) = 58 by 96 -> image padded to 64x96, reference 58x96.
    check("SeedVR2Resize max_resolution second pass",
          lambda: (lambda r: r[0].shape == (1, 64, 96, 3) and r[1].shape == (1, 58, 96, 3))(sr.resize(torch.rand(1, 60, 100, 3), 1.0, 0.5, 96, False)) or _fail())
    # 20x31, no downscale, x3.2 -> resolution 64 -> 64x99 -> image padded 64x112, reference cropped to 64x98.
    check("SeedVR2Resize odd long edge: reference cropped to even",
          lambda: (lambda r: r[0].shape == (1, 64, 112, 3) and r[1].shape == (1, 64, 98, 3))(sr.resize(torch.rand(1, 20, 31, 3), 3.2, 1.0, 0, False)) or _fail())
    check("SeedVR2Resize RGBA in -> RGB out", lambda: sr.resize(torch.rand(1, 16, 16, 4), 2.0, 1.0, 0, False)[0].shape == (1, 32, 32, 3) or _fail())
    check("SeedVR2Resize 3 frames -> image 5 frames (4n+1), reference 3",
          lambda: (lambda r: r[0].shape[0] == 5 and r[1].shape[0] == 3 and torch.equal(r[0][4], r[0][2]))(sr.resize(torch.rand(3, 16, 16, 3), 2.0, 1.0, 0, False)) or _fail())
    check("SeedVR2Resize 9 frames stay 9",
          lambda: (lambda r: r[0].shape[0] == 9 and r[1].shape[0] == 9)(sr.resize(torch.rand(9, 16, 16, 3), 2.0, 1.0, 0, False)) or _fail())
    check("SeedVR2Resize image and reference float16",
          lambda: (lambda r: r[0].dtype == torch.float16 and r[1].dtype == torch.float16)(sr.resize(torch.rand(1, 16, 16, 3), 2.0, 1.0, 0, True)) or _fail())

    class VideoAutoencoderKLWrapper:  # stand-in for comfy's SeedVR2 VAE: 4 pixel frames <-> 1 latent frame (first slice 5 <-> 2), 8x spatial
        use_slicing = True
        slicing_latent_min_size = 1
        slicing_sample_min_size = 4
        temporal_downsample_factor = 4
        device = torch.device("cpu")
        calls = []

        def _decode(self, z, memory_state=0, memory_cache=None):
            self.calls.append((tuple(z.shape), memory_state, memory_cache is not None))
            t = z.shape[2] * 4 - 3 if memory_state != 2 else z.shape[2] * 4
            return torch.full((z.shape[0], 3, t, z.shape[3] * 8, z.shape[4] * 8), float(z.shape[2]), dtype=torch.float16)

        def _encode(self, x, memory_state=0, memory_cache=None):
            self.calls.append((tuple(x.shape), memory_state, memory_cache is not None))
            t = (x.shape[2] + 3) // 4 if memory_state != 2 else x.shape[2] // 4
            h = torch.zeros((x.shape[0], 32, t, x.shape[3] // 8, x.shape[4] // 8), dtype=torch.float16)
            h[:, :16] = x.float().mean()  # the mean half carries the slice mean, so the range conversion is visible
            return h

    class FakeVAE:
        first_stage_model = VideoAutoencoderKLWrapper()
        vae_dtype = torch.float16
        device = torch.device("cpu")
        patcher = None
        disable_offload = True
        memory_used_decode = staticmethod(lambda shape, dtype: 0)
        process_output = staticmethod(lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0))

    se = m["seedvr2"].SeedVR2VAEEncode()
    check("SeedVR2VAEEncode 1 frame 32x48 -> latent (1, 16, 1, 4, 6) float32",
          lambda: (lambda r: r[0]["samples"].shape == (1, 16, 1, 4, 6) and r[0]["samples"].dtype == torch.float32)(se.encode(torch.ones(1, 32, 48, 3), FakeVAE(), 4096, 256)) or _fail())
    check("SeedVR2VAEEncode 9 frames -> 3 latent frames; slices 5 (INITIALIZING) + 4 (ACTIVE), shared cache",
          lambda: (VideoAutoencoderKLWrapper.calls.clear(),
                   se.encode(torch.ones(9, 16, 16, 3), FakeVAE(), 4096, 256)[0]["samples"].shape == (1, 16, 3, 2, 2)
                   and VideoAutoencoderKLWrapper.calls == [((1, 3, 5, 16, 16), 1, True), ((1, 3, 4, 16, 16), 2, True)])[1] or _fail())
    check("SeedVR2VAEEncode 7 frames: runt last slice (2) merged into the first one -> one 7-frame call",
          lambda: (VideoAutoencoderKLWrapper.calls.clear(), se.encode(torch.ones(7, 16, 16, 3), FakeVAE(), 4096, 256),
                   [c[0][2] for c in VideoAutoencoderKLWrapper.calls] == [7])[2] or _fail())
    check("SeedVR2VAEEncode x*2-1 then * 0.9152: ones -> 1 * 0.9152",
          lambda: torch.allclose(se.encode(torch.ones(1, 16, 16, 3), FakeVAE(), 4096, 256)[0]["samples"], torch.full((1, 16, 1, 2, 2), 0.9152)) or _fail())
    check("SeedVR2VAEEncode RGBA in -> 3 channels used", lambda: se.encode(torch.ones(1, 16, 16, 4), FakeVAE(), 4096, 256)[0]["samples"].shape == (1, 16, 1, 2, 2) or _fail())
    check("SeedVR2VAEEncode None -> ValueError", lambda: _raises(ValueError, lambda: se.encode(None, FakeVAE(), 4096, 256)))
    # 48x64 frames, tile 32 / overlap 16 -> 2 x 3 tiles on the pixel grid, blended on the latent grid; constant input stays constant.
    check("SeedVR2VAEEncode tiled: 9 frames 48x64, tile 32/16 -> (1, 16, 3, 6, 8), blend normalises to the input",
          lambda: (lambda r: r["samples"].shape == (1, 16, 3, 6, 8) and torch.allclose(r["samples"], torch.full((1, 16, 3, 6, 8), 0.9152), atol=1e-3))(
              se.encode(torch.ones(9, 48, 64, 3), FakeVAE(), 32, 16)[0]) or _fail())
    check("SeedVR2VAEEncode tiled: one causal cache per tile, INITIALIZING then ACTIVE, 6 tiles x 2 slices",
          lambda: (VideoAutoencoderKLWrapper.calls.clear(), se.encode(torch.ones(9, 48, 64, 3), FakeVAE(), 32, 16),
                   [c[1] for c in VideoAutoencoderKLWrapper.calls] == [1, 2] * 6)[2] or _fail())

    sd = m["seedvr2"].SeedVR2VAEDecode()
    check("SeedVR2VAEDecode 1 latent frame -> 1 frame, (B*T, H, W, 3) float16, even crop",
          lambda: (lambda r: r[0].shape == (1, 40, 56, 3) and r[0].dtype == torch.float16)(sd.decode({"samples": torch.zeros(1, 16, 1, 5, 7)}, FakeVAE(), 4096, 256)) or _fail())
    check("SeedVR2VAEDecode 3 latent frames -> 9 frames, slices INITIALIZING then ACTIVE with a shared cache",
          lambda: (VideoAutoencoderKLWrapper.calls.clear(),
                   sd.decode({"samples": torch.zeros(1, 16, 3, 2, 2)}, FakeVAE(), 4096, 256)[0].shape == (9, 16, 16, 3)
                   and VideoAutoencoderKLWrapper.calls == [((1, 16, 2, 2, 2), 1, True), ((1, 16, 1, 2, 2), 2, True)])[1] or _fail())
    check("SeedVR2VAEDecode values pass through (x + 1) / 2",
          lambda: torch.equal(sd.decode({"samples": torch.zeros(1, 16, 1, 2, 2)}, FakeVAE(), 4096, 256)[0], torch.full((1, 16, 16, 3), 1.0, dtype=torch.float16)) or _fail())
    check("SeedVR2VAEDecode wrong VAE -> ValueError", lambda: _raises(ValueError, lambda: sd.decode({"samples": torch.zeros(1, 16, 1, 2, 2)}, object(), 4096, 256)))
    check("SeedVR2VAEDecode 4-D latent -> ValueError", lambda: _raises(ValueError, lambda: sd.decode({"samples": torch.zeros(1, 16, 2, 2)}, FakeVAE(), 4096, 256)))
    # latent 6x8, tile 32 px -> 4 latent cells, overlap 16 px -> 2 cells: 2 x 3 tiles; constant frames stay constant after the blend.
    check("SeedVR2VAEDecode tiled: (1, 16, 3, 6, 8), tile 32/16 -> (9, 48, 64, 3) of ones",
          lambda: (lambda r: r.shape == (9, 48, 64, 3) and torch.equal(r, torch.ones(9, 48, 64, 3, dtype=torch.float16)))(
              sd.decode({"samples": torch.zeros(1, 16, 3, 6, 8)}, FakeVAE(), 32, 16)[0]) or _fail())
    check("SeedVR2VAEDecode tiled: 6 tiles x 2 slices, one cache per tile",
          lambda: (VideoAutoencoderKLWrapper.calls.clear(), sd.decode({"samples": torch.zeros(1, 16, 3, 6, 8)}, FakeVAE(), 32, 16),
                   [c[1] for c in VideoAutoencoderKLWrapper.calls] == [1, 2] * 6)[2] or _fail())

    pp = m["seedvr2"].SeedVR2PostProcess()
    dec = torch.rand(5, 34, 22, 3)
    ref = torch.rand(3, 32, 20, 3)
    check("SeedVR2PostProcess crops to the reference frames/size and to even, float16",
          lambda: (lambda r: r[0].shape == (3, 32, 20, 3) and r[0].dtype == torch.float16)(pp.process(dec, ref, "lab")) or _fail())
    check("SeedVR2PostProcess none = cropped input",
          lambda: torch.equal(pp.process(dec, ref, "none")[0], dec[:3, :32, :20].to(torch.float16)) or _fail())
    check("SeedVR2PostProcess lab stub passthrough = input through the range round trip",
          lambda: (pp.process(dec, ref, "lab")[0].float() - dec[:3, :32, :20]).abs().max().item() < 1e-3 or _fail())
    check("SeedVR2PostProcess alpha from the reference",
          lambda: (lambda r: r[0].shape == (3, 32, 20, 4) and torch.equal(r[0][..., 3], torch.ones(3, 32, 20, dtype=torch.float16)))(
              pp.process(dec, torch.cat([ref, torch.ones(3, 32, 20, 1)], dim=-1), "none")) or _fail())
    check("SeedVR2PostProcess odd reference -> even output", lambda: pp.process(torch.rand(1, 33, 21, 3), torch.rand(1, 33, 21, 3), "none")[0].shape == (1, 32, 20, 3) or _fail())
    check("SeedVR2PostProcess None -> ValueError", lambda: _raises(ValueError, lambda: pp.process(None, ref, "lab")))
    check("SeedVR2PostProcess bad method -> ValueError", lambda: _raises(ValueError, lambda: pp.process(dec, ref, "hsv")))

    # --- PostFx -------------------------------------------------------------
    pfa = m["postfx"].PostFxApply()
    themes = pfa.INPUT_TYPES()["required"]["theme"][0]
    default_theme = pfa.INPUT_TYPES()["required"]["theme"][1]["default"]
    img = torch.rand(2, 32, 40, 3)
    check("PostFxApply themes: 'none' first, portra_400 default", lambda: (themes[0] == "none" and default_theme.endswith("portra_400")) or _fail())
    check("PostFxApply theme none = the input object", lambda: pfa.apply(img, "none", "neutral", 1.0, 0, "fixed")[0] is img or _fail())
    check("PostFxApply default theme: same shape, float32, in [0, 1], changed",
          lambda: (lambda r: r.shape == img.shape and r.dtype == torch.float32 and r.min() >= 0 and r.max() <= 1 and not torch.equal(r, img))(
              pfa.apply(img, default_theme, "neutral", 1.0, 0, "fixed")[0]) or _fail())
    check("PostFxApply strength 0 = the input", lambda: torch.allclose(pfa.apply(img, default_theme, "neutral", 0.0, 0, "fixed")[0], img, atol=1e-5) or _fail())
    check("PostFxApply all-black mask is ignored",
          lambda: torch.equal(pfa.apply(img, default_theme, "neutral", 1.0, 0, "fixed", mask=torch.zeros(1, 32, 40))[0],
                              pfa.apply(img, default_theme, "neutral", 1.0, 0, "fixed")[0]) or _fail())
    half = torch.zeros(1, 32, 40); half[:, :, 20:] = 1.0
    check("PostFxApply half mask: left = source, right = effect",
          lambda: (lambda r, full: torch.allclose(r[:, :, :20], img[:, :, :20]) and torch.allclose(r[:, :, 20:], full[:, :, 20:]))(
              pfa.apply(img, default_theme, "neutral", 1.0, 0, "fixed", mask=half)[0], pfa.apply(img, default_theme, "neutral", 1.0, 0, "fixed")[0]) or _fail())
    check("PostFxApply batch_seed increment: frames get different grain",
          lambda: (lambda r: not torch.equal(r[0], r[1]))(pfa.apply(torch.rand(1, 32, 40, 3).expand(2, -1, -1, -1).contiguous(),
                                                                    default_theme, "night_flash", 1.0, 0, "increment")[0]) or _fail())
    pft = m["postfx"].PostFxTheme()
    look = pft.load(default_theme)[0]
    check("PostFxTheme -> look dict with a name", lambda: (isinstance(look, dict) and look.get("name")) or _fail())
    check("PostFxApply connected look overrides theme none", lambda: not torch.equal(pfa.apply(img, "none", "neutral", 1.0, 0, "fixed", look=look)[0], img) or _fail())
    pfc = m["postfx"].PostFxCustomLook()
    neutral_args = dict(temp=0.0, tint=0.0, exposure=0.0, contrast=0.0, vibrance=0.0, saturation=1.0,
                        grain=0.0, grain_size=1.5, vignette=0.0, halation=0.0, clarity=0.0)
    check("PostFxCustomLook all neutral on a look = same name, look untouched",
          lambda: (lambda r: r["name"] == look["name"] and r is not look)(pfc.build(look=look, **neutral_args)[0]) or _fail())
    check("PostFxCustomLook exposure moved -> '+custom' and an exposure block",
          lambda: (lambda r: r["name"].endswith("+custom") and r["exposure"]["stops"] == 1.0)(pfc.build(**dict(neutral_args, exposure=1.0))[0]) or _fail())
    pfl = m["postfx"].PostFxLut()
    check("PostFxLut none, no path -> neutral look", lambda: pfl.build("none", 1.0)[0].get("lut", {}).get("file") in (None, "") or _fail())
    check("PostFxLut missing file -> FileNotFoundError", lambda: _raises(FileNotFoundError, lambda: pfl.build("none", 1.0, lut_path=os.path.join(tmp, "missing.cube"))))
    check("PostFxLut dropdown lists nothing when luts/ is empty", lambda: pfl.INPUT_TYPES()["required"]["lut"][0][0] == "none" or _fail())
    pfs = m["postfx"].PostFxSignatureSheet()
    check("PostFxSignatureSheet -> one image", lambda: (lambda r: r.ndim == 4 and r.shape[0] == 1 and r.shape[-1] == 3)(pfs.build(torch.rand(1, 48, 64, 3), "signature", "neutral", 1.0, 5)[0]) or _fail())

    # --- Caption Audit ------------------------------------------------------
    ca = m["caption_audit"]
    ds = os.path.join(tmp, "captions")
    os.makedirs(ds)
    for i, text in enumerate(["sks1 woman, red scarf, studio lighting", "sks1 woman, red scarf, on a beach",
                              "sks1 woman, red scarf, night street", "sks1 woman, red scarf, close-up"]):
        with open(os.path.join(ds, f"{i}.txt"), "w") as fh:
            fh.write(text)
    node = ca.CaptionAudit()
    audit_kw = dict(trigger="sks1", class_words="woman", fuse="", critical_threshold=0.85, warn_threshold=0.6,
                    info_threshold=0.35, ngram_max=3, no_stopwords=False, recursive=False, table_rows=12)
    out = node.audit(ds, **audit_kw)
    check("CaptionAudit -> ui payload + 5 results", lambda: (set(out) == {"ui", "result"} and len(out["result"]) == 5) or _fail())
    check("CaptionAudit card is (1, H, 1280, 3) with H from table_rows",
          lambda: out["result"][0].shape == (1, ca.card_size(12)[1], 1280, 3) or _fail())
    check("CaptionAudit 'red scarf' in every caption -> critical >= 1", lambda: (out["result"][3] >= 1 and "red scarf" in out["result"][1]) or _fail())
    check("CaptionAudit report_json parses", lambda: json.loads(out["result"][2])["summary"]["critical"] == out["result"][3] or _fail())
    check("CaptionAudit fuse='red scarf' -> critical 0", lambda: node.audit(ds, **dict(audit_kw, fuse="red scarf"))["result"][3] == 0 or _fail())
    check("CaptionAudit preview lands in the temp dir", lambda: any(f.startswith("caption_audit_") for f in os.listdir(os.path.join(tmp, "temp"))) or _fail())
    check("CaptionAudit missing folder -> error card, critical 1, no raise",
          lambda: (lambda r: r["result"][3] == 1 and r["result"][2] == "{}")(node.audit(os.path.join(tmp, "nope"), **audit_kw)) or _fail())
    check("CaptionAudit thresholds out of order -> error card, critical 1",
          lambda: node.audit(ds, **dict(audit_kw, warn_threshold=0.9))["result"][3] == 1 or _fail())
    fp0 = ca.CaptionAudit.IS_CHANGED(ds)
    with open(os.path.join(ds, "0.txt"), "a") as fh:
        fh.write(", edited")
    os.utime(os.path.join(ds, "0.txt"), None)
    check("CaptionAudit IS_CHANGED moves when a caption is edited", lambda: ca.CaptionAudit.IS_CHANGED(ds) != fp0 or _fail())

    # A widget path comes from the workflow JSON, so it is confined to the
    # ComfyUI tree plus the roots named in BC_CAPTION_ROOTS.
    outside = tempfile.mkdtemp(prefix="bcnodes_outside_")
    with open(os.path.join(outside, "0.txt"), "w") as fh:
        fh.write("sks1 woman, red scarf, studio lighting")
    check("CaptionAudit outside the allowed roots -> error card, critical 1, no raise",
          lambda: (lambda r: r["result"][3] == 1 and r["result"][2] == "{}")(node.audit(outside, **audit_kw)) or _fail())
    check("CaptionAudit resolve_dir rejects an outside path", lambda: _raises(ValueError, lambda: ca.resolve_dir(outside)))
    check("CaptionAudit resolve_dir rejects '..' back out of a root",
          lambda: _raises(ValueError, lambda: ca.resolve_dir(os.path.join(ds, "..", "..", "..", "etc"))))
    check("CaptionAudit IS_CHANGED on a rejected path is stable, not a raise",
          lambda: (lambda v: isinstance(v, str) and v.startswith("rejected:") and v == ca.CaptionAudit.IS_CHANGED(outside))(ca.CaptionAudit.IS_CHANGED(outside)) or _fail())
    os.environ[ca.ROOTS_ENV] = outside
    try:
        check("CaptionAudit BC_CAPTION_ROOTS opens a root outside ComfyUI",
              lambda: ca.resolve_dir(outside) == os.path.realpath(outside) or _fail())
        check("CaptionAudit an opened root audits normally", lambda: node.audit(outside, **audit_kw)["result"][2] != "{}" or _fail())
    finally:
        del os.environ[ca.ROOTS_ENV]

    # --- Social Media Export ------------------------------------------------
    sme = m["social_media_export"].SocialMediaExport()
    req = sme.INPUT_TYPES()["required"]
    check("SocialMediaExport one checkbox per platform, instagram_feed on by default",
          lambda: (req["instagram_feed"][0] == "BOOLEAN" and req["instagram_feed"][1]["default"] is True and req["resize_mode"][0] == ["crop", "pad"]) or _fail())
    master = torch.rand(1, 90, 160, 3)
    check("SocialMediaExport no platform -> ValueError", lambda: _raises(ValueError, lambda: sme.export(master, "crop", 92, False, "social/export")))
    r = sme.export(master, "crop", 92, False, "social/export", instagram_feed=True)
    check("SocialMediaExport passes the input through and returns a report", lambda: (r["result"][0] is master and r["result"][1].startswith(m["social_export_core"].REPORT_HEADER.split("\n")[0])) or _fail())
    check("SocialMediaExport writes output/social/export_instagram_feed_00001_.jpg",
          lambda: os.path.isfile(os.path.join(tmp, "output", "social", "export_instagram_feed_00001_.jpg")) or _fail())
    check("SocialMediaExport pad mode, two platforms, batch of 2 -> 4 report lines",
          lambda: len(sme.export(torch.rand(2, 90, 160, 3), "pad", 80, True, "social/pad", instagram_feed=True, instagram_story=True)["result"][1].splitlines()) == 1 + 4 + len(m["social_export_core"].REPORT_HEADER.splitlines()) - 1 or _fail())

    # --- Save Image ---------------------------------------------------------
    si = m["save_image"]
    from datetime import datetime
    ts = datetime(2024, 5, 22, 9, 13, 58)
    sample_prompt = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
                                                    "positive": ["6", 0], "negative": ["7", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl/base_v1.safetensors"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry", "clip": ["4", 1]}},
        "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors", "cfg": 3.5}},
    }
    name = lambda keys, prefix="", named=False: si.custom_name([k.strip() for k in keys.split(",")], prefix, "-", sample_prompt, "512x768", ts, named)
    check("SaveImage name: widget keys, highest node wins, float trimmed", lambda: name("sampler_name, cfg, steps", "ComfyUI") == "ComfyUI-euler-3.5-20" or _fail())
    check("SaveImage name: 3.cfg picks node 3; unknown node falls back", lambda: name("3.cfg, 99.cfg") == "7.0-3.5" or _fail())
    check("SaveImage name: strftime, quoted literal, resolution, unknown key literal", lambda: name("%F %H-%M-%S, 'fixed', resolution, nope") == "2024-05-22 09-13-58-'fixed'-512x768-nope" or _fail())
    check("SaveImage name: ckpt_name strips the extension, ckpt_path is its folder, lora_name", lambda: name("ckpt_name, ckpt_path, lora_name") == "base_v1-sdxl-style" or _fail())
    check("SaveImage name: /sub steps into a folder, named_keys", lambda: name("seed, /steps", named=True) == os.sep.join(["seed=5", "steps=20"]) or _fail())
    check("SaveImage name: ../ climbs, illegal characters dropped", lambda: name("../seed, 'a:b?'") == os.sep.join(["..", "5-'ab'"]) or _fail())
    check("SaveImage name: no prompt -> prefix only", lambda: si.custom_name(["cfg"], "x", "-", None, "1x1", ts, False) == "x" or _fail())
    cdir = os.path.join(tmp, "counter")
    os.makedirs(cdir)
    for f in ("img-0001.webp", "img-0007.webp", "other-0009.webp", "0003.webp", "0002-img.webp"):
        open(os.path.join(cdir, f), "w").close()
    check("SaveImage counter: last -> 8", lambda: si.latest_counter(cdir, "img", 4, "last", ".webp") == 8 or _fail())
    check("SaveImage counter: first -> 3", lambda: si.latest_counter(cdir, "img", 4, "first", ".webp") == 3 or _fail())
    check("SaveImage counter: no name -> 4; other ext -> 1; missing dir -> 1",
          lambda: (si.latest_counter(cdir, "", 4, "last", ".webp") == 4 and si.latest_counter(cdir, "img", 4, "last", ".png") == 1
                   and si.latest_counter(os.path.join(tmp, "nope"), "img", 4, "last", ".webp") == 1) or _fail())
    node = si.SaveImage()
    save_kw = dict(filename_prefix="shot", filename_keys="sampler_name, 3.cfg", foldername_prefix="", foldername_keys="ckpt_name", delimiter="-",
                   save_job_data="basic, models, sampler, prompt", job_data_per_image=False, job_custom_text="note", save_metadata=True,
                   counter_digits=4, counter_position="last", one_counter_per_folder=True, image_preview=True, output_ext=".png", quality=90,
                   named_keys=False, prompt=sample_prompt, extra_pnginfo={"workflow": {"nodes": []}})
    pixels = torch.rand(2, 8, 12, 3)
    r = node.save_images(pixels, **save_kw)
    out_png = os.path.join(tmp, "output", "base_v1", "shot-euler-7.0-0001.png")
    check("SaveImage writes output/base_v1/shot-euler-7.0-0001.png + 0002, ui lists both with the subfolder",
          lambda: (os.path.isfile(out_png) and os.path.isfile(out_png.replace("0001", "0002"))
                   and r["ui"]["images"] == [{"filename": "shot-euler-7.0-0001.png", "subfolder": "base_v1", "type": "output"},
                                             {"filename": "shot-euler-7.0-0002.png", "subfolder": "base_v1", "type": "output"}]) or _fail())
    def png_meta():
        from PIL import Image
        with Image.open(out_png) as im:
            return json.loads(im.text["prompt"])["4"]["inputs"]["ckpt_name"] == "sdxl/base_v1.safetensors" and "workflow" in im.text and im.size == (12, 8)
    check("SaveImage PNG carries prompt + workflow text chunks", lambda: png_meta() or _fail())
    def job():
        with open(os.path.join(tmp, "output", "base_v1", "jobs.json")) as fh:
            (entry,) = json.load(fh).values()
        return (entry["resolution"] == "12x8" and entry["custom_text"] == "note" and entry["checkpoint"] == "sdxl/base_v1"
                and entry["loras"] == "style" and entry["sampler_parameters"]["sampler_name"] == "euler" and entry["positive_prompt"] == "a cat"
                and entry["negative_prompt"] == "blurry")
    check("SaveImage jobs.json: basic, models (loras), sampler, prompts read from KSampler links", lambda: job() or _fail())
    r = node.save_images(pixels[:1], **dict(save_kw, image_preview=False, output_ext=".webp", quality=100, job_data_per_image=True))
    check("SaveImage second run: counter continues, image_preview off -> no ui images, per-image json",
          lambda: (r["ui"]["images"] == [] and os.path.isfile(os.path.join(tmp, "output", "base_v1", "shot-euler-7.0-0001.webp"))
                   and os.path.isfile(os.path.join(tmp, "output", "base_v1", "shot-euler-7.0-0001.json"))) or _fail())
    def webp_meta():
        from PIL import Image
        with Image.open(os.path.join(tmp, "output", "base_v1", "shot-euler-7.0-0001.webp")) as im:
            exif = im.getexif()
        return exif[0x010F].startswith("Prompt: ") and json.loads(exif[0x010F][8:])["3"]["inputs"]["seed"] == 5 and exif[0x010E].startswith("Workflow: ")
    check("SaveImage WebP lossless carries prompt / workflow in EXIF", lambda: webp_meta() or _fail())
    r = node.save_images(pixels[:1], **dict(save_kw, filename_prefix="", filename_keys="", foldername_keys="", save_job_data="disabled", counter_digits=3, output_ext=".jpg"))
    check("SaveImage no name at all -> output/001.jpg, root subfolder ''",
          lambda: (os.path.isfile(os.path.join(tmp, "output", "001.jpg")) and r["ui"]["images"][0]["subfolder"] == "") or _fail())
    r = node.save_images(pixels[:1], **dict(save_kw, counter_digits=0, counter_position="first", save_job_data="disabled", save_metadata=False))
    check("SaveImage counter 0 -> plain name, overwrite", lambda: os.path.isfile(os.path.join(tmp, "output", "base_v1", "shot-euler-7.0.png")) or _fail())
    check("SaveImage None / empty batch -> no files, no raise",
          lambda: (node.save_images(None, **save_kw) == {"ui": {"images": []}} and node.save_images(torch.zeros(0, 8, 8, 3), **save_kw) == {"ui": {"images": []}}) or _fail())
    check("SaveImage output_ext combo has the base formats, .webp default",
          lambda: (lambda spec: set(si.BASE_EXTENSIONS) <= set(spec[0]) and spec[1]["default"] == ".webp")(si.SaveImage.INPUT_TYPES()["required"]["output_ext"]) or _fail())

    # --- Image Quality Gate -------------------------------------------------
    iqg = m["image_quality_gate"].ImageQualityGate()
    args = dict(blur_threshold=0.4, blur_var_threshold=30.0, sharpness_threshold=15.0, noise_threshold=25.0, clipping_threshold=0.02, entropy_threshold=5.0)
    noise = torch.rand(1, 96, 128, 3)
    r = iqg.analyze(noise, "custom", "full image", **args)
    check("ImageQualityGate -> badge (1, 310, 420, 3), int verdict, report, 5 floats",
          lambda: (r[0].shape == (1, 310, 420, 3) and r[1] in (0, 1, 2) and r[2].split(" ")[0] in ("PASS", "SO-SO", "FAIL") and len(r) == 8
                   and all(isinstance(float(x), float) for x in r[3:])) or _fail())
    check("ImageQualityGate flat grey -> entropy 0 -> FAIL (0)", lambda: (lambda r: r[1] == 0 and r[7] == 0.0)(iqg.analyze(torch.full((1, 96, 128, 3), 0.5), "custom", "full image", **args)) or _fail())
    check("ImageQualityGate every preset + center-weighted runs",
          lambda: all(iqg.analyze(noise, st, "center-weighted", **args)[1] in (0, 1, 2) for st in ("close-up", "medium", "wide / full-body")) or _fail())
    check("ImageQualityGate badge colour follows the verdict",
          lambda: (lambda b: torch.allclose(b[0, 5, 5], torch.tensor([170, 30, 30]) / 255.0))(iqg.analyze(torch.full((1, 96, 128, 3), 0.5), "custom", "full image", **args)[0]) or _fail())

    # --- Skin Texture -------------------------------------------------------
    st = m["skin_texture"]
    node = st.SkinTexture()
    kw = dict(sam3_model=st.DEFAULT_SAM3, texture=0.5, detail=0.6, pore_scale=1.0, feather=0, seed=3, threshold=0.5)
    src = torch.rand(1, 64, 80, 3) * 0.5 + 0.25
    ones = torch.ones(1, 64, 80)
    check("SkinTexture: the default checkpoint is offered even with no checkpoints on disk",
          lambda: st.DEFAULT_SAM3 in node.INPUT_TYPES()["required"]["sam3_model"][0] or _fail())
    check("SkinTexture None -> empty", lambda: node.run(None, **kw)[0].shape == (0, 64, 64, 3) or _fail())
    r = node.run(src, mask=ones, **kw)
    check("SkinTexture: mask path skips SAM 3, same shape, in [0, 1], changed",
          lambda: (r[0].shape == src.shape and r[0].min() >= 0 and r[0].max() <= 1 and not torch.equal(r[0], src) and torch.equal(r[1], ones)) or _fail())
    check("SkinTexture: texture 0 + detail 0 = the input", lambda: torch.allclose(node.run(src, mask=ones, **dict(kw, texture=0.0, detail=0.0))[0], src, atol=2e-3) or _fail())
    check("SkinTexture: zero mask = the input", lambda: torch.allclose(node.run(src, mask=torch.zeros(1, 64, 80), **kw)[0], src, atol=1e-6) or _fail())
    half = torch.zeros(1, 64, 80); half[:, :, 40:] = 1
    check("SkinTexture: texture stays inside the mask",
          lambda: (lambda o: torch.allclose(o[:, :, :40], src[:, :, :40], atol=1e-6) and not torch.allclose(o[:, :, 40:], src[:, :, 40:], atol=1e-3))(node.run(src, mask=half, **kw)[0]) or _fail())
    check("SkinTexture: exclude_mask carves the mask", lambda: torch.equal(node.run(src, mask=ones, exclude_mask=half, **kw)[1], 1 - half) or _fail())
    check("SkinTexture: feather softens the mask edge", lambda: (lambda mm: 0 < mm[0, 32, 40] < 1)(node.run(src, mask=half, **dict(kw, feather=8))[1]) or _fail())
    check("SkinTexture: mask of another size is resized", lambda: node.run(src, mask=torch.ones(1, 32, 40), **kw)[1].shape == (1, 64, 80) or _fail())
    check("SkinTexture: deterministic in the seed", lambda: torch.equal(node.run(src, mask=ones, **kw)[0], node.run(src, mask=ones, **kw)[0]) or _fail())
    check("SkinTexture: another seed, another field", lambda: not torch.equal(node.run(src, mask=ones, **dict(kw, seed=4))[0], r[0]) or _fail())
    check("SkinTexture: black stays black (luma gate)", lambda: torch.allclose(node.run(torch.zeros(1, 64, 80, 3), mask=ones, **kw)[0], torch.zeros(1, 64, 80, 3), atol=1e-6) or _fail())
    check("SkinTexture: face gate — no face = 1, small face fades", lambda: (st.face_gate(None, 100) == 1.0 and st.face_gate(torch.zeros(100, 100), 100) == 1.0
          and 0 < st.face_gate((lambda f: (f.__setitem__((slice(40, 50), slice(0, 10)), 1.0), f)[1])(torch.zeros(100, 100)), 100) < 1.0) or _fail())
    check("SkinTexture: batch of 3 with one mask", lambda: node.run(torch.rand(3, 64, 80, 3), mask=ones, **kw)[0].shape == (3, 64, 80, 3) or _fail())

    if failures:
        print(f"\nFAIL: {len(failures)}")
        sys.exit(1)
    print("\nOK")


def _fail():
    raise AssertionError("unexpected result")


def _raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"{exc.__name__} not raised")


if __name__ == "__main__":
    main()

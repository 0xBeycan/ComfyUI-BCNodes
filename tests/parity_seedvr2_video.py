"""BC_SeedVR2VAEEncode, BC_SeedVR2VAEDecode and BC_SeedVR2PostProcess against ComfyUI's own nodes, numerically.

    python tests/parity_seedvr2_video.py [--comfy PATH] [--vae PATH] [--frames N --height H --width W] [--tile T --overlap O]

Post-process: the real `SeedVR2PostProcessing` (comfy_extras/nodes_seedvr.py)
and the port run on identical synthetic clips for every method, with odd
sizes, extra frames and an alpha reference. Expected: the port equals the
native float32 result rounded to float16 (`torch.equal` after `.half()`).

VAE encode / decode (needs `--vae` and a GPU): a synthetic float16 clip is
encoded by `VAE.encode_tiled` (the native VAE Encode (Tiled) path) and by the
port — expected equal — then the native latent is decoded by
`VAE.decode_tiled` and by the port — expected equal after rounding the native
output to float16. `--tile` / `--overlap` are the tiled nodes' widgets (default
1024 / 256; a tile larger than the frame is the single-tile path). Peak VRAM
of every pass is printed; the ports' must not grow with `--frames`.

Needs a ComfyUI checkout (`--comfy`, `$COMFYUI_DIR`, or a sibling `ComfyUI`)
with its requirements installed. Exits non-zero on any mismatch.
"""

import argparse
import importlib
import os
import sys
import types

import torch

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BC_PKG = "bcnodes_under_test"
FP16_EPS = 2.0 ** -11  # one float16 ulp below 1.0


def load_comfy(comfy_dir):
    if not comfy_dir or not os.path.isfile(os.path.join(comfy_dir, "comfy", "sd.py")):
        sys.exit(f"ComfyUI checkout not found at {comfy_dir!r}; pass --comfy")
    saved_argv, sys.argv = sys.argv, sys.argv[:1]  # comfy.cli_args parses argv at import
    sys.path.insert(0, comfy_dir)
    try:
        import comfy.model_management  # noqa: F401
        import comfy.sd  # noqa: F401
        import comfy.utils  # noqa: F401
        import comfy_extras.nodes_seedvr as nodes_seedvr
    finally:
        sys.argv = saved_argv
    return nodes_seedvr


def bind_package(name, path):
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg


class Report:
    def __init__(self):
        self.failures = []

    def check(self, name, ours, native):
        native16 = native.to(torch.float16)
        if tuple(ours.shape) != tuple(native.shape):
            print(f"FAIL  {name}: shape {tuple(ours.shape)} vs {tuple(native.shape)}")
            self.failures.append(name)
            return
        diff = (ours.float() - native.float()).abs()
        exact16 = torch.equal(ours, native16)
        ok = exact16 or diff.max().item() <= FP16_EPS
        print(f"{'PASS' if ok else 'FAIL'}  {name}: shape {tuple(ours.shape)} max|diff|={diff.max().item():.3e} "
              f"mean|diff|={diff.mean().item():.3e} equal_after_fp16_round={exact16}")
        if not ok:
            self.failures.append(name)


def clip(frames, height, width, g, channels=3):
    coarse = torch.rand(frames, channels, height // 8 + 1, width // 8 + 1, generator=g)
    smooth = torch.nn.functional.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
    noise = torch.rand(frames, channels, height, width, generator=g) * 0.1
    return (smooth * 0.9 + noise).clamp(0, 1).permute(0, 2, 3, 1).contiguous()


def postprocess_cases(nodes_seedvr, port, report):
    native = nodes_seedvr.SeedVR2PostProcessing()
    g = torch.Generator().manual_seed(0)
    cases = [
        ("5 frames 64x96 vs 3 ref frames 64x96", clip(5, 64, 96, g), clip(3, 64, 96, g)),
        ("odd sizes: decoded 67x99, ref 65x97", clip(2, 67, 99, g), clip(2, 65, 97, g)),
        ("reference larger than decoded (resized down)", clip(1, 64, 96, g), clip(1, 80, 120, g)),
        ("alpha reference", clip(2, 64, 96, g), clip(2, 64, 96, g, channels=4)),
        ("float16 decoded input", clip(2, 64, 96, g).half(), clip(2, 64, 96, g)),
    ]
    for method in ("lab", "wavelet", "adain", "none"):
        for label, decoded, reference in cases:
            expected = native.execute(decoded.float(), reference, method).args[0]
            ours = port.process(decoded, reference, method)[0]
            report.check(f"postprocess {method}: {label}", ours, expected)


def decode_cases(vae_path, encode_port, port, frames, height, width, tile, overlap, report):
    import comfy.utils
    import comfy.sd
    import comfy.model_management as mm

    device = mm.get_torch_device()
    if device.type != "cuda":
        print("SKIP  decode: needs a CUDA device")
        return
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    g = torch.Generator().manual_seed(1)
    pixels = clip(frames, height, width, g).half()  # SeedVR2 Resize hands the encoder float16 frames

    label = f"{frames} frames {height}x{width} tile {tile}/{overlap}"
    torch.cuda.reset_peak_memory_stats(device)
    latent = vae.encode_tiled(pixels, tile_x=tile, tile_y=tile, overlap=overlap)  # VAEEncodeTiled(tile, overlap)
    native_peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(device)
    ours_latent = encode_port.encode(pixels, vae, tile, overlap)[0]["samples"]
    ours_peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    report.check(f"encode {label}", ours_latent, latent)
    print(f"      clip {frames}x{height}x{width} -> latent {tuple(latent.shape)}; peak VRAM during encode: native {native_peak:.2f} GiB, port {ours_peak:.2f} GiB")

    dec_overlap = tile // 4 if tile < overlap * 4 else overlap  # nodes.py VAEDecodeTiled
    torch.cuda.reset_peak_memory_stats(device)
    native = vae.decode_tiled(latent, tile_x=tile // 8, tile_y=tile // 8, overlap=dec_overlap // 8, tile_t=16, overlap_t=2)
    native = native.reshape(-1, native.shape[-3], native.shape[-2], native.shape[-1])
    native_peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    torch.cuda.reset_peak_memory_stats(device)
    ours = port.decode({"samples": latent}, vae, tile, overlap)[0]
    ours_peak = torch.cuda.max_memory_allocated(device) / 2 ** 30
    report.check(f"decode {label}", ours, native)
    print(f"      peak VRAM during decode: native {native_peak:.2f} GiB, port {ours_peak:.2f} GiB")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy", default=os.environ.get("COMFYUI_DIR", os.path.join(os.path.dirname(PKG_DIR), "ComfyUI")))
    p.add_argument("--vae", default=None, help="seedvr2 VAE safetensors; enables the decode checks (GPU)")
    p.add_argument("--frames", type=int, default=9)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--tile", type=int, default=1024)
    p.add_argument("--overlap", type=int, default=256)
    args = p.parse_args()

    nodes_seedvr = load_comfy(os.path.abspath(args.comfy))
    bind_package(BC_PKG, PKG_DIR)
    seedvr2 = importlib.import_module(f"{BC_PKG}.nodes.seedvr2")
    report = Report()
    print(f"ComfyUI: {os.path.abspath(args.comfy)}  torch {torch.__version__}")

    postprocess_cases(nodes_seedvr, seedvr2.SeedVR2PostProcess(), report)
    if args.vae:
        decode_cases(args.vae, seedvr2.SeedVR2VAEEncode(), seedvr2.SeedVR2VAEDecode(), args.frames, args.height, args.width, args.tile, args.overlap, report)
    else:
        print("SKIP  decode: pass --vae to run it")

    if report.failures:
        print(f"\nFAIL: {len(report.failures)} -> {report.failures}")
        sys.exit(1)
    print("\nOK")


if __name__ == "__main__":
    main()

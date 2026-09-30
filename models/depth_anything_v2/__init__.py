"""Depth Anything V2 Small: the vendored architecture (arch/, Apache-2.0), the weights
(weights.py), the model cache (loader.py) and the inference (inference.py).

Only the ViT-S checkpoint is supported: it is the only Depth Anything V2 size released under
Apache-2.0 (Base / Large / Giant are CC-BY-NC). One model, so no registry family. The weights
are the authors' .pth, fetched from Hugging Face through the pack's own downloader and loaded
with torch.load(weights_only=True) into a plain nn.Module — no transformers, no xFormers.
Everything beyond torch (folder_paths, comfy.model_management) is imported on the first run.
"""

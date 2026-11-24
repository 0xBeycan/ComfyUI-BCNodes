"""load(path) -> (VAE, note): a Wan 2.1 VAE file through core's comfy.sd.VAE, as core's Load VAE loads it.

A decoder whose last convolution gives image_channels x r^2 channels packs an r-times larger image into
them: Wan2.1-VAE-upscale2x gives 12 channels for a 2x image, and its model card unpacks them with
pixel_shuffle (decoder output [B, 12, F, H, W]). Core builds such a decoder (conv_out_channels) but
returns the packed channels as they are, so the loaded VAE is switched to a subclass whose decode and
decode_tiled unpack them: core's VAE Decode and VAE Decode (Tiled) then give the r-times larger image.
Its 3D tiled decode, which decode_tiled and decode's out-of-memory fallback run, keeps the packed
channels through the tiles. Encoding is untouched (the encoder takes image_channels). A plain Wan 2.1
VAE comes back as core loads it.
"""

import math
import os
from functools import cache

import torch.nn.functional as F


def load(path):
    import comfy.sd
    import comfy.utils

    sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    # as core's Load VAE: the reload factory a multi-GPU deep clone builds its per-device patcher from
    vae.patcher.cached_patcher_init = (comfy.sd.load_vae_patcher, (path, metadata, None))
    return prepare(vae, os.path.basename(path))


def prepare(vae, name):
    """A VAE core loaded -> (vae, note); its decoder's packed image unpacked on decode. ValueError when
    it is not a Wan 2.1 VAE or its decoder's channels hold no whole image."""
    conv_out = getattr(vae, "conv_out_channels", None)  # set by core for a Wan 2.1 layout only
    if conv_out is None or vae.latent_channels != 16:
        raise ValueError(f"{name} is not a Wan 2.1 VAE (Wan 2.1, Wan 2.2 A14B, Qwen-Image): pick its model, "
                         f"or load it with Load VAE.")
    channels = vae.output_channels
    if conv_out == channels:
        return vae, ""
    scale = math.isqrt(conv_out // channels)
    if scale < 2 or conv_out != channels * scale * scale:
        raise ValueError(f"{name}: its decoder gives {conv_out} channels, neither {channels} (a plain VAE) nor "
                         f"{channels} x a square (a packed larger image).")
    vae.__class__ = packed_vae_class(type(vae))
    vae.unpack_scale = scale
    return vae, f"the decoder packs a {scale}x image into {conv_out} channels: it decodes at {scale}x"


def unpack(images, scale):
    """Channels-last decoder output (..., H, W, C x scale^2) -> (..., H x scale, W x scale, C)."""
    return F.pixel_shuffle(images.movedim(-1, -3), scale).movedim(-3, -1)


@cache
def packed_vae_class(base):
    class PackedDecoderVAE(base):
        """A VAE whose decoder packs an unpack_scale-times larger image into its channels."""

        def decode(self, *args, **kwargs):
            return unpack(super().decode(*args, **kwargs), self.unpack_scale)

        def decode_tiled(self, *args, **kwargs):
            return unpack(super().decode_tiled(*args, **kwargs), self.unpack_scale)

        def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
            # core's own 3D tiled decode with the decoder's channel count, so the tiles keep the
            # packed channels whole; decode / decode_tiled unpack the result
            import comfy.utils

            decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())  # noqa: E731
            return self.process_output(comfy.utils.tiled_scale_multidim(
                samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio,
                out_channels=self.conv_out_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))

        def clone(self, *args, **kwargs):
            # core's clone builds a plain VAE: keep the unpacking on the copy
            copy = super().clone(*args, **kwargs)
            copy.__class__ = type(self)
            copy.unpack_scale = self.unpack_scale
            return copy

    return PackedDecoderVAE

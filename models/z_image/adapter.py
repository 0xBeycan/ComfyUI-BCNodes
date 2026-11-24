"""Z-Image's DyPE adapter (registered as "Z-Image" in the dype family): which models it takes, their values and
the embedder of one model call. The model is core's: comfy.supported_models.ZImage (Lumina 2's NextDiT at dim 3840,
the Flux VAE's latent at 8 px per cell), its rotary embedder at diffusion_model.rope_embedder (EmbedND) and the
RoPE length of an image axis in diffusion_model.axes_lens. Lumina 2, Ming-Image and the pixel-space Z-Image are
other model configs and are not taken."""

from .dype import DyPERope


class ZImageDyPE:
    name = "Z-Image"
    attribute = "rope_embedder"  # the diffusion model's rotary embedder, swapped for one call
    latent_scale = 8             # px per latent cell
    defaults = {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0}

    def matches(self, model):
        """Whether the ModelPatcher `model` is a Z-Image model."""
        from comfy.supported_models import ZImage

        return type(getattr(model.model, "model_config", None)) is ZImage

    def embedder(self, diffusion_model, method, scale, t, start_sigma, dype_scale, dype_exponent):
        """The rotary embedder of one call of `diffusion_model` (core's NextDiT), from the embedder it holds."""
        rope = getattr(diffusion_model, self.attribute)
        # both image axes take the longer of their two RoPE lengths (512 each on Z-Image), as the original
        return DyPERope(rope.theta, rope.axes_dim, max(diffusion_model.axes_lens[1:3]), method, scale, t, start_sigma,
                        dype_scale, dype_exponent)


ADAPTER = ZImageDyPE()

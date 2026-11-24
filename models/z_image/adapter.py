"""Z-Image's DyPE adapter (registered as "Z-Image" in the dype family): which models it takes, their values and
the embedder of one step. The model is core's: comfy.supported_models.ZImage (Lumina 2's NextDiT at dim 3840,
the Flux VAE's latent at 8 px per cell), its rotary embedder at diffusion_model.rope_embedder (EmbedND) and the
RoPE length of an image axis in diffusion_model.axes_lens. Lumina 2, Ming-Image and the pixel-space Z-Image are
other model configs and are not taken. Z-Image's ControlNets in core are model patches running inside the model
(they take its embedder's rotations): none carries an embedder of its own."""

from .dype import DyPERope


class ZImageDyPE:
    name = "Z-Image"
    attribute = "rope_embedder"  # the diffusion model's rotary embedder, swapped for one step
    latent_scale = 8             # px per latent cell
    # no yarn_alt_scaling: Z-Image's yarn has no alternative scaling (the original's Z-Image embedder ignores it)
    defaults = {"method": "vision_yarn", "base_resolution": 1024, "dype_scale": 2.0, "auto_threshold": 1920}

    def matches(self, model):
        """Whether the ModelPatcher `model` is a Z-Image model."""
        from comfy.supported_models import ZImage

        return type(getattr(model.model, "model_config", None)) is ZImage

    def own_embedder(self, control_model):
        return False

    def embedder(self, module, settings, height, width, scale, t, dype_exponent):
        """The rotary embedder of one step of `module` (core's NextDiT), from the embedder it holds."""
        rope = getattr(module, self.attribute)
        # both image axes take the longer of their two RoPE lengths (512 each on Z-Image), as the original
        return DyPERope(rope.theta, rope.axes_dim, max(module.axes_lens[1:3]), settings.method, scale, t,
                        settings.start_sigma, settings.dype_scale, dype_exponent)


ADAPTER = ZImageDyPE()

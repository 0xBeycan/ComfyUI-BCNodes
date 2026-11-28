"""Qwen-Image's DyPE adapter (registered as "Qwen-Image" in the dype family): which models it takes, their values
and the embedder of one call. The model is core's: comfy.supported_models.QwenImage (Qwen-Image and 2512; core's
config holds the Edit and Layered checkpoints too, which it cannot tell from them), its rotary embedder at
diffusion_model.pe_embedder (EmbedND), 2x2 patches of the Qwen VAE's latent at 8 px per cell. Qwen-Image 2.1, Krea-2
and Mage are other model configs and are not taken.

Of core's Qwen-Image ControlNets the InstantX one (comfy.ldm.qwen_image.controlnet.QwenImageControlNetModel) holds an
embedder of its own, replaced with the model's; the Fun ControlNet reads the diffusion model's and DiffSynth's runs
inside the model."""

from .dype import DyPERope


class QwenImageDyPE:
    name = "Qwen-Image"
    attribute = "pe_embedder"  # the rotary embedder of the diffusion model and of the InstantX ControlNet
    latent_scale = 8           # px per latent cell
    defaults = {"method": "vision_yarn", "base_resolution": 1328, "dype_scale": 2.0, "yarn_alt_scaling": False}

    def matches(self, model):
        """Whether the ModelPatcher `model` is a Qwen-Image model."""
        from comfy.supported_models import QwenImage

        return type(getattr(model.model, "model_config", None)) is QwenImage

    def own_embedder(self, control_model):
        from comfy.ldm.qwen_image.controlnet import QwenImageControlNetModel

        return isinstance(control_model, QwenImageControlNetModel)

    def embedder(self, module, rope, settings, height, width, scale, t, dype_exponent):
        """The rotary embedder of one call of `module` (core's QwenImageTransformer2DModel or its InstantX
        ControlNet), from `rope`, its own (core's EmbedND)."""
        base = settings.base_resolution
        # YaRN's original length: base_resolution in tokens (the original: base_resolution // 8 // 2)
        base_len = base // (self.latent_scale * module.patch_size)
        return DyPERope(rope.theta, rope.axes_dim, base_len, settings.method, (height / base, width / base), t,
                        settings.start_sigma, settings.dype_scale, dype_exponent, settings.yarn_alt_scaling)


ADAPTER = QwenImageDyPE()

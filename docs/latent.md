# Latent

## `BC_LatentUpscale` — Latent Upscale

`samples` (`LATENT`) + `model` + `upscale_model_name` (a weights file in `models/latent_upscale_models`) → `LATENT`, larger on height and width. The other fields of the latent (batch index, noise mask, …) are kept, as ComfyUI's own *Upscale Latent* keeps them; the sampler fits a noise mask to the new size itself.

| `model` | Latents | What it does |
| --- | --- | --- |
| Wan 2.1 2x | Wan 2.1, Wan 2.2 T2V / I2V A14B, Qwen-Image (16 channels, the Wan 2.1 VAE's latent space) | A small convolution network (2.6M parameters, float32) doubles the latent: the nearest-upscaled latent plus what the network adds, much closer to a real latent than bilinear or bislerp |

Use it between two sampling passes (hires fix): sample, *Latent Upscale*, sample again at a low denoise, decode. Without the second pass the decoder gets the upscaled latent as it is and adds no detail of its own.

An image latent (4D, e.g. *Empty SD3 Latent Image*'s) and a video latent (5D) both work. Frames are upscaled one by one (the convolutions are spatial); the network's channel attention pools each batch item whole, frames included, as it was trained. A latent with other than 16 channels (the Wan 2.2 TI2V-5B VAE's 48) stops the run with an error.

The weights file is yours to put in `models/latent_upscale_models`, ComfyUI's own folder for latent upscalers (Wan 2.1 2x: spacepxl's `wan21_latent_upscale_2x.safetensors`, 10 MB, under any name); nothing is downloaded. The file picked stays loaded until another is picked; the Process Monitor's full clear drops it. A file whose weights do not fit the model stops the run with an error naming it. The architecture is the author's (spacepxl, MIT), vendored in `models/wan_latent_upscale/arch/`.

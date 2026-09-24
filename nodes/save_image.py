"""Save Image node.

    BC_SaveImage (Save Image)

Saves an image batch under ComfyUI/output with folder and file names built
from prompt widget values (`sampler_name`, `13.cfg`, `ckpt_path`, `%F`,
`'fixed'`, `/subfolder`), a per-folder counter, an optional job JSON and the
prompt + workflow embedded per format (PNG text chunks; EXIF `Make` /
`ImageDescription` for WebP, AVIF, JXL, JPEG, JPEG 2000). With
`image_preview` on the images are listed in the queue / history gallery but
never drawn under the node (web/js/save_image.js), so the node keeps
whatever size it was given.

PIL and folder_paths are imported inside the methods; the optional AVIF and
JXL Pillow plugins are imported only when a file of that type is written.
"""

from datetime import datetime

from ..libs.files import COUNTER_POSITIONS
from ..libs.image import tensor_to_pil_u8
from ..libs.image_write import DEFAULT_QUALITY, output_extensions
from ..pipelines.save_image import save_batch

JOB_DATA_OPTIONS = ["disabled", "prompt", "basic, prompt", "basic, sampler, prompt", "basic, models, sampler, prompt"]


class SaveImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "multiline": False, "tooltip": "Fixed string the file name starts with"}),
                "filename_keys": ("STRING", {"default": "sampler_name, cfg, steps, %F %H-%M-%S", "multiline": True, "tooltip": (
                    "Comma separated keys appended to the file name, in this order. Any widget name of any node works "
                    "(`sampler_name, scheduler, cfg, denoise`, `ckpt_name`, `vae_name`, `model_name`); `13.cfg` picks node 13; "
                    "`resolution` is WxH; `%F %H-%M-%S` is a strftime format; `'text'` is a fixed string; `/sub` starts a subfolder.")}),
                "foldername_prefix": ("STRING", {"default": "", "multiline": False, "tooltip": "Fixed string the subfolder name starts with"}),
                "foldername_keys": ("STRING", {"default": "ckpt_name", "multiline": True, "tooltip": "Same rules as filename_keys; `/` or `../` make nested subfolders"}),
                "delimiter": ("STRING", {"default": "-", "multiline": False, "tooltip": "One character placed between the parts; `/` makes subfolders"}),
                "save_job_data": (JOB_DATA_OPTIONS, {"default": "disabled", "tooltip": "Append an entry per job to jobs.json in the subfolder: prompt texts, basic data, sampler settings, loaded models"}),
                "job_data_per_image": ("BOOLEAN", {"default": False, "tooltip": "One <image>.json per image instead of jobs.json"}),
                "job_custom_text": ("STRING", {"default": "", "multiline": False, "tooltip": "Free text saved with the job data"}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed prompt and workflow in the image (PNG text chunks, EXIF for the other formats)"}),
                "counter_digits": ("INT", {"default": 4, "min": 0, "max": 8, "step": 1, "display": "slider", "tooltip": (
                    "Digits of the image counter: 3 gives image_001.png. Continues from the highest counter in the subfolder; 0 disables it")}),
                "counter_position": (COUNTER_POSITIONS, {"default": "last", "tooltip": "image_001.png or 001_image.png"}),
                "one_counter_per_folder": ("BOOLEAN", {"default": True, "tooltip": "Unused"}),
                "image_preview": ("BOOLEAN", {"default": True, "tooltip": "List the saved images in the queue / history gallery. They are never drawn under the node"}),
                "output_ext": (output_extensions(), {"default": ".webp", "tooltip": "File format; AVIF and JXL appear when their Pillow plugins are installed"}),
                "quality": ("INT", {"default": DEFAULT_QUALITY, "min": 0, "max": 100, "step": 1, "display": "slider", "tooltip": (
                    "Encoder quality for JPEG / JXL / WebP / AVIF / JPEG 2000 (100 = lossless for WebP / AVIF / JXL); PNG maps it to compression level 0-9")}),
                "named_keys": ("BOOLEAN", {"default": False, "tooltip": "Prefix each value with its key: prefix-seed=123456-cfg=5.0-0001.webp"}),
            },
            "optional": {
                "positive_text_opt": ("STRING", {"forceInput": True, "tooltip": "Saved as positive_prompt in the job data"}),
                "negative_text_opt": ("STRING", {"forceInput": True, "tooltip": "Saved as negative_prompt in the job data"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "save image", "save image extended", "webp", "avif"]
    DESCRIPTION = (
        "Saves images under ComfyUI/output with folder and file names built from prompt values.\n"
        "\n"
        "Keys: any widget name (`sampler_name`, `cfg`, `ckpt_name`; the highest-numbered node wins), "
        "`13.cfg` for node 13, `ckpt_path` / `lora_path` / `control_net_path` for the model's folder, "
        "`resolution`, strftime formats (`%F` = 2024-05-22, `%H-%M-%S`), `'fixed text'`, and `/sub`, "
        "`./sub` or `../sub` to step into a folder.\n"
        "\n"
        "Prompt and workflow are embedded in every format except BMP: PNG text chunks, otherwise EXIF "
        "`Make` (prompt) and `ImageDescription` (workflow). ComfyUI loads PNG and WebP back."
    )

    def save_images(self, images, filename_prefix, filename_keys, foldername_prefix, foldername_keys, delimiter,
                    save_job_data, job_data_per_image, job_custom_text, save_metadata, counter_digits, counter_position,
                    one_counter_per_folder, image_preview, output_ext, quality, named_keys,
                    positive_text_opt=None, negative_text_opt=None, prompt=None, extra_pnginfo=None):
        import folder_paths
        from PIL import Image  # unused here: a missing Pillow must fail before the empty-batch guard

        if images is None or len(images) == 0:
            return {"ui": {"images": []}}
        if delimiter:
            delimiter = delimiter[0]
        output_dir = folder_paths.get_output_directory()
        frames = [tensor_to_pil_u8(image) for image in images]
        resolution = f"{frames[0].width}x{frames[0].height}"
        timestamp = datetime.now()

        results = save_batch(frames, output_dir, resolution, timestamp, filename_prefix, filename_keys, foldername_prefix,
                             foldername_keys, delimiter, save_job_data, job_data_per_image, job_custom_text, save_metadata,
                             counter_digits, counter_position, output_ext, quality, named_keys, positive_text_opt,
                             negative_text_opt, prompt, extra_pnginfo)
        return {"ui": {"images": results if image_preview else []}}


NODE_CLASS_MAPPINGS = {"BC_SaveImage": SaveImage}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_SaveImage": "Save Image"}

"""Save Image With Caption node.

    BC_SaveImageWithCaption (Save Image With Caption)

Saves an image batch as PNG files with ComfyUI's own naming (prefix_00001_.png) and, when a
caption is connected, the caption in a text file next to each image under the same name
(prefix_00001_.txt): the layout a training dataset needs. Small on purpose; Save Image is the node
with the name grammar, the formats and the job data.

`filename_prefix` takes the frontend's text replacements (%date:yyyy-MM-dd%, %Node.widget%;
web/js/save_image_with_caption.js) and ComfyUI's own (%year% ... %second%, %width%, %height%,
`sub/` folders), plus %batch_num%. folder_paths and comfy.cli_args are imported inside the method,
PIL inside tensor_to_pil_u8.
"""

from ..libs.image import tensor_to_pil_u8
from ..pipelines.save_image import CAPTION_EXTENSIONS, caption_extension, caption_folder, save_with_captions


class SaveImageWithCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save, as PNG."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": (
                    "The file name before the counter (prefix_00001_.png). May hold %date:yyyy-MM-dd% or "
                    "%Empty Latent Image.width% for values from nodes, %year% %month% %day% %hour% %minute% %second% "
                    "%width% %height%, %batch_num% (the image's index in the batch), and `sub/` folders.")}),
                "output_folder": ("STRING", {"default": "output", "tooltip": (
                    "Where to save: `output` is ComfyUI's output folder; `output/my_dataset` or `my_dataset` a folder "
                    "inside it; an absolute path any folder (created when missing).")}),
            },
            "optional": {
                "caption_file_extension": ("STRING", {"default": ".txt", "tooltip": (
                    f"The caption file's extension, a plain-text format: {', '.join(CAPTION_EXTENSIONS)}.")}),
                "caption": ("STRING", {"forceInput": True, "tooltip": (
                    "Saved next to each image under the image's name. Not connected: images only.")}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    OUTPUT_TOOLTIPS = ("The file name of the last image saved.",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/image"
    SEARCH_ALIASES = ["BCNodes", "save image", "caption", "dataset", "save caption"]
    DESCRIPTION = (
        "Saves images as PNG and, with a caption connected, the caption next to each image under the same "
        "name (image_00001_.png + image_00001_.txt), for training datasets.\n"
        "\n"
        "The counter continues from the files in the folder and never overwrites one. Prompt and workflow are "
        "embedded unless ComfyUI runs with --disable-metadata. Nothing is drawn under the node."
    )

    def save_images(self, images, filename_prefix, output_folder, caption_file_extension=".txt", caption=None,
                    prompt=None, extra_pnginfo=None):
        import folder_paths
        from comfy.cli_args import args

        extension = caption_extension(caption_file_extension) if caption is not None else None
        if images is None or len(images) == 0:
            return ("",)
        folder = caption_folder(output_folder, folder_paths.get_output_directory())
        # makes the folder; prefix sub/ folders stay inside it
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, folder, images[0].shape[1], images[0].shape[0])
        frames = (tensor_to_pil_u8(image) for image in images)
        return (save_with_captions(frames, folder, filename, counter, caption, extension, prompt, not args.disable_metadata,
                                   extra_pnginfo),)


NODE_CLASS_MAPPINGS = {"BC_SaveImageWithCaption": SaveImageWithCaption}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_SaveImageWithCaption": "Save Image With Caption"}

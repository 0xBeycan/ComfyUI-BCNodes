"""BC_ImageComparer (Image Comparer)

Two images, compared on the node with a draggable divider (web/js/comparer.js).
The images are written to ComfyUI's temp folder the way Preview Image does it
and handed to the frontend as a_images / b_images.
"""


class ImageComparer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "BCNodes/workflow"
    SEARCH_ALIASES = ['BCNodes', 'image comparer', 'compare images', 'before after']

    def compare(self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None):
        result = {"ui": {"a_images": [], "b_images": []}}
        previewer = None
        for key, images in (("a_images", image_a), ("b_images", image_b)):
            if images is None or len(images) == 0:
                continue
            if previewer is None:
                from nodes import PreviewImage

                previewer = PreviewImage()
            result["ui"][key] = previewer.save_images(images, "bcnodes.compare.", prompt, extra_pnginfo)["ui"]["images"]
        return result


NODE_CLASS_MAPPINGS = {
    "BC_ImageComparer": ImageComparer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_ImageComparer": "Image Comparer",
}

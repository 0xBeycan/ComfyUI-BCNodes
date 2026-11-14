"""BC_ImageComparer (Image Comparer)

Two images, compared on the node with a draggable divider (web/js/comparer.js).
The images are written to ComfyUI's temp folder the way Preview Image does it
and handed to the frontend as a_images / b_images.

The first input is shown as B (it fills the node), the second as A (drawn
from the left edge up to the pointer). The input keys keep their old names
(image_a first) so saved workflows keep their links; only the labels follow
the sides.
"""


class ImageComparer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE", {"display_name": "image_b"}),
                "image_b": ("IMAGE", {"display_name": "image_a"}),
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
        for key, images in (("b_images", image_a), ("a_images", image_b)):
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

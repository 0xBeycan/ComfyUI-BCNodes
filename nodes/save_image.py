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

import importlib.util
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")
COUNTER_POSITIONS = ["last", "first"]
BASE_EXTENSIONS = [".webp", ".png", ".jpg", ".jpeg", ".j2k", ".jp2", ".gif", ".tiff", ".bmp"]
JOB_DATA_OPTIONS = ["disabled", "prompt", "basic, prompt", "basic, sampler, prompt", "basic, models, sampler, prompt"]
DEFAULT_QUALITY = 90
# jpeg / webp / avif / jxl / tiff only; PNG is left out because Pillow then
# compresses at level 9 regardless of compress_level.
OPTIMIZE = True

# `x_path` keys need the matching `x_name` widget: the path is its parent dir.
PATH_KEYS = {"ckpt_path": "ckpt_name", "control_net_path": "control_net_name", "lora_path": "lora_name"}
NAME_KEYS = {name: path for path, name in PATH_KEYS.items()}
PLUGINS = {".avif": "pillow_avif", ".jxl": "pillow_jxl"}


def output_extensions():
    exts = list(BASE_EXTENSIONS)
    for ext, module in PLUGINS.items():
        if importlib.util.find_spec(module) is not None:
            exts.insert(0, ext)
    return exts


def strip_model_extension(value):
    if isinstance(value, str):
        for ext in MODEL_EXTENSIONS:
            value = value.removesuffix(ext)
    return value


def find_keys(prompt, wanted, found):
    """Walks `prompt` (a node, or the whole prompt) and records the value of
    every widget named in `wanted` into `found`. Later matches overwrite
    earlier ones, so with the whole prompt the highest node id wins. `x_path`
    and `x_name` come from the same widget; `x_path` takes precedence."""
    for key, value in prompt.items():
        if key in wanted:
            if isinstance(value, dict):
                value = value.get("content", "")  # checkpoint-with-thumbnail widget form
            if key in NAME_KEYS:
                path_key = NAME_KEYS[key]
                if path_key in wanted:
                    found[path_key] = strip_model_extension(str(Path(value).parent))
                else:
                    found[key] = strip_model_extension(Path(value).name)
            else:
                found[key] = strip_model_extension(value)
        elif isinstance(value, dict):
            find_keys(value, wanted, found)


def find_parameter_values(target_keys, prompt, found=None):
    """The job JSON's flat lookup: last occurrence of each target key anywhere
    in the prompt, plus every `lora*` widget joined as `loras`."""
    if found is None:
        found = {}
    loras = []
    for key, value in prompt.items():
        if "loras" in target_keys and re.match(r"lora(_name)?(_\d+)?", key) and value is not None:
            loras.append(str(strip_model_extension(value)))
        if isinstance(value, dict):
            find_parameter_values(target_keys, value, found)
        if key in target_keys:
            found[key] = strip_model_extension(value)
    if loras:
        found["loras"] = ", ".join(loras)
    if len(target_keys) == 1:
        return found.get(target_keys[0])
    return found


def custom_name(keys, prefix, delimiter, prompt, resolution, timestamp, named_keys):
    """Builds a folder or file name from `keys`, each one of:

        %F %H-%M-%S     strftime format
        'fixed'         quoted fixed string
        /sub or ../sub  a subfolder step before the key (trailing / dropped)
        13.cfg          widget `cfg` of node 13 (falls back to any node)
        cfg             widget `cfg` of the highest-numbered node that has it
        resolution      WxH of the first image
        ckpt_path       parent folder of the checkpoint widget
        anything else   kept as a literal
    """
    parts = []
    if prefix:
        parts.append(timestamp.strftime(prefix) if "%" in prefix else prefix)

    if prompt is not None and keys != [""]:
        found = {}
        for key in keys:
            if not key:
                continue
            value = None
            node_key = None

            if "%" in key:
                value = timestamp.strftime(key)

            if "/" in key:
                pieces = re.split("/+", key)
                if pieces[0] in ("", ".", ".."):
                    parts.append(pieces[0] + "/")
                    key = pieces[1]
                    if not key:
                        continue
                else:
                    key = pieces[0]

            if (key.startswith("'") and key.endswith("'")) or (key.startswith('"') and key.endswith('"')):
                value = key

            if value is None:
                split = key.split(".")
                if len(split) == 2:
                    if "" not in split:
                        if split[0].isdecimal():
                            node, node_key = split
                            scope = prompt[node] if node in prompt else prompt
                            if node not in prompt:
                                print(f"[BCNodes] Save Image: node #{node} not in the prompt, looking for `{node_key}` in every node")
                            find_keys(scope, wanted_keys(node_key), found)
                        else:
                            value = strip_model_extension(key)  # string.string: a literal
                    else:
                        value = key  # ".string", "string." or "."
                else:
                    node_key = key
                    if node_key == "resolution":
                        value = resolution
                    else:
                        find_keys(prompt, wanted_keys(node_key), found)

            if value is None and node_key is not None:
                if node_key in found:
                    value = f"{node_key}={found[node_key]}" if named_keys else found[node_key]
                else:
                    value = node_key

            if isinstance(value, str):
                if node_key not in PATH_KEYS:
                    value = strip_model_extension(value)
            elif isinstance(value, float):
                value = float(f"{value:.10g}")
            parts.append(str(value))

    parts = [p.strip() for p in parts if p]
    name = delimiter.join(parts).replace("/" + delimiter, "/").replace(delimiter + "/", "/").replace(delimiter + ".", ".")
    name = re.sub(r"\s+", " ", name).strip(delimiter).strip("/").strip(delimiter).rstrip(".")
    return re.sub(r'[*?:"<>|]', "", name).replace("/", os.sep)


def wanted_keys(node_key):
    if node_key in PATH_KEYS:
        return [PATH_KEYS[node_key], node_key]
    return [node_key]


def latest_counter(folder, filename, counter_digits, counter_position, output_ext):
    """1 + the highest counter among the files in `folder` that carry
    `filename` and `output_ext`; 1 when there are none. Gaps are ignored."""
    if not os.path.isdir(folder):
        return 1
    files = [f for f in os.listdir(folder) if f.endswith(output_ext)]
    if not files:
        return 1
    if counter_position not in COUNTER_POSITIONS:
        counter_position = COUNTER_POSITIONS[0]
    ext_len = len(output_ext)
    if not filename:
        counters = [int(f[:counter_digits]) if f[:counter_digits].isdecimal() and len(f) == counter_digits + ext_len else 0
                    for f in files]
    elif counter_position == "last":
        counters = [int(f[-(ext_len + counter_digits):-ext_len]) if f[-(ext_len + counter_digits):-ext_len].isdecimal() else 0
                    for f in files if f.startswith(filename)]
    else:
        counters = [int(f[:counter_digits]) if f[:counter_digits].isdecimal() else 0
                    for f in files if f[counter_digits + 1:].startswith(filename)]
    return max(counters) + 1 if counters else 1


def png_info(prompt, extra_pnginfo):
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    if prompt is not None:
        info.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for key, value in extra_pnginfo.items():
            info.add_text(key, json.dumps(value))
    return info


def exif_bytes(img, prompt, extra_pnginfo):
    """Prompt in `Make` (0x010f), workflow in `ImageDescription` (0x010e):
    two IFD0 tags close together, which is what ComfyUI's pnginfo reader
    parses back on load for WebP."""
    exif = img.getexif()
    exif[0x010F] = "Prompt: " + json.dumps(prompt if prompt is not None else {})
    exif[0x010E] = "Workflow: " + json.dumps((extra_pnginfo or {}).get("workflow", {}))
    return exif.tobytes()


def write_image(path, img, prompt, save_metadata, extra_pnginfo, quality):
    ext = Path(path).suffix
    if ext in PLUGINS:
        importlib.import_module(PLUGINS[ext])  # registers the Pillow codec
    if quality == 0:
        quality = DEFAULT_QUALITY
    kwargs = {}
    if ext in (".avif", ".webp", ".jxl"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        if quality == 100:
            kwargs["lossless"] = True
        else:
            kwargs["quality"] = quality
        kwargs["optimize"] = OPTIMIZE
    elif ext in (".j2k", ".jp2", ".jpc", ".jpf", ".jpx", ".j2c"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        if quality < 100:
            kwargs["irreversible"] = True
        else:
            kwargs["quality"] = quality
    elif ext in (".jpg", ".jpeg"):
        if save_metadata:
            kwargs["exif"] = exif_bytes(img, prompt, extra_pnginfo)
        kwargs["subsampling"] = 0
        kwargs["quality"] = quality
        kwargs["optimize"] = OPTIMIZE
    elif ext == ".tiff":
        kwargs["optimize"] = OPTIMIZE
    elif ext in (".png", ".gif"):
        if save_metadata:
            kwargs["pnginfo"] = png_info(prompt, extra_pnginfo)
        # quality 0-90 -> compress_level 0-9; above 90 buys nothing but time
        kwargs["compress_level"] = round(min(quality, 90) / 10)
    img.save(path, **kwargs)


def prompt_texts(prompt):
    """positive / negative prompt text from the prompt when nothing was wired
    into the `*_text_opt` inputs: Efficiency Loader widgets, else the `text`
    widget behind a KSampler's positive / negative links."""
    texts = {}
    if prompt is None:
        return texts

    def is_link(value):
        return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and len(value[0]) < 6 \
            and isinstance(value[1], (int, float))

    for node in prompt.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if class_type in ("Efficient Loader", "Eff. Loader SDXL"):
            if "positive" in inputs and "negative" in inputs:
                texts["positive_prompt"] = inputs["positive"]
                texts["negative_prompt"] = inputs["negative"]
        elif class_type in ("KSampler", "KSamplerAdvanced", "UltimateSDUpscale"):
            for side in ("positive", "negative"):
                ref = inputs.get(side, [None])[0] if side in inputs else None
                text = prompt.get(str(ref), {}).get("inputs", {}).get("text")
                if text is not None and not is_link(text):
                    texts[f"{side}_prompt"] = text
    return texts


def save_job(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text, resolution,
             folder, filename, timestamp):
    entry = {}
    if "basic" in save_job_data:
        if filename_prefix:
            entry["filename_prefix"] = filename_prefix
        entry["resolution"] = resolution
    if job_custom_text:
        entry["custom_text"] = job_custom_text
    if "models" in save_job_data:
        models = find_parameter_values(["ckpt_name", "loras", "vae_name", "model_name"], prompt or {})
        for out_key, key in (("checkpoint", "ckpt_name"), ("loras", "loras"), ("vae", "vae_name"), ("upscale_model", "model_name")):
            if models.get(key):
                entry[out_key] = models[key]
    if "sampler" in save_job_data:
        entry["sampler_parameters"] = find_parameter_values(["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"], prompt or {})
    if "prompt" in save_job_data:
        if positive_text_opt is not None:
            entry["positive_prompt"] = positive_text_opt
        if negative_text_opt is not None:
            entry["negative_prompt"] = negative_text_opt
        if positive_text_opt is None and negative_text_opt is None:
            entry.update(prompt_texts(prompt))

    path = os.path.join(folder, filename)
    existing = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                print(f"[BCNodes] Save Image: {path} is not valid JSON, starting it over")
    existing[timestamp.strftime("%c")] = entry
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=4)


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
        from PIL import Image

        if images is None or len(images) == 0:
            return {"ui": {"images": []}}
        if quality == 0:
            quality = DEFAULT_QUALITY
        if delimiter:
            delimiter = delimiter[0]
        output_dir = folder_paths.get_output_directory()
        frames = [Image.fromarray(np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in images]
        resolution = f"{frames[0].width}x{frames[0].height}"
        timestamp = datetime.now()

        folder_name = custom_name([k.strip() for k in foldername_keys.split(",")], foldername_prefix, delimiter, prompt,
                                  resolution, timestamp, named_keys)
        file_name = custom_name([k.strip() for k in filename_keys.split(",")], filename_prefix, delimiter, prompt,
                                resolution, timestamp, named_keys)
        if file_name:
            target = Path(os.path.join(output_dir, folder_name, file_name))
            folder, filename = target.parent, target.name
        else:
            folder, filename = Path(os.path.join(output_dir, folder_name)), ""
        os.makedirs(folder, exist_ok=True)
        counter = latest_counter(folder, filename, counter_digits, counter_position, output_ext)

        results = []
        for img in frames:
            if counter_digits > 0:
                if not filename:
                    image_name = f"{counter:0{counter_digits}}{output_ext}"
                elif counter_position == "last":
                    image_name = f"{filename}{delimiter}{counter:0{counter_digits}}{output_ext}"
                else:
                    image_name = f"{counter:0{counter_digits}}{delimiter}{filename}{output_ext}"
            else:
                image_name = f"{filename or filename_prefix}{output_ext}"
            image_path = os.path.join(folder, image_name)
            write_image(image_path, img, prompt, save_metadata, extra_pnginfo, quality)
            if save_job_data != "disabled" and job_data_per_image:
                save_job(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text,
                         resolution, folder, f"{image_name.removesuffix(output_ext)}.json", timestamp)
            subfolder = os.path.relpath(folder, output_dir)
            results.append({"filename": image_name, "subfolder": "" if subfolder == "." else subfolder, "type": "output"})
            counter += 1

        if save_job_data != "disabled" and not job_data_per_image:
            save_job(save_job_data, prompt, filename_prefix, positive_text_opt, negative_text_opt, job_custom_text,
                     resolution, folder, "jobs.json", timestamp)
        return {"ui": {"images": results if image_preview else []}}


NODE_CLASS_MAPPINGS = {"BC_SaveImage": SaveImage}
NODE_DISPLAY_NAME_MAPPINGS = {"BC_SaveImage": "Save Image"}

"""Save Image flow: folder and file names built from the prompt's widget values (custom_name
and its key grammar), the job JSON (save_job, prompt_texts, find_parameter_values) and the save
loop (save_batch: target folder, counter, image names, one write_image per frame, job data,
gallery entries). Takes PIL frames and plain values; the node keeps folder_paths, the clock and
the ui payload.
"""

import json
import os
import re
from pathlib import Path

from ..libs.files import latest_counter
from ..libs.image_write import write_image

MODEL_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")

# `x_path` keys need the matching `x_name` widget: the path is its parent dir.
PATH_KEYS = {"ckpt_path": "ckpt_name", "control_net_path": "control_net_name", "lora_path": "lora_name"}
NAME_KEYS = {name: path for path, name in PATH_KEYS.items()}


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


def save_batch(frames, output_dir, resolution, timestamp, filename_prefix, filename_keys, foldername_prefix,
               foldername_keys, delimiter, save_job_data, job_data_per_image, job_custom_text, save_metadata,
               counter_digits, counter_position, output_ext, quality, named_keys, positive_text_opt, negative_text_opt,
               prompt, extra_pnginfo):
    """Writes `frames` (PIL images) under output_dir/<folder name>/, the counter continued from
    the files already there, plus the job JSON when asked; returns the gallery entries
    ({filename, subfolder, type}) in frame order."""
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
    return results

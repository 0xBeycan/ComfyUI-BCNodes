"""Golden: Save Image's PROMPT interpretation and job JSON (target pipelines/save_image.py).

- custom_name over an extended fake prompt (placeholder model names): every key form of its
  docstring (strftime, quoted literals, /sub ./sub ../sub, 13.cfg with the node present and
  missing, plain widget keys where the highest node id wins, resolution, x_path / x_name,
  string.string literals, ".x" / "x." / ".", floats through :.10g, the thumbnail-widget dict),
  named_keys, delimiters, illegal characters, an empty key list and no prompt; each with the
  stdout it printed (the "node #... not in the prompt" line).
- find_parameter_values: the job JSON's flat lookup, incl. the quirks: two LoRA nodes (the last
  one wins), lora_strength collected by the lora prefix regex, a Power Lora row dict written as
  str(dict), nested dicts, a single target key.
- prompt_texts: Efficient Loader widgets, KSampler / KSamplerAdvanced / UltimateSDUpscale links,
  a text that is itself a link, a missing linked node, no prompt.
- save_job: the JSON written for all 5 save_job_data options, with and without the text inputs,
  appended to an existing file, and over a garbage jobs.json (stdout with the tmp dir replaced
  by /path/to/output).
"""

import json
import os
from datetime import datetime

import pytest

from _golden import Where, check, check_env, digest

ENV = {
    'platform': 'darwin-arm64',
    'python': '3.12.11',
}

GOLDEN = {
    'custom_name/defaults_file': "('ComfyUI-euler-4.123456789-30-2024-05-22 09-13-58', '')",
    'custom_name/defaults_folder': "('model_b', '')",
    'custom_name/highest_node_wins': "('4.123456789-model_b-lora_b', '')",
    'custom_name/node_scoped': "('7.0-4.123456789-3.5-123456', '')",
    'custom_name/node_missing': "('4.123456789-nope', '[BCNodes] Save Image: node #99 not in the prompt, looking for `cfg` in every node\\n[BCNodes] Save Image: node #98 not in the prompt, looking for `nope` in every node\\n')",
    'custom_name/strftime_and_literals': '("240522-2024-09-13-\'fixed\'-double-\'ab\'", \'\')',
    'custom_name/subfolders': "('run/euler./30../123456-sub-trail', '')",
    'custom_name/subfolder_only': "('run/', '')",
    'custom_name/paths': "('family_b-model_b.-cn-controlnet_a', '')",
    'custom_name/paths_named': "('ckpt_path=family_b_lora_name=lora_b_control_net_path=cn', '')",
    'custom_name/dotted_literals': "('a.b-file.x-x..-5-3', '[BCNodes] Save Image: node #1 not in the prompt, looking for `5` in every node\\n')",
    'custom_name/resolution_and_unknown': "('12x8-nope-a.b-xyzwv', '')",
    'custom_name/named_keys': '("p-seed=123456-cfg=4.123456789012345-denoise=1.0-12x8-\'q\'", \'\')',
    'custom_name/delimiter_slash': "('pre/euler/30', '')",
    'custom_name/delimiter_dot': "('pre.euler.30', '')",
    'custom_name/spaces_and_empty': "('spaced prefix-euler-normal', '')",
    'custom_name/empty_keys': "('only_prefix', '')",
    'custom_name/no_prompt': "('prefix', '')",
    'custom_name/no_prefix_no_keys': "('', '')",
    'find_parameter_values/models': "{'ckpt_name': {'content': 'family_b/model_b.ckpt', 'image': 'model_b.png'}, 'loras': 'lora_b, 0.6', 'vae_name': 'vae_a', 'model_name': 'upscaler_a'}",
    'find_parameter_values/sampler': "{'seed': 123456, 'steps': 30, 'cfg': 4.123456789012345, 'sampler_name': 'euler', 'scheduler': 'normal', 'denoise': 1.0}",
    'find_parameter_values/single_key': "'euler'",
    'find_parameter_values/single_key_missing': 'None',
    'find_parameter_values/loras_only': "'lora_b, 0.6'",
    'find_parameter_values/power_lora_rows': '{\'loras\': "{\'on\': True, \'lora\': \'lora_a.safetensors\', \'strength\': 1.0}, {\'on\': False, \'lora\': \'lora_b.safetensors\', \'strength\': 0.5}, not_a_lora", \'ckpt_name\': \'model_c\'}',
    'find_parameter_values/nested': "{'cfg': 1.5, 'loras': 'deep'}",
    'find_parameter_values/empty_prompt': '{}',
    'prompt_texts/ksampler_links': "{'positive_prompt': '<trigger>, a person, studio light', 'negative_prompt': 'blurry, lowres'}",
    'prompt_texts/efficient_loader': "{'positive_prompt': '<trigger>, a person', 'negative_prompt': 'lowres'}",
    'prompt_texts/text_is_a_link': "{'negative_prompt': ['long_id', 0]}",
    'prompt_texts/missing_linked_node': '{}',
    'prompt_texts/float_link_index': "{'positive_prompt': 'positive text'}",
    'prompt_texts/no_class_type': '{}',
    'prompt_texts/none': '{}',
    'save_job/disabled/no_text_inputs': '1285f81a3806c704bd104d39856f9635',
    'save_job/disabled/text_inputs': '81e8fa652036c451abd8cdca53b02f6d',
    'save_job/disabled/positive_only': '1285f81a3806c704bd104d39856f9635',
    'save_job/prompt/no_text_inputs': '690655d44de99c380c94d9348ad3ec0c',
    'save_job/prompt/text_inputs': 'efc72844035ab55dea506c0521957beb',
    'save_job/prompt/positive_only': '0f7c4bef293bf5d715a6d03c8dedf639',
    'save_job/basic, prompt/no_text_inputs': '309abb851489dacabf572bc59ede1662',
    'save_job/basic, prompt/text_inputs': '59cba62ba43ada147fa1333343b2f28d',
    'save_job/basic, prompt/positive_only': '4d18217eff12349c0326fe4fe53728df',
    'save_job/basic, sampler, prompt/no_text_inputs': '5e8569466c01336ab67cb3c1be532afe',
    'save_job/basic, sampler, prompt/text_inputs': 'f8205aeec941e9ee4e94245f4ed46800',
    'save_job/basic, sampler, prompt/positive_only': 'c30862ee266dc8fe2d3e4d795f51c230',
    'save_job/basic, models, sampler, prompt/no_text_inputs': '810525955ea531a075bd53f5f4ac391f',
    'save_job/basic, models, sampler, prompt/text_inputs': 'e9d32b0334b5446f7f9051b76c7e20f3',
    'save_job/basic, models, sampler, prompt/positive_only': 'bbc4e623c6f0a5ea94be28681ab59a21',
    "save_job_over_garbage/b'{not json'": '(b\'{\\n    "Wed May 22 09:13:58 2024": {\\n        "filename_prefix": "shot",\\n        "resolution": "12x8"\\n    }\\n}\', \'[BCNodes] Save Image: /path/to/output/sub/img.json is not valid JSON, starting it over\\n\')',
    "save_job_over_garbage/b''": '(b\'{\\n    "Wed May 22 09:13:58 2024": {\\n        "filename_prefix": "shot",\\n        "resolution": "12x8"\\n    }\\n}\', \'[BCNodes] Save Image: /path/to/output/sub/img.json is not valid JSON, starting it over\\n\')',
}

WHERE = Where({
    "custom_name": "pipelines.save_image:custom_name",
    "find_parameter_values": "pipelines.save_image:find_parameter_values",
    "prompt_texts": "pipelines.save_image:prompt_texts",
    "save_job": "pipelines.save_image:save_job",
})

TIMESTAMP = datetime(2024, 5, 22, 9, 13, 58)
LATER = datetime(2024, 5, 23, 18, 1, 2)

PROMPT = {
    "3": {"class_type": "KSampler", "inputs": {
        "seed": 123456, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0,
        "positive": ["6", 0], "negative": ["7", 0], "model": ["10", 0]}},
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "family_a/model_a.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "<trigger>, a person, studio light", "clip": ["4", 1]}},
    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, lowres", "clip": ["4", 1]}},
    "9": {"class_type": "LoraLoader", "inputs": {"lora_name": "styles/lora_a.safetensors", "strength_model": 0.8,
                                                  "cfg": 3.5, "model": ["4", 0]}},
    "10": {"class_type": "LoraLoader", "inputs": {"lora_name": "lora_b.pt", "lora_strength": 0.6, "model": ["9", 0]}},
    "11": {"class_type": "VAELoader", "inputs": {"vae_name": "vae_a.safetensors"}},
    "12": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "upscaler_a.pth"}},
    "13": {"class_type": "KSamplerAdvanced", "inputs": {"cfg": 4.123456789012345, "noise_seed": 7, "steps": 30,
                                                         "positive": ["6", 0], "negative": ["7", 0]}},
    "14": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "cn/controlnet_a.safetensors"}},
    "15": {"class_type": "CheckpointLoader|thumbnail", "inputs": {
        "ckpt_name": {"content": "family_b/model_b.ckpt", "image": "model_b.png"}}},
    "16": {"class_type": "Note", "inputs": {"label": "a.b", "empty": "", "weird": "x*y?z:<w>|\"v\""}},
}

# name -> (keys as typed in the widget, prefix, delimiter, named_keys, prompt: "full" | "none")
NAME_CASES = {
    "defaults_file": ("sampler_name, cfg, steps, %F %H-%M-%S", "ComfyUI", "-", False, "full"),
    "defaults_folder": ("ckpt_name", "", "-", False, "full"),
    "highest_node_wins": ("cfg, ckpt_name, lora_name", "", "-", False, "full"),
    "node_scoped": ("3.cfg, 13.cfg, 9.cfg, 3.seed", "", "-", False, "full"),
    "node_missing": ("99.cfg, 98.nope", "", "-", False, "full"),
    "strftime_and_literals": ("%Y, %H-%M, 'fixed', \"double\", 'a:b?'", "%y%m%d", "-", False, "full"),
    "subfolders": ("/sampler_name, ./steps, ../seed, sub//cfg, trail/", "run", "-", False, "full"),
    "subfolder_only": ("/, ./", "run", "-", False, "full"),
    "paths": ("ckpt_path, ckpt_name, lora_path, control_net_path, control_net_name", "", "-", False, "full"),
    "paths_named": ("ckpt_path, lora_name, control_net_path", "", "_", True, "full"),
    "dotted_literals": ("a.b, file.safetensors, .x, x., ., 1.5, 3.", "", "-", False, "full"),
    "resolution_and_unknown": ("resolution, nope, label, empty, weird", "", "-", False, "full"),
    "named_keys": ("seed, 13.cfg, denoise, resolution, 'q'", "p", "-", True, "full"),
    "delimiter_slash": ("sampler_name, steps", "pre", "/", False, "full"),
    "delimiter_dot": ("sampler_name, steps", "pre", ".", False, "full"),
    "spaces_and_empty": ("  sampler_name ,, scheduler  ,", " spaced   prefix ", "-", False, "full"),
    "empty_keys": ("", "only_prefix", "-", False, "full"),
    "no_prompt": ("cfg, 'x', resolution", "prefix", "-", False, "none"),
    "no_prefix_no_keys": ("", "", "-", False, "full"),
}


@pytest.mark.parametrize("name", list(NAME_CASES))
def test_custom_name(name, bcnodes, capsys):
    check_env(ENV)
    text, prefix, delimiter, named, prompt = NAME_CASES[name]
    keys = [k.strip() for k in text.split(",")]
    capsys.readouterr()
    value = WHERE["custom_name"](keys, prefix, delimiter, PROMPT if prompt == "full" else None, "12x8", TIMESTAMP, named)
    check(GOLDEN, f"custom_name/{name}", repr((value, capsys.readouterr().out)))


# name -> (target keys, prompt)
PARAMETER_CASES = {
    "models": (["ckpt_name", "loras", "vae_name", "model_name"], PROMPT),
    "sampler": (["seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"], PROMPT),
    "single_key": (["sampler_name"], PROMPT),
    "single_key_missing": (["nope"], PROMPT),
    "loras_only": (["loras"], PROMPT),
    "power_lora_rows": (["loras", "ckpt_name"], {
        "20": {"class_type": "Power Lora Loader", "inputs": {
            "lora_1": {"on": True, "lora": "lora_a.safetensors", "strength": 1.0},
            "lora_2": {"on": False, "lora": "lora_b.safetensors", "strength": 0.5},
            "lora_name": None,
            "loraish": "not_a_lora.safetensors",
            "ckpt_name": "model_c.safetensors"}}}),
    "nested": (["cfg", "loras"], {"a": {"b": {"cfg": 2.5, "lora_name": "deep.safetensors"}}, "cfg": 1.5}),
    "empty_prompt": (["ckpt_name", "loras"], {}),
}


@pytest.mark.parametrize("name", list(PARAMETER_CASES))
def test_find_parameter_values(name, bcnodes):
    check_env(ENV)
    keys, prompt = PARAMETER_CASES[name]
    check(GOLDEN, f"find_parameter_values/{name}", repr(WHERE["find_parameter_values"](keys, prompt)))


PROMPT_TEXT_CASES = {
    "ksampler_links": PROMPT,
    "efficient_loader": {"1": {"class_type": "Efficient Loader", "inputs": {"positive": "<trigger>, a person", "negative": "lowres"}},
                         "2": {"class_type": "Eff. Loader SDXL", "inputs": {"positive": "only positive"}}},
    "text_is_a_link": {"3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0], "negative": ["7", 0]}},
                       "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ["8", 0]}},
                       "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ["long_id", 0]}}},
    "missing_linked_node": {"3": {"class_type": "UltimateSDUpscale", "inputs": {"positive": ["60", 0]}}},
    "float_link_index": {"3": {"class_type": "KSampler", "inputs": {"positive": ["6", 0.0], "negative": ["7", 1]}},
                         "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive text"}},
                         "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ["7", 1.5]}}},
    "no_class_type": {"1": {"inputs": {"text": "orphan"}}},
    "none": None,
}


@pytest.mark.parametrize("name", list(PROMPT_TEXT_CASES))
def test_prompt_texts(name, bcnodes):
    check_env(ENV)
    check(GOLDEN, f"prompt_texts/{name}", repr(WHERE["prompt_texts"](PROMPT_TEXT_CASES[name])))


JOB_OPTIONS = ["disabled", "prompt", "basic, prompt", "basic, sampler, prompt", "basic, models, sampler, prompt"]
# name -> (positive_text_opt, negative_text_opt, job_custom_text, filename_prefix)
TEXT_INPUTS = {
    "no_text_inputs": (None, None, "", "ComfyUI"),
    "text_inputs": ("wired <trigger> text", "wired negative", "a note", ""),
    "positive_only": ("wired <trigger> text", None, "", "shot"),
}


def _read(path):
    with open(path, "rb") as f:
        return f.read()


@pytest.mark.parametrize("texts", list(TEXT_INPUTS))
@pytest.mark.parametrize("option", JOB_OPTIONS)
def test_save_job(option, texts, bcnodes, tmp_path, capsys):
    check_env(ENV)
    positive, negative, custom, prefix = TEXT_INPUTS[texts]
    folder = str(tmp_path)
    capsys.readouterr()
    WHERE["save_job"](option, PROMPT, prefix, positive, negative, custom, "12x8", folder, "jobs.json", TIMESTAMP)
    first = _read(os.path.join(folder, "jobs.json"))
    WHERE["save_job"](option, PROMPT, prefix, positive, negative, custom, "12x8", folder, "jobs.json", LATER)
    second = _read(os.path.join(folder, "jobs.json"))
    check(GOLDEN, f"save_job/{option}/{texts}", digest({"first": first, "second": second, "stdout": capsys.readouterr().out}))


@pytest.mark.parametrize("content", [b"{not json", b""])
def test_save_job_over_garbage(content, bcnodes, tmp_path, capsys):
    check_env(ENV)
    folder = tmp_path / "output" / "sub"
    folder.mkdir(parents=True)
    (folder / "img.json").write_bytes(content)
    capsys.readouterr()
    WHERE["save_job"]("basic, prompt", None, "shot", None, None, "", "12x8", str(folder), "img.json", TIMESTAMP)
    stdout = capsys.readouterr().out.replace(str(tmp_path / "output"), "/path/to/output")
    data = (folder / "img.json").read_bytes()
    assert json.loads(data)
    check(GOLDEN, f"save_job_over_garbage/{content!r}", repr((data, stdout)))

"""The LM catalog: a family's built-in models.yaml merged with the user's models.yaml files.

Schema, shared by the built-in file and the user files:

    models:
      <display name>:
        template: <one of the family's template ids>
        thinking: true | false             # the model has a thinking mode
        mtp: true | false                  # the file carries an MTP head
        backend: <backend name>            # optional; the family's default backend when absent
        defaults:                          # sampling per mode; thinking_on only on a thinking model
          thinking_off: {do_sample, temperature, top_p, top_k, min_p, presence_penalty, repetition_penalty, mtp}
          thinking_on: {...}
        precisions:                        # in the order the precision widget lists them
          <precision name>: {file: <file name>, url: <download URL>}

A built-in precision has a url; a user one may leave it out (the file must then be in the folder). A user
entry for a model the catalog already has overrides it: each field it gives replaces that field (defaults
per mode and per sampling field), each precision it names replaces that precision or is added. One console
line per overridden model names what changed. A user entry for a new model gives every field but backend,
complete in its own file. User files apply in the order given, a later one over an earlier one. `mtp: off`
reaches us as YAML's false and is read as 'off'.

yaml is imported inside load_catalog.
"""

import logging
import os
import sys
from dataclasses import dataclass

from .. import registry
from ..registry import LM_BACKEND
from .backend import MTP_CHOICES

MODES = ("thinking_off", "thinking_on")
SAMPLING_FIELDS = ("do_sample", "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "mtp")
MODEL_FIELDS = ("template", "thinking", "mtp", "backend", "defaults", "precisions")
REQUIRED_FIELDS = ("template", "thinking", "mtp", "defaults", "precisions")
PRECISION_FIELDS = ("file", "url")


@dataclass(frozen=True)
class Precision:
    name: str
    file: str
    url: "str | None"


@dataclass(frozen=True)
class Sampling:
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    presence_penalty: float
    repetition_penalty: float
    mtp: str  # one of backend.MTP_CHOICES


@dataclass(frozen=True)
class CatalogModel:
    name: str
    template: str
    thinking: bool
    mtp: bool
    backend: str
    defaults: dict    # 'thinking_off' -> Sampling, and 'thinking_on' -> Sampling on a thinking model
    precisions: dict  # precision name -> Precision, in the widget's order


class CatalogError(ValueError):
    """A catalog file that cannot be used; the message names the file, the entry and what to change."""


def load_catalog(builtin_path: str, user_files: list, templates: tuple, *, default_backend: str) -> dict:
    """{model name: CatalogModel}, the built-in models in their file's order, then the models the user files
    add. `templates`: the family's template ids; `default_backend`: the family's backend for an entry
    without one. Raises CatalogError naming the file and the entry. A model is checked complete in the file
    that adds it; an error after the merge names every user file that wrote the entry (the built-in entry,
    complete on its own, is not one to change)."""
    entries, origin = {}, {}  # origin: model name -> the user files that wrote its entry, in order
    for name, entry in _read(builtin_path, templates, builtin=True).items():
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise CatalogError(f"{builtin_path}: models.{name}: missing {', '.join(missing)}")
        _model(name, entry, builtin_path, default_backend)
        entries[name], origin[name] = entry, []
    for path in user_files:
        for name, entry in _read(path, templates, builtin=False).items():
            if name in entries:
                changed = _override(entries[name], entry)
                if changed:  # an empty entry overrides nothing
                    logging.info("[BCNodes] LM catalog: %s overrides %s: %s", path, name, ", ".join(changed))
                if entry:
                    origin[name].append(path)
            else:
                missing = [f for f in REQUIRED_FIELDS if f not in entry]
                if missing:
                    raise CatalogError(f"{path}: models.{name}: a new model needs {', '.join(missing)} "
                                       f"(only backend may be left out)")
                _model(name, entry, path, default_backend)
                entries[name], origin[name] = entry, [path]
    return {name: _model(name, entry, ", ".join(origin[name]) or builtin_path, default_backend)
            for name, entry in entries.items()}


def _read(path, templates, builtin):
    """{model name: entry} of one file, every entry checked as written: an entry is a dict holding only the
    fields it gives, defaults as {mode: {field: value}}, precisions as {name: Precision}."""
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except OSError as e:
        raise CatalogError(f"{path}: cannot be read: {e}") from e
    except UnicodeDecodeError as e:
        raise CatalogError(f"{path}: is not UTF-8 text ({e}); save it as UTF-8") from e
    except RecursionError as e:
        raise CatalogError(f"{path}: is nested too deeply to read; write the entries at most four levels below "
                           f"models: (model > defaults > mode > field, model > precisions > name > file)") from e
    except yaml.YAMLError as e:
        raise CatalogError(f"{path}: is not valid YAML: {e}") from e
    except ValueError as e:  # a plain value YAML reads as a date or number it cannot build: 2026-13-01, 0x_
        raise CatalogError(f"{path}: holds a value YAML reads as a date or a number but cannot convert ({e}); "
                           f"put that value in quotes to make it text") from e
    if not builtin and (data is None or data == {"models": None}):
        return {}  # an empty user file, or one whose entries are all commented out
    if not isinstance(data, dict) or set(data) != {"models"} or not isinstance(data["models"], dict):
        raise CatalogError(f"{path}: the file must hold one key, models:, mapping each model name to its entry")
    return {name: _entry(path, name, raw, templates, builtin) for name, raw in data["models"].items()}


def _entry(path, name, raw, templates, builtin):
    where = f"{path}: models.{name}"
    if not isinstance(name, str) or not name.strip():
        raise CatalogError(f"{path}: model name {name!r} must be a non-empty text")
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: the entry must be a mapping of {', '.join(MODEL_FIELDS)}")
    _known(where, raw, MODEL_FIELDS)
    entry = {}
    if "template" in raw:
        if raw["template"] not in templates:
            raise CatalogError(f"{where}: template {raw['template']!r} is not one of this family's: {', '.join(templates)}")
        entry["template"] = raw["template"]
    for field in ("thinking", "mtp"):
        if field in raw:
            if not isinstance(raw[field], bool):
                raise CatalogError(f"{where}: {field} must be true or false")
            entry[field] = raw[field]
    if "backend" in raw:
        backends = registry.names(LM_BACKEND)
        if raw["backend"] not in backends:
            raise CatalogError(f"{where}: backend {raw['backend']!r} is not one of: {', '.join(backends)}")
        entry["backend"] = raw["backend"]
    if "defaults" in raw:
        entry["defaults"] = _defaults(where, raw["defaults"])
    if "precisions" in raw:
        entry["precisions"] = _precisions(where, raw["precisions"], builtin)
    return entry


def _defaults(where, raw):
    if not isinstance(raw, dict):
        raise CatalogError(f"{where}: defaults must map {' and '.join(MODES)} to their sampling fields")
    _known(f"{where}.defaults", raw, MODES)
    out = {}
    for mode, block in raw.items():
        at = f"{where}.defaults.{mode}"
        if not isinstance(block, dict):
            raise CatalogError(f"{at}: must be a mapping of {', '.join(SAMPLING_FIELDS)}")
        _known(at, block, SAMPLING_FIELDS)
        out[mode] = {field: _sampling_value(at, field, value) for field, value in block.items()}
    return out


def _sampling_value(at, field, value):
    if field == "do_sample":
        if not isinstance(value, bool):
            raise CatalogError(f"{at}: do_sample must be true or false")
        return value
    if field == "top_k":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CatalogError(f"{at}: top_k must be a whole number")
        return value
    if field == "mtp":
        if value is False:
            return "off"
        if isinstance(value, int) and not isinstance(value, bool):
            value = str(value)
        if value not in MTP_CHOICES:
            raise CatalogError(f"{at}: mtp must be one of {', '.join(MTP_CHOICES)}")
        return value
    # abs(value) <= the largest float, not math.isfinite: an int too large for a float raises OverflowError there
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not abs(value) <= sys.float_info.max:
        raise CatalogError(f"{at}: {field} must be a number")
    return float(value)


def _precisions(where, raw, builtin):
    if not isinstance(raw, dict) or not raw:
        raise CatalogError(f"{where}: precisions must map each precision name to {{file: ..., url: ...}}")
    out = {}
    for name, spec in raw.items():
        at = f"{where}.precisions.{name}"
        if not isinstance(name, str) or not name.strip():
            raise CatalogError(f"{where}: precision name {name!r} must be a non-empty text")
        if not isinstance(spec, dict):
            raise CatalogError(f"{at}: must be a mapping {{file: ..., url: ...}}")
        _known(at, spec, PRECISION_FIELDS)
        file = spec.get("file")
        if not isinstance(file, str) or not file or os.path.basename(file) != file or file in (".", ".."):
            raise CatalogError(f"{at}: file must be a plain file name (no folder), e.g. model_bf16.safetensors")
        url = spec.get("url")
        if url is None and builtin:
            raise CatalogError(f"{at}: a built-in precision needs its url")
        if url is not None and (not isinstance(url, str) or not url.startswith(("https://", "http://"))):
            raise CatalogError(f"{at}: url must be an http(s) address")
        out[name] = Precision(name, file, url)
    return out


def _known(where, raw, fields):
    unknown = [str(k) for k in raw if k not in fields]
    if unknown:
        raise CatalogError(f"{where}: unknown key(s) {', '.join(unknown)}; the keys are {', '.join(fields)}")


def _override(base, user):
    """Writes the user entry's fields over `base` (in place); the names of what changed."""
    changed = []
    if user.get("thinking") is False and "thinking_on" in base["defaults"]:
        del base["defaults"]["thinking_on"]
        changed.append("defaults.thinking_on (dropped: thinking false)")
    for field, value in user.items():
        if field == "defaults":
            for mode, block in value.items():
                target = base["defaults"].setdefault(mode, {})
                for key, v in block.items():
                    target[key] = v
                    changed.append(f"defaults.{mode}.{key}")
        elif field == "precisions":
            for name, precision in value.items():
                changed.append(f"precisions.{name} ({'replaced' if name in base['precisions'] else 'added'})")
                base["precisions"][name] = precision
        else:
            base[field] = value
            changed.append(field)
    return changed


def _model(name, entry, files, default_backend):
    """The CatalogModel of a complete entry; `files` (text) names the file(s) that wrote it in the errors."""
    where = f"{files}: models.{name}"
    defaults = entry["defaults"]
    if not entry["thinking"] and "thinking_on" in defaults:
        raise CatalogError(f"{where}: thinking_on defaults on a model with thinking: false; remove them or set thinking: true")
    modes = MODES if entry["thinking"] else MODES[:1]
    sampling = {}
    for mode in modes:
        block = defaults.get(mode)
        if block is None:
            raise CatalogError(f"{where}: defaults.{mode} is missing")
        missing = [f for f in SAMPLING_FIELDS if f not in block]
        if missing:
            raise CatalogError(f"{where}: defaults.{mode} misses {', '.join(missing)}")
        sampling[mode] = Sampling(**{f: block[f] for f in SAMPLING_FIELDS})
    return CatalogModel(name=name, template=entry["template"], thinking=entry["thinking"], mtp=entry["mtp"],
                        backend=entry.get("backend", default_backend), defaults=sampling,
                        precisions=dict(entry["precisions"]))

"""Emulate: a rough estimate of a workflow's RAM and VRAM before it runs.

Built per component, never from one pixel ratio:
  - weights: fixed, from the safetensors header of every model file a node names (no model load);
  - tensors: exact from sizes (profiles.image_bytes, mask_bytes, wan_latent_bytes), propagated
    through the graph by a per-node-type profile (profiles.py);
  - output cache: every output stays in RAM until the prompt ends, summed in execution order;
  - node-internal transients (clone, stack, concat): from the profile, where its code was read. A
    node without a profile is listed as "not counted", never guessed.
Bypassed and muted nodes are listed, not added (they are not in the prompt the frontend sends;
the workflow says why). Subgraphs arrive expanded (`outer:inner` ids).

A video workflow (a video loader in the prompt) also gets a table of resolution x frame count.
Once a workflow has run armed, its measured nodes replace the formulas, scaled by the pixel count
of the scenario over the measured one.

`env` is the node layer's adapter: return_types(class_type), input_types(class_type),
model_path(name), weights(path), video_meta(name) -> {"width", "height", "frames", "fps"} or None,
image_size(name) -> (width, height) or None.
"""

from .profiles import PROFILES, Ctx, NotCounted, is_link

LABEL = "rough estimate; a real measurement exists only after the workflow has run once"
GIB = 1 << 30
GPUS = (("RTX 4090", 24 * GIB), ("RTX 5090", 32 * GIB))
RESOLUTIONS = (("480p", 480, 832), ("720p", 720, 1280), ("1080p", 1080, 1920))  # label, width, height (portrait)
FRAME_COUNTS = (81, 161, 321, 641)
VIDEO_LOADERS = ("VHS_LoadVideo", "VHS_LoadVideoPath", "BCVLoadVideo")
SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "NUMBER", "*"}
DATA = {"IMAGE", "MASK", "LATENT", "CONDITIONING", "AUDIO", "CLIP_VISION_OUTPUT"}
HANDLES = {"MODEL", "CLIP", "VAE", "CLIP_VISION", "CONTROL_NET", "UPSCALE_MODEL", "STYLE_MODEL", "GLIGEN"}
MODEL_FILES = (".safetensors", ".sft", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
MODES = {2: "muted", 4: "bypassed"}


# -- the graph -----------------------------------------------------------------------------------

def execution_order(prompt):
    """Node ids in a dependency order (Kahn), ties broken by prompt order."""
    deps = {nid: {v[0] for v in node.get("inputs", {}).values() if is_link(v) and v[0] in prompt} for nid, node in prompt.items()}
    order, done = [], set()
    while len(order) < len(prompt):
        ready = [nid for nid in prompt if nid not in done and deps[nid] <= done]
        if not ready:
            raise ValueError("the prompt has a dependency cycle")
        order.extend(ready)
        done.update(ready)
    return order


def skipped_nodes(workflow):
    """[{"id", "class_type", "title", "status"}] of muted and bypassed nodes, subgraph members included."""
    out = []
    graphs = [("", (workflow or {}).get("nodes", []))]
    graphs += [(f"{sg.get('name', sg.get('id'))}: ", sg.get("nodes", [])) for sg in ((workflow or {}).get("definitions") or {}).get("subgraphs", [])]
    for prefix, nodes in graphs:
        for n in nodes:
            if n.get("mode") in MODES:
                out.append({"id": str(n.get("id")), "class_type": n.get("type"), "title": prefix + (n.get("title") or n.get("type") or ""),
                            "status": MODES[n["mode"]]})
    return out


def _weights(node, env, cache):
    files = []
    for value in node.get("inputs", {}).values():
        if isinstance(value, str) and value.lower().endswith(MODEL_FILES):
            if value not in cache:
                path = env.model_path(value)
                cache[value] = None if path is None else {"name": value, **env.weights(path)}
            if cache[value] is not None:
                files.append(cache[value])
    return files


def estimate(prompt, env, scenario=None, measured=None, weight_cache=None):
    """Per-node rows and totals for one scenario ({"frames", "w", "h"} for the video loaders, None =
    the workflow as it is)."""
    weight_cache = {} if weight_cache is None else weight_cache
    measured_nodes = (measured or {}).get("nodes", {})
    ratio = _ratio(prompt, measured_nodes, scenario)
    upstream, models, types_of, rows = {}, {}, {}, []
    consumed = {}  # node id -> its output slots some node links
    for node in prompt.values():
        for value in node.get("inputs", {}).values():
            if is_link(value):
                consumed.setdefault(value[0], set()).add(value[1])
    cache = 0
    ram_peak = vram_peak = (0, None)
    for nid in execution_order(prompt):
        node = prompt[nid]
        row = {"id": nid, "class_type": node["class_type"], "title": (node.get("_meta") or {}).get("title") or node["class_type"]}
        files = _weights(node, env, weight_cache)
        types = types_of[nid] = env.return_types(node["class_type"])
        used = set()  # model files reaching this node through model-handle links (MODEL, VAE, ...)
        for value in node.get("inputs", {}).values():
            if is_link(value) and value[0] in models and _carries_models(types_of[value[0]], value[1]):
                used |= models[value[0]]
        models[nid] = used | {f["name"] for f in files}
        vram = sum(weight_cache[name]["bytes"] for name in used if weight_cache.get(name))
        row.update(weights=files, vram_need=vram)
        try:
            outputs, transient, note = _profile(node, types, Ctx(node, upstream, scenario, env, consumed.get(nid, set())))
            row["status"], row["note"] = "counted", note
        except NotCounted as e:
            outputs, transient = {}, None
            row["status"], row["note"] = "not counted", str(e)
        upstream[nid] = outputs
        out_bytes = sum(t["bytes"] for t in outputs.values() if not t.get("shared"))
        row["outputs"] = [{"slot": s, **{k: t[k] for k in ("type", "shape", "bytes")}, "shared": bool(t.get("shared"))}
                          for s, t in sorted(outputs.items())]
        m = measured_nodes.get(nid)
        if m and m.get("state") == "executed" and ratio is not None:
            # the measured node replaces the formula; downstream profiles keep the formula shapes
            added = m.get("new_bytes", m["output_bytes"])
            out_bytes = int(added * ratio)
            transient = int(max(0, m["ram_peak"] - m["ram_start"] - added) * ratio)
            row["status"], row["note"] = "measured", f"measured run, scaled x{ratio:.2f}"
        row.update(output_bytes=out_bytes, transient=transient)
        at_node = cache + out_bytes + (transient or 0)
        cache += out_bytes
        row.update(ram_at_node=at_node, cache_after=cache)
        ram_peak = max(ram_peak, (at_node, nid), key=lambda x: x[0])
        vram_peak = max(vram_peak, (vram, nid), key=lambda x: x[0])
        rows.append(row)
    weights_total = sum(w["bytes"] for w in weight_cache.values() if w)
    return {"rows": rows, "ram_peak": ram_peak[0], "ram_peak_node": ram_peak[1], "vram_peak": vram_peak[0],
            "vram_peak_node": vram_peak[1], "cache_total": cache, "weights_total": weights_total,
            "not_counted": [r["id"] for r in rows if r["status"] == "not counted"]}


def _carries_models(types, slot):
    """An output that is neither a number nor tensor data hands a model on (MODEL, VAE, CLIP, a pack's
    own model type)."""
    kind = types[slot] if slot < len(types) else None
    return kind is not None and kind not in SCALARS and kind not in DATA


def _profile(node, types, ctx):
    profile = PROFILES.get(node["class_type"])
    if profile is not None:
        try:
            return profile(ctx)
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:  # a widget value the profile cannot size
            raise NotCounted(f"cannot size {node['class_type']} from its inputs ({e!r})") from None
    data = [t for t in types if t not in SCALARS and t not in HANDLES]
    if data:
        raise NotCounted(f"no cost profile for {node['class_type']} (outputs {', '.join(data)})")
    return {}, 0, "outputs are numbers or model handles (weights are counted at the loaders)"


def _ratio(prompt, measured_nodes, scenario):
    """Scenario pixels over the measured run's pixels at the video loader, or None (no measurement,
    or no loader to scale by). 1.0 for the workflow as it is."""
    if not measured_nodes:
        return None
    if scenario is None:
        return 1.0
    for nid, node in prompt.items():
        m = measured_nodes.get(nid)
        if node["class_type"] in VIDEO_LOADERS and m and m.get("outputs"):
            n, h, w = m["outputs"][0]["shape"][:3]
            return scenario["frames"] * scenario["h"] * scenario["w"] / (n * h * w)
    return None


def fit(result, ram_limit, ram_now):
    """Per target GPU: VRAM need against it, and RAM need against the RAM limit: the RAM in use now
    (ComfyUI, loaded models, the last prompt's cached outputs: an upper bound) + the tensor peak +
    the weights that do not fit in VRAM."""
    out = []
    for name, vram in GPUS:
        spill = max(0, result["weights_total"] - vram)
        ram_need = (ram_now or 0) + result["ram_peak"] + spill
        out.append({"gpu": name, "vram": vram, "vram_fits": result["vram_peak"] <= vram, "ram_need": ram_need,
                    "ram_limit": ram_limit, "ram_fits": ram_limit is None or ram_need <= ram_limit})
    return out


def emulate(prompt, workflow, env, ram_limit, ram_now, measured=None, node_cost_s=None):
    """The Emulate tab: the workflow as it is, the scenario table for a video workflow, the fit
    check and what per-node measurement would add to the run."""
    weight_cache = {}
    current = estimate(prompt, env, None, measured, weight_cache)
    table = None
    if any(node["class_type"] in VIDEO_LOADERS for node in prompt.values()):
        # a loader's frames, when it outputs them (an unlinked heavy output is dropped)
        own = [r["outputs"][0]["shape"][0] for r in current["rows"]
               if r["class_type"] in VIDEO_LOADERS and r["outputs"] and r["outputs"][0]["type"] == "IMAGE"]
        frames = sorted(set(FRAME_COUNTS) | set(own))
        table = {"frames": frames, "rows": []}
        for label, w, h in RESOLUTIONS:
            cells = []
            for n in frames:
                r = estimate(prompt, env, {"frames": n, "w": w, "h": h}, measured, weight_cache)
                cells.append({"ram_peak": r["ram_peak"], "vram_peak": r["vram_peak"], "fit": fit(r, ram_limit, ram_now)})
            table["rows"].append({"label": label, "w": w, "h": h, "cells": cells})
    nodes = len(prompt)
    return {
        "label": LABEL, "current": current, "ram_now": ram_now, "fit": fit(current, ram_limit, ram_now), "table": table,
        "skipped": skipped_nodes(workflow), "calibrated": bool(measured),
        "measure_cost_s": None if node_cost_s is None else node_cost_s * nodes, "nodes": nodes,
    }

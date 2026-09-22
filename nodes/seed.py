"""BC_Seed (Seed)

A seed widget with three buttons (web/js/seed.js). The widget value -1 means
"a new random seed every run": the frontend swaps it for a concrete seed in
the prompt it sends (and in the workflow that goes into image metadata), so
the backend normally never sees -1. When it does — a prompt queued through
the API — the same happens here: a seed is drawn, written back into the
prompt and the workflow metadata, and returned. -1 is never passed on.
"""

import random

MAX_SEED = 0xFFFFFFFFFFFFFFFF
# Random seeds stay below 2**53 so the frontend (JavaScript numbers) shows and
# round-trips them exactly; anything larger would drift when an image is
# dropped back onto the canvas.
RANDOM_MAX = 2 ** 53 - 1
RANDOM = -1

_rng = random.Random()


def new_random_seed():
    return _rng.randint(0, RANDOM_MAX)


def _workflow_node(workflow, unique_id):
    """The serialized node for a prompt id; "outer:inner" ids point into a subgraph."""
    wanted = str(unique_id).split(":")[-1]
    graphs = [workflow] + list(((workflow or {}).get("definitions") or {}).get("subgraphs") or [])
    for g in graphs:
        for node in (g or {}).get("nodes") or []:
            if str(node.get("id")) == wanted:
                return node
    return None


def _record(seed, original, prompt, extra_pnginfo, unique_id):
    """Puts the drawn seed where the frontend would have put it."""
    if unique_id is None:
        return
    node = (prompt or {}).get(str(unique_id))
    if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
        node["inputs"]["seed"] = seed
    workflow = (extra_pnginfo or {}).get("workflow")
    wnode = _workflow_node(workflow, unique_id) if workflow else None
    if wnode is None:
        return
    values = wnode.get("widgets_values")
    if isinstance(values, list):
        for i, v in enumerate(values):
            if v == original:
                values[i] = seed
    named = wnode.get("widgets_values_named")
    if isinstance(named, dict) and named.get("seed") == original:
        named["seed"] = seed


class Seed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": RANDOM, "max": MAX_SEED, "control_after_generate": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "main"
    CATEGORY = "BCNodes/logic"
    SEARCH_ALIASES = ['BCNodes', 'seed', 'random seed']

    @classmethod
    def IS_CHANGED(cls, seed=0, **kwargs):
        # A concrete seed caches like any other input, so a fixed seed does not
        # re-run the graph below it. Only -1 (a prompt sent through the API
        # without the frontend's rewrite) must execute every time.
        return float("nan") if seed == RANDOM else seed

    def main(self, seed=0, prompt=None, extra_pnginfo=None, unique_id=None):
        seed = int(seed) if seed is not None else 0
        if seed == RANDOM:
            drawn = new_random_seed()
            print(f"[BCNodes] Seed: -1 received from the API, using random seed {drawn}")
            _record(drawn, RANDOM, prompt, extra_pnginfo, unique_id)
            seed = drawn
        return (seed,)


NODE_CLASS_MAPPINGS = {
    "BC_Seed": Seed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BC_Seed": "Seed",
}

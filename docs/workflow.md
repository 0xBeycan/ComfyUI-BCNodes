# Workflow

## `BC_ImageComparer` — Image Comparer

`image_a`, `image_b` (both optional) → shown on the node, drawn on the canvas itself so it moves with the node. A fills the node; while the pointer is over it, B is painted from the left edge up to the pointer with a divider line and A / B tags; leave the node and A shows alone. With more than one image per side a row of `A1 A2 B1 …` labels above the image picks the pair. The node keeps the size you give it; the image is letterboxed inside. The images are written to ComfyUI's temp folder like Preview Image does; the comparison lives with the run (it survives a tab switch, not a restart) and nothing is saved into the workflow file. Output node, no outputs.

## `BC_AnythingEverywhere` — Anything Everywhere

One node, any number of sources. Each source wired into the node is handed to **every unconnected input of the same type** in the workflow when the prompt is built — root graph and subgraph nodes alike. A connected slot takes the type and colour of its link (`VAE`, `CLIP`, …) and an empty `anything` slot is always kept at the bottom for the next one. The node properties `title_regex` and `input_regex` (right click → *Properties Panel*) narrow the targets by node title and input name; a node with a regex takes precedence over one without. A bypassed or muted Anything Everywhere does nothing.

On the canvas the node shows what it reaches: inputs it feeds get a glowing ring in the link's colour, every other free input a small dot (it could be fed), the node itself a green badge in its title bar (yellow when a regex restricts it, dim when it feeds nothing). The translucent phantom links from the node to the inputs it feeds are drawn when the node or the target is selected or under the pointer. Settings → *BCNodes › Anything Everywhere*: *Show links* (all off / selected nodes / mouseover node / selected and mouseover nodes / all on) and *Highlight connected and connectable inputs*.

Implementation: the API prompt that `graphToPrompt` returns is patched with the extra links, so queueing and *Export (API)* both contain them; no real links are drawn. Limits: the Anything Everywhere node and its source must sit in the root graph; a source that is a subgraph node's output is not resolved (a console warning says so).

**API / serverless mode:** the links only exist because the frontend wrote them into the prompt. A prompt exported from the frontend (*Export (API)*) carries them and runs anywhere. A prompt assembled without the frontend has no such links, so any input that depended on Anything Everywhere is simply missing and validation fails with `Required input is missing`. The Python side of the node is a no-op that is never executed.

## `BC_FastGroupsBypasser` — Fast Groups Bypasser

One toggle per group — the groups of the graph the node sits in and the groups inside every subgraph: on = the group's nodes are active, off = bypassed (subgraph nodes inside a group are switched together with their contents). The list follows the graph on a half-second tick — new, renamed and removed groups, and modes changed by other means. Right-click menu: *Bypass all*, *Enable all*, *Toggle all*.

Properties: `sort` (`position`, `alphanumeric` or `custom alphabet`), `customSortAlphabet` (letters, or comma-separated prefixes, that order the `custom alphabet` sort), `matchColors` (comma-separated group colours — names such as `red` or hex values — only matching groups are listed), `matchTitle` (regex; same), `showAllGraphs` (off = only the groups of the graph on screen), `toggleRestriction` (`default`, `max one` — switching a group on switches the others off — or `always one` — same, and the last active group cannot be switched off).

**API / serverless mode:** bypass is a frontend concept. When the frontend builds the prompt, bypassed nodes are already left out and their links routed around them, so an exported prompt reflects the toggles at export time and runs anywhere. A prompt assembled without the frontend cannot be switched by this node; it contains whatever nodes it was given. The Python side of the node is a no-op that is never executed.

## Align

With two or more items selected (nodes, groups, reroutes, subgraph nodes), the selection toolbox gains eight buttons: align left / horizontal centers / right / top / vertical centers / bottom, and — from three items — distribute horizontally / vertically. A group moves with its contents. Each action is one undo step.

For keeping things tidy while dragging, ComfyUI's own **Settings → LiteGraph → Canvas → Always snap to grid** does the job; the toolbox's **Arrange** menu re-stacks a selection vertically, horizontally or as a grid.

## `BC_AutoBypass` — Auto Bypass

A frontend-only node that watches a source (LoadImage, VHS_LoadVideo, ...) and flips its targets between **ACTIVE** and **BYPASS** automatically: empty source → targets bypassed, source loaded → targets active.

No Python execution. The node is virtual — it never appears in the prompt sent to the backend; it only rewrites `mode` on the nodes wired into it. It lives entirely in [`web/js/auto_bypass.js`](../web/js/auto_bypass.js).

With no Python side, the frontend would title the node with its key; the node's definition carries its display name, so it is titled Auto Bypass from the search box and the node library alike, and a workflow saved with the title `BC_AutoBypass` gets `Auto Bypass` when it loads.

### The problem it solves

Nodes such as `ImageResizeKJv2` have a **required** `image` input. When the upstream loader is bypassed, the link is gone and prompt validation fails with `Required input is missing: image`. In a workflow with an optional branch (a reference image that is sometimes there, sometimes not) you end up opening the subgraph and bypassing the resize node by hand every run. `Auto Bypass` does that for you.

### Inputs

| Slot | Type | Meaning |
| --- | --- | --- |
| `watch` | `*` | Output of the source to observe (e.g. `LoadImage.IMAGE`). |
| `force` | `BOOLEAN`, optional | Overrides the watch check: `true` → ACTIVE, `false` → BYPASS. See the note below. |
| `mode` | `COMBO`, optional | Socket of the `mode` widget. Wire it, or promote the widget out of a subgraph so it is set from the parent graph. See the note below. |
| `target_1..N` | `*` | Output of each node to control. Dynamic: connecting the last slot opens a new empty one, empty slots in the middle are removed. |

### Widgets

| Widget | Values | Meaning |
| --- | --- | --- |
| `mode` | `auto` / `force_enable` / `force_bypass` | `auto` follows the decision below. The two `force_*` values pin the targets regardless of inputs. |
| `status` | read-only | Current result and why, e.g. `BYPASS (image empty) -> 2 targets`. |

### Decision order

1. `mode` is `force_enable` / `force_bypass` → that.
2. `force` input is connected and resolves to a boolean → that.
3. Otherwise the `watch` source is **empty** when any of these holds:
   - `watch` is not connected;
   - the source node's mode is MUTE (2) or BYPASS (4);
   - the source's file widget (`image`, `video`, `audio`, `file`, `filename`, `model_file`, `path`, `url`) is an empty string / `None`.

Empty → every target is set to BYPASS. Not empty → every target is set to ACTIVE.

### Behavior

- Reroutes and virtual pass-through nodes (KJNodes Set/Get and the like) are followed to the real source, with cycle protection.
- Links that enter a subgraph through its input panel are followed out to the parent graph, so the node can live inside a subgraph while the loader sits outside.
- Every instance in the root graph and in every subgraph is evaluated together, on a 500 ms tick, on connection changes, and once more right before the prompt is built — so what gets queued always reflects the current state.
- A target that is itself a subgraph node gets its inner nodes set as well (the frontend does not propagate a subgraph node's mode into its body on its own).
- If `Auto Bypass` itself is muted or bypassed it stops touching its targets and says so in `status`.
- A source that is also listed as a target is not treated as "empty" because of its own mode — otherwise it would lock itself in BYPASS.

### Note on `mode`

`mode` is a widget with a socket. Inside a subgraph you can promote it (or drag its socket to the subgraph's input panel); the value then lives on the subgraph node in the parent graph and wins over the inner widget. The same works through nested subgraphs. When the socket is linked to a node instead, the value is read from that node's widget if it carries one of the three mode strings; otherwise the inner widget value applies.

### Note on `force`

The frontend cannot see values computed during execution. `force` only resolves when the connected node carries the boolean as a widget — a Primitive node, a BOOL constant node, and similar. If it comes from a node that computes the value at run time (e.g. `Is Mask Empty`) it cannot be read; `status` reports `force unresolved` and the watch check applies instead.

import { app } from "../../../scripts/app.js";

// Auto Bypass — a frontend-only (virtual) node. It never reaches the backend;
// it only rewrites `mode` on the nodes wired into its target_* slots based on
// whether the node wired into `watch` is "empty".
//
// Slots:
//   watch    (*)        the source being observed
//   force    (BOOLEAN)  overrides the watch check when it resolves to a value
//   mode     (COMBO)    widget input for the `mode` widget, so it can be wired
//                       or promoted out of a subgraph
//   target_N (*)        nodes to flip between ACTIVE and BYPASS (dynamic)
// Target slots are found by name, not index: workflows saved before the
// `mode` slot existed get it appended after their targets.

const NODE_TYPE = "BC_AutoBypass";
const NODE_TITLE = "Auto Bypass";
const NODE_CATEGORY = "BCNodes/workflow";

const MODE_ALWAYS = 0; // LiteGraph.ALWAYS
const MODE_MUTE = 2; // LiteGraph.NEVER
const MODE_BYPASS = 4; // ComfyUI bypass

// litegraph uses this origin_id for links that come from a subgraph's own input node.
// It is serialized as a string in current frontends, so compare via String().
const SUBGRAPH_INPUT_ID = "-10";

const WATCH_SLOT = 0;
const FORCE_SLOT = 1;
const MODE_SLOT_NAME = "mode";
const TARGET_PREFIX = "target_";

const MODES = ["auto", "force_enable", "force_bypass"];

// Widget names that hold a file reference on loader nodes (LoadImage.image,
// VHS_LoadVideo.video, LoadAudio.audio, ...). First match wins.
const FILE_WIDGET_NAMES = ["image", "video", "audio", "file", "filename", "model_file", "path", "url"];

const TICK_MS = 500;

const LGraphNode = LiteGraph.LGraphNode ?? window.LGraphNode;

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

function getRootGraph() {
	return app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
}

// Root graph plus every subgraph definition. Nodes inside a subgraph are shared
// by all of its instances, so visiting the definition once is enough.
function allGraphs() {
	const root = getRootGraph();
	if (!root) return [];
	const graphs = [root];
	const subgraphs = root.subgraphs;
	if (subgraphs?.values) {
		for (const sg of subgraphs.values()) graphs.push(sg);
	}
	return graphs;
}

function graphNodes(graph) {
	return graph?.nodes ?? graph?._nodes ?? [];
}

function getLink(graph, linkId) {
	if (graph == null || linkId == null) return null;
	return graph._links?.get?.(linkId) ?? graph.links?.get?.(linkId) ?? graph.links?.[linkId] ?? null;
}

function isEmptyValue(value) {
	if (value == null) return true;
	if (typeof value !== "string") return false;
	const v = value.trim();
	return v === "" || v.toLowerCase() === "none";
}

// The SubgraphNode instance (in a parent graph) whose body is `subgraph`.
// If the same subgraph is instantiated more than once the first one wins.
function findSubgraphNodeFor(subgraph) {
	for (const graph of allGraphs()) {
		for (const n of graphNodes(graph)) {
			if (n.isSubgraphNode?.() && (n.subgraph === subgraph || n.subgraph?.id === subgraph.id)) {
				return n;
			}
		}
	}
	return null;
}

// Origin {node, slot} of an input link. Crosses a subgraph input boundary by
// hopping to the SubgraphNode that wraps the current graph and following the
// matching outer link. If the outer slot is not linked but carries a promoted
// widget, the result is {value} — the value lives on the SubgraphNode, the
// inner widget is only a stale mirror.
function resolveLinkOrigin(graph, linkId, visitedGraphs = new Set()) {
	const link = getLink(graph, linkId);
	if (!link) return null;

	if (String(link.origin_id) === SUBGRAPH_INPUT_ID) {
		if (visitedGraphs.has(graph)) return null;
		visitedGraphs.add(graph);
		const wrapper = findSubgraphNodeFor(graph);
		if (!wrapper) return null;
		const innerName = graph.inputs?.[link.origin_slot]?.name;
		const outerInput =
			(innerName != null && wrapper.inputs?.find((i) => i.name === innerName)) ||
			wrapper.inputs?.[link.origin_slot];
		if (!outerInput) return null;
		if (outerInput.link == null) {
			const widget = wrapper.getWidgetFromSlot?.(outerInput);
			return widget ? { value: widget.value } : null;
		}
		return resolveLinkOrigin(wrapper.graph, outerInput.link, visitedGraphs);
	}

	const node = graph.getNodeById?.(link.origin_id);
	return node ? { node, slot: link.origin_slot } : null;
}

// Reroutes and other virtual nodes only forward a link; the real source is
// behind them. Our own node has no outputs so it can never be in a chain.
function isPassThrough(node) {
	if (!node) return false;
	const type = String(node.type ?? node.constructor?.type ?? "");
	if (type.includes("Reroute")) return true;
	return !!node.isVirtualNode;
}

// Walks a pass-through chain to the first real node. `visited` protects
// against cycles (a Reroute fed by its own output, Set/Get pairs pointing at
// each other, ...).
function followPassThrough(entry, visited = new Set()) {
	let current = entry;
	while (current?.node && isPassThrough(current.node)) {
		const n = current.node;
		if (visited.has(n)) return null;
		visited.add(n);

		let next = null;
		if (n.inputs?.[0]?.link != null) {
			next = resolveLinkOrigin(n.graph, n.inputs[0].link);
		} else if (typeof n.findSetter === "function") {
			// KJNodes GetNode: no input link, the source is its paired SetNode.
			const setter = n.findSetter(n.graph);
			if (setter) next = { node: setter, slot: 0 };
		}
		current = next;
	}
	if (!current) return null;
	return current.node || "value" in current ? current : null;
}

function resolveInputSource(node, slot) {
	const linkId = node.inputs?.[slot]?.link;
	if (linkId == null) return null;
	return followPassThrough(resolveLinkOrigin(node.graph, linkId), new Set([node]));
}

function findFileWidget(node) {
	const widgets = node.widgets ?? [];
	return widgets.find((w) => FILE_WIDGET_NAMES.includes(String(w.name ?? "").toLowerCase())) ?? null;
}

// The frontend cannot see execution-time values, so `force` only resolves when
// its source carries the boolean as a widget (PrimitiveNode, a BOOL constant
// node, ...). Anything else is reported as unresolved and ignored.
function readBooleanWidget(node) {
	const widgets = node?.widgets ?? [];
	const w = widgets.find((w) => w.type === "toggle" || typeof w.value === "boolean");
	return typeof w?.value === "boolean" ? w.value : null;
}

// Sets `mode` on a node and, for subgraph nodes, on every node inside — the
// frontend does not propagate a SubgraphNode's mode into its body on its own.
function setModeDeep(node, mode) {
	let changed = false;
	const stack = [node];
	const seen = new Set();
	while (stack.length) {
		const n = stack.pop();
		if (!n || seen.has(n)) continue;
		seen.add(n);
		if (n.mode !== mode) {
			n.mode = mode;
			changed = true;
		}
		if (n.isSubgraphNode?.() && n.subgraph) {
			stack.push(...graphNodes(n.subgraph));
		}
	}
	return changed;
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

class AutoBypassNode extends LGraphNode {
	static title = NODE_TITLE;
	static category = NODE_CATEGORY;
	static type = NODE_TYPE;

	constructor(title) {
		super(title ?? NODE_TITLE);
		this.isVirtualNode = true;
		this.serialize_widgets = true;
		this.properties = this.properties ?? {};
		this.properties["Node name for S&R"] = NODE_TYPE;

		this.addInput("watch", "*");
		this.addInput("force", "BOOLEAN");
		this.addModeInput();
		this.addInput(`${TARGET_PREFIX}1`, "*");

		this.modeWidget = this.addWidget("combo", MODE_SLOT_NAME, "auto", () => scheduleEvaluateAll(0), {
			values: MODES,
		});
		this.statusWidget = this.addWidget("text", "status", "", () => {}, {});
		// Display only. `disabled` would hide the value in current frontends,
		// so the click is swallowed and any edit is reverted on the next tick.
		this.statusWidget.onClick = () => {};
		this.statusWidget.mouse = () => true;
		this.statusWidget.callback = () => scheduleEvaluateAll(0);

		this.size = this.computeSize();
	}

	onConnectionsChange() {
		if (app.configuringGraph) return;
		scheduleEvaluateAll(50);
	}

	// `widget: { name }` ties the slot to the widget: the frontend draws the
	// socket on the widget row and lets it be linked or promoted.
	addModeInput() {
		return this.addInput(MODE_SLOT_NAME, "COMBO", { widget: { name: MODE_SLOT_NAME } });
	}

	targetSlots() {
		const slots = [];
		for (let i = 0; i < this.inputs.length; i++) {
			if (String(this.inputs[i]?.name ?? "").startsWith(TARGET_PREFIX)) slots.push(i);
		}
		return slots;
	}

	// Keeps exactly one empty target slot (the last one), target_N names
	// contiguous, and the mode slot present on nodes saved before it existed.
	stabilizeTargets() {
		let changed = false;
		const width = this.size[0];

		if (!this.inputs.some((i) => i?.name === MODE_SLOT_NAME)) {
			this.addModeInput();
			changed = true;
		}

		const slots = this.targetSlots();
		for (let k = slots.length - 2; k >= 0; k--) {
			if (this.inputs[slots[k]]?.link == null) {
				this.removeInput(slots[k]);
				changed = true;
			}
		}

		const remaining = this.targetSlots();
		const last = this.inputs[remaining[remaining.length - 1]];
		if (!last || last.link != null) {
			this.addInput(`${TARGET_PREFIX}1`, "*");
			changed = true;
		}

		this.targetSlots().forEach((slot, k) => {
			const name = `${TARGET_PREFIX}${k + 1}`;
			if (this.inputs[slot].name !== name) {
				this.inputs[slot].name = name;
				this.inputs[slot].label = undefined;
				changed = true;
			}
		});

		if (changed) {
			const computed = this.computeSize();
			this.setSize([Math.max(width, computed[0]), computed[1]]);
		}
		return changed;
	}

	getTargets() {
		const targets = [];
		for (const i of this.targetSlots()) {
			const src = resolveInputSource(this, i);
			if (src?.node && !targets.includes(src.node)) targets.push(src.node);
		}
		return targets;
	}

	// The mode slot, when linked, wins over the widget: a promoted widget keeps
	// its value on the SubgraphNode, a linked node keeps it in its own widget.
	// A PrimitiveNode writes straight into our widget, so the fallback covers it.
	readMode() {
		const slot = this.inputs.findIndex((i) => i?.name === MODE_SLOT_NAME);
		if (slot >= 0 && this.inputs[slot].link != null) {
			const src = resolveInputSource(this, slot);
			const value = src?.node ? src.node.widgets?.find((w) => MODES.includes(w.value))?.value : src?.value;
			if (MODES.includes(value)) return value;
		}
		const value = this.modeWidget?.value;
		return MODES.includes(value) ? value : "auto";
	}

	// Decision order: mode > force input > watch source.
	decide(targets) {
		const mode = this.readMode();
		if (mode === "force_enable") return { active: true, reason: "mode force_enable" };
		if (mode === "force_bypass") return { active: false, reason: "mode force_bypass" };

		let note = "";
		if (this.inputs[FORCE_SLOT]?.link != null) {
			const forceSrc = resolveInputSource(this, FORCE_SLOT);
			const value = readBooleanWidget(forceSrc?.node);
			if (value !== null) return { active: value, reason: `force ${value}` };
			note = "force unresolved, ";
		}

		if (this.inputs[WATCH_SLOT]?.link == null) {
			return { active: false, reason: note + "watch not connected" };
		}
		const src = resolveInputSource(this, WATCH_SLOT);
		if (!src?.node) return { active: false, reason: note + "watch source missing" };

		const source = src.node;
		// A source that is also a target would lock itself in BYPASS forever;
		// its mode is not a signal in that case.
		if (!targets.includes(source)) {
			if (source.mode === MODE_MUTE) return { active: false, reason: note + `${source.title} muted` };
			if (source.mode === MODE_BYPASS) return { active: false, reason: note + `${source.title} bypassed` };
		}

		const fileWidget = findFileWidget(source);
		if (fileWidget && isEmptyValue(fileWidget.value)) {
			return { active: false, reason: note + `${fileWidget.name} empty` };
		}
		return { active: true, reason: note + (fileWidget ? `${fileWidget.name} set` : `${source.title} active`) };
	}

	evaluate() {
		if (!this.graph) return false;
		let changed = this.stabilizeTargets();
		const targets = this.getTargets();

		let status;
		if (this.mode === MODE_MUTE || this.mode === MODE_BYPASS) {
			status = `OFF (node ${this.mode === MODE_MUTE ? "muted" : "bypassed"}) -> ${targets.length} target${targets.length === 1 ? "" : "s"}`;
		} else {
			const { active, reason } = this.decide(targets);
			const mode = active ? MODE_ALWAYS : MODE_BYPASS;
			for (const target of targets) {
				changed = setModeDeep(target, mode) || changed;
			}
			status = `${active ? "ACTIVE" : "BYPASS"} (${reason}) -> ${targets.length} target${targets.length === 1 ? "" : "s"}`;
		}

		if (this.statusWidget && this.statusWidget.value !== status) {
			this.statusWidget.value = status;
			changed = true;
		}
		return changed;
	}
}

// ---------------------------------------------------------------------------
// Global evaluation: every instance in every graph, on a debounced schedule.
// ---------------------------------------------------------------------------

let pending = null;

function evaluateAll() {
	if (app.configuringGraph) return;
	let changed = false;
	for (const graph of allGraphs()) {
		for (const node of graphNodes(graph)) {
			if (node.type === NODE_TYPE && typeof node.evaluate === "function") {
				changed = node.evaluate() || changed;
			}
		}
	}
	if (changed) app.graph?.setDirtyCanvas(true, true);
}

function scheduleEvaluateAll(ms) {
	if (pending) clearTimeout(pending);
	pending = setTimeout(() => {
		pending = null;
		evaluateAll();
	}, ms);
}

app.registerExtension({
	name: "BCNodes.AutoBypass",

	registerCustomNodes() {
		LiteGraph.registerNodeType(NODE_TYPE, AutoBypassNode);
		AutoBypassNode.category = NODE_CATEGORY;
	},

	setup() {
		// Widget edits and mode changes on other nodes have no event we can hook,
		// so we poll.
		setInterval(evaluateAll, TICK_MS);

		// Make the target modes deterministic at the moment the prompt is built,
		// independent of the tick.
		const graphToPrompt = app.graphToPrompt;
		if (typeof graphToPrompt === "function") {
			app.graphToPrompt = async function (...args) {
				evaluateAll();
				return graphToPrompt.apply(this, args);
			};
		}
	},

	loadedGraphNode(node) {
		if (node.type === NODE_TYPE) scheduleEvaluateAll(0);
	},
});

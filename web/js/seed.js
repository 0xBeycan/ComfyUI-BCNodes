import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Seed — the seed widget plus three buttons:
//   🎲 Randomize Each Time      sets the widget to -1
//   🎲 New Fixed Random         writes a concrete random seed into the widget
//   ♻️ (Use Last Queued Seed)   puts the seed of the last queued run back
//
// -1 never reaches the server from here: right before a prompt is sent, every
// BC_Seed at -1 is swapped for a fresh seed in both the API prompt and the
// workflow copy that ends up in image metadata. The widget itself stays -1.

const NODE_TYPE = "BC_Seed";
const RANDOM = -1;
const RANDOM_MAX = Number.MAX_SAFE_INTEGER; // 2**53 - 1: exact in JavaScript and in the metadata
const LAST_LABEL = "♻️ (Use Last Queued Seed)";

const randomSeed = () => Math.floor(Math.random() * RANDOM_MAX);

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

function allGraphs() {
	const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
	if (!root) return [];
	const graphs = [root];
	if (root.subgraphs?.values) for (const sg of root.subgraphs.values()) graphs.push(sg);
	return graphs;
}

// Prompt ids of nodes inside subgraphs look like "outer:inner"; the canvas
// node is the inner one, shared by every instance of that subgraph.
function canvasNode(promptId) {
	const wanted = String(promptId).split(":").pop();
	for (const g of allGraphs()) {
		const node = g.getNodeById?.(Number(wanted)) ?? g.getNodeById?.(wanted);
		if (node?.type === NODE_TYPE) return node;
	}
	return null;
}

function workflowNode(workflow, promptId) {
	const wanted = String(promptId).split(":").pop();
	const graphs = [workflow, ...(workflow?.definitions?.subgraphs ?? [])];
	for (const g of graphs) {
		const node = (g?.nodes ?? []).find((n) => String(n.id) === wanted);
		if (node) return node;
	}
	return null;
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

function setLastButton(node, seed) {
	const b = node.bcSeed.lastButton;
	if (seed === undefined || seed === node.bcSeed.widget.value) {
		b.label = LAST_LABEL;
		b.disabled = true;
	} else {
		b.label = `♻️ ${seed}`;
		b.disabled = false;
	}
	node.setDirtyCanvas?.(true, false);
}

function setup(node) {
	const widget = node.widgets?.find((w) => w.name === "seed");
	if (!widget) return;
	// Older frontends add a control_after_generate combo for any input named
	// "seed" regardless of the node def; the buttons replace it.
	const control = node.widgets.findIndex((w) => w.name === "control_after_generate");
	if (control >= 0) node.widgets.splice(control, 1);

	const uiOnly = (w) => {
		w.serialize = false;
		w.options = w.options ?? {};
		w.options.serialize = false;
		return w;
	};
	const randomize = uiOnly(node.addWidget("button", "randomize_each_time", null, () => {
		widget.value = RANDOM;
		setLastButton(node, node.bcSeed.lastSeed);
	}));
	randomize.label = "🎲 Randomize Each Time";
	const fixed = uiOnly(node.addWidget("button", "new_fixed_random", null, () => {
		widget.value = randomSeed();
		setLastButton(node, node.bcSeed.lastSeed);
	}));
	fixed.label = "🎲 New Fixed Random";
	const last = uiOnly(node.addWidget("button", "use_last_seed", null, () => {
		if (node.bcSeed.lastSeed === undefined) return;
		widget.value = node.bcSeed.lastSeed;
		setLastButton(node, node.bcSeed.lastSeed);
	}));
	node.bcSeed = { widget, lastButton: last, lastSeed: undefined };
	setLastButton(node, undefined);
	node.setSize(node.computeSize());
}

// ---------------------------------------------------------------------------
// Prompt rewrite
// ---------------------------------------------------------------------------

function rewriteSeeds(data) {
	const output = data?.output;
	if (!output) return;
	for (const [id, entry] of Object.entries(output)) {
		if (entry?.class_type !== NODE_TYPE || !entry.inputs || !("seed" in entry.inputs)) continue;
		const node = canvasNode(id);
		const original = entry.inputs.seed;
		let seed = original;
		if (original === RANDOM) {
			seed = randomSeed();
			entry.inputs.seed = seed;
			const wnode = workflowNode(data.workflow, id);
			if (wnode) {
				if (Array.isArray(wnode.widgets_values)) {
					wnode.widgets_values = wnode.widgets_values.map((v) => (v === RANDOM ? seed : v));
				}
				if (wnode.widgets_values_named && wnode.widgets_values_named.seed === RANDOM) {
					wnode.widgets_values_named.seed = seed;
				}
			}
		}
		if (node?.bcSeed) {
			node.bcSeed.lastSeed = seed;
			setLastButton(node, seed);
		}
	}
}

app.registerExtension({
	name: "BCNodes.Seed",

	setup() {
		const original = api.queuePrompt;
		api.queuePrompt = function (number, data, ...rest) {
			try {
				rewriteSeeds(data);
			} catch (e) {
				console.error("[BCNodes] Seed: could not rewrite seeds", e);
			}
			return original.call(this, number, data, ...rest);
		};
	},

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;
		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				setup(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};
	},
});

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { callJson } from "./bcnodes_api.js";

// PreFlight Outcome — the frontend half of its record list (nodes/preflight.py).
//
// The `record` combo lists the feedback store's predictions, newest first. The list comes from the
// server: GET /bcnodes/preflight/records when an Outcome node is created or loaded, and the
// bcnodes.preflight.records event the server sends to every open browser after PreFlight Report logs
// a prediction. So a new prediction shows in every Outcome node at once, without a page reload or
// Refresh; a node-definition refresh puts the server's current list in too, as for any combo.
//
// Only the list changes. A value is never rewritten on load (a workflow opened, a node pasted, undo),
// even when the list no longer holds it: the server accepts any record label and logs to its id (the
// label's first token), and record_id_override wins over it. The one value set here is that of a node
// the user has just added: it has no choice yet, so it starts on the newest record.
//
// The combo's options are a new array per node (the node definition's list is shared by every node of
// the type); the classic canvas and the Vue node renderer both read options.values, as ComfyUI's own
// Refresh writes it.

const NODE_TYPE = "BC_PreFlightOutcome";
const RECORD = "record";
const ROUTE = "/bcnodes/preflight/records";
const EVENT = "bcnodes.preflight.records";

function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// Calls fn after target[key] (a prototype method), with the same this and arguments; the original's
// result is returned and an error in fn is logged.
function after(target, key, fn) {
	if (!target) return;
	const original = target[key];
	target[key] = function (...args) {
		const r = original?.apply(this, args);
		try {
			fn.apply(this, args);
		} catch (e) {
			console.error("[BCNodes]", e);
		}
		return r;
	};
}

// ---------------------------------------------------------------------------
// The record list
// ---------------------------------------------------------------------------

let records = null; // the latest list, pending or settled: an array of labels, or null when it could not be read

function getRecords() {
	if (!records) {
		const pending = callJson(ROUTE)
			.then((data) => (Array.isArray(data.records) ? data.records : null))
			.catch((e) => {
				console.warn("[BCNodes] PreFlight: could not read the Outcome record list; the lists are left as they are", e);
				return null;
			})
			.then((labels) => {
				if (labels === null && records === pending) records = null; // read again next time
				return labels;
			});
		records = pending;
	}
	return records;
}

// fn(labels) with the latest list (never before the caller returns); nothing when it could not be read.
// A list that arrives while an older read is pending replaces it before fn runs.
function withRecords(fn) {
	const current = getRecords();
	current
		.then((labels) => {
			if (current !== records && records) return withRecords(fn);
			if (labels) fn(labels);
		})
		.catch((e) => console.error("[BCNodes]", e));
}

function allGraphs() {
	const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
	if (!root) return [];
	const graphs = [root];
	if (root.subgraphs?.values) for (const sg of root.subgraphs.values()) graphs.push(sg);
	return graphs;
}

function outcomeNodes() {
	const nodes = [];
	for (const g of allGraphs()) {
		for (const n of g.nodes ?? g._nodes ?? []) if (n.type === NODE_TYPE) nodes.push(n);
	}
	return nodes;
}

// The node's combo lists `labels`; its value stays.
function applyList(node, labels) {
	const w = widget(node, RECORD);
	if (!w?.options) return;
	w.options.values = [...labels];
	node.setDirtyCanvas?.(true, false);
}

// Nodes that were configured (loaded, pasted, restored by undo): their value is never set from here.
const configured = new WeakSet();

function installOutcomeNode(nodeType) {
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		const node = this;
		// A node made while a graph loads gets its saved value from configure right after this.
		const added = !app.configuringGraph;
		const initial = widget(node, RECORD)?.value;
		withRecords((labels) => {
			applyList(node, labels);
			const w = widget(node, RECORD);
			// A node the user has just added (configure never ran on it) and whose value nobody changed
			// starts on the newest record.
			if (added && w && !configured.has(node) && w.value === initial && labels.length) {
				w.value = labels[0];
				node.setDirtyCanvas?.(true, false);
			}
		});
	});
	after(proto, "onConfigure", function () {
		configured.add(this);
		withRecords((labels) => applyList(this, labels));
	});
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

app.registerExtension({
	name: "BCNodes.PreFlight",

	setup() {
		// Report logged a prediction: the event carries the whole list.
		api.addEventListener(EVENT, ({ detail }) => {
			if (!Array.isArray(detail?.records)) return;
			records = Promise.resolve(detail.records);
			for (const node of outcomeNodes()) applyList(node, detail.records);
		});
	},

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name === NODE_TYPE) installOutcomeNode(nodeType);
	},

	// A node-definition refresh gives every Outcome node the server's current list (ComfyUI writes the
	// combo options from the new definitions); the cached list is read again next time.
	refreshComboInNodes() {
		records = null;
	},
});

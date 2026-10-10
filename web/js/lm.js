import { app } from "../../../scripts/app.js";
import { callJson } from "./bcnodes_api.js";

// LM nodes (Qwen LM, and any later family node) and LM Config — the frontend half of nodes/lm.py.
// The data comes from GET /bcnodes/lm/catalog, read once (again after a node-definition refresh).
//
// LM node: the precision combo lists only the selected model's precisions, and `thinking` is disabled
// for a model without thinking while it is off. Applied on creation, on load, on a node-definition
// refresh and on a model or thinking change. Only a user model change also fits the values: the
// precision is kept when the new model has it, else the model's first, and thinking turns off on a
// model without it; not while the family's catalog has an error (it then holds the built-in models
// only, so a precision a user models.yaml adds would look invalid). Otherwise a value the model does
// not take is kept and the run's error names the valid ones: a saved workflow's values come back as
// saved. A model the catalog does not know changes nothing.
//
// The frontend also calls a widget's callback with its value unchanged (Rename Widget, a subgraph
// promote): only a callback with a value other than the one last seen is a change.
//
// LM Config: the backend applies the fields named in the hidden `edited` widget and every field fed by
// a link (it reads the links from the prompt; nothing here marks them). A user change to a field (its
// widget callback with a changed value; setting a value from code does not call it) adds it to
// `edited`. The other fields show the sampling defaults of the first LM node the config output is
// linked to (its model, thinking on / off), refreshed when that link changes, when that node's model
// or thinking changes, on "Reset to model defaults" (node menu; clears `edited`) and on a
// node-definition refresh. Never on load: a saved workflow's values come back as saved.
// On the node canvas un-edited fields are drawn with the disabled look and still take input. The Vue
// node renderer has no such look: its only greyed state is `disabled`, which also blocks input, so
// there un-edited fields look like edited ones.

const ROUTE = "/bcnodes/lm/catalog";
const CONFIG_TYPE = "BC_LM_CONFIG";
const EDITED = "edited";
// BC_LMConfig's fields in widget order; `edited` lists them in this order.
const FIELDS = ["do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty", "mtp"];
// The node canvas's disabled look: LGraphNode.drawWidgets halves the alpha of a disabled widget.
const DEFAULT_ALPHA = 0.5;

function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// Calls fn after target[key] (a prototype method or a widget callback), with the same this and
// arguments; the original's result is returned and an error in fn is logged.
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

// The value each watched widget was last seen with: at creation, on load, when set here, and at each
// change.
const seen = new WeakMap();

function see(w) {
	if (w) seen.set(w, w.value);
}

// Calls fn after w's callback when the value it is called with is not the one last seen.
function afterChange(w, fn) {
	if (!w) return;
	see(w);
	after(w, "callback", (value) => {
		if (value === seen.get(w)) return;
		seen.set(w, value);
		fn();
	});
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

let catalog = null; // the fetch, pending or settled: the catalog JSON, or null when it could not be read
let catalogDefs = null; // the node definitions of the last refresh, so one refresh reads the catalog once

function getCatalog() {
	if (!catalog) {
		catalog = callJson(ROUTE)
			.then((data) => {
				for (const [key, family] of Object.entries(data.families ?? {})) {
					if (family.error) console.warn(`[BCNodes] LM catalog, family ${key}: ${family.error}`);
				}
				return data;
			})
			.catch((e) => {
				console.warn("[BCNodes] LM: could not read the model catalog; precisions and config defaults are left as they are", e);
				return null;
			});
	}
	return catalog;
}

// fn(catalog) once the catalog is known (never before the caller returns); nothing when it could not be read.
function withCatalog(fn) {
	getCatalog()
		.then((cat) => cat && fn(cat))
		.catch((e) => console.error("[BCNodes]", e));
}

// A node-definition refresh calls refreshComboInNode(defs) on every node, then the extensions'
// refreshComboInNodes(defs), with the same defs; it may follow an edit of a user models.yaml, so the
// catalog is read again, once.
function catalogRefreshed(defs) {
	if (defs === catalogDefs) return;
	catalogDefs = defs;
	catalog = null;
}

// The catalog family of an LM node; undefined for a node type the catalog does not know.
function catalogFamily(cat, node) {
	return Object.values(cat.families ?? {}).find((f) => f.node === node.type);
}

// The catalog entry of the model an LM node has selected; undefined for a node type or a model the
// catalog does not know.
function catalogModel(cat, node) {
	const family = catalogFamily(cat, node);
	const name = widget(node, "model")?.value;
	return family?.models && Object.hasOwn(family.models, name) ? family.models[name] : undefined;
}

// ---------------------------------------------------------------------------
// LM node
// ---------------------------------------------------------------------------

// modelChanged: a user model change, which also fits the values.
function applyModel(node, cat, modelChanged = false) {
	const model = catalogModel(cat, node);
	if (!model) return;
	const fit = modelChanged && !catalogFamily(cat, node).error;
	const precision = widget(node, "precision");
	if (precision && model.precisions?.length) {
		// A new array: the node definition's list is shared by every node of the type.
		precision.options.values = [...model.precisions];
		if (fit && !model.precisions.includes(precision.value)) precision.value = model.precisions[0];
	}
	const thinking = widget(node, "thinking");
	if (thinking) {
		if (fit && !model.thinking && thinking.value) thinking.value = false;
		// A thinking left on stays enabled, so it can be turned off.
		thinking.disabled = !model.thinking && !thinking.value;
	}
	node.setDirtyCanvas?.(true, false);
}

// The config node plugged into this LM node shows its defaults when this is its first LM node.
function refreshLinkedConfig(node, cat) {
	const slot = node.findInputSlot("config");
	const config = slot >= 0 && node.graph ? node.getInputNode(slot) : null;
	if (config?.type === cat.config_node) refreshDefaults(config, cat);
}

function installLMNode(nodeType) {
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		const node = this;
		afterChange(widget(node, "model"), () =>
			withCatalog((cat) => {
				applyModel(node, cat, true);
				refreshLinkedConfig(node, cat);
			}),
		);
		after(widget(node, "thinking"), "callback", () =>
			withCatalog((cat) => {
				applyModel(node, cat);
				refreshLinkedConfig(node, cat);
			}),
		);
		withCatalog((cat) => applyModel(node, cat));
	});
	after(proto, "onConfigure", function () {
		see(widget(this, "model"));
		withCatalog((cat) => applyModel(this, cat));
	});
	after(proto, "refreshComboInNode", function (defs) {
		// Runs before the refresh puts the full precision list back, hence after the catalog read.
		catalogRefreshed(defs);
		withCatalog((cat) => applyModel(this, cat));
	});
}

// ---------------------------------------------------------------------------
// LM Config node
// ---------------------------------------------------------------------------

function editedNames(node) {
	const raw = String(widget(node, EDITED)?.value ?? "");
	return [...new Set(raw.split(",").map((s) => s.trim()).filter(Boolean))];
}

// Fields in widget order, then any other name as it was (the backend's error names it; Reset clears
// it), so one set of fields always gives the same string: `edited` is part of the cache key.
function setEdited(node, names) {
	const w = widget(node, EDITED);
	if (!w) return;
	w.value = [...FIELDS.filter((f) => names.includes(f)), ...names.filter((n) => !FIELDS.includes(n))].join(", ");
	node.setDirtyCanvas?.(true, false);
}

function markEdited(node, name) {
	const names = editedNames(node);
	if (!names.includes(name)) setEdited(node, [...names, name]);
}

// The first LM node the config output is linked to, among the catalog's family nodes.
function firstLinkedLM(node, cat) {
	const types = new Set(Object.values(cat.families ?? {}).map((f) => f.node));
	const slot = node.outputs?.findIndex((o) => o.type === CONFIG_TYPE) ?? -1;
	if (slot < 0 || !node.graph) return null;
	return node.getOutputNodes(slot)?.find((n) => types.has(n.type)) ?? null;
}

// Un-edited fields take the linked LM node's defaults for its model and mode. Values are set from
// code, so the field callbacks (which mark a field edited) are not called. Unlinked, or a model the
// catalog does not know: the values stay.
function refreshDefaults(node, cat) {
	const lm = firstLinkedLM(node, cat);
	const model = lm && catalogModel(cat, lm);
	if (!model) return;
	const mode = model.thinking && widget(lm, "thinking")?.value ? "thinking_on" : "thinking_off";
	const defaults = model.defaults?.[mode];
	if (!defaults) return;
	const edited = editedNames(node);
	for (const name of FIELDS) {
		const w = widget(node, name);
		if (!w || edited.includes(name) || !(name in defaults)) continue;
		const value = defaults[name];
		// A number outside the widget's range (a user models.yaml may give temperature 0) is shown at
		// the nearest end: ComfyUI refuses a prompt with a widget value out of range, even on a field
		// the backend leaves out (it still applies the model's own default).
		if (w.type === "combo") w.value = String(value);
		else if (typeof value === "number") w.value = Math.min(Math.max(value, w.options?.min ?? -Infinity), w.options?.max ?? Infinity);
		else w.value = value;
		see(w);
	}
	node.setDirtyCanvas?.(true, false);
}

function resetToDefaults(node) {
	setEdited(node, []);
	withCatalog((cat) => refreshDefaults(node, cat));
}

// Node canvas: an un-edited field is drawn at the disabled look's alpha but keeps taking input (a
// disabled widget takes none and hides its value). The Vue renderer never calls drawWidget for
// these widget types; it draws its own components.
function greyUnlessEdited(node, w) {
	const draw = w.drawWidget;
	if (typeof draw !== "function") return;
	w.drawWidget = function (ctx, options) {
		if (editedNames(node).includes(this.name)) return draw.call(this, ctx, options);
		ctx.save();
		try {
			ctx.globalAlpha *= DEFAULT_ALPHA;
			draw.call(this, ctx, options);
		} finally {
			ctx.restore();
		}
	};
}

function installConfigNode(nodeType) {
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		const node = this;
		// Hidden in both renderers: the node canvas skips a hidden widget in layout and drawing, the
		// Vue renderer filters on options.hidden, which this sets. It is still serialized.
		const edited = widget(node, EDITED);
		if (edited) edited.hidden = true;
		for (const name of FIELDS) {
			const w = widget(node, name);
			if (!w) continue;
			afterChange(w, () => markEdited(node, name));
			greyUnlessEdited(node, w);
		}
		node.setSize([node.size[0], node.computeSize()[1]]);
	});

	after(proto, "onConfigure", function () {
		for (const name of FIELDS) see(widget(this, name));
	});

	// The defaults follow the output's links. A link into a field needs nothing here: the backend applies it.
	after(proto, "onConnectionsChange", function (side) {
		if (app.configuringGraph) return;
		if (side === LiteGraph.OUTPUT) withCatalog((cat) => refreshDefaults(this, cat));
	});

	// The node menu: the legacy canvas and the Vue renderer's menu both read getExtraMenuOptions.
	after(proto, "getExtraMenuOptions", function (canvas, options) {
		options.push({ content: "Reset to model defaults", callback: () => resetToDefaults(this) });
	});

	after(proto, "refreshComboInNode", function (defs) {
		catalogRefreshed(defs);
		withCatalog((cat) => refreshDefaults(this, cat));
	});
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

// The config node makes a BC_LM_CONFIG; an LM node takes one on its `config` input.
function isConfigDef(nodeData) {
	return (nodeData?.output ?? []).includes(CONFIG_TYPE);
}

function isLMDef(nodeData) {
	const input = nodeData?.input ?? {};
	return [input.required?.config, input.optional?.config].some((spec) => spec?.[0] === CONFIG_TYPE);
}

app.registerExtension({
	name: "BCNodes.LM",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (isConfigDef(nodeData)) installConfigNode(nodeType);
		else if (isLMDef(nodeData)) installLMNode(nodeType);
	},

	// Called once per node-definition refresh with the defs the nodes got, so a refresh with no LM
	// node in the graph still drops the cached catalog.
	refreshComboInNodes(defs) {
		catalogRefreshed(defs);
	},
});

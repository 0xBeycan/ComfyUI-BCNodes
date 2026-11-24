import { app } from "../../../scripts/app.js";
import { callJson } from "./bcnodes_api.js";
import { after, afterChange, editedNames, installEditedFields, see, setEdited, widget } from "./edited_fields.js";

// LM nodes (Qwen LM, and any later family node), LM Loader and LM Config — the frontend half of nodes/lm.py.
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
// LM Loader (the catalog's loader_node): its family is the catalog family whose `name` its `family` widget
// holds (the family's display name; the catalog's keys are internal), and its model, precision and lora combos
// list only that family's models, the selected model's precisions and that family's LoRAs (None first), by the
// same rules: a user family change also fits the values (each kept when the new list has it, else the list's
// first: the family's first model, its first precision, None), a user model change fits the precision; not
// while that family's catalog has an error. A family the catalog does not know changes nothing.
//
// The frontend also calls a widget's callback with its value unchanged (Rename Widget, a subgraph
// promote): only a callback with a value other than the one last seen is a change.
//
// LM Config: the backend applies the fields named in the hidden `edited` widget and every field fed by
// a link (it reads the links from the prompt; nothing here marks them); edited_fields.js keeps `edited`
// and draws the un-edited fields greyed. The other fields show the sampling defaults of the first LM
// node the config output is linked to (its model, thinking on / off), refreshed when that link
// changes, when that node's model or thinking changes, on "Reset to model defaults" (node menu; clears
// `edited`) and on a node-definition refresh. Never on load: a saved workflow's values come back as saved.

const ROUTE = "/bcnodes/lm/catalog";
const CONFIG_TYPE = "BC_LM_CONFIG";
const LM_TYPE = "BC_LM"; // LM Loader's output
// BC_LMConfig's fields in widget order; `edited` lists them in this order.
const FIELDS = ["do_sample", "temperature", "top_k", "top_p", "min_p", "repetition_penalty", "presence_penalty", "mtp"];
// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

let catalog = null; // the fetch, pending or settled: the catalog JSON, or null when it could not be read
let catalogDefs = null; // the node definitions of the last refresh, so one refresh reads the catalog once

function getCatalog() {
	if (!catalog) {
		catalog = callJson(ROUTE)
			.then((data) => {
				for (const family of Object.values(data.families ?? {})) {
					if (family.error) console.warn(`[BCNodes] LM catalog, family ${family.name}: ${family.error}`);
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

// The catalog family of an LM node, or on LM Loader the one named by its family widget; undefined for a node
// type or a family the catalog does not know.
function catalogFamily(cat, node) {
	const families = Object.values(cat.families ?? {});
	if (node.type === cat.loader_node) {
		const name = widget(node, "family")?.value;
		return families.find((f) => f.name === name);
	}
	return families.find((f) => f.node === node.type);
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

// The combo w lists `values` (a new array: the node definition's list is shared by every node of the type);
// with `fit`, a value the list does not hold becomes its first.
function narrow(w, values, fit) {
	if (!w || !values?.length) return;
	w.options.values = [...values];
	if (fit && !values.includes(w.value)) {
		w.value = values[0];
		see(w);
	}
}

// modelChanged: a user model (or LM Loader family) change, which also fits the values.
function applyModel(node, cat, modelChanged = false) {
	const model = catalogModel(cat, node);
	if (!model) return;
	const fit = modelChanged && !catalogFamily(cat, node).error;
	narrow(widget(node, "precision"), model.precisions, fit);
	const thinking = widget(node, "thinking");
	if (thinking) {
		if (fit && !model.thinking && thinking.value) thinking.value = false;
		// A thinking left on stays enabled, so it can be turned off.
		thinking.disabled = !model.thinking && !thinking.value;
	}
	node.setDirtyCanvas?.(true, false);
}

// LM Loader: the model and lora combos list the selected family's, then the precision follows the model.
// familyChanged: a user family change, which also fits the values. On an LM node (no family widget): applyModel.
function applyFamily(node, cat, familyChanged = false) {
	if (widget(node, "family")) {
		const family = catalogFamily(cat, node);
		if (!family?.models) return;
		const fit = familyChanged && !family.error;
		narrow(widget(node, "model"), Object.keys(family.models), fit);
		narrow(widget(node, "lora"), family.loras, fit);
	}
	applyModel(node, cat, familyChanged);
}

// The config node plugged into this LM node shows its defaults when this is its first LM node.
function refreshLinkedConfig(node, cat) {
	const slot = node.findInputSlot("config");
	const config = slot >= 0 && node.graph ? node.getInputNode(slot) : null;
	if (config?.type === cat.config_node) refreshDefaults(config, cat);
}

// An LM node or LM Loader (a family widget, no thinking, no config input: the calls on the missing ones do
// nothing).
function installLMNode(nodeType) {
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		const node = this;
		afterChange(widget(node, "family"), () => withCatalog((cat) => applyFamily(node, cat, true)));
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
		withCatalog((cat) => applyFamily(node, cat));
	});
	after(proto, "onConfigure", function () {
		see(widget(this, "family"));
		see(widget(this, "model"));
		withCatalog((cat) => applyFamily(this, cat));
	});
	after(proto, "refreshComboInNode", function (defs) {
		// Runs before the refresh puts the full lists back, hence after the catalog read.
		catalogRefreshed(defs);
		withCatalog((cat) => applyFamily(this, cat));
	});
}

// ---------------------------------------------------------------------------
// LM Config node
// ---------------------------------------------------------------------------

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
	setEdited(node, [], FIELDS);
	withCatalog((cat) => refreshDefaults(node, cat));
}

function installConfigNode(nodeType) {
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		installEditedFields(this, FIELDS);
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

// The config node makes a BC_LM_CONFIG.
function isConfigDef(nodeData) {
	return (nodeData?.output ?? []).includes(CONFIG_TYPE);
}

// An LM node takes a BC_LM_CONFIG on its `config` input; LM Loader makes a BC_LM.
function isLMDef(nodeData) {
	const input = nodeData?.input ?? {};
	return (
		[input.required?.config, input.optional?.config].some((spec) => spec?.[0] === CONFIG_TYPE) ||
		(nodeData?.output ?? []).includes(LM_TYPE)
	);
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

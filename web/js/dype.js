import { app } from "../../../scripts/app.js";
import { after, installEditedFields, see, setEdited, widget } from "./edited_fields.js";

// DyPE Config — the frontend half of nodes/dype.py. The backend applies the fields named in the hidden `edited`
// widget and every field fed by a link; edited_fields.js keeps `edited` and draws the un-edited fields greyed.
// The un-edited fields show the node definition's defaults (Z-Image's automatic values; dype_exponent's, from the
// scale, has no single value). "Reset to automatic values" (node menu) clears `edited` and puts those defaults
// back. Never on load: a saved workflow's values come back as saved.

const CONFIG_TYPE = "BC_DYPE_CONFIG";
// BC_DyPEConfig's fields in widget order; `edited` lists them in this order.
const FIELDS = ["method", "base_resolution", "dype_scale", "dype_exponent", "auto_threshold"];

function resetToDefaults(node, defaults) {
	setEdited(node, [], FIELDS);
	for (const name of FIELDS) {
		const w = widget(node, name);
		if (!w || !(name in defaults)) continue;
		w.value = defaults[name];
		see(w);
	}
	node.setDirtyCanvas?.(true, false);
}

function installConfigNode(nodeType, nodeData) {
	const required = nodeData?.input?.required ?? {};
	const defaults = Object.fromEntries(FIELDS.filter((name) => required[name]?.[1] && "default" in required[name][1])
		.map((name) => [name, required[name][1].default]));
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		installEditedFields(this, FIELDS);
	});
	after(proto, "onConfigure", function () {
		for (const name of FIELDS) see(widget(this, name));
	});
	// The node menu: the legacy canvas and the Vue renderer's menu both read getExtraMenuOptions.
	after(proto, "getExtraMenuOptions", function (canvas, options) {
		options.push({ content: "Reset to automatic values", callback: () => resetToDefaults(this, defaults) });
	});
}

app.registerExtension({
	name: "BCNodes.DyPE",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if ((nodeData?.output ?? []).includes(CONFIG_TYPE)) installConfigNode(nodeType, nodeData);
	},
});

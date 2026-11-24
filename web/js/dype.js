import { app } from "../../../scripts/app.js";
import { after, afterChange, installEditedFields, see, setEdited, showValues, widget } from "./edited_fields.js";

// DyPE Config — the frontend half of nodes/dype.py. The backend applies the fields named in the hidden `edited`
// widget and every field fed by a link; edited_fields.js keeps `edited` and draws the un-edited fields greyed.
// The un-edited fields show the automatic values of the model selected in `model` (the node definition gives each
// model's under the model input's `automatic`; dype_exponent's, from the scale, has no single value and shows its
// widget default). A user model change shows the new model's; "Reset to automatic values" (node menu) clears
// `edited` and shows the selected model's. Never on load: a saved workflow's values come back as saved.

const CONFIG_TYPE = "BC_DYPE_CONFIG";
// BC_DyPEConfig's fields in widget order; `edited` lists them in this order. `model` is not one: it is always applied.
const FIELDS = ["method", "base_resolution", "dype_scale", "dype_exponent", "auto_threshold", "yarn_alt_scaling"];

function installConfigNode(nodeType, nodeData) {
	const required = nodeData?.input?.required ?? {};
	const defaults = Object.fromEntries(FIELDS.filter((name) => required[name]?.[1] && "default" in required[name][1])
		.map((name) => [name, required[name][1].default]));
	const automatic = required.model?.[1]?.automatic ?? {};
	// The values the un-edited fields show for the selected model: its automatic values over the widget defaults.
	const values = (node) => ({ ...defaults, ...(automatic[widget(node, "model")?.value] ?? {}) });
	const proto = nodeType.prototype;
	after(proto, "onNodeCreated", function () {
		installEditedFields(this, FIELDS);
		afterChange(widget(this, "model"), () => showValues(this, FIELDS, values(this)));
	});
	after(proto, "onConfigure", function () {
		for (const name of ["model", ...FIELDS]) see(widget(this, name));
	});
	// The node menu: the legacy canvas and the Vue renderer's menu both read getExtraMenuOptions.
	after(proto, "getExtraMenuOptions", function (canvas, options) {
		options.push({
			content: "Reset to automatic values",
			callback: () => {
				setEdited(this, [], FIELDS);
				showValues(this, FIELDS, values(this));
			},
		});
	});
}

app.registerExtension({
	name: "BCNodes.DyPE",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if ((nodeData?.output ?? []).includes(CONFIG_TYPE)) installConfigNode(nodeType, nodeData);
	},
});

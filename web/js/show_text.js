import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Show Text — one read-only, growing text box per element of the incoming
// text (a list gives several boxes). The boxes are re-created from the saved
// widget values when a workflow is loaded.

const NODE_TYPE = "BC_ShowText";
const VALUES = Symbol("bcnodes.showtext.values");

function clearBoxes(node) {
	const keep = node.widgets?.filter((w) => !String(w.name ?? "").startsWith("text_")) ?? [];
	for (const w of node.widgets ?? []) if (!keep.includes(w)) w.onRemove?.();
	node.widgets?.splice(0, node.widgets.length, ...keep);
}

function populate(node, text) {
	clearBoxes(node);
	let values = Array.isArray(text) ? [...text] : [text];
	values = values.flatMap((v) => (Array.isArray(v) ? v : [v]));
	if (values.length > 1 && !values[0]) values.shift();
	for (const value of values) {
		const w = ComfyWidgets.STRING(node, `text_${node.widgets?.length ?? 0}`, ["STRING", { multiline: true }], app).widget;
		// Saved with the workflow, kept out of the prompt: a box in the prompt
		// would change the node's cache key every time the shown text changes
		// and re-run everything below it once more.
		w.options = w.options ?? {};
		w.options.serialize = false;
		if (w.inputEl) {
			w.inputEl.readOnly = true;
			w.inputEl.style.opacity = 0.6;
		}
		w.value = value == null ? "" : String(value);
	}
	requestAnimationFrame(() => {
		const size = node.computeSize();
		size[0] = Math.max(size[0], node.size[0]);
		size[1] = Math.max(size[1], node.size[1]);
		node.setSize(size);
		app.graph?.setDirtyCanvas(true, false);
	});
}

app.registerExtension({
	name: "BCNodes.ShowText",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message, ...rest) {
			const r = onExecuted?.apply(this, [message, ...rest]);
			if (message?.text !== undefined) populate(this, message.text);
			return r;
		};

		// The frontend rebuilds widgets from the node def on configure, so the
		// saved values are captured first and the boxes recreated afterwards.
		const configure = nodeType.prototype.configure;
		nodeType.prototype.configure = function (info, ...rest) {
			this[VALUES] = info?.widgets_values;
			return configure?.apply(this, [info, ...rest]);
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			try {
				const values = this[VALUES];
				if (values?.length) requestAnimationFrame(() => populate(this, values));
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};
	},
});

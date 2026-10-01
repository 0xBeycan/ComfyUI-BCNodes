import { app } from "../../../scripts/app.js";

// Save Image With Caption — `filename_prefix` gets the frontend's text replacements when the prompt
// is queued (%date:yyyy-MM-dd%, %Empty Latent Image.width%), as ComfyUI's own Save Image does; the
// frontend applies them by itself only to its core save nodes. The server fills in the rest
// (%year% ... %second%, %width%, %height%, %batch_num%).

const NODE_TYPE = "BC_SaveImageWithCaption";

app.registerExtension({
	name: "BCNodes.SaveImageWithCaption",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			const widget = this.widgets?.find((w) => w.name === "filename_prefix");
			const applyTextReplacements = window.comfyAPI?.utils?.applyTextReplacements;
			if (widget && applyTextReplacements) widget.serializeValue = () => applyTextReplacements(app, widget.value);
			else console.warn("[BCNodes] Save Image With Caption: this frontend has no applyTextReplacements; %date:...% and %Node.widget% are saved as typed");
			return r;
		};
	},
});

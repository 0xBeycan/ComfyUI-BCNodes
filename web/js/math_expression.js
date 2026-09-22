import { app } from "../../../scripts/app.js";

// Math Expression — draws the last result in the node's corner and narrows
// the a / b / c sockets to the types the expression can use.

const NODE_TYPE = "BC_MathExpression";
const INPUT_TYPES = "INT,FLOAT,IMAGE,LATENT";

app.registerExtension({
	name: "BCNodes.MathExpression",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				for (const input of this.inputs ?? []) {
					if (["a", "b", "c"].includes(input.name)) input.type = INPUT_TYPES;
				}
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx, ...rest) {
			const r = onDrawForeground?.apply(this, [ctx, ...rest]);
			const out = app.nodeOutputs?.[String(this.id)];
			const value = out?.value?.[0];
			if (this.flags?.collapsed || value === undefined) return r;
			const text = String(value);
			ctx.save();
			ctx.font = "bold 12px sans-serif";
			ctx.fillStyle = "dodgerblue";
			ctx.textAlign = "right";
			ctx.fillText(text, this.size[0] - 6, LiteGraph.NODE_SLOT_HEIGHT * 3);
			ctx.restore();
			return r;
		};
	},
});

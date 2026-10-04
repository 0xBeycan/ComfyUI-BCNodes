import { app } from "../../../scripts/app.js";

// Math Expression — shows the last result and narrows the a / b / c sockets
// to the types the expression can use. The node canvas draws the result in
// the node's corner (onDrawForeground); the Vue renderer never calls that, so
// there a result row shows it instead. The row is not serialized and takes no
// height on the node canvas.

const NODE_TYPE = "BC_MathExpression";
const INPUT_TYPES = "INT,FLOAT,IMAGE,LATENT";
const RESULT_TYPE = "BC_MATH_RESULT";

function lastResult(node) {
	return app.nodeOutputs?.[String(node.id)]?.value?.[0];
}

class ResultRow {
	constructor() {
		this.type = RESULT_TYPE;
		this.name = "result";
		this.value = null;
		this.options = { serialize: false };
		this.serialize = false;
	}

	// -4: the node canvas adds 4 per widget, so the row takes no space there.
	computeSize(width) {
		return [width, LiteGraph.vueNodesMode ? LiteGraph.NODE_WIDGET_HEIGHT : -4];
	}

	draw(ctx, node, width, y, height) {
		if (!LiteGraph.vueNodesMode) return;
		const value = lastResult(node);
		ctx.save();
		ctx.font = "bold 12px sans-serif";
		ctx.textAlign = "right";
		ctx.fillStyle = value === undefined ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR : "dodgerblue";
		ctx.fillText(value === undefined ? "no result yet" : String(value), width - 15, y + height / 2 + 4);
		ctx.restore();
	}
}

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
				this.addCustomWidget(new ResultRow());
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		// The Vue renderer paints the row in a canvas of its own, only on triggerDraw.
		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (...args) {
			const r = onExecuted?.apply(this, args);
			this.widgets?.find((w) => w.type === RESULT_TYPE)?.triggerDraw?.();
			return r;
		};

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (ctx, ...rest) {
			const r = onDrawForeground?.apply(this, [ctx, ...rest]);
			const value = lastResult(this);
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

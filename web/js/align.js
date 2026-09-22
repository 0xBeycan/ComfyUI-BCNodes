import { app } from "../../../scripts/app.js";

// Align — alignment and distribution buttons for the selection toolbox.
// Works on whatever the canvas can select: nodes, groups, reroutes and
// subgraph nodes. Moving a group moves its contents; a node that sits inside a
// selected group is left to the group so it is not moved twice. Every action
// is one undo step.

const PREFIX = "BCNodes.Align.";

const ACTIONS = [
	{ id: "Left", label: "Align left", icon: "pi pi-align-left", axis: 0, mode: "min" },
	{ id: "CenterH", label: "Align horizontal centers", icon: "pi pi-align-center", axis: 0, mode: "center" },
	{ id: "Right", label: "Align right", icon: "pi pi-align-right", axis: 0, mode: "max" },
	{ id: "Top", label: "Align top", icon: "pi pi-align-left bcnodes-rot90", axis: 1, mode: "min" },
	{ id: "CenterV", label: "Align vertical centers", icon: "pi pi-align-center bcnodes-rot90", axis: 1, mode: "center" },
	{ id: "Bottom", label: "Align bottom", icon: "pi pi-align-right bcnodes-rot90", axis: 1, mode: "max" },
	{ id: "DistributeH", label: "Distribute horizontally", icon: "pi pi-arrows-h", axis: 0, mode: "distribute" },
	{ id: "DistributeV", label: "Distribute vertically", icon: "pi pi-arrows-v", axis: 1, mode: "distribute" },
];

function ensureStyle() {
	if (document.getElementById("bcnodes-align-style")) return;
	const style = document.createElement("style");
	style.id = "bcnodes-align-style";
	style.textContent = ".bcnodes-rot90{display:inline-block;transform:rotate(90deg)}";
	document.head.appendChild(style);
}

// Selected items minus those already carried by a selected group.
function topLevelSelection(canvas) {
	const items = [...(canvas?.selectedItems ?? [])];
	const groups = items.filter((i) => i.children instanceof Set || i._children instanceof Set);
	return items.filter((item) => !groups.some((g) => g !== item && (g.children ?? g._children).has(item)));
}

// [x, y, w, h] from pos / size (boundingRect is only refreshed while the
// canvas draws, so it can be stale right after a move). A node's rect
// includes its title bar; a group's pos already is its top-left corner.
function rectOf(item) {
	const w = item.size?.[0] ?? 0;
	const h = item.size?.[1] ?? 0;
	const isNode = typeof item.mode === "number" && item.constructor?.name !== "LGraphGroup";
	const title = isNode && !item.flags?.collapsed ? LiteGraph.NODE_TITLE_HEIGHT : 0;
	return [item.pos[0], item.pos[1] - title, w, h + title];
}

function moveBy(item, dx, dy) {
	if (!dx && !dy) return;
	if (typeof item.move === "function") item.move(dx, dy);
	else item.pos = [item.pos[0] + dx, item.pos[1] + dy];
}

function deltas(items, axis, mode) {
	const rects = items.map(rectOf);
	const start = (r) => r[axis];
	const size = (r) => r[axis + 2];
	const end = (r) => start(r) + size(r);

	if (mode === "distribute") {
		if (items.length < 3) return [];
		const order = rects.map((r, i) => i).sort((a, b) => start(rects[a]) - start(rects[b]));
		const first = rects[order[0]];
		const last = rects[order[order.length - 1]];
		const span = end(last) - start(first);
		const total = order.reduce((sum, i) => sum + size(rects[i]), 0);
		const gap = (span - total) / (order.length - 1);
		let cursor = start(first);
		const out = [];
		for (const i of order) {
			out.push([i, cursor - start(rects[i])]);
			cursor += size(rects[i]) + gap;
		}
		return out;
	}

	const lo = Math.min(...rects.map(start));
	const hi = Math.max(...rects.map(end));
	return rects.map((r, i) => {
		if (mode === "min") return [i, lo - start(r)];
		if (mode === "max") return [i, hi - end(r)];
		return [i, (lo + hi) / 2 - (start(r) + size(r) / 2)];
	});
}

function run(action) {
	const canvas = app.canvas;
	const graph = canvas?.graph;
	const items = topLevelSelection(canvas);
	if (items.length < 2) return;
	const moves = deltas(items, action.axis, action.mode).filter(([, d]) => Math.abs(d) > 0.01);
	if (!moves.length) return;

	graph?.beforeChange?.();
	for (const [i, d] of moves) moveBy(items[i], action.axis === 0 ? d : 0, action.axis === 1 ? d : 0);
	graph?.afterChange?.();
	canvas.setDirty(true, true);
}

app.registerExtension({
	name: "BCNodes.Align",

	commands: ACTIONS.map((a) => ({
		id: PREFIX + a.id,
		label: a.label,
		icon: a.icon,
		function: () => run(a),
	})),

	// Shown when two or more items are selected; distribute needs three.
	getSelectionToolboxCommands() {
		const count = app.canvas?.selectedItems?.size ?? 0;
		if (count < 2) return [];
		return ACTIONS.filter((a) => a.mode !== "distribute" || count >= 3).map((a) => PREFIX + a.id);
	},

	setup() {
		ensureStyle();
	},
});

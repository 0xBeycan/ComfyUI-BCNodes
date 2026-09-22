import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Power Lora Loader — one row per LoRA, drawn as a single canvas widget:
//   [●] lora file name ............ [◀ 1.00 ▶]
// Click the dot to toggle, the name to pick a file, the arrows to step the
// strength by 0.05, the number to type it. "+ Add LoRA" appends a row; a
// right click on a row toggles, moves or removes it. Each row is serialized
// as {on, lora, strength} and reaches the backend as lora_N in row order. A
// row keeps the name it was created with (numbers may have gaps): current
// frontends route widget renames through a store that silently refuses a
// name already in use, so rows are never renamed.

const NODE_TYPE = "BC_PowerLoraLoader";
const ROW_TYPE = "BC_LORA_ROW";
const STEP = 0.05;
const NONE = "None";

let loraList = null;
async function getLoraList() {
	if (loraList) return loraList;
	try {
		const res = await api.fetchApi("/object_info/LoraLoader");
		const info = await res.json();
		loraList = info?.LoraLoader?.input?.required?.lora_name?.[0] ?? [];
	} catch (e) {
		console.warn("[BCNodes] Power Lora Loader: could not list LoRAs", e);
		loraList = [];
	}
	return loraList;
}

function fmt(v) {
	return (Math.round(v * 100) / 100).toFixed(2);
}

class LoraRow {
	constructor(node, name, value) {
		this.type = ROW_TYPE;
		this.name = name;
		this.node = node;
		this.value = { on: true, lora: NONE, strength: 1.0, ...(value ?? {}) };
		this.options = { serialize: true };
		this.serialize = true;
		this.zones = {};
	}

	serializeValue() {
		return { on: !!this.value.on, lora: this.value.lora, strength: Number(this.value.strength) || 0 };
	}

	computeSize(width) {
		return [width, LiteGraph.NODE_WIDGET_HEIGHT];
	}

	draw(ctx, node, width, y, height) {
		// On the node canvas the row spans the node. The Vue-nodes legacy renderer
		// leaves its own width on the widget (`widget.width || nodeWidth`), which
		// would otherwise stick after switching back to the classic canvas.
		if (ctx.canvas === app.canvas?.canvas) width = node.size[0];
		const margin = 15;
		const left = margin;
		const right = width - margin;
		const midY = y + height / 2;
		const on = !!this.value.on;

		ctx.save();
		ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
		ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
		ctx.beginPath();
		ctx.roundRect(left, y, right - left, height, [height * 0.5]);
		ctx.fill();
		ctx.stroke();

		// toggle
		const tx = left + 14;
		ctx.beginPath();
		ctx.arc(tx, midY, 6, 0, Math.PI * 2);
		ctx.fillStyle = on ? "#7fbf6a" : "#555";
		ctx.fill();
		this.zones.toggle = [left, left + 28];

		// strength box
		const boxW = 92;
		const bx = right - boxW - 6;
		ctx.fillStyle = on ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
		ctx.font = "12px sans-serif";
		ctx.textAlign = "center";
		ctx.fillText("◀", bx + 10, midY + 4);
		ctx.fillText(fmt(this.value.strength), bx + boxW / 2, midY + 4);
		ctx.fillText("▶", bx + boxW - 10, midY + 4);
		this.zones.dec = [bx, bx + 20];
		this.zones.value = [bx + 20, bx + boxW - 20];
		this.zones.inc = [bx + boxW - 20, bx + boxW];

		// name
		const nameX = left + 30;
		const nameW = bx - nameX - 8;
		ctx.textAlign = "left";
		let label = this.value.lora || NONE;
		while (label.length > 4 && ctx.measureText(label).width > nameW) label = label.slice(0, -2);
		if (label !== (this.value.lora || NONE)) label += "…";
		ctx.fillText(label, nameX, midY + 4);
		this.zones.name = [nameX, bx - 4];
		ctx.restore();
	}

	mouse(event, pos, node) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		if (event.button === 2) return false;
		const x = pos[0];
		const within = (z) => z && x >= z[0] && x <= z[1];
		if (within(this.zones.toggle)) {
			this.value.on = !this.value.on;
		} else if (within(this.zones.dec)) {
			this.value.strength = Math.round((Number(this.value.strength) - STEP) * 100) / 100;
		} else if (within(this.zones.inc)) {
			this.value.strength = Math.round((Number(this.value.strength) + STEP) * 100) / 100;
		} else if (within(this.zones.value)) {
			app.canvas.prompt("Strength", this.value.strength, (v) => {
				const n = Number(v);
				if (!Number.isNaN(n)) this.value.strength = n;
				node.setDirtyCanvas(true, false);
			}, event);
		} else if (within(this.zones.name)) {
			getLoraList().then((list) => {
				// "dark" is what the frontend's Comfy.ContextMenuFilter keys on:
				// menus with that class and more than four entries get a filter box.
				new LiteGraph.ContextMenu([NONE, ...list], {
					event,
					title: "Choose a LoRA",
					className: "dark",
					scale: Math.max(1, app.canvas.ds?.scale ?? 1),
					callback: (v) => {
						this.value.lora = v;
						node.setDirtyCanvas(true, false);
					},
				});
			});
		} else {
			return false;
		}
		node.setDirtyCanvas(true, false);
		return true;
	}
}

function rows(node) {
	return node.widgets.filter((w) => w.type === ROW_TYPE);
}

function relayout(node) {
	node.setSize([node.size[0], node.computeSize()[1]]);
	node.setDirtyCanvas(true, true);
}

function nextRowName(node) {
	let n = 0;
	for (const w of rows(node)) {
		const m = /^lora_(\d+)$/.exec(w.name ?? "");
		if (m) n = Math.max(n, Number(m[1]));
	}
	return `lora_${n + 1}`;
}

function addRow(node, value) {
	const w = new LoraRow(node, nextRowName(node), value);
	node.addCustomWidget(w);
	// keep the add button last
	const add = node.widgets.find((x) => x.name === "add_lora");
	if (add) {
		node.widgets.splice(node.widgets.indexOf(add), 1);
		node.widgets.push(add);
	}
	relayout(node);
	return w;
}

function moveRow(node, w, dir) {
	const list = rows(node);
	const i = list.indexOf(w);
	const j = i + dir;
	if (i < 0 || j < 0 || j >= list.length) return;
	const a = node.widgets.indexOf(list[i]);
	const b = node.widgets.indexOf(list[j]);
	[node.widgets[a], node.widgets[b]] = [node.widgets[b], node.widgets[a]];
	relayout(node);
}

function removeRow(node, w) {
	const i = node.widgets.indexOf(w);
	if (i >= 0) node.widgets.splice(i, 1);
	relayout(node);
}

// The canvas does not pass the event on to getSlotMenuOptions; the row menu
// needs it for its position.
let lastPointerEvent = null;
document.addEventListener("pointerdown", (e) => (lastPointerEvent = e), true);

// What the canvas would offer for a real slot if the node had no getSlotMenuOptions.
function defaultSlotMenu(slot) {
	const options = [];
	if (slot?.output?.links?.length || slot?.input?.link != null) options.push({ content: "Disconnect Links", slot });
	const s = slot?.input || slot?.output;
	if (!s) return options;
	if (!s.nameLocked && !("link" in s && s.widget)) options.push({ content: "Rename Slot", slot });
	if (s.removable) options.push(null, s.locked ? "Cannot remove" : { content: "Remove Slot", slot, className: "danger" });
	return options;
}

app.registerExtension({
	name: "BCNodes.PowerLoraLoader",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				const add = this.addWidget("button", "add_lora", null, () => addRow(this));
				add.label = "➕ Add LoRA";
				add.serialize = false;
				add.options = add.options ?? {};
				add.options.serialize = false;
				getLoraList();
				this.setSize([Math.max(this.size[0], 340), this.computeSize()[1]]);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		// Rows are recreated from the saved values before litegraph applies them.
		const configure = nodeType.prototype.configure;
		nodeType.prototype.configure = function (info, ...rest) {
			const values = Array.isArray(info?.widgets_values) ? info.widgets_values : Object.values(info?.widgets_values_named ?? {});
			for (const w of rows(this)) this.widgets.splice(this.widgets.indexOf(w), 1);
			for (const v of values) if (v && typeof v === "object" && "lora" in v) addRow(this, v);
			return configure?.apply(this, [info, ...rest]);
		};

		// Right-click on a row: its own menu. The canvas asks getSlotInPosition
		// first and skips its default menu when getSlotMenuOptions returns
		// nothing, so a row is reported as a pseudo slot and handled here.
		const getSlotInPosition = nodeType.prototype.getSlotInPosition;
		nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
			const slot = getSlotInPosition?.apply(this, [canvasX, canvasY]);
			if (slot) return slot;
			const y = canvasY - this.pos[1];
			const row = rows(this).find((w) => w.last_y != null && y >= w.last_y && y < w.last_y + LiteGraph.NODE_WIDGET_HEIGHT);
			return row ? { widget: row, output: { type: ROW_TYPE } } : slot;
		};

		nodeType.prototype.getSlotMenuOptions = function (slot) {
			const row = slot?.widget;
			if (row?.type !== ROW_TYPE) return defaultSlotMenu(slot);
			const list = rows(this);
			const i = list.indexOf(row);
			new LiteGraph.ContextMenu(
				[
					{ content: `${row.value.on ? "⚫" : "🟢"} Toggle ${row.value.on ? "Off" : "On"}`, callback: () => { row.value.on = !row.value.on; this.setDirtyCanvas(true, false); } },
					{ content: "⬆️ Move Up", disabled: i <= 0, callback: () => moveRow(this, row, -1) },
					{ content: "⬇️ Move Down", disabled: i >= list.length - 1, callback: () => moveRow(this, row, 1) },
					{ content: "🗑️ Remove", callback: () => removeRow(this, row) },
				],
				{ title: row.value.lora || NONE, event: lastPointerEvent, scale: Math.max(1, app.canvas.ds?.scale ?? 1) },
			);
			return undefined;
		};
	},
});

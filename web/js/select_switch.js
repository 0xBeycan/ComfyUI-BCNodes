import { app } from "../../../scripts/app.js";
import { applyType, followedType } from "./wildcard_type.js";

// Select Switch — one input per named option; `selected` (a combo listing the
// option names) picks the input that comes out. The options are the node's
// inputs that are not widget sockets, so they are saved and restored with the
// workflow like any other slot. Under the combo one canvas widget lists them:
//   ● option_a .......................... ✕
// click a name to rename it (the link stays: only the slot's name changes),
// ✕ to remove it; "+ Add option" appends one. Empty, duplicate and reserved
// names are refused. The socket type follows what is connected
// (wildcard_type.js).

const NODE_TYPE = "BC_SelectSwitch";
const SELECTED = "selected";
// `selected` is the widget; `self` would collide with the backend method's
// first argument (nodes/select_switch.py).
const RESERVED = new Set([SELECTED, "self"]);
const ROWS_TYPE = "BC_SELECT_OPTION_ROWS";
const ROW_GAP = 4;

function optionSlots(node) {
	const slots = [];
	for (let i = 0; i < (node.inputs?.length ?? 0); i++) {
		const input = node.inputs[i];
		if (!input.widget && input.name !== SELECTED) slots.push(i);
	}
	return slots;
}

function optionNames(node) {
	return optionSlots(node).map((i) => node.inputs[i].name);
}

function selectedWidget(node) {
	return node.widgets?.find((w) => w.name === SELECTED);
}

function warn(detail) {
	const toast = app.extensionManager?.toast;
	if (toast?.add) toast.add({ severity: "warn", summary: "Select Switch", detail, life: 4000 });
	else window.alert(`Select Switch: ${detail}`);
}

// The trimmed name, or null (after telling the user why) when it cannot be used.
function checkName(node, raw, current) {
	const name = String(raw ?? "").trim();
	if (!name) {
		warn("An option name cannot be empty.");
		return null;
	}
	if (RESERVED.has(name)) {
		warn(`"${name}" is reserved; choose another option name.`);
		return null;
	}
	if (name !== current && (node.inputs ?? []).some((input) => input.name === name)) {
		warn(`An option named "${name}" already exists.`);
		return null;
	}
	return name;
}

function refresh(node) {
	const width = node.size[0];
	const slots = optionSlots(node);
	const names = slots.map((i) => node.inputs[i].name);
	for (const i of slots) {
		const input = node.inputs[i];
		// Renamed through the rows only, so the slot and the combo never disagree.
		input.nameLocked = true;
		input.label = undefined;
		input.localized_name = undefined;
	}

	const combo = selectedWidget(node);
	if (combo) {
		// A copy promoted out of a subgraph takes these values when the workflow
		// loads; the frontend does not pass later changes on to it.
		combo.options.values = names;
		if (!names.includes(combo.value)) combo.value = names[0] ?? "";
	}

	applyType(node, slots, followedType(node, slots));

	const computed = node.computeSize();
	node.setSize([Math.max(width, computed[0]), computed[1]]);
	node.setDirtyCanvas?.(true, true);
}

function addOption(node, event) {
	const taken = new Set(optionNames(node));
	let n = taken.size + 1;
	while (taken.has(`option_${n}`)) n++;
	app.canvas.prompt("Option name", `option_${n}`, (value) => {
		const name = checkName(node, value, null);
		if (name == null) return;
		node.addInput(name, followedType(node, optionSlots(node)));
		refresh(node);
	}, event);
}

function renameOption(node, slot, event) {
	const input = node.inputs[slot];
	const old = input.name;
	app.canvas.prompt("Rename option", old, (value) => {
		const name = checkName(node, value, old);
		if (name == null || name === old) return;
		input.name = name;
		const combo = selectedWidget(node);
		if (combo && combo.value === old) combo.value = name;
		refresh(node);
	}, event);
}

function removeOption(node, slot) {
	node.removeInput(slot);
	refresh(node);
}

class OptionRows {
	constructor(node) {
		this.type = ROWS_TYPE;
		this.name = "option_rows";
		this.node = node;
		this.value = null;
		this.options = { serialize: false };
		this.serialize = false;
	}

	rowCount() {
		return Math.max(1, optionSlots(this.node).length);
	}

	computeSize(width) {
		const h = LiteGraph.NODE_WIDGET_HEIGHT;
		return [width, this.rowCount() * (h + ROW_GAP) - ROW_GAP];
	}

	draw(ctx, node, width, y) {
		if (ctx.canvas === app.canvas?.canvas) width = node.size[0];
		const h = LiteGraph.NODE_WIDGET_HEIGHT;
		const left = 15;
		const right = width - 15;
		const slots = optionSlots(node);
		const selected = selectedWidget(node)?.value;

		ctx.save();
		ctx.font = "12px sans-serif";
		if (!slots.length) {
			ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
			ctx.textAlign = "center";
			ctx.fillText("No options yet: + Add option", width / 2, y + h / 2 + 4);
			ctx.restore();
			return;
		}
		slots.forEach((slot, k) => {
			const top = y + k * (h + ROW_GAP);
			const midY = top + h / 2;
			const name = node.inputs[slot].name;
			ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
			ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
			ctx.beginPath();
			ctx.roundRect(left, top, right - left, h, [h * 0.5]);
			ctx.fill();
			ctx.stroke();

			ctx.beginPath();
			ctx.arc(left + 14, midY, 5, 0, Math.PI * 2);
			ctx.fillStyle = name === selected ? "#7fbf6a" : "#555";
			ctx.fill();

			ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
			ctx.textAlign = "left";
			const nameW = right - 28 - (left + 28);
			let label = name;
			while (label.length > 4 && ctx.measureText(label).width > nameW) label = label.slice(0, -2);
			ctx.fillText(label === name ? name : `${label}…`, left + 28, midY + 4);

			ctx.fillStyle = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
			ctx.textAlign = "center";
			ctx.fillText("✕", right - 14, midY + 4);
		});
		ctx.restore();
	}

	mouse(event, pos, node) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		if (event.button === 2) return false;
		const h = LiteGraph.NODE_WIDGET_HEIGHT;
		const top = this.last_y ?? this.y ?? 0;
		const k = Math.floor((pos[1] - top) / (h + ROW_GAP));
		const slots = optionSlots(node);
		if (k < 0 || k >= slots.length) return false;
		const right = node.size[0] - 15;
		if (pos[0] >= right - 28) removeOption(node, slots[k]);
		else renameOption(node, slots[k], event);
		return true;
	}
}

app.registerExtension({
	name: "BCNodes.SelectSwitch",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				this.addCustomWidget(new OptionRows(this));
				const add = this.addWidget("button", "add_option", null, (...cb) => addOption(this, cb.find((a) => a instanceof Event)));
				add.label = "+ Add option";
				add.serialize = false;
				add.options = add.options ?? {};
				add.options.serialize = false;
				refresh(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			try {
				refresh(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		const onConnectionsChange = nodeType.prototype.onConnectionsChange;
		nodeType.prototype.onConnectionsChange = function (...args) {
			const r = onConnectionsChange?.apply(this, args);
			if (!app.configuringGraph) setTimeout(() => refresh(this), 0);
			return r;
		};
	},
});

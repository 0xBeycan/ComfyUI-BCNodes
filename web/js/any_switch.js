import { app } from "../../../scripts/app.js";

// Any Switch — wildcard inputs that grow as they are connected. Keeps one
// empty slot after the last connected one, drops empty slots in the middle,
// numbers them any_01, any_02, ... and lets the socket type follow whatever
// is connected (first input link, else the output's first target) so the
// canvas rejects mismatched links and shows the real type on the output.

const NODE_TYPE = "BC_AnySwitch";
const SLOT_RE = /^any_\d+$/;
const MIN_SLOTS = 2;

const slotName = (i) => `any_${String(i).padStart(2, "0")}`;

function listSlots(node) {
	const slots = [];
	for (let i = 0; i < node.inputs.length; i++) {
		if (SLOT_RE.test(String(node.inputs[i]?.name ?? ""))) slots.push(i);
	}
	return slots;
}

function connectedType(node) {
	const graph = node.graph;
	for (const i of listSlots(node)) {
		const link = graph?.links?.get?.(node.inputs[i].link) ?? graph?.links?.[node.inputs[i].link];
		if (link?.type && link.type !== "*") return link.type;
	}
	for (const id of node.outputs?.[0]?.links ?? []) {
		const link = graph?.links?.get?.(id) ?? graph?.links?.[id];
		const target = link ? graph?.getNodeById?.(link.target_id) : null;
		const type = target?.inputs?.[link?.target_slot]?.type;
		if (type && type !== "*") return type;
	}
	return "*";
}

function stabilize(node) {
	let changed = false;
	const width = node.size[0];

	let slots = listSlots(node);
	for (let k = slots.length - 2; k >= 0 && slots.length > MIN_SLOTS; k--) {
		if (node.inputs[slots[k]]?.link == null) {
			node.removeInput(slots[k]);
			slots = listSlots(node);
			changed = true;
		}
	}
	while (listSlots(node).length < MIN_SLOTS) {
		node.addInput(slotName(listSlots(node).length + 1), "*");
		changed = true;
	}
	slots = listSlots(node);
	if (node.inputs[slots[slots.length - 1]].link != null) {
		node.addInput(slotName(slots.length + 1), "*");
		changed = true;
	}

	const type = connectedType(node);
	listSlots(node).forEach((slot, k) => {
		const input = node.inputs[slot];
		const name = slotName(k + 1);
		if (input.name !== name) {
			input.name = name;
			input.label = undefined;
			input.localized_name = undefined;
			changed = true;
		}
		if (input.type !== type) {
			input.type = type;
			changed = true;
		}
	});
	const output = node.outputs?.[0];
	if (output && output.type !== type) {
		output.type = type;
		output.label = type;
		changed = true;
	}

	if (changed) {
		const computed = node.computeSize();
		node.setSize([Math.max(width, computed[0]), Math.max(node.size[1], computed[1])]);
		node.setDirtyCanvas?.(true, true);
	}
}

app.registerExtension({
	name: "BCNodes.AnySwitch",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				stabilize(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			try {
				stabilize(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		const onConnectionsChange = nodeType.prototype.onConnectionsChange;
		nodeType.prototype.onConnectionsChange = function (...args) {
			const r = onConnectionsChange?.apply(this, args);
			if (!app.configuringGraph) setTimeout(() => stabilize(this), 0);
			return r;
		};
	},
});

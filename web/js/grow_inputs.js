import { app } from "../../../scripts/app.js";

// Growing inputs: slots <prefix>1, <prefix>2, ... of one type, with exactly one empty slot after the last
// connected one (the next appears once the last is linked) and contiguous names when a slot in the middle is
// disconnected, never fewer than `min`. ComfyUI passes every linked input to the node whether declared or not
// (its optional mapping answers for any name: nodes/common.FlexibleOptionalInputType), so the extra slots need
// nothing on the Python side. Used by Join Image Lists (In1, In2, ...) and Claude (image_1, image_2, ...).

function listSlots(node, slotRe) {
	const slots = [];
	for (let i = 0; i < node.inputs.length; i++) {
		if (slotRe.test(String(node.inputs[i]?.name ?? ""))) slots.push(i);
	}
	return slots;
}

function stabilize(node, prefix, type, min, slotRe) {
	let changed = false;
	const width = node.size[0];

	// Drop every empty slot that is not the last one, but never go below the
	// `min` the backend declares.
	let slots = listSlots(node, slotRe);
	for (let k = slots.length - 2; k >= 0 && slots.length > min; k--) {
		if (node.inputs[slots[k]]?.link == null) {
			node.removeInput(slots[k]);
			slots = listSlots(node, slotRe);
			changed = true;
		}
	}

	while (listSlots(node, slotRe).length < min) {
		node.addInput(`${prefix}${listSlots(node, slotRe).length + 1}`, type);
		changed = true;
	}

	// One free slot at the end, always.
	slots = listSlots(node, slotRe);
	const last = node.inputs[slots[slots.length - 1]];
	if (last.link != null) {
		node.addInput(`${prefix}${slots.length + 1}`, type);
		changed = true;
	}

	listSlots(node, slotRe).forEach((slot, k) => {
		const name = `${prefix}${k + 1}`;
		if (node.inputs[slot].name !== name) {
			node.inputs[slot].name = name;
			node.inputs[slot].label = undefined;
			node.inputs[slot].localized_name = undefined;
			changed = true;
		}
	});

	if (changed) {
		const computed = node.computeSize();
		node.setSize([Math.max(width, computed[0]), Math.max(node.size[1], computed[1])]);
		node.setDirtyCanvas?.(true, true);
	}
	return changed;
}

// Called from an extension's beforeRegisterNodeDef with the node's type.
export function growInputs(nodeType, { prefix, type, min }) {
	const slotRe = new RegExp(`^${prefix}\\d+$`);
	const settle = (node) => stabilize(node, prefix, type, min, slotRe);

	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function (...args) {
		const r = onNodeCreated?.apply(this, args);
		try {
			settle(this);
		} catch (e) {
			console.error("[BCNodes]", e);
		}
		return r;
	};

	// Saved workflows carry their own slot list; settle it after configure
	// rather than on every connection event fired while loading.
	const onConfigure = nodeType.prototype.onConfigure;
	nodeType.prototype.onConfigure = function (...args) {
		const r = onConfigure?.apply(this, args);
		try {
			settle(this);
		} catch (e) {
			console.error("[BCNodes]", e);
		}
		return r;
	};

	const onConnectionsChange = nodeType.prototype.onConnectionsChange;
	nodeType.prototype.onConnectionsChange = function (...args) {
		const r = onConnectionsChange?.apply(this, args);
		if (!app.configuringGraph) settle(this);
		return r;
	};
}

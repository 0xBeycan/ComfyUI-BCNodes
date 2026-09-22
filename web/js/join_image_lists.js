import { app } from "../../../scripts/app.js";

// Join Image Lists — unbounded inputs. The backend declares In1 and In2; this
// extension keeps exactly one empty IMAGE slot after the last connected one
// (In3 appears once In1 and In2 are both linked, In4 once In3 is linked, ...)
// and keeps the names contiguous when a slot in the middle is disconnected.
// ComfyUI passes every linked input to the node whether declared or not, so
// the extra slots need nothing on the Python side.

const NODE_TYPE = "BC_JoinImageLists";
const SLOT_PREFIX = "In";
const SLOT_TYPE = "IMAGE";
const MIN_SLOTS = 2;
const SLOT_RE = /^In\d+$/;

function listSlots(node) {
	const slots = [];
	for (let i = 0; i < node.inputs.length; i++) {
		if (SLOT_RE.test(String(node.inputs[i]?.name ?? ""))) slots.push(i);
	}
	return slots;
}

function stabilize(node) {
	let changed = false;
	const width = node.size[0];

	// Drop every empty slot that is not the last one, but never go below the
	// two the backend requires.
	let slots = listSlots(node);
	for (let k = slots.length - 2; k >= 0 && slots.length > MIN_SLOTS; k--) {
		if (node.inputs[slots[k]]?.link == null) {
			node.removeInput(slots[k]);
			slots = listSlots(node);
			changed = true;
		}
	}

	while (listSlots(node).length < MIN_SLOTS) {
		node.addInput(`${SLOT_PREFIX}${listSlots(node).length + 1}`, SLOT_TYPE);
		changed = true;
	}

	// One free slot at the end, always.
	slots = listSlots(node);
	const last = node.inputs[slots[slots.length - 1]];
	if (last.link != null) {
		node.addInput(`${SLOT_PREFIX}${slots.length + 1}`, SLOT_TYPE);
		changed = true;
	}

	listSlots(node).forEach((slot, k) => {
		const name = `${SLOT_PREFIX}${k + 1}`;
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

app.registerExtension({
	name: "BCNodes.JoinImageLists",

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

		// Saved workflows carry their own slot list; settle it after configure
		// rather than on every connection event fired while loading.
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
			if (!app.configuringGraph) stabilize(this);
			return r;
		};
	},
});

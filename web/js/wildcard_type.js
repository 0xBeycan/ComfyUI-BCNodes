// Wildcard sockets whose type follows what is connected, shared by Any Switch
// and Select Switch. The type is taken from the first given input that has a
// typed link, else from the first typed target of output 0, else "*". Setting
// it on the inputs and the output lets the canvas reject mismatched links and
// show the real type on the output. No extension is registered here: ComfyUI
// loads every .js under web/ as an extension, and this module only exports.

function linkById(graph, id) {
	return graph?.links?.get?.(id) ?? graph?.links?.[id];
}

export function followedType(node, slots) {
	const graph = node.graph;
	for (const i of slots) {
		const link = linkById(graph, node.inputs[i].link);
		if (link?.type && link.type !== "*") return link.type;
	}
	for (const id of node.outputs?.[0]?.links ?? []) {
		const link = linkById(graph, id);
		const target = link ? graph?.getNodeById?.(link.target_id) : null;
		const type = target?.inputs?.[link?.target_slot]?.type;
		if (type && type !== "*") return type;
	}
	return "*";
}

// Sets `type` on the given inputs and on output 0 (type and label). Returns
// whether anything changed.
export function applyType(node, slots, type) {
	let changed = false;
	for (const i of slots) {
		const input = node.inputs[i];
		if (input.type !== type) {
			input.type = type;
			changed = true;
		}
	}
	const output = node.outputs?.[0];
	if (output && output.type !== type) {
		output.type = type;
		output.label = type;
		changed = true;
	}
	return changed;
}

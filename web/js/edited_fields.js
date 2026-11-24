// Config nodes whose backend applies only the fields the user changed and the fields fed by a link (nodes/common.py
// edited_fields): LM Config (lm.js) and DyPE Config (dype.js). The changed fields are named in the node's hidden
// `edited` widget. A user change to a field (its widget callback with a changed value; setting a value from code
// does not call it) adds it to `edited`. The frontend also calls a widget's callback with its value unchanged
// (Rename Widget, a subgraph promote): only a callback with a value other than the one last seen is a change.
// On the node canvas un-edited fields are drawn with the disabled look and still take input. The Vue node renderer
// has no such look: its only greyed state is `disabled`, which also blocks input, so there un-edited fields look like
// edited ones. Not an extension: the modules that use it register their own.

export const EDITED = "edited";
// The node canvas's disabled look: LGraphNode.drawWidgets halves the alpha of a disabled widget.
const DEFAULT_ALPHA = 0.5;

export function widget(node, name) {
	return node.widgets?.find((w) => w.name === name);
}

// Calls fn after target[key] (a prototype method or a widget callback), with the same this and
// arguments; the original's result is returned and an error in fn is logged.
export function after(target, key, fn) {
	if (!target) return;
	const original = target[key];
	target[key] = function (...args) {
		const r = original?.apply(this, args);
		try {
			fn.apply(this, args);
		} catch (e) {
			console.error("[BCNodes]", e);
		}
		return r;
	};
}

// The value each watched widget was last seen with: at creation, on load, when set here, and at each
// change.
const seen = new WeakMap();

export function see(w) {
	if (w) seen.set(w, w.value);
}

// Calls fn after w's callback when the value it is called with is not the one last seen.
export function afterChange(w, fn) {
	if (!w) return;
	see(w);
	after(w, "callback", (value) => {
		if (value === seen.get(w)) return;
		seen.set(w, value);
		fn();
	});
}

export function editedNames(node) {
	const raw = String(widget(node, EDITED)?.value ?? "");
	return [...new Set(raw.split(",").map((s) => s.trim()).filter(Boolean))];
}

// `fields` (the node's fields in widget order) first, then any other name as it was (the backend's error
// names it; a reset clears it), so one set of fields always gives the same string: `edited` is part of the
// cache key.
export function setEdited(node, names, fields) {
	const w = widget(node, EDITED);
	if (!w) return;
	w.value = [...fields.filter((f) => names.includes(f)), ...names.filter((n) => !fields.includes(n))].join(", ");
	node.setDirtyCanvas?.(true, false);
}

function markEdited(node, name, fields) {
	const names = editedNames(node);
	if (!names.includes(name)) setEdited(node, [...names, name], fields);
}

// Node canvas: an un-edited field is drawn at the disabled look's alpha but keeps taking input (a
// disabled widget takes none and hides its value). The Vue renderer never calls drawWidget for
// these widget types; it draws its own components.
function greyUnlessEdited(node, w) {
	const draw = w.drawWidget;
	if (typeof draw !== "function") return;
	w.drawWidget = function (ctx, options) {
		if (editedNames(node).includes(this.name)) return draw.call(this, ctx, options);
		ctx.save();
		try {
			ctx.globalAlpha *= DEFAULT_ALPHA;
			draw.call(this, ctx, options);
		} finally {
			ctx.restore();
		}
	};
}

// On a config node's creation: `edited` hidden, each field marked edited on a user change and drawn greyed
// until then.
export function installEditedFields(node, fields) {
	// Hidden in both renderers: the node canvas skips a hidden widget in layout and drawing, the
	// Vue renderer filters on options.hidden, which this sets. It is still serialized.
	const edited = widget(node, EDITED);
	if (edited) edited.hidden = true;
	for (const name of fields) {
		const w = widget(node, name);
		if (!w) continue;
		afterChange(w, () => markEdited(node, name, fields));
		greyUnlessEdited(node, w);
	}
	node.setSize([node.size[0], node.computeSize()[1]]);
}

import { app } from "../../../scripts/app.js";

// Fast Groups Bypasser — one row per group, drawn as a canvas widget:
//   Enable <group title> ............ yes ●
// Click the row to toggle: on = the group's nodes are active, off =
// bypassed. Groups of the graph the node sits in and of every subgraph are
// listed; the list follows the graph (groups added, renamed, removed) on a
// half-second tick, and the toggles follow the nodes' real modes, however
// they were changed.
//
// Properties: sort ("position" | "alphanumeric" | "custom alphabet"),
// customSortAlphabet (letters or comma-separated prefixes, used by "custom
// alphabet"), matchColors (comma-separated group colours or colour names;
// only matching groups are listed), matchTitle (regex; same),
// showAllGraphs (off = only the groups of the graph on screen),
// toggleRestriction ("default" | "max one" | "always one").
// Context menu: Bypass all / Enable all / Toggle all.

const NODE_TYPE = "BC_FastGroupsBypasser";
const ROW_TYPE = "BC_GROUP_ROW";
const MODE_ALWAYS = 0;
const MODE_BYPASS = 4;
const TICK_MS = 500;

const DEFAULTS = {
	sort: "position",
	customSortAlphabet: "",
	matchColors: "",
	matchTitle: "",
	showAllGraphs: true,
	toggleRestriction: "default",
};

const props = (node) => ({ ...DEFAULTS, ...(node.properties ?? {}) });

// ---------------------------------------------------------------------------
// Group membership and modes
// ---------------------------------------------------------------------------

// Membership is computed here from pos / size (title bar included) rather
// than through litegraph's recomputeInsideNodes, which depends on geometry
// that is only measured while the classic canvas draws the node.
function nodeRect(node) {
	const r = node.boundingRect;
	if (r && r[2] > 0 && r[3] > 0) return r;
	const title = node.flags?.collapsed ? 0 : LiteGraph.NODE_TITLE_HEIGHT;
	return [node.pos[0], node.pos[1] - title, node.size[0], node.size[1] + title];
}

function groupNodes(group) {
	const graph = group.graph;
	const [gx, gy, gw, gh] = group._bounding ?? [group.pos[0], group.pos[1], group.size[0], group.size[1]];
	const out = [];
	for (const node of graph?._nodes ?? graph?.nodes ?? []) {
		if (typeof node.mode !== "number") continue;
		const [x, y, w, h] = nodeRect(node);
		const cx = x + w / 2;
		const cy = y + h / 2;
		if (cx >= gx && cx < gx + gw && cy >= gy && cy < gy + gh) out.push(node);
	}
	return out;
}

function setModeDeep(node, mode) {
	const stack = [node];
	const seen = new Set();
	while (stack.length) {
		const n = stack.pop();
		if (!n || seen.has(n)) continue;
		seen.add(n);
		if (n.mode !== mode) n.mode = mode;
		if (n.isSubgraphNode?.() && n.subgraph) stack.push(...(n.subgraph.nodes ?? n.subgraph._nodes ?? []));
	}
}

function isActive(group) {
	const nodes = groupNodes(group);
	return nodes.length > 0 && nodes.some((n) => n.mode === MODE_ALWAYS);
}

function setGroup(group, on) {
	for (const n of groupNodes(group)) setModeDeep(n, on ? MODE_ALWAYS : MODE_BYPASS);
	group.graph?.setDirtyCanvas(true, false);
}

// ---------------------------------------------------------------------------
// Listing
// ---------------------------------------------------------------------------

const groupLabel = (g) => g.title || "(untitled group)";

function currentGraph() {
	return app.canvas?.getCurrentGraph?.() ?? app.canvas?.graph ?? app.graph;
}

// This graph's groups plus the ones inside every subgraph definition, so a
// node at the root reaches groups that live in subgraphs too.
function allGroups(node) {
	const graph = node.graph ?? app.graph;
	const groups = [...(graph?._groups ?? graph?.groups ?? [])];
	for (const sub of graph?.subgraphs?.values?.() ?? []) {
		if (sub !== graph) groups.push(...(sub._groups ?? sub.groups ?? []));
	}
	return groups;
}

// "red" / "#f00" / "#ff0000" → "#ff0000"; colour names map to the palette's
// group colour, which is what a group picked from the menu carries.
function normalizeColor(value) {
	let color = String(value ?? "").trim().toLowerCase();
	if (!color) return "";
	const named = LGraphCanvas.node_colors?.[color];
	if (named) color = named.groupcolor;
	color = color.replace("#", "").toLowerCase();
	if (color.length === 3) color = color.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
	return `#${color}`;
}

function customAlphabet(p) {
	const raw = (p.customSortAlphabet ?? "").replace(/\n/g, "");
	if (!raw.trim()) return null;
	const alphabet = raw.includes(",") ? raw.toLowerCase().split(",") : raw.toLowerCase().trim().split("");
	return alphabet.length ? alphabet : null;
}

function sortGroups(groups, p) {
	const alphabet = p.sort === "custom alphabet" ? customAlphabet(p) : null;
	if (alphabet) {
		const rank = (g) => alphabet.findIndex((a) => (g.title ?? "").toLowerCase().startsWith(a));
		return groups.sort((a, b) => {
			const ai = rank(a);
			const bi = rank(b);
			if (ai > -1 && bi > -1) return ai - bi || (a.title ?? "").localeCompare(b.title ?? "");
			if (ai > -1) return -1;
			if (bi > -1) return 1;
			return (a.title ?? "").localeCompare(b.title ?? "");
		});
	}
	if (p.sort === "alphanumeric" || p.sort === "custom alphabet") {
		return groups.sort((a, b) => (a.title ?? "").localeCompare(b.title ?? ""));
	}
	// By row, then column, snapped to 30px so slightly uneven groups keep a stable order.
	const cell = (v) => Math.floor(v / 30);
	return groups.sort((a, b) => cell(a.pos[1]) - cell(b.pos[1]) || cell(a.pos[0]) - cell(b.pos[0]));
}

function listGroups(node) {
	const p = props(node);
	let groups = allGroups(node);
	const colors = String(p.matchColors ?? "").split(",").map(normalizeColor).filter(Boolean);
	if (colors.length) groups = groups.filter((g) => colors.includes(normalizeColor(g.color)));
	const pattern = String(p.matchTitle ?? "").trim();
	if (pattern) {
		try {
			const re = new RegExp(pattern, "i");
			groups = groups.filter((g) => re.test(g.title ?? ""));
		} catch {
			// a half-typed regex: show everything until it parses
		}
	}
	if (!p.showAllGraphs) {
		const cur = currentGraph();
		groups = groups.filter((g) => g.graph === cur);
	}
	return sortGroups(groups, p);
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

const rows = (node) => (node.widgets ?? []).filter((w) => w.type === ROW_TYPE);

// Turn one row on or off, honouring toggleRestriction the way a click does:
// "max one" / "always one" switch the other rows off first, "always one"
// refuses to switch the last active row off.
function setRow(row, on, skipRestriction) {
	const restriction = props(row.node).toggleRestriction;
	let value = on;
	if (!skipRestriction) {
		if (value && restriction.includes(" one")) {
			for (const other of rows(row.node)) if (other !== row) setRow(other, false, true);
		} else if (!value && restriction === "always one") {
			value = rows(row.node).every((other) => other === row || !other.value);
		}
	}
	setGroup(row.group, value);
	row.value = value;
	row.node.setDirtyCanvas(true, false);
}

class GroupRow {
	constructor(node, group) {
		this.type = ROW_TYPE;
		this.name = groupLabel(group);
		this.node = node;
		this.group = group;
		this.value = isActive(group);
		this.options = { on: "yes", off: "no", serialize: false };
		this.serialize = false;
	}

	computeSize(width) {
		return [width, LiteGraph.NODE_WIDGET_HEIGHT];
	}

	draw(ctx, node, width, y, height, lowQuality = (app.canvas?.ds?.scale ?? 1) <= 0.5) {
		// On the node canvas the row spans the node. The Vue-nodes legacy renderer
		// leaves its own width on the widget (`widget.width || nodeWidth`), which
		// would otherwise stick after switching back to the classic canvas.
		if (ctx.canvas === app.canvas?.canvas) width = node.size[0];
		const margin = 15;
		const right = width - margin;
		const midY = y + height / 2;
		const on = !!this.value;

		ctx.save();
		ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
		ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
		ctx.beginPath();
		ctx.roundRect(margin, y, right - margin, height, [height * 0.5]);
		ctx.fill();
		ctx.stroke();

		// Right to left: the label on the left takes what is left over.
		let x = right - margin - 7;
		const radius = height * 0.36;
		ctx.fillStyle = on ? "#89A" : "#333";
		ctx.beginPath();
		ctx.arc(x - radius, midY, radius, 0, Math.PI * 2);
		ctx.fill();
		x -= radius * 2;

		if (!lowQuality) {
			ctx.font = "12px sans-serif";
			x -= 4;
			ctx.textAlign = "right";
			ctx.fillStyle = on ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
			ctx.fillText(on ? this.options.on : this.options.off, x, midY + 4);
			x -= Math.max(ctx.measureText(this.options.on).width, ctx.measureText(this.options.off).width) + 7;

			const labelX = margin + 10;
			const labelW = x - labelX;
			ctx.textAlign = "left";
			const full = `Enable ${groupLabel(this.group)}`;
			let label = full;
			while (label.length > 4 && ctx.measureText(label).width > labelW) label = label.slice(0, -2);
			if (label !== full) label += "…";
			ctx.fillText(label, labelX, midY + 4);
		}
		ctx.restore();
	}

	mouse(event, pos, node) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		if (event.button === 2) return false;
		setRow(this, !this.value);
		return true;
	}
}

// ---------------------------------------------------------------------------
// Refresh
// ---------------------------------------------------------------------------

function refresh(node) {
	try {
		refreshUnsafe(node);
	} catch (e) {
		console.error("[BCNodes] Fast Groups Bypasser:", e);
	}
}

function refreshUnsafe(node) {
	if (app.configuringGraph) return;
	const groups = listGroups(node);
	const widgets = node.widgets ?? [];
	const current = rows(node);
	const inSync = groups.length === current.length && groups.every((g, i) => current[i].group === g) &&
		(groups.length > 0 || widgets.some((w) => w.bcInfo));

	if (!inSync) {
		const width = node.size[0];
		if (node.widgets) node.widgets.splice(0, node.widgets.length);
		for (const group of groups) node.addCustomWidget(new GroupRow(node, group));
		if (!groups.length) {
			// The message is the label: a value would be right-aligned and
			// `disabled` would hide it in current frontends.
			const w = node.addWidget("text", "no groups in this graph", "", () => {}, {});
			w.onClick = () => {};
			w.mouse = () => true;
			w.bcInfo = true;
			w.serialize = false;
			w.options.serialize = false;
		}
		const computed = node.computeSize();
		node.setSize([Math.max(width, computed[0]), computed[1]]);
		node.setDirtyCanvas(true, true);
		return;
	}
	let dirty = false;
	for (const w of current) {
		const active = isActive(w.group);
		if (w.value !== active) {
			w.value = active;
			dirty = true;
		}
		const label = groupLabel(w.group);
		if (w.name !== label) {
			w.name = label;
			dirty = true;
		}
	}
	if (dirty) node.setDirtyCanvas(true, false);
}

// Bypass all / Enable all / Toggle all, honouring toggleRestriction.
function applyAll(node, action) {
	const list = rows(node);
	const restriction = props(node).toggleRestriction;
	const onlyOne = restriction.includes(" one");
	if (action === "bypass") {
		list.forEach((row, i) => setRow(row, restriction === "always one" && i === 0, true));
	} else if (action === "enable") {
		list.forEach((row, i) => setRow(row, !(onlyOne && i > 0), true));
	} else {
		let foundOne = false;
		for (const row of list) {
			const value = onlyOne && foundOne ? false : !row.value;
			foundOne = foundOne || value;
			setRow(row, value, true);
		}
		if (!foundOne && restriction === "always one" && list.length) setRow(list[list.length - 1], true, true);
	}
}

app.registerExtension({
	name: "BCNodes.FastGroupsBypasser",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;
		nodeType["@sort"] = { type: "combo", values: ["position", "alphanumeric", "custom alphabet"] };
		nodeType["@customSortAlphabet"] = { type: "string" };
		nodeType["@matchColors"] = { type: "string" };
		nodeType["@matchTitle"] = { type: "string" };
		nodeType["@showAllGraphs"] = { type: "boolean" };
		nodeType["@toggleRestriction"] = { type: "combo", values: ["default", "max one", "always one"] };

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			this.properties = { ...DEFAULTS, ...(this.properties ?? {}) };
			this.serialize_widgets = false;
			// No sockets: computeSize would still reserve a slot row that is
			// never drawn and leave a blank strip under the toggles.
			this.widgets_start_y = 6;
			return r;
		};

		const onAdded = nodeType.prototype.onAdded;
		nodeType.prototype.onAdded = function (...args) {
			const r = onAdded?.apply(this, args);
			clearInterval(this.bcTick);
			this.bcTick = setInterval(() => refresh(this), TICK_MS);
			setTimeout(() => refresh(this), 0);
			return r;
		};

		const onRemoved = nodeType.prototype.onRemoved;
		nodeType.prototype.onRemoved = function (...args) {
			clearInterval(this.bcTick);
			this.bcTick = null;
			return onRemoved?.apply(this, args);
		};

		const onDrawForeground = nodeType.prototype.onDrawForeground;
		nodeType.prototype.onDrawForeground = function (...args) {
			const r = onDrawForeground?.apply(this, args);
			const now = performance.now();
			if (!this.bcLastRefresh || now - this.bcLastRefresh > 250) {
				this.bcLastRefresh = now;
				refresh(this);
			}
			return r;
		};

		const onPropertyChanged = nodeType.prototype.onPropertyChanged;
		nodeType.prototype.onPropertyChanged = function (...args) {
			const r = onPropertyChanged?.apply(this, args);
			setTimeout(() => refresh(this), 0);
			return r;
		};

		const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
		nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
			const r = getExtraMenuOptions?.apply(this, [canvas, options]) ?? [];
			options.push(
				null,
				{ content: "Bypass all", callback: () => applyAll(this, "bypass") },
				{ content: "Enable all", callback: () => applyAll(this, "enable") },
				{ content: "Toggle all", callback: () => applyAll(this, "toggle") },
			);
			return r;
		};
	},
});

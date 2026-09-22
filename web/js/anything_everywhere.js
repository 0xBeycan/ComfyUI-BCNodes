import { app } from "../../../scripts/app.js";

// Anything Everywhere — whatever is wired into the node is handed to every
// unconnected input of the same type when the prompt is built. One node takes
// any number of sources: each connected slot takes the type (and colour) of
// its link and a fresh empty `anything` slot is kept at the bottom. Nothing
// is linked for real: the API prompt that graphToPrompt returns is patched, so
// queueing and "Export (API)" both see the links.
//
// On the canvas the node shows what it reaches: inputs it feeds get a glowing
// ring, every other free input a small dot (it could be fed), and the
// translucent "phantom" links are drawn when the node or the target is
// selected or under the pointer (setting: BCNodes › Anything Everywhere).
//
// The node properties title_regex / input_regex (Properties Panel) restrict
// which nodes and inputs receive it; a node with a regex wins over one
// without. Scope: the Anything Everywhere node and its sources must be in the
// root graph; targets may be anywhere (root or inside subgraphs). A bypassed
// or muted Anything Everywhere node does nothing.

const NODE_TYPE = "BC_AnythingEverywhere";
const MODE_ALWAYS = 0;
const EMPTY_LABEL = "anything";
const FIRST_EXTRA_SLOT = 11; // extra slots are anything11, anything12, …
const SUBGRAPH_INPUT_ID = -10;
const CACHE_MS = 100;

const SETTING_SHOW_LINKS = "BCNodes.AnythingEverywhere.ShowLinks";
const SETTING_HIGHLIGHT = "BCNodes.AnythingEverywhere.Highlight";
const SHOW_OFF = 0;
const SHOW_SELECTED = 1;
const SHOW_HOVER = 2;
const SHOW_SELECTED_OR_HOVER = 3;
const SHOW_ALL = 4;

function setting(id, fallback) {
	try {
		return app.ui?.settings?.getSettingValue?.(id) ?? fallback;
	} catch {
		return fallback;
	}
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

function rootGraph() {
	return app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
}

function allGraphs() {
	const root = rootGraph();
	if (!root) return [];
	const graphs = [root];
	if (root.subgraphs?.values) for (const sg of root.subgraphs.values()) graphs.push(sg);
	return graphs;
}

function linkOf(graph, id) {
	if (id == null) return null;
	return graph.links?.get?.(id) ?? graph.links?.[id] ?? null;
}

function propertyValue(node, name) {
	return String(node.properties?.[name] ?? "").trim();
}

function compile(pattern) {
	if (!pattern) return null;
	try {
		return new RegExp(pattern, "i");
	} catch {
		return null;
	}
}

// ---------------------------------------------------------------------------
// The node's slots: one per source plus one empty one
// ---------------------------------------------------------------------------

const isUE = (node) => node?.type === NODE_TYPE;
const isEmptySlot = (input) => input.link == null && (!input.type || input.type === "*");

function linkType(graph, link) {
	if (!link) return "*";
	let type;
	if (link.origin_id === SUBGRAPH_INPUT_ID || link.originIsIoNode) {
		type = graph.inputs?.[link.origin_slot]?.type;
	} else {
		type = graph.getNodeById?.(link.origin_id)?.outputs?.[link.origin_slot]?.type;
	}
	if (!type || type === "*") type = link.type;
	return type && type !== "*" ? type : "*";
}

function styleSlot(input, type) {
	if (type === "*") {
		input.type = "*";
		input.label = EMPTY_LABEL;
		input.color_on = undefined;
	} else {
		input.type = type;
		input.label = type;
		input.color_on = app.canvas?.default_connection_color_byType?.[type];
	}
}

function nextSlotName(node) {
	let n = FIRST_EXTRA_SLOT - 1;
	for (const input of node.inputs) {
		const m = /^anything(\d+)$/.exec(input.name ?? "");
		if (m) n = Math.max(n, Number(m[1]));
	}
	return `anything${n + 1}`;
}

// Connected slots carry their link's type, free slots are `anything`, and
// there is exactly one free slot, at the bottom.
function fixInputs(node) {
	if (!node.graph || app.configuringGraph) return;
	const graph = node.graph;
	for (const input of node.inputs) {
		if (input.link == null) {
			styleSlot(input, "*");
		} else {
			const type = input.type && input.type !== "*" ? input.type : linkType(graph, linkOf(graph, input.link));
			styleSlot(input, type);
		}
	}
	for (;;) {
		const empties = node.inputs.map((input, i) => (isEmptySlot(input) ? i : -1)).filter((i) => i >= 0);
		if (empties.length <= 1) break;
		node.removeInput(empties[0]);
	}
	if (!node.inputs.some(isEmptySlot)) node.addInput(nextSlotName(node), "*", { label: EMPTY_LABEL });
	node.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// Matching: which input gets which source
// ---------------------------------------------------------------------------

function typeMatches(inputType, sourceType) {
	if (!inputType || inputType === "*") return true;
	return String(inputType).split(",").map((t) => t.trim()).includes(sourceType);
}

// Sources: [{ueNode, ueSlot, type, promptId, slot, titleRe, inputRe, specific}]
function collectSources(root) {
	const out = [];
	for (const node of root.nodes ?? root._nodes ?? []) {
		if (!isUE(node) || node.mode !== MODE_ALWAYS) continue;
		const titleRe = compile(propertyValue(node, "title_regex"));
		const inputRe = compile(propertyValue(node, "input_regex"));
		node.inputs?.forEach((input, ueSlot) => {
			if (input.link == null) return;
			const link = linkOf(root, input.link);
			if (!link) return;
			const origin = root.getNodeById(link.origin_id);
			if (!origin) return;
			if (origin.isSubgraphNode?.() || origin.isVirtualNode) {
				console.warn(`[BCNodes] Anything Everywhere #${node.id}: sources from subgraph or virtual nodes are not supported`);
				return;
			}
			const type = link.type && link.type !== "*" ? link.type : origin.outputs?.[link.origin_slot]?.type;
			if (!type || type === "*") return;
			out.push({ ueNode: node, ueSlot, type, promptId: String(origin.id), slot: link.origin_slot, titleRe, inputRe, specific: !!(titleRe || inputRe) });
		});
	}
	out.sort((a, b) => Number(b.specific) - Number(a.specific));
	return out;
}

const isFreeInput = (input) => input.link == null && !input.widget;

// Connections: [{source, node, slot, graph}] — every free input that a source
// reaches. Recomputed at most every CACHE_MS and whenever a link changes.
const cache = { at: 0, list: [] };
const invalidate = () => (cache.at = 0);

function connections() {
	const now = performance.now();
	if (now - cache.at < CACHE_MS) return cache.list;
	const list = [];
	const root = rootGraph();
	if (root) {
		const srcs = collectSources(root);
		if (srcs.length) {
			for (const graph of allGraphs()) {
				for (const node of graph.nodes ?? graph._nodes ?? []) {
					if (isUE(node) || node.isVirtualNode) continue;
					node.inputs?.forEach((input, slot) => {
						if (!isFreeInput(input)) return;
						const source = srcs.find((s) =>
							typeMatches(input.type, s.type) &&
							(!s.titleRe || s.titleRe.test(node.title ?? "")) &&
							(!s.inputRe || s.inputRe.test(input.name ?? "")));
						if (source) list.push({ source, node, slot, graph });
					});
				}
			}
		}
	}
	cache.at = now;
	cache.list = list;
	return list;
}

// ---------------------------------------------------------------------------
// Prompt rewrite
// ---------------------------------------------------------------------------

function promptEntries(output, node, isRoot) {
	if (isRoot) {
		const e = output[String(node.id)];
		return e ? [e] : [];
	}
	const suffix = `:${node.id}`;
	return Object.keys(output).filter((k) => k.endsWith(suffix) && output[k].class_type === node.type).map((k) => output[k]);
}

function apply(result) {
	const output = result?.output;
	const root = rootGraph();
	if (!output || !root) return;
	invalidate();
	let count = 0;
	for (const { source, node, slot, graph } of connections()) {
		const input = node.inputs[slot];
		for (const entry of promptEntries(output, node, graph === root)) {
			if (entry.inputs[input.name] !== undefined) continue;
			entry.inputs[input.name] = [source.promptId, source.slot];
			count++;
		}
	}
	if (count) console.log(`[BCNodes] Anything Everywhere: filled ${count} input(s)`);
}

// ---------------------------------------------------------------------------
// Canvas: rings on fed inputs, dots on free ones, a badge on the node,
// phantom links
// ---------------------------------------------------------------------------

function slotPos(node, slot) {
	return node.getConnectionPos(true, slot);
}

function ring(ctx, x, y, radius, style) {
	ctx.save();
	ctx.shadowOffsetX = 0;
	ctx.shadowOffsetY = 0;
	Object.assign(ctx, style);
	ctx.beginPath();
	ctx.arc(x, y, radius, 0, Math.PI * 2);
	ctx.stroke();
	ctx.restore();
}

// Called inside drawNode, so the context is already at the node's origin.
function highlightNode(node, ctx) {
	if (!setting(SETTING_HIGHLIGHT, true)) return;
	if (isUE(node)) {
		const sending = connections().some((c) => c.source.ueNode === node);
		const restricted = !!(propertyValue(node, "title_regex") || propertyValue(node, "input_regex"));
		const alpha = sending ? 1 : 0.35;
		ctx.save();
		ctx.lineWidth = 2;
		ctx.strokeStyle = restricted ? `rgba(255, 255, 72, ${alpha})` : `rgba(72, 255, 72, ${alpha})`;
		ctx.beginPath();
		ctx.roundRect(5, 5 - LiteGraph.NODE_TITLE_HEIGHT, 20, 20, 6);
		ctx.stroke();
		ctx.restore();
		return;
	}
	if (!node.inputs?.length) return;
	const fed = new Set();
	for (const c of connections()) {
		if (c.node !== node) continue;
		fed.add(c.slot);
		const [x, y] = slotPos(node, c.slot);
		const color = LGraphCanvas.link_type_colors[c.source.type] ?? "white";
		ring(ctx, x - node.pos[0], y - node.pos[1], 6, { lineWidth: 1, strokeStyle: color, shadowColor: "white", shadowBlur: 4 });
		ring(ctx, x - node.pos[0], y - node.pos[1], 5, { lineWidth: 1, strokeStyle: "black", shadowBlur: 0 });
	}
	node.inputs.forEach((input, slot) => {
		if (fed.has(slot) || !isFreeInput(input)) return;
		const [x, y] = slotPos(node, slot);
		ring(ctx, x - node.pos[0], y - node.pos[1], 3, { lineWidth: 1, strokeStyle: "black", shadowColor: "green", shadowBlur: 4 });
	});
}

// "#rgb" / "#rrggbb" at 40 % opacity — the phantom look.
function translucent(color) {
	if (typeof color !== "string") return color;
	if (color.length === 4) return color + "6";
	if (color.length === 7) return color + "66";
	return color;
}

function shouldShow(canvas, mode, ueNode, target) {
	if (mode === SHOW_ALL) return true;
	if ((mode === SHOW_SELECTED || mode === SHOW_SELECTED_OR_HOVER) && (ueNode.selected || target.selected)) return true;
	if ((mode === SHOW_HOVER || mode === SHOW_SELECTED_OR_HOVER) && (canvas.node_over === ueNode || canvas.node_over === target)) return true;
	return false;
}

function drawPhantomLinks(canvas, ctx) {
	const mode = Number(setting(SETTING_SHOW_LINKS, SHOW_SELECTED_OR_HOVER));
	if (mode === SHOW_OFF) return;
	const visible = canvas.graph;
	const hq = canvas.highquality_render;
	const border = canvas.render_connections_border;
	ctx.save();
	canvas.highquality_render = false;
	canvas.render_connections_border = false;
	try {
		for (const { source, node, slot, graph } of connections()) {
			if (graph !== visible || source.ueNode.graph !== visible) continue;
			if (!shouldShow(canvas, mode, source.ueNode, node)) continue;
			const a = slotPos(source.ueNode, source.ueSlot);
			const b = slotPos(node, slot);
			const dx = b[0] - a[0];
			const dy = b[1] - a[1];
			// leaving an input socket: sideways when the target is mostly left or right, else up or down
			const startDir = Math.abs(dy) > Math.abs(dx) ? (dy > 0 ? LiteGraph.DOWN : LiteGraph.UP) : (dx < 0 ? LiteGraph.LEFT : LiteGraph.RIGHT);
			const color = LGraphCanvas.link_type_colors[source.type] ?? canvas.default_link_color;
			ctx.shadowColor = color;
			ctx.shadowBlur = 6;
			canvas.renderLink(ctx, a, b, null, true, 0, translucent(color), startDir, LiteGraph.LEFT);
		}
	} finally {
		canvas.highquality_render = hq;
		canvas.render_connections_border = border;
		ctx.restore();
	}
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

app.registerExtension({
	name: "BCNodes.AnythingEverywhere",

	settings: [
		{
			id: SETTING_SHOW_LINKS,
			category: ["BCNodes", "Anything Everywhere", "Show links"],
			name: "Show links",
			tooltip: "When the phantom links from an Anything Everywhere node to the inputs it feeds are drawn",
			type: "combo",
			options: [
				{ value: SHOW_OFF, text: "All off" },
				{ value: SHOW_SELECTED, text: "Selected nodes" },
				{ value: SHOW_HOVER, text: "Mouseover node" },
				{ value: SHOW_SELECTED_OR_HOVER, text: "Selected and mouseover nodes" },
				{ value: SHOW_ALL, text: "All on" },
			],
			defaultValue: SHOW_SELECTED_OR_HOVER,
			onChange: () => app.canvas?.setDirty(true, true),
		},
		{
			id: SETTING_HIGHLIGHT,
			category: ["BCNodes", "Anything Everywhere", "Highlight"],
			name: "Highlight connected and connectable inputs",
			type: "boolean",
			defaultValue: true,
			onChange: () => app.canvas?.setDirty(true, true),
		},
	],

	setup() {
		const original = app.graphToPrompt;
		app.graphToPrompt = async function (...args) {
			const result = await original.apply(this, args);
			try {
				apply(result);
			} catch (e) {
				console.error("[BCNodes] Anything Everywhere failed", e);
			}
			return result;
		};

		const drawNode = LGraphCanvas.prototype.drawNode;
		LGraphCanvas.prototype.drawNode = function (node, ctx, ...rest) {
			const r = drawNode.apply(this, [node, ctx, ...rest]);
			try {
				highlightNode(node, ctx);
			} catch (e) {
				console.error("[BCNodes] Anything Everywhere:", e);
			}
			return r;
		};

		const drawConnections = LGraphCanvas.prototype.drawConnections;
		LGraphCanvas.prototype.drawConnections = function (ctx, ...rest) {
			const r = drawConnections.apply(this, [ctx, ...rest]);
			try {
				drawPhantomLinks(this, ctx);
			} catch (e) {
				console.error("[BCNodes] Anything Everywhere:", e);
			}
			return r;
		};
	},

	afterConfigureGraph() {
		invalidate();
		for (const graph of allGraphs()) {
			for (const node of graph.nodes ?? graph._nodes ?? []) if (isUE(node)) fixInputs(node);
		}
	},

	beforeRegisterNodeDef(nodeType, nodeData) {
		// Any link change anywhere can change who feeds whom.
		const onConnectionsChange = nodeType.prototype.onConnectionsChange;
		nodeType.prototype.onConnectionsChange = function (side, slot, connected, link, ioSlot, ...rest) {
			const r = onConnectionsChange?.apply(this, [side, slot, connected, link, ioSlot, ...rest]);
			invalidate();
			if (isUE(this) && side === LiteGraph.INPUT && !app.configuringGraph) {
				const input = this.inputs?.[slot];
				if (input) {
					try {
						styleSlot(input, connected ? linkType(this.graph, link) : "*");
					} catch (e) {
						console.error("[BCNodes] Anything Everywhere:", e);
					}
				}
				setTimeout(() => {
					try {
						fixInputs(this);
					} catch (e) {
						console.error("[BCNodes] Anything Everywhere:", e);
					}
				}, 0);
			}
			return r;
		};

		if (nodeData?.name !== NODE_TYPE) return;
		nodeType["@title_regex"] = { type: "string" };
		nodeType["@input_regex"] = { type: "string" };

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			this.properties = this.properties ?? {};
			this.properties.title_regex = this.properties.title_regex ?? "";
			this.properties.input_regex = this.properties.input_regex ?? "";
			for (const input of this.inputs ?? []) if (input.link == null) styleSlot(input, "*");
			return r;
		};

		const onPropertyChanged = nodeType.prototype.onPropertyChanged;
		nodeType.prototype.onPropertyChanged = function (...args) {
			const r = onPropertyChanged?.apply(this, args);
			invalidate();
			return r;
		};
	},
});

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Image Comparer — drawn straight onto the node canvas, so it moves with the
// node and never lags behind it.
//
// The Vue renderer draws the widget in a canvas of its own instead: there the
// widget is as tall as the media at the widget's width, it is painted again
// only on triggerDraw, and the hover divider follows pointer events on that
// canvas (the node's onMouse* callbacks are not called there).
//
// Slide mode: A fills the node; while the pointer is over the node, B is
// painted from the left edge up to the pointer, with a divider line and A/B
// tags. Leave the node and A is shown alone. The node keeps whatever size
// the user gives it; the media is letterboxed inside.
//
// What is shown comes from app.nodeOutputs, the frontend's record of every
// node's last output: filled by the "executed" event, restored with the
// workflow tab and by a click in the queue history, empty after a restart.
// Nothing is saved into the workflow file — like Preview Image, the
// comparison lives with the run.

const IMAGE_NODE = "BC_ImageComparer";
const WIDGET_TYPE = "BC_COMPARER";
const TAG_FONT = "bold 12px sans-serif";

function viewUrl(info) {
	const q = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder ?? "", type: info.type ?? "temp" });
	return api.apiURL(`/view?${q}`);
}

// The node's last output, keyed the way the frontend stores it: the node id
// at the root, "<subgraph id>:<node id>" inside a subgraph.
function storedOutput(node) {
	const g = node.graph;
	const key = !g || g === (g.rootGraph ?? g) ? String(node.id) : `${g.id}:${node.id}`;
	return app.nodeOutputs?.[key];
}

// ---------------------------------------------------------------------------
// Media entries: {name, url, el}
// ---------------------------------------------------------------------------

// A source that fails to load (a temp file gone after a restart, typically)
// is marked and drawn as "not available" instead of loading forever.
function loadMedia(entry, widget) {
	if (entry.el) return entry.el;
	const img = new Image();
	img.src = entry.url;
	img.onload = () => widget.redraw();
	img.onerror = () => {
		entry.failed = true;
		widget.redraw();
	};
	entry.el = img;
	return entry.el;
}

function mediaSize(el, entry) {
	if (!el || entry?.failed) return [0, 0];
	return el.complete ? [el.naturalWidth, el.naturalHeight] : [0, 0];
}

// Letterbox [w, h] into the box [x, y, bw, bh].
function fitRect(w, h, x, y, bw, bh) {
	if (!w || !h) return null;
	const scale = Math.min(bw / w, bh / h);
	const dw = w * scale;
	const dh = h * scale;
	return [x + (bw - dw) / 2, y + (bh - dh) / 2, dw, dh];
}

// ---------------------------------------------------------------------------
// Widget
// ---------------------------------------------------------------------------

class ComparerWidget {
	constructor(node) {
		this.type = WIDGET_TYPE;
		this.name = "comparer";
		this.node = node;
		this.value = { entries: [] };
		this.serialize = false; // not written to the workflow file
		this.options = { serialize: false }; // not sent in the prompt
		this.seen = undefined; // the stored output the entries were built from
		this.a = null;
		this.b = null;
		this.hover = null; // local x while the pointer is over the node
		this.hits = [];
	}

	// ---- entries ------------------------------------------------------------

	// Rebuild the entries whenever the stored output is a different object:
	// a new run, a tab switch, a history click, a fresh workflow.
	sync() {
		const out = storedOutput(this.node);
		if (out === this.seen) return;
		this.seen = out;
		this.fromExecuted(out);
	}

	setEntries(entries) {
		this.value = { entries: entries.map((e) => ({ name: e.name, url: e.url, failed: false })) };
		const all = this.value.entries;
		this.select(all.find((e) => e.name.startsWith("A")), all.find((e) => e.name.startsWith("B")));
	}

	select(a, b) {
		this.a = a ?? null;
		this.b = b ?? null;
		for (const e of [this.a, this.b]) if (e) loadMedia(e, this);
		this.redraw();
	}

	// The node canvas repaints on setDirtyCanvas. The Vue renderer sizes its
	// canvas from the arranged height and paints it only on triggerDraw.
	redraw() {
		this.node.setDirtyCanvas(true, false);
		if (LiteGraph.vueNodesMode && this.node.graph) {
			this.node.arrange();
			this.triggerDraw?.();
		}
	}

	fromExecuted(message) {
		const entries = [];
		const a = message?.a_images ?? [];
		const b = message?.b_images ?? [];
		const many = a.length > 1 || b.length > 1;
		a.forEach((info, i) => entries.push({ name: many ? `A${i + 1}` : "A", url: viewUrl(info) }));
		b.forEach((info, i) => entries.push({ name: many ? `B${i + 1}` : "B", url: viewUrl(info) }));
		this.setEntries(entries);
	}

	// ---- layout -------------------------------------------------------------

	// On the node canvas the widget is a 20 px row and draws down to the node's
	// bottom. In the Vue renderer its height is its own: the selector row plus
	// the media at the widget's width (4:3 until an image has loaded).
	computeSize(width) {
		if (!LiteGraph.vueNodesMode) return [width, 20];
		const w = this.vueCanvas?.clientWidth || this.node.size[0];
		const src = this.source(this.a) ?? this.source(this.b);
		const ratio = src ? src.h / src.w : 0.75;
		return [width, this.selectorHeight() + Math.round((w - 8) * ratio) + 4];
	}

	selectorHeight() {
		return this.value.entries.length > 2 ? 22 : 0;
	}

	inBox(pos) {
		if (typeof this.y !== "number") return false;
		return within(this.box(this.y, this.node.size[0], this.node.size[1] - 4), pos[0], pos[1]);
	}

	// The media box: below the selector row when there is one.
	box(y, width, bottom) {
		const top = y + this.selectorHeight();
		return [4, top, width - 8, Math.max(20, bottom - top)];
	}

	// What to paint for one side: the element and its size.
	source(entry) {
		if (!entry) return null;
		const el = loadMedia(entry, this);
		const [w, h] = mediaSize(el, entry);
		if (!w || !h) return null;
		return { el, w, h };
	}

	paint(ctx, src, rect) {
		ctx.drawImage(src.el, 0, 0, src.w, src.h, rect[0], rect[1], rect[2], rect[3]);
	}

	draw(ctx, node, width, y, height) {
		// On the node canvas the widget spans the node. The Vue-nodes legacy renderer
		// leaves its own width on the widget (`widget.width || nodeWidth`), which
		// would otherwise stick after switching back to the classic canvas.
		const onNodeCanvas = ctx.canvas === app.canvas?.canvas;
		if (onNodeCanvas) width = node.size[0];
		else this.watchVueCanvas(ctx.canvas, width);
		guard(() => this.sync());
		this.hits = [];
		ctx.save();
		if (this.value.entries.length > 2) this.drawSelector(ctx, width, y);
		const [bx, by, bw, bh] = this.box(y, width, onNodeCanvas ? node.size[1] - 4 : y + height);
		this.vueBox = onNodeCanvas ? null : [bx, by, bw, bh];
		ctx.fillStyle = "#111";
		ctx.fillRect(bx, by, bw, bh);

		const srcA = this.source(this.a);
		const srcB = this.source(this.b);
		const rectA = srcA ? fitRect(srcA.w, srcA.h, bx, by, bw, bh) : null;
		const rectB = srcB ? fitRect(srcB.w, srcB.h, bx, by, bw, bh) : null;

		if (!rectA && !rectB) {
			const pending = [this.a, this.b].some((e) => e && !e.failed);
			ctx.fillStyle = "#666";
			ctx.font = "12px sans-serif";
			ctx.textAlign = "center";
			ctx.textBaseline = "middle";
			ctx.fillText(pending ? "loading…" : "run the workflow to compare", bx + bw / 2, by + bh / 2);
		} else if (!rectA || !rectB) {
			this.paint(ctx, srcA ?? srcB, rectA ?? rectB);
			this.tag(ctx, rectA ? "A" : "B", bx + bw - 8, by + 8, "right");
		} else {
			this.paint(ctx, srcA, rectA);
			if (this.hover != null) {
				const x = Math.min(Math.max(this.hover, bx), bx + bw);
				ctx.save();
				ctx.beginPath();
				ctx.rect(bx, by, x - bx, bh);
				ctx.clip();
				this.paint(ctx, srcB, rectB);
				ctx.restore();
				ctx.fillStyle = "rgba(255,255,255,.9)";
				ctx.fillRect(x - 1, by, 2, bh);
				this.tag(ctx, "B", bx + 8, by + 8, "left");
				this.tag(ctx, "A", bx + bw - 8, by + 8, "right");
			}
		}
		ctx.restore();
	}

	tag(ctx, text, x, y, align) {
		ctx.font = TAG_FONT;
		ctx.textBaseline = "top";
		ctx.textAlign = align;
		const w = ctx.measureText(text).width + 10;
		ctx.fillStyle = "rgba(0,0,0,.6)";
		ctx.fillRect(align === "left" ? x - 2 : x - w + 2, y - 2, w, 17);
		ctx.fillStyle = "#fff";
		ctx.fillText(text, align === "left" ? x + 3 : x - 3, y);
	}

	drawSelector(ctx, width, y) {
		ctx.font = "13px sans-serif";
		ctx.textBaseline = "top";
		ctx.textAlign = "left";
		const gap = 8;
		let total = 0;
		const items = this.value.entries.map((e) => {
			const w = ctx.measureText(e.name).width;
			total += w + gap;
			return { e, w };
		});
		let x = Math.max(0, (width - total + gap) / 2);
		for (const { e, w } of items) {
			const on = e === this.a || e === this.b;
			ctx.fillStyle = on ? "#ddd" : "#777";
			ctx.fillText(e.name, x, y + 2);
			this.hits.push({ x, y, w, h: 18, action: () => this.pick(e) });
			x += w + gap;
		}
	}

	pick(e) {
		if (e.name.startsWith("A")) this.select(e, this.b);
		else this.select(this.a, e);
	}

	// ---- pointer ------------------------------------------------------------

	// Vue renderer: the divider follows the pointer over the widget's own
	// canvas. Event offsets are CSS pixels of the element; the drawing uses
	// the width the renderer passed to draw().
	watchVueCanvas(canvas, width) {
		this.vueWidth = width;
		if (this.vueCanvas === canvas) return;
		this.vueCanvas = canvas;
		canvas.addEventListener("pointermove", (e) => {
			const k = this.vueWidth / (canvas.clientWidth || this.vueWidth);
			const x = e.offsetX * k;
			const hover = this.vueBox && within(this.vueBox, x, e.offsetY * k) ? x : null;
			if (hover === this.hover) return;
			this.hover = hover;
			this.triggerDraw?.();
		});
		canvas.addEventListener("pointerleave", () => {
			if (this.hover == null) return;
			this.hover = null;
			this.triggerDraw?.();
		});
	}

	// The selector row's hit areas are routed from the node's onMouseDown,
	// not from the widget row litegraph would limit `mouse` to.
	click(pos) {
		const h = this.hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w && pos[1] >= h.y && pos[1] <= h.y + h.h);
		if (!h) return false;
		h.action();
		return true;
	}

	// Clicks inside the widget's own row arrive here instead.
	mouse(event, pos) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		return this.click(pos);
	}
}

// ---------------------------------------------------------------------------
// Node wiring
// ---------------------------------------------------------------------------

function within([x, y, w, h], px, py) {
	return px >= x && px <= x + w && py >= y && py <= y + h;
}

function guard(fn) {
	try {
		fn();
	} catch (e) {
		console.error("[BCNodes] comparer:", e);
	}
}

function install(nodeType) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function (...args) {
		const r = onNodeCreated?.apply(this, args);
		this.bcComparer = new ComparerWidget(this);
		this.addCustomWidget(this.bcComparer);
		this.setSize([Math.max(this.size[0], 320), Math.max(this.size[1], 260)]);
		return r;
	};

	// The divider follows the pointer only while it is over the media box;
	// over the title or the selector row the node shows A alone.
	// A new run: the Vue renderer does not repaint the widget by itself.
	const onExecuted = nodeType.prototype.onExecuted;
	nodeType.prototype.onExecuted = function (...args) {
		const r = onExecuted?.apply(this, args);
		if (LiteGraph.vueNodesMode) guard(() => this.bcComparer?.sync());
		return r;
	};

	const onMouseMove = nodeType.prototype.onMouseMove;
	nodeType.prototype.onMouseMove = function (event, pos, ...rest) {
		const r = onMouseMove?.apply(this, [event, pos, ...rest]);
		const cmp = this.bcComparer;
		if (cmp) {
			cmp.hover = cmp.inBox(pos) ? pos[0] : null;
			this.setDirtyCanvas(true, false);
		}
		return r;
	};

	const onMouseEnter = nodeType.prototype.onMouseEnter;
	nodeType.prototype.onMouseEnter = function (event, pos, ...rest) {
		const r = onMouseEnter?.apply(this, [event, pos, ...rest]);
		const cmp = this.bcComparer;
		if (cmp && pos) cmp.hover = cmp.inBox(pos) ? pos[0] : null;
		return r;
	};

	const onMouseLeave = nodeType.prototype.onMouseLeave;
	nodeType.prototype.onMouseLeave = function (...args) {
		const r = onMouseLeave?.apply(this, args);
		if (this.bcComparer) {
			this.bcComparer.hover = null;
			this.setDirtyCanvas(true, false);
		}
		return r;
	};

	const onMouseDown = nodeType.prototype.onMouseDown;
	nodeType.prototype.onMouseDown = function (event, pos, ...rest) {
		if (this.bcComparer?.click(pos)) return true;
		return onMouseDown?.apply(this, [event, pos, ...rest]);
	};
}

app.registerExtension({
	name: "BCNodes.Comparers",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name === IMAGE_NODE) install(nodeType);
	},
});

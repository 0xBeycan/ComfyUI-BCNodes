import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Image Comparer and Video Comparer — drawn straight onto the node canvas,
// so they move with the node and never lag behind it.
//
// Slide mode: A fills the node; while the pointer is over the node, B is
// painted from the left edge up to the pointer, with a divider line and A/B
// tags. Leave the node and A is shown alone. The node keeps whatever size
// the user gives it; the media is letterboxed inside.
//
// Video: the same, from ONE hidden <video> that holds A and B side by side
// (nodes/video_comparer.py encodes them into a single file), each half
// painted into its own place — one decoder, one clock, nothing to keep in
// sync. Nothing plays until asked: the play button in the top row or a
// click on the video toggles playback, the seek bar at the bottom scrubs,
// the clip loops. Clips of different length were cut to the shorter one
// and a note says so; with an audio track a speaker button mutes it.
//
// What is shown comes from app.nodeOutputs, the frontend's record of every
// node's last output: filled by the "executed" event, restored with the
// workflow tab and by a click in the queue history, empty after a restart.
// Nothing is saved into the workflow file — like Preview Image, the
// comparison lives with the run.

const IMAGE_NODE = "BC_ImageComparer";
const VIDEO_NODE = "BC_VideoComparer";
const WIDGET_TYPE = "BC_COMPARER";
const BAR_H = 22;
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
// Media entries: {name, url, kind: "image" | "video", el}
// A video entry also carries sides (["A", "B"], ["A"] or ["B"]: which halves
// the file holds, left to right), audio and note.
// ---------------------------------------------------------------------------

// A source that fails to load (a temp file gone after a restart, typically)
// is marked and drawn as "not available" instead of loading forever.
function loadMedia(entry, node) {
	if (entry.el) return entry.el;
	const failed = () => {
		entry.failed = true;
		node.setDirtyCanvas(true, false);
	};
	if (entry.kind === "video") {
		const v = document.createElement("video");
		v.playsInline = true;
		v.preload = "auto";
		v.loop = true;
		v.src = entry.url;
		v.addEventListener("loadeddata", () => node.setDirtyCanvas(true, false));
		v.addEventListener("error", failed);
		entry.el = v;
	} else {
		const img = new Image();
		img.src = entry.url;
		img.onload = () => node.setDirtyCanvas(true, false);
		img.onerror = failed;
		entry.el = img;
	}
	return entry.el;
}

function mediaSize(el, entry) {
	if (!el || entry?.failed) return [0, 0];
	if (el.tagName === "VIDEO") return el.readyState >= 1 ? [el.videoWidth, el.videoHeight] : [0, 0];
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
	constructor(node, video) {
		this.type = WIDGET_TYPE;
		this.name = "comparer";
		this.node = node;
		this.video = video;
		this.value = { entries: [] };
		this.serialize = false; // not written to the workflow file
		this.options = { serialize: false }; // not sent in the prompt
		this.seen = undefined; // the stored output the entries were built from
		this.a = null;
		this.b = null;
		this.hover = null; // local x while the pointer is over the node
		this.hits = [];
		this.muted = false;
		this.raf = null;
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
		this.stop();
		this.value = { entries: entries.map((e) => ({ name: e.name, url: e.url, kind: e.kind, sides: e.sides, audio: e.audio, note: e.note, failed: false })) };
		const all = this.value.entries;
		if (this.video) {
			// One clip holds both sides; A and B are its halves.
			const clip = all.find((e) => e.kind === "video" && Array.isArray(e.sides));
			this.select(clip?.sides.includes("A") ? clip : null, clip?.sides.includes("B") ? clip : null);
		} else {
			this.select(all.find((e) => e.name.startsWith("A")), all.find((e) => e.name.startsWith("B")));
		}
	}

	select(a, b) {
		this.stop();
		this.a = a ?? null;
		this.b = b ?? null;
		for (const e of [this.a, this.b]) if (e) loadMedia(e, this.node);
		this.node.setDirtyCanvas(true, false);
	}

	fromExecuted(message) {
		const entries = [];
		if (this.video) {
			for (const info of message?.bc_video ?? []) {
				const sides = info.sides ?? ["A", "B"];
				const f = info.frames ?? {};
				let note = "";
				if (sides.length === 2 && f.A !== f.B && info.fps) {
					note = `A ${(f.A / info.fps).toFixed(2)}s · B ${(f.B / info.fps).toFixed(2)}s — cut to the shorter clip`;
				}
				entries.push({ name: sides.join(""), url: viewUrl(info), kind: "video", sides, audio: !!info.audio, note });
			}
		} else {
			const a = message?.a_images ?? [];
			const b = message?.b_images ?? [];
			const many = a.length > 1 || b.length > 1;
			a.forEach((info, i) => entries.push({ name: many ? `A${i + 1}` : "A", url: viewUrl(info), kind: "image" }));
			b.forEach((info, i) => entries.push({ name: many ? `B${i + 1}` : "B", url: viewUrl(info), kind: "image" }));
		}
		this.setEntries(entries);
	}

	// ---- layout -------------------------------------------------------------

	computeSize(width) {
		return [width, 20];
	}

	inBox(pos) {
		if (typeof this.y !== "number") return false;
		const [bx, by, bw, bh] = this.box(this.y);
		return pos[0] >= bx && pos[0] <= bx + bw && pos[1] >= by && pos[1] <= by + bh;
	}

	// The media box: below the top row (play / time / selector), above the seek bar.
	box(y) {
		const [w, h] = this.node.size;
		const top = y + (this.video || this.value.entries.length > 2 ? 22 : 0);
		const bottom = h - (this.video ? BAR_H : 4);
		return [4, top, w - 8, Math.max(20, bottom - top)];
	}

	// What to paint for one side: the element and the source rect inside it.
	// An image entry is the whole picture; a video entry is split into its
	// sides, left to right.
	source(entry, side) {
		if (!entry) return null;
		const el = loadMedia(entry, this.node);
		const [w, h] = mediaSize(el, entry);
		if (!w || !h) return null;
		const sides = entry.sides ?? [side];
		const i = sides.indexOf(side);
		if (i < 0) return null;
		const sw = w / sides.length;
		return { el, sx: i * sw, sw, sh: h };
	}

	paint(ctx, src, rect) {
		ctx.drawImage(src.el, src.sx, 0, src.sw, src.sh, rect[0], rect[1], rect[2], rect[3]);
	}

	draw(ctx, node, width, y) {
		// On the node canvas the widget spans the node. The Vue-nodes legacy renderer
		// leaves its own width on the widget (`widget.width || nodeWidth`), which
		// would otherwise stick after switching back to the classic canvas.
		if (ctx.canvas === app.canvas?.canvas) width = node.size[0];
		guard(() => this.sync());
		this.hits = [];
		ctx.save();
		if (this.video) this.drawTopRow(ctx, node, width, y);
		if (this.value.entries.length > 2) this.drawSelector(ctx, width, y);
		const [bx, by, bw, bh] = this.box(y);
		ctx.fillStyle = "#111";
		ctx.fillRect(bx, by, bw, bh);

		const srcA = this.source(this.a, "A");
		const srcB = this.source(this.b, "B");
		const rectA = srcA ? fitRect(srcA.sw, srcA.sh, bx, by, bw, bh) : null;
		const rectB = srcB ? fitRect(srcB.sw, srcB.sh, bx, by, bw, bh) : null;

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
		if (this.video) this.drawBar(ctx, node, y);
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
		// centred, but to the right of the play button when there is one
		let x = Math.max(this.video ? 120 : 0, (width - total + gap) / 2);
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

	// Hit areas live anywhere on the node (selector row, play button, seek
	// bar), so clicks are routed from the node's onMouseDown, not from the
	// widget row litegraph would limit `mouse` to.
	hitAt(pos) {
		return this.hits.find((h) => pos[0] >= h.x && pos[0] <= h.x + h.w && pos[1] >= h.y && pos[1] <= h.y + h.h);
	}

	click(pos) {
		const h = this.hitAt(pos);
		if (h) {
			if (h.seek) h.seek(pos[0]);
			else h.action?.();
			return true;
		}
		// A click on the video itself toggles playback, like a video player.
		if (this.video && this.clip() && this.inBox(pos)) {
			this.toggle();
			return true;
		}
		return false;
	}

	// Clicks inside the widget's own row (the top row) arrive here instead.
	mouse(event, pos) {
		if (event.type !== "pointerdown" && event.type !== "mousedown") return false;
		return this.click(pos);
	}

	// ---- video --------------------------------------------------------------

	// The one <video> behind A and B, once it has metadata.
	clip() {
		const el = (this.a ?? this.b)?.el;
		return el && el.tagName === "VIDEO" && el.readyState >= 1 ? el : null;
	}

	// Top row: play / pause at the left, time, the speaker when there is audio,
	// the note at the right.
	drawTopRow(ctx, node, width, y) {
		const v = this.clip();
		if (!v) return;
		const t = v.currentTime || 0;
		const d = v.duration || 0;
		const cy = y + 10;

		ctx.fillStyle = "#ddd";
		const cx = 18;
		if (!v.paused) {
			ctx.fillRect(cx - 5, cy - 6, 3, 12);
			ctx.fillRect(cx + 2, cy - 6, 3, 12);
		} else {
			ctx.beginPath();
			ctx.moveTo(cx - 5, cy - 6);
			ctx.lineTo(cx + 6, cy);
			ctx.lineTo(cx - 5, cy + 6);
			ctx.fill();
		}
		this.hits.push({ x: 4, y: y - 2, w: 30, h: 24, action: () => this.toggle() });

		ctx.font = "11px sans-serif";
		ctx.textBaseline = "middle";
		ctx.textAlign = "left";
		ctx.fillStyle = "#bbb";
		const time = `${t.toFixed(2)} / ${d.toFixed(2)}`;
		ctx.fillText(time, 34, cy);

		const entry = this.a ?? this.b;
		if (entry?.audio) {
			const sx = 34 + ctx.measureText(time).width + 12;
			this.drawSpeaker(ctx, sx, cy, this.muted);
			this.hits.push({ x: sx - 4, y: y - 2, w: 24, h: 24, action: () => this.toggleMute() });
		}
		if (entry?.note) {
			ctx.textAlign = "right";
			ctx.fillStyle = "#e0b060";
			ctx.font = "10px sans-serif";
			ctx.fillText(entry.note, width - 6, cy);
		}
	}

	// A small speaker: body + cone, two arcs when sounding, a slash when muted.
	drawSpeaker(ctx, x, cy, muted) {
		ctx.fillStyle = muted ? "#777" : "#ddd";
		ctx.strokeStyle = ctx.fillStyle;
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.moveTo(x, cy - 3);
		ctx.lineTo(x + 3, cy - 3);
		ctx.lineTo(x + 7, cy - 6);
		ctx.lineTo(x + 7, cy + 6);
		ctx.lineTo(x + 3, cy + 3);
		ctx.lineTo(x, cy + 3);
		ctx.closePath();
		ctx.fill();
		if (muted) {
			ctx.beginPath();
			ctx.moveTo(x + 9, cy - 4);
			ctx.lineTo(x + 15, cy + 4);
			ctx.stroke();
		} else {
			for (const r of [4, 7]) {
				ctx.beginPath();
				ctx.arc(x + 7, cy, r, -Math.PI / 3, Math.PI / 3);
				ctx.stroke();
			}
		}
	}

	// Bottom: the seek bar alone, full width.
	drawBar(ctx, node, y) {
		const [w, h] = node.size;
		const v = this.clip();
		if (!v) return;
		const t = v.currentTime || 0;
		const d = v.duration || 0;
		const top = h - BAR_H;
		const cy = top + BAR_H / 2;
		const tx = 10;
		const tw = w - 20;
		ctx.fillStyle = "#333";
		ctx.fillRect(tx, cy - 2, tw, 4);
		ctx.fillStyle = "#4a90e2";
		ctx.fillRect(tx, cy - 2, d ? (tw * Math.min(t, d)) / d : 0, 4);
		const px = tx + (d ? (tw * Math.min(t, d)) / d : 0);
		ctx.beginPath();
		ctx.arc(px, cy, 5, 0, Math.PI * 2);
		ctx.fillStyle = "#e6e6e6";
		ctx.fill();
		this.hits.push({ x: 4, y: top, w: w - 8, h: BAR_H, action: null, seek: (x) => this.seek(((Math.min(Math.max(x, tx), tx + tw) - tx) / tw) * d) });
	}

	toggle() {
		const v = this.clip();
		if (!v) return;
		if (!v.paused) return this.stop();
		v.muted = this.muted;
		v.play().catch(() => {});
		// Repaint on every frame while it plays.
		const step = () => {
			if (v.paused) {
				this.raf = null;
				this.node.setDirtyCanvas(true, false);
				return;
			}
			this.node.setDirtyCanvas(true, false);
			this.raf = requestAnimationFrame(step);
		};
		if (this.raf) cancelAnimationFrame(this.raf);
		this.raf = requestAnimationFrame(step);
	}

	stop() {
		this.clip()?.pause();
		if (this.raf) cancelAnimationFrame(this.raf);
		this.raf = null;
		this.node.setDirtyCanvas?.(true, false);
	}

	seek(t) {
		const v = this.clip();
		if (v) v.currentTime = Math.max(0, t);
		this.node.setDirtyCanvas(true, false);
	}

	toggleMute() {
		this.muted = !this.muted;
		const v = this.clip();
		if (v) v.muted = this.muted;
		this.node.setDirtyCanvas(true, false);
	}
}

// ---------------------------------------------------------------------------
// Node wiring
// ---------------------------------------------------------------------------

function guard(fn) {
	try {
		fn();
	} catch (e) {
		console.error("[BCNodes] comparer:", e);
	}
}

function install(nodeType, video) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function (...args) {
		const r = onNodeCreated?.apply(this, args);
		this.bcComparer = new ComparerWidget(this, video);
		this.addCustomWidget(this.bcComparer);
		this.setSize([Math.max(this.size[0], 320), Math.max(this.size[1], video ? 300 : 260)]);
		return r;
	};

	// The divider follows the pointer only while it is over the media box;
	// over the title, the top row or the seek bar the node shows A alone.
	const onMouseMove = nodeType.prototype.onMouseMove;
	nodeType.prototype.onMouseMove = function (event, pos, ...rest) {
		const r = onMouseMove?.apply(this, [event, pos, ...rest]);
		const cmp = this.bcComparer;
		if (cmp) {
			cmp.hover = cmp.inBox(pos) ? pos[0] : null;
			if (event.buttons & 1) {
				const seek = cmp.hits.find((h) => h.seek && pos[0] >= h.x && pos[0] <= h.x + h.w && pos[1] >= h.y && pos[1] <= h.y + h.h);
				if (seek) seek.seek(pos[0]);
			}
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

	// Never let a widget problem escape into graph teardown: an exception in
	// onRemoved aborts LGraph.clear() half way and corrupts the next workflow.
	const onRemoved = nodeType.prototype.onRemoved;
	nodeType.prototype.onRemoved = function (...args) {
		guard(() => this.bcComparer?.stop());
		return onRemoved?.apply(this, args);
	};
}

app.registerExtension({
	name: "BCNodes.Comparers",
	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name === IMAGE_NODE) install(nodeType, false);
		if (nodeData?.name === VIDEO_NODE) install(nodeType, true);
	},
});

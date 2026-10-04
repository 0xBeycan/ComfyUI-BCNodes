import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { callJson } from "./bcnodes_api.js";

// Process Monitor — the frontend half of nodes/process_monitor.py.
//
// One ComfyUI setting turns it on and off, live. The top bar shows RAM against its limit, VRAM
// and the GPU load while it is on; next to them the Full clear button (RAM and VRAM back to the
// reading taken when ComfyUI started, no restart; its outcome in a ComfyUI toast, the details in
// the button's tooltip), and a button that opens the modal. Both buttons are always there: the full
// clear, Emulate and the crash report work with the monitor off. The modal button turns red when the last
// run was killed. Modal tabs: Live, Emulate, Last run (a row click selects and centres the node),
// Crash, Settings.

const SETTING = "BCNodes.ProcessMonitor.Enabled";
const EVENT = "bcnodes.monitor";
const TABS = ["Live", "Emulate", "Last run", "Crash", "Settings"];
// the tensor census's "backing" of cpu memory (none on a GPU)
const BACKING = { ram: "RAM", file: "file (page cache)", unknown: "unknown" };
const CLEAR_TITLE = "Full clear: RAM and VRAM back to where they were right after ComfyUI started, without a restart "
	+ "(every model unloaded, every cached node output dropped, the freed memory given back to the system). "
	+ "Refused while a prompt runs or waits. The next run loads its models again, so it starts slower.";

const state = { status: null, sample: null, tab: "Live", modal: null, bar: null };

const call = (route, body) => callJson(`/bcnodes/monitor/${route}`, body);

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function gb(bytes) {
	return bytes == null ? "–" : `${(bytes / 2 ** 30).toFixed(bytes >= 10 * 2 ** 30 ? 1 : 2)} GB`;
}

function secs(s) {
	if (s == null) return "–";
	return s >= 120 ? `${(s / 60).toFixed(1)} min` : `${s.toFixed(s < 1 ? 3 : 1)} s`;
}

function el(tag, attrs = {}, ...children) {
	const e = document.createElement(tag);
	for (const [k, v] of Object.entries(attrs)) {
		if (k === "onclick") e.onclick = v;
		else if (k === "style") e.style.cssText = v;
		else e.setAttribute(k, v);
	}
	for (const c of children.flat()) if (c != null) e.append(c instanceof Node ? c : String(c));
	return e;
}

function table(head, rows, onRow) {
	return el("table", { class: "bcpm-table" },
		el("thead", {}, el("tr", {}, head.map((h) => el("th", {}, h)))),
		el("tbody", {}, rows.map((r) => {
			const tr = el("tr", {}, r.cells.map((c) => el("td", {}, c)));
			if (onRow && r.key != null) {
				tr.classList.add("bcpm-click");
				tr.onclick = () => onRow(r.key);
			}
			return tr;
		})));
}

function ensureStyle() {
	if (document.getElementById("bcpm-style")) return;
	const style = el("style", { id: "bcpm-style" });
	style.textContent = `
.bcpm-bar{display:flex;align-items:center;gap:6px;font:11px sans-serif;color:var(--fg-color,#ddd)}
.bcpm-meter{position:relative;min-width:70px;padding:0 6px;height:16px;border-radius:4px;background:var(--comfy-input-bg,#333);overflow:hidden}
.bcpm-meter>div{position:absolute;inset:0 auto 0 0;background:#3b82f6}
.bcpm-meter>span{position:relative;display:block;text-align:center;line-height:16px;white-space:nowrap}
.bcpm-btn{cursor:pointer;border:1px solid var(--border-color,#555);border-radius:4px;padding:1px 6px;background:var(--comfy-input-bg,#333);color:inherit}
.bcpm-btn.bcpm-red{background:#b91c1c;border-color:#ef4444;color:#fff}
.bcpm-btn:disabled{opacity:.6;cursor:default}
.bcpm-overlay{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.5);display:flex;align-items:center;justify-content:center}
.bcpm-panel{width:min(1100px,94vw);height:min(760px,90vh);display:flex;flex-direction:column;background:var(--comfy-menu-bg,#222);color:var(--fg-color,#ddd);border:1px solid var(--border-color,#555);border-radius:8px;font:12px sans-serif}
.bcpm-tabs{display:flex;gap:4px;padding:8px;border-bottom:1px solid var(--border-color,#555)}
.bcpm-tabs button.bcpm-on{background:#3b82f6;color:#fff}
.bcpm-body{flex:1;overflow:auto;padding:10px}
.bcpm-table{border-collapse:collapse;width:100%;margin:6px 0}
.bcpm-table th,.bcpm-table td{border-bottom:1px solid var(--border-color,#444);padding:3px 6px;text-align:left;vertical-align:top}
.bcpm-click{cursor:pointer}.bcpm-click:hover{background:rgba(59,130,246,.2)}
.bcpm-note{opacity:.75;margin:4px 0}.bcpm-warn{color:#f59e0b}.bcpm-bad{color:#ef4444}.bcpm-ok{color:#22c55e}
.bcpm-body pre{white-space:pre-wrap;font-size:11px}`;
	document.head.append(style);
}

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------

function meter(label) {
	const fill = el("div");
	const text = el("span", {}, label);
	return { root: el("div", { class: "bcpm-meter", title: label }, fill, text), fill, text };
}

function buildBar() {
	ensureStyle();
	const ram = meter("RAM");
	const vram = meter("VRAM");
	const gpu = meter("GPU");
	const meters = el("div", { class: "bcpm-bar" }, ram.root, vram.root, gpu.root);
	const clear = el("button", { class: "bcpm-btn", title: CLEAR_TITLE }, "Full clear");
	clear.onclick = fullClear;
	const button = el("button", { class: "bcpm-btn", title: "Process Monitor" }, "PM");
	button.onclick = openModal;
	const root = el("div", { class: "bcpm-bar" }, meters, clear, button);
	state.bar = { root, meters, button, clear, ram, vram, gpu };
	// The frontend rebuilds the top menu, action bar included, whenever its layout changes (the
	// right side panel opened or closed, focus mode, the app builder), so an element put into the
	// action bar is dropped with it. The legacy top-menu element (app.menu.element) is the place the
	// frontend keeps for custom scripts: each new top menu takes it into its action bar again. The
	// newer actionBarButtons extension field draws icon buttons only, no meters.
	if (app.menu?.element) app.menu.element.prepend(root);
	else console.warn("[BCNodes] Process Monitor: this ComfyUI frontend has no top menu (app.menu); the bar is not shown. Update the frontend.");
}

function setMeter(m, value, total, text) {
	m.fill.style.width = total ? `${Math.min(100, (100 * value) / total)}%` : "0";
	m.fill.style.background = total && value / total > 0.85 ? "#ef4444" : "#3b82f6";
	m.text.textContent = text;
}

function renderBar() {
	const bar = state.bar;
	if (!bar) return;
	const s = state.sample;
	const on = !!state.status?.enabled;
	bar.meters.style.display = on && s ? "" : "none";
	bar.button.classList.toggle("bcpm-red", !!state.status?.crash);
	bar.button.title = state.status?.crash ? "Process Monitor: the last run was killed (open the Crash tab)" : "Process Monitor";
	if (!on || !s) return;
	setMeter(bar.ram, s.ram, s.ram_limit, `RAM ${gb(s.ram)}`);
	bar.ram.root.title = `RAM ${gb(s.ram)} of ${gb(s.ram_limit)}`;
	const vramTotal = s.vram_total ?? null;
	setMeter(bar.vram, s.vram_device ?? s.vram, vramTotal, `VRAM ${gb(s.vram_device ?? s.vram)}`);
	bar.gpu.root.style.display = s.gpu_util == null ? "none" : "";
	if (s.gpu_util != null) setMeter(bar.gpu, s.gpu_util, 100, `GPU ${s.gpu_util}%`);
}

// "RAM 12.4 GB → 3.10 GB", with " (baseline …)" when asked and known; null when the readings lack the counter
function clearChange(r, label, key, withBaseline) {
	if (r.before[key] == null || r.after[key] == null) return null;
	const base = withBaseline && r.baseline?.[key] != null ? ` (baseline ${gb(r.baseline[key])})` : "";
	return `${label} ${gb(r.before[key])} → ${gb(r.after[key])}${base}`;
}

// The outcome as a ComfyUI toast; a frontend without the toast API gets a console line instead
function clearToast(severity, detail) {
	const toast = app.extensionManager?.toast;
	if (toast?.add) toast.add({ severity, summary: "Full clear", detail, life: severity === "success" ? 5000 : 15000 });
	else if (severity === "success") console.info(`[BCNodes] Full clear: ${detail}`);
	else console.warn(`[BCNodes] Full clear: ${detail}`);
}

async function fullClear() {
	const bar = state.bar;
	bar.clear.disabled = true;
	bar.clear.textContent = "Clearing…";
	try {
		const r = await call("clear", {});
		const failed = (r.steps.find((s) => s.name === "pack_models")?.detail ?? []).filter((h) => h.error);
		const short = [clearChange(r, "RAM", "ram"), clearChange(r, "VRAM", "vram_reserved")].filter(Boolean).join(", ");
		const details = [
			`full clear at ${new Date().toLocaleTimeString()}`,
			clearChange(r, "RAM", "ram", true),
			clearChange(r, "VRAM reserved", "vram_reserved", true),
			r.baseline ? null : "No baseline: the server's startup was not seen.",
			...r.steps.map((s) => `${s.name}: ${s.found}`),
			`Tensors still referenced after the clear: ${gb(r.remaining.total_bytes)}`,
		].filter(Boolean).join("\n");
		clearToast(failed.length ? "warn" : "success", `Cleared: ${short}${failed.length ? ` · ${failed.length} hook(s) failed` : ""}`);
		bar.clear.title = `${CLEAR_TITLE}\n\nLast: ${details}`;
	} catch (e) {
		const message = String(e.message ?? e);
		clearToast("error", `Not cleared: ${message}`);
		bar.clear.title = `${CLEAR_TITLE}\n\nLast: ${message}`;
	} finally {
		bar.clear.disabled = false;
		bar.clear.textContent = "Full clear";
	}
	refreshStatus();
}

// ---------------------------------------------------------------------------
// Modal
// ---------------------------------------------------------------------------

function openModal() {
	ensureStyle();
	closeModal();
	const tabs = el("div", { class: "bcpm-tabs" });
	const body = el("div", { class: "bcpm-body" });
	const close = el("button", { class: "bcpm-btn", style: "margin-left:auto" }, "Close");
	const overlay = el("div", { class: "bcpm-overlay" }, el("div", { class: "bcpm-panel" }, tabs, body));
	// A block body: an onclick that returns false cancels the click, which undoes a checkbox toggle
	// anywhere in the modal.
	overlay.onclick = (e) => {
		if (e.target === overlay) closeModal();
	};
	close.onclick = closeModal;
	for (const t of TABS) {
		const b = el("button", { class: "bcpm-btn" }, t);
		b.onclick = () => showTab(t);
		tabs.append(b);
	}
	tabs.append(close);
	document.body.append(overlay);
	state.modal = { overlay, tabs, body };
	showTab(state.tab);
}

function closeModal() {
	state.modal?.overlay.remove();
	state.modal = null;
}

async function showTab(name) {
	const m = state.modal;
	if (!m) return;
	state.tab = name;
	for (const b of m.tabs.querySelectorAll("button")) b.classList.toggle("bcpm-on", b.textContent === name);
	m.body.replaceChildren(el("div", { class: "bcpm-note" }, "Loading…"));
	try {
		const content = await RENDER[name]();
		if (state.modal === m && state.tab === name) m.body.replaceChildren(content);
	} catch (e) {
		m.body.replaceChildren(el("div", { class: "bcpm-bad" }, String(e.message ?? e)));
	}
}

function focusNode(id) {
	// Display ids: "12" in the root graph, "12:5" inside subgraph node 12 (its outer node is centred).
	const outer = String(id).split(":")[0];
	const graph = app.rootGraph ?? app.graph;
	const node = graph?.getNodeById?.(Number(outer)) ?? graph?.getNodeById?.(outer);
	if (!node) return;
	closeModal();
	app.canvas.selectNode(node, false);
	app.canvas.centerOnNode?.(node);
	app.canvas.setDirty(true, true);
}

function liveTab() {
	const s = state.sample;
	const st = state.status ?? {};
	const rows = [
		["RAM source", st.sources?.ram ?? "–"],
		["VRAM source", st.sources?.vram ?? "–"],
		["Per-node measurement", st.hook?.available ? "available" : st.hook?.message ?? "–"],
	];
	if (st.error) rows.push(["Monitor error", st.error]);
	if (s) {
		rows.push(["RAM", `${gb(s.ram)} of ${gb(s.ram_limit)}${s.ram_raw != null ? ` (with file cache ${gb(s.ram_raw)})` : ""}`]);
		if (s.swap != null) rows.push(["Host swap in use", gb(s.swap)]);
		rows.push(["VRAM (torch)", `${gb(s.vram)} allocated, ${gb(s.vram_reserved)} reserved`]);
		if (s.vram_device != null) rows.push(["VRAM (device, every process)", `${gb(s.vram_device)} of ${gb(s.vram_total)}`]);
		if (s.gpu_util != null) rows.push(["GPU load", `${s.gpu_util}%`]);
		if (s.run) rows.push(["Running", `prompt ${s.run.prompt_id}, node ${s.run.node ?? "–"} ${s.run.class_type ?? ""}, ${secs(s.run.elapsed)}`]);
		if (s.line) rows.push(["Line", s.line]);
	}
	return el("div", {},
		st.enabled ? null : el("div", { class: "bcpm-warn" }, `The monitor is off. Turn it on in Settings > BCNodes > Process Monitor.`),
		table(["", ""], rows.map((r) => ({ cells: r }))),
		el("div", { class: "bcpm-note" }, `Idle sampler time this session: ${secs(st.idle_sampler_s)}`));
}

async function emulateTab() {
	const box = el("div", {});
	const go = el("button", { class: "bcpm-btn" }, "Estimate this workflow");
	const out = el("div", {});
	go.onclick = async () => {
		out.replaceChildren(el("div", { class: "bcpm-note" }, "Estimating…"));
		try {
			const p = await app.graphToPrompt();
			out.replaceChildren(renderEstimate(await call("emulate", { prompt: p.output, workflow: p.workflow })));
		} catch (e) {
			out.replaceChildren(el("div", { class: "bcpm-bad" }, String(e.message ?? e)));
		}
	};
	box.append(go, out);
	return box;
}

function fitText(f) {
	const v = f.vram_fits ? el("span", { class: "bcpm-ok" }, "VRAM fits") : el("span", { class: "bcpm-bad" }, "VRAM does not fit");
	const r = f.ram_limit == null ? "no RAM limit known"
		: f.ram_fits ? el("span", { class: "bcpm-ok" }, `RAM ${gb(f.ram_need)} of ${gb(f.ram_limit)}`)
			: el("span", { class: "bcpm-bad" }, `RAM ${gb(f.ram_need)} over ${gb(f.ram_limit)}`);
	return el("span", {}, v, ", ", r);
}

function renderEstimate(r) {
	const c = r.current;
	const parts = [
		el("div", { class: "bcpm-warn" }, r.label),
		el("div", {}, `Tensor RAM peak ${gb(c.ram_peak)} at node ${c.ram_peak_node ?? "–"}; output cache ${gb(c.cache_total)}; `,
			`VRAM need ${gb(c.vram_peak)} at node ${c.vram_peak_node ?? "–"}; weights ${gb(c.weights_total)}`),
		el("div", { class: "bcpm-note" }, `RAM in use now ${gb(r.ram_now)} (ComfyUI, loaded models, the last prompt's cached outputs) is added in the fit check as an upper bound.`),
		r.calibrated ? el("div", { class: "bcpm-note" }, "Measured nodes of this workflow's armed run replace their formulas.") : null,
		el("div", { class: "bcpm-note" }, r.measure_cost_s == null ? "Measurement cost: unknown until one armed run in this session."
			: `Measurement adds about ${secs(r.measure_cost_s)} to a run (${r.nodes} nodes × this session's measured cost per node).`),
		table(["Target", "Fit"], r.fit.map((f) => ({ cells: [f.gpu, fitText(f)] }))),
	];
	if (r.table) {
		parts.push(el("h4", {}, "Resolution × frames (RAM peak / VRAM need)"));
		parts.push(table(["", ...r.table.frames.map(String)], r.table.rows.map((row) => ({
			cells: [`${row.label} ${row.w}×${row.h}`, ...row.cells.map((x) => `${gb(x.ram_peak)} / ${gb(x.vram_peak)}`)],
		}))));
	}
	if (c.not_counted.length) parts.push(el("div", { class: "bcpm-warn" }, `Not counted: ${c.not_counted.length} node(s), listed below.`));
	parts.push(table(["Node", "Status", "Outputs", "Transient", "RAM at node", "VRAM need", "Note"], c.rows.map((n) => ({
		key: n.id,
		cells: [`${n.id} ${n.title}`, n.status, gb(n.output_bytes), n.transient == null ? "not counted" : gb(n.transient),
			gb(n.ram_at_node), gb(n.vram_need), n.note ?? ""],
	})), focusNode));
	if (r.skipped.length) {
		parts.push(el("h4", {}, "Bypassed / muted (not in the totals)"));
		parts.push(table(["Node", "Status"], r.skipped.map((s) => ({ cells: [`${s.id} ${s.title}`, s.status] }))));
	}
	return el("div", {}, parts);
}

async function lastRunTab() {
	const st = state.status ?? {};
	const arm = el("button", { class: "bcpm-btn" }, st.armed ? "Armed: the next run is measured" : "Measure next run");
	arm.onclick = async () => {
		try {
			state.status = await call("arm", { armed: !state.status?.armed });
			showTab("Last run");
		} catch (e) {
			arm.after(el("div", { class: "bcpm-bad" }, String(e.message ?? e)));
		}
	};
	const { report: r } = await call("last_run");
	const parts = [arm, st.hook?.available ? null : el("div", { class: "bcpm-warn" }, st.hook?.message ?? "")];
	if (!r) return el("div", {}, parts, el("div", { class: "bcpm-note" }, "No finished run logged yet (the black box or an armed run writes one)."));
	const m = r.monitor ?? {};
	parts.push(
		el("div", {}, `Prompt ${r.prompt_id}: ${r.status}. Run ${secs(r.seconds)}, monitor ${secs(m.total_s)}`,
			r.overhead_pct == null ? "" : ` (${r.overhead_pct.toFixed(2)}%: hook ${secs(m.hook_s)}, sampler ${secs(m.sampler_s)}, snapshot ${secs(m.snapshot_s)})`),
		el("div", { class: "bcpm-note" }, r.profile),
		el("div", {}, `RAM peak ${gb(r.ram_peak)} of ${gb(r.ram_limit)} (${r.ram_source}); VRAM peak ${gb(r.vram_peak)}`),
		r.swap ? el("div", { class: "bcpm-warn" }, r.swap.text) : null);
	if (r.armed) {
		parts.push(table(["Node", "State", "Time", "RAM start → peak", "VRAM peak", "Outputs", "Cache", "Models"], r.nodes.map((n) => ({
			key: n.display ?? n.node,
			cells: [`${n.display ?? n.node} ${n.class_type}`, n.state === "cached" ? "from cache, not measured" : n.state,
				n.state === "cached" ? "–" : secs(n.seconds),
				n.state === "cached" ? "–" : `${gb(n.ram_start)} → ${gb(n.ram_peak)} (${n.peak_source})`,
				gb(n.vram_peak), `${gb(n.output_bytes)} ${(n.outputs ?? []).slice(0, 2).map((o) => `[${o.shape}] ${o.dtype}`).join(" ")}`,
				gb(n.cache), [...(n.models_loaded ?? []).map((x) => `+${x.name}`), ...(n.models_unloaded ?? []).map((x) => `−${x.name}`)].join(" ")],
		})), focusNode));
	} else {
		parts.push(el("div", { class: "bcpm-note" }, "Not measured per node: press \"Measure next run\". Node starts the black box logged:"));
		parts.push(table(["At", "Node", "RAM", "Cache"], r.timeline.map((t) => ({
			key: t.display ?? t.node, cells: [secs(t.at), `${t.display ?? t.node} ${t.class_type}`, gb(t.ram), gb(t.cache)],
		})), focusNode));
	}
	return el("div", {}, parts);
}

function censusTable(c) {
	return table(["Count", "Shape", "dtype", "Device", "Memory", "Each", "Distinct bytes"], c.groups.map((g) => ({
		cells: [`${g.count} ×`, `(${g.shape.join(", ")})`, g.dtype, g.device, BACKING[g.backing] ?? "–", gb(g.bytes_each), gb(g.bytes)],
	})));
}

function curveSvg(curve, limit) {
	if (!curve?.length) return null;
	const w = 600, h = 120;
	const tMax = curve[curve.length - 1][0] || 1;
	const yMax = Math.max(limit ?? 0, ...curve.map((p) => p[1] ?? 0)) || 1;
	const pts = curve.map(([t, v]) => `${((t / tMax) * w).toFixed(1)},${(h - ((v ?? 0) / yMax) * h).toFixed(1)}`).join(" ");
	const ns = "http://www.w3.org/2000/svg";
	const svg = document.createElementNS(ns, "svg");
	svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
	svg.setAttribute("style", "width:100%;max-width:600px;height:120px;background:var(--comfy-input-bg,#333)");
	const line = document.createElementNS(ns, "polyline");
	line.setAttribute("points", pts);
	line.setAttribute("style", "fill:none;stroke:#3b82f6;stroke-width:2");
	svg.append(line);
	if (limit) {
		const y = (h - (limit / yMax) * h).toFixed(1);
		const lim = document.createElementNS(ns, "line");
		for (const [k, v] of Object.entries({ x1: 0, x2: w, y1: y, y2: y, style: "stroke:#ef4444;stroke-dasharray:4" })) lim.setAttribute(k, v);
		svg.append(lim);
	}
	return svg;
}

async function crashTab() {
	const { report: r } = await call("crash");
	if (!r) return el("div", { class: "bcpm-note" }, "The last run ended normally (or no run was logged).");
	const n = r.node_start;
	const parts = [
		el("div", { class: r.cause.kind === "oom_kill" ? "bcpm-bad" : "bcpm-warn" }, r.cause.text),
		el("div", {}, `Prompt ${r.prompt_id}, died ${secs(r.ran_s)} after its start.`),
		el("div", {}, `Node ${r.node ?? "–"} (${r.class_type ?? "?"}), line: ${r.line ?? "–"}`),
		r.top ? el("div", { class: "bcpm-note" }, `innermost frame: ${r.top}`) : null,
		el("div", {}, `RAM ${gb(r.ram_last)} of ${gb(r.ram_limit)} at the last sample; at the node's start ${gb(n?.ram)}; `,
			`output cache ${gb(r.cache)}; growth ${r.growth_per_s == null ? "–" : `${gb(r.growth_per_s)}/s`} over the last 3 s`),
		curveSvg(r.curve, r.ram_limit),
		r.swap ? el("div", { class: "bcpm-warn" }, r.swap.text) : null,
	];
	if (n?.inputs?.length) {
		parts.push(el("h4", {}, "Node inputs"));
		parts.push(table(["Input", "Bytes", "Tensors"], n.inputs.map((i) => ({
			cells: [i.name, gb(i.bytes), i.tensors.map((t) => `[${t.shape}] ${t.dtype} ${t.device}`).join(", ")],
		}))));
	}
	const snap = r.snapshot;
	if (snap) {
		const c = snap.census;
		parts.push(el("h4", {}, `Tensors alive at the threshold (${gb(snap.ram)}, node ${snap.node} ${snap.class_type ?? ""}; ${snap.scope ?? "whole process"})`));
		// no census: the run died while it was taken; the stack record was written before it
		if (!c) parts.push(el("div", { class: "bcpm-note" }, "The run ended before the tensor census finished; the stack below was written at the threshold."));
		// file_bytes: absent in a run logged before the census told file-backed memory apart
		if (c?.file_bytes !== undefined) parts.push(el("div", { class: "bcpm-note" }, `${gb(c.total_bytes)} of memory in all, each byte once; `,
			c.file_bytes === null ? "file-backed part unknown (no /proc/self/maps on this system)"
				: `${gb(c.file_bytes)} of it file-backed (mapped from files: page cache, not the process's own RAM)`));
		if (c) parts.push(censusTable(c));
		parts.push(el("details", {}, el("summary", {}, "Stack of the execution thread"), el("pre", {}, snap.stack.join(""))));
	} else {
		parts.push(el("div", { class: "bcpm-note" }, "No snapshot: RAM never crossed the threshold before the end."));
	}
	return el("div", {}, parts);
}

function settingsTab() {
	const s = state.status?.settings ?? {};
	const blackBox = el("input", { type: "checkbox" });
	blackBox.checked = !!s.black_box;
	const threshold = el("input", { type: "number", min: "0.5", max: "0.99", step: "0.01", value: s.threshold ?? 0.85 });
	const stop = el("input", { type: "checkbox" });
	stop.checked = !!s.stop_at_threshold;
	const keep = el("input", { type: "number", min: "1", max: "500", step: "1", value: s.keep_runs ?? 20 });
	const msg = el("div", {});
	const save = el("button", { class: "bcpm-btn" }, "Save");
	save.onclick = async () => {
		try {
			state.status = await call("settings", { black_box: blackBox.checked, threshold: Number(threshold.value),
				stop_at_threshold: stop.checked, keep_runs: Number(keep.value) });
			msg.replaceChildren(el("span", { class: "bcpm-ok" }, "Saved."));
		} catch (e) {
			msg.replaceChildren(el("span", { class: "bcpm-bad" }, String(e.message ?? e)));
		}
	};
	const row = (label, input, note) => el("div", { style: "margin:8px 0" }, el("label", {}, input, " ", label),
		note ? el("div", { class: "bcpm-note" }, note) : null);
	return el("div", {},
		row("Black box: a log line every 100 ms during a run", blackBox, "Needed for the crash report. Run logs live in user/BCNodes/process_monitor/runs."),
		row("Threshold (fraction of the RAM limit) for the tensor snapshot", threshold),
		row("Experimental: stop the prompt at the threshold", stop,
			"Works only inside nodes that check ComfyUI's interrupt (e.g. between sampler steps); a running torch.stack or np.fromiter cannot be stopped."),
		row("Run logs kept", keep), save, msg);
}

const RENDER = { "Live": liveTab, "Emulate": emulateTab, "Last run": lastRunTab, "Crash": crashTab, "Settings": settingsTab };

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

async function refreshStatus() {
	try {
		state.status = await call("status");
		const live = await call("live");
		state.sample = live.sample;
	} catch (e) {
		console.warn("[BCNodes] Process Monitor status failed:", e);
	}
	renderBar();
}

async function setEnabled(enabled) {
	try {
		state.status = await call("enable", { enabled: !!enabled });
	} catch (e) {
		console.warn("[BCNodes] Process Monitor could not be switched:", e);
	}
	if (!enabled) state.sample = null;
	renderBar();
}

app.registerExtension({
	name: "BCNodes.ProcessMonitor",
	settings: [{
		id: SETTING,
		category: ["BCNodes", "Process Monitor", "Enabled"],
		name: "Process Monitor (RAM / VRAM bars, black box, per-node measurement)",
		tooltip: "Off: no thread, no hook, no file writes. On: live bars, a run log for the crash report, and per-node measurement when armed from the monitor's modal.",
		type: "boolean",
		defaultValue: true,
		onChange: (value) => setEnabled(value),
	}],

	setup() {
		buildBar();
		api.addEventListener(EVENT, (e) => {
			state.sample = e.detail;
			renderBar();
			if (state.modal && state.tab === "Live") state.modal.body.replaceChildren(liveTab());
		});
		// A finished run clears the crash flag; the monitor notices the end within 100 ms. A reconnect
		// can be a server that was restarted after a killed run: its crash turns the button red.
		const later = () => setTimeout(refreshStatus, 500);
		for (const ev of ["execution_success", "execution_error", "execution_interrupted", "reconnected"]) api.addEventListener(ev, later);
		refreshStatus();
	},
});

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// Auto Model Downloader — the frontend half of nodes/downloader.py.
//
// On the node: two token boxes (Hugging Face, Civitai), one line per model
// (URL + directory under models/ + "token required"), an "Add line" /
// "Remove last line" pair, a download button whose label follows the state of
// the files on disk, and a status row. The lines are UI only; their content
// lives in the hidden `entries` widget as JSON, which is what gets saved with
// the workflow and sent to the backend. Tokens are saved on the server and
// never written back into the UI or the workflow.
//
// On first open of a workflow whose models are missing, one modal asks to
// download them (progress bars inside). The answer is remembered server side.
// Downloads started from the node button report to the console and to the
// node's status row instead, and end with a small "done" modal.

const NODE_TYPE = "BC_AutoModelDownloader";
const ENTRIES = "entries";
const EVENT = "bcnodes.downloader";
const LINE_RE = /^(model|dir|hf|civitai)_(\d+)$/;
const SERVICES = [
	{ id: "huggingface", name: "Hugging Face", widget: "hf_token", label: "HF token" },
	{ id: "civitai", name: "Civitai", widget: "civitai_token", label: "Civitai token" },
];

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

async function call(route, body) {
	const res = await api.fetchApi(`/bcnodes/downloader/${route}`, {
		method: body === undefined ? "GET" : "POST",
		headers: { "Content-Type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
	});
	const data = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(data.error ?? `${res.status} ${res.statusText}`);
	return data;
}

// ---------------------------------------------------------------------------
// Graph helpers
// ---------------------------------------------------------------------------

function allGraphs() {
	const root = app.rootGraph ?? app.graph?.rootGraph ?? app.graph;
	if (!root) return [];
	const graphs = [root];
	if (root.subgraphs?.values) for (const sg of root.subgraphs.values()) graphs.push(sg);
	return graphs;
}

function downloaderNodes() {
	const nodes = [];
	for (const g of allGraphs()) {
		for (const n of g.nodes ?? g._nodes ?? []) if (n.type === NODE_TYPE) nodes.push(n);
	}
	return nodes;
}

function nodeEntries(node) {
	const w = node.widgets?.find((w) => w.name === ENTRIES);
	try {
		const data = JSON.parse(w?.value || "[]");
		return Array.isArray(data) ? data.filter((e) => e && (e.url?.trim() || e.dir?.trim())) : [];
	} catch {
		return [];
	}
}

function mergeEntries(nodes) {
	const seen = new Set();
	const out = [];
	for (const n of nodes) {
		for (const e of nodeEntries(n)) {
			const key = `${e.url?.trim()}\n${e.dir?.trim()}`;
			if (seen.has(key)) continue;
			seen.add(key);
			out.push({ url: e.url?.trim() ?? "", dir: e.dir?.trim() ?? "", hf: !!e.hf, civitai: !!e.civitai });
		}
	}
	return out;
}

// ---------------------------------------------------------------------------
// Node widgets
// ---------------------------------------------------------------------------

function uiOnly(widget) {
	widget.serialize = false; // not in widgets_values
	widget.options = widget.options ?? {};
	widget.options.serialize = false; // not in the prompt
	return widget;
}

// A display-only text row. `disabled` would hide the value in current
// frontends, so the click is swallowed instead.
function readOnly(widget) {
	widget.onClick = () => {};
	widget.mouse = () => true;
	return widget;
}

function hideWidget(widget) {
	widget.hidden = true;
	widget.computeSize = () => [0, -4];
}

function lineWidgets(node) {
	const lines = new Map();
	for (const w of node.widgets ?? []) {
		const m = LINE_RE.exec(w.name ?? "");
		if (!m) continue;
		const line = lines.get(m[2]) ?? {};
		line[m[1]] = w;
		lines.set(m[2], line);
	}
	return [...lines.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).map(([, l]) => l);
}

function syncEntries(node) {
	const entries = lineWidgets(node)
		.map((l) => ({ url: (l.model?.value ?? "").trim(), dir: (l.dir?.value ?? "").trim(), hf: !!l.hf?.value, civitai: !!l.civitai?.value }))
		.filter((e) => e.url || e.dir);
	node.bc.entries.value = JSON.stringify(entries);
	scheduleRefresh(node);
}

function addLine(node, model = "", dir = "", hf = false, civitai = false) {
	const n = lineWidgets(node).length + 1;
	const modelW = uiOnly(node.addWidget("text", `model_${n}`, model, () => syncEntries(node), {}));
	const dirW = uiOnly(node.addWidget("text", `dir_${n}`, dir, () => syncEntries(node), {}));
	const hfW = uiOnly(node.addWidget("toggle", `hf_${n}`, !!hf, () => syncEntries(node), { on: "yes", off: "no" }));
	const civW = uiOnly(node.addWidget("toggle", `civitai_${n}`, !!civitai, () => syncEntries(node), { on: "yes", off: "no" }));
	modelW.options.placeholder = "https://huggingface.co/owner/repo/resolve/main/model.safetensors";
	dirW.options.placeholder = "diffusion_models";
	hfW.label = "HF token needed?";
	civW.label = "Civitai token needed?";
	reorder(node);
}

function removeLastLine(node) {
	const lines = lineWidgets(node);
	if (lines.length <= 1) return;
	const last = lines[lines.length - 1];
	replaceWidgets(node, node.widgets.filter((w) => w !== last.model && w !== last.dir && w !== last.hf && w !== last.civitai));
	reorder(node);
	syncEntries(node);
}

function clearLines(node) {
	replaceWidgets(node, node.widgets.filter((w) => !LINE_RE.test(w.name ?? "")));
}

// In place: the frontend may hold a reference to the widgets array.
function replaceWidgets(node, widgets) {
	node.widgets.splice(0, node.widgets.length, ...widgets);
}

// entries (hidden), lines..., then the controls — and the size follows.
function reorder(node) {
	const { entries, tokens, controls } = node.bc;
	const lines = lineWidgets(node).flatMap((l) => [l.model, l.dir, l.hf, l.civitai]);
	replaceWidgets(node, [entries, ...tokens, ...lines, ...controls]);
	const width = node.size[0];
	const computed = node.computeSize();
	node.setSize([Math.max(width, computed[0]), computed[1]]);
	node.setDirtyCanvas?.(true, true);
}

function setStatus(node, text) {
	node.bc.status.value = text;
	node.setDirtyCanvas?.(true, false);
}

function setButton(node, label, disabled) {
	node.bc.button.label = label;
	node.bc.button.disabled = disabled;
	node.setDirtyCanvas?.(true, false);
}

// ---------------------------------------------------------------------------
// Tokens — typed on the node or in the dialog, stored on the server; the box
// is emptied again after saving so the secret never sits in the canvas.
// ---------------------------------------------------------------------------

let tokensPresent = { huggingface: false, civitai: false };

function applyTokenLabels(node) {
	for (const [i, s] of SERVICES.entries()) {
		const w = node.bc.tokens[i];
		w.label = tokensPresent[s.id] ? `${s.label} (saved)` : s.label;
	}
	node.setDirtyCanvas?.(true, false);
}

async function saveTokens(update) {
	const res = await call("tokens", update);
	tokensPresent = res.tokens ?? tokensPresent;
	for (const n of downloaderNodes()) {
		applyTokenLabels(n);
		scheduleRefresh(n);
	}
}

function addTokenWidget(node, service) {
	const w = uiOnly(node.addWidget("text", service.widget, "", (value) => {
		const text = (value ?? "").trim();
		setTimeout(() => {
			w.value = "";
			node.setDirtyCanvas?.(true, false);
		}, 0);
		if (!text) return;
		saveTokens({ [service.id]: text }).catch((e) => setStatus(node, `token not saved: ${e.message}`));
	}, {}));
	w.options.placeholder = "paste and press enter — kept on the server";
	w.label = service.label;
	return w;
}

// ---------------------------------------------------------------------------
// State from the backend
// ---------------------------------------------------------------------------

function scheduleRefresh(node) {
	clearTimeout(node.bc.refreshTimer);
	node.bc.refreshTimer = setTimeout(() => refreshStatus(node), 500);
}

async function refreshStatus(node) {
	const entries = nodeEntries(node);
	if (!entries.length) {
		setButton(node, "Download all models", true);
		setStatus(node, "no models listed");
		return null;
	}
	let info;
	try {
		info = await call("check", { entries });
	} catch (e) {
		setButton(node, "Download all models", false);
		setStatus(node, `check failed: ${e.message}`);
		return null;
	}
	node.bc.info = info;
	if (info.tokens) {
		tokensPresent = info.tokens;
		applyTokenLabels(node);
	}
	const bad = info.items.filter((i) => i.error);
	const missing = info.items.filter((i) => !i.error && !i.exists);
	const needsToken = tokenNeeds(info);
	if (bad.length) {
		setStatus(node, bad[0].error);
	} else if (needsToken) {
		setStatus(node, needsToken);
	} else if (missing.length) {
		setStatus(node, `missing: ${missing.map((i) => i.filename).join(", ")}`);
	} else {
		setStatus(node, `${info.items.length} model(s) present`);
	}
	if (info.job?.running) {
		setButton(node, "Downloading...", true);
	} else if (!missing.length && !bad.length) {
		setButton(node, "All models downloaded", true);
	} else if (missing.length === info.items.length - bad.length) {
		setButton(node, "Download all models", !!needsToken);
	} else {
		setButton(node, `Download missing models (${missing.length})`, !!needsToken);
	}
	return info;
}

function refreshAll() {
	for (const n of downloaderNodes()) refreshStatus(n);
}

// "needs Hugging Face token: a, b; needs Civitai token: c" or "".
function tokenNeeds(info) {
	const needs = info.missing_tokens ?? {};
	return SERVICES.filter((s) => needs[s.id]?.length)
		.map((s) => `needs ${s.name} token: ${needs[s.id].join(", ")}`)
		.join("; ");
}

async function downloadFromNode(node) {
	const entries = nodeEntries(node);
	if (!entries.length) return;
	let res;
	try {
		res = await call("start", { entries });
	} catch (e) {
		setStatus(node, `start failed: ${e.message}`);
		return;
	}
	if (!res.started) {
		if (res.job?.running) {
			setStatus(node, "a download is already running");
			setButton(node, "Downloading...", true);
		} else {
			refreshStatus(node);
		}
		return;
	}
	consoleJobs.add(res.job.id);
	console.log(`[BCNodes] downloading ${res.job.count} file(s): ${res.job.files.join(", ")}`);
	for (const n of downloaderNodes()) setButton(n, "Downloading...", true);
}

// ---------------------------------------------------------------------------
// Progress events
// ---------------------------------------------------------------------------

const consoleJobs = new Set(); // jobs started from a node button: log, no modal
const lastLogged = new Map(); // filename -> last logged 5% step
let activeModal = null;

function fmtMB(bytes) {
	return `${(bytes / 1e6).toFixed(0)} MB`;
}

function onEvent(ev) {
	const d = ev.detail ?? {};
	if (activeModal?.job === d.job) activeModal.update(d);

	if (d.event === "progress") {
		const pct = d.total ? Math.floor((100 * d.downloaded) / d.total) : 0;
		const text = d.total ? `${d.filename} ${pct}% (${fmtMB(d.downloaded)} / ${fmtMB(d.total)})` : `${d.filename} ${fmtMB(d.downloaded)}`;
		for (const n of downloaderNodes()) setStatus(n, `downloading ${d.index + 1}/${d.count}: ${text}`);
		const step = Math.floor(pct / 5);
		if (consoleJobs.has(d.job) && lastLogged.get(d.filename) !== step) {
			lastLogged.set(d.filename, step);
			console.log(`[BCNodes] ${d.index + 1}/${d.count} ${text}`);
		}
	} else if (d.event === "file_done") {
		if (consoleJobs.has(d.job)) console.log(`[BCNodes] done: ${d.filename}`);
	} else if (d.event === "file_error") {
		console.error(`[BCNodes] failed: ${d.filename}: ${d.error}`);
	} else if (d.event === "done") {
		lastLogged.clear();
		if (consoleJobs.has(d.job)) {
			consoleJobs.delete(d.job);
			const errors = d.errors ?? [];
			console.log(`[BCNodes] finished: ${d.count - errors.length}/${d.count} downloaded`);
			showDoneModal(d.count, errors);
		}
		refreshAll();
	}
}

// ---------------------------------------------------------------------------
// Modals — plain DOM so they work in every frontend version.
// ---------------------------------------------------------------------------

const STYLE = `
.bcnodes-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:system-ui,sans-serif}
.bcnodes-modal{background:#1e1e1e;color:#e6e6e6;border:1px solid #3a3a3a;border-radius:10px;min-width:460px;max-width:720px;max-height:80vh;overflow:auto;padding:18px 20px;box-shadow:0 12px 40px rgba(0,0,0,.6)}
.bcnodes-modal h3{margin:0 0 6px;font-size:16px}
.bcnodes-modal p{margin:0 0 12px;color:#bdbdbd;font-size:13px}
.bcnodes-row{padding:8px 0;border-top:1px solid #2c2c2c;font-size:13px}
.bcnodes-row .name{font-weight:600}
.bcnodes-row .dir{color:#9a9a9a;margin-left:6px}
.bcnodes-row .pct{float:right;color:#9a9a9a}
.bcnodes-bar{height:6px;background:#2c2c2c;border-radius:3px;margin-top:6px;overflow:hidden}
.bcnodes-bar>div{height:100%;width:0;background:#4a90e2;transition:width .2s}
.bcnodes-row.done .bcnodes-bar>div{background:#3cb371;width:100%}
.bcnodes-row.error .bcnodes-bar>div{background:#d9534f;width:100%}
.bcnodes-row .err{color:#e08585;font-size:12px;margin-top:4px;white-space:pre-wrap}
.bcnodes-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:14px}
.bcnodes-actions button{padding:7px 14px;border-radius:6px;border:1px solid #3a3a3a;background:#2a2a2a;color:#e6e6e6;cursor:pointer;font-size:13px}
.bcnodes-actions button.primary{background:#4a90e2;border-color:#4a90e2;color:#fff}
.bcnodes-actions button:disabled{opacity:.5;cursor:default}
.bcnodes-tokens{margin:0 0 12px;padding:10px 12px;background:#242424;border:1px solid #333;border-radius:8px}
.bcnodes-tokens label{display:block;font-size:12px;color:#bdbdbd;margin:6px 0 3px}
.bcnodes-tokens label b{color:#e6e6e6}
.bcnodes-tokens .need{color:#e0b060}
.bcnodes-tokens input{width:100%;box-sizing:border-box;padding:6px 8px;border-radius:5px;border:1px solid #3a3a3a;background:#1a1a1a;color:#e6e6e6;font-size:13px}
.bcnodes-row .tag{margin-left:6px;font-size:11px;color:#e0b060}
`;

function ensureStyle() {
	if (document.getElementById("bcnodes-downloader-style")) return;
	const style = document.createElement("style");
	style.id = "bcnodes-downloader-style";
	style.textContent = STYLE;
	document.head.appendChild(style);
}

function el(tag, cls, text) {
	const e = document.createElement(tag);
	if (cls) e.className = cls;
	if (text !== undefined) e.textContent = text;
	return e;
}

function openModal(title, text) {
	ensureStyle();
	const backdrop = el("div", "bcnodes-backdrop");
	const modal = el("div", "bcnodes-modal");
	modal.appendChild(el("h3", null, title));
	if (text) modal.appendChild(el("p", null, text));
	const list = el("div");
	const actions = el("div", "bcnodes-actions");
	modal.appendChild(list);
	modal.appendChild(actions);
	backdrop.appendChild(modal);
	document.body.appendChild(backdrop);
	return { backdrop, list, actions, close: () => backdrop.remove() };
}

function addButton(actions, label, primary, onClick) {
	const b = el("button", primary ? "primary" : null, label);
	b.onclick = onClick;
	actions.appendChild(b);
	return b;
}

function showDoneModal(count, errors) {
	const ok = count - errors.length;
	const m = openModal(errors.length ? "Download finished with errors" : "All models downloaded", `${ok}/${count} file(s) downloaded.`);
	for (const e of errors) {
		const row = el("div", "bcnodes-row error");
		row.appendChild(el("span", "name", e.filename));
		row.appendChild(el("div", "err", e.error));
		m.list.appendChild(row);
	}
	addButton(m.actions, "Close", true, m.close);
}

// The first-open prompt: lists what is missing, downloads with progress bars.
function showAskModal(entries, info) {
	const missing = info.items.filter((i) => !i.error && !i.exists);
	const m = openModal("Auto Model Downloader", "This workflow needs models that are not installed yet:");

	// Token boxes, shown when any missing file is marked as needing one.
	const needs = info.missing_tokens ?? {};
	const inputs = {};
	if (missing.some((i) => i.hf || i.civitai)) {
		const box = el("div", "bcnodes-tokens");
		for (const s of SERVICES) {
			const flag = s.id === "huggingface" ? "hf" : "civitai";
			const files = missing.filter((i) => i[flag]).map((i) => i.filename);
			const label = el("label");
			label.appendChild(el("b", null, `${s.name} token`));
			if (files.length) label.appendChild(el("span", "need", ` — required for ${files.join(", ")}`));
			if (info.tokens?.[s.id]) label.appendChild(el("span", null, " (saved; leave empty to keep)"));
			box.appendChild(label);
			const input = el("input");
			input.type = "password";
			input.placeholder = info.tokens?.[s.id] ? "saved" : `paste your ${s.name} token`;
			box.appendChild(input);
			inputs[s.id] = input;
		}
		m.list.appendChild(box);
	}

	const rows = new Map();
	for (const item of missing) {
		const row = el("div", "bcnodes-row");
		row.appendChild(el("span", "name", item.filename));
		row.appendChild(el("span", "dir", `→ models/${item.dir}`));
		const needs = [item.hf ? "Hugging Face" : null, item.civitai ? "Civitai" : null].filter(Boolean);
		if (needs.length) row.appendChild(el("span", "tag", `${needs.join(" + ")} token required`));
		row.appendChild(el("span", "pct", ""));
		const bar = el("div", "bcnodes-bar");
		bar.appendChild(el("div"));
		row.appendChild(bar);
		m.list.appendChild(row);
		rows.set(item.filename, row);
	}
	const bad = info.items.filter((i) => i.error);
	for (const item of bad) {
		const row = el("div", "bcnodes-row error");
		row.appendChild(el("span", "name", item.url || "(empty)"));
		row.appendChild(el("div", "err", item.error));
		m.list.appendChild(row);
	}

	const later = addButton(m.actions, "Not now", false, async () => {
		m.close();
		try {
			await call("dismiss", { key: info.key });
		} catch (e) {
			console.warn("[BCNodes] could not remember the answer:", e);
		}
	});
	let errorRow = null;
	const showError = (text) => {
		errorRow?.remove();
		errorRow = el("div", "bcnodes-row error", text);
		m.list.appendChild(errorRow);
	};
	const go = addButton(m.actions, "Download", true, async () => {
		go.disabled = true;
		later.disabled = true;
		const typed = {};
		for (const [id, input] of Object.entries(inputs)) if (input.value.trim()) typed[id] = input.value.trim();
		try {
			if (Object.keys(typed).length) await saveTokens(typed);
		} catch (e) {
			go.disabled = false;
			later.disabled = false;
			showError(`token not saved: ${e.message}`);
			return;
		}
		for (const input of Object.values(inputs)) input.value = "";
		let res;
		try {
			res = await call("start", { entries });
		} catch (e) {
			go.disabled = false;
			later.disabled = false;
			showError(e.message);
			return;
		}
		errorRow?.remove();
		if (!res.started && !res.job?.running) {
			m.close();
			refreshAll();
			return;
		}
		for (const n of downloaderNodes()) setButton(n, "Downloading...", true);
		activeModal = {
			job: res.job.id,
			update(d) {
				const row = rows.get(d.filename);
				if (row && d.event === "progress") {
					const pct = d.total ? Math.floor((100 * d.downloaded) / d.total) : 0;
					row.querySelector(".bcnodes-bar>div").style.width = `${pct}%`;
					row.querySelector(".pct").textContent = d.total ? `${pct}%  ${fmtMB(d.downloaded)} / ${fmtMB(d.total)}` : fmtMB(d.downloaded);
				} else if (row && d.event === "file_done") {
					row.classList.add("done");
					row.querySelector(".pct").textContent = "done";
				} else if (row && d.event === "file_error") {
					row.classList.add("error");
					row.querySelector(".pct").textContent = "failed";
					row.appendChild(el("div", "err", d.error));
				} else if (d.event === "done") {
					activeModal = null;
					later.remove();
					go.remove();
					addButton(m.actions, d.errors?.length ? "Close" : "Done", true, m.close);
				}
			},
		};
	});
}

async function askOnFirstOpen() {
	const nodes = downloaderNodes();
	if (!nodes.length) return;
	const entries = mergeEntries(nodes);
	if (!entries.length) return;
	let info;
	try {
		info = await call("check", { entries });
	} catch (e) {
		console.warn("[BCNodes] downloader check failed:", e);
		return;
	}
	if (!info.missing || info.seen || info.job?.running || activeModal) return;
	showAskModal(entries, info);
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

app.registerExtension({
	name: "BCNodes.AutoModelDownloader",

	setup() {
		api.addEventListener(EVENT, onEvent);
		call("tokens")
			.then((res) => {
				tokensPresent = res.tokens ?? tokensPresent;
				for (const n of downloaderNodes()) applyTokenLabels(n);
			})
			.catch(() => {});
	},

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function (...args) {
			const r = onNodeCreated?.apply(this, args);
			try {
				const entries = this.widgets.find((w) => w.name === ENTRIES);
				hideWidget(entries);
			// The only socket is the hidden entries widget's, so no slot row is
			// drawn; without this computeSize would still reserve one and leave
			// a blank strip under the last widget.
			this.widgets_start_y = 6;

				const tokens = SERVICES.map((s) => addTokenWidget(this, s));

				const add = uiOnly(this.addWidget("button", "add_line", null, () => addLine(this)));
				add.label = "Add line";
				const remove = uiOnly(this.addWidget("button", "remove_line", null, () => removeLastLine(this)));
				remove.label = "Remove last line";
				const button = uiOnly(this.addWidget("button", "download", null, () => downloadFromNode(this)));
				button.label = "Download all models";
				const status = readOnly(uiOnly(this.addWidget("text", "status", "", () => {}, {})));

				this.bc = { entries, tokens, controls: [add, remove, button, status], button, status, info: null, refreshTimer: null };
				addLine(this);
				applyTokenLabels(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};

		// The saved workflow only carries `entries`; rebuild the lines from it.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (...args) {
			const r = onConfigure?.apply(this, args);
			try {
				if (!this.bc) return r;
				clearLines(this);
				const entries = nodeEntries(this);
				if (entries.length) for (const e of entries) addLine(this, e.url, e.dir, e.hf, e.civitai);
				else addLine(this);
				scheduleRefresh(this);
			} catch (e) {
				console.error("[BCNodes]", e);
			}
			return r;
		};
	},

	afterConfigureGraph() {
		setTimeout(askOnFirstOpen, 300);
	},
});

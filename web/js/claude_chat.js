import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { callJson } from "./bcnodes_api.js";

// Claude Chat: the Claude tab of ComfyUI's sidebar. A chat with Claude through the Messages API and the key set in
// Settings -> BCNodes -> Claude; it touches no node and no workflow. Sessions live on the server
// (pipelines/claude/chat.py, the routes in nodes/claude.py): pick one, start one, delete one; each has its model,
// effort, max_tokens and system prompt. Enter sends (Shift+Enter: a new line); images are pasted, dropped or picked.
// The answer streams in (NDJSON); Stop ends the request, and a turn that did not end is not saved: its text and
// images come back to the input. Answers render as Markdown through the frontend's own renderer (sanitized).

const BASE = "/bcnodes/claude/chats";
const state = { sessions: [], options: null, current: null, attachments: [], controller: null };
let ui = null;

const css = `
.bc-chat{display:flex;flex-direction:column;height:100%;gap:6px;padding:8px;box-sizing:border-box;font-size:13px;color:var(--fg-color,#ddd)}
.bc-chat select,.bc-chat input,.bc-chat textarea{background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#444);border-radius:4px;padding:4px 6px;font:inherit}
.bc-chat button{background:var(--comfy-input-bg,#333);color:var(--input-text,#ddd);border:1px solid var(--border-color,#555);border-radius:4px;padding:4px 10px;cursor:pointer}
.bc-chat button:disabled{opacity:.5;cursor:default}
.bc-chat .row{display:flex;gap:6px;align-items:center}
.bc-chat .row select{flex:1;min-width:0}
.bc-chat details.settings{border:1px solid var(--border-color,#444);border-radius:4px;padding:4px 6px}
.bc-chat details.settings label{display:block;margin:4px 0 2px;opacity:.75;font-size:12px}
.bc-chat details.settings textarea{width:100%;box-sizing:border-box;min-height:60px;resize:vertical}
.bc-chat .messages{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:2px}
.bc-chat .msg{border-radius:6px;padding:6px 8px;overflow-wrap:anywhere}
.bc-chat .msg.user{background:var(--comfy-input-bg,#2a2a2a);white-space:pre-wrap;align-self:flex-end;max-width:92%}
.bc-chat .msg.assistant{border:1px solid var(--border-color,#3a3a3a)}
.bc-chat .msg.assistant pre{overflow-x:auto;background:rgba(0,0,0,.25);padding:6px;border-radius:4px}
.bc-chat .msg.assistant p{margin:.3em 0}
.bc-chat .msg details{opacity:.75;margin-bottom:4px}
.bc-chat .msg details div{white-space:pre-wrap;font-size:12px}
.bc-chat .meta{font-size:11px;opacity:.6;margin-top:4px}
.bc-chat .error{color:#e57373;white-space:pre-wrap}
.bc-chat .thumbs{display:flex;gap:4px;flex-wrap:wrap}
.bc-chat .thumbs img{height:56px;border-radius:4px;border:1px solid var(--border-color,#444)}
.bc-chat .thumbs span{position:relative}
.bc-chat .thumbs b{position:absolute;top:-4px;right:-4px;background:#000;border-radius:8px;padding:0 5px;cursor:pointer;font-size:11px}
.bc-chat textarea.input{width:100%;box-sizing:border-box;min-height:64px;resize:vertical}
`;

function el(tag, cls, text) {
	const e = document.createElement(tag);
	if (cls) e.className = cls;
	if (text !== undefined) e.textContent = text;
	return e;
}

function markdown(text) {
	const div = el("div");
	const render = app.extensionManager?.renderMarkdownToHtml;
	if (render) div.innerHTML = render(text);
	else div.textContent = text;
	return div;
}

function select(values, value, onChange) {
	const s = el("select");
	for (const v of values) s.appendChild(Object.assign(el("option", null, String(v)), { value: String(v) }));
	s.value = String(value);
	s.onchange = () => onChange(s.value);
	return s;
}

// --- server calls ------------------------------------------------------------------------------------------

async function loadList() {
	const res = await callJson(BASE);
	state.sessions = res.sessions;
	state.options = res;
}

async function open(id) {
	state.current = id ? await callJson(`${BASE}/${id}`) : null;
	render();
}

async function change(key, value) {
	if (!state.current) return; // applied when the first message creates the session
	state.current = await callJson(`${BASE}/${state.current.id}`, { [key]: value });
	await loadList();
	renderHeader();
}

// --- rendering ---------------------------------------------------------------------------------------------

function sessionTitle(s) {
	return s.title || `New chat (${new Date(s.updated * 1000).toLocaleString()})`;
}

function renderHeader() {
	const sessions = [{ id: "", title: "— new chat —", updated: 0 }, ...state.sessions];
	const pick = select(sessions.map((s) => s.id), state.current?.id ?? "", (id) => open(id).catch(showError));
	[...pick.options].forEach((o, i) => (o.textContent = i ? sessionTitle(sessions[i]) : sessions[0].title));
	const del = el("button", null, "Delete");
	del.disabled = !state.current;
	del.onclick = async () => {
		if (del.dataset.armed !== "1") {
			del.dataset.armed = "1";
			del.textContent = "Sure?";
			setTimeout(() => ((del.dataset.armed = ""), (del.textContent = "Delete")), 3000);
			return;
		}
		await callJson(`${BASE}/${state.current.id}/delete`, {});
		await loadList();
		await open(null);
	};
	ui.header.replaceChildren(pick, del);
	renderSettings();
}

function renderSettings() {
	const o = state.options;
	const s = state.current ?? ui.draft;
	const box = ui.settings;
	box.replaceChildren(el("summary", null, `Settings — ${s.model}, effort ${s.effort}`));
	const field = (label, control) => box.append(el("label", null, label), control);
	const set = (key) => (value) => {
		if (state.current) change(key, value).catch(showError);
		else (ui.draft[key] = value), renderSettings();
	};
	field("Model", select(o.models, s.model, set("model")));
	field("Effort", select(o.efforts, s.effort, set("effort")));
	const max = Object.assign(el("input"), { type: "number", min: 1, max: o.max_tokens, value: s.max_tokens });
	max.onchange = () => set("max_tokens")(Math.max(1, Math.min(o.max_tokens, parseInt(max.value, 10) || 1)));
	field("max_tokens (thinking and answer)", max);
	const system = Object.assign(el("textarea"), { value: s.system, placeholder: "No system prompt" });
	system.onchange = () => set("system")(system.value);
	field("System prompt", system);
}

function userBubble(content) {
	const bubble = el("div", "msg user");
	const images = content.filter((b) => b.type === "image");
	if (images.length) {
		const thumbs = el("div", "thumbs");
		for (const b of images) thumbs.appendChild(Object.assign(el("img"), { src: `data:${b.source.media_type};base64,${b.source.data}` }));
		bubble.appendChild(thumbs);
	}
	const text = content.filter((b) => b.type === "text" && !(images.length > 1 && /^Image \d+:$/.test(b.text)));
	if (text.length) bubble.appendChild(document.createTextNode(text.map((b) => b.text).join("\n")));
	return bubble;
}

function turnMeta(turn) {
	const u = turn.usage;
	const cache = u.cache_read_input_tokens || u.cache_creation_input_tokens
		? ` (cache read ${u.cache_read_input_tokens}, written ${u.cache_creation_input_tokens})` : "";
	const stop = turn.stop_reason === "end_turn" ? "" : ` · stopped: ${turn.stop_reason}`;
	return `${turn.model} · in ${u.input_tokens}${cache} · out ${u.output_tokens}${stop}`;
}

function assistantBubble(content, turn) {
	const bubble = el("div", "msg assistant");
	const thinking = content.filter((b) => b.type === "thinking" && b.thinking).map((b) => b.thinking).join("\n\n");
	if (thinking) {
		const d = el("details");
		d.append(el("summary", null, "Thinking"), el("div", null, thinking));
		bubble.appendChild(d);
	}
	for (const b of content) {
		if (b.type === "fallback") bubble.appendChild(el("div", "meta", `${b.from?.model} declined; ${b.to?.model} answered`));
	}
	bubble.appendChild(markdown(content.filter((b) => b.type === "text").map((b) => b.text).join("")));
	if (turn) bubble.appendChild(el("div", "meta", turnMeta(turn)));
	return bubble;
}

function renderMessages() {
	const list = ui.messages;
	list.replaceChildren();
	const messages = state.current?.messages ?? [];
	messages.forEach((m, i) => {
		list.appendChild(m.role === "user" ? userBubble(m.content) : assistantBubble(m.content, state.current.turns[(i - 1) / 2]));
	});
	if (!messages.length) list.appendChild(el("div", "meta", "Ask Claude anything. Billed to your Anthropic API key per token."));
	list.scrollTop = list.scrollHeight;
}

function renderAttachments() {
	ui.thumbs.replaceChildren(...state.attachments.map((a, i) => {
		const wrap = el("span");
		const x = el("b", null, "×");
		x.onclick = () => (state.attachments.splice(i, 1), renderAttachments());
		wrap.append(Object.assign(el("img"), { src: `data:${a.media_type};base64,${a.data}` }), x);
		return wrap;
	}));
}

function render() {
	renderHeader();
	renderMessages();
	renderAttachments();
	setBusy(!!state.controller);
}

function showError(e) {
	ui?.messages.appendChild(el("div", "msg error", String(e?.message ?? e)));
	ui?.messages.scrollTo(0, ui.messages.scrollHeight);
}

function setBusy(busy) {
	ui.send.textContent = busy ? "Stop" : "Send";
}

// --- attachments -------------------------------------------------------------------------------------------

function addFiles(files) {
	for (const file of files) {
		if (!file.type.startsWith("image/")) continue;
		const reader = new FileReader();
		reader.onload = () => {
			const [head, data] = String(reader.result).split(",", 2);
			state.attachments.push({ media_type: head.slice(5, head.indexOf(";")), data });
			renderAttachments();
		};
		reader.readAsDataURL(file);
	}
}

// --- a turn ------------------------------------------------------------------------------------------------

async function send() {
	if (state.controller) return state.controller.abort();
	const text = ui.input.value;
	const images = state.attachments;
	if (!text.trim() && !images.length) return;
	try {
		if (!state.current) {
			state.current = await callJson(BASE, { ...ui.draft });
			await loadList();
			renderHeader();
		}
	} catch (e) {
		return showError(e);
	}
	ui.input.value = "";
	state.attachments = [];
	renderAttachments();
	ui.messages.querySelector(".meta:only-child")?.remove();
	const userEl = userBubble([...images.map((a) => ({ type: "image", source: a })), { type: "text", text }]);
	ui.messages.appendChild(userEl);
	const bubble = el("div", "msg assistant");
	const thinking = el("details");
	const thinkingText = el("div");
	thinking.append(el("summary", null, "Thinking…"), thinkingText);
	const answer = el("div");
	bubble.append(answer);
	ui.messages.appendChild(bubble);
	let answerText = "";
	let pending = false;
	const paint = () => {
		pending = false;
		answer.replaceChildren(markdown(answerText));
		ui.messages.scrollTop = ui.messages.scrollHeight;
	};
	const restore = () => {
		userEl.remove();
		ui.input.value = text;
		state.attachments = images;
		renderAttachments();
	};
	const controller = new AbortController();
	state.controller = controller;
	setBusy(true);
	let outcome = null;
	try {
		const res = await api.fetchApi(`${BASE}/${state.current.id}/send`, {
			method: "POST", headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ text, images }), signal: controller.signal,
		});
		const reader = res.body.getReader();
		const decoder = new TextDecoder();
		let buf = "";
		for (;;) {
			const { value, done } = await reader.read();
			if (done) break;
			buf += decoder.decode(value, { stream: true });
			let nl;
			while ((nl = buf.indexOf("\n")) >= 0) {
				const event = JSON.parse(buf.slice(0, nl));
				buf = buf.slice(nl + 1);
				if (event.type === "text") {
					answerText += event.text;
					if (!pending) (pending = true), requestAnimationFrame(paint);
				} else if (event.type === "thinking") {
					if (!thinking.isConnected) bubble.prepend(thinking);
					thinkingText.textContent += event.text;
				} else if (event.type === "fallback") {
					bubble.appendChild(el("div", "meta", `${event.from} declined; ${event.to} answered`));
				} else if (event.type === "done" || event.type === "error") {
					outcome = event;
				}
			}
		}
		if (!outcome) throw new Error("The connection ended before the answer did; nothing was saved.");
		if (outcome.type === "error") {
			bubble.remove();
			restore();
			showError(outcome.error);
		} else {
			await loadList();
			await open(state.current.id);
		}
	} catch (e) {
		bubble.remove();
		restore();
		if (e.name === "AbortError") showError("Stopped; nothing was saved.");
		else showError(e);
	} finally {
		state.controller = null;
		setBusy(false);
	}
}

// --- mount -------------------------------------------------------------------------------------------------

async function mount(root) {
	if (!document.getElementById("bc-chat-css")) document.head.appendChild(Object.assign(el("style"), { id: "bc-chat-css", textContent: css }));
	const box = el("div", "bc-chat");
	ui = {
		header: el("div", "row"), settings: el("details", "settings"), messages: el("div", "messages"),
		thumbs: el("div", "thumbs"), input: Object.assign(el("textarea", "input"), { placeholder: "Message Claude (Enter to send, Shift+Enter for a new line; paste or drop images)" }),
		send: el("button", null, "Send"), draft: null,
	};
	const picker = Object.assign(el("input"), { type: "file", accept: "image/*", multiple: true, hidden: true });
	picker.onchange = () => (addFiles(picker.files), (picker.value = ""));
	const attach = el("button", null, "Image");
	attach.onclick = () => picker.click();
	ui.send.onclick = () => send();
	ui.input.onkeydown = (e) => {
		if (e.key === "Enter" && !e.shiftKey && !e.isComposing) (e.preventDefault(), send());
	};
	ui.input.onpaste = (e) => {
		const files = [...(e.clipboardData?.items ?? [])].filter((i) => i.kind === "file").map((i) => i.getAsFile());
		if (files.some((f) => f?.type.startsWith("image/"))) (e.preventDefault(), addFiles(files));
	};
	box.ondragover = (e) => e.preventDefault();
	box.ondrop = (e) => (e.preventDefault(), addFiles(e.dataTransfer?.files ?? []));
	const actions = el("div", "row");
	actions.append(attach, picker, el("span", "meta", ""), ui.send);
	actions.children[2].style.flex = "1";
	box.append(ui.header, ui.settings, ui.messages, ui.thumbs, ui.input, actions);
	root.replaceChildren(box);
	try {
		await loadList();
		const o = state.options;
		ui.draft = { model: o.models[0], effort: "high", max_tokens: o.default_max_tokens, system: "" };
		const keep = state.current?.id ?? state.sessions[0]?.id;
		await open(keep && state.sessions.some((s) => s.id === keep) ? keep : null);
	} catch (e) {
		showError(e);
	}
}

app.registerExtension({
	name: "BCNodes.ClaudeChat",
	setup() {
		app.extensionManager?.registerSidebarTab?.({
			id: "bcnodes-claude-chat",
			icon: "pi pi-comments",
			title: "Claude",
			tooltip: "Claude Chat (your Anthropic API key, Settings -> BCNodes -> Claude)",
			type: "custom",
			render: (root) => {
				mount(root);
			},
		});
	},
});

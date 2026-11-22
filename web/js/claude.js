import { app } from "../../../scripts/app.js";
import { callJson } from "./bcnodes_api.js";
import { growInputs } from "./grow_inputs.js";

// Claude (nodes/claude.py): the growing image slots (image_1, image_2, ..., grow_inputs.js) and the Anthropic API
// key in ComfyUI's settings (BCNodes -> Claude). The key goes to the pack's token store
// (libs/download.py, user/BCNodes/downloader_tokens.json) through the downloader's token route, which answers only
// whether a key is stored, never the key. The setting renders its own field and never calls the settings
// setter, so the key is not written into ComfyUI's settings file either.

const TOKENS = "/bcnodes/downloader/tokens";
const SERVICE = "anthropic";

function button(text, onClick) {
	const b = document.createElement("button");
	b.textContent = text;
	b.style.cssText = "padding:4px 10px;border-radius:4px;border:1px solid #555;background:#333;color:#ddd;cursor:pointer";
	b.onclick = onClick;
	return b;
}

function keyField() {
	const box = document.createElement("div");
	box.style.cssText = "display:flex;gap:6px;align-items:center;flex-wrap:wrap";
	const input = document.createElement("input");
	input.type = "password";
	input.autocomplete = "off";
	input.placeholder = "sk-ant-...";
	input.style.cssText = "flex:1;min-width:180px;padding:4px 6px;border-radius:4px;border:1px solid #555;background:#1a1a1a;color:#ddd";
	const status = document.createElement("span");
	status.style.cssText = "font-size:12px;opacity:0.8";
	const show = (tokens) => (status.textContent = tokens?.[SERVICE] ? "saved" : "not set");
	const send = async (value) => {
		try {
			show((await callJson(TOKENS, { [SERVICE]: value })).tokens);
			input.value = "";
		} catch (e) {
			status.textContent = `not saved: ${e.message}`;
		}
	};
	box.append(
		input,
		button("Save", () => input.value.trim() && send(input.value.trim())),
		button("Clear", () => send("")),
		status,
	);
	callJson(TOKENS)
		.then((res) => show(res.tokens))
		.catch((e) => (status.textContent = `unknown: ${e.message}`));
	return box;
}

app.registerExtension({
	name: "BCNodes.Claude",

	settings: [{
		id: "BCNodes.Claude.APIKey",
		category: ["BCNodes", "Claude", "API key"],
		name: "Anthropic API key (Claude node)",
		tooltip: "A Claude Console API key for the Claude node. Kept on the ComfyUI server (user/BCNodes), never in a workflow or the settings file, and never shown again: the field only says whether one is saved. Clear removes it.",
		type: () => keyField(),
		defaultValue: "",
	}],

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== "BC_Claude") return;
		growInputs(nodeType, { prefix: "image_", type: "IMAGE", min: 1 });
	},
});

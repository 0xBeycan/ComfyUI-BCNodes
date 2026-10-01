import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// The toast of the unused-heavy-outputs helper (nodes/common.py): the server sends EVENT when
// another pack's on_prompt handler runs after this pack's, which turns the RAM saving of unused
// outputs off for that run.
const EVENT = "bcnodes.unused_outputs";

app.registerExtension({
	name: "BCNodes.UnusedOutputs",
	setup() {
		api.addEventListener(EVENT, ({ detail }) => {
			const toast = app.extensionManager?.toast;
			if (toast?.add) toast.add({ severity: "warn", summary: "BCNodes", detail: detail?.message, life: 10000 });
			else console.warn(`BCNodes: ${detail?.message}`);
		});
	},
});

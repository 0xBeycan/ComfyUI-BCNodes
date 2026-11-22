import { app } from "../../../scripts/app.js";
import { growInputs } from "./grow_inputs.js";

// Join Image Lists — unbounded inputs. The backend declares In1 and In2; this
// extension keeps exactly one empty IMAGE slot after the last connected one
// (In3 appears once In1 and In2 are both linked, In4 once In3 is linked, ...)
// and keeps the names contiguous when a slot in the middle is disconnected
// (grow_inputs.js).

app.registerExtension({
	name: "BCNodes.JoinImageLists",

	beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData?.name !== "BC_JoinImageLists") return;
		growInputs(nodeType, { prefix: "In", type: "IMAGE", min: 2 });
	},
});

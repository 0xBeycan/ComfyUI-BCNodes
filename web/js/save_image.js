import { app } from "../../../scripts/app.js";

// Save Image — the saved images go to the queue / history gallery through the
// node's ui.images, but are never drawn under the node, so the node keeps the
// size the user gave it.
//
// The frontend draws output images from two places: the classic canvas
// renderer's prototype onDrawBackground (updatePreviews -> image preview
// widget, which grows the node) and the Vue node renderer, which honours
// node.hideOutputImages. Both are switched off per instance here.

const NODE_TYPE = "BC_SaveImage";

app.registerExtension({
	name: "BCNodes.SaveImage",

	nodeCreated(node) {
		if (node.constructor?.comfyClass !== NODE_TYPE) return;
		node.hideOutputImages = true;
		node.onDrawBackground = function () {};
	},
});

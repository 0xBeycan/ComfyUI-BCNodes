import { api } from "../../../scripts/api.js";

// JSON calls to the pack's own server routes, shared by the downloader and the Process Monitor.
// GET without a body, POST with one; a non-2xx answer throws with the server's `error` text.
export async function callJson(path, body) {
	const res = await api.fetchApi(path, {
		method: body === undefined ? "GET" : "POST",
		headers: { "Content-Type": "application/json" },
		body: body === undefined ? undefined : JSON.stringify(body),
	});
	const data = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(data.error ?? `${res.status} ${res.statusText}`);
	return data;
}

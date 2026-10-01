"""Process Monitor: server half (no node). The frontend half is web/js/process_monitor.js.

This module is the ComfyUI surface of pipelines/process_monitor: its HTTP routes under
/bcnodes/monitor/, the `bcnodes.monitor` live event, and the adapters that hand the pipeline
ComfyUI's prompt queue, folders and node classes. It registers only inside a running ComfyUI
server (PromptServer.instance exists); `server`, `aiohttp`, `folder_paths`, PIL and PyAV are
imported inside the functions that use them.

At startup the monitor runs when its saved setting says so (user/BCNodes/process_monitor/
settings.json, written by the ComfyUI setting "BCNodes.ProcessMonitor.Enabled"; on when there is no
file), so a server queued through the API without a browser still gets the black box.
"""

import asyncio
import logging
import os

from ..libs.download import user_file
from ..libs.safetensors_info import weights_info
from ..pipelines.process_monitor.monitor import Monitor

EVENT = "bcnodes.monitor"


class ServerProbe:
    """What the monitor reads from ComfyUI's server: the running prompt and its outcome."""

    def __init__(self, server):
        self.server = server

    def running(self):
        """(prompt_id, extra_data) of the running prompt, or None. The queue item is
        (number, prompt_id, prompt, extra_data, outputs, ...) on every ComfyUI version."""
        items = list(self.server.prompt_queue.currently_running.values())
        return (items[0][1], items[0][3]) if items else None

    def status(self, prompt_id):
        entry = self.server.prompt_queue.history.get(prompt_id) or {}
        status = entry.get("status") or {}
        if any(m[0] == "execution_interrupted" for m in status.get("messages", [])):
            return "interrupted"
        return status.get("status_str") or "unknown"

    def last_node(self):
        return getattr(self.server, "last_node_id", None)

    def push(self, payload):
        self.server.send_sync(EVENT, payload)


class ComfyEnv:
    """What Emulate reads from ComfyUI: node classes, model folders, input files (headers only)."""

    @staticmethod
    def _classes():
        # ComfyUI's node classes, reached through the executor's own reference to its `nodes`
        # module: an absolute `import nodes` here would be ambiguous with the pack's nodes/.
        import execution

        return execution.nodes.NODE_CLASS_MAPPINGS

    def return_types(self, class_type):
        cls = self._classes().get(class_type)
        return [str(t) for t in getattr(cls, "RETURN_TYPES", ())] if cls is not None else []

    def input_types(self, class_type):
        return self._classes()[class_type].INPUT_TYPES()

    def model_path(self, name):
        import folder_paths

        for folder in list(folder_paths.folder_names_and_paths):
            path = folder_paths.get_full_path(folder, name)
            if path is not None:
                return path
        return None

    def weights(self, path):
        return weights_info(path)

    def _input_path(self, name):
        import folder_paths

        if not isinstance(name, str) or not name:
            return None
        path = name if os.path.isabs(name) else folder_paths.get_annotated_filepath(name)
        return path if os.path.isfile(path) else None

    def video_meta(self, name):
        import av

        path = self._input_path(name)
        if path is None:
            return None
        with av.open(path) as container:
            s = container.streams.video[0]
            fps = float(s.average_rate or s.guessed_rate)
            frames = s.frames or int(float(container.duration or 0) / 1e6 * fps)
            return {"width": s.codec_context.width, "height": s.codec_context.height, "frames": frames, "fps": fps}

    def image_size(self, name):
        from PIL import Image

        path = self._input_path(name)
        if path is None:
            return None
        with Image.open(path) as im:
            w, h = im.size
            return (h, w) if im.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)  # EXIF rotated by 90 degrees


def _register():
    try:
        from server import PromptServer
    except ImportError:  # not under a ComfyUI server (tests, the import gate)
        return None
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return None
    from aiohttp import web

    try:
        monitor = Monitor(user_file("process_monitor"), ServerProbe(server))
    except (OSError, ValueError) as e:
        logging.error("[BCNodes] Process Monitor: cannot read %s (%s); the monitor stays off. Delete the file to reset it.",
                      user_file(os.path.join("process_monitor", "settings.json")), e)
        return None
    env = ComfyEnv()
    routes = server.routes

    async def off_loop(fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    def fail(e, status=400):
        return web.json_response({"error": str(e)}, status=status)

    @routes.get("/bcnodes/monitor/status")
    async def _status(request):
        return web.json_response(await off_loop(monitor.status))

    @routes.post("/bcnodes/monitor/enable")
    async def _enable(request):
        body = await request.json()
        await off_loop(monitor.set_enabled, bool(body.get("enabled")))
        return web.json_response(await off_loop(monitor.status))

    @routes.post("/bcnodes/monitor/arm")
    async def _arm(request):
        body = await request.json()
        try:
            monitor.arm(bool(body.get("armed", True)))
        except RuntimeError as e:
            return fail(e, 409)
        return web.json_response(await off_loop(monitor.status))

    @routes.post("/bcnodes/monitor/settings")
    async def _settings(request):
        try:
            monitor.update_settings(await request.json())
        except ValueError as e:
            return fail(e)
        return web.json_response(await off_loop(monitor.status))

    @routes.get("/bcnodes/monitor/live")
    async def _live(request):
        return web.json_response({"enabled": monitor.enabled, "sample": monitor.last})

    @routes.get("/bcnodes/monitor/last_run")
    async def _last_run(request):
        return web.json_response({"report": await off_loop(monitor.last_run)})

    @routes.get("/bcnodes/monitor/crash")
    async def _crash(request):
        return web.json_response({"report": await off_loop(monitor.crash)})

    @routes.post("/bcnodes/monitor/emulate")
    async def _emulate(request):
        body = await request.json()
        prompt, workflow = body.get("prompt") or {}, body.get("workflow") or {}
        if not prompt:
            return fail("send the workflow's prompt (app.graphToPrompt().output)")
        try:
            return web.json_response(await off_loop(monitor.emulate, prompt, workflow, env))
        except ValueError as e:
            return fail(e)

    if monitor.settings.enabled:
        monitor.start()
    return monitor


MONITOR = _register()

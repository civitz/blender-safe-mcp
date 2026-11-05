"""Blender integration: preferences, API key file, sidebar panel, server lifecycle, op guard."""

import sys
import time
from contextlib import contextmanager

import bpy

from . import keystore
from .mcp_http import MAX_BODY_BYTES, McpHttpServer
from .ops import ALWAYS_ALLOWED, log, registry, undo_push
from .server import OpError

# History ops move along the undo stack; pushing a step after them would discard redo.
NO_UNDO_PUSH = frozenset({"undo", "redo"})

ADDON_ID = __package__
DEFAULT_PORT = 9877

_state = {"server": None, "key": "", "tripped": None}


def _addon_version():
    v = getattr(sys.modules.get(ADDON_ID), "bl_info", {}).get("version", (0, 0, 0))
    return ".".join(str(x) for x in v)


def prefs():
    return bpy.context.preferences.addons[ADDON_ID].preferences


def _ensure_key():
    """Load the key from its file (creating it on first use). Main thread only;
    the server threads read the cached copy in _state."""
    if not _state["key"]:
        _state["key"], _ = keystore.load_or_create()
    return _state["key"]


# --------------------------------------------------------------------------- guard

@contextmanager
def _guard(op):
    if op.name in ALWAYS_ALLOWED:
        yield
        return
    if _state["tripped"]:
        raise OpError("disabled", f"safety tripwire fired earlier: {_state['tripped']}; "
                                  "restart the server from the Blender Safe MCP panel")
    if bpy.context.preferences.filepaths.use_scripts_auto_execute:
        raise OpError("unsafe_blender", "Blender's 'Auto Run Python Scripts' is enabled. "
                      "Ask the user to disable it in the Blender Safe MCP panel before continuing.")
    if op.mutates and not prefs().allow_modify:
        raise OpError("read_only", "scene changes are disabled; ask the user to enable "
                      "'Allow scene changes' in the Blender Safe MCP panel")
    if not op.mutates:
        yield
        return
    texts = {t.as_pointer() for t in bpy.data.texts}
    libs = {lb.as_pointer() for lb in bpy.data.libraries}
    images = _image_snapshot()
    refused = False
    try:
        yield
    except OpError:
        # Ops check their inputs before changing anything, so a refusal normally leaves the
        # scene as it was; recording a step anyway would put an empty step on the stack, and
        # the next undo would land on it and seem to do nothing. Crashes still get a step,
        # since they may leave a partial edit.
        refused = True
        raise
    finally:
        new_texts = [t for t in bpy.data.texts if t.as_pointer() not in texts]
        new_libs = [lb for lb in bpy.data.libraries if lb.as_pointer() not in libs]
        if new_texts or new_libs:
            for t in new_texts:
                bpy.data.texts.remove(t)
            for lb in new_libs:
                bpy.data.libraries.remove(lb)
            _trip(f"op {op.name} created text/library datablocks")
        bad_images = _image_violations(images)
        if bad_images:
            _trip(f"op {op.name} {bad_images}")
        if op.name not in NO_UNDO_PUSH and not refused:
            undo_push(f"MCP: {op.name}")
    if _state["tripped"]:
        raise OpError("disabled", "safety tripwire fired; server disabled")


def _trip(reason):
    _state["tripped"] = reason
    log.critical("TRIPWIRE: %s; server disabled", reason)
    srv = _state["server"]
    if srv is not None:
        srv.reset_sessions()  # every later op is refused above until a restart


def _image_snapshot():
    return {i.as_pointer(): (i.filepath_raw, i.source, i.packed_file is not None)
            for i in bpy.data.images}


def _image_violations(before):
    """Image tripwire. Empirical (Blender 5.1): Image.pack() on a generated image leaves
    filepath_raw '' but flips `source` GENERATED -> FILE and sets packed_file; so that one
    transition (with the path unchanged and the image now packed) is legitimate. Anything
    else that gives an image a path, re-sources it or unpacks it trips. New images must be
    packed, or generated/viewer with no path (render results); offenders are removed.
    Returns a description of the first violation, or ''."""
    why = ""
    for img in list(bpy.data.images):
        ptr = img.as_pointer()
        packed = img.packed_file is not None
        old = before.get(ptr)
        if old is None:
            if (packed and not img.filepath_raw) or (
                    not packed and not img.filepath_raw and img.source in ("GENERATED", "VIEWER")):
                continue
            why = why or "created an image that is neither embedded nor generated"
            bpy.data.images.remove(img)
            continue
        path, source, was_packed = old
        if img.filepath_raw != path:
            why = why or "changed an image's file path"
        elif source != img.source and not (source == "GENERATED" and img.source == "FILE"
                                           and packed):
            why = why or "changed an image's source"
        elif was_packed and not packed:
            why = why or "unpacked an image"
    return why


registry.guard = _guard


# --------------------------------------------------------------------------- server

def _pump():
    srv = _state["server"]
    if srv is None or not srv.running:
        return None
    srv.pump()
    return 0.05


def start_server():
    stop_server()
    p = prefs()
    _ensure_key()
    _state["tripped"] = None
    srv = McpHttpServer(lambda: _state["key"], registry.dispatch, host=p.host, port=p.port,
                        server_version=_addon_version())
    srv.max_body_bytes = _body_cap(p)
    srv.start()
    _state["server"] = srv
    log.info("server started (allow_modify=%s, accept_images=%s, body cap %d bytes)",
             p.allow_modify, p.accept_images, srv.max_body_bytes)
    if not bpy.app.timers.is_registered(_pump):
        bpy.app.timers.register(_pump, first_interval=0.05, persistent=True)


def stop_server():
    srv, _state["server"] = _state["server"], None
    if srv is not None:
        srv.stop()
    if bpy.app.timers.is_registered(_pump):
        bpy.app.timers.unregister(_pump)


def _http_url(p):
    srv = _state["server"]
    if srv is not None:
        return srv.url
    host = "127.0.0.1" if p.host in ("0.0.0.0", "::", "") else p.host
    return f"http://{host}:{p.port}/mcp"


def _http_command(p):
    return (f'claude mcp add --transport http blender-safe-mcp {_http_url(p)} '
            f'--header "Authorization: Bearer {_ensure_key()}"')


# --------------------------------------------------------------------------- UI

def _allow_modify_changed(self, context):
    log.warning("scene changes %s by the user", "ENABLED" if self.allow_modify else "disabled")


def _body_cap(p):
    """HTTP body cap: today's 1 MB, unless uploads are accepted (then max_request_mb)."""
    return p.max_request_mb * 1024 * 1024 if p.accept_images else MAX_BODY_BYTES


def _apply_body_cap(self, context=None):
    srv = _state["server"]
    if srv is not None:
        srv.max_body_bytes = _body_cap(self)


def _accept_images_changed(self, context):
    log.warning("image uploads %s by the user", "ENABLED" if self.accept_images else "disabled")
    _apply_body_cap(self)


class SAFEMCP_Preferences(bpy.types.AddonPreferences):
    bl_idname = ADDON_ID

    host: bpy.props.StringProperty(
        name="Bind address", default="127.0.0.1",
        description="127.0.0.1 = this machine only. Anything else exposes the port on your network")
    port: bpy.props.IntProperty(name="Port", default=DEFAULT_PORT, min=1024, max=65535,
                                description="MCP endpoint: http://<bind address>:<port>/mcp")
    allow_modify: bpy.props.BoolProperty(
        name="Allow scene changes", default=False,
        description="When off, the agent can only look at the scene",
        update=_allow_modify_changed)
    accept_images: bpy.props.BoolProperty(
        name="Accept image uploads", default=True,
        description="Let the agent create images and write pixels (embedded in the .blend, "
                    "never saved to disk). When off, the HTTP request size cap drops to 1 MB",
        update=_accept_images_changed)
    max_image_pixels: bpy.props.IntProperty(
        name="Max image pixels", default=16_777_216, min=1, max=268_435_456,
        description="Largest image (width x height) the agent may create, upload or resize to")
    max_request_mb: bpy.props.IntProperty(
        name="Max request size (MB)", default=64, min=1, max=1024,
        description="HTTP request body cap while image uploads are accepted",
        update=_apply_body_cap)
    autostart: bpy.props.BoolProperty(name="Start with Blender", default=False)
    show_key: bpy.props.BoolProperty(name="Show key", default=False,
                                     options={"SKIP_SAVE"})

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "host")
        layout.prop(self, "port")
        layout.prop(self, "autostart")
        layout.label(text="Key, start/stop and permissions are in the 3D view sidebar → Blender Safe MCP.")
        layout.label(text=f"API key file: {keystore.default_path()}")


class SAFEMCP_OT_start(bpy.types.Operator):
    bl_idname = "blender_safe_mcp.start"
    bl_label = "Start server"

    def execute(self, context):
        try:
            start_server()
        except OSError as e:
            self.report({"ERROR"}, f"Could not listen: {e}")
            return {"CANCELLED"}
        return {"FINISHED"}


class SAFEMCP_OT_stop(bpy.types.Operator):
    bl_idname = "blender_safe_mcp.stop"
    bl_label = "Stop server"

    def execute(self, context):
        stop_server()
        return {"FINISHED"}


class SAFEMCP_OT_regenerate_key(bpy.types.Operator):
    bl_idname = "blender_safe_mcp.regenerate_key"
    bl_label = "Regenerate key"
    bl_description = "Create a new API key; clients must use the new one"

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        _state["key"] = keystore.regenerate()
        log.warning("API key regenerated; the old one no longer works")
        if _state["server"] is not None:
            _state["server"].reset_sessions()
        self.report({"INFO"}, "New key generated; update your MCP config")
        return {"FINISHED"}


class SAFEMCP_OT_copy(bpy.types.Operator):
    bl_idname = "blender_safe_mcp.copy"
    bl_label = "Copy"
    what: bpy.props.EnumProperty(items=[("KEY", "Key", ""), ("COMMAND", "claude mcp add", "")])

    def execute(self, context):
        context.window_manager.clipboard = (_ensure_key() if self.what == "KEY"
                                            else _http_command(prefs()))
        self.report({"INFO"}, "Copied to clipboard")
        return {"FINISHED"}


class SAFEMCP_OT_disable_autoexec(bpy.types.Operator):
    bl_idname = "blender_safe_mcp.disable_autoexec"
    bl_label = "Disable Auto Run Python Scripts"

    def execute(self, context):
        context.preferences.filepaths.use_scripts_auto_execute = False
        context.preferences.is_dirty = True
        return {"FINISHED"}


class SAFEMCP_PT_panel(bpy.types.Panel):
    bl_label = "Blender Safe MCP"
    bl_idname = "SAFEMCP_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Blender Safe MCP"

    def draw(self, context):
        layout = self.layout
        p = prefs()
        srv = _state["server"]

        if context.preferences.filepaths.use_scripts_auto_execute:
            box = layout.box()
            box.alert = True
            box.label(text="Auto Run Python Scripts is ON", icon="ERROR")
            box.operator("blender_safe_mcp.disable_autoexec")

        if _state["tripped"]:
            box = layout.box()
            box.alert = True
            box.label(text="Tripwire fired, server disabled", icon="CANCEL")

        if srv is not None and srv.running:
            layout.label(text=f"Serving {srv.url}", icon="LINKED")
            last = srv.last_client
            if last is None:
                layout.label(text="No requests yet", icon="BLANK1")
            else:
                ago = int(time.monotonic() - last[1])
                layout.label(text=f"Last request: {last[0]}, {ago} s ago", icon="USER")
            layout.operator("blender_safe_mcp.stop", icon="PAUSE")
        else:
            layout.label(text="Stopped", icon="UNLINKED")
            layout.operator("blender_safe_mcp.start", icon="PLAY")

        layout.prop(p, "allow_modify", toggle=True,
                    icon="UNLOCKED" if p.allow_modify else "LOCKED")

        box = layout.box()
        box.label(text="Images")
        box.prop(p, "accept_images")
        sub = box.column()
        sub.enabled = p.accept_images
        sub.prop(p, "max_image_pixels")
        sub.prop(p, "max_request_mb")

        box = layout.box()
        box.label(text="API key (kept across reinstalls)")
        row = box.row(align=True)
        key = _state["key"]
        if p.show_key:
            row.label(text=key or "(created on start)")
        else:
            row.label(text=("•" * 12 + key[-4:]) if key else "(created on start)")
        row.prop(p, "show_key", text="", icon="HIDE_OFF" if p.show_key else "HIDE_ON")
        row = box.row(align=True)
        row.operator("blender_safe_mcp.copy", text="Copy key", icon="COPYDOWN").what = "KEY"
        row.operator("blender_safe_mcp.copy", text="Copy 'claude mcp add'",
                     icon="COPYDOWN").what = "COMMAND"
        box.operator("blender_safe_mcp.regenerate_key", icon="FILE_REFRESH")


CLASSES = (SAFEMCP_Preferences, SAFEMCP_OT_start, SAFEMCP_OT_stop, SAFEMCP_OT_regenerate_key,
           SAFEMCP_OT_copy, SAFEMCP_OT_disable_autoexec, SAFEMCP_PT_panel)


def _autostart():
    try:
        p = prefs()
        _ensure_key()
        if p.autostart:
            start_server()
    except Exception as e:  # noqa: BLE001
        log.error("autostart failed", exc_info=e)
    return None


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.app.timers.register(_autostart, first_interval=0.5)


def unregister():
    stop_server()
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)

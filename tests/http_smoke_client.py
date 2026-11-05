"""MCP SDK client for blender_http_smoke.py: drives real ops over HTTP in both protocol eras.
Reads BLENDER_SAFE_MCP_URL and BLENDER_SAFE_MCP_KEY from the environment."""

import base64
import http.client
import json
import os
import sys
import traceback
import urllib.parse
import zlib
import struct

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from blender_safe_mcp.imaging import decode_png  # noqa: E402

import anyio
import httpx2
import mcp
from mcp.client.streamable_http import streamable_http_client

URL, KEY = os.environ["BLENDER_SAFE_MCP_URL"], os.environ["BLENDER_SAFE_MCP_KEY"]
failures = []


def data(r):
    assert not r.is_error, r.content[0].text
    return json.loads(r.content[0].text)


async def session(mode, body):
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {KEY}"}, timeout=120) as hc:
        async with mcp.Client(streamable_http_client(URL, http_client=hc), mode=mode) as c:
            await body(c)


async def modern(c):
    assert c.protocol_version == "2026-07-28", c.protocol_version
    tools = {t.name: t for t in (await c.list_tools()).tools}
    assert len(tools) >= 40 and "list_ops" not in tools, sorted(tools)
    assert tools["delete_objects"].annotations.destructive_hint is True
    assert tools["add_primitive"].annotations.destructive_hint is False
    assert tools["get_scene_info"].annotations.read_only_hint is True

    info = data(await c.call_tool("get_scene_info", {}))
    assert "objects" in info or "name" in info, info
    data(await c.call_tool("add_primitive", {"kind": "cube", "name": "HttpCube"}))
    obj = data(await c.call_tool("get_object_info", {"name": "HttpCube"}))
    assert obj["name"] == "HttpCube", obj
    data(await c.call_tool("set_transform", {"name": "HttpCube", "location": [1, 2, 3]}))
    assert data(await c.call_tool("get_object_info", {"name": "HttpCube"}))["location"] == [1, 2, 3]

    r = await c.call_tool("add_primitive", {"kind": "cube", "evil": "__import__('os')"})
    assert r.is_error and r.content[0].text.startswith("invalid_params"), r.content[0].text
    r = await c.call_tool("set_property", {"target": {"kind": "scene"},
                                           "path": ["render", "filepath"], "value": "/tmp/x"})
    assert r.is_error, r.content[0].text

    # a real render travels back as a PNG image (the server side selected Cycles: CPU-only)
    data(await c.call_tool("add_camera", {"name": "HttpCam", "location": [6, -6, 4],
                                          "look_at": [0, 0, 0]}))
    data(await c.call_tool("set_active_camera", {"name": "HttpCam"}))
    r = await c.call_tool("render_image", {"max_size": 64, "samples": 1})
    assert not r.is_error, r.content[0].text
    assert r.content[1].type == "image" and r.content[1].mime_type == "image/png"

    # get_uv_layout: an op that returns an image built in pure Python
    data(await c.call_tool("uv_project", {"object": "HttpCube", "method": "cube"}))
    r = await c.call_tool("get_uv_layout", {"object": "HttpCube", "size": 128})
    assert not r.is_error, r.content[0].text
    assert r.content[1].type == "image" and r.content[1].mime_type == "image/png"
    assert json.loads(r.content[0].text)["width"] == 128

    logs = data(await c.call_tool("get_debug_logs", {"limit": 20}))
    assert any("http" in rec.get("message", "") for rec in logs["records"]), logs


async def legacy(c):
    assert c.protocol_version == "2025-11-25", c.protocol_version
    assert c.server_info.name == "blender-safe-mcp"
    data(await c.call_tool("delete_objects", {"names": ["HttpCube"]}))
    r = await c.call_tool("get_object_info", {"name": "HttpCube"})
    assert r.is_error and r.content[0].text.startswith("not_found"), r.content[0].text
    # a refused change must not leave an empty undo step for the next undo to land on
    r = await c.call_tool("delete_objects", {"names": ["NoSuchObject"]})
    assert r.is_error and r.content[0].text.startswith("not_found"), r.content[0].text
    u = data(await c.call_tool("undo", {}))
    assert u["steps_done"] == 1 and u["objects_added"] == ["HttpCube"], u
    assert data(await c.call_tool("get_object_info", {"name": "HttpCube"}))["name"] == "HttpCube"
    # redo reports what it changed, and says so when there's nothing left to redo
    r = data(await c.call_tool("redo", {}))
    assert r["steps_done"] == 1 and r["objects_removed"] == ["HttpCube"], r
    r = await c.call_tool("redo", {})
    assert r.is_error and r.content[0].text.startswith("nothing_to_redo"), r.content[0].text
    u = data(await c.call_tool("undo", {}))
    assert u["objects_added"] == ["HttpCube"], u


def png_of(w, h, rnd):
    """Valid RGBA PNG; incompressible noise when rnd, so size ~ w*h*4."""
    raw = b"".join(b"\x00" + (os.urandom(w * 4) if rnd else bytes(w * 4)) for _ in range(h))

    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 0 if rnd else 6)) + chunk(b"IEND", b""))


async def images(c):
    # ~3.2 MB PNG (4.3 MB of base64 in the JSON body): over the old 1 MB cap, under the 8 MB one
    png = png_of(900, 900, True)
    assert 3_000_000 < len(png) < 3_500_000, len(png)
    info = data(await c.call_tool("upload_image", {
        "name": "HttpImg", "format": "png", "data": base64.b64encode(png).decode()}))
    assert info["size"] == [900, 900] and info["kind"] == "embedded", info
    r = await c.call_tool("get_image", {"name": "HttpImg", "preview_max": 64})
    assert not r.is_error and r.content[1].type == "image", r.content[0].text
    assert "/" not in r.content[0].text.replace("\\/", "")
    # undo probe (informational): what does Blender's undo do to an image op?
    data(await c.call_tool("create_image", {"name": "HttpUndo", "width": 4, "height": 4,
                                            "color": [1, 0, 0, 1]}))
    data(await c.call_tool("fill_image_rect", {"name": "HttpUndo", "rect": [0, 0, 4, 4],
                                               "color": [0, 1, 0, 1]}))
    async def first_pixel(name):
        r = await c.call_tool("get_image", {"name": name})
        return list(decode_png(base64.b64decode(r.content[1].data), 1 << 20)[2][:4])

    print("UNDO PROBE pixel before undo:", await first_pixel("HttpUndo"), flush=True)
    u = data(await c.call_tool("undo", {}))
    print("UNDO PROBE pixel after undoing the fill (red was [255,0,0,255]):",
          await first_pixel("HttpUndo"), flush=True)
    names = [i["name"] for i in data(await c.call_tool("list_images", {}))["images"]]
    print(f"UNDO PROBE after fill: steps={u['steps_done']} image still listed={'HttpUndo' in names}",
          flush=True)
    u = data(await c.call_tool("undo", {}))
    names = [i["name"] for i in data(await c.call_tool("list_images", {}))["images"]]
    print(f"UNDO PROBE after create: steps={u['steps_done']} image still listed={'HttpUndo' in names}",
          flush=True)
    # a 10 MB declared body is refused with 413 once authenticated (cap is 8 MB here)
    u = urllib.parse.urlparse(URL)
    for auth, want in ((True, 413), (False, 401)):
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        conn.putrequest("POST", u.path)
        hdrs = [("Content-Type", "application/json"), ("Content-Length", str(10 * 1024 * 1024))]
        if auth:
            hdrs.append(("Authorization", f"Bearer {KEY}"))
        for k, v in hdrs:
            conn.putheader(k, v)
        conn.endheaders()
        st = conn.getresponse().status
        conn.close()
        assert st == want, (auth, st)


for label, mode, body in [("2026-07-28 via discover", "auto", modern),
                          ("handshake era", "legacy", legacy),
                          ("images + body cap", "auto", images)]:
    try:
        anyio.run(session, mode, body)
        print(f"ok   {label}", flush=True)
    except BaseException:  # noqa: BLE001
        failures.append(f"{label}:\n{traceback.format_exc()}")
        print(f"FAIL {label}", flush=True)

print("\n".join(failures))
sys.exit(1 if failures else 0)

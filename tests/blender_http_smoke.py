"""End-to-end MCP-over-HTTP check inside real Blender (headless is fine):

    BLENDER_SAFE_MCP_SMOKE_CLIENT="uv run --python 3.15 --with 'mcp>=2.3,<3' --no-project \
        python -I tests/http_smoke_client.py" \
    blender --background --factory-startup --python-exit-code 1 \
        --python tests/blender_http_smoke.py

Serves the real op registry (real guard, real RNA) on a free port, runs the client command
with BLENDER_SAFE_MCP_URL / BLENDER_SAFE_MCP_KEY in its environment, pumps the main-thread queue until the
client exits, and exits with the client's status. Background Blender doesn't run timers while
a script runs, so this loop stands in for the addon's `bpy.app.timers` pump.
"""

import os
import secrets
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402

from blender_safe_mcp import blender as B  # noqa: E402  (installs the guard)
from blender_safe_mcp.mcp_http import McpHttpServer  # noqa: E402
from blender_safe_mcp.ops import registry  # noqa: E402

PREFS = SimpleNamespace(allow_modify=True, accept_images=True, max_image_pixels=16_777_216,
                        max_request_mb=8)
B.prefs = lambda: PREFS
bpy.context.preferences.filepaths.use_scripts_auto_execute = False
# Cycles renders on the CPU, so render_image works without a GPU.
import addon_utils  # noqa: E402
addon_utils.enable("cycles", default_set=True)
bpy.context.scene.render.engine = "CYCLES"  # set_property can't (see README)

cmd = os.environ.get("BLENDER_SAFE_MCP_SMOKE_CLIENT")
if not cmd:
    print("set BLENDER_SAFE_MCP_SMOKE_CLIENT to the client command (see the docstring)")
    sys.exit(2)

key = secrets.token_urlsafe(32)
server = McpHttpServer(lambda: key, registry.dispatch, port=0, server_version="smoke")
server.max_body_bytes = B._body_cap(PREFS)  # 8 MB: ~3 MB uploads pass, 10 MB don't
server.start()
print(f"serving {server.url}", flush=True)
env = dict(os.environ, BLENDER_SAFE_MCP_URL=server.url, BLENDER_SAFE_MCP_KEY=key)
proc = subprocess.Popen(shlex.split(cmd), env=env)
deadline = time.monotonic() + float(os.environ.get("BLENDER_SAFE_MCP_SMOKE_TIMEOUT", "300"))
try:
    while proc.poll() is None:
        if time.monotonic() > deadline:
            proc.kill()
            print("client timed out")
            break
        if not server.pump():
            time.sleep(0.005)
finally:
    server.stop()
code = proc.wait()
print("\nHTTP SMOKE OK" if code == 0 else f"\nHTTP SMOKE FAILED (client exit {code})")
sys.exit(0 if code == 0 else 1)

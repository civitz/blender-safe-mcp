bl_info = {
    "name": "Blender Safe MCP",
    "author": "Roberto Piva",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Blender Safe MCP",
    "description": "MCP server (HTTP) with a closed set of operations: no Python execution, "
                   "no file paths, no URLs",
    "category": "Interface",
}

import sys  # noqa: E402

# Blender re-executes only this file when the addon is reinstalled or reloaded;
# submodules already in sys.modules would keep running their old code. Drop them
# so the imports below load the files that are on disk now. The reload reuses this
# package's module object, so its old submodule attributes must go too: otherwise
# `from . import props` returns the stale attribute instead of importing the new file.
for _name in [m for m in sys.modules if m.startswith(__name__ + ".")]:
    del sys.modules[_name]
    globals().pop(_name[len(__name__) + 1:].split(".")[0], None)

# The server / mcp_http / schema / registry / props modules are pure Python so they can be
# tested outside Blender; only import the bpy-dependent parts inside Blender.
try:
    import bpy  # noqa: F401
except ImportError:
    bpy = None

if bpy is not None:
    from .blender import register, unregister  # noqa: F401

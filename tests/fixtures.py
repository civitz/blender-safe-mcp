"""A small test registry standing in for the addon's ops."""
from blender_safe_mcp import logbuf
from blender_safe_mcp.imaging import encode_png_rgba
from blender_safe_mcp.registry import Registry, WithBlobs
from blender_safe_mcp.schema import Int, Obj, Opt, Str
from blender_safe_mcp.server import OpError

ADDON_LOG, ADDON_RING = logbuf.setup("blender_safe_mcp", console=False)
KEY = "test-key-0123456789abcdefghijklmnopqrstuvwxyz"


def make_registry():
    reg = Registry()

    @reg.op("list_ops")
    def list_ops():
        return {"ops": reg.describe()}

    @reg.op("echo", Obj({"text": Str(max_len=100), "n": Opt(Int(0, 10), 1)}))
    def echo(text, n):
        """Echo text n times."""
        return {"text": text * n}

    @reg.op("png", Obj({"size": Opt(Int(1, 64), 2)}))
    def png(size):
        """A tiny red PNG."""
        data = encode_png_rgba(size, size, b"\xff\x00\x00\xff" * size * size)
        return WithBlobs({"image": {"$blob": 0}, "size": size}, [data])

    @reg.op("refuse", mutates=True)
    def refuse():
        raise OpError("read_only", "nope")

    @reg.op("crash")
    def crash():
        return {}["missing"]

    @reg.op("get_debug_logs", Obj({"limit": Opt(Int(1, 500), 100),
                                   "min_level": Opt(Str(), "DEBUG"),
                                   "since_id": Opt(Int(0, 2 ** 53), 0),
                                   "errors_only": Opt(__import__("blender_safe_mcp.schema").schema.Bool(), False)}))
    def get_debug_logs(limit, min_level, since_id, errors_only):
        return ADDON_RING.query(limit=limit, min_level=min_level, since_id=since_id,
                                errors_only=errors_only)

    return reg

"""Op registry and dispatch. Pure Python, no bpy.

An op is a named function with a strict parameter schema. Dispatch is the only
way anything from the network reaches Blender: unknown op -> error, params that
don't validate -> error, blobs to an op that doesn't take them -> error.
"""

import re

from .schema import Obj, ValidationError
from .server import OpError

OP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class WithBlobs:
    """Return value of an op that sends binary data back (e.g. PNG images)."""

    def __init__(self, result, blobs):
        self.result, self.blobs = result, list(blobs)


class Op:
    def __init__(self, name, fn, params, mutates, max_blobs, description, destructive=False):
        self.name, self.fn, self.params = name, fn, params
        self.mutates, self.max_blobs, self.description = mutates, max_blobs, description
        # Removes or irreversibly rewrites existing data (a hint for MCP clients).
        self.destructive = destructive


class Registry:
    def __init__(self):
        self.ops = {}
        # Hook: guard(op) -> context manager wrapped around every call. The
        # Blender layer uses it for policy checks, undo steps and tripwires.
        self.guard = None

    def op(self, name, params=None, mutates=False, max_blobs=0, description=None,
           destructive=False):
        if not OP_NAME_RE.match(name):
            raise ValueError(f"bad op name {name!r}")
        if destructive and not mutates:
            raise ValueError(f"{name}: destructive ops must be mutating")
        params = params if params is not None else Obj({})

        def deco(fn):
            if name in self.ops:
                raise ValueError(f"duplicate op {name}")
            desc = description or (fn.__doc__ or "").strip()
            self.ops[name] = Op(name, fn, params, mutates, max_blobs, desc, destructive)
            return fn
        return deco

    def describe(self):
        return [{"name": o.name, "description": o.description, "mutates": o.mutates,
                 "destructive": o.destructive,
                 "accepts_images": o.max_blobs, "input_schema": o.params.json_schema()}
                for o in sorted(self.ops.values(), key=lambda o: o.name)]

    def dispatch(self, name, params, blobs=()):
        op = self.ops.get(name)
        if op is None:
            raise OpError("unknown_op", f"unknown op {name!r}")
        if len(blobs) > op.max_blobs:
            raise OpError("bad_request", f"{name} accepts at most {op.max_blobs} binary attachments")
        try:
            args = op.params.validate(params)
        except ValidationError as e:
            raise OpError("invalid_params", str(e)) from None
        if op.max_blobs:
            args["blobs"] = list(blobs)
        if self.guard is not None:
            with self.guard(op):
                out = op.fn(**args)
        else:
            out = op.fn(**args)
        if isinstance(out, WithBlobs):
            return out.result, out.blobs
        return out, []

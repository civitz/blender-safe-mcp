"""Generate (or check) TOOLS.md from the addon's op registry.

Run inside Blender, because the op modules import bpy:

    blender --background --factory-startup --python-exit-code 1 \
        --python tests/gen_tools_md.py -- [--check]

Without --check the file is rewritten. With --check Blender exits non-zero when TOOLS.md differs
from what the registry produces (blender_smoke.py runs the same check). The data is the one
`tools/list` serves, plus the parameter defaults, which the JSON Schema does not carry.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
TOOLS_MD = os.path.join(HERE, "..", "TOOLS.md")

from blender_safe_mcp import blender as _blender  # noqa: E402,F401  (registers every op)
from blender_safe_mcp import schema as S  # noqa: E402
from blender_safe_mcp.ops import registry  # noqa: E402

# Display order. Every registered op must appear in exactly one group; render() fails otherwise,
# so a new op cannot be forgotten.
GROUPS = [
    ("Session and diagnostics", "What the server is doing, and undo/redo of its own steps.",
     ["get_status", "list_ops", "get_debug_logs", "undo", "redo"]),
    ("Looking at the scene", "Read-only inspection, viewport capture and rendering.",
     ["get_scene_info", "list_objects", "get_object_info", "get_property", "capture_viewport",
      "set_viewport", "render_image"]),
    ("Creating and editing objects", "Primitives, lights, cameras, text, transforms, parenting, "
     "the generic property setter, animation and the world.",
     ["add_primitive", "add_mesh", "add_light", "add_camera", "add_empty", "add_text",
      "set_active_camera", "set_transform", "set_parent", "duplicate_object", "delete_objects",
      "join_objects", "rename", "set_object_mode", "set_property", "set_frame",
      "insert_keyframe", "set_world"]),
    ("Modifiers", "", ["add_modifier", "remove_modifier", "apply_modifiers",
                       "set_modifier_texture"]),
    ("Materials and shader nodes", "",
     ["create_material", "assign_material", "get_shader_graph", "add_shader_node",
      "remove_shader_node", "link_shader_nodes", "unlink_shader_socket", "set_node_input",
      "set_node_image"]),
    ("Transform and placement", "Cursor, origins and world-space placement of whole objects.",
     ["set_cursor", "set_origin", "apply_transform", "align_objects", "distribute_objects",
      "snap_objects", "drop_to_surface", "look_at"]),
    ("Mesh analysis and repair", "",
     ["analyze_mesh", "find_close_vertices", "get_boundary_loops", "get_mesh_elements",
      "repair_mesh", "merge_vertices", "fill_hole", "remove_duplicate_faces", "delete_faces",
      "create_face"]),
    ("Mesh editing", "Operate on a selector, in Object mode, through bmesh.",
     ["extrude", "inset", "transform_elements", "delete_elements", "dissolve", "set_edge_flags",
      "normals", "bevel", "loop_cut", "bisect", "subdivide", "bridge_edge_loops", "spin",
      "smooth_vertices", "symmetrize", "separate", "snap_vertices", "triangulate_faces",
      "poke_faces"]),
    ("Curves", "", ["add_curve", "add_curve_primitive", "get_curve_info", "set_spline",
                    "add_spline", "remove_spline", "set_curve_shape", "convert_object"]),
    ("Textures and UVs", "",
     ["create_texture", "set_texture_image", "list_uv_maps", "uv_maps", "uv_project",
      "uv_unwrap", "transform_uvs", "get_uv_layout"]),
    ("Images", "Pixels live in packed or generated images; no path is ever set or returned.",
     ["list_images", "get_image", "create_image", "upload_image", "write_image_region",
      "fill_image_rect", "set_image_settings", "delete_image"]),
]

TITLES = {"select": "Selector", "splines": "Spline"}  # shared type names that differ from the parameter name
SHARED_MIN_USES = 3  # an object schema used by this many ops or more is documented once


def _key(spec):
    import json
    return json.dumps(spec.json_schema(), sort_keys=True)


def _walk_objs(spec, name, found):
    """Collect (param name, Obj) pairs below `spec`."""
    if isinstance(spec, S.Obj):
        found.append((name, spec))
        for k, f in spec.fields.items():
            _walk_objs(f.spec, k, found)
    elif isinstance(spec, S.List):
        _walk_objs(spec.item, name, found)
    elif isinstance(spec, S.Union):
        for o in spec.options:
            _walk_objs(o, name, found)


def _shared_types():
    """{schema key: (Title, Obj)} for object schemas many ops use (the selector, ...)."""
    uses, first = {}, {}
    for op in registry.ops.values():
        seen = set()
        found = []
        for k, f in op.params.fields.items():
            _walk_objs(f.spec, k, found)
        for name, obj in found:
            key = _key(obj)
            if key not in seen:
                seen.add(key)
                uses[key] = uses.get(key, 0) + 1
                first.setdefault(key, (name, obj))
    cands = {k for k, n in uses.items() if n >= SHARED_MIN_USES and first[k][1].fields}
    # an object that only occurs inside another shared type (the selector's box, sphere, ...)
    # is documented as part of it
    nested = set()
    for key in cands:
        inner = []
        for k, f in first[key][1].fields.items():
            _walk_objs(f.spec, k, inner)
        nested |= {_key(o) for _, o in inner if _key(o) in cands and uses[_key(o)] <= uses[key]}
    shared = {}
    for key in sorted(cands - nested, key=lambda k: (-uses[k], first[k][0])):
        name, obj = first[key]
        title = TITLES.get(name) or name.replace("_", " ").title().replace(" ", "")
        shared[key] = (title, obj)
    return shared


def _num(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def _range(lo, hi):
    if lo is None and hi is None:
        return ""
    return f" {_num(lo) if lo is not None else '…'}..{_num(hi) if hi is not None else '…'}"


def _type(spec, shared):
    """One-line type summary of a spec."""
    if isinstance(spec, S.Obj):
        hit = shared.get(_key(spec))
        return f"[{hit[0]}](#{hit[0].lower()})" if hit else "object"
    if isinstance(spec, S.Enum):
        return " \\| ".join(f"`{c}`" for c in spec.choices)
    if isinstance(spec, S.Int):
        return "integer" + _range(spec.min, spec.max)
    if isinstance(spec, S.Float):
        return "number" + _range(spec.min, spec.max)
    if isinstance(spec, S.Bool):
        return "boolean"
    if isinstance(spec, S.Str):
        extra = f", pattern `{spec.pattern.pattern}`" if spec.pattern else ""
        return f"string ≤{spec.max_len}{extra}"
    if isinstance(spec, S.Any):
        return "scalar or array of scalars"
    if isinstance(spec, S.Union):
        return " or ".join(_type(o, shared) for o in spec.options)
    if isinstance(spec, S.List):
        item = spec.item
        n = f"{spec.min_len}" if spec.min_len == spec.max_len else f"{spec.min_len}..{spec.max_len}"
        if isinstance(item, S.Float) and spec.min_len == spec.max_len:
            return f"number[{n}]" + _range(item.min, item.max)
        return f"array[{n}] of {_type(item, shared)}"
    return type(spec).__name__


def _default(f):
    if f.required:
        return "**required**"
    d = f.default
    if d is None:
        return "none"
    if isinstance(d, bool):
        return "true" if d else "false"
    return f"`{d}`" if not isinstance(d, str) else f'`"{d}"`'


def _desc(spec):
    return (spec.description or "").replace("|", "\\|").replace("\n", " ")


def _inner_objs(spec):
    """Objects directly inside a spec, looking through arrays and alternatives."""
    if isinstance(spec, S.Obj):
        return [spec]
    if isinstance(spec, S.List):
        return _inner_objs(spec.item)
    if isinstance(spec, S.Union):
        return [o for opt in spec.options for o in _inner_objs(opt)]
    return []


def _rows(obj, shared, prefix=""):
    """Table rows for the fields of `obj`; non-shared nested objects get dotted rows."""
    rows = []
    for k, f in obj.fields.items():
        spec = f.spec
        rows.append(f"| `{prefix}{k}` | {_type(spec, shared)} | {_default(f)} | {_desc(spec)} |")
        for inner in _inner_objs(spec):
            if inner.fields and _key(inner) not in shared:
                rows += _rows(inner, shared, f"{prefix}{k}.")
    return rows


TABLE_HEAD = ["| Parameter | Type | Default | Notes |", "|---|---|---|---|"]


def _flags(op):
    flags = []
    if not op.mutates:
        flags.append("read-only")
    else:
        flags.append("changes the scene")
        if op.destructive:
            flags.append("destructive")
    if op.max_blobs:
        flags.append(f"accepts {op.max_blobs} binary attachment{'s' if op.max_blobs > 1 else ''}")
    return ", ".join(flags)


def render():
    ops = registry.ops
    listed = [n for _, _, names in GROUPS for n in names]
    if sorted(listed) != sorted(ops):
        missing = sorted(set(ops) - set(listed))
        extra = sorted(set(listed) - set(ops))
        dupes = sorted({n for n in listed if listed.count(n) > 1})
        raise SystemExit(f"GROUPS in gen_tools_md.py is out of date: not grouped {missing}, "
                         f"unknown {extra}, listed twice {dupes}")
    shared = _shared_types()
    out = [
        "# Tools reference",
        "",
        "<!-- Generated by tests/gen_tools_md.py from the op registry. Do not edit by hand. -->",
        "",
        f"{len(ops)} tools. Every tool is an MCP tool of the same name; this is the data "
        "`tools/list` serves (descriptions, types, ranges), plus the parameter defaults. "
        "Unknown parameters are rejected. A tool marked *read-only* works while "
        "\"Allow scene changes\" is off; *changes the scene* tools are refused with "
        "`read_only` until the user turns it on, and each one is a single undo step. "
        "*Destructive* is the MCP `destructiveHint`: the tool removes or irreversibly rewrites "
        "existing data. Errors come back as `{code, message}`; the codes are listed in the "
        "[README](README.md#errors).",
        "",
        "Regenerate with `blender --background --factory-startup --python "
        "tests/gen_tools_md.py`.",
        "",
        "## Contents",
        "",
    ]
    for title, _, names in GROUPS:
        anchor = title.lower().replace(" ", "-")
        out.append(f"- [{title}](#{anchor}) ({len(names)}): "
                   + ", ".join(f"[`{n}`](#{n})" for n in names))
    if shared:
        out.append("- [Shared types](#shared-types): "
                   + ", ".join(f"[{t}](#{t.lower()})" for t, _ in shared.values()))
    out.append("")
    for title, intro, names in GROUPS:
        out += [f"## {title}", ""]
        if intro:
            out += [intro, ""]
        for n in names:
            op = ops[n]
            out += [f"### {n}", "", f"*{_flags(op)}*", "", op.description.strip(), ""]
            if op.params.fields:
                out += TABLE_HEAD + _rows(op.params, shared) + [""]
            else:
                out += ["No parameters.", ""]
    if shared:
        out += ["## Shared types", "",
                "Object parameters that many tools take, documented once.", ""]
        for title, obj in shared.values():
            out += [f"### {title}", ""]
            if obj.description:
                out += [obj.description, ""]
            out += TABLE_HEAD + _rows(obj, shared) + [""]
    return "\n".join(out).rstrip("\n") + "\n"


def check():
    """True when TOOLS.md matches the registry."""
    try:
        with open(TOOLS_MD, encoding="utf-8") as fh:
            return fh.read() == render()
    except FileNotFoundError:
        return False


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        if not check():
            print("TOOLS.md is out of date: regenerate it with tests/gen_tools_md.py")
            sys.exit(1)
        print("TOOLS.md is up to date")
    else:
        with open(TOOLS_MD, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {os.path.normpath(TOOLS_MD)} ({len(registry.ops)} tools)")

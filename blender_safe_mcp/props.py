"""Structural, allow-listed RNA property access. No bpy import: works on anything
exposing Blender's `bl_rna` introspection API, so tests can use fakes.

Paths are JSON lists, never strings that get resolved. Every segment must be a
real RNA property of the current struct (not a Python attribute, so no
`__class__`, no methods). A COLLECTION segment must be followed by an int index
or a str item name. Every struct reached must be allow-listed; a few are
deny-listed outright. Values are checked against RNA metadata (type, enum
items, hard ranges, array length) plus our own budget caps.
"""

import math

from .server import OpError

# Structs a path may pass through or end on. Matched against the struct's
# whole base chain, so "Modifier" covers SubsurfModifier etc.
# Never add "ID", "Struct", "PropertyGroup" or "bpy_struct": everything inherits them.
ALLOWED_STRUCTS = frozenset({
    "Object", "Mesh", "Curve", "Light", "Camera", "Material", "World", "Scene",
    "RenderSettings", "SceneEEVEE", "CyclesRenderSettings", "CyclesObjectSettings",
    "Modifier", "Node", "NodeSocket", "NodeTree", "MaterialSlot", "ObjectDisplay",
    "View3DShading", "ColorRamp", "ColorRampElement", "CameraDOFSettings",
    "CurveMapping", "TexMapping", "ColorMapping",
    "Spline", "BezierSplinePoint", "SplinePoint", "Texture",
})

# Never reachable, even if a base class is allowed.
DENIED_STRUCTS = frozenset({
    "ShaderNodeScript", "ShaderNodeTexIES",
    "ShaderNodeTexPointDensity", "CompositorNodeImage", "CompositorNodeOutputFile",
    "GeometryNodeImportOBJ", "GeometryNodeImportSTL", "GeometryNodeImportPLY",
    "GeometryNodeImportCSV", "GeometryNodeImportVDB", "GeometryNodeImportText",
    # Image/ImageUser (and the TexImage/ImageTexture `image` pointers) are reachable only
    # through the image ops: pixels, settings and assignment all have their own checks there.
    "Image", "ImageUser", "Text", "Library", "Driver", "DriverVariable", "FCurve", "AnimData",
    "Preferences", "WindowManager", "Window", "Screen", "Area", "Region",
    "FFmpegSettings", "ImageFormatSettings", "BakeSettings",
})

# (struct in base chain, prop) that can be read but not written here. Texture.type is
# writable in RNA: changing it would turn a procedural texture into an IMAGE one, which only
# the dedicated image ops may create.
READONLY_PROPS = frozenset({("Texture", "type")})

# Runtime (Python-defined) properties are denied except on these structs:
# another addon's properties may have update callbacks with side effects.
RUNTIME_OK_STRUCTS = frozenset({"CyclesRenderSettings", "CyclesObjectSettings"})

DENIED_SUBTYPES = frozenset({"FILE_PATH", "DIR_PATH", "FILE_NAME", "BYTE_STRING", "PASSWORD"})

DENIED_PROPS = frozenset({
    "filepath", "filepath_raw", "directory", "script", "bytecode", "bytecode_hash",
    "use_autopack", "frame_path", "use_scripts_auto_execute", "library", "library_weak_reference",
    "override_library", "use_fake_user", "use_extra_user", "is_evaluated", "original",
    "driver_namespace", "rna_type", "bl_rna", "bl_idname", "id_properties_ensure",
})

# String properties that may be written, as (struct in base chain, prop).
WRITABLE_STRINGS = frozenset({
    ("ID", "name"), ("TextCurve", "body"), ("Node", "label"), ("Node", "name"),
    ("Modifier", "name"),
    # names of UV maps / attributes (plain strings, not paths: subtype NONE is still checked)
    ("ShaderNodeUVMap", "uv_map"), ("ShaderNodeNormalMap", "uv_map"),
    ("ShaderNodeAttribute", "attribute_name"),
})
MAX_STRING = 4096

# Budget caps on top of RNA hard limits: (struct in base chain, prop) -> (min, max).
NUMERIC_CAPS = {
    ("SubsurfModifier", "levels"): (0, 4),
    ("SubsurfModifier", "render_levels"): (0, 4),
    ("MultiresModifier", "levels"): (0, 4),
    ("MultiresModifier", "render_levels"): (0, 4),
    ("MultiresModifier", "sculpt_levels"): (0, 4),
    ("ArrayModifier", "count"): (1, 500),
    ("BevelModifier", "segments"): (1, 32),
    ("ScrewModifier", "steps"): (1, 512),
    ("ScrewModifier", "render_steps"): (1, 512),
    ("ScrewModifier", "iterations"): (1, 64),
    ("RemeshModifier", "octree_depth"): (1, 8),
    ("RemeshModifier", "voxel_size"): (0.01, 1e6),
    ("RemeshModifier", "adaptivity"): (0.0, 1.0),
    ("SolidifyModifier", "thickness"): (-1e4, 1e4),
    ("WireframeModifier", "thickness"): (-1e4, 1e4),
    ("SmoothModifier", "iterations"): (0, 100),
    ("CorrectiveSmoothModifier", "iterations"): (0, 100),
    ("LaplacianSmoothModifier", "iterations"): (0, 100),
    ("DecimateModifier", "iterations"): (0, 100),
    ("Curve", "resolution_u"): (1, 64),
    ("Curve", "render_resolution_u"): (0, 64),
    ("Curve", "bevel_resolution"): (0, 32),
    ("Spline", "resolution_u"): (1, 64),
    ("Spline", "order_u"): (2, 6),
    ("RenderSettings", "resolution_x"): (4, 4096),
    ("RenderSettings", "resolution_y"): (4, 4096),
    ("RenderSettings", "resolution_percentage"): (1, 100),
    ("RenderSettings", "threads"): (1, 64),
    ("CyclesRenderSettings", "samples"): (1, 1024),
    ("CyclesRenderSettings", "preview_samples"): (1, 1024),
    ("CyclesRenderSettings", "max_bounces"): (0, 64),
    ("SceneEEVEE", "taa_render_samples"): (1, 512),
    ("SceneEEVEE", "taa_samples"): (1, 512),
    ("Scene", "frame_start"): (0, 100000),
    ("Scene", "frame_end"): (0, 100000),
    ("Scene", "frame_current"): (-100000, 100000),
}


def _chain(struct_rna):
    ids = []
    while struct_rna is not None:
        ids.append(struct_rna.identifier)
        struct_rna = getattr(struct_rna, "base", None)
    return ids


def struct_ids(value):
    rna = getattr(value, "bl_rna", None)
    return _chain(rna) if rna is not None else []


def _check_struct(value, where):
    ids = struct_ids(value)
    if not ids:
        raise OpError("denied", f"{where}: not an RNA struct")
    if any(i in DENIED_STRUCTS for i in ids):
        raise OpError("denied", f"{where}: {ids[0]} is not accessible")
    if not any(i in ALLOWED_STRUCTS for i in ids):
        raise OpError("denied", f"{where}: {ids[0]} is not accessible")
    return ids


def _rna_prop(value, ids, name, where):
    if not isinstance(name, str):
        raise OpError("invalid_path", f"{where}: expected a property name, got {name!r}")
    prop = value.bl_rna.properties.get(name)
    if prop is None:
        raise OpError("invalid_path", f"{where}: {ids[0]} has no property {name!r}")
    if name in DENIED_PROPS or getattr(prop, "subtype", "NONE") in DENIED_SUBTYPES:
        raise OpError("denied", f"{where}: {ids[0]}.{name} is not accessible")
    if getattr(prop, "is_runtime", False) and not any(i in RUNTIME_OK_STRUCTS for i in ids) \
            and getattr(getattr(prop, "fixed_type", None), "identifier", None) not in RUNTIME_OK_STRUCTS:
        raise OpError("denied", f"{where}: addon-defined property {name!r} is not accessible")
    return prop


def walk(root, path):
    """Follow `path` from `root`. Returns (owner, owner_ids, prop, display_path).

    If the path ends on a property name, `owner` holds it and `prop` is its RNA
    definition (not followed). If it ends on a collection item, or is empty,
    `owner` is that struct and `prop` is None.
    """
    if not isinstance(path, list) or len(path) > 16:
        raise OpError("invalid_path", "path must be a list of at most 16 segments")
    value = root
    ids = _check_struct(value, "root")
    shown = []
    i = 0
    while i < len(path):
        seg = path[i]
        prop = _rna_prop(value, ids, seg, ".".join(shown) or "root")
        shown.append(seg)
        if i == len(path) - 1:
            return value, ids, prop, ".".join(shown)
        if prop.type == "POINTER":
            value = getattr(value, prop.identifier)
            i += 1
        elif prop.type == "COLLECTION":
            key = path[i + 1]
            coll = getattr(value, prop.identifier)
            if isinstance(key, bool) or not isinstance(key, (int, str)):
                raise OpError("invalid_path", f"{'.'.join(shown)}: index must be int or name")
            if isinstance(key, int):
                if not -len(coll) <= key < len(coll):
                    raise OpError("not_found", f"{'.'.join(shown)}[{key}]: index out of range")
                value = coll[key]
            else:
                value = coll.get(key)
            shown[-1] = f"{seg}[{key!r}]"
            i += 2
        else:
            raise OpError("invalid_path", f"{'.'.join(shown)}: is a value, not a struct")
        if value is None:
            raise OpError("not_found", f"{'.'.join(shown)}: is empty")
        ids = _check_struct(value, ".".join(shown))
    return value, ids, None, ".".join(shown) or "root"


def resolve_struct(root, path):
    """Like walk, but a trailing POINTER is followed; returns the struct reached."""
    owner, ids, prop, where = walk(root, path)
    if prop is None:
        return owner
    if prop.type != "POINTER":
        raise OpError("invalid_path", f"{where}: is not a struct")
    value = getattr(owner, prop.identifier)
    if value is None:
        raise OpError("not_found", f"{where}: is empty")
    _check_struct(value, where)
    return value


def _array_len(prop):
    dims = [d for d in (getattr(prop, "array_dimensions", None) or ()) if d]
    if len(dims) > 1:
        return -1  # matrices: not supported
    return getattr(prop, "array_length", 0) or 0


def _cap(ids, name):
    for sid in ids:
        cap = NUMERIC_CAPS.get((sid, name))
        if cap:
            return cap
    return None


def _number(prop, ids, value, where, is_int):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OpError("invalid_value", f"{where}: expected a finite number")
    if is_int:
        if isinstance(value, float):
            if not value.is_integer():
                raise OpError("invalid_value", f"{where}: expected an integer")
            value = int(value)
    else:
        value = float(value)
    lo, hi = getattr(prop, "hard_min", None), getattr(prop, "hard_max", None)
    cap = _cap(ids, prop.identifier)
    if cap:
        lo = cap[0] if lo is None else max(lo, cap[0])
        hi = cap[1] if hi is None else min(hi, cap[1])
    if lo is not None and value < lo or hi is not None and value > hi:
        raise OpError("out_of_range", f"{where}: {value} outside allowed range [{lo}, {hi}]")
    return value


def coerce(prop, ids, value, where):
    """Validate `value` for writing to `prop`. Returns the Python value to setattr."""
    if getattr(prop, "is_readonly", False) or any((sid, prop.identifier) in READONLY_PROPS
                                                  for sid in ids):
        raise OpError("denied", f"{where}: is read-only")
    t = prop.type
    if t in ("BOOLEAN", "INT", "FLOAT"):
        n = _array_len(prop)
        if n < 0:
            raise OpError("denied", f"{where}: matrix properties can't be set directly")

        def one(v):
            if t == "BOOLEAN":
                if not isinstance(v, bool):
                    raise OpError("invalid_value", f"{where}: expected true/false")
                return v
            return _number(prop, ids, v, where, t == "INT")
        if n:
            if not isinstance(value, list) or len(value) != n:
                raise OpError("invalid_value", f"{where}: expected a list of {n} values")
            return [one(v) for v in value]
        return one(value)
    if t == "ENUM":
        items = [it.identifier for it in prop.enum_items]
        if not items:
            raise OpError("denied", f"{where}: dynamic enum, set it through a dedicated op")
        if getattr(prop, "is_enum_flag", False):
            if not isinstance(value, list) or any(v not in items for v in value):
                raise OpError("invalid_value", f"{where}: expected a list drawn from {items}")
            return set(value)
        if value not in items:
            raise OpError("invalid_value", f"{where}: must be one of {items}")
        return value
    if t == "STRING":
        if not any((sid, prop.identifier) in WRITABLE_STRINGS for sid in ids):
            raise OpError("denied", f"{where}: this text property can't be set")
        if not isinstance(value, str) or "\x00" in value:
            raise OpError("invalid_value", f"{where}: expected a string")
        limit = getattr(prop, "length_max", 0) or MAX_STRING
        if len(value) > min(limit, MAX_STRING):
            raise OpError("invalid_value", f"{where}: string too long")
        return value
    raise OpError("denied", f"{where}: {t.lower()} properties can't be set here; use a dedicated op")


def set_prop(root, path, value):
    if not path:
        raise OpError("invalid_path", "path must name a property")
    owner, ids, prop, where = walk(root, path)
    if prop is None:
        raise OpError("invalid_path", f"{where}: path must end on a property name")
    if prop.type in ("POINTER", "COLLECTION"):
        raise OpError("denied", f"{where}: {prop.type.lower()} properties can't be set here")
    v = coerce(prop, ids, value, where)
    try:
        setattr(owner, prop.identifier, v)
    except (TypeError, ValueError, AttributeError) as e:
        raise OpError("invalid_value", f"{where}: {e}") from None
    return to_json(getattr(owner, prop.identifier), prop)


def to_json(value, prop=None, depth=0):
    """Convert an RNA value to plain JSON (no structs leak out)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "bl_rna"):
        out = {"type": value.bl_rna.identifier}
        name = getattr(value, "name", None)
        if isinstance(name, str):
            out["name"] = name
        return out
    if depth < 2:
        try:
            return [to_json(v, None, depth + 1) for v in value]
        except TypeError:
            pass
    return repr(value)[:200]


def get_prop(root, path):
    owner, ids, prop, where = walk(root, path)
    if prop is None:
        return describe_struct(owner)
    value = getattr(owner, prop.identifier)
    if prop.type == "COLLECTION":
        items = list(value)
        return {"collection": True, "length": len(items),
                "items": [to_json(v) for v in items[:100]]}
    return to_json(value, prop)


def describe_prop(prop, ids):
    d = {"name": prop.identifier, "type": prop.type, "readonly": bool(getattr(prop, "is_readonly", False))}
    sub = getattr(prop, "subtype", "NONE")
    if sub and sub != "NONE":
        d["subtype"] = sub
    desc = getattr(prop, "description", "")
    if desc:
        d["description"] = desc[:200]
    if prop.type in ("INT", "FLOAT"):
        cap = _cap(ids, prop.identifier)
        lo, hi = getattr(prop, "hard_min", None), getattr(prop, "hard_max", None)
        if cap:
            lo, hi = max(lo, cap[0]) if lo is not None else cap[0], min(hi, cap[1]) if hi is not None else cap[1]
        d["min"], d["max"] = _finite(lo), _finite(hi)
    if prop.type in ("BOOLEAN", "INT", "FLOAT"):
        n = _array_len(prop)
        if n:
            d["array_length"] = n
    if prop.type == "ENUM":
        d["items"] = [it.identifier for it in prop.enum_items][:64]
        if getattr(prop, "is_enum_flag", False):
            d["multiple"] = True
    if prop.type in ("POINTER", "COLLECTION"):
        ft = getattr(prop, "fixed_type", None)
        d["struct"] = getattr(ft, "identifier", None)
    try:
        if prop.type in ("POINTER", "COLLECTION"):
            d["settable"] = False
        else:
            coerce_check = prop.type != "STRING" or any((s, prop.identifier) in WRITABLE_STRINGS for s in ids)
            d["settable"] = (not d["readonly"]) and coerce_check \
                and not any((s, prop.identifier) in READONLY_PROPS for s in ids) and _array_len(prop) >= 0 \
                and not (prop.type == "ENUM" and not d.get("items"))
    except Exception:  # noqa: BLE001
        d["settable"] = False
    return d


def _finite(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) and abs(x) < 1e30 else None


def describe_struct(value):
    """List the accessible properties of an allowed struct."""
    ids = _check_struct(value, "root")
    props = []
    for prop in value.bl_rna.properties:
        name = prop.identifier
        try:
            _rna_prop(value, ids, name, "root")
        except OpError:
            continue
        props.append(describe_prop(prop, ids))
    return {"struct": ids[0], "bases": ids[1:], "properties": props}

"""The op vocabulary. Everything the network can make Blender do is in this file.

Rules for adding an op:
  * parameters are validated by its schema; no free-form strings reach bpy.ops,
    path_resolve, exec/eval, driver expressions, file paths or URLs;
  * look things up by name in bpy.data collections, never by RNA path strings;
  * generic property writes go through props.set_prop (allow-listed RNA walk);
  * creation uses bpy.data / bmesh, not bpy.ops (no context tricks needed).
"""

import math
import os
import shutil
import tempfile

import bmesh
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

from . import logbuf, props
from .imaging import encode_png_rgba
from .registry import Registry, WithBlobs
from .schema import (Any, Bool, Color, Enum, Float, Int, List, Obj, Opt, Str, Union,
                     ValidationError, Vec)
from .server import OpError

registry = Registry()
op = registry.op
log, RING = logbuf.setup("blender_safe_mcp")

# Ops that keep working when the guard would otherwise refuse (read-only diagnostics).
ALWAYS_ALLOWED = frozenset({"list_ops", "get_status", "get_debug_logs"})

# --------------------------------------------------------------------------- shared specs

NAME = Str(max_len=63, description="Datablock name")
LOC = Vec(3, -1e6, 1e6, description="XYZ in scene units")
ROT = Vec(3, -36000, 36000, description="Euler XYZ in degrees")
SCALE = Vec(3, -1e4, 1e4, description="XYZ scale")
PATH = List(Union(Str(max_len=63), Int(-100000, 100000)), min_len=0, max_len=16,
            description='RNA path as a list, e.g. ["modifiers", "Bevel", "width"] or '
                        '["data", "energy"]. Collections are followed by an index or item name.')
TARGET_KINDS = ("object", "material", "world", "scene", "mesh", "light", "camera", "curve", "texture")
TARGET = Obj({
    "kind": Enum(TARGET_KINDS),
    "name": Opt(NAME),
}, description="Datablock to start from. 'scene' and 'world' default to the current ones.")
NODE_OWNER = Obj({"kind": Enum(("material", "world")), "name": Opt(NAME)},
                 description="Whose shader node tree. world defaults to the scene's world.")

class _Settings(Obj):
    """Flat {property: value} map, validated later against RNA by props.set_prop."""

    def validate(self, value, path=""):
        if not isinstance(value, dict) or len(value) > 32:
            raise ValidationError(path, "expected an object with at most 32 entries")
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or not 0 < len(k) <= 63:
                raise ValidationError(path, "bad property name")
            out[k] = Any().validate(v, f"{path}.{k}")
        return out

    def json_schema(self):
        return {"type": "object", "maxProperties": 32,
                "additionalProperties": Any().json_schema(),
                "description": "RNA property -> value, e.g. {\"levels\": 2}"}


SETTINGS = _Settings()


MAX_POLYS = 5_000_000
MAX_MESH_VERTS = 100_000

ALLOWED_MODIFIERS = (
    "SUBSURF", "BEVEL", "ARRAY", "MIRROR", "SOLIDIFY", "BOOLEAN", "DECIMATE", "SMOOTH",
    "CORRECTIVE_SMOOTH", "LAPLACIANSMOOTH", "WIREFRAME", "SCREW", "TRIANGULATE", "WELD",
    "REMESH", "SIMPLE_DEFORM", "CAST", "WAVE", "EDGE_SPLIT", "SHRINKWRAP", "SKIN",
    "WEIGHTED_NORMAL", "DISPLACE", "BUILD", "CURVE",
)

ALLOWED_SHADER_NODES = (
    "ShaderNodeBsdfPrincipled", "ShaderNodeBsdfDiffuse", "ShaderNodeBsdfGlossy",
    "ShaderNodeBsdfGlass", "ShaderNodeBsdfTransparent", "ShaderNodeBsdfTranslucent",
    "ShaderNodeBsdfRefraction", "ShaderNodeEmission", "ShaderNodeMixShader",
    "ShaderNodeAddShader", "ShaderNodeBackground", "ShaderNodeHoldout",
    "ShaderNodeOutputMaterial", "ShaderNodeOutputWorld", "ShaderNodeVolumePrincipled",
    "ShaderNodeVolumeAbsorption", "ShaderNodeVolumeScatter", "ShaderNodeSubsurfaceScattering",
    "ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexWave", "ShaderNodeTexGradient",
    "ShaderNodeTexChecker", "ShaderNodeTexBrick", "ShaderNodeTexMagic", "ShaderNodeTexSky",
    "ShaderNodeTexWhiteNoise", "ShaderNodeTexCoord", "ShaderNodeMapping", "ShaderNodeValToRGB",
    "ShaderNodeRGBCurve", "ShaderNodeMath", "ShaderNodeVectorMath", "ShaderNodeMix",
    "ShaderNodeMapRange", "ShaderNodeClamp", "ShaderNodeBump", "ShaderNodeNormal",
    "ShaderNodeRGB", "ShaderNodeValue", "ShaderNodeHueSaturation", "ShaderNodeBrightContrast",
    "ShaderNodeGamma", "ShaderNodeInvert", "ShaderNodeLayerWeight", "ShaderNodeFresnel",
    "ShaderNodeSeparateXYZ", "ShaderNodeCombineXYZ", "ShaderNodeSeparateColor",
    "ShaderNodeCombineColor", "ShaderNodeRGBToBW", "ShaderNodeBlackbody",
    "ShaderNodeWavelength", "ShaderNodeLightPath", "ShaderNodeAmbientOcclusion",
    "ShaderNodeObjectInfo", "ShaderNodeNewGeometry", "ShaderNodeVectorRotate",
    "ShaderNodeDisplacement", "ShaderNodeVectorDisplacement",
    "ShaderNodeUVMap", "ShaderNodeNormalMap", "ShaderNodeAttribute",
    "ShaderNodeTexImage", "ShaderNodeTexEnvironment",  # images: assign with set_node_image
)


# --------------------------------------------------------------------------- helpers

def _scene():
    return bpy.context.scene


def _obj(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise OpError("not_found", f"no object named {name!r}")
    return obj


def _material(name):
    mat = bpy.data.materials.get(name)
    if mat is None:
        raise OpError("not_found", f"no material named {name!r}")
    return mat


def _target(spec):
    kind, name = spec["kind"], spec.get("name")
    if kind == "scene":
        if name is None:
            return _scene()
        coll = bpy.data.scenes
    elif kind == "world":
        if name is None:
            if _scene().world is None:
                raise OpError("not_found", "the scene has no world")
            return _scene().world
        coll = bpy.data.worlds
    else:
        coll = {"object": bpy.data.objects, "material": bpy.data.materials,
                "mesh": bpy.data.meshes, "light": bpy.data.lights,
                "camera": bpy.data.cameras, "curve": bpy.data.curves,
                "texture": bpy.data.textures}[kind]
    if name is None:
        raise OpError("invalid_params", f"name is required for kind {kind!r}")
    value = coll.get(name)
    if value is None:
        raise OpError("not_found", f"no {kind} named {name!r}")
    return value


def _link(obj):
    _scene().collection.objects.link(obj) if bpy.context.view_layer is None else \
        bpy.context.view_layer.active_layer_collection.collection.objects.link(obj)


def _place(obj, location, rotation, scale):
    if location is not None:
        obj.location = location
    if rotation is not None:
        obj.rotation_euler = Euler([math.radians(a) for a in rotation], "XYZ")
    if scale is not None:
        obj.scale = scale


def _vec(v):
    return [round(float(x), 6) for x in v]


def _world_bbox(obj):
    if obj.type not in {"MESH", "CURVE", "FONT", "SURFACE", "META"}:
        return None
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return {"min": _vec(lo), "max": _vec(hi)}


def _cursor_info():
    cur = _scene().cursor
    rot = cur.rotation_euler if cur.rotation_mode == "XYZ" else cur.matrix.to_euler("XYZ")
    return {"location": _vec(cur.location),
            "rotation_deg": _vec(math.degrees(a) for a in rot)}


def _obj_summary(obj):
    # dimensions / matrix_world are only refreshed by a depsgraph update; without it the
    # summary of an object we just changed reports the old size. Cheap when nothing is dirty.
    bpy.context.view_layer.update()
    return {
        "name": obj.name,
        "type": obj.type,
        "location": _vec(obj.location),
        "rotation_deg": _vec(math.degrees(a) for a in obj.rotation_euler),
        "scale": _vec(obj.scale),
        "dimensions": _vec(obj.dimensions),
        "parent": obj.parent.name if obj.parent else None,
        "visible": obj.visible_get() if hasattr(obj, "visible_get") else not obj.hide_viewport,
    }


def _find_node_by_type(tree, node_type):
    return next((n for n in tree.nodes if n.type == node_type), None)


def _socket(sockets, key, where):
    if isinstance(key, int):
        if not 0 <= key < len(sockets):
            raise OpError("not_found", f"{where}: socket index {key} out of range")
        return sockets[key]
    enabled = [s for s in sockets if getattr(s, "enabled", True)]
    for pool in (enabled, list(sockets)):
        for s in pool:
            if s.identifier == key:
                return s
        for s in pool:
            if s.name == key:
                return s
    raise OpError("not_found", f"{where}: no socket {key!r}; have "
                  f"{[s.identifier for s in enabled]}")


def _node_tree(owner):
    if owner["kind"] == "material":
        if owner.get("name") is None:
            raise OpError("invalid_params", "material name required")
        idb = _material(owner["name"])
    else:
        idb = _target({"kind": "world", "name": owner.get("name")})
    if idb.node_tree is None:
        try:
            idb.use_nodes = True
        except (AttributeError, TypeError):
            pass
    if idb.node_tree is None:
        raise OpError("not_found", "no node tree")
    return idb.node_tree


def _node(tree, name):
    node = tree.nodes.get(name)
    if node is None:
        raise OpError("not_found", f"no node named {name!r}; have {[n.name for n in tree.nodes][:50]}")
    return node


def _apply_settings(struct, settings, where):
    applied = {}
    for key, value in (settings or {}).items():
        try:
            applied[key] = props.set_prop(struct, [key], value)
        except OpError as e:
            raise OpError(e.code, f"{where}: {e.message}") from None
    return applied


# --------------------------------------------------------------------------- geometry budget

def _array_curve_count(obj, m):
    """Conservative copy count of an Array modifier fitted to a curve: control-polygon length
    of the curve divided by the length of one array step (both in world units)."""
    cobj = m.curve
    if cobj is None or cobj.type != "CURVE":
        return 1
    length = 0.0
    for s in cobj.data.splines:
        pts = [Vector(p.co[:3]) for p in (s.bezier_points if s.type == "BEZIER" else s.points)]
        if s.use_cyclic_u and pts:
            pts.append(pts[0])
        length += sum((b - a).length for a, b in zip(pts, pts[1:]))
        if s.type == "BEZIER":  # handles can bulge outside the control polygon
            length += sum((p.handle_right - p.co).length + (p.handle_left - p.co).length
                          for p in s.bezier_points)
    length *= max(cobj.matrix_world.to_scale())
    step = Vector((0.0, 0.0, 0.0))
    if m.use_relative_offset:
        step += Vector(m.relative_offset_displace[i] * obj.dimensions[i] for i in range(3))
    if m.use_constant_offset:
        step += Vector(m.constant_offset_displace) * max(obj.matrix_world.to_scale())
    if m.use_object_offset and m.offset_object is not None:
        step += (obj.matrix_world.inverted() @ m.offset_object.matrix_world).translation
    if step.length < 1e-6:
        return 1
    return int(length / step.length) + 2


def _estimate_polys(obj):
    """Upper-bound-ish estimate of evaluated face count. Runs before Blender evaluates."""
    base = len(obj.data.polygons) if obj.type == "MESH" else 1000
    est = float(max(base, 1))
    dims = obj.dimensions
    for m in obj.modifiers:
        if not (m.show_viewport or m.show_render):
            continue
        t = m.type
        if t in ("SUBSURF", "MULTIRES"):
            est *= 4 ** max(getattr(m, "levels", 0), getattr(m, "render_levels", 0))
        elif t == "ARRAY":
            if m.fit_type == "FIT_CURVE":
                est *= _array_curve_count(obj, m)
            elif m.fit_type != "FIXED_COUNT":
                raise OpError("budget", "only Array fit_type FIXED_COUNT or FIT_CURVE is allowed")
            else:
                est *= m.count
        elif t == "SCREW":
            est = est * max(m.steps, m.render_steps) * m.iterations + max(m.steps, m.render_steps)
        elif t == "BEVEL":
            est *= 1 + m.segments
        elif t == "REMESH":
            if m.mode == "VOXEL":
                area = 2 * (dims.x * dims.y + dims.y * dims.z + dims.x * dims.z) or 1.0
                est = area / (m.voxel_size ** 2)
            else:
                est = 6 * 4 ** m.octree_depth
        elif t in ("SOLIDIFY", "MIRROR", "TRIANGULATE"):
            est *= 2 if t != "MIRROR" else 2 ** sum(bool(a) for a in m.use_axis)
        elif t in ("WIREFRAME", "SKIN"):
            est *= 6
    return int(est)


def _check_budget(obj):
    est = _estimate_polys(obj)
    if est > MAX_POLYS:
        raise OpError("budget", f"{obj.name}: would produce ~{est:,} faces (limit {MAX_POLYS:,})")


# =========================================================================== read ops

@op("list_ops", description="List every operation this addon accepts, with JSON schemas.")
def list_ops():
    return {"ops": registry.describe()}


@op("get_debug_logs", Obj({
    "limit": Opt(Int(1, 500), 100),
    "min_level": Opt(Enum(("DEBUG", "INFO", "WARNING", "ERROR")), "DEBUG"),
    "since_id": Opt(Int(0, 2 ** 53), 0, ),
    "errors_only": Opt(Bool(), False),
}), description="Recent log records from the Blender addon (connections, every op with its "
                "outcome and timing, refusals, and full tracebacks of crashes). Pass the "
                "returned last_id as since_id to get only newer records.")
def get_debug_logs(limit, min_level, since_id, errors_only):
    return RING.query(limit=limit, min_level=min_level, since_id=since_id,
                      errors_only=errors_only)


@op("get_status", description="Blender version, current file, scene and safety state.")
def get_status():
    return {
        "blender_version": bpy.app.version_string,
        "file": os.path.basename(bpy.data.filepath) or None,
        "scene": _scene().name,
        "auto_run_scripts": bpy.context.preferences.filepaths.use_scripts_auto_execute,
    }


@op("get_scene_info", description="Scene summary: render engine, frame range, camera, world, "
    "and up to 100 objects with transforms.")
def get_scene_info():
    sc = _scene()
    objs = list(sc.objects)
    return {
        "name": sc.name,
        "render_engine": sc.render.engine,
        "frame_range": [sc.frame_start, sc.frame_end],
        "frame_current": sc.frame_current,
        "resolution": [sc.render.resolution_x, sc.render.resolution_y],
        "camera": sc.camera.name if sc.camera else None,
        "world": sc.world.name if sc.world else None,
        "unit_scale": sc.unit_settings.scale_length,
        "cursor": _cursor_info(),
        "object_count": len(objs),
        "objects": [_obj_summary(o) for o in objs[:100]],
        "materials": [m.name for m in bpy.data.materials][:200],
    }


@op("list_objects", Obj({
    "query": Opt(Str(max_len=63)),
    "type": Opt(Enum(("MESH", "CURVE", "FONT", "LIGHT", "CAMERA", "EMPTY", "ARMATURE",
                      "SURFACE", "META", "VOLUME", "GPENCIL", "GREASEPENCIL", "POINTCLOUD",
                      "CURVES", "LATTICE", "SPEAKER", "LIGHT_PROBE"))),
    "limit": Opt(Int(1, 500), 50),
}), description="Objects in the scene, filtered by name substring and/or type.")
def list_objects(query, type, limit):
    out = []
    q = (query or "").lower()
    for o in _scene().objects:
        if q and q not in o.name.lower():
            continue
        if type and o.type != type:
            continue
        out.append(_obj_summary(o))
        if len(out) >= limit:
            break
    return {"objects": out}


@op("get_object_info", Obj({"name": NAME}),
    description="Details for one object: transform, world bounding box, mesh stats, "
                "materials, modifiers, children.")
def get_object_info(name):
    obj = _obj(name)
    info = _obj_summary(obj)
    info["world_bounding_box"] = _world_bbox(obj)
    info["children"] = [c.name for c in obj.children][:100]
    info["collections"] = [c.name for c in obj.users_collection]
    info["materials"] = [s.material.name if s.material else None for s in obj.material_slots]
    info["modifiers"] = [{"name": m.name, "type": m.type, "show_viewport": m.show_viewport}
                         for m in obj.modifiers]
    if obj.type == "MESH":
        me = obj.data
        info["mesh"] = {"name": me.name, "vertices": len(me.vertices), "edges": len(me.edges),
                        "faces": len(me.polygons)}
    elif obj.data is not None:
        info["data"] = {"name": obj.data.name, "type": obj.data.bl_rna.identifier}
    return info


@op("get_property", Obj({"target": TARGET, "path": PATH}),
    description="Read a property through an allow-listed RNA path. An empty path, or a path "
                "ending on a struct, lists that struct's accessible properties.")
def get_property(target, path):
    root = _target(target)
    if not path:
        return props.describe_struct(root)
    owner, ids, prop, where = props.walk(root, path)
    if prop is None:
        return props.describe_struct(owner)
    if prop.type == "POINTER" and getattr(owner, prop.identifier) is not None:
        try:
            return props.describe_struct(props.resolve_struct(root, path))
        except OpError:
            pass
    return {"path": where, "value": props.get_prop(root, path)}


# =========================================================================== create

PRIMITIVES = ("cube", "uv_sphere", "ico_sphere", "cylinder", "cone", "plane", "grid",
              "circle", "torus", "monkey")


def _build_primitive(bm, kind, size, segments, rings, subdivisions, minor_radius):
    r = size / 2
    if kind == "cube":
        bmesh.ops.create_cube(bm, size=size)
    elif kind == "uv_sphere":
        bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=rings, radius=r)
    elif kind == "ico_sphere":
        bmesh.ops.create_icosphere(bm, subdivisions=subdivisions, radius=r)
    elif kind in ("cylinder", "cone"):
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                              radius1=r, radius2=r if kind == "cylinder" else 0.0, depth=size)
    elif kind == "plane":
        bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=r)
    elif kind == "grid":
        bmesh.ops.create_grid(bm, x_segments=segments, y_segments=segments, size=r)
    elif kind == "circle":
        bmesh.ops.create_circle(bm, cap_ends=True, segments=segments, radius=r)
    elif kind == "monkey":
        bmesh.ops.create_monkey(bm)
        bmesh.ops.scale(bm, vec=(r, r, r), verts=bm.verts)
    elif kind == "torus":
        R, rr = r, minor_radius if minor_radius is not None else r * 0.25
        verts = []
        for i in range(segments):
            u = 2 * math.pi * i / segments
            ring = []
            for j in range(rings):
                v = 2 * math.pi * j / rings
                ring.append(bm.verts.new(((R + rr * math.cos(v)) * math.cos(u),
                                          (R + rr * math.cos(v)) * math.sin(u),
                                          rr * math.sin(v))))
            verts.append(ring)
        for i in range(segments):
            for j in range(rings):
                a, b = verts[i][j], verts[(i + 1) % segments][j]
                c, d = verts[(i + 1) % segments][(j + 1) % rings], verts[i][(j + 1) % rings]
                bm.faces.new((a, b, c, d))


@op("add_primitive", Obj({
    "kind": Enum(PRIMITIVES),
    "name": Opt(NAME),
    "size": Opt(Float(0.0001, 1e5, description="Overall size (diameter / edge length)"), 2.0),
    "segments": Opt(Int(3, 256), 32),
    "rings": Opt(Int(3, 128), 16),
    "subdivisions": Opt(Int(1, 6), 2),
    "minor_radius": Opt(Float(0.0001, 1e5)),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True, description="Add a mesh primitive.")
def add_primitive(kind, name, size, segments, rings, subdivisions, minor_radius,
                  location, rotation_deg, scale):
    name = name or kind.replace("_", " ").title().replace(" ", "")
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    try:
        _build_primitive(bm, kind, size, segments, rings, subdivisions, minor_radius)
        bm.to_mesh(mesh)
    finally:
        bm.free()
    obj = bpy.data.objects.new(name, mesh)
    _link(obj)
    _place(obj, location, rotation_deg, scale)
    return _obj_summary(obj)


@op("add_mesh", Obj({
    "name": NAME,
    "vertices": List(Vec(3, -1e6, 1e6), 1, MAX_MESH_VERTS),
    "faces": Opt(List(List(Int(0, MAX_MESH_VERTS - 1), 3, 64), 0, MAX_MESH_VERTS), []),
    "edges": Opt(List(List(Int(0, MAX_MESH_VERTS - 1), 2, 2), 0, MAX_MESH_VERTS), []),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True, description="Create a mesh object from explicit vertices and faces "
                              "(face = list of vertex indices).")
def add_mesh(name, vertices, faces, edges, location, rotation_deg, scale):
    n = len(vertices)
    for f in faces:
        if max(f) >= n or len(set(f)) != len(f):
            raise OpError("invalid_params", "face references a missing or repeated vertex")
    for e in edges:
        if max(e) >= n or e[0] == e[1]:
            raise OpError("invalid_params", "edge references a missing or repeated vertex")
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, edges, faces)
    mesh.validate()  # fixes degenerate geometry in place
    obj = bpy.data.objects.new(name, mesh)
    _link(obj)
    _place(obj, location, rotation_deg, scale)
    return _obj_summary(obj)


@op("add_light", Obj({
    "type": Enum(("POINT", "SUN", "SPOT", "AREA")),
    "name": Opt(NAME),
    "energy": Opt(Float(0, 1e7, description="Watts (sun: W/m²)"), 1000.0),
    "color": Opt(Color()),
    "size": Opt(Float(0, 1e4, description="Radius / area size / sun angle (deg)")),
    "spot_angle_deg": Opt(Float(1, 180)),
    "location": Opt(LOC), "rotation_deg": Opt(ROT),
}), mutates=True, description="Add a light.")
def add_light(type, name, energy, color, size, spot_angle_deg, location, rotation_deg):
    name = name or type.title() + "Light"
    light = bpy.data.lights.new(name, type)
    light.energy = energy
    if color:
        light.color = color
    if size is not None:
        if type == "SUN":
            light.angle = math.radians(min(size, 180))
        elif type == "AREA":
            light.size = size
        else:
            light.shadow_soft_size = size
    if spot_angle_deg is not None and type == "SPOT":
        light.spot_size = math.radians(spot_angle_deg)
    obj = bpy.data.objects.new(name, light)
    _link(obj)
    _place(obj, location, rotation_deg, None)
    return _obj_summary(obj)


@op("add_camera", Obj({
    "name": Opt(NAME),
    "lens_mm": Opt(Float(1, 5000), 50.0),
    "location": Opt(LOC), "rotation_deg": Opt(ROT),
    "look_at": Opt(LOC),
    "make_active": Opt(Bool(), True),
}), mutates=True, description="Add a camera, optionally aimed at a point and made the scene camera.")
def add_camera(name, lens_mm, location, rotation_deg, look_at, make_active):
    cam = bpy.data.cameras.new(name or "Camera")
    cam.lens = lens_mm
    obj = bpy.data.objects.new(cam.name, cam)
    _link(obj)
    _place(obj, location, rotation_deg, None)
    if look_at is not None:
        _aim(obj, look_at)
    if make_active:
        _scene().camera = obj
    return _obj_summary(obj)


@op("add_text", Obj({
    "body": Str(max_len=1024, multiline=True),
    "name": Opt(NAME),
    "size": Opt(Float(0.0001, 1e4), 1.0),
    "extrude": Opt(Float(0, 1e3), 0.0),
    "align_x": Opt(Enum(("LEFT", "CENTER", "RIGHT", "JUSTIFY", "FLUSH")), "LEFT"),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True, description="Add a 3D text object.")
def add_text(body, name, size, extrude, align_x, location, rotation_deg, scale):
    curve = bpy.data.curves.new(name or "Text", "FONT")
    curve.body = body
    curve.size = size
    curve.extrude = extrude
    curve.align_x = align_x
    obj = bpy.data.objects.new(curve.name, curve)
    _link(obj)
    _place(obj, location, rotation_deg, scale)
    return _obj_summary(obj)


@op("add_empty", Obj({
    "name": Opt(NAME),
    "display": Opt(Enum(("PLAIN_AXES", "ARROWS", "SINGLE_ARROW", "CIRCLE", "CUBE", "SPHERE",
                         "CONE")), "PLAIN_AXES"),
    "size": Opt(Float(0.0001, 1e4), 1.0),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True, description="Add an empty (useful as a parent or pivot).")
def add_empty(name, display, size, location, rotation_deg, scale):
    obj = bpy.data.objects.new(name or "Empty", None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    _link(obj)
    _place(obj, location, rotation_deg, scale)
    return _obj_summary(obj)


# =========================================================================== edit objects

def _aim(obj, point):
    direction = Vector(point) - obj.matrix_world.translation
    if direction.length < 1e-9:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


@op("set_transform", Obj({
    "name": NAME, "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
    "look_at": Opt(LOC, None),
}), mutates=True, description="Set location / rotation / scale; look_at aims the object's -Z "
                              "axis (cameras, lights) at a point.")
def set_transform(name, location, rotation_deg, scale, look_at):
    obj = _obj(name)
    _place(obj, location, rotation_deg, scale)
    if look_at is not None:
        bpy.context.view_layer.update()
        _aim(obj, look_at)
    return _obj_summary(obj)


@op("delete_objects", Obj({"names": List(NAME, 1, 500)}), mutates=True, destructive=True,
    description="Delete objects (and their data if nothing else uses it).")
def delete_objects(names):
    objs = [_obj(n) for n in names]
    deleted = []
    for obj in objs:
        data = obj.data
        deleted.append(obj.name)
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            coll = _data_collection(data)
            if coll is not None:
                coll.remove(data)
    return {"deleted": deleted}


def _data_collection(data):
    ids = props.struct_ids(data)
    for ident, coll in (("Mesh", bpy.data.meshes), ("Light", bpy.data.lights),
                        ("Camera", bpy.data.cameras), ("Curve", bpy.data.curves)):
        if ident in ids:
            return coll
    return None


@op("duplicate_object", Obj({"name": NAME, "new_name": Opt(NAME), "linked": Opt(Bool(), False),
                             "location": Opt(LOC)}), mutates=True,
    description="Duplicate an object; linked=true shares the mesh data.")
def duplicate_object(name, new_name, linked, location):
    src = _obj(name)
    obj = src.copy()
    if src.data is not None and not linked:
        obj.data = src.data.copy()
    if new_name:
        obj.name = new_name
    _link(obj)
    if location is not None:
        obj.location = location
    return _obj_summary(obj)


@op("set_parent", Obj({"child": NAME, "parent": Opt(NAME), "keep_transform": Opt(Bool(), True)}),
    mutates=True, description="Parent an object to another (or clear the parent if omitted).")
def set_parent(child, parent, keep_transform):
    c = _obj(child)
    mw = c.matrix_world.copy()
    if parent is None:
        c.parent = None
    else:
        p = _obj(parent)
        node = p
        while node is not None:
            if node == c:
                raise OpError("invalid_params", "that would create a parent loop")
            node = node.parent
        c.parent = p
        c.matrix_parent_inverse = p.matrix_world.inverted() if keep_transform else Matrix()
    if keep_transform:
        c.matrix_world = mw
    return _obj_summary(c)


@op("rename", Obj({"kind": Enum(("object", "material", "mesh", "light", "camera", "curve",
                                 "world")), "name": NAME, "new_name": NAME}), mutates=True,
    description="Rename a datablock. Blender may append .001 if the name is taken.")
def rename(kind, name, new_name):
    idb = _target({"kind": kind, "name": name})
    idb.name = new_name
    return {"name": idb.name}


@op("set_property", Obj({"target": TARGET, "path": PATH, "value": Any()}), mutates=True,
    description="Set one property through an allow-listed RNA path (no file paths, scripts, "
                "drivers or pointers). Use get_property to discover names, ranges and enum items.")
def set_property(target, path, value):
    root = _target(target)
    owner, ids, prop, where = props.walk(root, path)
    if prop is None:
        raise OpError("invalid_path", f"{where}: path must end on a property")
    old = props.to_json(getattr(owner, prop.identifier), prop)
    new = props.set_prop(root, path, value)
    obj = root if isinstance(root, bpy.types.Object) else None
    if obj is not None and path and path[0] == "modifiers":
        try:
            _check_budget(obj)
        except OpError:
            if not getattr(prop, "is_enum_flag", False):
                setattr(owner, prop.identifier, old)
            raise
    return {"path": where, "value": new}


# =========================================================================== modifiers



@op("add_modifier", Obj({
    "object": NAME,
    "type": Enum(ALLOWED_MODIFIERS),
    "name": Opt(NAME),
    "settings": Opt(SETTINGS, None),
    "target_object": Opt(NAME, None),
    "curve_object": Opt(NAME, None),
}), mutates=True, description="Add a modifier and set its properties. target_object sets the "
                              "Boolean/Shrinkwrap target, Mirror/Array offset object or the "
                              "CURVE (Curve Deform) modifier's curve object (must be a CURVE). "
                              "curve_object (ARRAY only, must be a CURVE) sets the curve for "
                              "Array's fit-to-curve and switches fit_type to FIT_CURVE.")
def add_modifier(object, type, name, settings, target_object, curve_object):
    obj = _obj(object)
    if obj.type not in {"MESH", "CURVE", "FONT", "SURFACE", "LATTICE"}:
        raise OpError("invalid_params", f"{obj.name} ({obj.type}) can't take modifiers")
    if len(obj.modifiers) >= 16:
        raise OpError("budget", "at most 16 modifiers per object")
    mod = obj.modifiers.new(name or type.title(), type)
    if mod is None:
        raise OpError("invalid_params", f"{type} isn't available for {obj.type}")
    try:
        if target_object is not None:
            tgt = _obj(target_object)
            if tgt == obj:
                raise OpError("invalid_params", "target_object can't be the object itself")
            attr = {"BOOLEAN": "object", "SHRINKWRAP": "target", "MIRROR": "mirror_object",
                    "ARRAY": "offset_object", "CAST": "object", "SIMPLE_DEFORM": "origin",
                    "CURVE": "object"}.get(type)
            if attr is None:
                raise OpError("invalid_params", f"{type} has no target object")
            if type == "CURVE" and tgt.type != "CURVE":
                raise OpError("invalid_params", f"{tgt.name} is a {tgt.type}; Curve Deform "
                                                "needs a CURVE object")
            setattr(mod, attr, tgt)
            if type == "ARRAY":
                mod.use_object_offset = True
        if curve_object is not None:
            if type != "ARRAY":
                raise OpError("invalid_params", "curve_object only applies to ARRAY")
            cur = _obj(curve_object)
            if cur.type != "CURVE":
                raise OpError("invalid_params", f"{cur.name} is a {cur.type}; Array fit-to-curve "
                                                "needs a CURVE object")
            mod.curve = cur
            mod.fit_type = "FIT_CURVE"
        applied = _apply_settings(mod, settings, mod.name)
        _check_budget(obj)
    except Exception:
        obj.modifiers.remove(mod)
        raise
    return {"object": obj.name, "modifier": mod.name, "type": mod.type, "settings": applied}


@op("remove_modifier", Obj({"object": NAME, "name": NAME}), mutates=True, destructive=True,
    description="Remove a modifier.")
def remove_modifier(object, name):
    obj = _obj(object)
    mod = obj.modifiers.get(name)
    if mod is None:
        raise OpError("not_found", f"{obj.name} has no modifier {name!r}")
    obj.modifiers.remove(mod)
    return {"removed": name}


@op("apply_modifiers", Obj({"object": NAME}), mutates=True, destructive=True,
    description="Bake the whole modifier stack of a mesh object into its mesh.")
def apply_modifiers(object):
    obj = _obj(object)
    if obj.type != "MESH":
        raise OpError("invalid_params", "only mesh objects")
    _check_budget(obj)
    new_mesh = _bake_modifiers(obj)
    return {"object": obj.name, "faces": len(new_mesh.polygons)}


def _bake_modifiers(obj):
    """Replace obj's mesh with its evaluated (modified) mesh and clear the stack."""
    dg = bpy.context.evaluated_depsgraph_get()
    new_mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(dg), preserve_all_data_layers=True,
                                               depsgraph=dg)
    old = obj.data
    obj.modifiers.clear()
    obj.data = new_mesh
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return new_mesh


@op("join_objects", Obj({
    "target": NAME,
    "others": List(NAME, 1, 100, description="Mesh objects merged into target and removed"),
    "apply_modifiers": Opt(Bool(description="Bake the others' modifiers first so they keep "
                                            "their look (their stacks are lost otherwise)"), True),
    "merge_distance": Opt(Float(0.0, 1.0, description="Afterwards, weld vertices closer than "
                                                      "this (0/omitted = don't)")),
}), mutates=True, destructive=True, description="Join mesh objects into one (like Ctrl+J): the others' geometry, "
                              "UVs and materials move into target, keeping their world placement; "
                              "the others are removed. Target's own modifiers then apply to the "
                              "whole result. For a watertight solid from overlapping/touching "
                              "shapes, use a BOOLEAN modifier (UNION) + apply_modifiers instead.")
def join_objects(target, others, apply_modifiers, merge_distance):
    tgt = _obj(target)
    objs = [_obj(n) for n in others]
    if len({o.name for o in objs}) != len(objs) or tgt in objs:
        raise OpError("invalid_params", "objects must be distinct and not include the target")
    view_objects = bpy.context.view_layer.objects
    for o in (tgt, *objs):
        if o.type != "MESH":
            raise OpError("invalid_params", f"{o.name} is a {o.type}; only meshes can be joined")
        if view_objects.get(o.name) is None:
            raise OpError("invalid_params", f"{o.name} is not in the current view layer")
        if o.mode != "OBJECT":
            raise OpError("unavailable", f"{o.name} is in {o.mode} mode; ask the user to return "
                                         "to Object mode")
    est = sum(_estimate_polys(o) for o in (tgt, *objs))
    if est > MAX_POLYS:
        raise OpError("budget", f"joined mesh would have ~{est:,} faces (limit {MAX_POLYS:,})")

    if apply_modifiers:
        for o in objs:
            if len(o.modifiers):
                _bake_modifiers(o)
    if tgt.data.users > 1:
        tgt.data = tgt.data.copy()  # don't grow every linked duplicate of the target

    selected = [tgt, *objs]
    override = dict(_window_override() or {})
    override.update(active_object=tgt, object=tgt, selected_objects=selected,
                    selected_editable_objects=selected)
    joined = [o.name for o in objs]
    with bpy.context.temp_override(**override):
        if bpy.ops.object.join() != {"FINISHED"}:
            raise OpError("failed", "Blender refused the join (see get_debug_logs)")

    welded = 0
    if merge_distance:
        before = len(tgt.data.vertices)
        bm = bmesh.new()
        try:
            bm.from_mesh(tgt.data)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
            bm.to_mesh(tgt.data)
        finally:
            bm.free()
        tgt.data.update()
        welded = before - len(tgt.data.vertices)

    me = tgt.data
    result = _obj_summary(tgt)
    result.update({
        "joined": joined,
        "mesh": {"vertices": len(me.vertices), "faces": len(me.polygons)},
        "welded_vertices": welded,
        "materials": [m.name if m else None for m in me.materials],
    })
    if len(tgt.modifiers):
        result["note"] = ("target modifiers now apply to the joined geometry: "
                          + ", ".join(m.name for m in tgt.modifiers))
    return result


# =========================================================================== mesh quality

MAX_ANALYZE_FACES = 2_000_000
MAX_SELF_INTERSECT_FACES = 300_000


def _mesh_bm(obj, evaluated=False):
    if obj.type != "MESH":
        raise OpError("invalid_params", f"{obj.name} is a {obj.type}, not a mesh")
    bm = bmesh.new()
    if evaluated:
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        try:
            bm.from_mesh(me)
        finally:
            ev.to_mesh_clear()
    elif obj.mode == "EDIT":
        bm.free()
        bm = bmesh.from_edit_mesh(obj.data).copy()  # read the live edit-mode state
    else:
        bm.from_mesh(obj.data)
    if len(bm.faces) > MAX_ANALYZE_FACES:
        bm.free()
        raise OpError("budget", f"{obj.name} has more than {MAX_ANALYZE_FACES:,} faces")
    return bm


def _components(n, pairs):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return find


def _analyze_bm(bm, obj, max_holes=20):
    """Topology report. A closed 2-manifold has: no boundary edges, every edge
    shared by exactly 2 faces, no bowtie vertices, no wire/loose elements and
    consistent winding. Self-intersections are reported separately: a surface can
    be manifold and still pass through itself."""
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for i, v in enumerate(bm.verts):
        v.index = i
    mw = obj.matrix_world

    boundary = [e for e in bm.edges if len(e.link_faces) == 1]
    nonmanifold = [e for e in bm.edges if len(e.link_faces) > 2]
    wire = [e for e in bm.edges if not e.link_faces]
    loose = [v for v in bm.verts if not v.link_edges]
    bowtie = [v for v in bm.verts if v.link_faces and not v.is_manifold and not v.is_boundary]
    flipped = sum(1 for e in bm.edges if len(e.link_loops) == 2
                  and e.link_loops[0].vert == e.link_loops[1].vert)
    degenerate = sum(1 for f in bm.faces if f.calc_area() < 1e-12)

    # holes = connected loops of boundary edges
    holes = []
    if boundary:
        find = _components(len(bm.verts), ((e.verts[0].index, e.verts[1].index) for e in boundary))
        groups = {}
        for e in boundary:
            groups.setdefault(find(e.verts[0].index), []).append(e)
        for edges in sorted(groups.values(), key=len, reverse=True):
            verts = {v for e in edges for v in e.verts}
            c = sum((v.co for v in verts), Vector()) / len(verts)
            holes.append({"edges": len(edges),
                          "perimeter": round(sum(e.calc_length() for e in edges), 4),
                          "center_world": _vec(mw @ c)})

    # shells = groups of faces connected through edges
    find = _components(len(bm.verts), ((e.verts[0].index, e.verts[1].index)
                                       for e in bm.edges if e.link_faces))
    shells = len({find(f.verts[0].index) for f in bm.faces})

    report = {
        "vertices": len(bm.verts), "edges": len(bm.edges), "faces": len(bm.faces),
        "shells": shells,
        "holes": len(holes),
        "boundary_edges": len(boundary),
        "non_manifold_edges": len(nonmanifold),
        "bowtie_vertices": len(bowtie),
        "wire_edges": len(wire),
        "loose_vertices": len(loose),
        "inconsistent_normal_edges": flipped,
        "degenerate_faces": degenerate,
        "euler_characteristic": len(bm.verts) - len(bm.edges) + len(bm.faces),
    }
    report["is_manifold"] = not (boundary or nonmanifold or bowtie or wire or loose or flipped)
    if holes:
        report["hole_list"] = holes[:max_holes]
    if report["is_manifold"]:
        report["volume_world"] = round(bm.calc_volume(signed=True) * mw.determinant(), 6)
        if report["volume_world"] < 0:
            report["note"] = "normals point inward (negative volume)"

    if len(bm.faces) <= MAX_SELF_INTERSECT_FACES:
        from mathutils.bvhtree import BVHTree
        tree = BVHTree.FromBMesh(bm)
        faces = bm.faces
        count = 0
        for a, b in tree.overlap(tree):
            if a < b and not ({v.index for v in faces[a].verts} & {v.index for v in faces[b].verts}):
                count += 1
        report["self_intersecting_face_pairs"] = count
    else:
        report["self_intersecting_face_pairs"] = None
    return report


@op("analyze_mesh", Obj({
    "object": NAME,
    "evaluated": Opt(Bool(description="Analyze the result of the modifier stack "
                                      "instead of the base mesh"), False),
}), description="Check mesh topology: holes (with locations), non-manifold edges, bowtie "
                "vertices, loose geometry, inconsistent normals, shells, degenerate faces, "
                "self-intersections, and volume when closed. is_manifold is true only for a "
                "closed, consistently oriented surface.")
def analyze_mesh(object, evaluated):
    obj = _obj(object)
    bm = _mesh_bm(obj, evaluated)
    try:
        report = _analyze_bm(bm, obj)
    finally:
        bm.free()
    report["object"] = obj.name
    report["mode"] = obj.mode
    return report


@op("repair_mesh", Obj({
    "object": NAME,
    "merge_distance": Opt(Float(0.0, 0.1, description="Weld vertices closer than this "
                                                      "(0 = don't)"), 0.0001),
    "dissolve_degenerate": Opt(Bool(), True),
    "remove_loose": Opt(Bool(description="Delete vertices/edges that belong to no face"), True),
    "fill_holes_max_sides": Opt(Int(0, 100000, description="Cap holes bounded by up to this "
                                                            "many edges with a new face "
                                                            "(0 = don't fill)"), 0),
    "recalculate_normals": Opt(Bool(description="Make face winding consistent, "
                                                "pointing outward"), True),
}), mutates=True, description="Clean up a mesh and report before/after analyze_mesh results. "
                              "Filling caps each hole with one n-gon: it closes the surface, "
                              "but can't recreate missing detail.")
def repair_mesh(object, merge_distance, dissolve_degenerate, remove_loose,
                fill_holes_max_sides, recalculate_normals):
    obj = _obj(object)
    if obj.mode != "OBJECT":
        raise OpError("unavailable", f"{obj.name} is in {obj.mode} mode; use set_object_mode first")
    bm = _mesh_bm(obj)
    actions = {}
    try:
        before = _analyze_bm(bm, obj)
        if merge_distance:
            n = len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=merge_distance)
            actions["welded_vertices"] = n - len(bm.verts)
        if dissolve_degenerate:
            n = len(bm.faces)
            bmesh.ops.dissolve_degenerate(bm, dist=max(merge_distance, 1e-6), edges=list(bm.edges))
            actions["degenerate_faces_removed"] = n - len(bm.faces)
        if remove_loose:
            wire = [e for e in bm.edges if not e.link_faces]
            if wire:
                bmesh.ops.delete(bm, geom=wire, context="EDGES")
            loose = [v for v in bm.verts if not v.link_edges]
            if loose:
                bmesh.ops.delete(bm, geom=loose, context="VERTS")
            actions["loose_removed"] = {"edges": len(wire), "vertices": len(loose)}
        if fill_holes_max_sides:
            boundary = [e for e in bm.edges if len(e.link_faces) == 1]
            res = bmesh.ops.holes_fill(bm, edges=boundary, sides=fill_holes_max_sides)
            actions["holes_filled"] = len(res["faces"])
        if recalculate_normals:
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            actions["normals_recalculated"] = True
        if obj.data.users > 1:
            obj.data = obj.data.copy()  # don't silently change linked duplicates
        bm.to_mesh(obj.data)
        obj.data.update()
        after = _analyze_bm(bm, obj)
    finally:
        bm.free()
    return {"object": obj.name, "actions": actions, "before": before, "after": after}


# ----------------------------------------------------------------- vertex-level editing
#
# Vertices are addressed by index into the object's mesh (the order Blender keeps in
# Object mode). Indices change after any topology edit, so every op returns a fresh
# analysis and callers must re-query before the next edit.

MAX_VERTEX_INDEX = 10_000_000
VIDX = Int(0, MAX_VERTEX_INDEX, description="Vertex index (changes after topology edits)")


def _boundary_edges(bm):
    return [e for e in bm.edges if len(e.link_faces) == 1]


def _loop_of(edges_by_vert, start_vert):
    """Edges of the boundary component containing start_vert."""
    seen_v, seen_e, stack = {start_vert}, set(), [start_vert]
    while stack:
        v = stack.pop()
        for e in edges_by_vert.get(v, ()):
            if e not in seen_e:
                seen_e.add(e)
                o = e.other_vert(v)
                if o not in seen_v:
                    seen_v.add(o)
                    stack.append(o)
    return seen_e, seen_v


def _ordered_loop(edges, verts):
    """Walk a boundary component in order (best effort if it isn't a simple loop)."""
    adj = {}
    for e in edges:
        a, b = e.verts
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = min(verts, key=lambda v: v.index)
    order, prev, cur = [start], None, start
    while True:
        nxt = [n for n in adj[cur] if n is not prev]
        if not nxt or nxt[0] is start or len(order) > len(verts):
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    simple = len(order) == len(verts) and all(len(adj[v]) == 2 for v in verts)
    return order, simple


class _MeshEdit:
    """Load an object's mesh into bmesh, let the op edit it, write it back."""

    def __init__(self, obj):
        if obj.type != "MESH":
            raise OpError("invalid_params", f"{obj.name} is a {obj.type}, not a mesh")
        if obj.mode != "OBJECT":
            raise OpError("unavailable", f"{obj.name} is in {obj.mode} mode; use set_object_mode")
        self.obj = obj

    def __enter__(self):
        self.bm = _mesh_bm(self.obj)
        self.bm.verts.ensure_lookup_table()
        return self.bm

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                if self.obj.data.users > 1:
                    self.obj.data = self.obj.data.copy()
                self.bm.to_mesh(self.obj.data)
                self.obj.data.update()
                self.report = _analyze_bm(self.bm, self.obj, max_holes=10)
        finally:
            self.bm.free()
        return False


def _verts(bm, indices, where="vertices"):
    n = len(bm.verts)
    bad = [i for i in indices if i >= n]
    if bad:
        raise OpError("not_found", f"{where}: no vertex {bad[0]} (mesh has {n}); re-query indices")
    return [bm.verts[i] for i in indices]


def _summary(report):
    keys = ("vertices", "faces", "shells", "holes", "boundary_edges", "non_manifold_edges",
            "bowtie_vertices", "inconsistent_normal_edges", "is_manifold",
            "self_intersecting_face_pairs", "volume_world")
    return {k: report[k] for k in keys if k in report}


@op("get_boundary_loops", Obj({
    "object": NAME,
    "max_loops": Opt(Int(1, 200), 50),
    "max_vertices_per_loop": Opt(Int(3, 2000), 200),
}), description="List every hole (open boundary) as an ordered loop of vertex indices with "
                "world positions. Use it to see where the mesh is open before stitching.")
def get_boundary_loops(object, max_loops, max_vertices_per_loop):
    obj = _obj(object)
    bm = _mesh_bm(obj)
    try:
        bm.verts.ensure_lookup_table()
        for i, v in enumerate(bm.verts):
            v.index = i
        by_vert = {}
        for e in _boundary_edges(bm):
            for v in e.verts:
                by_vert.setdefault(v, []).append(e)
        loops, done = [], set()
        mw = obj.matrix_world
        for v in sorted(by_vert, key=lambda v: v.index):
            if v in done:
                continue
            edges, verts = _loop_of(by_vert, v)
            done |= verts
            order, simple = _ordered_loop(edges, verts)
            loops.append({
                "vertex_count": len(verts), "simple_loop": simple,
                "perimeter": round(sum(e.calc_length() for e in edges), 4),
                "vertices": [{"i": u.index, "co": _vec(mw @ u.co)} for u in order[:max_vertices_per_loop]],
            })
        loops.sort(key=lambda l: -l["vertex_count"])
        return {"object": obj.name, "loops": len(loops), "loop_list": loops[:max_loops]}
    finally:
        bm.free()


@op("find_close_vertices", Obj({
    "object": NAME,
    "distance": Float(0.0, 1.0, description="World-space search radius"),
    "boundary_only": Opt(Bool(), True),
    "max_pairs": Opt(Int(1, 5000), 500),
}), description="Pairs of vertices closer than `distance` that aren't already joined by an edge "
                "(by default only open-boundary vertices): candidates for merge_vertices.")
def find_close_vertices(object, distance, boundary_only, max_pairs):
    from mathutils.kdtree import KDTree
    obj = _obj(object)
    bm = _mesh_bm(obj)
    try:
        bm.verts.ensure_lookup_table()
        mw = obj.matrix_world
        cand = ([v for v in bm.verts if v.is_boundary] if boundary_only else list(bm.verts))
        kd = KDTree(len(cand))
        for k, v in enumerate(cand):
            kd.insert(mw @ v.co, k)
        kd.balance()
        pairs = []
        for a, v in enumerate(cand):
            for _co, b, d in kd.find_range(mw @ v.co, distance):
                if b <= a:
                    continue
                u = cand[b]
                if any(e.other_vert(v) is u for e in v.link_edges):
                    continue
                pairs.append({"a": v.index, "b": u.index, "distance": round(d, 6),
                              "a_co": _vec(mw @ v.co), "b_co": _vec(mw @ u.co)})
        pairs.sort(key=lambda p: p["distance"])
        return {"object": obj.name, "pairs_found": len(pairs), "pairs": pairs[:max_pairs]}
    finally:
        bm.free()


@op("merge_vertices", Obj({
    "object": NAME,
    "groups": List(List(VIDX, 2, 32), 1, 5000,
                   description="Each group of vertex indices is merged into one vertex"),
    "position": Opt(Enum(("center", "first")), "center"),
}), mutates=True, description="Merge chosen vertex groups (e.g. the two sides of a crack) into "
                              "single vertices. Indices refer to the mesh before this call.")
def merge_vertices(object, groups, position):
    obj = _obj(object)
    flat = [i for g in groups for i in g]
    if len(flat) != len(set(flat)):
        raise OpError("invalid_params", "a vertex appears in more than one group")
    ed = _MeshEdit(obj)
    with ed as bm:
        resolved = [_verts(bm, g, "groups") for g in groups]
        for vs in resolved:
            co = (sum((v.co for v in vs), Vector()) / len(vs)) if position == "center" else vs[0].co.copy()
            bmesh.ops.pointmerge(bm, verts=vs, merge_co=co)
    return {"object": obj.name, "merged_groups": len(groups),
            "removed_vertices": len(flat) - len(groups), "after": _summary(ed.report)}


@op("fill_hole", Obj({
    "object": NAME,
    "vertex": VIDX,
    "method": Opt(Enum(("triangles", "ngon")), "triangles"),
}), mutates=True, description="Fill the single hole whose boundary passes through `vertex`: "
                              "one n-gon, or triangulated (better for non-flat holes).")
def fill_hole(object, vertex, method):
    obj = _obj(object)
    ed = _MeshEdit(obj)
    with ed as bm:
        (v,) = _verts(bm, [vertex], "vertex")
        if not v.is_boundary:
            raise OpError("invalid_params", f"vertex {vertex} is not on a hole boundary")
        by_vert = {}
        for e in _boundary_edges(bm):
            for u in e.verts:
                by_vert.setdefault(u, []).append(e)
        edges, verts = _loop_of(by_vert, v)
        res = bmesh.ops.holes_fill(bm, edges=list(edges), sides=len(edges))
        faces = res["faces"]
        if not faces:
            raise OpError("failed", "Blender could not fill that hole (not a closed loop?)")
        if method == "triangles":
            tri = bmesh.ops.triangulate(bm, faces=faces, quad_method="BEAUTY", ngon_method="BEAUTY")
            faces = tri["faces"]
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
        filled = {"loop_vertices": len(verts), "new_faces": len(faces)}
    return {"object": obj.name, "filled": filled, "after": _summary(ed.report)}


@op("create_face", Obj({
    "object": NAME,
    "vertices": List(VIDX, 3, 64, description="Vertex indices in order around the new face"),
}), mutates=True, description="Create one face from existing vertices (to close a small gap by "
                              "hand). Refuses if it would make an edge shared by 3+ faces.")
def create_face(object, vertices):
    obj = _obj(object)
    if len(set(vertices)) != len(vertices):
        raise OpError("invalid_params", "repeated vertex")
    ed = _MeshEdit(obj)
    with ed as bm:
        vs = _verts(bm, vertices)
        for a, b in zip(vs, vs[1:] + vs[:1]):
            e = bm.edges.get((a, b))
            if e is not None and len(e.link_faces) >= 2:
                raise OpError("invalid_params", f"edge {a.index}-{b.index} already has two faces; "
                                                "a third would make it non-manifold")
        if bm.faces.get(vs) is not None:
            raise OpError("invalid_params", "that face already exists")
        bm.faces.new(vs)
        bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    return {"object": obj.name, "after": _summary(ed.report)}


@op("remove_duplicate_faces", Obj({"object": NAME}), mutates=True,
    description="Where several faces use exactly the same vertices (e.g. two overlapping "
                "copies of a region after merge_vertices), keep one and delete the rest; "
                "then delete edges/vertices left without faces.")
def remove_duplicate_faces(object):
    obj = _obj(object)
    ed = _MeshEdit(obj)
    with ed as bm:
        seen, dupes = {}, []
        for f in bm.faces:
            key = frozenset(v.index for v in f.verts)
            if key in seen:
                dupes.append(f)
            else:
                seen[key] = f
        if dupes:
            bmesh.ops.delete(bm, geom=dupes, context="FACES_ONLY")
        wire = [e for e in bm.edges if not e.link_faces]
        if wire:
            bmesh.ops.delete(bm, geom=wire, context="EDGES")
        loose = [v for v in bm.verts if not v.link_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
        removed = {"faces": len(dupes), "edges": len(wire), "vertices": len(loose)}
    return {"object": obj.name, "removed": removed, "after": _summary(ed.report)}


@op("delete_faces", Obj({
    "object": NAME,
    "faces": List(List(VIDX, 3, 64), 1, 5000,
                  description="Each face given by its vertex indices (any order)"),
}), mutates=True, description="Delete specific faces, identified by their vertices; edges and "
                              "vertices left without faces are removed too.")
def delete_faces(object, faces):
    obj = _obj(object)
    ed = _MeshEdit(obj)
    with ed as bm:
        targets = []
        for idx in faces:
            f = bm.faces.get(_verts(bm, idx, "faces"))
            if f is None:
                raise OpError("not_found", f"no face with vertices {idx}")
            targets.append(f)
        bmesh.ops.delete(bm, geom=list(set(targets)), context="FACES_ONLY")
        wire = [e for e in bm.edges if not e.link_faces]
        if wire:
            bmesh.ops.delete(bm, geom=wire, context="EDGES")
        loose = [v for v in bm.verts if not v.link_edges]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")
    return {"object": obj.name, "deleted_faces": len(set(targets)), "after": _summary(ed.report)}


@op("set_object_mode", Obj({"object": NAME, "mode": Enum(("OBJECT", "EDIT"))}), mutates=True,
    description="Switch an object between Object and Edit mode (leaving Edit mode commits "
                "the edit-mode changes to the mesh).")
def set_object_mode(object, mode):
    obj = _obj(object)
    if obj.mode == mode:
        return {"object": obj.name, "mode": obj.mode, "changed": False}
    if bpy.context.view_layer.objects.get(obj.name) is None:
        raise OpError("invalid_params", f"{obj.name} is not in the current view layer")
    if mode == "EDIT" and obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META",
                                           "LATTICE", "ARMATURE"}:
        raise OpError("invalid_params", f"{obj.type} objects have no Edit mode")
    override = dict(_window_override() or {})
    override.update(active_object=obj, object=obj, selected_objects=[obj],
                    selected_editable_objects=[obj], edit_object=obj if obj.mode == "EDIT" else None)
    prev_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = obj
    try:
        with bpy.context.temp_override(**override):
            if bpy.ops.object.mode_set(mode=mode) != {"FINISHED"}:
                raise OpError("failed", "Blender refused the mode change")
    finally:
        if mode == "OBJECT" and prev_active is not None and prev_active != obj:
            bpy.context.view_layer.objects.active = prev_active
    return {"object": obj.name, "mode": obj.mode, "changed": True}


# =========================================================================== materials

PRINCIPLED_INPUTS = {
    "base_color": ("Base Color",), "metallic": ("Metallic",), "roughness": ("Roughness",),
    "ior": ("IOR",), "alpha": ("Alpha",),
    "transmission": ("Transmission Weight", "Transmission"),
    "emission_color": ("Emission Color", "Emission"),
    "emission_strength": ("Emission Strength",),
    "coat": ("Coat Weight", "Clearcoat"), "sheen": ("Sheen Weight", "Sheen"),
    "subsurface": ("Subsurface Weight", "Subsurface"),
    "specular": ("Specular IOR Level", "Specular"),
}


def _set_input(node, names, value):
    for n in names:
        sock = node.inputs.get(n)
        if sock is not None:
            if isinstance(value, list) and len(value) == 3 and sock.type == "RGBA":
                value = list(value) + [1.0]
            sock.default_value = value
            return True
    return False


@op("create_material", Obj({
    "name": NAME,
    "base_color": Opt(Color()), "metallic": Opt(Float(0, 1)), "roughness": Opt(Float(0, 1)),
    "ior": Opt(Float(1, 4)), "alpha": Opt(Float(0, 1)), "transmission": Opt(Float(0, 1)),
    "emission_color": Opt(Color()), "emission_strength": Opt(Float(0, 1e5)),
    "coat": Opt(Float(0, 1)), "sheen": Opt(Float(0, 1)), "subsurface": Opt(Float(0, 1)),
    "specular": Opt(Float(0, 2)),
    "assign_to": Opt(List(NAME, 0, 200), []),
}), mutates=True, description="Create a Principled BSDF material (or update it if it exists) "
                              "and optionally assign it to objects.")
def create_material(name, assign_to, **inputs):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    tree = _node_tree({"kind": "material", "name": mat.name})
    bsdf = _find_node_by_type(tree, "BSDF_PRINCIPLED")
    if bsdf is None:
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        out = _find_node_by_type(tree, "OUTPUT_MATERIAL") or tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(bsdf.outputs[0], out.inputs[0])
    missing = []
    for key, value in inputs.items():
        if value is None:
            continue
        if not _set_input(bsdf, PRINCIPLED_INPUTS[key], value):
            missing.append(key)
    if inputs.get("base_color"):
        mat.diffuse_color = list(inputs["base_color"]) + [1.0]
    if inputs.get("alpha") is not None and inputs["alpha"] < 1:
        for attr, val in (("blend_method", "BLEND"), ("surface_render_method", "BLENDED")):
            try:
                setattr(mat, attr, val)
            except (AttributeError, TypeError):
                pass
    for obj_name in assign_to:
        _assign(_obj(obj_name), mat, None)
    return {"material": mat.name, "unsupported_inputs": missing, "assigned_to": assign_to}


def _assign(obj, mat, slot):
    if obj.data is None or not hasattr(obj.data, "materials"):
        raise OpError("invalid_params", f"{obj.name} can't hold materials")
    mats = obj.data.materials
    if slot is None:
        if len(mats) == 0:
            mats.append(mat)
        else:
            for i in range(len(mats)):
                mats[i] = mat
    else:
        while len(mats) <= slot:
            mats.append(None)
        mats[slot] = mat


@op("assign_material", Obj({"object": NAME, "material": NAME, "slot": Opt(Int(0, 63))}),
    mutates=True, description="Assign a material to an object (all slots, or one slot).")
def assign_material(object, material, slot):
    obj = _obj(object)
    _assign(obj, _material(material), slot)
    return {"object": obj.name, "materials": [m.name if m else None for m in obj.data.materials]}


@op("set_world", Obj({"color": Opt(Color()), "strength": Opt(Float(0, 1e4))}), mutates=True,
    description="Set the world background color/strength (creates a world if needed).")
def set_world(color, strength):
    sc = _scene()
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    tree = _node_tree({"kind": "world", "name": sc.world.name})
    bg = _find_node_by_type(tree, "BACKGROUND")
    if bg is None:
        bg = tree.nodes.new("ShaderNodeBackground")
        out = _find_node_by_type(tree, "OUTPUT_WORLD") or tree.nodes.new("ShaderNodeOutputWorld")
        tree.links.new(bg.outputs[0], out.inputs[0])
    if color is not None:
        bg.inputs["Color"].default_value = list(color) + [1.0]
        sc.world.color = color
    if strength is not None:
        bg.inputs["Strength"].default_value = strength
    return {"world": sc.world.name}


# =========================================================================== shader nodes

SOCKET = Union(Str(max_len=63), Int(0, 255), description="Socket identifier, name or index")


@op("get_shader_graph", Obj({"owner": NODE_OWNER}),
    description="Nodes (type, inputs with values, outputs) and links of a material/world.")
def get_shader_graph(owner):
    tree = _node_tree(owner)
    nodes = []
    for n in tree.nodes:
        inputs = []
        for s in n.inputs:
            if not getattr(s, "enabled", True):
                continue
            d = {"id": s.identifier, "name": s.name, "type": s.type, "linked": s.is_linked}
            if hasattr(s, "default_value") and not s.is_linked:
                d["value"] = props.to_json(s.default_value)
            inputs.append(d)
        nodes.append({
            "name": n.name, "type": n.bl_idname, "label": n.label,
            "location": _vec(n.location),
            "inputs": inputs,
            "outputs": [{"id": s.identifier, "name": s.name, "type": s.type}
                        for s in n.outputs if getattr(s, "enabled", True)],
        })
    links = [{"from_node": l.from_node.name, "from_socket": l.from_socket.identifier,
              "to_node": l.to_node.name, "to_socket": l.to_socket.identifier}
             for l in tree.links]
    return {"nodes": nodes, "links": links}


@op("add_shader_node", Obj({
    "owner": NODE_OWNER,
    "type": Enum(ALLOWED_SHADER_NODES),
    "name": Opt(NAME),
    "location": Opt(Vec(2, -1e5, 1e5)),
    "settings": Opt(SETTINGS, None),
}), mutates=True, description="Add a shader node; settings are node properties "
                              "(e.g. {\"operation\": \"MULTIPLY\"}). Set input values with set_node_input.")
def add_shader_node(owner, type, name, location, settings):
    tree = _node_tree(owner)
    if len(tree.nodes) >= 500:
        raise OpError("budget", "at most 500 nodes per tree")
    node = tree.nodes.new(type)
    try:
        if name:
            node.name = name
        if location:
            node.location = location
        _apply_settings(node, settings, node.name)
    except Exception:
        tree.nodes.remove(node)
        raise
    return {"name": node.name, "type": node.bl_idname,
            "inputs": [s.identifier for s in node.inputs if getattr(s, "enabled", True)],
            "outputs": [s.identifier for s in node.outputs if getattr(s, "enabled", True)]}


@op("link_shader_nodes", Obj({
    "owner": NODE_OWNER, "from_node": NAME, "from_socket": SOCKET,
    "to_node": NAME, "to_socket": SOCKET,
}), mutates=True, description="Connect an output socket to an input socket.")
def link_shader_nodes(owner, from_node, from_socket, to_node, to_socket):
    tree = _node_tree(owner)
    a, b = _node(tree, from_node), _node(tree, to_node)
    out = _socket(a.outputs, from_socket, a.name)
    inp = _socket(b.inputs, to_socket, b.name)
    link = tree.links.new(out, inp)
    return {"linked": [a.name, out.identifier, b.name, inp.identifier],
            "valid": getattr(link, "is_valid", True)}


@op("unlink_shader_socket", Obj({"owner": NODE_OWNER, "node": NAME, "socket": SOCKET}),
    mutates=True, description="Remove links going into an input socket.")
def unlink_shader_socket(owner, node, socket):
    tree = _node_tree(owner)
    n = _node(tree, node)
    s = _socket(n.inputs, socket, n.name)
    removed = 0
    for l in list(s.links):
        tree.links.remove(l)
        removed += 1
    return {"removed": removed}


@op("remove_shader_node", Obj({"owner": NODE_OWNER, "node": NAME}), mutates=True, destructive=True,
    description="Delete a node.")
def remove_shader_node(owner, node):
    tree = _node_tree(owner)
    tree.nodes.remove(_node(tree, node))
    return {"removed": node}


@op("set_node_input", Obj({"owner": NODE_OWNER, "node": NAME, "socket": SOCKET, "value": Any()}),
    mutates=True, description="Set an unlinked input socket's value (colors are RGBA).")
def set_node_input(owner, node, socket, value):
    tree = _node_tree(owner)
    n = _node(tree, node)
    s = _socket(n.inputs, socket, n.name)
    if not hasattr(s, "default_value"):
        raise OpError("invalid_params", f"{s.identifier} has no value (shader/closure socket)")
    if isinstance(value, list) and len(value) == 3 and s.type == "RGBA":
        value = value + [1.0]
    return {"value": props.set_prop(s, ["default_value"], value)}


# =========================================================================== animation

KEYABLE = ("location", "rotation_euler", "scale", "hide_viewport", "hide_render")


@op("insert_keyframe", Obj({
    "object": NAME, "property": Enum(KEYABLE), "frame": Int(-100000, 100000),
    "value": Opt(Union(Vec(3, -1e6, 1e6), Bool())),
}), mutates=True, description="Keyframe an object's transform (rotation in degrees) or visibility.")
def insert_keyframe(object, property, frame, value):
    obj = _obj(object)
    if value is not None:
        if property in ("hide_viewport", "hide_render"):
            if not isinstance(value, bool):
                raise OpError("invalid_params", "visibility value must be true/false")
        elif isinstance(value, bool):
            raise OpError("invalid_params", "transform value must be [x, y, z]")
        elif property == "rotation_euler":
            value = [math.radians(a) for a in value]
        setattr(obj, property, value)
    obj.keyframe_insert(data_path=property, frame=frame)
    return {"object": obj.name, "property": property, "frame": frame}


@op("set_frame", Obj({"frame": Int(-100000, 100000)}), mutates=True,
    description="Jump to a frame.")
def set_frame(frame):
    _scene().frame_set(frame)
    return {"frame": _scene().frame_current}


@op("set_active_camera", Obj({"name": NAME}), mutates=True, description="Make a camera the scene camera.")
def set_active_camera(name):
    obj = _obj(name)
    if obj.type != "CAMERA":
        raise OpError("invalid_params", f"{name} is not a camera")
    _scene().camera = obj
    return {"camera": obj.name}


# =========================================================================== history

def _window_override():
    wm = bpy.context.window_manager
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                return {"window": win, "screen": win.screen, "area": area}
    if wm.windows:
        return {"window": wm.windows[0], "screen": wm.windows[0].screen}
    return None


def undo_push(message):
    """Record one undo step. Best effort, but never silent: a step that isn't recorded
    means the next undo reverts more (or something else) than this op."""
    ov = _window_override()
    if ov is None:
        log.warning("no undo step for %r: no Blender window to record it in", message)
        return
    try:
        with bpy.context.temp_override(**ov):
            ret = bpy.ops.ed.undo_push(message=message[:60])
    except Exception as e:  # noqa: BLE001 - undo is best effort
        log.warning("no undo step for %r: %s: %s", message, type(e).__name__, e)
        return
    if ret != {"FINISHED"}:
        log.warning("no undo step for %r: Blender returned %s", message, sorted(ret))


def _objects_digest():
    """Plain-value snapshot of every object, to report what an undo/redo changed. Holds no
    bpy references: undo reloads data, and stale references would crash Blender."""
    out = {}
    for o in bpy.data.objects:
        data = o.data
        mesh = (len(data.vertices), len(data.polygons)) if o.type == "MESH" and data else None
        out[o.name] = (o.type, tuple(round(v, 6) for row in o.matrix_world for v in row),
                       data.name if data else None, mesh,
                       tuple(m.name if m else None for m in getattr(data, "materials", ()) or ()),
                       tuple((m.name, m.type) for m in o.modifiers))
    return out


def _history_step(kind, steps):
    ov = _window_override()
    if ov is None:
        raise OpError("unavailable", f"{kind} needs a Blender window")
    call = bpy.ops.ed.undo if kind == "undo" else bpy.ops.ed.redo
    before = _objects_digest()
    done = 0
    with bpy.context.temp_override(**ov):
        for _ in range(steps):
            try:
                ret = call()
            except RuntimeError as e:  # poll() fails when there's no step left
                ret = {f"poll failed: {e}"}
            if ret != {"FINISHED"}:
                log.info("%s stopped after %d of %d steps: Blender returned %s",
                         kind, done, steps, sorted(ret))
                break
            done += 1
    if done == 0:
        raise OpError("nothing_to_" + kind,
                      f"Blender has no step to {kind} (or refused it); nothing changed")
    after = _objects_digest()
    changed = sorted(n for n in before.keys() & after.keys() if before[n] != after[n])
    result = {
        "steps_done": done, "steps_requested": steps,
        "objects_added": sorted(after.keys() - before.keys())[:50],
        "objects_removed": sorted(before.keys() - after.keys())[:50],
        "objects_changed": changed[:50],
    }
    if not (result["objects_added"] or result["objects_removed"] or changed):
        result["note"] = (f"the {kind}ne step(s) changed nothing this report tracks (objects, "
                          "transforms, mesh sizes, material slots, modifiers): e.g. a material "
                          "colour or property, or the user's own edits in Blender (a selection "
                          "is an undo step too). Check the scene before undoing further.")
    return result


@op("undo", Obj({"steps": Opt(Int(1, 50), 1)}), mutates=True, destructive=True,
    description="Undo the last steps of Blender's undo history. Each MCP change is one step, "
                "but so is each edit the user makes in Blender (even a selection). Returns the "
                "steps actually undone and which objects were added, removed or changed.")
def undo(steps):
    return _history_step("undo", steps)


@op("redo", Obj({"steps": Opt(Int(1, 50), 1)}), mutates=True, destructive=True,
    description="Redo undone steps. Returns the steps actually redone and which objects "
                "were added, removed or changed. Any new change discards the redo history.")
def redo(steps):
    return _history_step("redo", steps)


# =========================================================================== images

def _view3d():
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == "VIEW_3D":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is not None:
                    return area.spaces.active, region
    raise OpError("unavailable", "no 3D viewport open")


def _view_state(r3d):
    q = r3d.view_rotation
    eye = r3d.view_location + q @ Vector((0.0, 0.0, r3d.view_distance))
    fwd = q @ Vector((0.0, 0.0, -1.0))
    return {
        "perspective": r3d.view_perspective,
        "center": _vec(r3d.view_location),
        "distance": round(r3d.view_distance, 6),
        "eye": _vec(eye),
        "azimuth_deg": round(math.degrees(math.atan2(-fwd.y, -fwd.x)), 3),
        "elevation_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, -fwd.z)))), 3),
    }


@op("set_viewport", Obj({
    "orbit_right_deg": Opt(Float(-360, 360, description="Orbit the view to the right around "
                                                        "its center (world Z axis)")),
    "orbit_up_deg": Opt(Float(-180, 180, description="Orbit the view upward (see more from above)")),
    "center": Opt(LOC),
    "distance": Opt(Float(0.001, 1e6)),
    "perspective": Opt(Enum(("PERSP", "ORTHO", "CAMERA"))),
}), description="Move the user's 3D viewport (not the scene camera): orbit, recenter, zoom, or "
                "switch perspective. Returns the resulting view. Changes no scene data.")
def set_viewport(orbit_right_deg, orbit_up_deg, center, distance, perspective):
    space, region = _view3d()
    r3d = space.region_3d
    if perspective is not None:
        r3d.view_perspective = perspective
    elif (orbit_right_deg or orbit_up_deg) and r3d.view_perspective == "CAMERA":
        r3d.view_perspective = "PERSP"  # orbiting leaves the camera view, as in the UI
    q = r3d.view_rotation.copy()
    if orbit_right_deg:
        q = Quaternion((0.0, 0.0, 1.0), math.radians(orbit_right_deg)) @ q
    if orbit_up_deg:
        q = q @ Quaternion((1.0, 0.0, 0.0), -math.radians(orbit_up_deg))
    q.normalize()
    r3d.view_rotation = q
    if center is not None:
        r3d.view_location = center
    if distance is not None:
        r3d.view_distance = distance
    r3d.update()  # recompute view/window matrices now, not at the next redraw
    region.tag_redraw()
    return _view_state(r3d)


@op("capture_viewport", Obj({
    "max_size": Opt(Int(64, 2048), 800),
    "camera": Opt(Union(Bool(), NAME, description="true = scene camera, or a camera name"), False),
}), description="Return a PNG of the 3D viewport as currently shown, or through a camera "
                "(OpenGL preview, not a render). Fast; use it often to check your work.")
def capture_viewport(max_size, camera):
    import gpu
    import numpy as np

    space, region = _view3d()
    sc = _scene()
    if camera:
        cam = sc.camera if camera is True else _obj(camera)
        if cam is None or cam.type != "CAMERA":
            raise OpError("invalid_params", "no such camera")
        rx, ry = sc.render.resolution_x, sc.render.resolution_y
        s = max_size / max(rx, ry)
        w, h = max(1, int(rx * s)), max(1, int(ry * s))
        dg = bpy.context.evaluated_depsgraph_get()
        view = cam.matrix_world.inverted()
        proj = cam.calc_matrix_camera(dg, x=w, y=h, scale_x=sc.render.pixel_aspect_x,
                                      scale_y=sc.render.pixel_aspect_y)
    else:
        r3d = space.region_3d
        # The matrices are otherwise only refreshed by a redraw, so a capture right after
        # set_viewport (same timer tick) showed the previous view.
        r3d.update()
        s = min(1.0, max_size / max(region.width, region.height))
        w, h = max(1, int(region.width * s)), max(1, int(region.height * s))
        view, proj = r3d.view_matrix, r3d.window_matrix
    off = gpu.types.GPUOffScreen(w, h)
    try:
        off.draw_view3d(sc, bpy.context.view_layer, space, region, view, proj,
                        do_color_management=True)
        buf = off.texture_color.read()
    finally:
        off.free()
    buf.dimensions = w * h * 4
    arr = np.asarray(buf, dtype=np.uint8)  # RGBA8 offscreen texture -> 0..255
    png = encode_png_rgba(w, h, arr.tobytes(), flip_vertical=True)
    return WithBlobs({"image": {"$blob": 0}, "width": w, "height": h}, [png])


@op("render_image", Obj({
    "max_size": Opt(Int(64, 2048), 800),
    "samples": Opt(Int(1, 256), 32),
}), description="Render the scene camera with the current engine at reduced size/samples and "
                "return a PNG. Slow (seconds to minutes); Blender is blocked meanwhile.")
def render_image(max_size, samples):
    sc = _scene()
    if sc.camera is None:
        raise OpError("invalid_params", "the scene has no camera")
    r = sc.render
    saved = {"filepath": r.filepath, "pct": r.resolution_percentage,
             "fmt": r.image_settings.file_format}
    eng = None
    if r.engine == "CYCLES" and hasattr(sc, "cycles"):
        eng = ("cycles", "samples", sc.cycles.samples)
        sc.cycles.samples = samples
    elif hasattr(sc, "eevee"):
        eng = ("eevee", "taa_render_samples", sc.eevee.taa_render_samples)
        sc.eevee.taa_render_samples = samples
    tmp = tempfile.mkdtemp(prefix="blender_safe_mcp_render_")
    try:
        r.resolution_percentage = max(1, min(100, int(100 * max_size /
                                                      max(r.resolution_x, r.resolution_y))))
        r.image_settings.file_format = "PNG"
        r.filepath = os.path.join(tmp, "render.png")
        bpy.ops.render.render(write_still=True)
        path = os.path.join(tmp, "render.png")
        with open(path, "rb") as f:
            data = f.read()
    finally:
        r.filepath, r.resolution_percentage = saved["filepath"], saved["pct"]
        r.image_settings.file_format = saved["fmt"]
        if eng:
            setattr(getattr(sc, eng[0]), eng[1], eng[2])
        shutil.rmtree(tmp, ignore_errors=True)
    return WithBlobs({"image": {"$blob": 0}, "engine": r.engine}, [data])


# =========================================================================== more ops
# Registered on the same registry; imported last because they use the helpers above.
from . import (ops_curve, ops_image, ops_mesh, ops_mesh2, ops_texture,  # noqa: E402
               ops_transform)

OP_MODULES = (ops_transform, ops_mesh, ops_mesh2, ops_curve, ops_texture, ops_image)

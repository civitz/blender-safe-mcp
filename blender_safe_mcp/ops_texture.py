"""Procedural textures and UV ops (plan phase 5).

Textures are created with bpy.data.textures.new (IMAGE ones start empty; phase 6 assigns).
UV projection, transforms and the layout picture are plain bmesh / Python math
(uvmath.py, imaging.py); the only bpy.ops calls are the three fixed uv.* unwrappers in
uv_unwrap, run in Edit mode with everything they touch restored in a finally block.
Every op validates fully and raises OpError before it changes anything.
"""

import math

import bpy

import bmesh

from . import imaging, uvmath
from .ops import (NAME, SETTINGS, _apply_settings, _mesh_bm, _obj, _window_override,
                  op, ALLOWED_MODIFIERS)
from .ops_mesh import SELECT, SPACE, _mw, _normal_matrix, _open_edit, _resolve
from .ops_transform import _run_operator
from .registry import WithBlobs
from .schema import Bool, Enum, Float, Int, Obj, Opt, Spec, Str, Union, Vec
from .server import OpError

MAX_TEXTURES = 1000
MAX_UV_MAPS = 8                 # Blender's limit
MAX_PROJECT_FACES = 500_000     # pure-Python loops
MAX_LAYOUT_FACES = 100_000
MAX_UNWRAP_FACES = 200_000
UV2 = Vec(2, -1e6, 1e6, description="U, V")
BLOCKED_TEXTURE_TYPES = ("NONE",)


class _NameOrNull(Spec):
    """A datablock name, or null (required key, so omitting it can't clear by accident)."""

    def __init__(self, description=None):
        self.description = description

    def validate(self, value, path=""):
        return None if value is None else NAME.validate(value, path)

    def json_schema(self):
        return self._with_desc({"anyOf": [NAME.json_schema(), {"type": "null"}]})


# --------------------------------------------------------------------------- textures

def _texture_types():
    items = bpy.types.Texture.bl_rna.properties["type"].enum_items
    return [i.identifier for i in items if i.identifier not in BLOCKED_TEXTURE_TYPES]


@op("create_texture", Obj({
    "name": NAME,
    "type": Str(max_len=32, pattern="[A-Z_]+",
                description="CLOUDS, DISTORTED_NOISE, BLEND, MAGIC, MARBLE, MUSGRAVE, NOISE, "
                            "STUCCI, VORONOI, WOOD or IMAGE (checked against Blender's list; an IMAGE "
                            "texture starts empty: assign one with set_texture_image)"),
    "settings": Opt(SETTINGS, None),
}), mutates=True,
    description="Create a procedural texture datablock (for Displace / Wave modifiers, see "
                "set_modifier_texture). 'settings' are texture properties (e.g. "
                "{\"noise_scale\": 0.5, \"use_color_ramp\": true}), applied through the same "
                "allow-listed RNA walker as set_property. Read or change it later with "
                "get_property/set_property using target {\"kind\": \"texture\", \"name\": ...}. "
                "Returns the (possibly suffixed) name.")
def create_texture(name, type, settings):
    allowed = _texture_types()
    if type not in allowed:
        raise OpError("invalid_params", f"texture type must be one of {allowed}")
    if len(bpy.data.textures) >= MAX_TEXTURES:
        raise OpError("budget", f"at most {MAX_TEXTURES} textures")
    tex = bpy.data.textures.new(name, type)
    try:
        applied = _apply_settings(tex, settings, tex.name)
    except Exception:
        bpy.data.textures.remove(tex)
        raise
    return {"name": tex.name, "type": tex.type, "settings": applied}


def _texture(name):
    tex = bpy.data.textures.get(name)
    if tex is None:
        raise OpError("not_found", f"no texture named {name!r}")
    if tex.type in BLOCKED_TEXTURE_TYPES:
        raise OpError("invalid_params", f"{tex.name} is a {tex.type} texture; it can't be "
                                        "used here")
    if tex.type == "IMAGE" and tex.image is not None:
        from .ops_image import assignable  # late: ops_image imports this module
        if not assignable(tex.image):
            raise OpError("invalid_params", f"{tex.name} uses an image that can't be used "
                                            "here (movie, sequence, tiled or render image)")
    return tex


@op("set_modifier_texture", Obj({
    "object": NAME,
    "modifier": NAME,
    "texture": _NameOrNull(description="Texture name, or null to clear"),
    "texture_coords": Opt(Enum(("LOCAL", "GLOBAL", "OBJECT", "UV"))),
    "texture_coords_object": Opt(NAME, None),
    "uv_layer": Opt(NAME, None),
}), mutates=True,
    description="Point a modifier's texture slot at a procedural texture (DISPLACE and WAVE "
                "have one), or clear it with null. texture_coords picks the mapping; OBJECT "
                "needs texture_coords_object (an object name) and UV needs uv_layer (an "
                "existing UV map of the mesh; see list_uv_maps).")
def set_modifier_texture(object, modifier, texture, texture_coords, texture_coords_object,
                         uv_layer):
    obj = _obj(object)
    mod = obj.modifiers.get(modifier)
    if mod is None:
        raise OpError("not_found", f"{obj.name} has no modifier {modifier!r}")
    prop = mod.bl_rna.properties.get("texture")
    if mod.type not in ALLOWED_MODIFIERS or prop is None \
            or getattr(prop.fixed_type, "identifier", None) != "Texture":
        raise OpError("invalid_params", f"{mod.name} ({mod.type}) has no texture slot")
    tex = None if texture is None else _texture(texture)
    coords_obj = None if texture_coords_object is None else _obj(texture_coords_object)
    if uv_layer is not None:
        if obj.type != "MESH" or uv_layer not in obj.data.uv_layers:
            raise OpError("not_found", f"{obj.name} has no UV map {uv_layer!r}")
    coords = texture_coords if texture_coords is not None else mod.texture_coords
    if texture_coords == "OBJECT" and coords_obj is None and mod.texture_coords_object is None:
        raise OpError("invalid_params", "texture_coords OBJECT needs texture_coords_object")
    if texture_coords == "UV" and uv_layer is None and not mod.uv_layer \
            and not (obj.type == "MESH" and len(obj.data.uv_layers)):
        raise OpError("invalid_params", f"texture_coords UV needs a UV map; {obj.name} has none")
    mod.texture = tex
    if texture_coords is not None:
        mod.texture_coords = texture_coords
    if coords_obj is not None:
        mod.texture_coords_object = coords_obj
    if uv_layer is not None:
        mod.uv_layer = uv_layer
    return {"object": obj.name, "modifier": mod.name, "texture": tex.name if tex else None,
            "texture_coords": coords, "uv_layer": mod.uv_layer or None,
            "texture_coords_object": mod.texture_coords_object.name
            if mod.texture_coords_object else None}


# --------------------------------------------------------------------------- UV maps

def _mesh_obj(name, need_object_mode=True):
    obj = _obj(name)
    if obj.type != "MESH":
        raise OpError("invalid_params", f"{obj.name} is a {obj.type}, not a mesh")
    if obj.data.library is not None:
        raise OpError("invalid_params", f"{obj.name}: linked data cannot be edited")
    if need_object_mode and obj.mode != "OBJECT":
        raise OpError("unavailable", f"{obj.name} is in {obj.mode} mode; use set_object_mode")
    return obj


def _uv_report(me):
    layers = me.uv_layers
    return {"count": len(layers), "max": MAX_UV_MAPS,
            "maps": [{"name": l.name, "active": l.active, "active_render": l.active_render}
                     for l in layers]}


@op("list_uv_maps", Obj({"object": NAME}),
    description="A mesh's UV maps: name, active (the one UV ops edit by default) and "
                "active_render, plus the count and the maximum (8).")
def list_uv_maps(object):
    obj = _obj(object)
    if obj.type != "MESH":
        raise OpError("invalid_params", f"{obj.name} is a {obj.type}, not a mesh")
    return {"object": obj.name, **_uv_report(obj.data)}


@op("uv_maps", Obj({
    "object": NAME,
    "action": Enum(("add", "remove", "rename", "set_active", "set_active_render")),
    "name": Opt(NAME, None),
    "new_name": Opt(NAME, None),
}), mutates=True, destructive=True,
    description="Change a mesh's UV maps (list_uv_maps reads them). add (name optional, "
                "default 'UVMap'; a new map is copied from the active one and becomes active; "
                "at most 8); remove / rename (new_name) / set_active / set_active_render need "
                "name. The mesh must be in Object mode. Returns the maps (name, active, "
                "active_render).")
def uv_maps(object, action, name, new_name):
    obj = _mesh_obj(object)
    me = obj.data
    layers = me.uv_layers
    if action != "add" and name is None:
        raise OpError("invalid_params", f"{action} needs 'name'")
    if action != "rename" and new_name is not None:
        raise OpError("invalid_params", "new_name is only used by rename")
    if action == "add":
        if (name or "UVMap") in layers:
            raise OpError("invalid_params", f"{obj.name} already has a UV map {name or 'UVMap'!r}")
        if len(layers) >= MAX_UV_MAPS:
            raise OpError("budget", f"a mesh can have at most {MAX_UV_MAPS} UV maps")
        layers.new(name=name or "UVMap", do_init=True)
    else:
        layer = layers.get(name)
        if layer is None:
            raise OpError("not_found", f"{obj.name} has no UV map {name!r}")
        if action == "rename":
            if new_name is None:
                raise OpError("invalid_params", "rename needs new_name")
            if new_name != name and new_name in layers:
                raise OpError("invalid_params", f"{obj.name} already has a UV map {new_name!r}")
            layer.name = new_name
        elif action == "set_active":
            layers.active = layer
        elif action == "set_active_render":
            layer.active_render = True
        else:
            layers.remove(layer)
    me.update()
    return {"object": obj.name, **_uv_report(me)}


# --------------------------------------------------------------------------- shared UV helpers

def _bm_uv_layer(bm, name, create):
    """The UV layer `name` (or the active one), creating a 'UVMap' if the mesh has none and
    `create`. Raises not_found / invalid_params."""
    layers = bm.loops.layers.uv
    if name is not None:
        layer = layers.get(name)
        if layer is None:
            raise OpError("not_found", f"no UV map {name!r}; see list_uv_maps")
        return layer
    if len(layers):
        return layers.active or layers[0]
    if not create:
        raise OpError("invalid_params", "the mesh has no UV map; create one with uv_maps or "
                                        "uv_project / uv_unwrap")
    return layers.new("UVMap")


def _faces_of(bm, obj, select, limit, what):
    bm.faces.ensure_lookup_table()
    if select is None:
        faces = list(bm.faces)
    else:
        faces = _resolve(bm, obj, select, "faces")
    if not faces:
        raise OpError("invalid_params", "the mesh has no faces")
    if len(faces) > limit:
        raise OpError("budget", f"{what} handles at most {limit:,} faces at a time "
                                f"(selected {len(faces):,})")
    return faces


def _layout(faces, layer):
    """[(vertex indices, [(u, v)...])] for stats / drawing."""
    return [([l.vert.index for l in f.loops], [tuple(l[layer].uv) for l in f.loops])
            for f in faces]


def _stats(faces, layer):
    lay = _layout(faces, layer)
    st = uvmath.layout_stats(lay)
    return st, [uv for _, uv in lay]


# --------------------------------------------------------------------------- uv_project

@op("uv_project", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "method": Enum(("cube", "cylinder", "sphere", "planar")),
    "uv_map": Opt(NAME, None),
    "axis": Opt(Enum(("x", "y", "z")), None),
    "scale": Opt(Float(1e-6, 1e6), 1.0),
    "space": Opt(SPACE, "local"),
    "normalize": Opt(Bool(), True),
}), mutates=True,
    description="Write UVs by pure projection math (no Blender operator). 'select' picks the "
                "faces (default all). method: planar (onto the plane perpendicular to axis; "
                "x->(y,z), y->(z,x), z->(x,y)), cube (each face along its dominant normal axis; "
                "normalised, the six directions form a 3x2 atlas in the lower two thirds of "
                "0..1), cylinder (angle around axis, height along it) or sphere (longitude / "
                "latitude around axis). Cylinder and sphere handle the wrap per face (no face "
                "spans the seam; pole vertices take their neighbours' angle); faces that "
                "surround the axis (cylinder caps) come out as flat strips: use cube or "
                "planar for those. axis (default z) applies to everything but cube. "
                "normalize (default true) maps the result into 0..1 (planar/cube keep their "
                "aspect, cylinder/sphere fit each axis); scale then multiplies the UVs (>1 "
                "tiles beyond 0..1). space: local (default) or world. uv_map defaults to the "
                "active map and a mesh with none gets 'UVMap'. Returns the UV bounds, island "
                "count and zero-area face count.")
def uv_project(object, select, method, uv_map, axis, scale, space, normalize):
    if axis is not None and method == "cube":
        raise OpError("invalid_params", "cube projection takes no axis")
    obj, ed = _open_edit(object)
    with ed as bm:
        faces = _faces_of(bm, obj, select, MAX_PROJECT_FACES, "uv_project")
        layer = _bm_uv_layer(bm, uv_map, create=True)
        ax = "xyz".index(axis or "z")
        if space == "world":
            mw = _mw(obj)
            nm = _normal_matrix(mw)
            pos = [[tuple(mw @ l.vert.co) for l in f.loops] for f in faces]
            normals = [tuple(nm @ f.normal) for f in faces]
        else:
            pos = [[tuple(l.vert.co) for l in f.loops] for f in faces]
            normals = [tuple(f.normal) for f in faces]
        if method == "planar":
            uvs = uvmath.project_planar(pos, ax, normalize, scale)
        elif method == "cube":
            uvs = uvmath.project_cube(pos, normals, normalize, scale)
        elif method == "cylinder":
            uvs = uvmath.project_cylinder(pos, ax, normalize, scale)
        else:
            uvs = uvmath.project_sphere(pos, ax, normalize, scale)
        for f, face_uv in zip(faces, uvs):
            for loop, uv in zip(f.loops, face_uv):
                loop[layer].uv = uv
        stats, _ = _stats(faces, layer)
        used = layer.name
    return {"object": obj.name, "uv_map": used, "method": method, "faces": len(faces),
            "uv_bounds": stats["bounds"], "islands": stats["islands"],
            "zero_area_faces": stats["zero_area_faces"]}


# --------------------------------------------------------------------------- transform_uvs

@op("transform_uvs", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "uv_map": Opt(NAME, None),
    "translate": Opt(UV2),
    "rotate_deg": Opt(Float(-36000, 36000), 0.0),
    "scale": Opt(UV2),
    "pivot": Opt(Union(Enum(("center", "origin")), Obj({"point": UV2})), "center",),
}), mutates=True,
    description="Move / rotate / scale the UVs of the selected faces (default all). Order: "
                "scale about the pivot, rotate about it (counter-clockwise in UV space), then "
                "translate. pivot: 'center' (middle of the affected UVs' bounds, default), "
                "'origin' (0,0) or {\"point\": [u, v]}. Welded UVs stay welded. Returns the "
                "new UV bounds.")
def transform_uvs(object, select, uv_map, translate, rotate_deg, scale, pivot):
    translate = translate or [0.0, 0.0]
    scale = scale or [1.0, 1.0]
    if translate == [0.0, 0.0] and rotate_deg == 0 and scale == [1.0, 1.0]:
        raise OpError("invalid_params", "give translate, rotate_deg or scale")
    obj, ed = _open_edit(object)
    with ed as bm:
        faces = _faces_of(bm, obj, select, MAX_PROJECT_FACES, "transform_uvs")
        layer = _bm_uv_layer(bm, uv_map, create=False)
        loops = [l for f in faces for l in f.loops]
        pts = [tuple(l[layer].uv) for l in loops]
        if pivot == "origin":
            piv = (0.0, 0.0)
        elif isinstance(pivot, dict):
            piv = tuple(pivot["point"])
        else:
            us, vs = [p[0] for p in pts], [p[1] for p in pts]
            piv = ((min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2)
        new = uvmath.transform_points(pts, translate, rotate_deg, scale, piv)
        if not all(math.isfinite(c) and abs(c) < 1e9 for p in new for c in p):
            raise OpError("out_of_range", "result UVs would be out of range")
        for l, uv in zip(loops, new):
            l[layer].uv = uv
        stats, _ = _stats(faces, layer)
        used = layer.name
    return {"object": obj.name, "uv_map": used, "faces": len(faces),
            "uv_bounds": stats["bounds"]}


# --------------------------------------------------------------------------- get_uv_layout

@op("get_uv_layout", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "uv_map": Opt(NAME, None),
    "size": Opt(Int(32, 2048), 512),
}), description="Return a PNG of the UV edges of the selected faces (default all) in the "
                "0..1 square (grey border; edges outside 0..1 are orange and the view grows "
                "to include them), plus stats: island count, UV bounds, faces with zero UV "
                "area, and the view window.")
def get_uv_layout(object, select, uv_map, size):
    obj = _obj(object)
    if obj.type != "MESH":
        raise OpError("invalid_params", f"{obj.name} is a {obj.type}, not a mesh")
    bm = _mesh_bm(obj)
    try:
        faces = _faces_of(bm, obj, select, MAX_LAYOUT_FACES, "get_uv_layout")
        layer = _bm_uv_layer(bm, uv_map, create=False)
        stats, polys = _stats(faces, layer)
        used = layer.name
    finally:
        bm.free()
    rgba, (u0, v0, span) = imaging.rasterize_uv_layout(size, polys)
    png = imaging.encode_png_rgba(size, size, rgba)
    return WithBlobs({"image": {"$blob": 0}, "width": size, "height": size, "object": obj.name,
                      "uv_map": used, "faces": len(faces), "islands": stats["islands"],
                      "uv_bounds": stats["bounds"], "zero_area_faces": stats["zero_area_faces"],
                      "view": {"min": [u0, v0], "span": span}}, [png])


# --------------------------------------------------------------------------- uv_unwrap

UV_METHODS = ("smart_project", "unwrap", "lightmap")
# Indirection so the smoke test can substitute a failing operator to exercise the restore path.
_UV_OPERATORS = {
    "smart_project": lambda p: bpy.ops.uv.smart_project(
        angle_limit=p["angle"], island_margin=p["margin"], correct_aspect=p["correct_aspect"],
        scale_to_bounds=False),
    "unwrap": lambda p: bpy.ops.uv.unwrap(
        method=p["unwrap_method"], correct_aspect=p["correct_aspect"], margin=p["margin"]),
    "lightmap": lambda p: bpy.ops.uv.lightmap_pack(
        PREF_CONTEXT="SEL_FACES", PREF_PACK_IN_ONE=True, PREF_NEW_UVLAYER=False,
        PREF_MARGIN_DIV=p["margin"]),
}
_DEFAULT_MARGIN = {"smart_project": 0.0, "unwrap": 0.001, "lightmap": 0.1}


def _flags(coll):
    out = [False] * len(coll)
    coll.foreach_get("select", out)
    return out


def _set_flags(coll, flags):
    coll.foreach_set("select", flags)


@op("uv_unwrap", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "method": Enum(UV_METHODS),
    "uv_map": Opt(NAME, None),
    "angle_limit_deg": Opt(Float(0, 89), 66.0),
    "margin": Opt(Float(0, 1), None),
    "correct_aspect": Opt(Bool(), True),
    "unwrap_method": Opt(Enum(("ANGLE_BASED", "CONFORMAL", "MINIMUM_STRETCH")), "ANGLE_BASED"),
}), mutates=True,
    description="Unwrap UVs with Blender's own algorithms (runs the operator in Edit mode and "
                "restores selection, mode and active UV map afterwards). 'select' picks the "
                "faces (default all). method: smart_project (angle_limit_deg, default 66), "
                "unwrap (honours seams, see set_edge_flags; unwrap_method ANGLE_BASED / "
                "CONFORMAL / MINIMUM_STRETCH) or lightmap (one packed island per face). "
                "margin is the island spacing (defaults: 0 smart_project, 0.001 unwrap, "
                "0.1 lightmap). uv_map picks the map to write (default active; a mesh with "
                "none gets 'UVMap'). The object must be in Object mode. Needs a Blender window "
                "context; in background mode the operators run but results may differ.")
def uv_unwrap(object, select, method, uv_map, angle_limit_deg, margin, correct_aspect,
              unwrap_method):
    obj = _mesh_obj(object)
    me = obj.data
    if bpy.context.view_layer.objects.get(obj.name) is None:
        raise OpError("invalid_params", f"{obj.name} is not in the current view layer")
    if len(me.polygons) == 0:
        raise OpError("invalid_params", f"{obj.name} has no faces")
    if len(me.polygons) > MAX_UNWRAP_FACES:
        raise OpError("budget", f"uv_unwrap handles at most {MAX_UNWRAP_FACES:,} faces")
    if uv_map is not None and uv_map not in me.uv_layers:
        raise OpError("not_found", f"{obj.name} has no UV map {uv_map!r}; see list_uv_maps")
    bm = _mesh_bm(obj)
    try:
        faces = _faces_of(bm, obj, select, MAX_UNWRAP_FACES, "uv_unwrap")
        face_idx = [f.index for f in faces]
    finally:
        bm.free()
    hidden = [i for i in face_idx if me.polygons[i].hide]
    if hidden:
        raise OpError("invalid_params", f"face {hidden[0]} is hidden; unhide it in Blender first")
    params = {"angle": math.radians(angle_limit_deg),
              "margin": _DEFAULT_MARGIN[method] if margin is None else margin,
              "correct_aspect": correct_aspect, "unwrap_method": unwrap_method}
    runner = _UV_OPERATORS[method]       # looked up before anything changes

    # ---- nothing above changed the scene; everything below is restored in `finally`
    if me.users > 1:
        obj.data = me = me.copy()
    created = None
    if not len(me.uv_layers):
        me.uv_layers.new(name="UVMap", do_init=False)
        created = "UVMap"
    prev = {"v": _flags(me.vertices), "e": _flags(me.edges), "f": _flags(me.polygons),
            "active_uv": me.uv_layers.active_index}
    ts = bpy.context.scene.tool_settings
    prev_mode, prev_sync = tuple(ts.mesh_select_mode), ts.use_uv_select_sync
    used = uv_map or me.uv_layers.active.name
    ok = False

    def work():
        me.uv_layers.active = me.uv_layers[used]
        ts.use_uv_select_sync = True
        ts.mesh_select_mode = (False, False, True)
        bpy.context.view_layer.objects.active = obj      # _run_operator restores it
        override = dict(_window_override() or {})
        override.update(active_object=obj, object=obj)
        try:
            with bpy.context.temp_override(**override):
                if bpy.ops.object.mode_set(mode="EDIT") != {"FINISHED"}:
                    raise OpError("failed", "Blender refused to enter Edit mode")
                ebm = bmesh.from_edit_mesh(me)
                ebm.select_mode = {"FACE"}
                for coll in (ebm.faces, ebm.edges, ebm.verts):
                    for e in coll:
                        e.select_set(False)
                ebm.faces.ensure_lookup_table()
                for i in face_idx:
                    ebm.faces[i].select_set(True)
                ebm.select_flush_mode()
                bmesh.update_edit_mesh(me, loop_triangles=False, destructive=False)
                del ebm
                return runner(params)
        finally:
            leave_edit_mode(obj, override)

    try:
        _run_operator([obj], False, work)
        ok = True
    finally:
        try:
            if obj.mode != "OBJECT":      # leave_edit_mode failed: one more try
                leave_edit_mode(obj, {})
            _set_flags(me.vertices, prev["v"])
            _set_flags(me.edges, prev["e"])
            _set_flags(me.polygons, prev["f"])
            if not ok and created is not None:
                me.uv_layers.remove(me.uv_layers[created])
            elif prev["active_uv"] >= 0:
                me.uv_layers.active_index = prev["active_uv"]
            me.update()
        except Exception:  # noqa: BLE001  (never mask the real error)
            pass
        ts.use_uv_select_sync = prev_sync
        ts.mesh_select_mode = prev_mode
    bm = _mesh_bm(obj)
    try:
        bm.faces.ensure_lookup_table()
        layer = _bm_uv_layer(bm, used, create=False)
        stats, _ = _stats([bm.faces[i] for i in face_idx], layer)
    finally:
        bm.free()
    return {"object": obj.name, "uv_map": used, "method": method, "faces": len(face_idx),
            "uv_bounds": stats["bounds"], "islands": stats["islands"],
            "zero_area_faces": stats["zero_area_faces"], "mode": obj.mode}


def leave_edit_mode(obj, override):
    if obj.mode == "OBJECT":
        return
    with bpy.context.temp_override(**override):
        bpy.ops.object.mode_set(mode="OBJECT")

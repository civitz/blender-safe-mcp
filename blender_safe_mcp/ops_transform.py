"""Transform ops: 3D cursor, origin placement, applying transforms (plan phase 1).

Same rules as ops.py: strict schemas, objects looked up by name, and bpy.ops only as fixed
calls inside temp_override, with the user's selection / active object restored afterwards.
"""

import math

import bpy
from mathutils import Euler, Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

from .ops import LOC, MAX_ANALYZE_FACES, NAME, ROT, _cursor_info, _obj, _obj_summary, _vec, _window_override, op
from .schema import Bool, Enum, Float, List, Obj, Opt, Union, Vec
from .server import OpError

TRANSFORM_TYPES = {"MESH", "CURVE", "FONT", "SURFACE"}
ORIGIN_TYPES = {
    "geometry_bounds": ("ORIGIN_GEOMETRY", "BOUNDS"),
    "geometry_median": ("ORIGIN_GEOMETRY", "MEDIAN"),
    "center_of_mass_surface": ("ORIGIN_CENTER_OF_MASS", "MEDIAN"),
    "center_of_mass_volume": ("ORIGIN_CENTER_OF_VOLUME", "MEDIAN"),
    "cursor": ("ORIGIN_CURSOR", "MEDIAN"),
    "point": ("ORIGIN_CURSOR", "MEDIAN"),
    "geometry_to_origin": ("GEOMETRY_ORIGIN", "MEDIAN"),
}
GEOMETRY_TYPES = {"MESH", "CURVE", "FONT", "SURFACE", "META"}


def _world_loc(obj):
    return _vec(obj.matrix_world.translation)


def _evaluated(obj):
    return obj.evaluated_get(bpy.context.evaluated_depsgraph_get())


def _eval_points(obj, limit=None):
    """World-space vertices of the evaluated geometry (curves / text / surfaces / metaballs
    are tessellated with to_mesh, like convert_object), or None if there are more than
    `limit` or none at all."""
    ev = _evaluated(obj)
    owned = obj.type != "MESH"
    me = ev.to_mesh() if owned else ev.data
    try:
        n = len(me.vertices) if me is not None else 0
        if n == 0 or (limit is not None and n > limit):
            return None
        flat = [0.0] * (n * 3)
        me.vertices.foreach_get("co", flat)
        mw = ev.matrix_world
        return [mw @ Vector(flat[i:i + 3]) for i in range(0, n * 3, 3)]
    finally:
        if owned:
            ev.to_mesh_clear()


def _finite(vs):
    return all(math.isfinite(x) for v in vs for x in v)


def _eval_box(obj):
    """(min, max) world box of the evaluated geometry. Only a mesh's evaluated bound_box is
    trusted: for curves it can be plainly wrong (Blender 5.1, a bevelled NURBS spiral 0.5
    tall reported local z -1..1.45, so drop_to_surface cast from under the table and
    missed). Other types use their tessellated vertices, then the origin."""
    pts = None
    if obj.type == "MESH":
        ev = _evaluated(obj)
        pts = [ev.matrix_world @ Vector(c) for c in ev.bound_box]
        if not _finite(pts):
            pts = None
    if pts is None:
        pts = _eval_points(obj) or [obj.matrix_world.translation.copy()]
    return (Vector(min(p[i] for p in pts) for i in range(3)),
            Vector(max(p[i] for p in pts) for i in range(3)))


def _bounds_center(obj):
    lo, hi = _eval_box(obj)
    return (lo + hi) / 2


def _median(obj):
    ev = _evaluated(obj)
    me = ev.data
    n = len(me.vertices)
    if n == 0:
        return obj.matrix_world.translation.copy()
    flat = [0.0] * (n * 3)
    me.vertices.foreach_get("co", flat)
    s = Vector((sum(flat[0::3]), sum(flat[1::3]), sum(flat[2::3])))
    return ev.matrix_world @ (s / n)


@op("set_cursor", Obj({
    "location": Opt(LOC),
    "rotation_deg": Opt(ROT),
    "to_object": Opt(NAME),
    "to": Opt(Enum(("origin", "bounds_center", "median"))),
}), mutates=True,
    description="Move the 3D cursor (used by set_origin mode 'cursor'). Give at most one of "
                "location (world space) or to_object (or neither, to only rotate it). With to_object, 'to' picks the spot: "
                "'origin' (default), 'bounds_center' (center of its evaluated bounding box) or "
                "'median' (median of the evaluated mesh vertices; non-mesh objects use their "
                "origin). rotation_deg optionally sets the cursor rotation too. Returns the "
                "new cursor.")
def set_cursor(location, rotation_deg, to_object, to):
    if location is not None and to_object is not None:
        raise OpError("invalid_params", "give at most one of location and to_object")
    if location is None and to_object is None and rotation_deg is None:
        raise OpError("invalid_params", "give location, to_object or rotation_deg")
    if to is not None and to_object is None:
        raise OpError("invalid_params", "'to' is only used together with to_object")
    if to_object is not None:
        obj = _obj(to_object)
        mode = to or "origin"
        if mode == "origin" or obj.type not in GEOMETRY_TYPES:
            target = obj.matrix_world.translation.copy()
        elif mode == "bounds_center":
            target = _bounds_center(obj)
        else:
            target = _median(obj) if obj.type == "MESH" else obj.matrix_world.translation.copy()
    elif location is not None:
        target = Vector(location)
    else:
        target = None
    cur = bpy.context.scene.cursor
    if target is not None:
        cur.location = target
    if rotation_deg is not None:
        cur.rotation_mode = "XYZ"
        cur.rotation_euler = Euler([math.radians(a) for a in rotation_deg], "XYZ")
    return _cursor_info()


# --------------------------------------------------------------------------- operator runner

def _targets(names):
    objs = []
    for n in names:
        obj = _obj(n)
        if obj in objs:
            raise OpError("invalid_params", f"{n} is listed twice")
        objs.append(obj)
    view_objects = bpy.context.view_layer.objects
    for o in objs:
        if o.type not in TRANSFORM_TYPES:
            raise OpError("invalid_params", f"{o.name} is a {o.type}; only MESH, CURVE, FONT "
                                            "and SURFACE objects are supported")
        if view_objects.get(o.name) is None:
            raise OpError("invalid_params", f"{o.name} is not in the current view layer")
        if o.mode != "OBJECT":
            raise OpError("unavailable", f"{o.name} is in {o.mode} mode; ask the user to return "
                                         "to Object mode")
        if o.data.library is not None:
            raise OpError("invalid_params", f"{o.name}: linked data cannot be edited")
    return objs


def _check_shared(objs, make_single_user):
    shared = [o.name for o in objs if o.data.users > 1]
    if shared and not make_single_user:
        raise OpError("invalid_params", f"{', '.join(shared)}: data is shared with other "
                                        "objects; pass make_single_user=true to give each its "
                                        "own copy first")


def _run_operator(objs, make_single_user, call):
    """Run call() with objs as the operator's selection; leave selection, active object and
    mode as they were, whatever happens."""
    vl = bpy.context.view_layer
    prev_active = vl.objects.active
    prev_selected = {o for o in vl.objects if o.select_get()}
    if make_single_user:
        for o in objs:
            if o.data.users > 1:
                o.data = o.data.copy()
    override = dict(_window_override() or {})
    override.update(active_object=objs[0], object=objs[0], selected_objects=list(objs),
                    selected_editable_objects=list(objs))
    try:
        with bpy.context.temp_override(**override):
            if call() != {"FINISHED"}:
                raise OpError("failed", "Blender refused the operation (see get_debug_logs)")
    finally:
        try:
            for o in vl.objects:
                if o.select_get() != (o in prev_selected):
                    o.select_set(o in prev_selected)
            if vl.objects.active is not prev_active:
                vl.objects.active = prev_active
        except Exception:  # noqa: BLE001  (best effort; never mask the real error)
            pass
        vl.update()


@op("set_origin", Obj({
    "objects": List(NAME, 1, 500),
    "mode": Enum(tuple(ORIGIN_TYPES)),
    "point": Opt(LOC),
    "make_single_user": Opt(Bool(), False),
}), mutates=True,
    description="Move object origins without moving the geometry in the world (like Object > "
                "Set Origin). mode: geometry_bounds / geometry_median (center of the geometry), "
                "center_of_mass_surface / center_of_mass_volume, cursor (the 3D cursor, see "
                "set_cursor), point (a world-space 'point', required for this mode and only "
                "there; the cursor is left as it was), or geometry_to_origin (moves the "
                "geometry onto the origin instead). MESH/CURVE/FONT/SURFACE only. Data shared "
                "by several objects is refused unless make_single_user=true. Returns each "
                "object's new world location.")
def set_origin(objects, mode, point, make_single_user):
    if (point is not None) != (mode == "point"):
        raise OpError("invalid_params", "'point' is required for mode 'point' and only "
                                        "allowed there")
    objs = _targets(objects)
    _check_shared(objs, make_single_user)
    otype, center = ORIGIN_TYPES[mode]
    cur = bpy.context.scene.cursor
    saved = cur.location.copy()
    try:
        if mode == "point":
            cur.location = Vector(point)
        _run_operator(objs, make_single_user,
                      lambda: bpy.ops.object.origin_set(type=otype, center=center))
    finally:
        cur.location = saved
    return {"objects": [{"name": o.name, "world_location": _world_loc(o)} for o in objs]}


@op("apply_transform", Obj({
    "objects": List(NAME, 1, 500),
    "location": Opt(Bool(), False),
    "rotation": Opt(Bool(), True),
    "scale": Opt(Bool(), True),
    "make_single_user": Opt(Bool(), False),
}), mutates=True, destructive=True,
    description="Bake the object transform into the geometry (like Ctrl+A): afterwards the "
                "chosen parts of location / rotation / scale are identity and the vertices "
                "carry them (vertex indices stay the same, positions change). Default applies "
                "rotation and scale. Negative scales are handled (normals stay outward). "
                "MESH/CURVE/FONT/SURFACE only; shared data is refused unless "
                "make_single_user=true. Returns each object's new transform.")
def apply_transform(objects, location, rotation, scale, make_single_user):
    if not (location or rotation or scale):
        raise OpError("invalid_params", "enable at least one of location, rotation, scale")
    objs = _targets(objects)
    _check_shared(objs, make_single_user)
    _run_operator(objs, make_single_user, lambda: bpy.ops.object.transform_apply(
        location=location, rotation=rotation, scale=scale, properties=True,
        isolate_users=False))
    return {"objects": [_obj_summary(o) for o in objs]}


# --------------------------------------------------------------------------- phase 2: placement
# Plain math on matrix_world and world-space bounding boxes of the evaluated objects. Every
# op plans all moves first and only then writes, so a refusal leaves the scene untouched.



AXES = {"x": 0, "y": 1, "z": 2}
MAX_VERTS_SAMPLED = 20_000      # more than this: sample the bounding-box face instead
BBOX_GRID = 8
MAX_RAYS = 400_000
RAY_BACKOFF = 1e-4
MAX_SNAP_VERTS = 2_000_000


def _movable(names):
    """Like _targets but for any object type (cameras, lights, empties...)."""
    objs = []
    view_objects = bpy.context.view_layer.objects
    for n in names:
        obj = _obj(n)
        if obj in objs:
            raise OpError("invalid_params", f"{n} is listed twice")
        if view_objects.get(obj.name) is None:
            raise OpError("invalid_params", f"{obj.name} is not in the current view layer")
        if obj.mode != "OBJECT":
            raise OpError("unavailable", f"{obj.name} is in {obj.mode} mode; ask the user to "
                                         "return to Object mode")
        objs.append(obj)
    bpy.context.view_layer.update()
    return objs


def _wbox(obj):
    """(min, max) of the evaluated object's world-space bounding box; a point for objects
    without geometry."""
    if obj.type not in GEOMETRY_TYPES:
        p = obj.matrix_world.translation.copy()
        return p, p.copy()
    return _eval_box(obj)


def _depth(obj):
    d = 0
    while obj.parent is not None:
        obj, d = obj.parent, d + 1
    return d


def _apply_matrices(plan):
    """plan: [(obj, target matrix_world)]. Parents first; refresh the depsgraph when a
    parent in the plan moved, so a child's local transform is computed against it."""
    plan = sorted(plan, key=lambda p: _depth(p[0]))
    objs = {p[0] for p in plan}
    for obj, mat in plan:
        a = obj.parent
        while a is not None and a not in objs:
            a = a.parent
        if a is not None:
            bpy.context.view_layer.update()
        obj.matrix_world = mat
    bpy.context.view_layer.update()


def _moved(obj):
    return {"name": obj.name, "world_location": _world_loc(obj)}


def _shifted(obj, delta):
    m = obj.matrix_world.copy()
    m.translation = m.translation + Vector(delta)
    return m


@op("align_objects", Obj({
    "objects": List(NAME, 1, 500),
    "axes": List(Enum(("x", "y", "z")), 1, 3),
    "mode": Enum(("min", "center", "max")),
    "relative_to": Opt(Union(Enum(("first", "world_origin", "cursor", "selection_bounds")),
                             Obj({"object": NAME})), "first"),
}), mutates=True,
    description="Move objects so their world-space bounding boxes (evaluated, so modifiers "
                "count) line up on the chosen axes: mode min / center / max matches that side of "
                "each box to the reference. relative_to: 'first' (default: objects[0], which "
                "stays put), 'world_origin', 'cursor', 'selection_bounds' (the box around all "
                "listed objects) or {\"object\": name} (an object that stays put; it is skipped "
                "if also listed). Any object type works; cameras, lights and empties count as a "
                "point at their origin. Returns each object's new world location.")
def align_objects(objects, axes, mode, relative_to):
    if len(set(axes)) != len(axes):
        raise OpError("invalid_params", "axes must be unique")
    objs = _movable(objects)
    boxes = {o: _wbox(o) for o in objs}
    skip = None
    if isinstance(relative_to, dict):
        ref_obj = _obj(relative_to["object"])
        ref = _wbox(ref_obj)
        skip = ref_obj
    elif relative_to == "first":
        ref, skip = boxes[objs[0]], objs[0]
    elif relative_to == "world_origin":
        ref = (Vector(), Vector())
    elif relative_to == "cursor":
        c = bpy.context.scene.cursor.location.copy()
        ref = (c, c.copy())
    else:
        ref = (Vector(min(b[0][i] for b in boxes.values()) for i in range(3)),
               Vector(max(b[1][i] for b in boxes.values()) for i in range(3)))
    pick = {"min": lambda b, i: b[0][i], "max": lambda b, i: b[1][i],
            "center": lambda b, i: (b[0][i] + b[1][i]) / 2}[mode]
    plan = []
    for o in objs:
        if o is skip:
            continue
        delta = Vector()
        for a in axes:
            i = AXES[a]
            delta[i] = pick(ref, i) - pick(boxes[o], i)
        plan.append((o, _shifted(o, delta)))
    _apply_matrices(plan)
    return {"objects": [_moved(o) for o in objs]}


@op("distribute_objects", Obj({
    "objects": List(NAME, 3, 500),
    "axis": Enum(("x", "y", "z")),
    "mode": Opt(Enum(("centers", "gaps")), "centers"),
}), mutates=True,
    description="Spread 3+ objects evenly along a world axis. The two extreme objects (by "
                "current box center on that axis) stay put and the ones between them are "
                "moved. mode 'centers' (default): equal distance between bounding-box centers; "
                "'gaps': equal empty space between neighbouring bounding boxes (negative if "
                "they overlap). Returns the objects in their new order along the axis, with "
                "world locations.")
def distribute_objects(objects, axis, mode):
    objs = _movable(objects)
    i = AXES[axis]
    boxes = {o: _wbox(o) for o in objs}
    order = sorted(objs, key=lambda o: (boxes[o][0][i] + boxes[o][1][i]) / 2)
    first, last = boxes[order[0]], boxes[order[-1]]
    plan = []
    if mode == "centers":
        c0 = (first[0][i] + first[1][i]) / 2
        c1 = (last[0][i] + last[1][i]) / 2
        step = (c1 - c0) / (len(order) - 1)
        for k, o in enumerate(order[1:-1], 1):
            b = boxes[o]
            d = Vector()
            d[i] = c0 + step * k - (b[0][i] + b[1][i]) / 2
            plan.append((o, _shifted(o, d)))
    else:
        mid = order[1:-1]
        span = last[0][i] - first[1][i] - sum(boxes[o][1][i] - boxes[o][0][i] for o in mid)
        gap = span / (len(order) - 1)
        cursor = first[1][i] + gap
        for o in mid:
            b = boxes[o]
            d = Vector()
            d[i] = cursor - b[0][i]
            plan.append((o, _shifted(o, d)))
            cursor += b[1][i] - b[0][i] + gap
    _apply_matrices(plan)
    return {"objects": [_moved(o) for o in order]}


def _mesh_target(name, what):
    t = _obj(name)
    if t.type != "MESH":
        raise OpError("invalid_params", f"{what} {t.name} is a {t.type}; a MESH is required")
    if len(_evaluated(t).data.polygons) > MAX_ANALYZE_FACES:
        raise OpError("budget", f"{t.name} has more than {MAX_ANALYZE_FACES:,} faces")
    return t


def _bvh(target):
    return BVHTree.FromObject(target, bpy.context.evaluated_depsgraph_get())


def _nearest_surface(target, bvh, point):
    inv = target.matrix_world.inverted()
    loc, _n, _i, _d = bvh.find_nearest(inv @ point)
    if loc is None:
        raise OpError("not_found", f"{target.name} has no faces")
    return target.matrix_world @ loc


@op("snap_objects", Obj({
    "objects": List(NAME, 1, 500),
    "to": Union(Obj({"grid": Float(1e-6, 1e6)}), Enum(("cursor",)),
                Obj({"object": NAME,
                     "target": Opt(Enum(("origin", "nearest_vertex", "nearest_surface",
                                         "bounds_center")), "origin")})),
}), mutates=True,
    description="Move objects' origins onto a snap target (rotation and scale are kept; world "
                "space; a parented object ends with that world location). to: {\"grid\": "
                "increment} rounds the world location to the grid; \"cursor\" (the 3D cursor); "
                "{\"object\": name, \"target\": ...} with target 'origin' (default), "
                "'bounds_center' (center of its evaluated bounding box), 'nearest_vertex' or "
                "'nearest_surface' (closest point on the evaluated mesh to each moved object's "
                "origin; these two need a MESH target). The target object must not be in "
                "'objects'. Returns each object's new world location.")
def snap_objects(objects, to):
    objs = _movable(objects)
    plan = []
    if isinstance(to, dict) and "grid" in to:
        g = to["grid"]
        for o in objs:
            p = o.matrix_world.translation
            plan.append((o, _shifted(o, Vector(round(c / g) * g for c in p) - p)))
    elif to == "cursor":
        c = bpy.context.scene.cursor.location
        plan = [(o, _shifted(o, c - o.matrix_world.translation)) for o in objs]
    else:
        mode = to["target"]
        if mode in ("nearest_vertex", "nearest_surface"):
            t = _mesh_target(to["object"], "snap target")
        else:
            t = _obj(to["object"])
        if t in objs:
            raise OpError("invalid_params", f"{t.name} is both the snap target and a moved "
                                            "object")
        if mode == "origin":
            dest = lambda _p: t.matrix_world.translation.copy()  # noqa: E731
        elif mode == "bounds_center":
            c = _bounds_center(t)
            dest = lambda _p: c  # noqa: E731
        elif mode == "nearest_surface":
            bvh = _bvh(t)
            dest = lambda p: _nearest_surface(t, bvh, p)  # noqa: E731
        else:
            ev = _evaluated(t)
            n = len(ev.data.vertices)
            if n == 0:
                raise OpError("not_found", f"{t.name} has no vertices")
            if n > MAX_SNAP_VERTS:
                raise OpError("budget", f"{t.name} has more than {MAX_SNAP_VERTS:,} vertices")
            kd = KDTree(n)
            mw = ev.matrix_world
            for k, v in enumerate(ev.data.vertices):
                kd.insert(mw @ v.co, k)
            kd.balance()
            dest = lambda p: Vector(kd.find(p)[0])  # noqa: E731
        for o in objs:
            p = o.matrix_world.translation
            plan.append((o, _shifted(o, dest(p) - p)))
    _apply_matrices(plan)
    return {"objects": [_moved(o) for o in objs]}


def _sample_points(obj, direction):
    """World-space points to cast from: all evaluated vertices (curves, text etc. included,
    tessellated), or (over 20 000, or no geometry) an 8x8 grid on the bounding-box face the
    object would land on."""
    if obj.type in GEOMETRY_TYPES:
        pts = _eval_points(obj, MAX_VERTS_SAMPLED)
        if pts:
            return pts
    lo, hi = _wbox(obj)
    a = max(range(3), key=lambda k: abs(direction[k]))
    fixed = lo[a] if direction[a] < 0 else hi[a]
    b, c = [k for k in range(3) if k != a]
    pts = []
    for u in range(BBOX_GRID):
        for w in range(BBOX_GRID):
            p = Vector()
            p[a] = fixed
            p[b] = lo[b] + (hi[b] - lo[b]) * u / (BBOX_GRID - 1)
            p[c] = lo[c] + (hi[c] - lo[c]) * w / (BBOX_GRID - 1)
            pts.append(p)
    return pts


def _cast(points, direction, trees, floor_z):
    """Nearest hit over all trees: (distance, normal, label) or None."""
    best = None
    for p in points:
        # Start a hair behind the point: a ray starting exactly on a surface (an object
        # already resting on it) can miss it, and the next vertex then pushed it in.
        start = p - direction * RAY_BACKOFF
        for t, bvh, inv in trees:
            loc, nrm, _i, dist = bvh.ray_cast(inv @ start, inv.to_3x3() @ direction)
            if loc is None:
                continue
            # local-space ray: the direction was transformed too, so rescale the distance
            world_hit = t.matrix_world @ loc
            d = (world_hit - p).dot(direction)
            if d < -2 * RAY_BACKOFF:
                continue  # behind the point (only reachable through the back-off)
            if best is None or d < best[0]:
                n = (inv.to_3x3().transposed() @ nrm).normalized()
                best = (d, n, t.name)
        if floor_z is not None and abs(direction.z) > 1e-9:
            d = (floor_z - p.z) / direction.z
            if d >= 0 and (best is None or d < best[0]):
                best = (d, Vector((0, 0, 1)), "floor_z")
    return best


@op("drop_to_surface", Obj({
    "objects": List(NAME, 1, 500),
    "direction": Opt(Vec(3, -1, 1), [0, 0, -1]),
    "targets": Opt(List(NAME, 1, 100)),
    "floor_z": Opt(Float(-1e6, 1e6)),
    "offset": Opt(Float(-1e6, 1e6), 0.0),
    "align_to_normal": Opt(Bool(), False),
}), mutates=True,
    description="Drop objects onto surfaces, like Blender's 'drop to floor'. Rays go along "
                "'direction' (default [0,0,-1]) from the object's evaluated vertices (large "
                "meshes: an 8x8 grid on the bounding-box face that leads; non-mesh objects: "
                "their origin) against 'targets' (mesh names; default every visible mesh not "
                "being moved) and optionally the plane z=floor_z. The object moves along "
                "the direction by the smallest hit distance minus 'offset' (so it rests on "
                "the surface, leaving a gap of offset). align_to_normal first rotates it "
                "about its origin so its local +Z (up) lines up with the surface normal at "
                "the closest hit. An object with no hit at all makes the whole call fail with "
                "not_found and nothing moves. Returns, per object, the new world location, "
                "the distance moved and what it landed on.")
def drop_to_surface(objects, direction, targets, floor_z, offset, align_to_normal):
    d = Vector(direction)
    if d.length < 1e-6:
        raise OpError("invalid_params", "direction must not be zero")
    d.normalize()
    objs = _movable(objects)
    if targets is not None:
        tobjs = [_mesh_target(n, "target") for n in targets]
        clash = [t.name for t in tobjs if t in objs]
        if clash:
            raise OpError("invalid_params", f"{', '.join(clash)} is both a target and "
                                            "a moved object")
    else:
        tobjs = [o for o in bpy.context.view_layer.objects
                 if o.type == "MESH" and o not in objs and o.visible_get()]
        tobjs = [t for t in tobjs if len(_evaluated(t).data.polygons) <= MAX_ANALYZE_FACES]
    if not tobjs and floor_z is None:
        raise OpError("not_found", "no target meshes (and no floor_z) to drop onto")
    trees = [(t, _bvh(t), t.matrix_world.inverted()) for t in tobjs]
    samples = {o: _sample_points(o, d) for o in objs}
    if sum(len(s) for s in samples.values()) > MAX_RAYS:
        raise OpError("budget", f"too many rays (limit {MAX_RAYS:,}); drop fewer objects")
    plan, report = [], []
    for o in objs:
        pts = samples[o]
        mat = o.matrix_world.copy()
        hit = _cast(pts, d, trees, floor_z)
        if hit is None:
            raise OpError("not_found", f"{o.name}: nothing to drop onto along {list(direction)}")
        if align_to_normal:
            n = hit[1]
            if n.dot(d) > 0:
                n = -n
            up = mat.to_quaternion() @ Vector((0, 0, 1))
            rot = up.rotation_difference(n).to_matrix().to_4x4()
            o_loc = mat.translation.copy()
            pivot = Matrix.Translation(o_loc) @ rot @ Matrix.Translation(-o_loc)
            mat = pivot @ mat
            pts = [pivot @ p for p in pts]
            hit = _cast(pts, d, trees, floor_z)
            if hit is None:
                raise OpError("not_found", f"{o.name}: nothing below after aligning to the "
                                           "normal")
        dist = hit[0] - offset
        mat.translation = mat.translation + d * dist
        plan.append((o, mat))
        report.append((o, round(dist, 6), hit[2]))
    _apply_matrices(plan)
    return {"objects": [dict(_moved(o), moved=dist, landed_on=on) for o, dist, on in report]}


@op("look_at", Obj({
    "object": NAME,
    "target": Union(Obj({"object": NAME}), Obj({"point": LOC})),
    "track_axis": Opt(Enum(("-Z", "Z", "-X", "X", "-Y", "Y")), "-Z"),
    "up_axis": Opt(Enum(("X", "Y", "Z")), "Y"),
}), mutates=True,
    description="Rotate an object so its track_axis (default local -Z, which is what cameras "
                "and spot lights look along) points at a world point or at another object's "
                "origin, keeping up_axis (default local Y) as upright as possible. Location "
                "and scale are unchanged. track_axis and up_axis must be different axes. "
                "Returns the object summary.")
def look_at(object, target, track_axis, up_axis):
    if track_axis.lstrip("-") == up_axis:
        raise OpError("invalid_params", "track_axis and up_axis must be different axes")
    obj = _movable([object])[0]
    if "object" in target:
        dest = _obj(target["object"])
        if dest is obj:
            raise OpError("invalid_params", "an object cannot look at itself")
        point = dest.matrix_world.translation.copy()
    else:
        point = Vector(target["point"])
    loc, _rot, scale = obj.matrix_world.decompose()
    direction = point - loc
    if direction.length < 1e-9:
        raise OpError("invalid_params", "the target is at the object's origin")
    quat = direction.to_track_quat(track_axis, up_axis)
    obj.matrix_world = Matrix.LocRotScale(loc, quat, scale)
    return _obj_summary(obj)

"""Curve ops and object conversion (plan phase 4).

Curves are built from data (bpy.data.curves.new + splines.new). The only bpy.ops call is the
fixed object.convert(target='CURVE') in convert_object. Every op validates everything and
estimates the resulting geometry before it changes anything.
"""

import math

import bpy

from .ops import (LOC, MAX_ANALYZE_FACES, MAX_MESH_VERTS, MAX_POLYS, NAME, ROT, SCALE,
                  _analyze_bm, _data_collection, _estimate_polys, _link, _mesh_bm, _obj,
                  _obj_summary, _place, _summary, _vec, op)
from .ops_transform import _run_operator
from .schema import Bool, Enum, Float, Int, List, Obj, Opt, Str, Union, Vec
from .server import OpError

MAX_SPLINES = 256
MAX_SPLINE_POINTS = 10_000
MAX_CURVE_POINTS = 50_000

SPLINE_TYPES = {"bezier": "BEZIER", "poly": "POLY", "nurbs": "NURBS"}
RNA_TO_TYPE = {v: k for k, v in SPLINE_TYPES.items()}
# Blender's point handle enum has FREE / VECTOR / ALIGNED / AUTO ("Automatic", which is the
# clamped kind); "auto_clamped" is accepted as an alias of auto.
HANDLE_TYPES = {"auto": "AUTO", "auto_clamped": "AUTO", "vector": "VECTOR",
                "aligned": "ALIGNED", "free": "FREE"}

CO = Vec(3, -1e6, 1e6)
POINT = Union(
    CO,
    Obj({
        "co": CO,
        "handle_left": Opt(CO), "handle_right": Opt(CO),
        "handle_type": Opt(Enum(tuple(HANDLE_TYPES),
                          description="bezier only; default 'free' when a handle position is "
                                      "given, else 'auto'")),
        "tilt_deg": Opt(Float(-36000, 36000)),
        "radius": Opt(Float(0.0, 1e4)),
        "weight": Opt(Float(0.0001, 1e4, description="nurbs / poly only (default 1)")),
    }),
    description="[x, y, z], or an object: {co, handle_left?, handle_right?, handle_type?} for "
                "bezier, {co, weight?} for nurbs / poly; tilt_deg and radius for any.")
SPLINE = Obj({
    "type": Enum(tuple(SPLINE_TYPES)),
    "points": List(POINT, 2, MAX_SPLINE_POINTS),
    "cyclic": Opt(Bool(), False),
    "order": Opt(Int(2, 6, description="nurbs only; default min(4, point count)")),
    "resolution": Opt(Int(1, 64, description="segments per span; default the curve's")),
}, description="One spline. World/object-local coordinates are the curve object's local space.")
FILL = ("FULL", "BACK", "FRONT", "HALF")


# --------------------------------------------------------------------------- normalisation

def _normalize_point(p, stype, where):
    if isinstance(p, list):
        p = {"co": p}
    d = {k: v for k, v in p.items() if v is not None}
    if stype != "BEZIER" and any(k in d for k in ("handle_left", "handle_right", "handle_type")):
        raise OpError("invalid_params", f"{where}: handles only exist on bezier splines")
    if stype == "BEZIER" and "weight" in d:
        raise OpError("invalid_params", f"{where}: weight doesn't apply to bezier points")
    ht = None
    if stype == "BEZIER":
        explicit = "handle_left" in d or "handle_right" in d
        ht = HANDLE_TYPES[d["handle_type"]] if "handle_type" in d else \
            ("FREE" if explicit else "AUTO")
    return {"co": tuple(d["co"]), "hl": tuple(d["handle_left"]) if "handle_left" in d else None,
            "hr": tuple(d["handle_right"]) if "handle_right" in d else None,
            "hlt": ht, "hrt": ht, "tilt": math.radians(d.get("tilt_deg", 0.0)),
            "radius": d.get("radius", 1.0), "weight": d.get("weight", 1.0)}


def _normalize_spline(spec, where):
    stype = SPLINE_TYPES[spec["type"]]
    pts = [_normalize_point(p, stype, f"{where}.points[{i}]")
           for i, p in enumerate(spec["points"])]
    n = len(pts)
    order = None
    if stype == "NURBS":
        order = spec.get("order") or min(4, n)
        if order > n:
            raise OpError("invalid_params", f"{where}: order {order} needs at least {order} "
                                            f"points, got {n}")
    elif spec.get("order") is not None:
        raise OpError("invalid_params", f"{where}: order only applies to nurbs splines")
    return {"type": stype, "points": pts, "cyclic": bool(spec.get("cyclic")), "order": order,
            "resolution": spec.get("resolution"), "extras": {}}


def _normalize_all(specs, where="splines"):
    out = [_normalize_spline(s, f"{where}[{i}]") for i, s in enumerate(specs)]
    if sum(len(s["points"]) for s in out) > MAX_CURVE_POINTS:
        raise OpError("budget", f"at most {MAX_CURVE_POINTS:,} points per curve")
    return out


# --------------------------------------------------------------------------- budget

def _faces_for(stype, n, cyclic, res, ring):
    if stype == "POLY":
        segs = n if cyclic else n - 1
    else:
        segs = (n if cyclic else n - 1) * res
    return max(segs, 1) * ring


def _ring(bevel_depth, bevel_res, extrude, bevel_mode, bevel_obj):
    """Faces per curve segment contributed by the cross-section."""
    if bevel_mode == "OBJECT" and bevel_obj is not None:
        c = bevel_obj.data
        return max(1, sum(len(s.points) * max(s.resolution_u, 1) + len(s.bezier_points) *
                          max(s.resolution_u, 1) for s in c.splines))
    ring = 0
    if bevel_depth > 0:
        ring = 4 * (bevel_res + 1)
    if extrude > 0:
        ring += 2
    return ring


def _check_curve_budget(default_res, splines, bevel_depth, bevel_res, extrude, bevel_mode="ROUND",
                        bevel_obj=None):
    """splines: iterable of (type, point count, cyclic, resolution or None)."""
    ring = _ring(bevel_depth, bevel_res, extrude, bevel_mode, bevel_obj)
    total = 0
    for stype, n, cyclic, res in splines:
        total += _faces_for(stype, n, cyclic, res or default_res, max(ring, 1))
    if total > MAX_POLYS:
        raise OpError("budget", f"curve would produce ~{total:,} faces (limit {MAX_POLYS:,})")


def _existing(curve, skip=None):
    return [(s.type, len(s.bezier_points) if s.type == "BEZIER" else len(s.points),
             s.use_cyclic_u, s.resolution_u) for i, s in enumerate(curve.splines) if i != skip]


def _curve_estimate_list(curve, norm_splines, skip=None):
    return _existing(curve, skip) + [(s["type"], len(s["points"]), s["cyclic"], s["resolution"])
                                     for s in norm_splines]


def _curve_faces_of(obj):
    """Face estimate for an existing curve object (used when converting)."""
    c = obj.data
    ring = _ring(c.bevel_depth, c.bevel_resolution, c.extrude, c.bevel_mode, c.bevel_object)
    return sum(_faces_for(t, n, cy, r, max(ring, 1)) for t, n, cy, r in _existing(c))


# --------------------------------------------------------------------------- spline data

def _build_spline(curve, d):
    s = curve.splines.new(d["type"])
    pts = d["points"]
    if d["type"] == "BEZIER":
        s.bezier_points.add(len(pts) - 1)
        for bp, p in zip(s.bezier_points, pts):
            bp.handle_left_type, bp.handle_right_type = p["hlt"], p["hrt"]
            bp.co = p["co"]
            if p["hl"] is not None:
                bp.handle_left = p["hl"]
            if p["hr"] is not None:
                bp.handle_right = p["hr"]
            bp.tilt, bp.radius = p["tilt"], p["radius"]
    else:
        s.points.add(len(pts) - 1)
        for sp, p in zip(s.points, pts):
            sp.co = (*p["co"], p["weight"])
            sp.tilt, sp.radius = p["tilt"], p["radius"]
    s.use_cyclic_u = d["cyclic"]
    if d["type"] == "NURBS":
        s.order_u = d["order"]
        s.use_endpoint_u = d["extras"].get("use_endpoint_u", not d["cyclic"])
        s.use_bezier_u = d["extras"].get("use_bezier_u", False)
    if d["resolution"]:
        s.resolution_u = d["resolution"]
    for k, v in d["extras"].items():
        if k not in ("use_endpoint_u", "use_bezier_u"):
            setattr(s, k, v)
    return s


EXTRAS = ("use_smooth", "material_index", "tilt_interpolation", "radius_interpolation")


def _dump_spline(s):
    """Full-fidelity copy of a spline in the normalized format, to rebuild it later."""
    if s.type == "BEZIER":
        pts = [{"co": tuple(b.co), "hl": tuple(b.handle_left), "hr": tuple(b.handle_right),
                "hlt": b.handle_left_type, "hrt": b.handle_right_type, "tilt": b.tilt,
                "radius": b.radius, "weight": 1.0} for b in s.bezier_points]
    else:
        pts = [{"co": tuple(p.co)[:3], "hl": None, "hr": None, "hlt": None, "hrt": None,
                "tilt": p.tilt, "radius": p.radius, "weight": p.co[3]} for p in s.points]
    extras = {k: getattr(s, k) for k in EXTRAS}
    if s.type == "NURBS":
        extras["use_endpoint_u"] = s.use_endpoint_u
        extras["use_bezier_u"] = s.use_bezier_u
    return {"type": s.type, "points": pts, "cyclic": s.use_cyclic_u,
            "order": s.order_u if s.type == "NURBS" else None,
            "resolution": s.resolution_u, "extras": extras}


def _spline_info(i, s, max_points):
    bez = s.type == "BEZIER"
    items = s.bezier_points if bez else s.points
    n = len(items)
    pts = []
    for it in list(items)[:max_points]:
        if bez:
            pts.append({"co": _vec(it.co), "handle_left": _vec(it.handle_left),
                        "handle_right": _vec(it.handle_right),
                        "handle_left_type": it.handle_left_type,
                        "handle_right_type": it.handle_right_type,
                        "tilt_deg": round(math.degrees(it.tilt), 4),
                        "radius": round(it.radius, 6)})
        else:
            pts.append({"co": _vec(it.co[:3]), "weight": round(it.co[3], 6),
                        "tilt_deg": round(math.degrees(it.tilt), 4),
                        "radius": round(it.radius, 6)})
    info = {"index": i, "type": RNA_TO_TYPE[s.type], "cyclic": s.use_cyclic_u,
            "resolution": s.resolution_u, "point_count": n, "points": pts,
            "points_truncated": n > max_points}
    if s.type == "NURBS":
        info["order"] = s.order_u
    return info


def _curve_obj(name, edit=True):
    obj = _obj(name)
    if obj.type != "CURVE":
        raise OpError("invalid_params", f"{obj.name} is a {obj.type}; need a CURVE object "
                                        "(text objects have no editable splines)")
    if edit:
        if obj.mode != "OBJECT":
            raise OpError("unavailable", f"{obj.name} is in {obj.mode} mode; ask the user to "
                                         "return to Object mode")
        if obj.data.library is not None:
            raise OpError("invalid_params", f"{obj.name}: linked data cannot be edited")
    return obj


SINGLE_USER = Opt(Bool(description=(
    "the curve data is shared with other objects: give this object its own copy first "
    "(otherwise shared data is refused, since the edit would change every user)")), False)


def _check_shared(obj, make_single_user):
    if obj.data.users > 1 and not make_single_user:
        raise OpError("invalid_params", f"{obj.name}: its curve data is shared with other "
                                        "objects; pass make_single_user=true to give it its "
                                        "own copy first")


def _own_data(obj):
    """Call after all validation, right before mutating."""
    if obj.data.users > 1:
        obj.data = obj.data.copy()
    return obj.data


def _total_points(curve, skip=None):
    return sum(len(s.bezier_points) if s.type == "BEZIER" else len(s.points)
               for i, s in enumerate(curve.splines) if i != skip)


def _curve_result(obj):
    c = obj.data
    return {"object": obj.name, "curve": c.name, "splines": len(c.splines),
            "points": _total_points(c)}


# --------------------------------------------------------------------------- ops

@op("add_curve", Obj({
    "name": Opt(NAME),
    "splines": List(SPLINE, 1, MAX_SPLINES),
    "dimensions": Opt(Enum(("2D", "3D")), "3D"),
    "bevel_depth": Opt(Float(0.0, 1e3)),
    "bevel_resolution": Opt(Int(0, 32)),
    "extrude": Opt(Float(0.0, 1e3)),
    "fill_mode": Opt(Enum(FILL)),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True,
    description="Create a curve object from explicit splines (bezier / poly / nurbs; at most "
                f"{MAX_SPLINES} splines, {MAX_SPLINE_POINTS:,} points each, {MAX_CURVE_POINTS:,} "
                "in total). Points are in the object's local space. bevel_depth / extrude give "
                "the curve thickness (a round tube for bevel_depth, resolution via "
                "bevel_resolution); cyclic splines with fill_mode FULL (2D) make flat shapes. "
                "convert_object turns it into a mesh.")
def add_curve(name, splines, dimensions, bevel_depth, bevel_resolution, extrude, fill_mode,
              location, rotation_deg, scale):
    norm = _normalize_all(splines)
    _check_curve_budget(12, [(s["type"], len(s["points"]), s["cyclic"], s["resolution"])
                             for s in norm], bevel_depth or 0.0,
                        4 if bevel_resolution is None else bevel_resolution, extrude or 0.0)
    return _make_curve(name or "Curve", norm, dimensions, bevel_depth, bevel_resolution,
                       extrude, fill_mode, location, rotation_deg, scale)


def _make_curve(name, norm, dimensions, bevel_depth, bevel_resolution, extrude, fill_mode,
                location, rotation_deg, scale):
    curve = bpy.data.curves.new(name, "CURVE")
    try:
        curve.dimensions = dimensions
        for d in norm:
            _build_spline(curve, d)
        if bevel_depth is not None:
            curve.bevel_depth = bevel_depth
        if bevel_resolution is not None:
            curve.bevel_resolution = bevel_resolution
        if extrude is not None:
            curve.extrude = extrude
        if fill_mode is not None:
            curve.fill_mode = fill_mode
        obj = bpy.data.objects.new(curve.name, curve)
    except Exception:
        bpy.data.curves.remove(curve)
        raise
    _link(obj)
    _place(obj, location, rotation_deg, scale)
    out = _obj_summary(obj)
    out.update(splines=len(curve.splines), points=_total_points(curve))
    return out


KAPPA = 0.5522847498307936


def _bez(co, hl, hr, ht="ALIGNED"):
    return {"co": co, "handle_left": hl, "handle_right": hr, "handle_type":
            {"ALIGNED": "aligned", "AUTO": "auto"}[ht]}


def _primitive_splines(kind, radius, size, turns, height):
    if kind == "bezier_circle":
        r, k = radius, radius * KAPPA
        return [{"type": "bezier", "cyclic": True, "points": [
            _bez([r, 0, 0], [r, -k, 0], [r, k, 0]),
            _bez([0, r, 0], [k, r, 0], [-k, r, 0]),
            _bez([-r, 0, 0], [-r, k, 0], [-r, -k, 0]),
            _bez([0, -r, 0], [-k, -r, 0], [k, -r, 0])]}]
    if kind == "bezier_curve":
        s = size / 2
        return [{"type": "bezier", "points": [
            _bez([-s, 0, 0], [-1.5 * s, -0.5 * s, 0], [-0.5 * s, 0.5 * s, 0]),
            _bez([s, 0, 0], [0.5 * s, 0.5 * s, 0], [1.5 * s, -0.5 * s, 0])]}]
    if kind == "nurbs_circle":
        r, w = radius, math.sqrt(0.5)
        ring = [([r, 0, 0], 1), ([r, r, 0], w), ([0, r, 0], 1), ([-r, r, 0], w),
                ([-r, 0, 0], 1), ([-r, -r, 0], w), ([0, -r, 0], 1), ([r, -r, 0], w)]
        return [{"type": "nurbs", "cyclic": True, "order": 3,
                 "points": [{"co": c, "weight": wt} for c, wt in ring]}]
    if kind == "nurbs_path":
        n = 5
        return [{"type": "nurbs", "order": 5, "points": [
            [size * (i / (n - 1) - 0.5), 0, 0] for i in range(n)]}]
    # spiral: a NURBS helix along +Z
    n = max(4, int(turns * 12) + 1)
    pts = []
    for i in range(n):
        t = i / (n - 1)
        a = 2 * math.pi * turns * t
        pts.append([radius * math.cos(a), radius * math.sin(a), height * t])
    return [{"type": "nurbs", "order": 4, "points": pts}]


@op("add_curve_primitive", Obj({
    "kind": Enum(("bezier_circle", "bezier_curve", "nurbs_circle", "nurbs_path", "spiral")),
    "name": Opt(NAME),
    "radius": Opt(Float(0.0001, 1e5, description="circles and spiral (default 1)")),
    "size": Opt(Float(0.0001, 1e5, description="bezier_curve / nurbs_path length (default 2)")),
    "turns": Opt(Float(0.1, 50, description="spiral only (default 3)")),
    "height": Opt(Float(-1e5, 1e5, description="spiral only (default 2)")),
    "bevel_depth": Opt(Float(0.0, 1e3)),
    "location": Opt(LOC), "rotation_deg": Opt(ROT), "scale": Opt(SCALE),
}), mutates=True,
    description="Add a ready-made curve: bezier_circle (4 points), bezier_curve (2 points, "
                "along X), nurbs_circle (8 weighted points), nurbs_path (5 points, along X) or "
                "spiral (helix along +Z). Built from data like add_curve; edit it with "
                "set_spline.")
def add_curve_primitive(kind, name, radius, size, turns, height, bevel_depth, location,
                        rotation_deg, scale):
    circle = kind in ("bezier_circle", "nurbs_circle")
    allowed = {"radius": circle or kind == "spiral", "size": kind in ("bezier_curve", "nurbs_path"),
               "turns": kind == "spiral", "height": kind == "spiral"}
    given = {"radius": radius, "size": size, "turns": turns, "height": height}
    for k, v in given.items():
        if v is not None and not allowed[k]:
            raise OpError("invalid_params", f"{k} doesn't apply to {kind}")
    norm = _normalize_all(_primitive_splines(
        kind, 1.0 if radius is None else radius, 2.0 if size is None else size,
        3.0 if turns is None else turns, 2.0 if height is None else height))
    if kind == "nurbs_circle":  # as Blender's own circle: closed loop of quadratic arcs
        norm[0]["extras"] = {"use_endpoint_u": True, "use_bezier_u": True}
    _check_curve_budget(12, [(s["type"], len(s["points"]), s["cyclic"], s["resolution"])
                             for s in norm], bevel_depth or 0.0, 4, 0.0)
    return _make_curve(name or kind.replace("_", " ").title(), norm, "3D", bevel_depth, None,
                       None, None, location, rotation_deg, scale)


@op("get_curve_info", Obj({
    "object": NAME,
    "max_points_per_spline": Opt(Int(0, MAX_SPLINE_POINTS), 200),
}), description="Read a curve object: every spline (type, cyclic, order, resolution) with its "
                "points (co; bezier handles and handle types; nurbs weights; tilt_deg; radius), "
                "plus bevel / extrude settings and the names of the bevel and taper objects. "
                "Point order is the index set_spline works with.")
def get_curve_info(object, max_points_per_spline):
    obj = _curve_obj(object, edit=False)
    c = obj.data
    return {
        "object": obj.name, "curve": c.name, "dimensions": c.dimensions,
        "resolution_u": c.resolution_u, "render_resolution_u": c.render_resolution_u,
        "bevel_mode": c.bevel_mode, "bevel_depth": round(c.bevel_depth, 6),
        "bevel_resolution": c.bevel_resolution, "extrude": round(c.extrude, 6),
        "offset": round(c.offset, 6), "fill_mode": c.fill_mode,
        "bevel_object": c.bevel_object.name if c.bevel_object else None,
        "taper_object": c.taper_object.name if c.taper_object else None,
        "use_path": c.use_path, "path_duration": c.path_duration,
        "points": _total_points(c),
        "splines": [_spline_info(i, s, max_points_per_spline)
                    for i, s in enumerate(c.splines)],
    }


def _check_index(c, index):
    if not 0 <= index < len(c.splines):
        raise OpError("not_found", f"no spline {index} (curve has {len(c.splines)})")


@op("set_spline", Obj({"object": NAME, "index": Int(0, MAX_SPLINES - 1), "spline": SPLINE,
                       "make_single_user": SINGLE_USER}),
    mutates=True, destructive=True,
    description="Replace spline `index` of a curve object with a new one (same schema as "
                "add_curve; the spline keeps its position and its smooth / material_index "
                "settings). Use get_curve_info for the current points and indices.")
def set_spline(object, index, spline, make_single_user):
    obj = _curve_obj(object)
    _check_shared(obj, make_single_user)
    c = obj.data
    _check_index(c, index)
    new = _normalize_spline(spline, "spline")
    if _total_points(c, skip=index) + len(new["points"]) > MAX_CURVE_POINTS:
        raise OpError("budget", f"at most {MAX_CURVE_POINTS:,} points per curve")
    _check_curve_budget(c.resolution_u, _curve_estimate_list(c, [new], skip=index),
                        c.bevel_depth, c.bevel_resolution, c.extrude, c.bevel_mode, c.bevel_object)
    c = _own_data(obj)
    old = c.splines[index]
    new["extras"] = {k: getattr(old, k) for k in EXTRAS}
    tail = [_dump_spline(s) for s in list(c.splines)[index + 1:]]
    for s in list(c.splines)[index:]:
        c.splines.remove(s)
    for d in [new, *tail]:
        _build_spline(c, d)
    return _curve_result(obj)


@op("add_spline", Obj({"object": NAME, "spline": SPLINE, "make_single_user": SINGLE_USER}),
    mutates=True,
    description="Append a spline to a curve object (same schema as add_curve). Returns the "
                "curve's spline and point counts; the new spline's index is splines-1.")
def add_spline(object, spline, make_single_user):
    obj = _curve_obj(object)
    _check_shared(obj, make_single_user)
    c = obj.data
    new = _normalize_spline(spline, "spline")
    if len(c.splines) >= MAX_SPLINES:
        raise OpError("budget", f"at most {MAX_SPLINES} splines per curve")
    if _total_points(c) + len(new["points"]) > MAX_CURVE_POINTS:
        raise OpError("budget", f"at most {MAX_CURVE_POINTS:,} points per curve")
    _check_curve_budget(c.resolution_u, _curve_estimate_list(c, [new]), c.bevel_depth,
                        c.bevel_resolution, c.extrude, c.bevel_mode, c.bevel_object)
    _build_spline(_own_data(obj), new)
    return _curve_result(obj)


@op("remove_spline", Obj({"object": NAME, "index": Int(0, MAX_SPLINES - 1),
                          "make_single_user": SINGLE_USER}), mutates=True,
    destructive=True,
    description="Remove spline `index` from a curve object; later splines move down by one.")
def remove_spline(object, index, make_single_user):
    obj = _curve_obj(object)
    _check_shared(obj, make_single_user)
    _check_index(obj.data, index)
    c = _own_data(obj)
    c.splines.remove(c.splines[index])
    return _curve_result(obj)


def _curve_target(name, what):
    t = _obj(name)
    if t.type != "CURVE":
        raise OpError("invalid_params", f"{what} {t.name} is a {t.type}; it must be a CURVE")
    return t


def _reaches(start_curve, goal_curve):
    """True if goal_curve is reachable from start_curve through bevel / taper objects."""
    seen, stack = set(), [start_curve]
    while stack:
        c = stack.pop()
        if c == goal_curve:
            return True
        if c in seen:
            continue
        seen.add(c)
        for o in (c.bevel_object, c.taper_object):
            if o is not None and o.type == "CURVE":
                stack.append(o.data)
    return False


@op("set_curve_shape", Obj({
    "object": NAME,
    "bevel_mode": Opt(Enum(("ROUND", "OBJECT", "PROFILE"))),
    "bevel_object": Opt(Str(max_len=63, min_len=0, description=(
        "name of a CURVE object used as cross-section; '' clears it"))),
    "taper_object": Opt(Str(max_len=63, min_len=0, description=(
        "name of a CURVE object that scales the thickness along the length; '' clears it"))),
    "make_single_user": SINGLE_USER,
}), mutates=True,
    description="Set what shapes a curve's thickness: bevel_mode ROUND (bevel_depth), OBJECT "
                "(cross-section from bevel_object) or PROFILE; a taper_object. Targets must be "
                "CURVE objects, not the curve itself, and may not form a loop (A bevelled by B "
                "while B is bevelled by A). Giving bevel_object without bevel_mode switches the "
                "mode to OBJECT.")
def set_curve_shape(object, bevel_mode, bevel_object, taper_object, make_single_user):
    obj = _curve_obj(object)
    _check_shared(obj, make_single_user)
    c = obj.data
    if bevel_mode is None and bevel_object is None and taper_object is None:
        raise OpError("invalid_params", "give bevel_mode, bevel_object or taper_object")
    new = {}
    for key, name in (("bevel_object", bevel_object), ("taper_object", taper_object)):
        if name is None:
            continue
        if name == "":
            new[key] = None
            continue
        t = _curve_target(name, key)
        if t == obj or t.data == c:
            raise OpError("invalid_params", f"{key} can't be the curve itself")
        if _reaches(t.data, c):
            raise OpError("invalid_params", f"{key} {t.name} is already shaped by {obj.name}: "
                                            "that would be a loop")
        new[key] = t
    mode = bevel_mode or ("OBJECT" if new.get("bevel_object") else c.bevel_mode)
    bobj = new["bevel_object"] if "bevel_object" in new else c.bevel_object
    if mode == "OBJECT" and bobj is None:
        raise OpError("invalid_params", "bevel_mode OBJECT needs a bevel_object")
    _check_curve_budget(c.resolution_u, _existing(c), c.bevel_depth, c.bevel_resolution,
                        c.extrude, mode, bobj)
    c = _own_data(obj)
    c.bevel_mode = mode
    for key, val in new.items():
        setattr(c, key, val)
    return {"object": obj.name, "bevel_mode": c.bevel_mode,
            "bevel_object": c.bevel_object.name if c.bevel_object else None,
            "taper_object": c.taper_object.name if c.taper_object else None}


# --------------------------------------------------------------------------- convert_object

MESH_SOURCES = {"MESH", "CURVE", "FONT", "SURFACE", "META"}
CURVE_SOURCES = {"MESH", "FONT"}
SUPPORTED = ("MESH target from MESH, CURVE, FONT, SURFACE, META; "
             "CURVE target from MESH, FONT")


def _mesh_summary(me, obj):
    if len(me.polygons) > MAX_ANALYZE_FACES:
        return {"vertices": len(me.vertices), "faces": len(me.polygons)}
    bm = _mesh_bm(obj)
    try:
        return _summary(_analyze_bm(bm, obj, max_holes=10))
    finally:
        bm.free()


def _convert_to_mesh(src, keep_original, new_name):
    vl = bpy.context.view_layer
    if vl.objects.get(src.name) is None:
        raise OpError("invalid_params", f"{src.name} is not in the current view layer")
    if src.type == "MESH":
        est = _estimate_polys(src)
    elif src.type in ("CURVE", "SURFACE", "FONT"):
        est = _curve_faces_of(src) if src.type == "CURVE" else 0
    else:
        est = 0
    if est > MAX_POLYS:
        raise OpError("budget", f"{src.name}: would produce ~{est:,} faces (limit {MAX_POLYS:,})")
    vl.update()
    dg = bpy.context.evaluated_depsgraph_get()
    ev = src.evaluated_get(dg)
    tmp = ev.to_mesh()
    try:
        nf = len(tmp.polygons)
    finally:
        ev.to_mesh_clear()
    if nf > MAX_POLYS:
        raise OpError("budget", f"{src.name}: evaluated result has {nf:,} faces "
                                f"(limit {MAX_POLYS:,})")

    baked = [m.name for m in src.modifiers if m.show_viewport]
    not_baked = [m.name for m in src.modifiers if not m.show_viewport]

    mesh = bpy.data.meshes.new_from_object(ev, preserve_all_data_layers=True, depsgraph=dg)
    try:
        want = src.name if (new_name is None and not keep_original) else None
        obj = bpy.data.objects.new(new_name or src.name + "_mesh", mesh)
    except Exception:
        bpy.data.meshes.remove(mesh)
        raise
    obj.matrix_basis = src.matrix_basis.copy()
    obj.parent = src.parent
    if src.parent is not None and src.parent_type == "OBJECT":
        obj.matrix_parent_inverse = src.matrix_parent_inverse.copy()
    for coll in src.users_collection:
        coll.objects.link(obj)
    src_mats = list(src.data.materials)
    if len(mesh.materials) != len(src_mats):
        mesh.materials.clear()
        for m in src_mats:
            mesh.materials.append(m)
    for i, slot in enumerate(src.material_slots):
        if i < len(obj.material_slots) and slot.link == "OBJECT":
            obj.material_slots[i].link = "OBJECT"
            obj.material_slots[i].material = slot.material
    warnings = []
    if src.animation_data and src.animation_data.action:
        warnings.append("animation on the source object was not carried over")
    if getattr(src.data, "shape_keys", None):
        warnings.append("shape keys were not carried over (the evaluated shape was baked)")

    if not keep_original:
        src_name, data = src.name, src.data
        vl.update()  # the new object's matrix_world is stale until now
        for child in list(src.children):
            mw = child.matrix_world.copy()
            child.parent = obj
            child.matrix_parent_inverse = obj.matrix_world.inverted()
            child.matrix_world = mw
        bpy.data.objects.remove(src, do_unlink=True)
        if data is not None and data.users == 0:
            coll = _data_collection(data)
            if coll is not None:
                coll.remove(data)
        if want is not None:
            obj.name = src_name
    else:
        src_name = src.name
    mesh.name = obj.name
    vl.update()
    out = _obj_summary(obj)
    out.update(source=src_name, kept_original=keep_original, modifiers_baked=baked,
               modifiers_not_baked=not_baked, after=_mesh_summary(mesh, obj))
    if warnings:
        out["warnings"] = warnings
    return out


def _convert_to_curve(src, keep_original, new_name):
    vl = bpy.context.view_layer
    if vl.objects.get(src.name) is None:
        raise OpError("invalid_params", f"{src.name} is not in the current view layer")
    if src.data.library is not None:
        raise OpError("invalid_params", f"{src.name}: linked data cannot be converted")
    if src.type == "MESH":
        if len(src.data.vertices) > MAX_MESH_VERTS:
            raise OpError("budget", f"{src.name}: more than {MAX_MESH_VERTS:,} vertices")
        _check_budget_stack(src)
    old_data = src.data
    before = set(bpy.data.objects)
    before_data = (set(bpy.data.meshes), set(bpy.data.curves))
    _run_operator([src], False, lambda: bpy.ops.object.convert(
        target="CURVE", keep_original=keep_original))
    created = [o for o in bpy.data.objects if o not in before]
    if keep_original:
        if len(created) != 1:
            raise OpError("failed", "conversion didn't create exactly one object")
        obj = created[0]
    else:
        obj = src
        if old_data is not obj.data and old_data.users == 0:
            coll = _data_collection(old_data)
            if coll is not None:
                coll.remove(old_data)
    # keep_original leaves an intermediate datablock behind: drop what nothing uses
    for coll, old in ((bpy.data.meshes, before_data[0]), (bpy.data.curves, before_data[1])):
        for d in [d for d in coll if d not in old and d.users == 0]:
            coll.remove(d)
    if obj.type != "CURVE":
        raise OpError("failed", f"conversion produced a {obj.type}, not a curve")
    if new_name is not None:
        obj.name = new_name
    out = _obj_summary(obj)
    out.update(source=src.name, kept_original=keep_original, **{
        k: v for k, v in _curve_result(obj).items() if k in ("splines", "points")})
    return out


def _check_budget_stack(obj):
    if _estimate_polys(obj) > MAX_POLYS:
        raise OpError("budget", f"{obj.name}: modifier stack is over the geometry budget")


@op("convert_object", Obj({
    "object": NAME,
    "target": Enum(("MESH", "CURVE")),
    "keep_original": Opt(Bool(), False),
    "new_name": Opt(NAME),
}), mutates=True, destructive=True,
    description="Convert an object. MESH from MESH / CURVE / FONT / SURFACE / META: bakes the "
                "evaluated result (the whole modifier stack, bevel / extrude of curves, text "
                "shapes) into a new mesh object with the same matrix, parent, collections and "
                "materials; the new object has no modifiers. CURVE from MESH (edges become "
                "poly splines) or FONT (letter outlines). Without keep_original the source is "
                "deleted (destructive; its children are re-parented keeping their world "
                "transforms) and the new object takes its name; with keep_original the new "
                "object is named new_name or '<name>_mesh'. Other "
                "types (lights, cameras, empties, CURVE to CURVE ...) are refused. Returns the "
                "new object and, for MESH, its topology summary.")
def convert_object(object, target, keep_original, new_name):
    src = _obj(object)
    ok = MESH_SOURCES if target == "MESH" else CURVE_SOURCES
    if src.type not in ok:
        raise OpError("invalid_params", f"can't convert {src.name} ({src.type}) to {target}; "
                                        f"supported: {SUPPORTED}")
    if src.mode != "OBJECT":
        raise OpError("unavailable", f"{src.name} is in {src.mode} mode; ask the user to "
                                     "return to Object mode")
    if new_name is not None:
        other = bpy.data.objects.get(new_name)
        if other is not None and not (other == src and not keep_original):
            raise OpError("invalid_params", f"an object named {new_name!r} already exists")
    if target == "MESH":
        return _convert_to_mesh(src, keep_original, new_name)
    return _convert_to_curve(src, keep_original, new_name)

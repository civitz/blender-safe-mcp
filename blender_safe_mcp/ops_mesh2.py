"""More mesh-editing ops (plan phase 3b): bevel, loop_cut, bisect, subdivide, bridge_edge_loops,
spin, smooth_vertices, symmetrize, separate, snap_vertices, triangulate_faces, poke_faces.

Same rules and helpers as ops_mesh.py (selector, `_MeshEdit`, `created` index lists, geometry
budget). Every op estimates its geometry growth before running a bmesh operator and validates
everything before the edit; the edit happens on a copy that is written back only on success, so
a refusal (or a bmesh error, mapped to invalid_params) leaves the scene untouched.
"""

import math

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

import bmesh

from .ops import LOC, NAME, _summary, op
from .ops_mesh import (DIR3, ELEM_IDX, SELECT, SPACE, _compact, _check_growth, _counts,
                       _created, _mw, _open_edit, _reindex, _resolve,
                       _unit_scale)
from .ops_transform import MAX_SNAP_VERTS, _bvh, _mesh_target, _nearest_surface
from .schema import Bool, Enum, Float, Int, List, Obj, Opt, Union
from .server import OpError

MAX_SEPARATE_PARTS = 256
MAX_SEPARATE_WORK = 30_000_000    # parts x (vertices + faces): each part is built from a copy
MAX_SMOOTH_WORK = 20_000_000      # vertices x repeat


# --------------------------------------------------------------------------- helpers

def _snapshot(bm):
    """Python wrappers of every element, to tell survivors from new elements after an op."""
    return list(bm.verts), list(bm.edges), list(bm.faces)


def _fresh(bm, snap):
    """(verts, edges, faces) now in `bm` that were not in `snap`. Wrappers of elements an op
    deleted become invalid, so memory reused by a new element is not mistaken for a survivor.
    Elements an op modified in place count as survivors. bmesh.ops.subdivide_edges invalidates
    every VERTEX wrapper (edges and faces are fine), so subdivide-style ops take new vertices
    from the op's own `geom` result and edges/faces from this diff."""
    out = []
    for seq, old in zip((bm.verts, bm.edges, bm.faces), snap):
        keep = {e for e in old if e.is_valid}
        out.append([e for e in seq if e not in keep])
    return out


def _geom_split(geom):
    v = [g for g in geom if isinstance(g, bmesh.types.BMVert)]
    e = [g for g in geom if isinstance(g, bmesh.types.BMEdge)]
    f = [g for g in geom if isinstance(g, bmesh.types.BMFace)]
    return v, e, f


def _bmo(fn, bm, **kw):
    """Run a bmesh operator; its argument errors become invalid_params (nothing is written)."""
    try:
        return fn(bm, **kw)
    except (ValueError, RuntimeError) as e:
        raise OpError("invalid_params", f"{fn.__name__} failed: {e}") from None


def _result(obj, ed, bm_counts, created, **extra):
    out = {"object": obj.name, "created": created, "counts": bm_counts,
           "after": _summary(ed.report)}
    out.update(extra)
    return out


def _inverse(obj, mw):
    try:
        return mw.inverted()
    except ValueError:
        raise OpError("invalid_params", f"{obj.name} has a zero scale") from None


# --------------------------------------------------------------------------- bevel

@op("bevel", Obj({
    "object": NAME,
    "select": SELECT,
    "affect": Opt(Enum(("edges", "vertices")), "edges"),
    "offset": Float(0, 1e6),
    "offset_type": Opt(Enum(("offset", "width", "depth", "percent")), "offset"),
    "segments": Opt(Int(1, 32), 1),
    "profile": Opt(Float(0, 1), 0.5),
    "clamp_overlap": Opt(Bool(), True),
    "space": Opt(SPACE, "world"),
}), mutates=True,
    description="Bevel (chamfer/round) selected edges, or vertices with affect='vertices', like "
                "Blender's Ctrl+B. offset_type: 'offset' (distance each new edge is moved "
                "from the original), 'width' (total width of the bevel), 'depth' (depth "
                "perpendicular to the original faces) or 'percent' (offset is 0..100 percent "
                "of the adjacent edges). segments 1..32 (1 = flat chamfer), profile 0..1 "
                "(0.5 = circular round-over, lower is flatter/concave, higher is convex), "
                "clamp_overlap stops geometry from overlapping. Lengths are in `space` "
                "(world needs uniform object scale). The selector is resolved in edges (or "
                "vertices). Refused when the estimated result exceeds the geometry budget. "
                "Returns created {verts, edges, faces}: every element that did not exist "
                "before, including the existing faces that were rebuilt around the bevel.")
def bevel(object, select, affect, offset, offset_type, segments, profile, clamp_overlap, space):
    if offset <= 0:
        raise OpError("invalid_params", "offset must be > 0")
    if offset_type == "percent" and offset > 100:
        raise OpError("invalid_params", "offset is a percentage here: it must be <= 100")
    obj, ed = _open_edit(object)
    domain = "verts" if affect == "vertices" else "edges"
    k = 1.0 if offset_type == "percent" else _unit_scale(obj, space)
    with ed as bm:
        sel = _resolve(bm, obj, select, domain)
        touched = set(sel) if domain == "verts" else {v for e in sel for v in e.verts}
        extra = len(sel) * segments + len(touched) * (segments + 1) ** 2
        _check_growth(bm, extra)
        snap = _snapshot(bm)
        _bmo(bmesh.ops.bevel, bm, geom=list(sel), offset=offset * k,
             offset_type=offset_type.upper(), profile_type="SUPERELLIPSE", segments=segments,
             profile=profile, affect="VERTICES" if affect == "vertices" else "EDGES",
             clamp_overlap=clamp_overlap)
        created = _created(bm, *_fresh(bm, snap))
        counts = _counts(bm)
    return _result(obj, ed, counts, created)


# --------------------------------------------------------------------------- loop_cut

@op("loop_cut", Obj({
    "object": NAME,
    "edge": ELEM_IDX,
    "cuts": Opt(Int(1, 64), 1),
    "expect_counts": Opt(List(Int(0, 100_000_000), 3, 3)),
}), mutates=True,
    description="Loop cut, like Blender's Ctrl+R: walk the edge ring from `edge` (the edges "
                "parallel to it across consecutive quads, stopping at triangles/n-gons and "
                "boundaries) and cut `cuts` (1..64) evenly spaced loops across it. `edge` is "
                "the index of an edge that crosses the loop you want (it is cut in the "
                "middle), from get_mesh_elements. Pass expect_counts from that result to "
                "guard against stale indices. Returns created {verts, edges, faces}, "
                "cut_edges (the new edges running along the cut, e.g. to select them with "
                "{edges: ...} and slide/scale them), and ring_edges (how many edges were cut).")
def loop_cut(object, edge, cuts, expect_counts):
    obj, ed = _open_edit(object)
    sel_spec = {"edge_ring": edge}
    if expect_counts is not None:
        sel_spec["expect_counts"] = expect_counts
    with ed as bm:
        ring = _resolve(bm, obj, sel_spec, "edges")
        faces = {f for e in ring for f in e.link_faces}
        if not any(len(f.verts) == 4 for f in faces):
            raise OpError("invalid_params", f"edge {edge} is not part of a quad ring; loop_cut "
                                            "needs quad faces next to the edge")
        _check_growth(bm, len(faces) * cuts)
        snap = _snapshot(bm)
        ret = _bmo(bmesh.ops.subdivide_edges, bm, edges=list(ring), cuts=cuts,
                   use_grid_fill=True)
        created = _created(bm, _geom_split(ret["geom"])[0], *_fresh(bm, snap)[1:])
        inner = [g for g in ret["geom_inner"] if isinstance(g, bmesh.types.BMEdge)]
        cut_edges = sorted({g.index for g in inner})
        counts = _counts(bm)
    return _result(obj, ed, counts, created, cut_edges=_compact(cut_edges),
                   cut_edges_count=len(cut_edges), ring_edges=len(ring))


# --------------------------------------------------------------------------- bisect

@op("bisect", Obj({
    "object": NAME,
    "plane_point": LOC,
    "plane_normal": DIR3,
    "space": Opt(SPACE, "world"),
    "clear_inner": Opt(Bool(), False),
    "clear_outer": Opt(Bool(), False),
    "fill": Opt(Bool(), False),
    "select": Opt(SELECT),
}), mutates=True, destructive=True,
    description="Cut the mesh with a plane, like Blender's Bisect: new edges and vertices are "
                "created where the plane crosses faces. plane_normal points to the 'outer' "
                "side. clear_inner / clear_outer delete the geometry on that side (not both). "
                "fill closes the cut with new faces and needs exactly one of the clear flags "
                "(otherwise the cut edges would end up non-manifold); on a closed mesh it "
                "leaves a closed, manifold mesh. `select` limits the cut to those faces "
                "(default: the whole mesh). space applies to the plane (default world). "
                "Returns created {verts, edges, faces}, cut_edges (the new edges lying on the "
                "plane) and filled_faces. Refused if it would delete the entire mesh.")
def bisect(object, plane_point, plane_normal, space, clear_inner, clear_outer, fill, select):
    n = Vector(plane_normal)
    if n.length < 1e-9:
        raise OpError("invalid_params", "plane_normal must not be zero")
    if clear_inner and clear_outer:
        raise OpError("invalid_params", "clear_inner and clear_outer together would delete "
                                        "everything cut; give one")
    if fill and not (clear_inner or clear_outer):
        raise OpError("invalid_params", "fill needs clear_inner or clear_outer (without one the "
                                        "cut stays inside the mesh and filling it would make "
                                        "it non-manifold)")
    obj, ed = _open_edit(object)
    with ed as bm:
        if select is not None:
            fs = _resolve(bm, obj, select, "faces")
            geom = list(fs) + list({e for f in fs for e in f.edges}) + \
                list({v for f in fs for v in f.verts})
        else:
            geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        mw = Matrix.Identity(4) if space == "local" else _mw(obj)
        inv = _inverse(obj, mw)
        co = inv @ Vector(plane_point)
        no = mw.to_3x3().transposed() @ n   # normals transform by the inverse transpose
        if no.length < 1e-12:
            raise OpError("invalid_params", "the plane normal is degenerate in object space")
        no.normalize()
        _check_growth(bm, sum(1 for g in geom if isinstance(g, bmesh.types.BMFace)) + 1)
        snap = _snapshot(bm)
        ret = _bmo(bmesh.ops.bisect_plane, bm, geom=geom, dist=1e-6, plane_co=co,
                   plane_no=no, clear_outer=clear_outer, clear_inner=clear_inner)
        cut = [g for g in ret["geom_cut"] if isinstance(g, bmesh.types.BMEdge) and g.is_valid]
        filled = []
        if fill and cut:
            filled = list(_bmo(bmesh.ops.edgenet_fill, bm, edges=cut, mat_nr=0)["faces"])
            # match the winding of the neighbouring faces
            for f in filled:
                for e in f.edges:
                    others = [g for g in e.link_faces if g is not f]
                    if len(e.link_faces) == 2 and others:
                        la = next(l for l in f.loops if l.edge is e)
                        lb = next(l for l in others[0].loops if l.edge is e)
                        if la.vert is lb.vert:
                            bmesh.ops.reverse_faces(bm, faces=[f])
                        break
        if not bm.verts:
            raise OpError("invalid_params", "the cut would delete the entire mesh")
        created = _created(bm, *_fresh(bm, snap))
        cut_idx = sorted({e.index for e in cut if e.is_valid})
        counts = _counts(bm)
    return _result(obj, ed, counts, created, cut_edges=_compact(cut_idx),
                   cut_edges_count=len(cut_idx), filled_faces=len(filled))


# --------------------------------------------------------------------------- subdivide

@op("subdivide", Obj({
    "object": NAME,
    "select": SELECT,
    "domain": Opt(Enum(("faces", "edges")), "faces"),
    "cuts": Opt(Int(1, 10), 1),
    "smoothness": Opt(Float(0, 1), 0.0),
}), mutates=True,
    description="Subdivide the selected faces (all edges of the selected faces) or edges, like "
                "Blender's Subdivide: each cut adds evenly spaced vertices, quads and "
                "triangles are filled with a regular grid ((cuts+1)^2 faces per face). "
                "smoothness 0..1 pushes the new vertices towards a smooth surface (0 = flat). "
                "cuts 1..10; refused when the estimated result exceeds the geometry budget. "
                "Faces next to a subdivided edge are split to stay watertight (they become "
                "triangles/n-gons as in Blender). Returns created {verts, edges, faces}.")
def subdivide(object, select, domain, cuts, smoothness):
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, domain)
        if domain == "faces":
            edges = list({e for f in sel for e in f.edges})
            est = sum(((cuts + 1) ** 2) * max(1, (len(f.verts) + 2) // 4) for f in sel)
        else:
            edges = list(sel)
            adj = {f for e in sel for f in e.link_faces}
            est = len(adj) * (cuts + 1) ** 2
        _check_growth(bm, est)
        snap = _snapshot(bm)
        ret = _bmo(bmesh.ops.subdivide_edges, bm, edges=edges, cuts=cuts,
                   smooth=smoothness, use_grid_fill=True)
        created = _created(bm, _geom_split(ret["geom"])[0], *_fresh(bm, snap)[1:])
        counts = _counts(bm)
    return _result(obj, ed, counts, created)


# --------------------------------------------------------------------------- bridge

@op("bridge_edge_loops", Obj({
    "object": NAME,
    "select": SELECT,
    "twist": Opt(Int(-1000, 1000), 0),
    "merge": Opt(Bool(), False),
    "merge_factor": Opt(Float(0, 1), 0.5),
    "use_pairs": Opt(Bool(), False),
    "use_cyclic": Opt(Bool(), False),
}), mutates=True,
    description="Bridge edge loops, like Blender's Bridge Edge Loops: connect two or more "
                "separate loops (open or closed chains of edges, e.g. the rims of two holes) "
                "with a strip of quads. The selector is resolved in edges and must contain at "
                "least two loops. twist rotates the vertex matching between the loops, merge "
                "joins the loops into one (merge_factor 0..1 = where the merged vertices "
                "end up between the two), use_pairs bridges the loops two by two instead of "
                "in a chain, use_cyclic also closes the last loop to the first. Use "
                "get_boundary_loops to find hole rims. Returns created {edges, faces}.")
def bridge_edge_loops(object, select, twist, merge, merge_factor, use_pairs, use_cyclic):
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, "edges")
        parent = {}

        def find(v):
            while parent.setdefault(v, v) != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v
        for e in sel:
            parent[find(e.verts[0])] = find(e.verts[1])
        if len({find(v) for e in sel for v in e.verts}) < 2:
            raise OpError("invalid_params", "the selection contains a single connected chain "
                                            "of edges; bridge needs at least two separate loops")
        if use_pairs and len({find(e.verts[0]) for e in sel}) % 2:
            raise OpError("invalid_params", "use_pairs needs an even number of loops")
        _check_growth(bm, len(sel))
        snap = _snapshot(bm)
        ret = _bmo(bmesh.ops.bridge_loops, bm, edges=list(sel), use_pairs=use_pairs,
                   use_cyclic=use_cyclic, use_merge=merge, merge_factor=merge_factor,
                   twist_offset=twist)
        if not ret["faces"] and not merge:
            raise OpError("invalid_params", "nothing was bridged: the selection is not a set "
                                            "of loops")
        created = _created(bm, *_fresh(bm, snap))
        counts = _counts(bm)
    return _result(obj, ed, counts, created)


# --------------------------------------------------------------------------- spin

@op("spin", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "axis": DIR3,
    "center": Opt(LOC),
    "angle_deg": Float(-36000, 36000),
    "steps": Int(1, 512),
    "duplicate": Opt(Bool(), False),
    "merge_seam": Opt(Bool(), True),
    "space": Opt(SPACE, "world"),
}), mutates=True,
    description="Spin (lathe/revolve) the selected geometry around an axis, like Blender's "
                "Spin: each selected edge sweeps a ring of quads, e.g. a profile curve becomes "
                "a vase and a circle a torus. The selector is resolved in vertices (default "
                "all); edges/faces whose vertices are all selected are included. axis is the "
                "direction of the axis through `center` (default the space origin), "
                "angle_deg is the total angle (right-hand rule about the axis, not zero, up "
                "to +-36000) split into `steps` (1..512) steps, giving steps+1 copies of the "
                "profile unless |angle| is 360. duplicate keeps the original geometry as a "
                "separate profile instead of sweeping it. merge_seam welds the first and "
                "last profile together for a full 360 turn (so the result is closed). "
                "Refused when the estimated result exceeds the geometry budget. world space "
                "needs uniform object scale. Returns created {verts, edges, faces}.")
def spin(object, select, axis, center, angle_deg, steps, duplicate, merge_seam, space):
    ax = Vector(axis)
    if ax.length < 1e-9:
        raise OpError("invalid_params", "axis must not be zero")
    if abs(angle_deg) < 1e-9:
        raise OpError("invalid_params", "angle_deg must not be zero")
    obj, ed = _open_edit(object)
    _unit_scale(obj, space)    # world-space axes need a uniform scale (else the sweep shears)
    with ed as bm:
        verts = _resolve(bm, obj, select or {"all": True}, "verts")
        vs = set(verts)
        edges = [e for e in bm.edges if e.verts[0] in vs and e.verts[1] in vs]
        faces = [f for f in bm.faces if all(v in vs for v in f.verts)]
        _check_growth(bm, steps * (len(edges) + len(faces)))
        mw = Matrix.Identity(4) if space == "local" else _mw(obj)
        inv = _inverse(obj, mw)
        cen = inv @ Vector(center if center is not None else (0, 0, 0))
        axl = mw.to_3x3().inverted_safe() @ ax
        if axl.length < 1e-12:
            raise OpError("invalid_params", "the axis is degenerate in object space")
        axl.normalize()
        full = abs(abs(angle_deg) - 360.0) < 1e-6
        snap = _snapshot(bm)
        _bmo(bmesh.ops.spin, bm, geom=list(verts) + edges + faces, cent=cen, axis=axl,
             dvec=Vector((0, 0, 0)), angle=math.radians(angle_deg), steps=steps,
             use_merge=bool(merge_seam and full), use_duplicate=duplicate)
        new_v, new_e, new_f = _fresh(bm, snap)
        if not faces and new_f:
            # sweeping bare edges leaves the quads' winding to the edges' stored vertex order,
            # which is arbitrary: make the swept surface consistent (outward when closed)
            _bmo(bmesh.ops.recalc_face_normals, bm, faces=new_f)
        created = _created(bm, new_v, new_e, new_f)
        counts = _counts(bm)
    return _result(obj, ed, counts, created)


# --------------------------------------------------------------------------- smooth

@op("smooth_vertices", Obj({
    "object": NAME,
    "select": SELECT,
    "factor": Opt(Float(0, 1), 0.5),
    "repeat": Opt(Int(1, 100), 1),
    "axes": Opt(Obj({"x": Opt(Bool(), True), "y": Opt(Bool(), True), "z": Opt(Bool(), True)})),
}), mutates=True,
    description="Smooth the selected vertices (Laplacian: each moves towards the average of "
                "its neighbours), like Blender's Smooth Vertices. factor 0..1 per pass, repeat "
                "1..100 passes, axes {x, y, z} (object-local axes, default all) restricts the "
                "movement. Topology is unchanged. Returns moved (vertices that changed) and "
                "max_displacement (largest move, world units).")
def smooth_vertices(object, select, factor, repeat, axes):
    ax = axes or {"x": True, "y": True, "z": True}
    if not any(ax.values()):
        raise OpError("invalid_params", "axes: at least one of x, y, z must be true")
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, "verts")
        if len(sel) * repeat > MAX_SMOOTH_WORK:
            raise OpError("budget", f"{len(sel):,} vertices x {repeat} passes is over the "
                                    f"limit of {MAX_SMOOTH_WORK:,}")
        before = [v.co.copy() for v in sel]
        for _ in range(repeat):
            _bmo(bmesh.ops.smooth_vert, bm, verts=list(sel), factor=factor,
                 use_axis_x=ax["x"], use_axis_y=ax["y"], use_axis_z=ax["z"])
        m3 = _mw(obj).to_3x3()
        deltas = [(m3 @ (v.co - b)).length for v, b in zip(sel, before)]
        counts = _counts(bm)
    return {"object": obj.name, "moved": sum(1 for d in deltas if d > 1e-9),
            "max_displacement": max(deltas, default=0.0), "counts": counts,
            "after": _summary(ed.report)}


# --------------------------------------------------------------------------- symmetrize

SYM_DIRECTIONS = {"-X>+X": "-X", "+X>-X": "X", "-Y>+Y": "-Y", "+Y>-Y": "Y",
                  "-Z>+Z": "-Z", "+Z>-Z": "Z"}


@op("symmetrize", Obj({
    "object": NAME,
    "direction": Enum(tuple(SYM_DIRECTIONS)),
    "threshold": Opt(Float(0, 1), 1e-4),
}), mutates=True, destructive=True,
    description="Make the mesh symmetric about the object's local origin plane, like Blender's "
                "Symmetrize. direction '-X>+X' keeps the geometry on the negative X side and "
                "replaces the positive side with its mirror image ('+X>-X' the opposite; the "
                "same for Y and Z), in the object's local axes. Vertices within `threshold` "
                "(local units) of the plane are snapped onto it and welded. The half being "
                "replaced is deleted. Refused when the doubled mesh would exceed the geometry "
                "budget. Returns created {verts, edges, faces} (the mirrored half).")
def symmetrize(object, direction, threshold):
    obj, ed = _open_edit(object)
    with ed as bm:
        if not bm.faces and not bm.edges and not bm.verts:
            raise OpError("invalid_params", f"{obj.name} has no geometry")
        _check_growth(bm, len(bm.faces))
        snap = _snapshot(bm)
        _bmo(bmesh.ops.symmetrize, bm,
             input=list(bm.verts) + list(bm.edges) + list(bm.faces),
             direction=SYM_DIRECTIONS[direction], dist=threshold)
        created = _created(bm, *_fresh(bm, snap))
        counts = _counts(bm)
    return _result(obj, ed, counts, created)


# --------------------------------------------------------------------------- separate

def _loose_parts(bm):
    """Vertex-index sets of the connected components (wire edges and lone vertices count)."""
    parent = list(range(len(bm.verts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in bm.edges:
        a, b = find(e.verts[0].index), find(e.verts[1].index)
        if a != b:
            parent[a] = b
    groups = {}
    for v in bm.verts:
        groups.setdefault(find(v.index), set()).add(v.index)
    return sorted(groups.values(), key=min)


def _part_from_faces(bm, keep):
    """Copy of bm holding only the faces whose index is in `keep` (loose geometry dropped)."""
    p = bm.copy()
    _reindex(p)
    drop = [f for f in p.faces if f.index not in keep]
    if drop:
        bmesh.ops.delete(p, geom=drop, context="FACES")
    wire = [e for e in p.edges if not e.link_faces]
    if wire:
        bmesh.ops.delete(p, geom=wire, context="EDGES")
    lone = [v for v in p.verts if not v.link_edges]
    if lone:
        bmesh.ops.delete(p, geom=lone, context="VERTS")
    return p


def _part_from_verts(bm, keep):
    p = bm.copy()
    _reindex(p)
    drop = [v for v in p.verts if v.index not in keep]
    if drop:
        bmesh.ops.delete(p, geom=drop, context="VERTS")
    return p


@op("separate", Obj({
    "object": NAME,
    "mode": Enum(("selection", "loose_parts", "material")),
    "select": Opt(SELECT),
    "name_prefix": Opt(NAME),
}), mutates=True, destructive=True,
    description="Split a mesh object into several, like Blender's Separate (P). Each new "
                "object is a copy of the source object (same transform, parent, collections, "
                "modifiers, material slots, UV maps) with its own mesh holding one part; the "
                "separated geometry is removed from the source. mode 'selection': the faces "
                "chosen by `select` (required, resolved in faces) become ONE new object; "
                "refused if that would leave the source empty. 'loose_parts': every "
                "disconnected piece (wire edges and lone vertices count as pieces) except the "
                "first (the one containing the lowest vertex index), which stays in the "
                "source. 'material': one new object per used material slot except the "
                "lowest-numbered one, which stays (material slots are kept on every object so "
                "face material indices stay valid). At most 256 new objects. name_prefix "
                "names them (Blender appends .001 etc. if taken; default: the source name). "
                "Shape keys are not carried over. Returns new_objects (names, with vertex and "
                "face counts) and the source's counts.")
def separate(object, mode, select, name_prefix):
    obj, ed = _open_edit(object)
    if mode == "selection" and select is None:
        raise OpError("invalid_params", "mode 'selection' needs `select`")
    if mode != "selection" and select is not None:
        raise OpError("invalid_params", "`select` only applies to mode 'selection'")
    src_mesh = obj.data
    parts = []
    try:
        with ed as bm:
            _reindex(bm)
            if mode == "selection":
                sel = _resolve(bm, obj, select, "faces")
                chosen = {f.index for f in sel}
                gone = set()
                for v in bm.verts:
                    if v.link_faces and all(f.index in chosen for f in v.link_faces) \
                            and all(e.link_faces for e in v.link_edges):
                        gone.add(v.index)
                if len(gone) == len(bm.verts):
                    raise OpError("invalid_params", "that would leave the source object empty; "
                                                    "select fewer faces")
                groups = [("faces", chosen)]
                to_remove = ("faces", chosen)
            elif mode == "loose_parts":
                comps = _loose_parts(bm)
                if len(comps) < 2:
                    raise OpError("invalid_params", f"{obj.name} has a single connected part; "
                                                    "nothing to separate")
                groups = [("verts", c) for c in comps[1:]]
                to_remove = ("verts", set().union(*comps[1:]))
            else:
                by_mat = {}
                for f in bm.faces:
                    by_mat.setdefault(f.material_index, set()).add(f.index)
                if len(by_mat) < 2:
                    raise OpError("invalid_params", f"{obj.name} uses a single material slot; "
                                                    "nothing to separate")
                order = sorted(by_mat)
                groups = [("faces", by_mat[m]) for m in order[1:]]
                to_remove = ("faces", set().union(*(by_mat[m] for m in order[1:])))
            if len(groups) > MAX_SEPARATE_PARTS:
                raise OpError("budget", f"would create {len(groups):,} objects (limit "
                                        f"{MAX_SEPARATE_PARTS})")
            if len(groups) > 1 and len(groups) * (len(bm.verts) + len(bm.faces)) > MAX_SEPARATE_WORK:
                raise OpError("budget", "too many parts for a mesh this size; separate fewer "
                                        "parts at a time (use mode 'selection')")
            for kind, keep in groups:
                parts.append((_part_from_faces if kind == "faces" else _part_from_verts)(bm, keep))
            kind, rm = to_remove
            if kind == "faces":
                bm.faces.ensure_lookup_table()
                bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index in rm],
                                 context="FACES")
            else:
                bm.verts.ensure_lookup_table()
                bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index in rm],
                                 context="VERTS")
            counts = _counts(bm)
        # source written back; now create the objects (validation is all done)
        prefix = name_prefix or obj.name
        mats = list(src_mesh.materials)
        collections = list(obj.users_collection) or [bpy.context.scene.collection]
        made = []
        try:
            for p in parts:
                me = bpy.data.meshes.new(prefix)
                p.to_mesh(me)
                for m in mats:
                    me.materials.append(m)
                new = obj.copy()
                new.data = me
                new.name = prefix
                for c in collections:
                    c.objects.link(new)
                made.append(new)
        except Exception:
            for new in made:
                bpy.data.objects.remove(new)
            raise
        info = [{"name": o.name, "vertices": len(o.data.vertices), "faces": len(o.data.polygons)}
                for o in made]
    finally:
        for p in parts:
            p.free()
    return {"object": obj.name, "mode": mode, "new_objects": [i["name"] for i in info],
            "parts": info, "counts": counts, "after": _summary(ed.report)}


# --------------------------------------------------------------------------- snap

@op("snap_vertices", Obj({
    "object": NAME,
    "select": SELECT,
    "to": Union(Obj({"grid": Float(1e-6, 1e6)}), Enum(("cursor",)),
                Obj({"object": NAME, "mode": Enum(("nearest_surface", "nearest_vertex"))})),
    "space": Opt(SPACE, "world"),
}), mutates=True,
    description="Snap the selected vertices (the selector is resolved in vertices), like "
                "Blender's snapping. to: {\"grid\": increment} rounds each vertex coordinate "
                "to the grid in `space` (default world); \"cursor\" moves them all onto the 3D "
                "cursor; {\"object\": name, \"mode\": 'nearest_surface' | 'nearest_vertex'} "
                "moves each onto the closest point on the surface / the closest vertex of "
                "another MESH object (its evaluated mesh, world space). The target must "
                "not be the edited object. Topology is unchanged. Returns moved (vertices "
                "that changed) and max_distance (world units).")
def snap_vertices(object, select, to, space):
    obj, ed = _open_edit(object)
    target = None
    if isinstance(to, dict) and "object" in to:
        target = _mesh_target(to["object"], "snap target")
        if target == obj:
            raise OpError("invalid_params", f"{obj.name} is both the snap target and the "
                                            "edited object")
    with ed as bm:
        sel = _resolve(bm, obj, select, "verts")
        mw = Matrix.Identity(4) if space == "local" else _mw(obj)
        mw_world = _mw(obj)
        inv = _inverse(obj, mw)
        inv_world = _inverse(obj, mw_world)
        if isinstance(to, dict) and "grid" in to:
            g = to["grid"]
            plan = [(v, inv @ Vector(round(c / g) * g for c in (mw @ v.co))) for v in sel]
        elif to == "cursor":
            c = inv_world @ bpy.context.scene.cursor.location
            plan = [(v, c.copy()) for v in sel]
        elif to["mode"] == "nearest_surface":
            bvh = _bvh(target)
            plan = [(v, inv_world @ _nearest_surface(target, bvh, mw_world @ v.co))
                    for v in sel]
        else:
            ev = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
            n = len(ev.data.vertices)
            if n == 0:
                raise OpError("not_found", f"{target.name} has no vertices")
            if n > MAX_SNAP_VERTS:
                raise OpError("budget", f"{target.name} has more than {MAX_SNAP_VERTS:,} "
                                        "vertices")
            kd = KDTree(n)
            tw = ev.matrix_world
            for i, tv in enumerate(ev.data.vertices):
                kd.insert(tw @ tv.co, i)
            kd.balance()
            plan = [(v, inv_world @ Vector(kd.find(mw_world @ v.co)[0])) for v in sel]
        m3 = mw_world.to_3x3()
        moved, far = 0, 0.0
        for v, new in plan:
            d = (m3 @ (new - v.co)).length
            if d > 1e-9:
                moved += 1
                far = max(far, d)
            v.co = new
        counts = _counts(bm)
    return {"object": obj.name, "moved": moved, "max_distance": far, "counts": counts,
            "after": _summary(ed.report)}


# --------------------------------------------------------------------------- tris, poke

@op("triangulate_faces", Obj({
    "object": NAME,
    "select": SELECT,
    "quad_method": Opt(Enum(("BEAUTY", "FIXED", "FIXED_ALTERNATE", "SHORTEST_DIAGONAL",
                             "LONGEST_DIAGONAL")), "BEAUTY"),
    "ngon_method": Opt(Enum(("BEAUTY", "EAR_CLIP")), "BEAUTY"),
}), mutates=True,
    description="Triangulate the selected faces (triangles are left alone), like Blender's "
                "Ctrl+T. quad_method picks the diagonal of quads (BEAUTY, FIXED, "
                "FIXED_ALTERNATE, SHORTEST_DIAGONAL, LONGEST_DIAGONAL), ngon_method how "
                "n-gons are split (BEAUTY, EAR_CLIP). Returns created {edges, faces}: the "
                "triangles that came out of quads/n-gons (including the one that keeps the "
                "original face's identity) and the new diagonals; `triangulated` = number of "
                "faces that were split.")
def triangulate_faces(object, select, quad_method, ngon_method):
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, "faces")
        todo = [f for f in sel if len(f.verts) > 3]
        _check_growth(bm, sum(len(f.verts) - 3 for f in todo))
        ret = _bmo(bmesh.ops.triangulate, bm, faces=list(sel), quad_method=quad_method,
                   ngon_method=ngon_method)
        created = _created(bm, edges=ret["edges"], faces=ret["faces"])
        counts = _counts(bm)
    return _result(obj, ed, counts, created, triangulated=len(todo))


@op("poke_faces", Obj({
    "object": NAME,
    "select": SELECT,
    "offset": Opt(Float(-1e6, 1e6), 0.0),
    "relative": Opt(Bool(), False),
    "center": Opt(Enum(("mean_weighted", "mean", "bounds")), "mean_weighted"),
    "space": Opt(SPACE, "world"),
}), mutates=True,
    description="Poke faces, like Blender's Poke Faces: add a vertex in the middle of each "
                "selected face and connect it to the face's corners (a triangle fan). offset "
                "moves the new vertex along the face normal (negative = into the mesh); with "
                "relative=true it is a fraction of the face's size instead of a length "
                "(lengths are in `space`, world needs uniform object scale). center: "
                "'mean_weighted' (area-weighted centre), 'mean' or 'bounds'. Returns created "
                "{verts, edges, faces}.")
def poke_faces(object, select, offset, relative, center, space):
    obj, ed = _open_edit(object)
    k = 1.0 if relative else _unit_scale(obj, space)
    with ed as bm:
        sel = _resolve(bm, obj, select, "faces")
        _check_growth(bm, sum(len(f.verts) - 1 for f in sel))
        bm.normal_update()
        ret = _bmo(bmesh.ops.poke, bm, faces=list(sel), offset=offset * k,
                   center_mode=center.upper(), use_relative_offset=relative)
        new_v = ret["verts"]
        new_e = {e for v in new_v for e in v.link_edges}
        created = _created(bm, new_v, new_e, ret["faces"])
        counts = _counts(bm)
    return _result(obj, ed, counts, created)

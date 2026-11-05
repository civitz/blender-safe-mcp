"""Mesh selector and core mesh-editing ops (plan phase 3a; 3b builds on the same helpers).

Every edit op takes a *selector* (SELECT) instead of raw indices, because indices change after
any topology edit. `_resolve(bm, obj, select, domain)` turns a selector into BMesh elements;
ops return the indices of the elements they create (`created`), which match what
get_mesh_elements reports afterwards (bm.to_mesh writes elements in bm iteration order).

Rules as in ops.py: strict schemas, objects looked up by name, bmesh only (no bpy.ops), and every
op validates everything (selector, geometry budget, degenerate directions) and raises OpError
*before* the bmesh is modified. The edit happens on a copy (_MeshEdit) that is written back only
when the op finishes without error.
"""

import heapq
import math

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

import bmesh

from .falloff import FALLOFFS, falloff_weight
from .ops import LOC, MAX_POLYS, NAME, SCALE, _MeshEdit, _mesh_bm, _obj, _summary, _vec, op
from .schema import Bool, Enum, Float, Int, List, Obj, Opt, Str, Union, Vec
from .server import OpError

MAX_ELEM_INDEX = 9_999_999
ELEM_IDX = Int(0, MAX_ELEM_INDEX, description="Element index (changes after topology edits)")
MAX_SELECT_LIST = 100_000
MAX_RETURNED = 5000          # cap on items in index lists in results (the *_count keys give the totals)
MAX_PROPORTIONAL_VERTS = 500_000
DOMAINS = ("verts", "edges", "faces")

IDX_ITEMS = List(Union(ELEM_IDX, List(ELEM_IDX, 2, 2)), 1, MAX_SELECT_LIST,
                 description="Indices; an item may be a [first, last] inclusive range, as "
                             "ops return them (e.g. [3, [16, 767]])")

DIR3 = Vec(3, -1e6, 1e6, description="XYZ direction (need not be normalised)")
SPACE = Enum(("world", "local"), description="Coordinate space of positions/vectors "
                                             "(default world)")

SELECT = Obj({
    "verts": Opt(IDX_ITEMS),
    "edges": Opt(IDX_ITEMS),
    "faces": Opt(IDX_ITEMS),
    "all": Opt(Bool(description="Every element of the op's domain")),
    "normal": Opt(Obj({
        "direction": DIR3,
        "max_angle_deg": Float(0, 180),
    }, description="Faces whose normal is within max_angle_deg of direction")),
    "box": Opt(Obj({"min": LOC, "max": LOC},
                   description="Elements fully inside the axis-aligned box (faces: all "
                               "vertices; edges: both vertices)")),
    "sphere": Opt(Obj({"center": LOC, "radius": Float(1e-9, 1e6)},
                      description="Elements fully inside the sphere")),
    "material": Opt(Union(Str(max_len=63), Int(0, 63)),),
    "boundary": Opt(Bool(description="Boundary (hole) edges and their vertices")),
    "edge_loop": Opt(ELEM_IDX),
    "edge_ring": Opt(ELEM_IDX),
    "combine": Opt(Enum(("and", "or")), "and"),
    "space": Opt(Enum(("world", "local")), "world"),
    "grow": Opt(Int(0, 20), 0),
    "invert": Opt(Bool()),
    "expect_counts": Opt(List(Int(0, 100_000_000), 3, 3)),
}, description="Which elements to act on. Criteria (verts, edges, faces, all, normal, box, "
               "sphere, material, boundary, edge_loop, edge_ring) are combined with `combine` "
               "(default and); then `grow` (steps of adjacency), then `invert`. Element "
               "domains convert like Blender's select modes: a face is selected iff all its "
               "vertices (or, from an edge selection, all its edges) are; an edge iff both its "
               "vertices are; selecting a face or edge selects its vertices. `space` applies to "
               "normal, box and sphere. `expect_counts` is the [vertices, edges, faces] from an "
               "earlier result: if the mesh changed since, the call fails with stale_indices "
               "instead of acting on the wrong elements.")

CRITERIA = ("verts", "edges", "faces", "all", "normal", "box", "sphere", "material", "boundary",
            "edge_loop", "edge_ring")


# --------------------------------------------------------------------------- helpers

def _mw(obj):
    # matrix_world is only refreshed by a depsgraph update; ops chained in one tick need it.
    bpy.context.view_layer.update()
    return obj.matrix_world.copy()


def _normal_matrix(mw):
    return mw.to_3x3().inverted_safe().transposed()


def _counts(bm):
    return [len(bm.verts), len(bm.edges), len(bm.faces)]


def _reindex(bm):
    for s in (bm.verts, bm.edges, bm.faces):
        s.ensure_lookup_table()
        s.index_update()


def _compact(indices):
    """Sorted unique indices as a compact list: runs of 2+ become [first, last] (inclusive),
    single indices stay plain ints; capped at MAX_RETURNED items. The selector accepts it back."""
    out, idx = [], sorted(set(indices))
    i = 0
    while i < len(idx) and len(out) < MAX_RETURNED:
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(idx[i] if i == j else [idx[i], idx[j]])
        i = j + 1
    return out


def _expand(items, n, key):
    """Selector index items (ints or [first, last] ranges) -> index set, all < n."""
    singles, ranges = set(), []
    for it in items:
        a, b = (it, it) if isinstance(it, int) else it
        if a > b:
            raise OpError("invalid_params", f"{key}: range [{a}, {b}] has first > last")
        if b >= n:
            raise OpError("not_found", f"{key}: no {key[:-1]} {b} (mesh has {n}); "
                                       "re-query indices")
        if a == b:
            singles.add(a)
        else:
            ranges.append((a, b))
    # merge ranges first so overlapping ones cost no more than the mesh size
    ranges.sort()
    merged = []
    for a, b in ranges:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    for a, b in merged:
        singles.update(range(a, b + 1))
    return singles


def _created(bm, verts=(), edges=(), faces=()):
    """Indices of new elements, by kind, compacted (the *_count keys give the real totals)."""
    _reindex(bm)
    out = {}
    for key, elems in (("verts", verts), ("edges", edges), ("faces", faces)):
        idx = {e.index for e in elems}
        out[key] = _compact(idx)
        out[key + "_count"] = len(idx)
    return out


def _open_edit(object):
    obj = _obj(object)
    if obj.type == "MESH" and obj.data.library is not None:
        raise OpError("invalid_params", f"{obj.name}: linked data cannot be edited")
    return obj, _MeshEdit(obj)


def _check_growth(bm, extra_faces):
    if len(bm.faces) + extra_faces > MAX_POLYS:
        raise OpError("budget", f"would produce ~{len(bm.faces) + extra_faces:,} faces "
                                f"(limit {MAX_POLYS:,})")


def _elem(seq, i, kind):
    if i >= len(seq):
        raise OpError("not_found", f"no {kind} {i} (mesh has {len(seq)}); re-query indices")
    return seq[i]


# --------------------------------------------------------------------------- the selector

def _make_converters(bm):
    """{(src, dst): fn(index set) -> index set} for the domain conversions in SELECT."""
    E, F = bm.edges, bm.faces
    # edges/faces -> verts
    def e2v(s):
        return {v.index for i in s for v in E[i].verts}

    def f2v(s):
        return {v.index for i in s for v in F[i].verts}

    def v2e(s):
        return {e.index for e in E if e.verts[0].index in s and e.verts[1].index in s}

    def f2e(s):
        return {e.index for i in s for e in F[i].edges}

    def v2f(s):
        return {f.index for f in F if all(v.index in s for v in f.verts)}

    def e2f(s):
        return {f.index for f in F if all(e.index in s for e in f.edges)}
    ident = (lambda s: set(s))
    return {("verts", "verts"): ident, ("edges", "edges"): ident, ("faces", "faces"): ident,
            ("edges", "verts"): e2v, ("faces", "verts"): f2v, ("verts", "edges"): v2e,
            ("faces", "edges"): f2e, ("verts", "faces"): v2f, ("edges", "faces"): e2f}


def _grow(bm, domain, idx):
    """One step of growth by vertex adjacency (Blender's Select More)."""
    if domain == "verts":
        return idx | {o.index for i in idx for e in bm.verts[i].link_edges
                      for o in e.verts}
    if domain == "edges":
        vs = {v.index for i in idx for v in bm.edges[i].verts}
        return idx | {e.index for v in vs for e in bm.verts[v].link_edges}
    vs = {v.index for i in idx for v in bm.faces[i].verts}
    return idx | {f.index for v in vs for f in bm.verts[v].link_faces}


def _edge_loop(e0):
    """Edge indices of the loop through e0: continue through valence-4 interior vertices
    (surrounded by quads) to the edge that shares no face with the current one, and along a
    boundary through valence-3 boundary vertices. Stops at poles and non-quads."""
    out = {e0.index}
    for start in e0.verts:
        e, v = e0, start
        while True:
            nxt = _loop_next(e, v)
            if nxt is None or nxt.index in out:
                break
            out.add(nxt.index)
            e, v = nxt, nxt.other_vert(v)
    return out


def _loop_next(e, v):
    if len(e.link_faces) > 2:
        return None
    edges = v.link_edges
    if e.is_boundary:
        if len(edges) == 3 and v.is_boundary:
            others = [c for c in edges if c.index != e.index and c.is_boundary]
            return others[0] if len(others) == 1 else None
        return None
    if len(edges) != 4 or len(e.link_faces) != 2 or any(len(f.verts) != 4 for f in v.link_faces):
        return None
    fs = {f.index for f in e.link_faces}
    opp = [c for c in edges if c.index != e.index and not ({f.index for f in c.link_faces} & fs)]
    return opp[0] if len(opp) == 1 else None


def _edge_ring(e0):
    """Edge indices of the ring through e0: the parallel edges across consecutive quads."""
    out = {e0.index}
    for f0 in e0.link_faces:
        e, f = e0, f0
        while len(f.verts) == 4:
            ev = {v.index for v in e.verts}
            opp = [c for c in f.edges if not ({v.index for v in c.verts} & ev)]
            if len(opp) != 1 or opp[0].index in out:
                break
            out.add(opp[0].index)
            e = opp[0]
            nxt = [g for g in e.link_faces if g.index != f.index]
            if len(e.link_faces) != 2 or not nxt:
                break
            f = nxt[0]
    return out


def _resolve(bm, obj, select, domain):
    """Selector -> list of BMVert/BMEdge/BMFace of `domain`, sorted by index.

    Raises stale_indices (expect_counts mismatch, checked first), not_found (index out of
    range / unknown material), invalid_params (no criteria, or nothing matched). Reads only;
    the bmesh is not modified apart from refreshing element indices."""
    if domain not in DOMAINS:
        raise ValueError(domain)
    ec = select.get("expect_counts")
    if ec is not None and list(ec) != _counts(bm):
        raise OpError("stale_indices", f"mesh has {_counts(bm)} [vertices, edges, faces] but "
                                       f"the selector expects {list(ec)}: indices from an "
                                       "earlier result are out of date; re-query them")
    present = [k for k in CRITERIA if select.get(k) is not None and select.get(k) is not False]
    if not present:
        raise OpError("invalid_params", "selector has no criteria; give one of "
                                        f"{list(CRITERIA)} (all: true selects everything)")
    _reindex(bm)
    seqs = {"verts": bm.verts, "edges": bm.edges, "faces": bm.faces}
    conv = _make_converters(bm)
    local = select.get("space", "world") == "local"
    mw = Matrix.Identity(4) if local else _mw(obj)

    parts = []  # (domain, index set)
    for key in ("verts", "edges", "faces"):
        if select.get(key) is not None:
            parts.append((key, _expand(select[key], len(seqs[key]), key)))
    if select.get("all"):
        parts.append((domain, set(range(len(seqs[domain])))))
    if select.get("normal") is not None:
        n = select["normal"]
        d = Vector(n["direction"])
        if d.length < 1e-9:
            raise OpError("invalid_params", "normal.direction must not be zero")
        d.normalize()
        nm = _normal_matrix(mw)
        cos_min = math.cos(math.radians(n["max_angle_deg"]))
        bm.faces.ensure_lookup_table()
        for f in bm.faces:
            f.normal_update()
        parts.append(("faces", {f.index for f in bm.faces
                                if (nm @ f.normal).normalized().dot(d) >= cos_min - 1e-9}))
    if select.get("box") is not None:
        lo, hi = select["box"]["min"], select["box"]["max"]
        if any(a > b for a, b in zip(lo, hi)):
            raise OpError("invalid_params", "box.min must be <= box.max on every axis")
        parts.append(("verts", {v.index for v in bm.verts
                                if all(lo[k] <= c <= hi[k] for k, c in enumerate(mw @ v.co))}))
    if select.get("sphere") is not None:
        c = Vector(select["sphere"]["center"])
        r = select["sphere"]["radius"]
        parts.append(("verts", {v.index for v in bm.verts if ((mw @ v.co) - c).length <= r}))
    if select.get("material") is not None:
        m = select["material"]
        if isinstance(m, str):
            slots = [i for i, s in enumerate(obj.material_slots)
                     if s.material is not None and s.material.name == m]
            if not slots:
                raise OpError("not_found", f"{obj.name} has no material slot using {m!r}")
        else:
            if m >= max(len(obj.material_slots), 1):
                raise OpError("not_found", f"{obj.name} has no material slot {m}")
            slots = [m]
        parts.append(("faces", {f.index for f in bm.faces if f.material_index in slots}))
    if select.get("boundary"):
        parts.append(("edges", {e.index for e in bm.edges if len(e.link_faces) == 1}))
    for key, walk in (("edge_loop", _edge_loop), ("edge_ring", _edge_ring)):
        if select.get(key) is not None:
            parts.append(("edges", walk(_elem(bm.edges, select[key], "edge"))))

    sets = [conv[(d, domain)](s) if d != domain else set(s) for d, s in parts]
    result = sets[0]
    for s in sets[1:]:
        result = result & s if select.get("combine", "and") == "and" else result | s
    for _ in range(select.get("grow") or 0):
        result = _grow(bm, domain, result)
    if select.get("invert"):
        result = set(range(len(seqs[domain]))) - result
    if not result:
        raise OpError("invalid_params", "selector matched nothing")
    seq = seqs[domain]
    return [seq[i] for i in sorted(result)]


# --------------------------------------------------------------------------- read op

@op("get_mesh_elements", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "domain": Opt(Enum(DOMAINS), "verts"),
    "space": Opt(Enum(("world", "local")), "world"),
    "offset": Opt(Int(0, MAX_ELEM_INDEX), 0),
    "limit": Opt(Int(1, 2000), 200),
}), description="List mesh elements (vertices, edges or faces) chosen by a selector "
                "(default: all), with the indices the edit ops use. verts: index, co, normal; "
                "edges: index, verts, length, is_boundary, seam, sharp; faces: index, verts, "
                "center, normal, area, material_index. `total` is the number of matches, "
                "`counts` is [vertices, edges, faces] for the selector's expect_counts. "
                "Positions are in `space` (default world). Works in Edit mode too (reads the "
                "live edit mesh). An empty match is an empty list, not an error.")
def get_mesh_elements(object, select, domain, space, offset, limit):
    obj = _obj(object)
    bm = _mesh_bm(obj)
    try:
        mw = _mw(obj) if space == "world" else Matrix.Identity(4)
        nm = _normal_matrix(mw)
        if select is None:
            select = {"all": True}
        try:
            elems = _resolve(bm, obj, select, domain)
        except OpError as e:
            if e.message != "selector matched nothing":
                raise
            elems = []
        total = len(elems)
        page = elems[offset: offset + limit]
        out = []
        for el in page:
            if domain == "verts":
                out.append({"index": el.index, "co": _vec(mw @ el.co),
                            "normal": _vec((nm @ el.normal).normalized())})
            elif domain == "edges":
                a, b = el.verts
                out.append({"index": el.index, "verts": [a.index, b.index],
                            "length": round(float((mw @ a.co - mw @ b.co).length), 6),
                            "is_boundary": len(el.link_faces) == 1, "seam": bool(el.seam),
                            "sharp": not el.smooth})
            else:
                el.normal_update()
                pts = [mw @ v.co for v in el.verts]
                newell = Vector()
                for p, q in zip(pts, pts[1:] + pts[:1]):
                    newell += p.cross(q)
                out.append({"index": el.index, "verts": [v.index for v in el.verts],
                            "center": _vec(sum(pts, Vector()) / len(pts)),
                            "normal": _vec((nm @ el.normal).normalized()),
                            "area": round(newell.length / 2.0, 6),
                            "material_index": el.material_index})
        return {"object": obj.name, "domain": domain, "space": space, "total": total,
                "offset": offset, "counts": _counts(bm), "elements": out}
    finally:
        bm.free()


# --------------------------------------------------------------------------- extrude / inset

def _unit_scale(obj, space):
    """Factor turning world lengths into object-local lengths (uniform scale only)."""
    if space == "local":
        return 1.0
    s = obj.matrix_world.to_scale()
    if max(s) - min(s) > 1e-4 * max(abs(max(s)), 1e-9) or min(s) <= 0:
        raise OpError("invalid_params", f"{obj.name} has a non-uniform or negative scale; give "
                                        "lengths with space='local' (object units)")
    return 1.0 / s[0]


@op("extrude", Obj({
    "object": NAME,
    "select": SELECT,
    "mode": Opt(Enum(("region", "individual_faces", "edges", "verts")), "region"),
    "distance": Opt(Float(-1e6, 1e6)),
    "vector": Opt(LOC),
    "space": Opt(Enum(("world", "local")), "world"),
}), mutates=True,
    description="Extrude selected elements, like Blender's E. mode: 'region' (faces; adjacent "
                "faces extrude together, side walls are made, the original faces are removed "
                "as in the interactive tool), 'individual_faces', 'edges' (edge-only extrude: "
                "new edge ring plus side faces) or 'verts' (a new vertex and edge per vertex). "
                "The selector is resolved in faces (region, individual_faces), edges or "
                "vertices. Move the new geometry with `distance` (along the normal: the "
                "region's area-weighted average face normal; each face's own normal for "
                "individual_faces; average vertex normal for edges; each vertex's own normal "
                "for verts) or with `vector` (+ `space`); give neither to extrude in place and "
                "then use transform_elements. Returns created {verts, edges, faces} indices "
                "(+ *_count), and cap_faces (the moved copies of the original faces) for the "
                "face modes, so you can chain extrude/inset/transform_elements.")
def extrude(object, select, mode, distance, vector, space):
    if distance is not None and vector is not None:
        raise OpError("invalid_params", "give distance or vector, not both")
    obj, ed = _open_edit(object)
    domain = {"region": "faces", "individual_faces": "faces", "edges": "edges",
              "verts": "verts"}[mode]
    with ed as bm:
        sel = _resolve(bm, obj, select, domain)
        mw = Matrix.Identity(4) if space == "local" else _mw(obj)
        try:
            inv3 = mw.to_3x3().inverted()
        except ValueError:
            raise OpError("invalid_params", f"{obj.name} has a zero scale") from None
        nm = _normal_matrix(mw)
        bm.normal_update()

        def to_local(world_vec):
            return inv3 @ world_vec

        def along(normal_vec, amount):
            n = nm @ normal_vec
            if n.length < 1e-12:
                raise OpError("invalid_params", "the selection has no usable normal "
                                                "(zero-area or opposing faces); use `vector`")
            return to_local(n.normalized() * amount)

        # ---- compute displacements first (validation), then mutate
        if mode == "region":
            extra = len(sel) + sum(len(f.verts) for f in sel)
        elif mode == "individual_faces":
            extra = len(sel) + sum(len(f.verts) for f in sel)
        elif mode == "edges":
            extra = len(sel) * 2
        else:
            extra = 0
        _check_growth(bm, extra)
        fixed = None
        if vector is not None:
            fixed = to_local(Vector(vector))
        if mode == "region":
            if distance is not None:
                fixed = along(sum((f.normal * f.calc_area() for f in sel), Vector()), distance)
            es = {e for f in sel for e in f.edges}
            vs = {v for f in sel for v in f.verts}
            ret = bmesh.ops.extrude_face_region(bm, geom=list(sel) + list(es) + list(vs))["geom"]
            new_v = [g for g in ret if isinstance(g, bmesh.types.BMVert)]
            if fixed is not None:
                for v in new_v:
                    v.co += fixed
            nv = set(new_v)
            new_e = {e for v in new_v for e in v.link_edges}
            new_f = {f for v in new_v for f in v.link_faces}
            caps = [f for f in new_f if all(v in nv for v in f.verts)]
        elif mode == "individual_faces":
            caps = list(bmesh.ops.extrude_discrete_faces(bm, faces=list(sel))["faces"])
            for f in caps:
                f.normal_update()
                d = along(f.normal, distance) if distance is not None else fixed
                if d is not None:
                    for v in f.verts:
                        v.co += d
            new_v = list({v for f in caps for v in f.verts})
            new_e = {e for v in new_v for e in v.link_edges}
            new_f = {f for v in new_v for f in v.link_faces}
        elif mode == "edges":
            if distance is not None:
                fixed = along(sum((v.normal for e in sel for v in e.verts), Vector()), distance)
            ret = bmesh.ops.extrude_edge_only(bm, edges=list(sel))["geom"]
            new_v = [g for g in ret if isinstance(g, bmesh.types.BMVert)]
            if fixed is not None:
                for v in new_v:
                    v.co += fixed
            new_e = {e for v in new_v for e in v.link_edges}
            new_f = {f for v in new_v for f in v.link_faces}
            caps = []
        else:
            moves = {v.index: along(v.normal, distance) for v in sel} if distance is not None else {}
            old = set(sel)
            ret = bmesh.ops.extrude_vert_indiv(bm, verts=list(sel))
            new_v = list(ret["verts"])
            new_e = set(ret["edges"])
            new_f = set()
            caps = []
            for e in new_e:  # each new edge joins an old vertex to its new copy
                a, b = e.verts
                o, n_ = (a, b) if a in old else (b, a)
                if distance is not None:
                    n_.co += moves[o.index]
                elif fixed is not None:
                    n_.co += fixed
        created = _created(bm, new_v, new_e, new_f)
        _reindex(bm)
        cap_idx = sorted(f.index for f in caps)
        counts = _counts(bm)
    res = {"object": obj.name, "mode": mode, "created": created, "counts": counts,
           "after": _summary(ed.report)}
    if mode in ("region", "individual_faces"):
        res["cap_faces"] = _compact(cap_idx)
    return res


@op("inset", Obj({
    "object": NAME,
    "select": SELECT,
    "thickness": Float(0, 1e6),
    "depth": Opt(Float(-1e6, 1e6), 0.0),
    "individual": Opt(Bool(), False),
    "even_offset": Opt(Bool(), True),
    "outset": Opt(Bool(), False),
    "space": Opt(Enum(("world", "local")), "world"),
}), mutates=True,
    description="Inset faces, like Blender's I: a smaller copy of each face (or of the whole "
                "region's outline when individual=false) surrounded by a rim of new faces. "
                "thickness is the inset distance, depth moves the inner faces along the "
                "normal (negative = into the mesh), even_offset keeps thickness constant "
                "around corners, outset grows instead of shrinks (region only). Lengths are in "
                "`space` (world needs uniform object scale). Returns created.faces = the inner "
                "faces (they keep the input faces' identity), created.rim_faces = the new rim "
                "faces, plus new verts/edges, so 'inset then extrude' can chain with "
                "select {faces: created.faces}.")
def inset(object, select, thickness, depth, individual, even_offset, outset, space):
    if individual and outset:
        raise OpError("invalid_params", "outset only applies to region inset (individual=false)")
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, "faces")
        k = _unit_scale(obj, space)
        _check_growth(bm, sum(len(f.verts) for f in sel))
        bm.normal_update()
        if individual:
            rim = bmesh.ops.inset_individual(bm, faces=list(sel), thickness=thickness * k,
                                             depth=depth * k, use_even_offset=even_offset)["faces"]
        else:
            rim = bmesh.ops.inset_region(bm, faces=list(sel), use_boundary=True,
                                         use_even_offset=even_offset, thickness=thickness * k,
                                         depth=depth * k, use_outset=outset)["faces"]
        inner = [f for f in sel if f.is_valid]
        inner_v = {v for f in inner for v in f.verts}
        inner_e = {e for f in inner for e in f.edges}
        new_v = {v for f in rim for v in f.verts} - inner_v
        new_e = {e for f in rim for e in f.edges} - inner_e
        created = _created(bm, new_v, new_e, inner)
        created["rim_faces"] = _compact(f.index for f in rim)
        created["rim_faces_count"] = len(rim)
        counts = _counts(bm)
    return {"object": obj.name, "created": created, "counts": counts, "after": _summary(ed.report)}


# --------------------------------------------------------------------------- transform

def _proportional_weights(bm, sel_verts, prop, to_space):
    """{vertex index: weight} for the *unselected* vertices within the radius."""
    radius, kind = prop["radius"], prop["falloff"]
    chosen = {v.index for v in sel_verts}
    weights = {}

    def add(i, d):
        w = falloff_weight(kind, d, radius)
        if w > 0.0:
            weights[i] = w
            if len(weights) + len(chosen) > MAX_PROPORTIONAL_VERTS:
                raise OpError("budget", "proportional editing would affect more than "
                                        f"{MAX_PROPORTIONAL_VERTS:,} vertices; use a smaller "
                                        "radius")
    if prop["connected_only"]:
        dist = {i: 0.0 for i in chosen}
        heap = [(0.0, i) for i in chosen]
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist.get(i, math.inf):
                continue
            v = bm.verts[i]
            for e in v.link_edges:
                o = e.other_vert(v)
                nd = d + (to_space @ v.co - to_space @ o.co).length
                if nd <= radius and nd < dist.get(o.index, math.inf):
                    dist[o.index] = nd
                    heapq.heappush(heap, (nd, o.index))
        for i, d in dist.items():
            if i not in chosen:
                add(i, d)
    else:
        kd = KDTree(len(chosen))
        for n, i in enumerate(sorted(chosen)):
            kd.insert(to_space @ bm.verts[i].co, n)
        kd.balance()
        for v in bm.verts:
            if v.index in chosen:
                continue
            _, _, d = kd.find(to_space @ v.co)
            if d <= radius:
                add(v.index, d)
    return weights


@op("transform_elements", Obj({
    "object": NAME,
    "select": SELECT,
    "translate": Opt(LOC),
    "rotate": Opt(Obj({"axis": DIR3, "angle_deg": Float(-36000, 36000)})),
    "scale": Opt(SCALE),
    "pivot": Opt(Union(Enum(("median", "bounds_center", "cursor")), Obj({"point": LOC})),
                 "median"),
    "space": Opt(Enum(("world", "local")), "world"),
    "proportional": Opt(Obj({
        "radius": Float(1e-6, 1e6),
        "falloff": Opt(Enum(FALLOFFS), "smooth"),
        "connected_only": Opt(Bool(), False),
    })),
}), mutates=True,
    description="Move, rotate and/or scale selected mesh elements (the selector is resolved to "
                "vertices) without changing topology. Applied in the order scale, rotate, "
                "translate, all about `pivot` (median / bounds_center of the selection, the "
                "3D cursor, or {point: [x,y,z]}), in `space` axes (default world). "
                "`proportional` also moves unselected vertices within `radius` (in `space` "
                "units) of the selection, with weight falloff(distance/radius) using Blender's "
                "curves (smooth = 3t^2-2t^3, sphere, root, linear, sharp, constant, "
                "inverse_square, where t = 1 - distance/radius); the weight scales the "
                "translation, the rotation angle, and scale as lerp(1, s, w). With "
                "connected_only the distance is measured along edges instead of straight. "
                "Returns moved (selected) and affected (also weighted) vertex counts.")
def transform_elements(object, select, translate, rotate, scale, pivot, space, proportional):
    if translate is None and rotate is None and scale is None:
        raise OpError("invalid_params", "give at least one of translate, rotate, scale")
    axis = None
    if rotate is not None:
        axis = Vector(rotate["axis"])
        if axis.length < 1e-9:
            raise OpError("invalid_params", "rotate.axis must not be zero")
        axis.normalize()
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, "verts")
        local = space == "local"
        mw = Matrix.Identity(4) if local else _mw(obj)
        try:
            inv = mw.inverted()
        except ValueError:
            raise OpError("invalid_params", f"{obj.name} has a zero scale") from None
        pts = [mw @ v.co for v in sel]
        if pivot == "median":
            c = sum(pts, Vector()) / len(pts)
        elif pivot == "bounds_center":
            c = Vector([(min(p[k] for p in pts) + max(p[k] for p in pts)) / 2 for k in range(3)])
        elif pivot == "cursor":
            c = Vector(bpy.context.scene.cursor.location)
            if local:
                c = _mw(obj).inverted_safe() @ c
        else:
            c = Vector(pivot["point"])
        weights = {}
        if proportional is not None:
            weights = _proportional_weights(bm, sel, proportional, mw)
        t = Vector(translate) if translate is not None else Vector()
        moved = [(v, p, 1.0) for v, p in zip(sel, pts)]
        moved += [(bm.verts[i], mw @ bm.verts[i].co, w) for i, w in weights.items()]
        for v, p, w in moved:
            q = p - c
            if scale is not None:
                q = Vector((q[0] * (1 + (scale[0] - 1) * w), q[1] * (1 + (scale[1] - 1) * w),
                            q[2] * (1 + (scale[2] - 1) * w)))
            if rotate is not None:
                q = Matrix.Rotation(math.radians(rotate["angle_deg"]) * w, 3, axis) @ q
            v.co = inv @ (c + q + t * w)
        counts = _counts(bm)
    return {"object": obj.name, "moved": len(sel), "affected": len(moved), "counts": counts,
            "after": _summary(ed.report)}


# --------------------------------------------------------------------------- delete / dissolve

DELETE_CONTEXT = {"verts": ("VERTS", "verts"), "edges": ("EDGES", "edges"),
                  "faces": ("FACES", "faces"), "only_faces": ("FACES_ONLY", "faces"),
                  "edges_faces": ("EDGES_FACES", "edges")}


@op("delete_elements", Obj({
    "object": NAME,
    "select": SELECT,
    "what": Enum(tuple(DELETE_CONTEXT)),
}), mutates=True, destructive=True,
    description="Delete selected mesh elements, like Blender's X menu. what: 'verts' (vertices "
                "and everything using them), 'edges' (edges and the faces using them), "
                "'faces' (faces plus edges/vertices used by nothing else), 'only_faces' (faces "
                "only; edges and vertices stay, leaving a hole) or 'edges_faces' (edges and "
                "faces, keeping vertices). The selector is resolved in the domain `what` "
                "deletes. Indices change afterwards; use the returned counts.")
def delete_elements(object, select, what):
    ctx, domain = DELETE_CONTEXT[what]
    obj, ed = _open_edit(object)
    with ed as bm:
        sel = _resolve(bm, obj, select, domain)
        before = _counts(bm)
        bmesh.ops.delete(bm, geom=list(sel), context=ctx)
        counts = _counts(bm)
    return {"object": obj.name, "deleted": len(sel), "removed": [b - a for b, a in zip(before, counts)],
            "counts": counts, "after": _summary(ed.report)}


@op("dissolve", Obj({
    "object": NAME,
    "select": SELECT,
    "what": Enum(("verts", "edges", "faces", "limited")),
    "angle_deg": Opt(Float(0, 180)),
    "use_boundary_tear": Opt(Bool()),
    "use_face_split": Opt(Bool()),
    "use_verts": Opt(Bool()),
    "dissolve_boundaries": Opt(Bool()),
}), mutates=True, destructive=True,
    description="Dissolve selected elements, merging the surrounding geometry instead of "
                "leaving holes. what: 'verts' (remove vertices, merging their faces; "
                "use_boundary_tear, use_face_split), 'edges' (merge the faces on both sides; "
                "use_verts removes vertices left with two edges, default true; "
                "use_face_split), 'faces' (merge selected faces into one; use_verts as for "
                "edges, default true) or 'limited' (limited dissolve of the selected "
                "vertices: merge nearly flat faces/collinear edges below angle_deg, default "
                "5; dissolve_boundaries also dissolves along the open boundary). Options that "
                "do not apply to `what` are refused. Returns created.faces = the faces that are new or whose "
                "vertices changed (the merged face, and with use_verts also its neighbours).")
def dissolve(object, select, what, angle_deg, use_boundary_tear, use_face_split, use_verts,
             dissolve_boundaries):
    allowed = {"verts": {"use_boundary_tear", "use_face_split"},
               "edges": {"use_verts", "use_face_split"},
               "faces": {"use_verts"},
               "limited": {"angle_deg", "dissolve_boundaries"}}[what]
    given = {"angle_deg": angle_deg, "use_boundary_tear": use_boundary_tear,
             "use_face_split": use_face_split, "use_verts": use_verts,
             "dissolve_boundaries": dissolve_boundaries}
    extra = [k for k, v in given.items() if v is not None and k not in allowed]
    if extra:
        raise OpError("invalid_params", f"{extra} do not apply to what={what!r}")
    obj, ed = _open_edit(object)
    domain = {"verts": "verts", "edges": "edges", "faces": "faces", "limited": "verts"}[what]
    with ed as bm:
        sel = _resolve(bm, obj, select, domain)
        before = _counts(bm)
        # Dissolve creates no vertices, so a face is new when its vertex set (by the indices
        # _resolve just assigned, which deletion leaves alone) did not exist before.
        old_faces = {frozenset(v.index for v in f.verts) for f in bm.faces}
        if what == "verts":
            bmesh.ops.dissolve_verts(bm, verts=list(sel), use_face_split=bool(use_face_split),
                                     use_boundary_tear=bool(use_boundary_tear))
        elif what == "edges":
            bmesh.ops.dissolve_edges(bm, edges=list(sel),
                                     use_verts=True if use_verts is None else use_verts,
                                     use_face_split=bool(use_face_split))
        elif what == "faces":
            bmesh.ops.dissolve_faces(bm, faces=list(sel),
                                     use_verts=True if use_verts is None else use_verts)
        else:
            vs = set(sel)
            es = [e for v in sel for e in v.link_edges if e.verts[0] in vs and e.verts[1] in vs]
            bmesh.ops.dissolve_limit(
                bm, angle_limit=math.radians(5.0 if angle_deg is None else angle_deg),
                use_dissolve_boundaries=bool(dissolve_boundaries), verts=list(sel),
                edges=list(set(es)))
        merged = [f for f in bm.faces if frozenset(v.index for v in f.verts) not in old_faces]
        created = _created(bm, faces=merged)
        counts = _counts(bm)
    return {"object": obj.name, "created": created, "removed": [b - a for b, a in zip(before, counts)],
            "counts": counts, "after": _summary(ed.report)}


# --------------------------------------------------------------------------- flags, normals

CREASE_LAYER = "crease_edge"            # Blender 4.x/5.x store these as float edge attributes
BEVEL_WEIGHT_LAYER = "bevel_weight_edge"


@op("set_edge_flags", Obj({
    "object": NAME,
    "select": SELECT,
    "seam": Opt(Bool()),
    "sharp": Opt(Bool()),
    "crease": Opt(Float(0, 1)),
    "bevel_weight": Opt(Float(0, 1)),
}), mutates=True,
    description="Set UV seam, sharp edge, subdivision crease (0..1) and/or bevel weight (0..1) "
                "on the selected edges (the selector is resolved in edges; a face or vertex "
                "selection gives the edges it contains). Give at least one flag. Crease and "
                "bevel weight are stored in the float edge attributes 'crease_edge' and "
                "'bevel_weight_edge'; setting 0 leaves the attribute in place with value 0.")
def set_edge_flags(object, select, seam, sharp, crease, bevel_weight):
    if seam is None and sharp is None and crease is None and bevel_weight is None:
        raise OpError("invalid_params", "give at least one of seam, sharp, crease, bevel_weight")
    obj, ed = _open_edit(object)
    with ed as bm:
        # Adding a custom-data layer reallocates the elements, so do it before resolving.
        layers = {}
        for key, name, val in (("crease", CREASE_LAYER, crease),
                               ("bevel_weight", BEVEL_WEIGHT_LAYER, bevel_weight)):
            if val is not None:
                layers[key] = (bm.edges.layers.float.get(name) or bm.edges.layers.float.new(name),
                               val)
        sel = _resolve(bm, obj, select, "edges")
        for e in sel:
            if seam is not None:
                e.seam = seam
            if sharp is not None:
                e.smooth = not sharp
            for layer, val in layers.values():
                e[layer] = val
        counts = _counts(bm)
    return {"object": obj.name, "edges_changed": len(sel), "counts": counts,
            "after": _summary(ed.report)}


@op("normals", Obj({
    "object": NAME,
    "select": Opt(SELECT),
    "action": Enum(("recalculate_outside", "recalculate_inside", "flip")),
}), mutates=True,
    description="Fix face orientation for the selected faces (default: all). "
                "'recalculate_outside' / 'recalculate_inside' make the winding consistent and "
                "point normals out of / into the (selected) surface, 'flip' reverses each "
                "selected face. repair_mesh(recalculate_normals) does the whole-mesh outward "
                "case as one step of a cleanup; use this for a subset, inside normals or a "
                "plain flip. Returns flipped = number of faces whose normal changed side.")
def normals(object, select, action):
    obj, ed = _open_edit(object)
    with ed as bm:
        faces = _resolve(bm, obj, select or {"all": True}, "faces")
        bm.normal_update()
        before = [f.normal.copy() for f in faces]
        if action == "flip":
            bmesh.ops.reverse_faces(bm, faces=list(faces))
        else:
            bmesh.ops.recalc_face_normals(bm, faces=list(faces))
            if action == "recalculate_inside":
                bmesh.ops.reverse_faces(bm, faces=list(faces))
        bm.normal_update()
        flipped = sum(1 for f, b in zip(faces, before) if f.normal.dot(b) < 0)
        counts = _counts(bm)
    return {"object": obj.name, "faces": len(faces), "flipped": flipped, "counts": counts,
            "after": _summary(ed.report)}

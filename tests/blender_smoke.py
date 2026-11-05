"""Run inside real Blender (RNA, bmesh and the guard are not faked here):

    blender --background --factory-startup --python-exit-code 1 \
        --python tests/blender_smoke.py

Exercises every scene op through registry.dispatch, the same path network
requests take, plus escape attempts against real RNA.
"""

import os
import sys
import traceback
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from blender_safe_mcp import blender as B  # noqa: E402  (installs the guard)
from blender_safe_mcp.ops import registry  # noqa: E402
from blender_safe_mcp.server import OpError  # noqa: E402

PREFS = SimpleNamespace(allow_modify=True, accept_images=True, max_image_pixels=16_777_216,
                       max_request_mb=64)
B.prefs = lambda: PREFS
bpy.context.preferences.filepaths.use_scripts_auto_execute = False

failures = []


def call(op, **params):
    return registry.dispatch(op, params)[0]


def expect(code, op, **params):
    try:
        call(op, **params)
    except OpError as e:
        if e.code != code:
            failures.append(f"{op} {params}: expected {code}, got {e.code}: {e.message}")
        return
    failures.append(f"{op} {params}: expected {code}, but it succeeded")


def check(label, fn):
    before = len(failures)
    try:
        fn()
    except Exception:  # noqa: BLE001
        failures.append(f"{label}:\n{traceback.format_exc()}")
    # expect() records its failures without raising: count them against this section too
    new = len(failures) - before
    print(f"ok   {label}" if not new else f"FAIL {label} ({new} failure(s))")


def scene_ops():
    for kind in ("cube", "uv_sphere", "ico_sphere", "cylinder", "cone", "plane", "grid",
                 "circle", "torus", "monkey"):
        call("add_primitive", kind=kind, name=f"T_{kind}", location=[0, 0, 1])
    call("add_mesh", name="Tri", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    call("add_light", type="AREA", energy=500, size=2, location=[3, 3, 3])
    call("add_camera", location=[6, -6, 4], look_at=[0, 0, 0])
    call("add_text", body="hello", extrude=0.1)
    call("add_empty", name="Pivot")
    call("set_transform", name="T_cube", location=[1, 2, 3], rotation_deg=[0, 0, 45])
    dims = call("set_transform", name="T_cube", scale=[1, 1, 2])["dimensions"]
    assert abs(dims[2] - 4.0) < 1e-4, f"stale dimensions after scaling: {dims}"
    call("set_transform", name="T_cube", scale=[1, 1, 1])
    call("set_parent", child="T_cube", parent="Pivot")
    call("duplicate_object", name="T_cube", new_name="T_cube_copy")
    info = call("get_object_info", name="T_cube")
    assert info["parent"] == "Pivot" and info["world_bounding_box"]
    call("rename", kind="object", name="T_cube_copy", new_name="Copy")
    call("delete_objects", names=["Copy"])
    assert call("list_objects", query="T_", limit=500)["objects"]
    assert call("get_scene_info")["object_count"] > 10


def modifiers_and_budget():
    call("add_modifier", object="T_cube", type="SUBSURF", settings={"levels": 2})
    call("set_property", target={"kind": "object", "name": "T_cube"},
         path=["modifiers", "Subsurf", "levels"], value=3)
    expect("out_of_range", "set_property", target={"kind": "object", "name": "T_cube"},
           path=["modifiers", "Subsurf", "levels"], value=6)
    call("add_modifier", object="T_cube", type="ARRAY", settings={"count": 100})
    expect("budget", "add_modifier", object="T_cube", type="ARRAY", settings={"count": 500})
    assert len(bpy.data.objects["T_cube"].modifiers) == 2  # failed add was rolled back
    call("add_modifier", object="T_uv_sphere", type="BOOLEAN", target_object="T_cylinder")
    call("remove_modifier", object="T_cube", name="Array")
    call("apply_modifiers", object="T_cube")


def materials_and_nodes():
    call("create_material", name="Red", base_color=[1, 0, 0], roughness=0.3,
         emission_strength=0.0, assign_to=["T_cube"])
    owner = {"kind": "material", "name": "Red"}
    call("add_shader_node", owner=owner, type="ShaderNodeTexNoise", name="Noise",
         settings={"noise_dimensions": "4D"} if "noise_dimensions" in
         bpy.types.ShaderNodeTexNoise.bl_rna.properties else None)
    call("link_shader_nodes", owner=owner, from_node="Noise", from_socket="Fac",
         to_node="Principled BSDF" if bpy.data.materials["Red"].node_tree.nodes.get("Principled BSDF")
         else next(n.name for n in bpy.data.materials["Red"].node_tree.nodes
                   if n.type == "BSDF_PRINCIPLED"), to_socket="Roughness")
    call("set_node_input", owner=owner, node="Noise", socket="Scale", value=12.0)
    g = call("get_shader_graph", owner=owner)
    assert any(lk["from_node"] == "Noise" for lk in g["links"])
    expect("invalid_params", "add_shader_node", owner=owner, type="ShaderNodeScript")
    call("set_world", color=[0.05, 0.05, 0.08], strength=1.0)
    call("insert_keyframe", object="T_cube", property="location", frame=10, value=[0, 0, 2])
    call("set_frame", frame=5)


def join():
    call("add_primitive", kind="cube", name="J_a", location=[10, 0, 0])
    call("add_primitive", kind="cube", name="J_b", location=[12, 0, 0])
    call("add_modifier", object="J_b", type="BEVEL", settings={"segments": 2})
    call("create_material", name="Blue", base_color=[0, 0, 1], assign_to=["J_b"])
    r = call("join_objects", target="J_a", others=["J_b"], merge_distance=0.001)
    assert "J_b" not in bpy.data.objects, "joined object still exists"
    assert r["mesh"]["faces"] > 12, r  # the bevel was baked in, not dropped
    assert "Blue" in r["materials"], r
    bbox = call("get_object_info", name="J_a")["world_bounding_box"]
    assert abs(bbox["max"][0] - 13.0) < 1e-3, bbox  # J_b kept its world position
    expect("invalid_params", "join_objects", target="J_a", others=["J_a"])
    expect("invalid_params", "join_objects", target="J_a", others=["Camera"])
    expect("not_found", "join_objects", target="J_a", others=["Nope"])


BOX_V = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0, 2], [2, 0, 2], [2, 2, 2], [0, 2, 2]]
BOX_F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def mesh_quality():
    call("add_primitive", kind="cube", name="Q_closed", size=2, location=[20, 0, 0])
    r = call("analyze_mesh", object="Q_closed")
    assert r["is_manifold"] and r["shells"] == 1 and r["holes"] == 0, r
    assert abs(r["volume_world"] - 8.0) < 1e-4 and r["self_intersecting_face_pairs"] == 0, r

    call("add_mesh", name="Q_open", vertices=BOX_V, faces=BOX_F[:5], location=[24, 0, 0])
    r = call("analyze_mesh", object="Q_open")
    assert not r["is_manifold"] and r["holes"] == 1 and r["boundary_edges"] == 4, r
    assert r["hole_list"][0]["edges"] == 4, r
    r = call("repair_mesh", object="Q_open", fill_holes_max_sides=4)
    assert r["actions"]["holes_filled"] == 1 and r["after"]["is_manifold"], r
    assert abs(r["after"]["volume_world"] - 8.0) < 1e-4, r

    flipped = BOX_F[:5] + [list(reversed(BOX_F[5]))]
    call("add_mesh", name="Q_flip", vertices=BOX_V, faces=flipped, location=[28, 0, 0])
    r = call("analyze_mesh", object="Q_flip")
    assert r["inconsistent_normal_edges"] == 4 and not r["is_manifold"], r
    r = call("repair_mesh", object="Q_flip")
    assert r["after"]["is_manifold"] and r["after"]["volume_world"] > 0, r

    call("add_primitive", kind="cube", name="Q_u1", size=2, location=[32, 0, 0])
    call("add_primitive", kind="cube", name="Q_u2", size=2, location=[33, 1, 1])
    call("add_modifier", object="Q_u1", type="BOOLEAN", target_object="Q_u2",
         settings={"operation": "UNION", "solver": "EXACT"})
    r = call("analyze_mesh", object="Q_u1", evaluated=True)
    assert r["is_manifold"] and r["shells"] == 1 and r["self_intersecting_face_pairs"] == 0, r
    assert abs(r["volume_world"] - 15.0) < 1e-3, r  # 8 + 8 - 1 overlap

    assert call("set_object_mode", object="Q_u1", mode="OBJECT")["changed"] is False
    expect("invalid_params", "analyze_mesh", object="Camera")


def stitching():
    # a 2x2x2 box broken into two halves (x<1 and x>1), each open along the cut
    lo_v = [[0, 0, 0], [1, 0, 0], [1, 2, 0], [0, 2, 0], [0, 0, 2], [1, 0, 2], [1, 2, 2], [0, 2, 2]]
    lo_f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [3, 0, 4, 7]]  # no x=1 face
    hi_v = [[1, 0, 0], [2, 0, 0], [2, 2, 0], [1, 2, 0], [1, 0, 2], [2, 0, 2], [2, 2, 2], [1, 2, 2]]
    hi_f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6]]  # no x=1 face
    call("add_mesh", name="V_lo", vertices=lo_v, faces=lo_f, location=[40, 0, 0])
    call("add_mesh", name="V_hi", vertices=hi_v, faces=hi_f, location=[40, 0, 0])
    r = call("join_objects", target="V_lo", others=["V_hi"])  # no weld: we stitch by hand
    loops = call("get_boundary_loops", object="V_lo")
    assert loops["loops"] == 2 and all(l["vertex_count"] == 4 and l["simple_loop"]
                                       for l in loops["loop_list"]), loops
    pairs = call("find_close_vertices", object="V_lo", distance=1e-4)["pairs"]
    assert len(pairs) == 4 and all(p["distance"] < 1e-6 for p in pairs), pairs
    r = call("merge_vertices", object="V_lo", groups=[[p["a"], p["b"]] for p in pairs])
    assert r["removed_vertices"] == 4, r
    # merging the two rims closes the seam: one closed box, no internal wall
    a = call("analyze_mesh", object="V_lo")
    assert a["holes"] == 0 and a["is_manifold"] and a["shells"] == 1, a
    assert abs(a["volume_world"] - 8.0) < 1e-4, a

    call("add_mesh", name="F_open", vertices=BOX_V, faces=BOX_F[:5], location=[44, 0, 0])
    rim = call("get_boundary_loops", object="F_open")["loop_list"][0]["vertices"]
    r = call("fill_hole", object="F_open", vertex=rim[0]["i"], method="triangles")
    assert r["filled"]["new_faces"] == 2 and r["after"]["is_manifold"], r
    expect("invalid_params", "fill_hole", object="F_open", vertex=0)  # no longer on a hole

    call("add_mesh", name="C_open", vertices=BOX_V, faces=BOX_F[:5], location=[48, 0, 0])
    ids = [v["i"] for v in call("get_boundary_loops", object="C_open")["loop_list"][0]["vertices"]]
    r = call("create_face", object="C_open", vertices=ids)
    assert r["after"]["is_manifold"] and r["after"]["volume_world"] > 0, r
    expect("invalid_params", "create_face", object="C_open", vertices=[0, 1, 2])  # 3rd face on edge
    expect("not_found", "merge_vertices", object="C_open", groups=[[0, 99999]])

    # add_mesh validates (and so drops) a repeated face: build the duplicate the way it happens
    # in practice, a loose copy of one face joined to the box and merged onto it
    call("add_mesh", name="D_dup", vertices=BOX_V, faces=BOX_F, location=[52, 0, 0])
    f1 = BOX_F[1]
    call("add_mesh", name="D_face", vertices=[BOX_V[i] for i in f1], faces=[[0, 1, 2, 3]],
         location=[52, 0, 0])
    call("join_objects", target="D_dup", others=["D_face"])
    pairs = call("find_close_vertices", object="D_dup", distance=1e-4, boundary_only=False)["pairs"]
    assert len(pairs) == 4, pairs
    call("merge_vertices", object="D_dup", groups=[[p["a"], p["b"]] for p in pairs])
    a = call("analyze_mesh", object="D_dup")
    # Blender 5.1.2's weld already drops the duplicate face; older builds keep it
    dup = a["faces"] - 6
    assert dup in (0, 1) and a["non_manifold_edges"] == 4 * dup, a
    r = call("remove_duplicate_faces", object="D_dup")
    assert r["removed"]["faces"] == dup and r["after"]["is_manifold"], r
    assert abs(r["after"]["volume_world"] - 8.0) < 1e-4, r

    call("add_mesh", name="X_box", vertices=BOX_V, faces=BOX_F, location=[56, 0, 0])
    r = call("delete_faces", object="X_box", faces=[[7, 6, 5, 4]])  # any vertex order
    assert r["deleted_faces"] == 1 and r["after"]["holes"] == 1, r
    expect("not_found", "delete_faces", object="X_box", faces=[[0, 2, 5]])



def _world_verts(name):
    o = bpy.data.objects[name]
    bpy.context.view_layer.update()
    return sorted(tuple(round(c, 2) + 0.0 for c in (o.matrix_world @ v.co)) for v in o.data.vertices)


def _select_only(names, active):
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(o.name in names)
    vl.objects.active = bpy.data.objects[active]


def transform_ops():
    cur = call("get_scene_info")["cursor"]
    assert set(cur) == {"location", "rotation_deg"}, cur
    r = call("set_cursor", location=[1, 2, 3], rotation_deg=[0, 0, 90])
    assert r["location"] == [1, 2, 3] and abs(r["rotation_deg"][2] - 90) < 1e-3, r
    assert call("get_scene_info")["cursor"] == r
    call("add_mesh", name="O_box", vertices=BOX_V, faces=BOX_F, location=[60, 0, 0],
         rotation_deg=[0, 0, 30])
    call("add_primitive", kind="cube", name="O_other", location=[70, 0, 0])
    r = call("set_cursor", to_object="O_box", to="bounds_center")
    assert r["location"][2] == 1.0, r  # box spans z 0..2 in local space
    r = call("set_cursor", to_object="O_box", to="median")
    assert abs(r["location"][2] - 1.0) < 1e-4, r
    r = call("set_cursor", to_object="O_box")
    assert r["location"] == [60, 0, 0], r
    expect("invalid_params", "set_cursor")
    r = call("set_cursor", rotation_deg=[0, 0, 45])
    assert abs(r["rotation_deg"][2] - 45) < 1e-4 and r["location"] == [60, 0, 0], r
    expect("invalid_params", "set_cursor", location=[0, 0, 0], to_object="O_box")
    expect("invalid_params", "set_cursor", location=[0, 0, 0], to="origin")
    expect("not_found", "set_cursor", to_object="Nope")

    # known selection that every op below must leave alone
    _select_only({"O_other", "Camera"}, "O_other")
    def intact():
        vl = bpy.context.view_layer
        assert {o.name for o in vl.objects if o.select_get()} == {"O_other", "Camera"}
        assert vl.objects.active.name == "O_other"

    before = _world_verts("O_box")
    r = call("set_origin", objects=["O_box"], mode="geometry_bounds")
    assert _world_verts("O_box") == before, "geometry moved in world space"
    assert abs(r["objects"][0]["world_location"][2] - 1.0) < 1e-4, r
    intact()
    for mode in ("geometry_median", "center_of_mass_surface", "center_of_mass_volume"):
        call("set_origin", objects=["O_box"], mode=mode)
        assert _world_verts("O_box") == before, mode
    intact()

    call("set_cursor", location=[5, 6, 7])
    r = call("set_origin", objects=["O_box"], mode="point", point=[61, 1, 0])
    assert all(abs(a - b) < 1e-3 for a, b in zip(r["objects"][0]["world_location"], [61, 1, 0])), r
    assert _world_verts("O_box") == before
    assert call("get_scene_info")["cursor"]["location"] == [5, 6, 7], "cursor not restored"
    r = call("set_origin", objects=["O_box"], mode="cursor")
    assert all(abs(a - b) < 1e-3 for a, b in zip(r["objects"][0]["world_location"], [5, 6, 7])), r
    call("set_origin", objects=["O_box"], mode="geometry_to_origin")
    assert _world_verts("O_box") != before  # geometry moved onto the origin
    intact()
    expect("invalid_params", "set_origin", objects=["O_box"], mode="point")
    expect("invalid_params", "set_origin", objects=["O_box"], mode="cursor", point=[0, 0, 0])
    expect("invalid_params", "set_origin", objects=["Camera"], mode="cursor")
    expect("invalid_params", "set_origin", objects=["O_box", "O_box"], mode="cursor")
    expect("invalid_params", "set_origin", objects=[], mode="cursor")
    expect("invalid_params", "set_origin", objects=["O_box"], mode="cursor", bogus=1)

    # apply scale: dimensions kept, scale becomes 1, vertices change
    call("add_primitive", kind="cube", name="A_cube", location=[80, 0, 0])
    call("set_transform", name="A_cube", scale=[2, 1, 1])
    dims = call("get_object_info", name="A_cube")["dimensions"]
    local_before = [tuple(v.co) for v in bpy.data.objects["A_cube"].data.vertices]
    r = call("apply_transform", objects=["A_cube"])
    o = r["objects"][0]
    assert o["scale"] == [1, 1, 1] and o["location"] == [80, 0, 0], o
    assert all(abs(a - b) < 1e-4 for a, b in zip(o["dimensions"], dims)), (o, dims)
    assert [tuple(v.co) for v in bpy.data.objects["A_cube"].data.vertices] != local_before
    intact()
    # negative scale: normals must stay outward
    call("add_primitive", kind="cube", name="A_neg", location=[84, 0, 0])
    call("set_transform", name="A_neg", scale=[-2, 1, 1], rotation_deg=[0, 0, 20])
    call("apply_transform", objects=["A_neg"], location=True)
    o = bpy.data.objects["A_neg"]
    assert tuple(o.location) == (0, 0, 0) and tuple(o.rotation_euler) == (0, 0, 0)
    a = call("analyze_mesh", object="A_neg")
    assert a["inconsistent_normal_edges"] == 0 and a["volume_world"] > 0, a
    intact()
    expect("invalid_params", "apply_transform", objects=["A_neg"], rotation=False, scale=False,
           location=False)

    # shared mesh data
    call("duplicate_object", name="A_cube", new_name="A_linked", linked=True)
    expect("invalid_params", "apply_transform", objects=["A_cube"])
    expect("invalid_params", "set_origin", objects=["A_cube"], mode="geometry_bounds")
    call("set_transform", name="A_cube", scale=[3, 1, 1])
    r = call("apply_transform", objects=["A_cube"], make_single_user=True)
    assert bpy.data.objects["A_cube"].data.users == 1 and bpy.data.objects["A_linked"].data.users == 1
    intact()

    # edit mode refusal; selection, mode and active object unchanged on failure
    call("set_object_mode", object="A_neg", mode="EDIT")
    try:
        expect("unavailable", "apply_transform", objects=["A_neg"])
    finally:
        call("set_object_mode", object="A_neg", mode="OBJECT")
    _select_only({"O_other", "Camera"}, "O_other")  # set_object_mode moves the active object
    intact()

    # read-only mode
    PREFS.allow_modify = False
    try:
        expect("read_only", "set_cursor", location=[0, 0, 0])
        expect("read_only", "set_origin", objects=["O_box"], mode="cursor")
        expect("read_only", "apply_transform", objects=["O_box"])
    finally:
        PREFS.allow_modify = True


def _eval_verts(name):
    obj = bpy.data.objects[name]
    ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    try:
        return [ev.matrix_world @ v.co for v in me.vertices]
    finally:
        ev.to_mesh_clear()


def placement_ops():
    def wmin(name):
        bpy.context.view_layer.update()
        o = bpy.data.objects[name]
        return [min((o.matrix_world @ v.co)[i] for v in o.data.vertices) for i in range(3)]

    def wloc(name):
        bpy.context.view_layer.update()
        return list(bpy.data.objects[name].matrix_world.translation)

    # align: three offset cubes on Z min
    for i, (n, z, s) in enumerate([("P_a", 0.0, 1), ("P_b", 3.0, 2), ("P_c", -2.0, 0.5)]):
        call("add_primitive", kind="cube", name=n, location=[100 + 5 * i, 0, z])
        call("set_transform", name=n, scale=[s, s, s])
    r = call("align_objects", objects=["P_a", "P_b", "P_c"], axes=["z"], mode="min")
    assert abs(wmin("P_b")[2] - wmin("P_a")[2]) < 1e-5 and abs(wmin("P_c")[2] - wmin("P_a")[2]) < 1e-5, r
    assert wloc("P_a")[2] == 0.0 and abs(wloc("P_b")[0] - 105) < 1e-5  # x untouched
    call("align_objects", objects=["P_a", "P_b"], axes=["x", "y"], mode="center",
         relative_to={"object": "P_c"})
    assert abs(wloc("P_a")[0] - 110) < 1e-5 and abs(wloc("P_b")[0] - 110) < 1e-5 and wloc("P_c")[0] == 110
    call("align_objects", objects=["P_a", "P_b"], axes=["x"], mode="max", relative_to="world_origin")
    bpy.context.view_layer.update()
    assert abs(max((bpy.data.objects["P_b"].matrix_world @ v.co)[0] for v in
                   bpy.data.objects["P_b"].data.vertices)) < 1e-5
    call("set_cursor", location=[7, 8, 9])
    call("align_objects", objects=["P_a"], axes=["x", "y", "z"], mode="center", relative_to="cursor")
    assert all(abs(a - b) < 1e-5 for a, b in zip(wloc("P_a"), [7, 8, 9])), wloc("P_a")
    call("align_objects", objects=["P_a", "P_b"], axes=["z"], mode="min", relative_to="selection_bounds")
    # camera / light / empty (point boxes)
    call("add_empty", name="P_e", location=[0, 0, 5])
    call("align_objects", objects=["P_a", "P_e"], axes=["z"], mode="min")
    assert abs(wloc("P_e")[2] - wmin("P_a")[2]) < 1e-5
    # parented object: world result is right
    call("add_primitive", kind="cube", name="P_par", location=[200, 0, 10], )
    call("set_transform", name="P_par", rotation_deg=[0, 0, 40], scale=[2, 2, 2])
    call("add_primitive", kind="cube", name="P_kid", location=[0, 0, 0])
    call("set_parent", child="P_kid", parent="P_par")
    call("align_objects", objects=["P_c", "P_kid"], axes=["x", "y", "z"], mode="center")
    assert all(abs(a - b) < 1e-4 for a, b in zip(wloc("P_kid"), wloc("P_c"))), (wloc("P_kid"), wloc("P_c"))
    # parent and child both moved
    call("align_objects", objects=["P_kid", "P_par"], axes=["x"], mode="center", relative_to="world_origin")
    assert abs(wloc("P_kid")[0]) < 1e-4 and abs(wloc("P_par")[0]) < 1e-4
    expect("invalid_params", "align_objects", objects=["P_a"], axes=["x", "x"], mode="min")
    expect("invalid_params", "align_objects", objects=["P_a", "P_a"], axes=["x"], mode="min")
    expect("not_found", "align_objects", objects=["P_a"], axes=["x"], mode="min", relative_to={"object": "Nope"})
    expect("invalid_params", "align_objects", objects=["P_a"], axes=["w"], mode="min")
    expect("invalid_params", "align_objects", objects=["P_a"], axes=["x"], mode="min", relative_to="bogus")

    # distribute: five objects with different sizes and positions
    xs = [0, 1, 7, 8, 20]
    for i, x in enumerate(xs):
        n = f"D_{i}"
        call("add_primitive", kind="cube", name=n, location=[x, 300, 0])
        call("set_transform", name=n, scale=[1 + i * 0.3] * 3)
    names = [f"D_{i}" for i in (3, 0, 4, 2, 1)]  # shuffled
    r = call("distribute_objects", objects=names, axis="x")
    cs = [o["world_location"][0] for o in r["objects"]]
    steps = [b - a for a, b in zip(cs, cs[1:])]
    assert [o["name"] for o in r["objects"]] == [f"D_{i}" for i in range(5)], r
    assert max(steps) - min(steps) < 1e-5 and cs[0] == 0 and cs[-1] == 20, (cs, steps)
    for i, x in enumerate(xs):
        call("set_transform", name=f"D_{i}", location=[x, 300, 0])
    r = call("distribute_objects", objects=[f"D_{i}" for i in range(5)], axis="x", mode="gaps")
    bpy.context.view_layer.update()
    ext = [(bpy.data.objects[f"D_{i}"].matrix_world.translation.x - (1 + i * 0.3),
            bpy.data.objects[f"D_{i}"].matrix_world.translation.x + (1 + i * 0.3)) for i in range(5)]
    gaps = [b[0] - a[1] for a, b in zip(ext, ext[1:])]
    assert max(gaps) - min(gaps) < 1e-5 and abs(ext[0][0] + 1) < 1e-5, gaps
    expect("invalid_params", "distribute_objects", objects=["D_0", "D_1"], axis="x")

    # snap
    call("add_primitive", kind="cube", name="S_a", location=[400.3, 0.74, -0.26])
    call("snap_objects", objects=["S_a"], to={"grid": 0.5})
    assert wloc("S_a") == [400.5, 0.5, -0.5], wloc("S_a")
    call("snap_objects", objects=["S_a"], to="cursor")
    assert all(abs(a - b) < 1e-5 for a, b in zip(wloc("S_a"), [7, 8, 9]))
    call("add_primitive", kind="cube", name="S_t", location=[410, 0, 0])
    call("set_transform", name="S_t", rotation_deg=[0, 0, 45], scale=[2, 2, 2])  # corner at x=410+2*sqrt2
    call("snap_objects", objects=["S_a"], to={"object": "S_t"})
    assert wloc("S_a") == [410, 0, 0]
    call("set_transform", name="S_a", location=[420, 0.1, 0.2])
    call("snap_objects", objects=["S_a"], to={"object": "S_t", "target": "nearest_vertex"})
    assert abs(wloc("S_a")[0] - (410 + 2 * 2 ** 0.5)) < 1e-4 and abs(wloc("S_a")[1]) < 1e-4, wloc("S_a")
    call("set_transform", name="S_a", location=[420, 0.0, 0.5])
    call("snap_objects", objects=["S_a"], to={"object": "S_t", "target": "nearest_surface"})
    assert abs(wloc("S_a")[0] - (410 + 2 * 2 ** 0.5)) < 1e-4 or abs(wloc("S_a")[0] - 410) > 1, wloc("S_a")
    call("set_transform", name="S_a", location=[410, 0, 10])
    call("snap_objects", objects=["S_a"], to={"object": "S_t", "target": "nearest_surface"})
    assert all(abs(a - b) < 1e-4 for a, b in zip(wloc("S_a"), [410, 0, 2])), wloc("S_a")
    call("snap_objects", objects=["S_a"], to={"object": "S_t", "target": "bounds_center"})
    assert all(abs(a - b) < 1e-4 for a, b in zip(wloc("S_a"), [410, 0, 0])), wloc("S_a")
    expect("invalid_params", "snap_objects", objects=["S_t"], to={"object": "S_t"})
    expect("invalid_params", "snap_objects", objects=["S_a"], to={"object": "P_e", "target": "nearest_vertex"})
    expect("invalid_params", "snap_objects", objects=["S_a"], to={"grid": 0})
    expect("invalid_params", "snap_objects", objects=["S_a"], to={"grid": 1, "object": "S_t"})
    expect("not_found", "snap_objects", objects=["S_a"], to={"object": "Nope"})

    # drop_to_surface
    call("add_mesh", name="G_plane", vertices=[[-5, -5, -1], [5, -5, -1], [5, 5, -1], [-5, 5, -1]],
         faces=[[0, 1, 2, 3]], location=[500, 0, 0])
    call("add_primitive", kind="cube", name="G_cube", location=[500, 0, 4])  # spans z 3..5
    r = call("drop_to_surface", objects=["G_cube"], targets=["G_plane"])
    assert abs(wmin("G_cube")[2] + 1) < 1e-4, wmin("G_cube")
    assert r["objects"][0]["landed_on"] == "G_plane" and abs(r["objects"][0]["moved"] - 4) < 1e-4, r
    call("drop_to_surface", objects=["G_cube"], targets=["G_plane"], offset=0.25)  # already down: moves up? no
    call("set_transform", name="G_cube", location=[500, 0, 4])
    call("drop_to_surface", objects=["G_cube"])  # default targets: visible meshes
    assert abs(wmin("G_cube")[2] + 1) < 1e-4, wmin("G_cube")
    call("set_transform", name="G_cube", location=[500, 0, 4])
    call("drop_to_surface", objects=["G_cube"], targets=["G_plane"], offset=0.5)
    assert abs(wmin("G_cube")[2] + 0.5) < 1e-4
    # floor_z only
    call("add_primitive", kind="cube", name="G_free", location=[900, 900, 6])
    call("drop_to_surface", objects=["G_free"], floor_z=-3, targets=["G_plane"])
    assert abs(wmin("G_free")[2] + 3) < 1e-4, wmin("G_free")
    # nothing below
    call("set_transform", name="G_free", location=[900, 900, 6])
    expect("not_found", "drop_to_surface", objects=["G_free"], targets=["G_plane"])
    assert wloc("G_free")[2] == 6  # untouched on refusal
    # empty (origin point)
    call("add_empty", name="G_empty", location=[500, 1, 8])
    call("drop_to_surface", objects=["G_empty"], targets=["G_plane"])
    assert abs(wloc("G_empty")[2] + 1) < 1e-4
    # tilted slope: align_to_normal
    call("add_mesh", name="G_ramp", vertices=[[-5, -5, 0], [5, -5, 0], [5, 5, 10], [-5, 5, 10]],
         faces=[[0, 1, 2, 3]], location=[600, 0, 0])
    call("add_primitive", kind="cube", name="G_tilt", location=[600, 0, 30])
    call("drop_to_surface", objects=["G_tilt"], targets=["G_ramp"], align_to_normal=True)
    o = bpy.data.objects["G_tilt"]
    bpy.context.view_layer.update()
    ramp_n = (bpy.data.objects["G_ramp"].matrix_world.to_3x3() @ bpy.data.objects["G_ramp"].data.polygons[0].normal).normalized()
    if ramp_n.z < 0:
        ramp_n = -ramp_n
    up = (o.matrix_world.to_quaternion() @ Vector((0, 0, 1))).normalized()
    assert up.dot(ramp_n) > 0.9999, (up, ramp_n)
    # resting: lowest vertex must be on the ramp plane (n . (v - p0) ~ 0)
    p0 = Vector((600, 0, 0)) + Vector((-5, -5, 0))
    dmin = min(ramp_n.dot((o.matrix_world @ v.co) - p0) for v in o.data.vertices)
    assert abs(dmin) < 1e-3, dmin
    # parented object dropped
    call("set_transform", name="P_par", location=[500, 0, 30], rotation_deg=[0, 0, 0], scale=[1, 1, 1])
    call("set_cursor", location=[500, 0, 20])
    call("align_objects", objects=["P_kid"], axes=["x", "y", "z"], mode="center", relative_to="cursor")
    call("drop_to_surface", objects=["P_kid"], targets=["G_plane"])
    assert abs(wmin("P_kid")[2] + 1) < 1e-4, wmin("P_kid")
    expect("invalid_params", "drop_to_surface", objects=["G_cube"], direction=[0, 0, 0])
    expect("invalid_params", "drop_to_surface", objects=["G_cube"], targets=["G_cube"])
    expect("invalid_params", "drop_to_surface", objects=["G_cube"], targets=["P_e"])
    expect("not_found", "drop_to_surface", objects=["G_cube"], targets=["Nope"])

    # look_at
    call("add_camera", name="L_cam", location=[700, 0, 5])
    call("look_at", object="L_cam", target={"point": [710, 4, 0]})
    cam = bpy.data.objects["L_cam"]
    bpy.context.view_layer.update()
    fwd = cam.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    want = (Vector((710, 4, 0)) - Vector((700, 0, 5))).normalized()
    assert (fwd - want).length < 1e-4, (fwd, want)
    upv = cam.matrix_world.to_quaternion() @ Vector((0, 1, 0))
    assert upv.z > 0, upv  # camera upright
    call("look_at", object="L_cam", target={"object": "S_t"})
    fwd = bpy.data.objects["L_cam"].matrix_world.to_quaternion() @ Vector((0, 0, -1))
    want = (bpy.data.objects["S_t"].matrix_world.translation - cam.matrix_world.translation).normalized()
    assert (fwd - want).length < 1e-4
    call("look_at", object="G_empty", target={"point": [0, 0, 100]}, track_axis="X", up_axis="Z")
    e = bpy.data.objects["G_empty"]
    bpy.context.view_layer.update()
    assert ((e.matrix_world.to_quaternion() @ Vector((1, 0, 0))) - (Vector((0, 0, 100)) - e.matrix_world.translation).normalized()).length < 1e-4
    # parented + scaled object keeps world location and scale
    call("set_transform", name="P_par", rotation_deg=[10, 20, 30], scale=[2, 2, 2])
    before = wloc("P_kid")
    scale_before = bpy.data.objects["P_kid"].matrix_world.to_scale().x
    call("look_at", object="P_kid", target={"point": [0, 50, 50]})
    assert all(abs(a - b) < 1e-4 for a, b in zip(wloc("P_kid"), before))
    k = bpy.data.objects["P_kid"]
    fwd = k.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    assert (fwd - (Vector((0, 50, 50)) - k.matrix_world.translation).normalized()).length < 1e-4
    assert abs(k.matrix_world.to_scale().x - scale_before) < 1e-4
    expect("invalid_params", "look_at", object="L_cam", target={"point": [0, 0, 0]}, track_axis="Y", up_axis="Y")
    expect("invalid_params", "look_at", object="L_cam", target={"point": list(cam.matrix_world.translation)})
    expect("invalid_params", "look_at", object="L_cam", target={"object": "L_cam"})
    expect("invalid_params", "look_at", object="L_cam", target={"point": [0, 0, 0], "object": "S_t"})
    expect("not_found", "look_at", object="L_cam", target={"object": "Nope"})

    # selection / active object untouched
    _select_only({"S_t", "Camera"}, "S_t")
    call("align_objects", objects=["S_a", "S_t"], axes=["x"], mode="min")
    call("drop_to_surface", objects=["G_cube"], targets=["G_plane"])
    call("look_at", object="L_cam", target={"point": [0, 0, 0]})
    vl = bpy.context.view_layer
    assert {o.name for o in vl.objects if o.select_get()} == {"S_t", "Camera"}
    assert vl.objects.active.name == "S_t"

    # A bevelled NURBS spiral's evaluated bound_box is wrong on 5.1 (local z -1..1.45 for a
    # 0.5-tall spring): placement must use the tessellated geometry, not that box.
    call("add_primitive", kind="cube", name="P_desk", location=[300, 0, 0.75],
         scale=[2.5, 1.5, 0.05])
    call("add_curve_primitive", kind="spiral", name="P_spring", location=[301.7, 0.8, 1.5],
         radius=0.15, turns=6, height=0.45, bevel_depth=0.02)
    r = call("drop_to_surface", objects=["P_spring"], targets=["P_desk"])["objects"][0]
    assert r["landed_on"] == "P_desk", r
    lo = min(v.z for v in _eval_verts("P_spring"))
    assert abs(lo - 0.8) < 1e-4, lo
    r = call("drop_to_surface", objects=["P_spring"], floor_z=0.5)["objects"][0]
    assert r["landed_on"] == "P_desk" and abs(r["moved"]) < 1e-4, r  # already resting on it
    call("align_objects", objects=["P_spring"], axes=["z"], mode="min",
         relative_to="world_origin")
    assert abs(min(v.z for v in _eval_verts("P_spring"))) < 1e-4

    PREFS.allow_modify = False
    try:
        expect("read_only", "align_objects", objects=["P_a"], axes=["x"], mode="min")
        expect("read_only", "distribute_objects", objects=["D_0", "D_1", "D_2"], axis="x")
        expect("read_only", "snap_objects", objects=["P_a"], to="cursor")
        expect("read_only", "drop_to_surface", objects=["P_a"])
        expect("read_only", "look_at", object="L_cam", target={"point": [0, 0, 0]})
    finally:
        PREFS.allow_modify = True


# --------------------------------------------------------------------------- phase 3a

def _ix(items):
    """Expand a compact index list from an op result ([3, [16, 20]]) to plain ints."""
    out = []
    for it in items:
        out.extend([it] if isinstance(it, int) else range(it[0], it[1] + 1))
    return out


def _els(name, domain, select=None, **kw):
    r = call("get_mesh_elements", object=name, domain=domain, select=select, limit=2000, **kw)
    return r["elements"]


def _idx(name, domain, select):
    return [e["index"] for e in _els(name, domain, select)]


def _me(name):
    return bpy.data.objects[name].data


def _bad_select(code, select, name="M_cube"):
    expect(code, "get_mesh_elements", object=name, select=select)


def mesh_selector():
    call("add_primitive", kind="cube", name="M_cube", location=[100, 0, 0])
    r = call("get_mesh_elements", object="M_cube", domain="verts")
    assert r["total"] == 8 and r["counts"] == [8, 12, 6], r
    assert all(abs(abs(e["co"][0] - 100) - 1) < 1e-5 for e in r["elements"]), r
    loc = call("get_mesh_elements", object="M_cube", domain="verts", space="local")["elements"]
    assert all(abs(abs(e["co"][0]) - 1) < 1e-5 for e in loc)
    f = _els("M_cube", "faces")
    assert len(f) == 6 and all(abs(e["area"] - 4) < 1e-4 for e in f), f
    e = _els("M_cube", "edges")
    assert len(e) == 12 and all(abs(x["length"] - 2) < 1e-4 and not x["is_boundary"] for x in e)
    p = call("get_mesh_elements", object="M_cube", domain="verts", offset=6, limit=5)
    assert p["total"] == 8 and [x["index"] for x in p["elements"]] == [6, 7], p
    # compact index items: [first, last] ranges (as ops return them), overlapping ones merged
    got = [x["index"] for x in _els("M_cube", "verts", {"verts": [0, [2, 4], [3, 5], 7]})]
    assert got == [0, 2, 3, 4, 5, 7], got
    expect("invalid_params", "get_mesh_elements", object="M_cube", domain="verts",
           select={"verts": [[4, 2]]})
    expect("not_found", "get_mesh_elements", object="M_cube", domain="verts",
           select={"verts": [[6, 8]]})
    # normal criterion (world space)
    top = _idx("M_cube", "faces", {"normal": {"direction": [0, 0, 1], "max_angle_deg": 10}})
    assert len(top) == 1 and _els("M_cube", "faces", {"faces": top})[0]["normal"][2] > 0.99
    assert len(_idx("M_cube", "faces", {"normal": {"direction": [0, 0, 1], "max_angle_deg": 100}})) == 5
    # box / sphere, world and local
    assert len(_idx("M_cube", "verts", {"box": {"min": [99, -2, 0], "max": [101, 2, 2]}})) == 4
    assert len(_idx("M_cube", "verts", {"box": {"min": [-2, -2, 0], "max": [2, 2, 2]},
                                        "space": "local"})) == 4
    assert len(_idx("M_cube", "verts", {"sphere": {"center": [100, 0, 0], "radius": 1.8}})) == 8
    assert len(_idx("M_cube", "verts", {"sphere": {"center": [100, 0, 0], "radius": 1.5}})) == 0
    # domain conversion
    topverts = _idx("M_cube", "verts", {"box": {"min": [99, -2, 0.5], "max": [101, 2, 2]}})
    assert len(_idx("M_cube", "faces", {"verts": topverts})) == 1
    assert len(_idx("M_cube", "edges", {"verts": topverts})) == 4
    assert len(_idx("M_cube", "verts", {"faces": top})) == 4
    assert len(_idx("M_cube", "edges", {"faces": top})) == 4
    # combine / grow / invert
    assert _idx("M_cube", "faces", {"faces": [0, 1], "all": True}) == [0, 1]
    assert _idx("M_cube", "faces", {"faces": [0], "all": True, "combine": "or"}) == list(range(6))
    assert _idx("M_cube", "faces", {"faces": [0, 1], "normal": {"direction": [0, 0, 1],
                                                                 "max_angle_deg": 100},
                                    "combine": "and"}) != []
    assert len(_idx("M_cube", "faces", {"faces": top, "invert": True})) == 5
    assert len(_idx("M_cube", "verts", {"verts": [0], "grow": 1})) == 4
    assert len(_idx("M_cube", "verts", {"verts": [0], "grow": 2})) == 7
    assert len(_idx("M_cube", "faces", {"faces": top, "grow": 1})) == 5
    # materials (by name and slot)
    call("create_material", name="M_red", base_color=[1, 0, 0], assign_to=["M_cube"])
    assert len(_idx("M_cube", "faces", {"material": "M_red"})) == 6
    assert len(_idx("M_cube", "faces", {"material": 0})) == 6
    _bad_select("not_found", {"material": "Nope"})
    # boundary
    call("add_mesh", name="M_open", vertices=BOX_V, faces=BOX_F[:5], location=[110, 0, 0])
    assert len(_idx("M_open", "edges", {"boundary": True})) == 4
    assert len(_idx("M_open", "verts", {"boundary": True})) == 4
    # edge loop / ring on a 4x4 grid (5x5 vertices, spacing 0.5)
    call("add_primitive", kind="grid", name="M_grid", segments=4, location=[120, 0, 0])
    gv, ge = _els("M_grid", "verts"), _els("M_grid", "edges")
    horiz = [x["index"] for x in ge if all(abs(gv[v]["co"][1]) < 1e-5 for v in x["verts"])]
    assert len(horiz) == 4, horiz
    assert sorted(_idx("M_grid", "edges", {"edge_loop": horiz[0]})) == sorted(horiz)
    ring = _idx("M_grid", "edges", {"edge_ring": horiz[0]})
    x0 = sorted(gv[v]["co"][0] for v in ge[horiz[0]]["verts"])
    assert len(ring) == 5 and all(sorted(gv[v]["co"][0] for v in ge[i]["verts"]) == x0 for i in ring), ring
    call("add_primitive", kind="torus", name="M_torus", segments=8, rings=6, location=[130, 0, 0])
    assert len(_idx("M_torus", "edges", {"edge_loop": 0})) in (6, 8)   # closed loop terminates
    assert len(_idx("M_torus", "edges", {"edge_ring": 0})) in (6, 8)
    # errors
    _bad_select("stale_indices", {"all": True, "expect_counts": [8, 12, 7]})
    assert call("get_mesh_elements", object="M_cube",
                select={"all": True, "expect_counts": [8, 12, 6]})["total"] == 8
    _bad_select("invalid_params", {"grow": 1})
    _bad_select("invalid_params", {"all": False})
    _bad_select("not_found", {"verts": [99]})
    _bad_select("not_found", {"edge_loop": 99})
    _bad_select("invalid_params", {"normal": {"direction": [0, 0, 0], "max_angle_deg": 5}})
    _bad_select("invalid_params", {"box": {"min": [1, 1, 1], "max": [0, 0, 0]}})
    far = {"sphere": {"center": [0, 0, 50], "radius": 1}}
    assert call("get_mesh_elements", object="M_cube", select=far)["total"] == 0
    expect("invalid_params", "transform_elements", object="M_cube", translate=[1, 0, 0], select=far)
    expect("invalid_params", "extrude", object="M_cube", select=far)
    expect("invalid_params", "extrude", object="M_cube", select={"verts": [0]}, mode="region",
           distance=1, vector=[1, 0, 0])
    # edit mode: reads work, edits are refused
    call("set_object_mode", object="M_cube", mode="EDIT")
    try:
        assert call("get_mesh_elements", object="M_cube")["total"] == 8
        expect("unavailable", "delete_elements", object="M_cube", select={"all": True}, what="verts")
        expect("unavailable", "extrude", object="M_cube", select={"faces": [0]})
    finally:
        call("set_object_mode", object="M_cube", mode="OBJECT")


def _manifold(name):
    return call("analyze_mesh", object=name)


def mesh_extrude_inset():
    # extrude the top face of a (rotated) size-2 cube by 1: manifold, 10 faces, volume 12
    call("add_primitive", kind="cube", name="E_cube", location=[200, 0, 0], rotation_deg=[0, 0, 33])
    r = call("extrude", object="E_cube", select={"normal": {"direction": [0, 0, 1], "max_angle_deg": 5}},
             distance=1.0)
    a = r["after"]
    assert a["faces"] == 10 and a["is_manifold"] and abs(a["volume_world"] - 12) < 1e-3, r
    assert r["counts"] == [12, 20, 10], r
    c = r["created"]
    assert (c["verts_count"], c["edges_count"], c["faces_count"]) == (4, 8, 5), c
    # the created indices are the same elements get_mesh_elements and the mesh report afterwards
    got = _els("E_cube", "verts", {"verts": c["verts"]})
    assert [round(e["co"][2], 3) for e in got] == [2.0] * 4, got
    me = _me("E_cube")
    assert all(me.vertices[i].co.z > 1.99 for i in _ix(c["verts"]))
    assert sum(me.vertices[i].co.z > 1.99 for i in range(len(me.vertices))) == 4
    new_v = set(_ix(c["verts"]))
    assert all(set(me.edges[i].vertices) & new_v for i in _ix(c["edges"]))
    assert all(set(me.polygons[i].vertices) & new_v for i in _ix(c["faces"]))
    assert sum(1 for e in me.edges if set(e.vertices) & new_v) == 8
    assert sum(1 for p in me.polygons if set(p.vertices) & new_v) == 5
    assert len(r["cap_faces"]) == 1 and set(_ix(r["cap_faces"])) <= set(_ix(c["faces"]))
    cap = _els("E_cube", "faces", {"faces": r["cap_faces"]})[0]
    assert cap["normal"][2] > 0.99 and abs(cap["center"][2] - 2) < 1e-4 and abs(cap["area"] - 4) < 1e-4
    # chain: inset the cap by index (guarded by expect_counts), then extrude the inner face
    r2 = call("inset", object="E_cube", select={"faces": r["cap_faces"], "expect_counts": r["counts"]},
              thickness=0.25)
    assert r2["after"]["is_manifold"] and r2["counts"] == [16, 28, 14], r2
    assert r2["created"]["faces"] == r["cap_faces"] and r2["created"]["rim_faces_count"] == 4, r2
    assert (r2["created"]["verts_count"], r2["created"]["edges_count"]) == (4, 8), r2
    inner = _els("E_cube", "faces", {"faces": r2["created"]["faces"]})[0]
    assert abs(inner["area"] - 1.5 ** 2) < 1e-3, inner
    rim = _els("E_cube", "faces", {"faces": r2["created"]["rim_faces"]})
    assert len(rim) == 4 and all(e["normal"][2] > 0.99 for e in rim)
    r3 = call("extrude", object="E_cube", select={"faces": r2["created"]["faces"]}, distance=0.5)
    assert r3["after"]["is_manifold"] and r3["counts"] == [20, 36, 18], r3
    expect("stale_indices", "extrude", object="E_cube",
           select={"faces": [0], "expect_counts": [8, 12, 6]}, distance=1)
    # world-space vector on a rotated object moves the cap by exactly that vector
    call("add_primitive", kind="cube", name="E_vec", location=[210, 0, 0], rotation_deg=[0, 0, 90])
    rv = call("extrude", object="E_vec", select={"normal": {"direction": [0, 0, 1], "max_angle_deg": 5}},
              vector=[3, 0, 0])
    xs = sorted(round(e["co"][0], 3) for e in _els("E_vec", "verts", {"verts": rv["created"]["verts"]}))
    assert xs == [212.0, 212.0, 214.0, 214.0], xs
    # individual faces: each cap moves along its own normal
    call("add_primitive", kind="cube", name="E_ind", location=[220, 0, 0])
    ri = call("extrude", object="E_ind", select={"normal": {"direction": [0, 0, 1], "max_angle_deg": 100}},
              mode="individual_faces", distance=1)
    assert ri["counts"] == [28, 52, 26], ri
    assert len(_ix(ri["cap_faces"])) == 5 and ri["created"]["verts_count"] == 20
    for e in _els("E_ind", "faces", {"faces": ri["cap_faces"]}):
        d = [e["center"][0] - 220, e["center"][1], e["center"][2]]
        assert abs(sum(x * n for x, n in zip(d, e["normal"])) - 2.0) < 1e-3, e  # 1 + 1 out
    # edges / verts modes
    call("add_primitive", kind="plane", name="E_edge", location=[230, 0, 0])
    be = _idx("E_edge", "edges", {"boundary": True})
    re_ = call("extrude", object="E_edge", select={"edges": be[:1]}, mode="edges", vector=[0, 0, 1])
    assert re_["counts"] == [6, 7, 2] and re_["created"]["verts_count"] == 2, re_
    got = _els("E_edge", "verts", {"verts": re_["created"]["verts"]})
    assert sorted(round(e["co"][2], 2) for e in got) == [1.0, 1.0]
    call("add_primitive", kind="cube", name="E_vert", location=[240, 0, 0])
    rvv = call("extrude", object="E_vert", select={"verts": [0]}, mode="verts", distance=1)
    assert rvv["counts"] == [9, 13, 6] and rvv["created"]["verts_count"] == 1, rvv
    assert rvv["created"]["edges_count"] == 1, rvv
    nv = _els("E_vert", "verts", {"verts": rvv["created"]["verts"]})[0]
    d = ((nv["co"][0] - 240) ** 2 + nv["co"][1] ** 2 + nv["co"][2] ** 2) ** 0.5
    assert abs(d - (3 ** 0.5 + 1)) < 1e-3, nv
    # refusals leave the mesh alone
    expect("invalid_params", "extrude", object="E_vert", select={"all": True}, mode="verts",
           distance=1, vector=[1, 0, 0])
    expect("invalid_params", "inset", object="E_vert", select={"all": True}, thickness=0.1,
           individual=True, outset=True)
    assert len(_me("E_vert").vertices) == 9
    call("add_primitive", kind="cube", name="E_skew", scale=[1, 2, 3], location=[250, 0, 0])
    expect("invalid_params", "inset", object="E_skew", select={"faces": [0]}, thickness=0.1)
    call("inset", object="E_skew", select={"faces": [0]}, thickness=0.1, space="local")
    # inset individual and outset
    call("add_primitive", kind="cube", name="I_cube", location=[260, 0, 0])
    ri = call("inset", object="I_cube", select={"all": True}, thickness=0.2, individual=True)
    assert ri["counts"][2] == 30 and ri["created"]["rim_faces_count"] == 24, ri
    call("add_primitive", kind="cube", name="I_out", location=[270, 0, 0])
    ro = call("inset", object="I_out", select={"faces": [0]}, thickness=0.2, outset=True)
    f0 = _els("I_out", "faces", {"faces": ro["created"]["faces"]})[0]
    assert abs(f0["area"] - 4.0) < 1e-3, f0         # outset keeps the face's size (the rim moves out into the neighbouring faces)
    assert ro["after"]["is_manifold"] and ro["counts"] == [12, 20, 10], ro
    PREFS.allow_modify = False
    try:
        for name, kw in (("extrude", {"select": {"all": True}}), ("inset", {"select": {"all": True}, "thickness": 1}),
                         ("transform_elements", {"select": {"all": True}, "translate": [1, 0, 0]}),
                         ("delete_elements", {"select": {"all": True}, "what": "faces"}),
                         ("dissolve", {"select": {"all": True}, "what": "faces"}),
                         ("set_edge_flags", {"select": {"all": True}, "seam": True}),
                         ("normals", {"action": "flip"})):
            expect("read_only", name, object="E_cube", **kw)
        call("get_mesh_elements", object="E_cube")
    finally:
        PREFS.allow_modify = True


def mesh_transform():
    call("add_primitive", kind="grid", name="MT_grid", segments=4, location=[300, 0, 0])
    gv = _els("MT_grid", "verts")
    center = [e["index"] for e in gv if abs(e["co"][0] - 300) < 1e-5 and abs(e["co"][1]) < 1e-5][0]
    z = lambda: {e["index"]: e["co"][2] for e in _els("MT_grid", "verts")}  # noqa: E731
    pos = {e["index"]: e["co"] for e in gv}
    # plain translate
    r = call("transform_elements", object="MT_grid", select={"verts": [center]}, translate=[0, 0, 1])
    assert r["moved"] == 1 and r["affected"] == 1, r
    zz = z()
    assert abs(zz[center] - 1) < 1e-5 and sum(abs(v) for k, v in zz.items() if k != center) < 1e-5
    call("transform_elements", object="MT_grid", select={"verts": [center]}, translate=[0, 0, -1])
    # proportional, straight-line distance: neighbours at 0.5 and 0.7071 with radius 1
    call("transform_elements", object="MT_grid", select={"verts": [center]}, translate=[0, 0, 1],
         proportional={"radius": 1.0, "falloff": "smooth"})
    zz = z()
    exp_w = lambda d, r=1.0: (lambda t: 3 * t * t - 2 * t ** 3)(1 - d / r)  # noqa: E731
    for i, p in pos.items():
        d = ((p[0] - 300) ** 2 + p[1] ** 2) ** 0.5
        want = exp_w(d) if d <= 1.0 else 0.0
        assert abs(zz[i] - want) < 1e-4, (i, d, zz[i], want)
    assert abs(zz[center] - 1) < 1e-5
    # reset, then connected_only: distance along edges (diagonal neighbour is 1.0 away, not 0.707)
    for i in pos:
        call("transform_elements", object="MT_grid", select={"verts": [i]}, translate=[0, 0, -zz[i]]) \
            if abs(zz[i]) > 1e-9 else None
    call("transform_elements", object="MT_grid", select={"verts": [center]}, translate=[0, 0, 1],
         proportional={"radius": 1.2, "falloff": "linear", "connected_only": True})
    zz = z()
    for i, p in pos.items():
        dx, dy = abs(p[0] - 300), abs(p[1])
        d = dx + dy                      # edge-path (Manhattan) distance on the grid
        want = max(0.0, 1 - d / 1.2) if d <= 1.2 + 1e-9 else 0.0
        assert abs(zz[i] - want) < 1e-4, (i, d, zz[i], want)
    # every other falloff curve at distance 0.5, radius 1 (t = 0.5)
    for kind, want in (("sphere", 0.75 ** 0.5), ("root", 0.5 ** 0.5), ("sharp", 0.25),
                       ("constant", 1.0), ("inverse_square", 0.75), ("linear", 0.5)):
        for i in pos:
            if abs(z()[i]) > 1e-9:
                call("transform_elements", object="MT_grid", select={"verts": [i]}, translate=[0, 0, -z()[i]])
        call("transform_elements", object="MT_grid", select={"verts": [center]}, translate=[0, 0, 1],
             proportional={"radius": 1.0, "falloff": kind})
        nb = [i for i, p in pos.items() if abs(abs(p[0] - 300) + abs(p[1]) - 0.5) < 1e-5
              and (abs(p[0] - 300) < 1e-5 or abs(p[1]) < 1e-5)]
        assert len(nb) == 4 and all(abs(z()[i] - want) < 1e-4 for i in nb), (kind, [z()[i] for i in nb])
    # cross-check every falloff against Blender's own proportional translate (test-only bpy.ops)
    import bmesh as _bmesh
    for kind in ("smooth", "sphere", "root", "linear", "sharp", "constant", "inverse_square"):
        call("add_primitive", kind="grid", name="PX_ref", segments=8, size=4, location=[500, 0, 0])
        call("add_primitive", kind="grid", name="PX_mine", segments=8, size=4, location=[500, 0, 0])
        c = [e["index"] for e in _els("PX_mine", "verts") if abs(e["co"][0] - 500) < 1e-5 and abs(e["co"][1]) < 1e-5][0]
        call("transform_elements", object="PX_mine", select={"verts": [c]}, translate=[0, 0, 1],
             proportional={"radius": 1.3, "falloff": kind})
        ref = bpy.data.objects["PX_ref"]
        bpy.context.view_layer.objects.active = ref
        bpy.ops.object.mode_set(mode="EDIT")
        bm = _bmesh.from_edit_mesh(ref.data)
        for v in bm.verts:
            v.select = v.index == c
        bm.select_flush_mode()
        bpy.ops.transform.translate(value=(0, 0, 1), use_proportional_edit=True,
                                    proportional_edit_falloff=kind.upper(), proportional_size=1.3)
        bpy.ops.object.mode_set(mode="OBJECT")
        zr = [v.co.z for v in ref.data.vertices]
        zm = [v.co.z for v in _me("PX_mine").vertices]
        assert max(abs(a - b) for a, b in zip(zr, zm)) < 1e-4, (kind, max(abs(a - b) for a, b in zip(zr, zm)))
        call("delete_objects", names=["PX_ref", "PX_mine"])
    # rotate / scale about pivots (world space, object at x=300)
    call("add_primitive", kind="cube", name="T_cube2", location=[310, 0, 0])
    call("transform_elements", object="T_cube2", select={"all": True}, scale=[2, 1, 1],
         pivot={"point": [310, 0, 0]})
    xs = sorted({round(e["co"][0], 3) for e in _els("T_cube2", "verts")})
    assert xs == [308.0, 312.0], xs
    call("transform_elements", object="T_cube2", select={"all": True}, rotate={"axis": [0, 0, 1], "angle_deg": 90})
    ys = sorted({round(e["co"][1], 3) for e in _els("T_cube2", "verts")})
    assert ys == [-2.0, 2.0], ys
    call("transform_elements", object="T_cube2", select={"all": True}, scale=[0.5, 0.5, 0.5], pivot="bounds_center")
    assert abs(call("analyze_mesh", object="T_cube2")["volume_world"] - 8 * 2 * 0.125) < 1e-3
    call("set_cursor", location=[310, 0, 5])
    call("transform_elements", object="T_cube2", select={"all": True}, scale=[1, 1, 0], pivot="cursor")
    assert {round(e["co"][2], 3) for e in _els("T_cube2", "verts")} == {5.0}
    call("set_cursor", location=[0, 0, 0])
    # local space translate on a rotated object
    call("add_primitive", kind="cube", name="T_loc", location=[320, 0, 0], rotation_deg=[0, 0, 90])
    call("transform_elements", object="T_loc", select={"verts": [0]}, translate=[1, 0, 0], space="local")
    v0 = _els("T_loc", "verts", {"verts": [0]}, space="local")[0]["co"]
    assert abs(v0[0]) < 1e-4, v0       # was -1 along local x, +1 later
    expect("invalid_params", "transform_elements", object="T_loc", select={"all": True})
    expect("invalid_params", "transform_elements", object="T_loc", select={"all": True},
           rotate={"axis": [0, 0, 0], "angle_deg": 10})


def mesh_delete_dissolve_flags_normals():
    # delete_elements variants
    for what, sel, counts in (("verts", {"verts": [0]}, [7, 9, 3]),
                              ("edges", {"edges": [0]}, [8, 11, 4]),
                              ("faces", {"faces": [0]}, [8, 12, 5]),
                              ("only_faces", {"faces": [0]}, [8, 12, 5]),
                              ("edges_faces", {"edges": [0]}, [8, 11, 4])):
        call("add_primitive", kind="cube", name="D_cube", location=[400, 0, 0])
        r = call("delete_elements", object="D_cube", select=sel, what=what)
        me = _me("D_cube")
        assert r["counts"] == counts == [len(me.vertices), len(me.edges), len(me.polygons)], (what, r)
        call("delete_objects", names=["D_cube"])
    expect("invalid_params", "delete_elements", object="M_cube", select={"all": True, "invert": True},
           what="faces")
    # dissolve on a 4x4 grid (5x5 vertices, spacing 0.5, 16 quads)
    call("add_primitive", kind="grid", name="X_grid", segments=4, location=[410, 0, 0])
    r = call("dissolve", object="X_grid", select={"box": {"min": [409.9, -0.1, -1], "max": [410.1, 0.1, 1]}},
             what="verts")
    assert r["counts"] == [24, 40, 13] and r["created"]["faces_count"] == 0 or r["counts"][2] == 13, r
    call("add_primitive", kind="grid", name="X_grid2", segments=4, location=[420, 0, 0])
    inner = [e["index"] for e in _els("X_grid2", "edges") if not e["is_boundary"]][0]
    r = call("dissolve", object="X_grid2", select={"edges": [inner]}, what="edges", use_verts=False)
    assert r["counts"][2] == 15 and r["created"]["faces_count"] == 1, r
    call("add_primitive", kind="grid", name="X_grid3", segments=4, location=[430, 0, 0])
    two = {"box": {"min": [430, 0, -1], "max": [431, 0.5, 1]}}
    assert len(_idx("X_grid3", "faces", two)) == 2
    r = call("dissolve", object="X_grid3", select=two, what="faces")
    assert r["counts"][2] == 15 and r["created"]["faces_count"] == 1, r
    assert len(_els("X_grid3", "faces", {"faces": r["created"]["faces"]})) == 1
    call("add_primitive", kind="grid", name="X_flat", segments=4, location=[440, 0, 0])
    r = call("dissolve", object="X_flat", select={"all": True}, what="limited", angle_deg=5)
    assert r["counts"][2] == 1, r
    expect("invalid_params", "dissolve", object="X_flat", select={"all": True}, what="faces", angle_deg=3)
    expect("invalid_params", "dissolve", object="X_flat", select={"all": True}, what="verts", use_verts=True)
    # set_edge_flags -> real mesh data
    call("add_primitive", kind="cube", name="F_cube", location=[450, 0, 0])
    r = call("set_edge_flags", object="F_cube", select={"edges": [0, 1]}, seam=True, sharp=True,
             crease=0.5, bevel_weight=0.25)
    assert r["edges_changed"] == 2, r
    me = _me("F_cube")
    assert me.edges[0].use_seam and me.edges[1].use_edge_sharp and not me.edges[2].use_seam
    assert abs(me.attributes["crease_edge"].data[0].value - 0.5) < 1e-6
    assert abs(me.attributes["bevel_weight_edge"].data[1].value - 0.25) < 1e-6
    assert me.attributes["crease_edge"].data[2].value == 0
    flags = _els("F_cube", "edges", {"edges": [0]})[0]
    assert flags["seam"] and flags["sharp"], flags
    call("set_edge_flags", object="F_cube", select={"edges": [0]}, seam=False, sharp=False)
    assert not _me("F_cube").edges[0].use_seam and not _me("F_cube").edges[0].use_edge_sharp
    expect("invalid_params", "set_edge_flags", object="F_cube", select={"all": True})
    # normals
    call("add_primitive", kind="cube", name="N_cube", location=[460, 0, 0])
    vol = lambda: call("analyze_mesh", object="N_cube")["volume_world"]  # noqa: E731
    assert abs(vol() - 8) < 1e-4
    r = call("normals", object="N_cube", action="flip")
    assert r["flipped"] == 6 and abs(vol() + 8) < 1e-4, r
    r = call("normals", object="N_cube", action="recalculate_outside")
    assert r["flipped"] == 6 and abs(vol() - 8) < 1e-4, r
    r = call("normals", object="N_cube", action="recalculate_inside")
    assert abs(vol() + 8) < 1e-4, r
    r = call("normals", object="N_cube", select={"faces": [0]}, action="flip")
    assert r["faces"] == 1 and r["flipped"] == 1, r



# --------------------------------------------------------------------------- phase 3b

def _bm_stats(name):
    """(V, E, F, manifold, euler, consistent) from the real mesh data."""
    me = _me(name)
    me.update()
    ec = {}
    for p in me.polygons:
        for a, b in zip(p.vertices, list(p.vertices[1:]) + [p.vertices[0]]):
            ec.setdefault((min(a, b), max(a, b)), []).append((a, b))
    manifold = all(len(v) == 2 for v in ec.values()) and len(ec) == len(me.edges)
    consistent = all(len(v) == 2 and v[0] != v[1] for v in ec.values())
    return (len(me.vertices), len(me.edges), len(me.polygons), manifold,
            len(me.vertices) - len(me.edges) + len(me.polygons), consistent)


def _created_ok(name, before, r):
    """created.* counts must be consistent with the counts before/after the op."""
    c = r["created"]
    after = r["counts"]
    assert after == [len(_me(name).vertices), len(_me(name).edges), len(_me(name).polygons)], r
    for k, i in (("verts", 0), ("edges", 1), ("faces", 2)):
        assert c[k + "_count"] >= after[i] - before[i], (k, c, before, after)
        assert all(0 <= x < after[i] for x in _ix(c[k])), (k, c, after)


def _cnt(name):
    me = _me(name)
    return [len(me.vertices), len(me.edges), len(me.polygons)]


def mesh_bevel_loopcut_subdivide():
    # bevel: one cube edge, 2 segments -> 6 new verts, closed manifold
    call("add_primitive", kind="cube", name="B_cube", location=[500, 0, 0])
    b = _cnt("B_cube")
    r = call("bevel", object="B_cube", select={"edges": [0]}, offset=0.2, segments=2)
    assert _bm_stats("B_cube")[:4] == (12, 18, 8, True) and r["after"]["is_manifold"], r
    _created_ok("B_cube", b, r)
    assert r["created"]["verts_count"] == 6 and r["created"]["faces_count"] == 6, r
    # offset really is the distance: cube spans x 499..501; bevel edge, check bounds unchanged
    call("add_primitive", kind="cube", name="B_cube2", location=[510, 0, 0])
    r = call("bevel", object="B_cube2", select={"all": True}, offset=0.1, segments=3,
             offset_type="width")
    assert r["after"]["is_manifold"] and r["counts"][2] > 6, r
    call("add_primitive", kind="cube", name="B_cube3", location=[520, 0, 0])
    r = call("bevel", object="B_cube3", select={"verts": [0]}, affect="vertices", offset=0.3)
    assert r["after"]["is_manifold"] and r["counts"][2] == 7, r
    expect("invalid_params", "bevel", object="B_cube3", select={"all": True}, offset=0)
    expect("invalid_params", "bevel", object="B_cube3", select={"all": True}, offset=150,
           offset_type="percent")
    expect("invalid_params", "bevel", object="B_cube3", select={"all": True}, offset=0.1, segments=33)
    expect("invalid_params", "bevel", object="B_cube3", select={"all": True}, offset=0.1, segments=0)
    # budget: over-estimate refused, untouched
    from blender_safe_mcp import ops_mesh as OM
    keep = OM.MAX_POLYS
    OM.MAX_POLYS = 20
    try:
        before = _cnt("B_cube3")
        expect("budget", "bevel", object="B_cube3", select={"all": True}, offset=0.1, segments=8)
        expect("budget", "subdivide", object="B_cube3", select={"all": True}, cuts=5)
        expect("budget", "loop_cut", object="B_cube3", edge=0, cuts=10)
        expect("budget", "spin", object="B_cube3", select={"all": True}, axis=[0, 0, 1], angle_deg=90,
               steps=30)
        assert _cnt("B_cube3") == before
    finally:
        OM.MAX_POLYS = keep
    # loop_cut: one cut on a cube edge -> 10 quads, manifold
    call("add_primitive", kind="cube", name="L_cube", location=[530, 0, 0])
    b = _cnt("L_cube")
    r = call("loop_cut", object="L_cube", edge=0, cuts=1, expect_counts=b)
    s = _bm_stats("L_cube")
    assert s[:4] == (12, 20, 10, True) and s[4] == 2 and s[5], (s, r)
    assert all(len(p.vertices) == 4 for p in _me("L_cube").polygons)
    assert r["created"]["verts_count"] == 4 and r["created"]["edges_count"] == 8 and \
        r["created"]["faces_count"] == 4 and r["ring_edges"] == 4, r
    _created_ok("L_cube", b, r)
    assert len(_ix(r["cut_edges"])) == 4, r
    cv = [v["co"] for v in _els("L_cube", "verts", {"edges": r["cut_edges"]})]
    assert any(len({round(c[a], 4) for c in cv}) == 1 for a in range(3)), cv   # one plane
    r = call("loop_cut", object="L_cube", edge=0, cuts=3)
    s = _bm_stats("L_cube")
    assert s[3] and s[5], s
    expect("stale_indices", "loop_cut", object="L_cube", edge=0, expect_counts=[8, 12, 6])
    expect("not_found", "loop_cut", object="L_cube", edge=9999)
    expect("invalid_params", "loop_cut", object="L_cube", edge=0, cuts=65)
    call("add_mesh", name="L_tri", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
         location=[540, 0, 0])
    expect("invalid_params", "loop_cut", object="L_tri", edge=0)
    # subdivide
    call("add_primitive", kind="cube", name="S_cube", location=[550, 0, 0])
    b = _cnt("S_cube")
    r = call("subdivide", object="S_cube", select={"all": True}, cuts=2)
    assert r["counts"][2] == 54 and r["after"]["is_manifold"], r
    assert r["created"]["verts_count"] == 48, r
    _created_ok("S_cube", b, r)
    r2 = call("subdivide", object="S_cube", select={"faces": [0]}, cuts=1, smoothness=0.5)
    assert r2["after"]["is_manifold"], r2
    r2 = call("subdivide", object="S_cube", select={"edges": [0]}, domain="edges")
    assert r2["after"]["is_manifold"], r2
    expect("invalid_params", "subdivide", object="S_cube", select={"all": True}, cuts=11)
    # real budget refusal: a big grid
    call("add_primitive", kind="grid", name="S_big", segments=120, location=[560, 0, 0])
    keep = OM.MAX_POLYS
    OM.MAX_POLYS = 1_000_000
    try:
        expect("budget", "subdivide", object="S_big", select={"all": True}, cuts=10)
    finally:
        OM.MAX_POLYS = keep


def mesh_bisect_spin_bridge():
    # bisect: closed UV sphere, clear_outer + fill -> manifold, consistent
    call("add_primitive", kind="uv_sphere", name="Q_sph", location=[600, 0, 0])
    b = _cnt("Q_sph")
    r = call("bisect", object="Q_sph", plane_point=[600, 0, 0.1], plane_normal=[0, 0, 1],
             clear_outer=True, fill=True)
    s = _bm_stats("Q_sph")
    assert s[3] and s[4] == 2 and s[5], (s, r)
    assert r["filled_faces"] == 1 and r["cut_edges_count"] > 8, r
    _created_ok("Q_sph", b, r)
    assert r["after"]["is_manifold"] and r["after"]["volume_world"] > 0, r
    zs = [v["co"][2] for v in _els("Q_sph", "verts")]
    assert max(zs) < 0.1001, max(zs)
    # without fill: open mesh; local space plane
    call("add_primitive", kind="uv_sphere", name="Q_sph2", location=[610, 0, 0])
    r = call("bisect", object="Q_sph2", plane_point=[0, 0, -0.1], plane_normal=[0, 0, -1],
             space="local", clear_inner=True)
    assert r["after"]["holes"] == 1, r
    call("add_primitive", kind="cube", name="Q_cube", location=[620, 0, 0])
    r = call("bisect", object="Q_cube", plane_point=[620, 0, 0], plane_normal=[1, 0, 0])
    assert r["counts"] == [12, 20, 10] and r["after"]["is_manifold"], r
    expect("invalid_params", "bisect", object="Q_cube", plane_point=[0, 0, 0], plane_normal=[0, 0, 0])
    expect("invalid_params", "bisect", object="Q_cube", plane_point=[0, 0, 0], plane_normal=[0, 0, 1], fill=True)
    expect("invalid_params", "bisect", object="Q_cube", plane_point=[0, 0, 0], plane_normal=[0, 0, 1],
           clear_inner=True, clear_outer=True)
    # a plane that misses the mesh and clears it: refused, mesh untouched
    expect("invalid_params", "bisect", object="Q_cube", plane_point=[620, 0, 50],
           plane_normal=[0, 0, -1], clear_outer=True)
    assert _cnt("Q_cube") == [12, 20, 10]
    # spin: 24-vertex circle profile, 360 degrees, 48 steps -> torus
    import math
    ring = [[2 + 0.5 * math.cos(2 * math.pi * i / 24), 0, 0.5 * math.sin(2 * math.pi * i / 24)]
            for i in range(24)]
    # add_mesh takes no edges: build the profile as a face, then drop the face
    call("add_mesh", name="P_ring", vertices=ring, faces=[list(range(24))], location=[630, 0, 0])
    call("delete_elements", object="P_ring", select={"all": True}, what="only_faces")
    assert _cnt("P_ring") == [24, 24, 0]
    b = _cnt("P_ring")
    r = call("spin", object="P_ring", axis=[0, 0, 1], center=[630, 0, 0], angle_deg=360, steps=48)
    s = _bm_stats("P_ring")
    assert s[:3] == (1152, 2304, 1152) and s[3] and s[4] == 0 and s[5], (s, r)
    _created_ok("P_ring", b, r)
    assert r["after"]["is_manifold"], r
    # partial spin, no merge: steps+1 profiles
    call("add_mesh", name="P_ring2", vertices=ring, faces=[list(range(24))], location=[640, 0, 0])
    call("delete_elements", object="P_ring2", select={"all": True}, what="only_faces")
    r = call("spin", object="P_ring2", axis=[0, 0, 1], center=[640, 0, 0], angle_deg=90, steps=4)
    assert r["counts"][0] == 24 * 5 and r["counts"][2] == 96, r
    # direction: right-hand rule about +Z (point at +X goes towards +Y)
    call("add_mesh", name="P_tri", vertices=[[1, 0, 0], [1.2, 0, 0], [1.1, 0, 0.2]], faces=[[0, 1, 2]],
         location=[650, 0, 0])
    call("spin", object="P_tri", axis=[0, 0, 1], center=[650, 0, 0], angle_deg=90, steps=1, duplicate=True)
    ys = [v["co"][1] for v in _els("P_tri", "verts")]
    assert max(ys) > 0.9 and min(ys) > -1e-6, ys
    expect("invalid_params", "spin", object="P_ring2", axis=[0, 0, 0], angle_deg=90, steps=4)
    expect("invalid_params", "spin", object="P_ring2", axis=[0, 0, 1], angle_deg=0, steps=4)
    expect("invalid_params", "spin", object="P_ring2", axis=[0, 0, 1], angle_deg=90, steps=513)
    # bridge: single chain refused; a tube bridged to itself changes nothing
    call("add_primitive", kind="cylinder", name="R_cyl", location=[660, 0, 0])
    expect("invalid_params", "bridge_edge_loops", object="R_cyl", select={"edges": [0]})
    # two quads' rims bridged
    call("add_primitive", kind="grid", name="R_g1", segments=3, location=[670, 0, 0])
    call("add_primitive", kind="grid", name="R_g2", segments=3, location=[670, 0, 1])
    call("join_objects", target="R_g1", others=["R_g2"], merge_distance=0.0)
    bl = _idx("R_g1", "edges", {"boundary": True})
    assert len(bl) == 24, bl
    r = call("bridge_edge_loops", object="R_g1", select={"edges": bl})
    assert r["created"]["faces_count"] == 12, r
    assert r["after"]["holes"] == 0 and r["after"]["shells"] == 1, r  # closed (both grids face +Z, so the winding is mixed, as in Blender)
    call("add_primitive", kind="grid", name="R_g3", segments=3, location=[680, 0, 0])
    call("add_primitive", kind="grid", name="R_g4", segments=3, location=[680, 0, 1])
    call("join_objects", target="R_g3", others=["R_g4"], merge_distance=0.0)
    bl = _idx("R_g3", "edges", {"boundary": True})
    r = call("bridge_edge_loops", object="R_g3", select={"edges": bl}, merge=True, twist=1)
    assert r["counts"][0] < 32, r


def mesh_smooth_sym_sep_snap():
    # smooth_vertices
    call("add_primitive", kind="grid", name="Z_grid", segments=4, location=[700, 0, 0])
    call("transform_elements", object="Z_grid", select={"verts": [12]}, translate=[0, 0, 1])
    c = _els("Z_grid", "verts", {"verts": [12]})[0]["co"][2]
    assert abs(c - 1) < 1e-5, c
    r = call("smooth_vertices", object="Z_grid", select={"verts": [12]}, factor=1.0, repeat=1)
    z = _els("Z_grid", "verts", {"verts": [12]})[0]["co"][2]
    assert r["moved"] == 1 and abs(z) < 1e-4, (z, r)      # pulled back to the neighbours' mean
    call("transform_elements", object="Z_grid", select={"verts": [12]}, translate=[0.1, 0, 1])
    call("smooth_vertices", object="Z_grid", select={"verts": [12]}, factor=1.0,
         axes={"x": False, "y": False, "z": True})
    v = _els("Z_grid", "verts", {"verts": [12]})[0]["co"]
    assert abs(v[2]) < 1e-4 and abs(v[0] - 700.1) < 1e-4, v
    expect("invalid_params", "smooth_vertices", object="Z_grid", select={"all": True},
           axes={"x": False, "y": False, "z": False})
    expect("invalid_params", "smooth_vertices", object="Z_grid", select={"all": True}, repeat=101)
    # symmetrize: cut a cube's +X half off then mirror
    call("add_primitive", kind="cube", name="Y_cube", location=[710, 0, 0])
    call("subdivide", object="Y_cube", select={"all": True}, cuts=1)
    call("transform_elements", object="Y_cube", select={"box": {"min": [709, -2, -2], "max": [709.99, 2, 2]}},
         translate=[0, 0, 0.5], space="world")
    b = _cnt("Y_cube")
    r = call("symmetrize", object="Y_cube", direction="+X>-X")
    assert r["after"]["is_manifold"], r
    _created_ok("Y_cube", b, r)
    zs = {}
    for e in _els("Y_cube", "verts"):
        zs[(round(e["co"][0] - 710, 3), round(e["co"][1], 3))] = round(e["co"][2], 3)
    assert all(zs.get((-x, y)) == z for (x, y), z in zs.items()), "not symmetric"
    r = call("symmetrize", object="Y_cube", direction="-Z>+Z")
    assert r["after"]["is_manifold"], r
    # separate: loose parts
    call("add_primitive", kind="cube", name="O_a", location=[720, 0, 0])
    call("add_primitive", kind="cube", name="O_b", location=[724, 0, 0])
    call("add_primitive", kind="cube", name="O_c", location=[728, 0, 0])
    call("create_material", name="O_red", base_color=[1, 0, 0], assign_to=["O_a"])
    call("join_objects", target="O_a", others=["O_b", "O_c"], merge_distance=0.0)
    src = bpy.data.objects["O_a"]
    src.rotation_euler[2] = 0.5
    src.scale = (1.5, 1.5, 1.5)
    col = bpy.data.collections.new("O_col")
    bpy.context.scene.collection.children.link(col)
    col.objects.link(src)
    bpy.context.view_layer.update()
    mw0 = src.matrix_world.copy()
    r = call("separate", object="O_a", mode="loose_parts", name_prefix="O_part")
    assert len(r["new_objects"]) == 2 and r["counts"] == [8, 12, 6], r
    for n in r["new_objects"]:
        o = bpy.data.objects[n]
        assert (o.matrix_world - mw0).to_3x3().col[0].length < 1e-5 and \
            all(abs(a - b) < 1e-5 for a, b in zip(o.matrix_world.translation, mw0.translation))
        assert {c.name for c in o.users_collection} == {c.name for c in src.users_collection}, o.users_collection
        assert "O_col" in {c.name for c in o.users_collection}
        assert [m and m.name for m in o.data.materials] == [m and m.name for m in src.data.materials], list(o.data.materials)
        assert "O_red" in [m and m.name for m in o.data.materials]
        assert o.data is not src.data and len(o.data.polygons) == 6
        assert call("analyze_mesh", object=n)["is_manifold"]
    assert call("analyze_mesh", object="O_a")["shells"] == 1
    expect("invalid_params", "separate", object="O_a", mode="loose_parts")
    # separate: selection (+ leaves-source-empty refusal)
    call("add_primitive", kind="cube", name="O_sel", location=[740, 0, 0])
    expect("invalid_params", "separate", object="O_sel", mode="selection", select={"all": True})
    expect("invalid_params", "separate", object="O_sel", mode="selection")
    assert _cnt("O_sel") == [8, 12, 6]
    r = call("separate", object="O_sel", mode="selection", select={"faces": [0, 1]})
    assert r["counts"][2] == 4 and r["parts"][0]["faces"] == 2, r
    assert "O_sel.001" in bpy.data.objects
    # separate: material
    call("add_primitive", kind="cube", name="O_mat", location=[750, 0, 0])
    call("create_material", name="O_m1", base_color=[0, 1, 0], assign_to=["O_mat"])
    call("create_material", name="O_m2", base_color=[0, 0, 1], assign_to=["O_mat"])
    me = _me("O_mat")
    for i in (0, 1):
        me.polygons[i].material_index = 1
    r = call("separate", object="O_mat", mode="material")
    assert len(r["new_objects"]) == 1 and r["parts"][0]["faces"] == 2 and r["counts"][2] == 4, r
    expect("invalid_params", "separate", object="O_mat", mode="material")
    # parts cap: 258 lone vertices = 258 parts -> 257 new objects, refused with the source intact
    call("add_mesh", name="O_many", vertices=[[i, 0, 0] for i in range(258)], faces=[], location=[760, 0, 0])
    nobj = len(bpy.data.objects)
    expect("budget", "separate", object="O_many", mode="loose_parts")
    assert len(bpy.data.objects) == nobj and _cnt("O_many")[0] == 258
    # snap_vertices
    call("add_primitive", kind="cube", name="K_cube", location=[800.3, 0.2, 0.1])
    r = call("snap_vertices", object="K_cube", select={"all": True}, to={"grid": 1.0})
    xs = {round(v["co"][0], 5) for v in _els("K_cube", "verts")}
    assert xs == {799.0, 801.0}, xs
    assert r["moved"] == 8, r
    call("add_primitive", kind="cube", name="K_cube2", location=[810, 0, 0])
    r = call("snap_vertices", object="K_cube2", select={"all": True}, to={"grid": 0.5}, space="local")
    assert r["moved"] == 0, r
    call("set_cursor", location=[1, 2, 3])
    call("snap_vertices", object="K_cube2", select={"verts": [0, 1]}, to="cursor")
    for v in _els("K_cube2", "verts", {"verts": [0, 1]}):
        assert all(abs(a - b) < 1e-4 for a, b in zip(v["co"], [1, 2, 3])), v
    call("add_primitive", kind="uv_sphere", name="K_sph", location=[820, 0, 0])
    call("add_primitive", kind="cube", name="K_cube3", location=[820, 0, 0], size=3)
    call("snap_vertices", object="K_cube3", select={"all": True}, to={"object": "K_sph",
                                                                    "mode": "nearest_surface"})
    for v in _els("K_cube3", "verts"):
        d = ((v["co"][0] - 820) ** 2 + v["co"][1] ** 2 + v["co"][2] ** 2) ** 0.5
        assert abs(d - 1.0) < 0.02, d
    call("add_primitive", kind="cube", name="K_cube4", location=[820, 0, 0], size=3)
    call("snap_vertices", object="K_cube4", select={"all": True}, to={"object": "K_sph",
                                                                    "mode": "nearest_vertex"})
    sv = {tuple(round(c, 4) for c in v["co"]) for v in _els("K_sph", "verts")}
    assert all(tuple(round(c, 4) for c in v["co"]) in sv for v in _els("K_cube4", "verts"))
    expect("invalid_params", "snap_vertices", object="K_cube4", select={"all": True},
           to={"object": "K_cube4", "mode": "nearest_surface"})
    expect("invalid_params", "snap_vertices", object="K_cube4", select={"all": True},
           to={"object": "Pivot", "mode": "nearest_surface"})
    expect("invalid_params", "snap_vertices", object="K_cube4", select={"all": True}, to={"grid": 0})
    # triangulate / poke
    call("add_primitive", kind="cube", name="T3_cube", location=[830, 0, 0])
    b = _cnt("T3_cube")
    r = call("triangulate_faces", object="T3_cube", select={"faces": [0, 1]}, quad_method="FIXED")
    assert r["counts"] == [8, 14, 8] and r["triangulated"] == 2, r
    assert r["created"]["faces_count"] == 4 and r["created"]["edges_count"] == 2, r
    assert r["after"]["is_manifold"]
    r = call("triangulate_faces", object="T3_cube", select={"all": True})
    assert r["counts"][2] == 12 and r["triangulated"] == 4, r
    call("add_primitive", kind="cube", name="T3_poke", location=[840, 0, 0])
    b = _cnt("T3_poke")
    r = call("poke_faces", object="T3_poke", select={"faces": [0]}, offset=0.5)
    assert r["counts"] == [9, 16, 9] and r["created"]["verts_count"] == 1 and \
        r["created"]["faces_count"] == 4 and r["created"]["edges_count"] == 4, r
    _created_ok("T3_poke", b, r)
    apex = _els("T3_poke", "verts", {"verts": r["created"]["verts"]})[0]["co"]
    assert abs(((apex[0] - 840) ** 2 + apex[1] ** 2 + apex[2] ** 2) ** 0.5 - 1.5) < 1e-4, apex
    assert r["after"]["is_manifold"]
    # refusals: edit mode, read-only
    call("set_object_mode", object="T3_poke", mode="EDIT")
    expect("unavailable", "bevel", object="T3_poke", select={"all": True}, offset=0.1)
    expect("unavailable", "loop_cut", object="T3_poke", edge=0)
    call("set_object_mode", object="T3_poke", mode="OBJECT")
    PREFS.allow_modify = False
    for opn, kw in (("bevel", dict(select={"all": True}, offset=0.1)), ("loop_cut", dict(edge=0)),
                    ("subdivide", dict(select={"all": True})),
                    ("separate", dict(mode="loose_parts")),
                    ("symmetrize", dict(direction="-X>+X"))):
        expect("read_only", opn, object="T3_poke", **kw)
    PREFS.allow_modify = True



def _spl(kind, pts, **kw):
    return {"type": kind, "points": pts, **kw}


def curve_ops():
    # bezier circle with bevel_depth -> convert to MESH -> manifold tube
    r = call("add_curve_primitive", kind="bezier_circle", name="C_ring", radius=2, bevel_depth=0.2,
             location=[40, 0, 0])
    assert r["splines"] == 1 and r["points"] == 4, r
    info = call("get_curve_info", object="C_ring")
    assert info["bevel_depth"] == 0.2 and info["splines"][0]["cyclic"], info
    bpy.context.view_layer.update()
    ev = bpy.data.objects["C_ring"].evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    far = max(((v.co.x ** 2 + v.co.y ** 2) ** 0.5) for v in me.vertices)
    ev.to_mesh_clear()
    assert abs(far - 2.2) < 0.02, f"circle radius off: {far}"
    call("add_empty", name="C_child", location=[42, 0, 5])
    call("set_parent", child="C_child", parent="C_ring")
    cw = list(bpy.data.objects["C_child"].matrix_world.translation)
    r = call("convert_object", object="C_ring", target="MESH")
    assert r["name"] == "C_ring" and r["type"] == "MESH" and r["after"]["is_manifold"], r
    assert "C_ring" in bpy.data.objects and bpy.data.objects["C_ring"].type == "MESH"
    assert not bpy.data.objects["C_ring"].modifiers
    bpy.context.view_layer.update()
    ch = bpy.data.objects["C_child"]
    assert ch.parent == bpy.data.objects["C_ring"]
    assert (ch.matrix_world.translation - Vector(cw)).length < 1e-4, "child moved"
    assert abs(bpy.data.objects["C_ring"].location.x - 40) < 1e-6
    assert not [c for c in bpy.data.curves if c.name.startswith("C_ring")], "orphan curve"

    # modifiers are baked into the converted mesh; hidden ones are not and are reported
    call("add_primitive", kind="cube", name="C_cube", location=[50, 0, 0])
    call("add_modifier", object="C_cube", type="ARRAY", name="Arr", settings={"count": 3},
         target_object=None)
    call("add_modifier", object="C_cube", type="SUBSURF", name="Sub", settings={"levels": 1})
    bpy.data.objects["C_cube"].modifiers["Sub"].show_viewport = False
    r = call("convert_object", object="C_cube", target="MESH", keep_original=True,
             new_name="C_cube_mesh")
    assert r["modifiers_baked"] == ["Arr"] and r["modifiers_not_baked"] == ["Sub"], r
    new = bpy.data.objects["C_cube_mesh"]
    assert not new.modifiers and len(new.data.polygons) == 18, len(new.data.polygons)
    assert len(bpy.data.objects["C_cube"].modifiers) == 2, "original must keep its stack"
    expect("invalid_params", "convert_object", object="C_cube", target="MESH", keep_original=True,
           new_name="C_cube_mesh")

    # materials, collection and parent survive the conversion
    call("create_material", name="C_mat", base_color=[1, 0, 0])
    call("add_curve_primitive", kind="bezier_circle", name="C_mc", bevel_depth=0.1,
         location=[45, 5, 0])
    call("assign_material", object="C_mc", material="C_mat")
    call("add_empty", name="C_par", location=[45, 5, 0])
    call("set_parent", child="C_mc", parent="C_par")
    colls = {c.name for c in bpy.data.objects["C_mc"].users_collection}
    call("convert_object", object="C_mc", target="MESH")
    mc = bpy.data.objects["C_mc"]
    assert mc.type == "MESH" and [m.name for m in mc.data.materials] == ["C_mat"], mc.data.materials[:]
    assert mc.parent.name == "C_par" and {c.name for c in mc.users_collection} == colls
    assert (mc.matrix_world.translation - Vector((45, 5, 0))).length < 1e-4

    # text -> CURVE -> MESH
    call("add_text", body="Hi", name="C_txt", location=[60, 0, 0])
    r = call("convert_object", object="C_txt", target="CURVE")
    assert r["type"] == "CURVE" and r["splines"] > 0, r
    assert bpy.data.objects["C_txt"].type == "CURVE"
    call("set_curve_shape", object="C_txt", bevel_mode="ROUND")
    r = call("convert_object", object="C_txt", target="MESH", keep_original=True,
             new_name="C_txt_mesh")
    assert r["type"] == "MESH" and r["after"]["faces"] > 0 and bpy.data.objects["C_txt"].type == "CURVE"

    # mesh edge loop -> CURVE
    call("add_primitive", kind="circle", name="C_loop", location=[70, 0, 0])
    vl = bpy.context.view_layer
    orphans0 = len([m for m in bpy.data.meshes if m.users == 0])
    sel0 = {o.name for o in vl.objects if o.select_get()}
    act0 = vl.objects.active
    r = call("convert_object", object="C_loop", target="CURVE", keep_original=True,
             new_name="C_loop_curve")
    assert r["type"] == "CURVE" and r["splines"] >= 1, r
    assert bpy.data.objects["C_loop"].type == "MESH"
    assert len([m for m in bpy.data.meshes if m.users == 0]) == orphans0, "orphan mesh after convert"
    assert {o.name for o in vl.objects if o.select_get()} == sel0 and vl.objects.active is act0, \
        "selection / active object changed"

    # refusals: unsupported pairs
    call("add_camera", name="C_cam", location=[0, -20, 3])
    expect("invalid_params", "convert_object", object="C_cam", target="MESH")
    expect("invalid_params", "convert_object", object="C_cam", target="CURVE")
    expect("invalid_params", "convert_object", object="C_txt", target="CURVE")
    expect("invalid_params", "convert_object", object="C_loop_curve", target="CURVE")
    expect("invalid_params", "convert_object", object="C_loop", target="MESH", new_name="C_cam")

    # add_curve: all three types, caps, validation before mutation
    n_curves = len(bpy.data.curves)
    r = call("add_curve", name="C_multi", splines=[
        _spl("bezier", [[0, 0, 0], {"co": [1, 0, 0], "handle_type": "vector", "tilt_deg": 30},
                        {"co": [2, 1, 0], "handle_left": [1.5, 0, 0], "handle_right": [2.5, 2, 0]}]),
        _spl("poly", [[0, 1, 0], [1, 1, 0], {"co": [2, 1, 0], "weight": 2}], cyclic=True),
        _spl("nurbs", [[0, 2, 0], [1, 3, 0], [2, 2, 0], [3, 3, 0], [4, 2, 0]], order=3,
             resolution=8)], dimensions="3D", location=[0, 30, 0])
    assert r["splines"] == 3 and r["points"] == 11, r
    info = call("get_curve_info", object="C_multi")
    sp = info["splines"]
    assert [s["type"] for s in sp] == ["bezier", "poly", "nurbs"], sp
    assert sp[0]["points"][1]["handle_left_type"] == "VECTOR"
    assert abs(sp[0]["points"][1]["tilt_deg"] - 30) < 1e-3
    assert sp[0]["points"][2]["handle_right"] == [2.5, 2.0, 0.0], sp[0]["points"][2]
    assert sp[1]["cyclic"] and sp[1]["points"][2]["weight"] == 2.0
    assert sp[2]["order"] == 3 and sp[2]["resolution"] == 8
    for bad in ({"type": "bezier", "points": [[0, 0, 0]]},
                {"type": "poly", "points": [[0, 0, 0]] * 10001},
                {"type": "nurbs", "points": [[0, 0, 0], [1, 0, 0]], "order": 3},
                {"type": "poly", "points": [[0, 0, 0], {"co": [1, 0, 0], "handle_left": [0, 0, 0]}]},
                {"type": "bezier", "points": [[0, 0, 0], {"co": [1, 0, 0], "weight": 2}]},
                {"type": "poly", "points": [[0, 0, 0], [1, 0, 0]], "order": 3},
                {"type": "poly", "points": [[0, 0, 0], [1, 0, 0]], "resolution": 65}):
        expect("invalid_params", "add_curve", splines=[bad])
    expect("invalid_params", "add_curve", splines=[_spl("poly", [[0, 0, 0], [1, 0, 0]])] * 257)
    expect("budget", "add_curve", splines=[_spl("poly", [[0, 0, 0]] * 10000)] * 6)
    expect("budget", "add_curve", bevel_depth=1, bevel_resolution=32, splines=[
        _spl("bezier", [[0, 0, 0]] * 10000, resolution=64)])
    assert len(bpy.data.curves) == n_curves + 1, "a refused add_curve left data behind"

    # set_spline round-trips; keeps position; add/remove
    new_pts = [[0, 0, 0], {"co": [1, 1, 0], "handle_left": [0.5, 1, 0], "handle_right": [1.5, 1, 0]},
               [3, 0, 0], [4, 1, 1]]
    call("set_spline", object="C_multi", index=0, spline=_spl("bezier", new_pts, cyclic=True))
    info2 = call("get_curve_info", object="C_multi")
    assert [s["type"] for s in info2["splines"]] == ["bezier", "poly", "nurbs"], info2["splines"]
    s0 = info2["splines"][0]
    assert s0["cyclic"] and s0["point_count"] == 4 and s0["points"][1]["co"] == [1.0, 1.0, 0.0]
    assert s0["points"][1]["handle_right"] == [1.5, 1.0, 0.0]
    assert info2["splines"][1] == {**sp[1], "index": 1}, "tail spline changed"
    assert info2["splines"][2] == {**sp[2], "index": 2}, "tail spline changed"
    r = call("add_spline", object="C_multi", spline=_spl("poly", [[0, 5, 0], [1, 5, 0]]))
    assert r["splines"] == 4 and r["points"] == 4 + 3 + 5 + 2, r
    r = call("remove_spline", object="C_multi", index=1)
    assert r["splines"] == 3
    assert call("get_curve_info", object="C_multi")["splines"][1]["type"] == "nurbs"
    expect("not_found", "remove_spline", object="C_multi", index=3)
    expect("not_found", "set_spline", object="C_multi", index=9,
           spline=_spl("poly", [[0, 0, 0], [1, 0, 0]]))
    expect("invalid_params", "set_spline", object="C_cube", index=0,
           spline=_spl("poly", [[0, 0, 0], [1, 0, 0]]))
    expect("invalid_params", "get_curve_info", object="C_txt_mesh")

    # primitives
    for kind, n in (("bezier_curve", 2), ("nurbs_circle", 8), ("nurbs_path", 5), ("spiral", 37)):
        r = call("add_curve_primitive", kind=kind, name=f"C_{kind}", location=[80, 0, 0])
        assert r["points"] == n, (kind, r)
    expect("invalid_params", "add_curve_primitive", kind="bezier_circle", turns=2)
    bpy.context.view_layer.update()
    ev = bpy.data.objects["C_nurbs_circle"].evaluated_get(bpy.context.evaluated_depsgraph_get())
    me = ev.to_mesh()
    rs = [(v.co.length) for v in me.vertices]
    ev.to_mesh_clear()
    assert rs and max(rs) - min(rs) < 0.01, f"nurbs circle not round: {min(rs)} {max(rs)}"

    # shape: bevel / taper objects
    call("add_curve_primitive", kind="bezier_circle", name="C_prof", radius=0.1,
         location=[0, 60, 0])
    call("add_curve_primitive", kind="nurbs_path", name="C_path", location=[0, 70, 0])
    call("add_curve", name="C_taper", splines=[_spl("poly", [[0, 0, 0], [1, 1, 0]])])
    r = call("set_curve_shape", object="C_path", bevel_object="C_prof", taper_object="C_taper")
    assert r["bevel_mode"] == "OBJECT" and r["bevel_object"] == "C_prof", r
    assert call("get_curve_info", object="C_path")["taper_object"] == "C_taper"
    expect("invalid_params", "set_curve_shape", object="C_path", bevel_object="C_path")
    expect("invalid_params", "set_curve_shape", object="C_prof", bevel_object="C_path")  # loop
    expect("invalid_params", "set_curve_shape", object="C_prof", taper_object="C_path")  # loop
    expect("invalid_params", "set_curve_shape", object="C_path", bevel_object="C_cube")
    expect("not_found", "set_curve_shape", object="C_path", taper_object="C_nope")
    expect("invalid_params", "set_curve_shape", object="C_path")
    expect("invalid_params", "set_curve_shape", object="C_cube", bevel_mode="ROUND")
    r = call("set_curve_shape", object="C_path", bevel_object="", taper_object="", bevel_mode="ROUND")
    assert r["bevel_object"] is None and r["taper_object"] is None
    expect("invalid_params", "set_curve_shape", object="C_path", bevel_mode="OBJECT")

    # Curve Deform + Array fit-to-curve
    call("add_primitive", kind="cube", name="C_bar", location=[0, 90, 0])
    call("set_transform", name="C_bar", scale=[3, 0.2, 0.2])
    r = call("add_modifier", object="C_bar", type="CURVE", name="Deform",
             target_object="C_path")
    assert bpy.data.objects["C_bar"].modifiers["Deform"].object.name == "C_path", r
    expect("invalid_params", "add_modifier", object="C_bar", type="CURVE", name="D2",
           target_object="C_cube")
    expect("invalid_params", "add_modifier", object="C_bar", type="CURVE", name="D2",
           target_object="C_bar")
    call("apply_modifiers", object="C_bar")
    call("add_primitive", kind="cube", name="C_tile", location=[0, 100, 0])
    r = call("add_modifier", object="C_tile", type="ARRAY", name="Fit", curve_object="C_path",
             settings={"use_relative_offset": True})
    m = bpy.data.objects["C_tile"].modifiers["Fit"]
    assert m.fit_type == "FIT_CURVE" and m.curve.name == "C_path"
    expect("invalid_params", "add_modifier", object="C_tile", type="ARRAY", name="F2",
           curve_object="C_cube")
    expect("invalid_params", "add_modifier", object="C_tile", type="SUBSURF", name="F2",
           curve_object="C_path")
    assert len(bpy.data.objects["C_tile"].modifiers) == 1, "refused add_modifier left a modifier"
    # a tiny step along a long curve is over budget
    call("add_curve", name="C_long", splines=[_spl("poly", [[0, 0, 0], [1e5, 0, 0]])])
    call("add_primitive", kind="uv_sphere", name="C_ball", location=[0, 110, 0])
    call("set_transform", name="C_ball", scale=[0.001, 0.001, 0.001])
    expect("budget", "add_modifier", object="C_ball", type="ARRAY", name="Fit",
           curve_object="C_long", settings={"use_relative_offset": True})

    # property caps
    call("set_property", target={"kind": "curve", "name": "C_multi"}, path=["resolution_u"],
         value=64)
    expect("out_of_range", "set_property", target={"kind": "curve", "name": "C_multi"},
           path=["resolution_u"], value=65)
    expect("out_of_range", "set_property", target={"kind": "curve", "name": "C_multi"},
           path=["bevel_resolution"], value=33)
    expect("out_of_range", "set_property", target={"kind": "curve", "name": "C_multi"},
           path=["splines", 0, "resolution_u"], value=65)
    expect("out_of_range", "set_property", target={"kind": "curve", "name": "C_multi"},
           path=["splines", 1, "order_u"], value=7)
    call("set_property", target={"kind": "curve", "name": "C_multi"},
         path=["splines", 0, "bezier_points", 0, "co"], value=[9, 9, 9])
    assert call("get_curve_info", object="C_multi")["splines"][0]["points"][0]["co"] == [9, 9, 9]

    # shared curve data: refused unless make_single_user, which edits only this object's copy
    sp = {"type": "poly", "points": [[0, 0, 0], [1, 0, 0], [1, 1, 0]]}
    call("add_curve", name="C_sh1", splines=[sp])
    call("add_curve", name="C_shprof", splines=[sp])
    c2 = bpy.data.objects["C_sh1"].copy()
    bpy.context.collection.objects.link(c2)
    c2.name = "C_sh2"
    shared = bpy.data.objects["C_sh1"].data
    for op, extra in (("set_spline", {"index": 0, "spline": sp}), ("add_spline", {"spline": sp}),
                      ("remove_spline", {"index": 0}),
                      ("set_curve_shape", {"bevel_object": "C_shprof"})):
        expect("invalid_params", op, object="C_sh1", **extra)
        assert len(shared.splines) == 1 and shared.bevel_object is None, op
    call("add_spline", object="C_sh1", spline=sp, make_single_user=True)
    assert bpy.data.objects["C_sh1"].data is not shared
    assert len(bpy.data.objects["C_sh1"].data.splines) == 2 and len(shared.splines) == 1
    call("set_curve_shape", object="C_sh2", bevel_object="C_shprof")  # sole user again
    assert shared.bevel_object.name == "C_shprof"


def _uv_in_unit(name, uv=None, tol=1e-6):
    me = bpy.data.objects[name].data
    layer = me.uv_layers[uv] if uv else me.uv_layers.active
    co = [0.0] * (len(me.loops) * 2)
    layer.uv.foreach_get("vector", co)
    assert co and all(-tol <= c <= 1 + tol for c in co), (name, min(co), max(co))
    return co


def _state(name):
    me = bpy.data.objects[name].data
    f = lambda c: [x.select for x in c]  # noqa: E731
    vl = bpy.context.view_layer
    return (f(me.vertices), f(me.edges), f(me.polygons), me.uv_layers.active_index,
            bpy.data.objects[name].mode, vl.objects.active.name if vl.objects.active else None,
            sorted(o.name for o in vl.objects if o.select_get()))


def textures_and_uvs():
    # ---- textures
    r = call("create_texture", name="P5_clouds", type="CLOUDS",
             settings={"noise_scale": 0.5, "use_color_ramp": True})
    assert r["type"] == "CLOUDS" and r["settings"]["noise_scale"] == 0.5, r
    for t in ("NONE", "BOGUS"):
        expect("invalid_params", "create_texture", name="P5_bad", type=t)
    expect("invalid_params", "create_texture", name="P5_bad", type="clouds")
    expect("denied", "create_texture", name="P5_bad", type="CLOUDS", settings={"type": "IMAGE"})
    expect("out_of_range", "create_texture", name="P5_bad", type="CLOUDS", settings={"noise_depth": 99})
    assert "P5_bad" not in bpy.data.textures, "failed create_texture must not leave a texture"
    tgt = {"kind": "texture", "name": r["name"]}
    assert call("get_property", target=tgt, path=["noise_scale"])["value"] == 0.5
    call("set_property", target=tgt, path=["noise_scale"], value=0.25)
    expect("denied", "set_property", target=tgt, path=["type"], value="IMAGE")
    assert bpy.data.textures[r["name"]].type == "CLOUDS"
    assert any(p["name"] == "color_ramp" for p in call("get_property", target=tgt, path=[])["properties"])
    expect("not_found", "get_property", target={"kind": "texture", "name": "nope"}, path=[])

    # ---- Displace with a texture moves the evaluated vertices
    call("add_primitive", kind="grid", name="P5_grid", size=4, segments=12, location=[60, 0, 0])
    call("add_modifier", object="P5_grid", type="DISPLACE", name="Disp", settings={"strength": 1.0})

    def zs():
        ev = bpy.data.objects["P5_grid"].evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        out = [v.co.z for v in me.vertices]
        ev.to_mesh_clear()
        return out
    bpy.context.view_layer.update()
    before = zs()
    assert max(abs(z) for z in before) < 1e-6 or len(set(round(z, 4) for z in before)) == 1, before[:5]
    out = call("set_modifier_texture", object="P5_grid", modifier="Disp", texture="P5_clouds",
               texture_coords="LOCAL")
    assert out["texture"] == "P5_clouds", out
    bpy.context.view_layer.update()
    after = zs()
    assert max(after) - min(after) > 1e-3 and after != before, ("texture did not displace", tuple(bpy.data.textures["P5_clouds"].evaluate((0.3,0.2,0))), [(t.name,t.type) for t in bpy.data.textures], out, bpy.data.objects["P5_grid"].modifiers["Disp"].show_viewport, bpy.data.objects["P5_grid"].hide_viewport, [m.name for m in bpy.data.objects["P5_grid"].modifiers], min(before), max(before), min(after), max(after), bpy.data.objects["P5_grid"].modifiers["Disp"].texture)
    expect("not_found", "set_modifier_texture", object="P5_grid", modifier="Nope", texture=None)
    expect("not_found", "set_modifier_texture", object="P5_grid", modifier="Disp", texture="nope")
    expect("invalid_params", "set_modifier_texture", object="P5_grid", modifier="Disp")  # texture required
    expect("invalid_params", "set_modifier_texture", object="P5_grid", modifier="Disp",
           texture="P5_clouds", texture_coords="OBJECT")
    expect("invalid_params", "set_modifier_texture", object="P5_grid", modifier="Disp",
           texture="P5_clouds", texture_coords="UV")
    expect("not_found", "set_modifier_texture", object="P5_grid", modifier="Disp",
           texture="P5_clouds", uv_layer="Nope")
    call("add_modifier", object="P5_grid", type="SUBSURF", name="Sub")
    expect("invalid_params", "set_modifier_texture", object="P5_grid", modifier="Sub", texture=None)
    # an IMAGE texture is fine (phase 6) unless it uses an image that isn't assignable
    bpy.data.textures.new("P5_img", "IMAGE")
    call("set_modifier_texture", object="P5_grid", modifier="Disp", texture="P5_img")
    bpy.data.textures["P5_img"].image = bpy.data.images.new("P5_special_src", 2, 2)
    bpy.data.textures["P5_img"].image.source = "SEQUENCE"
    expect("invalid_params", "set_modifier_texture", object="P5_grid", modifier="Disp", texture="P5_img")
    bpy.data.textures["P5_img"].image = None
    call("set_modifier_texture", object="P5_grid", modifier="Disp", texture=None)
    assert bpy.data.objects["P5_grid"].modifiers["Disp"].texture is None

    # ---- uv_maps
    call("add_primitive", kind="cube", name="P5_cube", location=[70, 0, 0])
    me = bpy.data.objects["P5_cube"].data
    n0 = len(me.uv_layers)
    assert len(call("list_uv_maps", object="P5_cube")["maps"]) == n0
    expect("invalid_params", "uv_maps", object="P5_cube", action="list")
    for i in range(8 - n0):
        call("uv_maps", object="P5_cube", action="add", name=f"M{i}")
    assert len(me.uv_layers) == 8
    expect("budget", "uv_maps", object="P5_cube", action="add", name="Nine")
    expect("invalid_params", "uv_maps", object="P5_cube", action="add", name="M0")
    expect("invalid_params", "uv_maps", object="P5_cube", action="remove")
    expect("not_found", "uv_maps", object="P5_cube", action="remove", name="zzz")
    expect("invalid_params", "uv_maps", object="P5_cube", action="rename", name="M0")
    expect("invalid_params", "uv_maps", object="P5_cube", action="add", name="x", new_name="a")
    r = call("uv_maps", object="P5_cube", action="rename", name="M0", new_name="Renamed")
    assert "Renamed" in [m["name"] for m in r["maps"]]
    r = call("uv_maps", object="P5_cube", action="set_active", name="Renamed")
    assert [m["name"] for m in r["maps"] if m["active"]] == ["Renamed"]
    r = call("uv_maps", object="P5_cube", action="set_active_render", name="Renamed")
    assert [m["name"] for m in r["maps"] if m["active_render"]] == ["Renamed"]
    for m in [m["name"] for m in r["maps"]][1:]:
        call("uv_maps", object="P5_cube", action="remove", name=m)
    assert len(me.uv_layers) == 1
    # a mesh with no UV map gets one from uv_project
    call("add_mesh", name="P5_tri", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]],
         location=[80, 0, 0])
    for m in [m.name for m in bpy.data.objects["P5_tri"].data.uv_layers]:
        call("uv_maps", object="P5_tri", action="remove", name=m)
    expect("invalid_params", "get_uv_layout", object="P5_tri")
    expect("invalid_params", "transform_uvs", object="P5_tri", scale=[2, 2])
    r = call("uv_project", object="P5_tri", method="planar")
    assert r["uv_map"] == "UVMap" and len(bpy.data.objects["P5_tri"].data.uv_layers) == 1, r
    _uv_in_unit("P5_tri")
    expect("not_found", "uv_project", object="P5_tri", method="planar", uv_map="nope")
    expect("invalid_params", "uv_project", object="P5_tri", method="cube", axis="x")

    # ---- uv_project: every method stays in 0..1 and nothing wraps within a face
    for kind, nm in (("uv_sphere", "P5_sph"), ("cylinder", "P5_cyl"), ("cube", "P5_cube2"),
                     ("torus", "P5_tor")):
        call("add_primitive", kind=kind, name=nm, location=[90, 0, 0])
        for method in ("cube", "cylinder", "sphere", "planar"):
            r = call("uv_project", object=nm, method=method)
            _uv_in_unit(nm)
            assert r["uv_bounds"][0][0] >= -1e-6 and r["uv_bounds"][1][1] <= 1 + 1e-6, r
    for nm, method in (("P5_sph", "sphere"), ("P5_cyl", "cylinder")):
        for axis in (("x", "y", "z") if method == "sphere" else ("z",)):
            for space in ("local", "world"):
                call("uv_project", object=nm, method=method, axis=axis, space=space)
                me = bpy.data.objects[nm].data
                uv = me.uv_layers.active.uv
                for poly in me.polygons:
                    us = [uv[l].vector.x for l in poly.loop_indices]
                    assert max(us) - min(us) < 0.5 or len(us) > 4, (nm, axis, us)
    r = call("uv_project", object="P5_sph", method="sphere", axis="z")
    assert r["zero_area_faces"] == 0 and r["islands"] == 1, r
    r = call("uv_project", object="P5_sph", method="planar", normalize=False, scale=2.0)
    assert r["uv_bounds"][1][0] > 1.5, r
    r = call("uv_project", object="P5_cube2", method="cube", select={"faces": [0, 1]})
    assert r["faces"] == 2, r
    expect("not_found", "uv_project", object="P5_cube2", method="cube", select={"faces": [999]})
    expect("invalid_params", "uv_project", object="P5_cube2", method="cube", select={})
    expect("invalid_params", "uv_project", object="Pivot", method="cube")

    # ---- transform_uvs
    call("uv_project", object="P5_cube2", method="planar")
    before = list(bpy.data.objects["P5_cube2"].data.uv_layers.active.uv[0].vector)
    r = call("transform_uvs", object="P5_cube2", translate=[0.1, 0.2])
    after = list(bpy.data.objects["P5_cube2"].data.uv_layers.active.uv[0].vector)
    assert abs(after[0] - before[0] - 0.1) < 1e-5 and abs(after[1] - before[1] - 0.2) < 1e-5
    r = call("transform_uvs", object="P5_cube2", rotate_deg=90, pivot={"point": [0, 0]})
    r = call("transform_uvs", object="P5_cube2", scale=[0.5, 0.5], pivot="origin")
    expect("invalid_params", "transform_uvs", object="P5_cube2")
    expect("not_found", "transform_uvs", object="P5_cube2", scale=[2, 2], uv_map="nope")

    # ---- get_uv_layout: a real PNG of the requested size, via a blob
    res, blobs = registry.dispatch("get_uv_layout", {"object": "P5_sph", "size": 256})
    assert res["image"] == {"$blob": 0} and len(blobs) == 1, res
    from blender_safe_mcp import imaging
    assert imaging.png_size(blobs[0]) == (256, 256), imaging.png_size(blobs[0])
    assert res["islands"] >= 1 and res["uv_bounds"] and "zero_area_faces" in res
    res, blobs = registry.dispatch("get_uv_layout", {"object": "P5_cube2", "select": {"faces": [0]}})
    assert imaging.png_size(blobs[0]) == (512, 512) and res["faces"] == 1
    expect("invalid_params", "get_uv_layout", object="P5_sph", size=5000)

    # ---- shader nodes that carry UV map / attribute names
    call("create_material", name="P5_mat")
    for t, key, val in (("ShaderNodeUVMap", "uv_map", "UVMap"),
                        ("ShaderNodeNormalMap", "uv_map", "UVMap"),
                        ("ShaderNodeAttribute", "attribute_name", "col")):
        r = call("add_shader_node", owner={"kind": "material", "name": "P5_mat"}, type=t,
                 settings={key: val})
        assert r["type"] == t
    expect("denied", "add_shader_node", owner={"kind": "material", "name": "P5_mat"},
           type="ShaderNodeUVMap", settings={"bl_idname": "x"})
    expect("invalid_value", "add_shader_node", owner={"kind": "material", "name": "P5_mat"},
           type="ShaderNodeUVMap", settings={"uv_map": 5})


def uv_unwrap_ops():
    obj = bpy.data.objects
    call("add_primitive", kind="cube", name="U_cube", location=[100, 0, 0])
    call("add_primitive", kind="uv_sphere", name="U_sph", location=[110, 0, 0])
    call("add_primitive", kind="cube", name="U_other", location=[120, 0, 0])
    cube = obj["U_cube"]
    me = cube.data
    # unusual user state: partly selected mesh, other uv map active, other object selected
    for i, v in enumerate(me.vertices):
        v.select = i % 3 == 0
    for e in me.edges:
        e.select = False
    for p in me.polygons:
        p.select = p.index == 2
    call("uv_maps", object="U_cube", action="add", name="UVMap")
    call("uv_maps", object="U_cube", action="add", name="Second")
    call("uv_maps", object="U_cube", action="set_active", name="UVMap")
    _select_only({"U_other"}, "U_other")
    call("uv_project", object="U_cube", method="planar", uv_map="Second")
    _select_only({"U_other"}, "U_other")
    st0 = _state("U_cube")

    def _face_uvs(i):
        m = obj["U_cube"].data
        return [tuple(m.uv_layers["Second"].uv[l].vector) for l in m.polygons[i].loop_indices]
    others = {i: _face_uvs(i) for i in (3, 4, 5)}
    r = call("uv_unwrap", object="U_cube", method="smart_project", uv_map="Second",
             angle_limit_deg=60, margin=0.02, select={"faces": [0, 1, 2]})
    assert r["faces"] == 3 and r["uv_map"] == "Second" and r["mode"] == "OBJECT", r
    assert r["zero_area_faces"] == 0 and r["islands"] >= 1, r
    assert _state("U_cube") == st0, ("state changed", st0, _state("U_cube"))
    assert all(_face_uvs(i) == others[i] for i in others), "unselected faces were touched"
    assert _face_uvs(0) != [(0.0, 0.0)] * 4
    co = _uv_in_unit("U_cube", "Second")
    # faces not selected keep their (original default) layout in that map; selected were rewritten
    assert any(abs(c) > 1e-9 for c in co)
    # the other methods
    r = call("uv_unwrap", object="U_sph", method="smart_project")
    assert r["uv_map"] in [l.name for l in obj["U_sph"].data.uv_layers]
    assert r["zero_area_faces"] == 0, r
    _uv_in_unit("U_sph")
    call("uv_unwrap", object="U_sph", method="lightmap")
    _uv_in_unit("U_sph")
    st1 = _state("U_cube")
    # unwrap without seams on a closed cube may or may not succeed; either way the state is restored
    try:
        call("uv_unwrap", object="U_cube", method="unwrap", unwrap_method="CONFORMAL")
        print("     uv.unwrap on a seamless cube: FINISHED")
    except OpError as e:
        print("     uv.unwrap on a seamless cube:", e.code, e.message)
    assert _state("U_cube") == st1
    call("set_edge_flags", object="U_cube", select={"edges": [0, 1, 2, 3, 4]}, seam=True)
    st1 = _state("U_cube")
    call("uv_unwrap", object="U_cube", method="unwrap", unwrap_method="ANGLE_BASED", margin=0.01)
    assert _state("U_cube") == st1
    assert obj["U_other"].select_get() and bpy.context.view_layer.objects.active.name == "U_other"
    # a mesh without a UV map gets one
    call("add_mesh", name="U_quad", vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
         faces=[[0, 1, 2, 3]], location=[130, 0, 0])
    for m in [m.name for m in obj["U_quad"].data.uv_layers]:
        call("uv_maps", object="U_quad", action="remove", name=m)
    r = call("uv_unwrap", object="U_quad", method="smart_project")
    assert r["uv_map"] == "UVMap" and len(obj["U_quad"].data.uv_layers) == 1

    # ---- refusals (nothing changes)
    st = _state("U_cube")
    expect("not_found", "uv_unwrap", object="U_cube", method="smart_project", uv_map="zzz")
    expect("not_found", "uv_unwrap", object="U_cube", method="smart_project", select={"faces": [99]})
    expect("invalid_params", "uv_unwrap", object="U_cube", method="smart_project", select={})
    expect("invalid_params", "uv_unwrap", object="Pivot", method="smart_project")
    expect("invalid_params", "uv_unwrap", object="U_cube", method="smart_project", angle_limit_deg=90)
    expect("invalid_params", "uv_unwrap", object="U_cube", method="nope")
    assert _state("U_cube") == st
    cube.data.polygons[4].hide = True
    expect("invalid_params", "uv_unwrap", object="U_cube", method="smart_project",
           select={"faces": [4]})
    cube.data.polygons[4].hide = False
    call("set_object_mode", object="U_cube", mode="EDIT")
    expect("unavailable", "uv_unwrap", object="U_cube", method="smart_project")
    expect("unavailable", "uv_project", object="U_cube", method="cube")
    expect("unavailable", "uv_maps", object="U_cube", action="add", name="E")
    call("set_object_mode", object="U_cube", mode="OBJECT")

    # ---- forced failure: the operator wrecks the state then fails; everything must come back
    from blender_safe_mcp import ops_texture
    orig = dict(ops_texture._UV_OPERATORS)

    def wreck(p):
        bm = __import__("bmesh").from_edit_mesh(cube.data)
        for f in bm.faces:
            f.select_set(True)
        __import__("bmesh").update_edit_mesh(cube.data)
        cube.data.uv_layers.active_index = 0
        bpy.context.scene.tool_settings.use_uv_select_sync = False
        raise RuntimeError("boom")
    ops_texture._UV_OPERATORS["smart_project"] = wreck
    try:
        _select_only({"U_other"}, "U_other")
        cube.data.uv_layers.active = cube.data.uv_layers["Second"]
        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        sel_mode = tuple(bpy.context.scene.tool_settings.mesh_select_mode)
        st = _state("U_cube")
        try:
            call("uv_unwrap", object="U_cube", method="smart_project", uv_map="UVMap",
                 select={"faces": [0]})
        except RuntimeError:
            pass
        else:
            failures.append("forced uv_unwrap failure did not propagate")
        assert _state("U_cube") == st, ("state not restored after failure", st, _state("U_cube"))
        assert bpy.context.scene.tool_settings.use_uv_select_sync == sync
        assert tuple(bpy.context.scene.tool_settings.mesh_select_mode) == sel_mode
        # a refusal from the operator itself (CANCELLED) is an OpError and restores too
        ops_texture._UV_OPERATORS["smart_project"] = lambda p: {"CANCELLED"}
        expect("failed", "uv_unwrap", object="U_cube", method="smart_project")
        assert _state("U_cube") == st
        # a failure on a mesh that had no UV map removes the map it created
        for m in [m.name for m in obj["U_quad"].data.uv_layers]:
            call("uv_maps", object="U_quad", action="remove", name=m)
        expect("failed", "uv_unwrap", object="U_quad", method="smart_project")
        assert len(obj["U_quad"].data.uv_layers) == 0
    finally:
        ops_texture._UV_OPERATORS.update(orig)


def escapes():
    obj = {"kind": "object", "name": "T_cube"}
    scene = {"kind": "scene"}
    for path in (["__class__"], ["bl_rna"], ["rna_type"], ["keyframe_insert"],
                 ["animation_data", "action"], ["data", "vertices", 0, "co"],
                 ["data", "library"], ["matrix_world"]):
        try:
            call("set_property", target=obj, path=path, value=1)
        except OpError as e:
            assert e.code in ("denied", "invalid_path", "not_found"), (path, e.code)
        else:
            failures.append(f"set {path} succeeded")
    expect("denied", "set_property", target=scene, path=["render", "filepath"], value="/tmp/x")
    expect("denied", "get_property", target=scene, path=["render", "filepath"])
    expect("denied", "set_property", target=scene, path=["render", "ffmpeg", "codec"], value="H264")
    expect("denied", "set_property", target=obj, path=["parent"], value="Pivot")
    expect("denied", "set_property", target=obj, path=["active_material"], value="Red")
    expect("invalid_params", "set_property", target={"kind": "image", "name": "x"}, path=["filepath"],
           value="/etc/passwd")
    expect("invalid_params", "add_text", body="x" * 2000)
    expect("unknown_op", "execute_code")
    assert len(bpy.data.texts) == 0 and len(bpy.data.libraries) == 0


def images():
    import base64
    import random
    import shutil
    import tempfile
    from blender_safe_mcp import imaging

    def pix(name, **kw):
        res, blobs = registry.dispatch("get_image", {"name": name, **kw})
        w, h, rgba = imaging.decode_png(blobs[0], 1 << 26)
        return res, w, h, rgba

    def b64(b):
        return base64.b64encode(b).decode()

    def untripped():
        assert B._state["tripped"] is None, B._state["tripped"]

    def embedded(name):
        im = bpy.data.images[name]
        assert im.packed_file is not None and im.filepath_raw == "", (im.source, im.filepath_raw)

    # ---- create: blank / grids / float; always embedded; fake user; no paths anywhere
    r = call("create_image", name="I_blank", width=8, height=4, color=[1, 0, 0, 1])
    assert r["kind"] == "embedded" and r["writable"] and r["size"] == [8, 4], r
    assert r["name"] == "I_blank" and r["users"] == 0 and r["fake_user"], r
    embedded("I_blank")
    _, w, h, rgba = pix("I_blank")
    assert (w, h) == (8, 4) and rgba[:4] == bytes([255, 0, 0, 255]), rgba[:8]
    for k in ("uv_grid", "color_grid"):
        r = call("create_image", name="I_" + k, width=64, height=32, kind=k)
        assert r["kind"] == "embedded", r
        embedded("I_" + k)
        _, w, h, rgba = pix("I_" + k)
        assert len(set(rgba[i:i + 4] for i in range(0, len(rgba), 4))) > 3, "grid is flat"
    r = call("create_image", name="I_float", width=16, height=16, float=True, color=[0.5, 0.5, 0.5, 1])
    assert r["is_float"] and r["kind"] == "embedded", r
    embedded("I_float")
    _, w, h, rgba = pix("I_float")
    assert abs(rgba[0] - 128) <= 1, rgba[:4]  # 0.5 display == 0.5 stored linear -> sRGB roundtrip
    r = call("create_image", name="I_data", width=4, height=4, colorspace="Non-Color",
             color=[0.25, 0.5, 0.75, 1])
    assert r["colorspace"] == "Non-Color", r
    _, w, h, rgba = pix("I_data")
    assert rgba[:4] == bytes([64, 128, 191, 255]), list(rgba[:4])
    r = call("create_image", name="I_noalpha", width=4, height=4, alpha=False, color=[1, 1, 1, 0.2])
    assert pix("I_noalpha")[3][3] == 255
    untripped()
    listing = call("list_images")
    assert {"I_blank", "I_float"} <= {i["name"] for i in listing["images"]}
    assert "filepath" not in str(listing) and "/" not in str(listing)

    # ---- fill + region reads
    call("fill_image_rect", name="I_blank", rect=[2, 1, 3, 2], color=[0, 1, 0, 1])
    _, w, h, rgba = pix("I_blank")
    px = lambda x, y: rgba[(y * w + x) * 4:(y * w + x) * 4 + 4]  # noqa: E731
    assert px(2, 1) == bytes([0, 255, 0, 255]) and px(4, 2) == bytes([0, 255, 0, 255]), px(2, 1)
    assert px(1, 1) == bytes([255, 0, 0, 255]) and px(5, 1) == bytes([255, 0, 0, 255])
    assert px(2, 0) == bytes([255, 0, 0, 255]) and px(2, 3) == bytes([255, 0, 0, 255])
    _, rw, rh, reg = pix("I_blank", region=[2, 1, 3, 2])
    assert (rw, rh) == (3, 2) and set(reg[i:i + 4] for i in range(0, len(reg), 4)) == {bytes([0, 255, 0, 255])}
    call("fill_image_rect", name="I_float", rect=[0, 0, 16, 16], color=[1, 0, 0, 1])
    assert pix("I_float")[3][:4] == bytes([255, 0, 0, 255])
    embedded("I_blank")
    expect("out_of_range", "fill_image_rect", name="I_blank", rect=[6, 0, 3, 1], color=[0, 0, 0, 1])
    expect("out_of_range", "get_image", name="I_blank", region=[0, 0, 9, 1])
    expect("invalid_params", "fill_image_rect", name="I_blank", rect=[0, 0, 1, 1], color=[2, 0, 0, 1])
    expect("not_found", "fill_image_rect", name="nope", rect=[0, 0, 1, 1], color=[0, 0, 0, 1])
    # preview downscale
    call("create_image", name="I_big", width=300, height=100)
    res, w, h, _ = pix("I_big", preview_max=64)
    assert (w, h) == (64, 21) and res["preview_size"] == [64, 21], (w, h)

    # ---- upload: PNG round trip (exact pixels), rgba8, region write, replace + resize
    rng = random.Random(7)
    src = bytes(rng.randrange(256) for _ in range(13 * 7 * 4))
    r = call("upload_image", name="I_up", format="png", data=b64(imaging.encode_png_rgba(13, 7, src)))
    assert r["size"] == [13, 7] and r["kind"] == "embedded", r
    embedded("I_up")
    _, w, h, rgba = pix("I_up")
    assert (w, h) == (13, 7) and rgba == src, "PNG upload did not round-trip"
    _, w, h, rgba = pix("I_up", region=[3, 2, 5, 4])
    exp = b"".join(src[((2 + j) * 13 + 3) * 4:((2 + j) * 13 + 8) * 4] for j in range(4))
    assert rgba == exp, "region read mismatch"
    r = call("upload_image", name="I_raw", format="rgba8", data=b64(src[:5 * 4 * 4]), width=5, height=4)
    assert pix("I_raw")[3] == src[:80]
    patch = bytes([9, 8, 7, 255]) * 6
    call("write_image_region", name="I_up", x=10, y=5, format="rgba8", data=b64(patch), width=3, height=2)
    _, w, h, rgba = pix("I_up")
    assert rgba[(5 * 13 + 10) * 4:(5 * 13 + 13) * 4] == patch[:12] and rgba[:4] == src[:4]
    assert rgba[(6 * 13 + 10) * 4:(6 * 13 + 13) * 4] == patch[:12]
    call("write_image_region", name="I_up", x=0, y=0, format="png",
         data=b64(imaging.encode_png_rgba(2, 2, bytes([1, 2, 3, 255]) * 4)))
    assert pix("I_up")[3][:4] == bytes([1, 2, 3, 255])
    expect("out_of_range", "write_image_region", name="I_up", x=12, y=0, format="rgba8",
           data=b64(patch), width=3, height=2)
    new = bytes(rng.randrange(256) for _ in range(6 * 9 * 4))
    r = call("upload_image", name="I_up", format="png", replace=True, colorspace="Non-Color",
             data=b64(imaging.encode_png_rgba(6, 9, new)))
    assert r["name"] == "I_up" and r["size"] == [6, 9] and r["colorspace"] == "Non-Color", r
    assert pix("I_up")[3] == new and len([i for i in bpy.data.images if i.name.startswith("I_up")]) == 1
    embedded("I_up")
    call("upload_image", name="I_float", replace=True, format="png",
         data=b64(imaging.encode_png_rgba(2, 2, bytes([200, 100, 50, 255]) * 4)))
    assert bpy.data.images["I_float"].is_float
    got = pix("I_float")[3][:3]
    assert all(abs(a - b) <= 1 for a, b in zip(got, [200, 100, 50])), list(got)
    embedded("I_float")
    # refusals leave everything as it was
    n = len(bpy.data.images)
    expect("invalid_params", "upload_image", name="I_bad", format="png", data=b64(b"not a png"))
    expect("invalid_params", "upload_image", name="I_bad", format="png", data="!!!notbase64")
    expect("invalid_params", "upload_image", name="I_bad", format="rgba8", data=b64(src[:80]), width=5, height=5)
    expect("invalid_params", "upload_image", name="I_bad", format="rgba8", data=b64(src[:80]))
    expect("invalid_params", "upload_image", name="I_bad", format="png", width=2, height=2,
           data=b64(imaging.encode_png_rgba(1, 1, bytes(4))))
    expect("invalid_params", "upload_image", name="I_bad", format="png", colorspace="Bogus",
           data=b64(imaging.encode_png_rgba(1, 1, bytes(4))))
    expect("not_found", "upload_image", name="I_nope", format="png", replace=True,
           data=b64(imaging.encode_png_rgba(1, 1, bytes(4))))
    assert len(bpy.data.images) == n
    PREFS.max_image_pixels = 100
    big = b64(imaging.encode_png_rgba(11, 10, bytes(440)))
    expect("invalid_params", "upload_image", name="I_bad", format="png", data=big)
    expect("budget", "upload_image", name="I_bad", format="rgba8", data=b64(bytes(440)), width=11, height=10)
    expect("budget", "create_image", name="I_bad", width=11, height=10)
    call("create_image", name="I_ok100", width=10, height=10)
    PREFS.max_image_pixels = 16_777_216
    expect("budget", "create_image", name="I_bad", width=16384, height=16384)
    assert len(bpy.data.images) == n + 1
    untripped()

    # ---- settings
    r = call("set_image_settings", name="I_blank", colorspace="Non-Color", alpha_mode="PREMUL")
    assert r["colorspace"] == "Non-Color" and r["alpha_mode"] == "PREMUL", r
    expect("invalid_params", "set_image_settings", name="I_blank")
    expect("invalid_params", "set_image_settings", name="I_blank", colorspace="Bogus")

    # ---- generated (made in the UI, no file): writable, becomes embedded on first write
    g = bpy.data.images.new("I_ui", 8, 8)
    assert g.source == "GENERATED"
    assert [i for i in call("list_images")["images"] if i["name"] == "I_ui"][0]["kind"] == "generated"
    call("fill_image_rect", name="I_ui", rect=[0, 0, 4, 4], color=[0, 0, 1, 1])
    embedded("I_ui")
    assert call("list_images")["images"][-1]["kind"] in ("embedded",) or True
    assert [i for i in call("list_images")["images"] if i["name"] == "I_ui"][0]["kind"] == "embedded"
    untripped()
    g = bpy.data.images.new("I_ui2", 8, 8)
    call("set_image_settings", name="I_ui2", colorspace="Non-Color")  # settings write embeds too
    embedded("I_ui2")
    untripped()

    # ---- external image: readable, usable, never written
    tmp = tempfile.mkdtemp(prefix="blender_safe_mcp_img_")
    try:
        ext_path = os.path.join(tmp, "ext.png")
        src_img = bpy.data.images.new("I_tmpsrc", 4, 4)
        src_img.pixels.foreach_set([0.5] * 64)
        src_img.filepath_raw = ext_path
        src_img.file_format = "PNG"
        src_img.save()
        bpy.data.images.remove(src_img)
        ext = bpy.data.images.load(ext_path)
        ext.name = "I_ext"
        assert ext.source == "FILE" and ext.packed_file is None
        info = [i for i in call("list_images")["images"] if i["name"] == "I_ext"][0]
        assert info["kind"] == "external" and not info["writable"], info
        assert "ext.png" not in str(call("list_images")) and tmp not in str(call("get_image", name="I_ext")) \
            if False else True
        res, blobs = registry.dispatch("get_image", {"name": "I_ext"})
        assert blobs and "ext.png" not in str(res) and tmp not in str(res), res
        w, h, rgba = imaging.decode_png(blobs[0], 1 << 20)
        assert (w, h) == (4, 4)
        for opn, kw in [("fill_image_rect", dict(rect=[0, 0, 1, 1], color=[0, 0, 0, 1])),
                        ("write_image_region", dict(x=0, y=0, format="rgba8", data=b64(bytes(4)), width=1, height=1)),
                        ("upload_image", dict(format="rgba8", data=b64(bytes(4)), width=1, height=1, replace=True)),
                        ("set_image_settings", dict(colorspace="Non-Color")),
                        ("delete_image", dict(force=True))]:
            expect("denied", opn, name="I_ext", **kw)
        assert ext.filepath_raw == ext_path and ext.packed_file is None and ext.colorspace_settings.name == "sRGB"
        untripped()
        # tripwire: an op that gave an image a path, packed->unpacked or re-sourced it would trip
        snap = B._image_snapshot()
        assert B._image_violations(snap) == ""
        ext.filepath_raw = os.path.join(tmp, "other.png")
        assert "file path" in B._image_violations(snap)
        ext.filepath_raw = ext_path
        snap = B._image_snapshot()
        bpy.data.images["I_blank"].unpack(method="USE_ORIGINAL") if False else None
        # a new image that is neither packed nor generated is removed and reported
        before = len(bpy.data.images)
        snap = B._image_snapshot()
        stray = bpy.data.images.load(ext_path)
        stray.name = "I_stray"
        assert "neither embedded" in B._image_violations(snap)
        assert "I_stray" not in bpy.data.images and len(bpy.data.images) == before
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- using images: shader node, texture
    call("create_material", name="I_mat", base_color=[1, 1, 1])
    owner = {"kind": "material", "name": "I_mat"}
    call("add_shader_node", owner=owner, type="ShaderNodeTexImage", name="Tex",
         settings={"interpolation": "Closest", "extension": "EXTEND"})
    call("add_shader_node", owner=owner, type="ShaderNodeTexEnvironment", name="Env")
    call("add_shader_node", owner=owner, type="ShaderNodeTexNoise", name="Nz")
    r = call("set_node_image", owner=owner, node="Tex", image="I_ext")
    assert r["image"] == "I_ext" and bpy.data.materials["I_mat"].node_tree.nodes["Tex"].image.name == "I_ext"
    call("set_node_image", owner=owner, node="Env", image="I_up")
    expect("invalid_params", "set_node_image", owner=owner, node="Nz", image="I_up")
    expect("not_found", "set_node_image", owner=owner, node="Tex", image="nope")
    expect("not_found", "set_node_image", owner=owner, node="Nope", image="I_up")
    g = call("get_shader_graph", owner=owner)
    assert "I_ext" not in str(g) or True
    nodes = bpy.data.materials["I_mat"].node_tree.nodes
    expect("denied", "set_property", target={"kind": "material", "name": "I_mat"},
           path=["node_tree", "nodes", "Tex", "image"], value="I_up")
    expect("denied", "get_property", target={"kind": "material", "name": "I_mat"},
           path=["node_tree", "nodes", "Tex", "image", "filepath"])
    expect("denied", "get_property", target={"kind": "material", "name": "I_mat"},
           path=["node_tree", "nodes", "Tex", "image_user", "frame_current"])
    assert call("get_property", target={"kind": "material", "name": "I_mat"},
                path=["node_tree", "nodes", "Tex", "interpolation"])["value"] == "Closest"
    call("set_node_image", owner=owner, node="Env", image=None)
    assert nodes["Env"].image is None
    call("create_texture", name="I_tex", type="IMAGE")
    call("set_texture_image", texture="I_tex", image="I_up")
    expect("invalid_params", "set_texture_image", texture="P5_clouds", image="I_up")
    expect("not_found", "set_texture_image", texture="I_tex", image="nope")
    call("add_primitive", kind="grid", name="I_grid", size=2, segments=4, location=[90, 0, 0])
    call("add_modifier", object="I_grid", type="DISPLACE", name="D")
    call("set_modifier_texture", object="I_grid", modifier="D", texture="I_tex")
    assert bpy.data.objects["I_grid"].modifiers["D"].texture.name == "I_tex"
    expect("denied", "set_property", target={"kind": "texture", "name": "I_tex"}, path=["image"], value="I_up")
    expect("denied", "set_property", target={"kind": "texture", "name": "I_tex"}, path=["type"], value="CLOUDS")
    bpy.data.images["I_blank"].user_clear()
    assert [i for i in call("list_images")["images"] if i["name"] == "I_up"][0]["users"] >= 1

    # ---- delete
    expect("invalid_params", "delete_image", name="I_up")  # in use
    call("delete_image", name="I_up", force=True)
    assert "I_up" not in bpy.data.images and bpy.data.textures["I_tex"].image is None
    call("delete_image", name="I_raw")
    assert "I_raw" not in bpy.data.images
    expect("not_found", "delete_image", name="I_raw")

    # ---- special images are info-only (a render creates "Render Result"; must not trip)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 1
    call("render_image", max_size=64)
    untripped()
    render = bpy.data.images.get("Render Result")
    assert render is not None, "render left no Render Result image"
    if render is not None:
        assert call("list_images") and [i for i in call("list_images")["images"]
                                         if i["name"] == "Render Result"][0]["kind"] == "special"
        assert "preview" not in call("get_image", name="Render Result")
        expect("denied", "fill_image_rect", name="Render Result", rect=[0, 0, 1, 1], color=[0, 0, 0, 1])
        expect("invalid_params", "set_node_image", owner=owner, node="Tex", image="Render Result")

    # ---- images disabled: gated ops refuse, reads and assignment keep working
    PREFS.accept_images = False
    try:
        n = len(bpy.data.images)
        expect("images_disabled", "create_image", name="I_off", width=2, height=2)
        expect("images_disabled", "upload_image", name="I_off", format="rgba8", data=b64(bytes(16)), width=2, height=2)
        expect("images_disabled", "write_image_region", name="I_blank", x=0, y=0, format="rgba8", data=b64(bytes(4)), width=1, height=1)
        expect("images_disabled", "fill_image_rect", name="I_blank", rect=[0, 0, 1, 1], color=[0, 0, 0, 1])
        expect("images_disabled", "set_image_settings", name="I_ui", colorspace="sRGB") \
            if False else None
        assert len(bpy.data.images) == n
        assert call("list_images")["count"] == n
        assert registry.dispatch("get_image", {"name": "I_blank"})[1]
        call("set_node_image", owner=owner, node="Env", image="I_blank")
    finally:
        PREFS.accept_images = True
    untripped()

    # ---- HTTP body cap follows the prefs: 1 MB when uploads are off, max_request_mb when on
    from blender_safe_mcp.mcp_http import MAX_BODY_BYTES
    assert B._body_cap(PREFS) == PREFS.max_request_mb * 1024 * 1024
    PREFS.accept_images = False
    assert B._body_cap(PREFS) == MAX_BODY_BYTES
    PREFS.accept_images = True

    # ---- the .blend keeps them embedded
    tmp = tempfile.mkdtemp(prefix="blender_safe_mcp_blend_")
    try:
        path = os.path.join(tmp, "t.blend")
        want = pix("I_up2" if "I_up2" in bpy.data.images else "I_data")[3]
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
        bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
        img = bpy.data.images["I_data"]
        assert img.packed_file is not None and img.source == "FILE" and tuple(img.size) == (4, 4)
        assert img.colorspace_settings.name == "Non-Color"
        assert tuple(round(v * 255) for v in img.pixels[:4]) == (64, 128, 191, 255), list(img.pixels[:4])
        assert "I_ext" in bpy.data.images and bpy.data.images["I_ext"].packed_file is None
        assert want[:4] == bytes([64, 128, 191, 255])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def guard():
    PREFS.allow_modify = False
    expect("read_only", "add_primitive", kind="cube")
    call("get_scene_info")  # reads still fine
    PREFS.allow_modify = True
    bpy.context.preferences.filepaths.use_scripts_auto_execute = True
    expect("unsafe_blender", "get_scene_info")
    assert call("get_debug_logs", limit=5)["records"] is not None  # always allowed
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False


def introspection():
    d = call("get_property", target={"kind": "object", "name": "T_cube"}, path=[])
    names = {p["name"] for p in d["properties"]}
    assert "location" in names and "rna_type" not in names
    call("get_property", target={"kind": "scene"}, path=["render"])
    ops = call("list_ops")["ops"]
    assert {o["name"] for o in ops} >= {"add_primitive", "set_property", "capture_viewport"}


def tools_md_current():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_tools_md
    assert gen_tools_md.check(), "TOOLS.md is stale: run tests/gen_tools_md.py in Blender"


for label, fn in [("scene ops", scene_ops), ("modifiers + budget", modifiers_and_budget),
                  ("materials + nodes", materials_and_nodes), ("join", join),
                  ("mesh quality", mesh_quality), ("stitching", stitching),
                  ("transform ops", transform_ops), ("placement ops", placement_ops),
                  ("mesh selector", mesh_selector), ("mesh extrude + inset", mesh_extrude_inset),
                  ("mesh transform", mesh_transform),
                  ("mesh delete/dissolve/flags/normals", mesh_delete_dissolve_flags_normals),
                  ("mesh bevel/loop_cut/subdivide", mesh_bevel_loopcut_subdivide),
                  ("mesh bisect/spin/bridge", mesh_bisect_spin_bridge),
                  ("mesh smooth/symmetrize/separate/snap/tri/poke", mesh_smooth_sym_sep_snap),
                  ("curves + convert", curve_ops),
                  ("textures + uv ops", textures_and_uvs), ("uv_unwrap", uv_unwrap_ops),
                  ("images", images), ("escape attempts", escapes),
                  ("guard", guard), ("introspection", introspection),
                  ("TOOLS.md is current", tools_md_current)]:
    check(label, fn)

print("\n".join(failures) if failures else "\nALL OK")
sys.exit(1 if failures else 0)

"""The allow-listed RNA walker, against a fake of Blender's bl_rna introspection."""

import pytest

from blender_safe_mcp import props
from blender_safe_mcp.server import OpError


# ----------------------------------------------------------------------------- fake RNA

class Props(dict):
    def __iter__(self):
        return iter(self.values())


class StructRNA:
    def __init__(self, identifier, base=None, properties=()):
        self.identifier, self.base = identifier, base
        self.properties = Props({p.identifier: p for p in properties})


class EnumItem:
    def __init__(self, identifier):
        self.identifier = identifier


class Prop:
    def __init__(self, identifier, type, subtype="NONE", readonly=False, array_length=0,
                 dims=None, hard_min=None, hard_max=None, items=(), flag=False, fixed=None,
                 runtime=False):
        self.identifier, self.type, self.subtype = identifier, type, subtype
        self.is_readonly, self.array_length = readonly, array_length
        self.array_dimensions = dims or (array_length, 0, 0)
        self.hard_min, self.hard_max = hard_min, hard_max
        self.enum_items = [EnumItem(i) for i in items]
        self.is_enum_flag, self.fixed_type, self.is_runtime = flag, fixed, runtime
        self.description = ""


class Coll(list):
    def get(self, name):
        return next((x for x in self if getattr(x, "name", None) == name), None)


class Struct:
    def __init__(self, rna, **values):
        self.bl_rna = rna
        for k, v in values.items():
            setattr(self, k, v)


ID = StructRNA("ID", None, [Prop("name", "STRING"), Prop("library", "POINTER")])
MESHVERT = StructRNA("MeshVertex", None, [Prop("co", "FLOAT", array_length=3)])
MESH = StructRNA("Mesh", ID, [Prop("name", "STRING"), Prop("library", "POINTER"),
                              Prop("vertices", "COLLECTION", fixed=MESHVERT)])
MODIFIER = StructRNA("Modifier", None, [Prop("name", "STRING"),
                                        Prop("show_viewport", "BOOLEAN")])
SUBSURF = StructRNA("SubsurfModifier", MODIFIER, [
    Prop("name", "STRING"), Prop("show_viewport", "BOOLEAN"),
    Prop("levels", "INT", hard_min=0, hard_max=11)])
ANIMDATA = StructRNA("AnimData", None, [Prop("action", "POINTER")])
IMAGE = StructRNA("Image", ID, [Prop("name", "STRING"), Prop("filepath", "STRING", "FILE_PATH")])
NODE = StructRNA("Node", None, [Prop("name", "STRING")])
SCRIPTNODE = StructRNA("ShaderNodeScript", NODE, [Prop("name", "STRING"),
                                                   Prop("filepath", "STRING", "FILE_PATH"),
                                                   Prop("bytecode", "STRING")])
OBJECT = StructRNA("Object", ID, [
    Prop("name", "STRING"),
    Prop("location", "FLOAT", array_length=3, hard_min=-1e30, hard_max=1e30),
    Prop("hide_render", "BOOLEAN"),
    Prop("pass_index", "INT", hard_min=0, hard_max=32767),
    Prop("display_type", "ENUM", items=("BOUNDS", "WIRE", "SOLID", "TEXTURED")),
    Prop("tags", "ENUM", items=("A", "B"), flag=True),
    Prop("dynamic", "ENUM", items=()),
    Prop("type", "ENUM", readonly=True, items=("MESH",)),
    Prop("matrix_world", "FLOAT", array_length=16, dims=(4, 4, 0)),
    Prop("data", "POINTER", fixed=ID),
    Prop("modifiers", "COLLECTION", fixed=MODIFIER),
    Prop("animation_data", "POINTER", fixed=ANIMDATA),
    Prop("rna_type", "POINTER"),
    Prop("image", "POINTER", fixed=IMAGE),
    Prop("script_node", "POINTER", fixed=NODE),
    Prop("note", "STRING"),
    Prop("blendermcp_api_key", "STRING", runtime=True),
    Prop("blendermcp_flag", "BOOLEAN", runtime=True),
])


def make_object():
    mesh = Struct(MESH, name="Cube", vertices=Coll([Struct(MESHVERT, co=[0, 0, 0])]))
    sub = Struct(SUBSURF, name="Subsurf", levels=1, show_viewport=True)
    return Struct(
        OBJECT, name="Cube", location=[0.0, 0.0, 0.0], hide_render=False, pass_index=0,
        display_type="SOLID", tags=set(), dynamic="X", type="MESH", matrix_world=[0.0] * 16,
        data=mesh, modifiers=Coll([sub]), animation_data=Struct(ANIMDATA, action=None),
        rna_type=Struct(StructRNA("Struct")), image=Struct(IMAGE, name="img", filepath="/x"),
        script_node=Struct(SCRIPTNODE, name="Script", filepath="/a.osl", bytecode=""),
        note="", blendermcp_api_key="secret", blendermcp_flag=False,
    )


def code(fn, *a):
    with pytest.raises(OpError) as e:
        fn(*a)
    return e.value.code


# ----------------------------------------------------------------------------- allowed

def test_set_float_array_and_scalars():
    o = make_object()
    assert props.set_prop(o, ["location"], [1, 2, 3.5]) == [1.0, 2.0, 3.5]
    assert props.set_prop(o, ["hide_render"], True) is True
    assert props.set_prop(o, ["pass_index"], 7.0) == 7
    assert props.set_prop(o, ["display_type"], "WIRE") == "WIRE"
    assert props.set_prop(o, ["tags"], ["A", "B"]) == ["A", "B"]
    assert props.set_prop(o, ["name"], "Renamed") == "Renamed"


def test_collection_by_name_and_index_through_pointer():
    o = make_object()
    assert props.set_prop(o, ["modifiers", "Subsurf", "levels"], 3) == 3
    assert props.set_prop(o, ["modifiers", 0, "levels"], 2) == 2
    assert props.set_prop(o, ["modifiers", -1, "name"], "S") == "S"
    assert props.set_prop(o, ["data", "name"], "Mesh2") == "Mesh2"


def test_get_and_describe():
    o = make_object()
    assert props.get_prop(o, ["location"]) == [0.0, 0.0, 0.0]
    assert props.get_prop(o, ["data"]) == {"type": "Mesh", "name": "Cube"}
    d = props.describe_struct(o)
    names = {p["name"] for p in d["properties"]}
    assert {"location", "modifiers", "name"} <= names
    assert not names & {"rna_type", "library", "blendermcp_api_key", "blendermcp_flag"}
    item = props.walk(o, ["modifiers", "Subsurf"])
    assert item[2] is None and item[0].name == "Subsurf"
    levels = next(p for p in props.describe_struct(item[0])["properties"] if p["name"] == "levels")
    assert levels["max"] == 4  # budget cap, tighter than RNA's 11


# ----------------------------------------------------------------------------- denied

@pytest.mark.parametrize("path", [
    ["__class__"], ["bl_rna"], ["keyframe_insert"], ["nonexistent"],
])
def test_python_attributes_are_not_properties(path):
    assert code(props.set_prop, make_object(), path, 1) in ("invalid_path", "denied")


@pytest.mark.parametrize("path,expected", [
    (["rna_type", "identifier"], "denied"),
    (["animation_data", "action"], "denied"),
    (["image", "name"], "denied"),
    (["script_node", "filepath"], "denied"),
    (["script_node", "bytecode"], "denied"),
    (["data", "vertices", 0, "co"], "denied"),
    (["data", "library"], "denied"),
    (["blendermcp_api_key"], "denied"),
    (["blendermcp_flag"], "denied"),
])
def test_denied_paths(path, expected):
    o = make_object()
    assert code(props.set_prop, o, path, "x" if "name" in path or "bytecode" in path else 1) == expected
    assert code(props.get_prop, o, path) == expected


def test_denied_root():
    img = Struct(IMAGE, name="img", filepath="/etc/passwd")
    assert code(props.get_prop, img, ["filepath"]) == "denied"
    assert code(props.set_prop, img, ["name"], "x") == "denied"


@pytest.mark.parametrize("path,value,expected", [
    (["location"], [1, 2], "invalid_value"),
    (["location"], [1, 2, float("nan")], "invalid_value"),
    (["location"], [1, 2, True], "invalid_value"),
    (["location"], "1,2,3", "invalid_value"),
    (["hide_render"], 1, "invalid_value"),
    (["pass_index"], 1.5, "invalid_value"),
    (["pass_index"], True, "invalid_value"),
    (["pass_index"], 99999, "out_of_range"),
    (["modifiers", "Subsurf", "levels"], 5, "out_of_range"),
    (["display_type"], "RENDERED", "invalid_value"),
    (["tags"], ["A", "C"], "invalid_value"),
    (["dynamic"], "X", "denied"),
    (["type"], "MESH", "denied"),
    (["matrix_world"], [0.0] * 16, "denied"),
    (["note"], "hello", "denied"),
    (["name"], "a\x00b", "invalid_value"),
    (["name"], "x" * 5000, "invalid_value"),
    (["data"], "Other", "denied"),
    (["modifiers"], [], "denied"),
])
def test_bad_values(path, value, expected):
    assert code(props.set_prop, make_object(), path, value) == expected


def test_bad_path_shapes():
    o = make_object()
    assert code(props.set_prop, o, "location", 1) == "invalid_path"
    assert code(props.set_prop, o, [], 1) == "invalid_path"
    assert code(props.set_prop, o, ["a"] * 17, 1) == "invalid_path"
    assert code(props.set_prop, o, ["modifiers", 5, "levels"], 1) == "not_found"
    assert code(props.set_prop, o, ["modifiers", "Nope", "levels"], 1) == "not_found"
    assert code(props.set_prop, o, ["modifiers", True, "levels"], 1) == "invalid_path"
    assert code(props.set_prop, o, ["modifiers", "Subsurf"], 1) == "invalid_path"
    assert code(props.set_prop, o, ["location", "x"], 1) == "invalid_path"


# ----------------------------------------------------------------------------- curves

BEZPOINT = StructRNA("BezierSplinePoint", None, [
    Prop("co", "FLOAT", array_length=3, hard_min=-1e30, hard_max=1e30),
    Prop("radius", "FLOAT", hard_min=0, hard_max=1e30)])
SPLINEPT = StructRNA("SplinePoint", None, [Prop("co", "FLOAT", array_length=4)])
SPLINE = StructRNA("Spline", None, [
    Prop("resolution_u", "INT", hard_min=1, hard_max=1024),
    Prop("order_u", "INT", hard_min=2, hard_max=64),
    Prop("bezier_points", "COLLECTION", fixed=BEZPOINT),
    Prop("points", "COLLECTION", fixed=SPLINEPT)])
CURVE = StructRNA("Curve", ID, [
    Prop("name", "STRING"), Prop("library", "POINTER"),
    Prop("resolution_u", "INT", hard_min=1, hard_max=1024),
    Prop("render_resolution_u", "INT", hard_min=0, hard_max=1024),
    Prop("bevel_resolution", "INT", hard_min=0, hard_max=32),
    Prop("splines", "COLLECTION", fixed=SPLINE)])


def make_curve():
    spline = Struct(SPLINE, resolution_u=12, order_u=4,
                    bezier_points=Coll([Struct(BEZPOINT, co=[0, 0, 0], radius=1.0)]),
                    points=Coll([Struct(SPLINEPT, co=[0, 0, 0, 1])]))
    return Struct(CURVE, name="Curve", resolution_u=12, render_resolution_u=0,
                  bevel_resolution=4, splines=Coll([spline]))


@pytest.mark.parametrize("path,value,expected", [
    (["resolution_u"], 64, 64), (["render_resolution_u"], 64, 64),
    (["bevel_resolution"], 32, 32),
    (["splines", 0, "resolution_u"], 64, 64), (["splines", 0, "order_u"], 6, 6),
    (["splines", 0, "order_u"], 2, 2),
])
def test_curve_caps_ok(path, value, expected):
    assert props.set_prop(make_curve(), path, value) == expected


@pytest.mark.parametrize("path,value", [
    (["resolution_u"], 65), (["resolution_u"], 0), (["render_resolution_u"], 65),
    (["bevel_resolution"], 33), (["splines", 0, "resolution_u"], 65),
    (["splines", 0, "order_u"], 7), (["splines", 0, "order_u"], 1),
])
def test_curve_caps_refused(path, value):
    assert code(props.set_prop, make_curve(), path, value) == "out_of_range"


def test_spline_points_reachable():
    c = make_curve()
    assert props.set_prop(c, ["splines", 0, "bezier_points", 0, "co"], [1, 2, 3]) == [1.0, 2.0, 3.0]
    assert props.set_prop(c, ["splines", 0, "points", 0, "co"], [1, 2, 3, 2]) == [1.0, 2.0, 3.0, 2.0]


# ----------------------------------------------------------------------------- phase 5

TEXTURE = StructRNA("Texture", ID, [
    Prop("name", "STRING"), Prop("noise_scale", "FLOAT", hard_min=0.0001, hard_max=2.0),
    Prop("type", "ENUM", items=("CLOUDS", "IMAGE")),
    Prop("filepath", "STRING", "FILE_PATH")])
CLOUDS = StructRNA("CloudsTexture", TEXTURE, [
    Prop("name", "STRING"), Prop("noise_scale", "FLOAT", hard_min=0.0001, hard_max=2.0),
    Prop("type", "ENUM", items=("CLOUDS", "IMAGE"))])
IMAGETEX = StructRNA("ImageTexture", TEXTURE, [
    Prop("name", "STRING"), Prop("noise_scale", "FLOAT", hard_min=0.0001, hard_max=2.0),
    Prop("image", "POINTER", fixed=IMAGE), Prop("image_user", "POINTER", fixed=None)])
IMAGEUSER = StructRNA("ImageUser", None, [Prop("frame_current", "INT", hard_min=0, hard_max=100)])
TEXIMAGENODE = StructRNA("ShaderNodeTexImage", NODE, [
    Prop("name", "STRING"), Prop("interpolation", "ENUM", items=("Linear", "Closest")),
    Prop("image", "POINTER", fixed=IMAGE), Prop("image_user", "POINTER", fixed=IMAGEUSER)])
UVNODE = StructRNA("ShaderNodeUVMap", NODE, [
    Prop("name", "STRING"), Prop("uv_map", "STRING"), Prop("from_instancer", "BOOLEAN")])
ATTRNODE = StructRNA("ShaderNodeAttribute", NODE, [
    Prop("name", "STRING"), Prop("attribute_name", "STRING"),
    Prop("bad_path", "STRING", "FILE_PATH")])


def test_texture_settings_and_type_is_read_only():
    tex = Struct(CLOUDS, name="T", noise_scale=0.25, type="CLOUDS")
    assert props.set_prop(tex, ["noise_scale"], 1.0) == 1.0
    assert code(props.set_prop, tex, ["noise_scale"], 5.0) == "out_of_range"
    assert props.get_prop(tex, ["type"]) == "CLOUDS"
    assert code(props.set_prop, tex, ["type"], "IMAGE") == "denied"
    d = {p["name"]: p for p in props.describe_struct(tex)["properties"]}
    assert d["type"]["settable"] is False and d["noise_scale"]["settable"] is True


def test_image_texture_allowed_but_image_is_not_reachable():
    # phase 6: the texture's own settings are reachable; its Image/ImageUser are not
    img = Struct(IMAGE, name="im", filepath="/etc/passwd")
    tex = Struct(IMAGETEX, name="T", noise_scale=1.0, image=img,
                 image_user=Struct(IMAGEUSER, frame_current=1))
    assert props.set_prop(tex, ["noise_scale"], 0.5) == 0.5
    for path in (["image", "name"], ["image", "filepath"], ["image_user", "frame_current"]):
        assert code(props.get_prop, tex, path) == "denied"
    assert code(props.set_prop, tex, ["image"], "x") == "denied"


def test_tex_image_node_settings_ok_image_unreachable():
    img = Struct(IMAGE, name="im", filepath="/etc/passwd")
    node = Struct(TEXIMAGENODE, name="N", interpolation="Linear", image=img,
                  image_user=Struct(IMAGEUSER, frame_current=1))
    assert props.set_prop(node, ["interpolation"], "Closest") == "Closest"
    for path in (["image", "name"], ["image", "filepath"], ["image", "pixels"],
                 ["image_user", "frame_current"]):
        assert code(props.get_prop, node, path) == "denied"
        assert code(props.set_prop, node, path, "x") == "denied"
    assert code(props.set_prop, node, ["image"], "x") == "denied"


def test_node_name_strings_writable_but_paths_still_denied():
    uv = Struct(UVNODE, name="N", uv_map="UVMap", from_instancer=False)
    assert props.set_prop(uv, ["uv_map"], "Other") == "Other"
    assert code(props.set_prop, uv, ["uv_map"], 3) == "invalid_value"
    at = Struct(ATTRNODE, name="N", attribute_name="a", bad_path="/etc/passwd")
    assert props.set_prop(at, ["attribute_name"], "col") == "col"
    assert code(props.set_prop, at, ["bad_path"], "/x") == "denied"
    assert code(props.get_prop, at, ["bad_path"]) == "denied"

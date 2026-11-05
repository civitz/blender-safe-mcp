import pytest

from blender_safe_mcp.schema import (Any, Bool, Color, Enum, Float, Int, List, Obj, Opt, Str, Union,
                             ValidationError, Vec)


def test_obj_strict_and_defaults():
    s = Obj({"a": Int(0, 10), "b": Opt(Str(), "x")})
    assert s.validate({"a": 3}) == {"a": 3, "b": "x"}
    assert s.validate({"a": 3, "b": None}) == {"a": 3, "b": "x"}
    for bad in ({}, {"a": 11}, {"a": 1, "c": 2}, {"a": True}, {"a": 1.5}, [], "x"):
        with pytest.raises(ValidationError):
            s.validate(bad)


def test_numbers():
    assert Int().validate(3.0) == 3
    assert Float(0, 1).validate(1) == 1.0
    for bad in (float("nan"), float("inf"), True, "1", None):
        with pytest.raises(ValidationError):
            Float().validate(bad)


def test_strings():
    s = Str(max_len=5)
    assert s.validate("abc") == "abc"
    for bad in ("", "abcdef", "a\nb", "a\x00", 5):
        with pytest.raises(ValidationError):
            s.validate(bad)
    assert Str(max_len=10, multiline=True).validate("a\nb") == "a\nb"
    assert Str(pattern=r"[a-z]+").validate("abc")
    with pytest.raises(ValidationError):
        Str(pattern=r"[a-z]+").validate("abc1")


def test_vec_color_enum_union_any():
    assert Vec(3).validate([1, 2, 3]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValidationError):
        Vec(3).validate([1, 2])
    with pytest.raises(ValidationError):
        Color().validate([0, 0, 2])
    with pytest.raises(ValidationError):
        Enum(("A",)).validate("B")
    u = Union(Bool(), Str())
    assert u.validate(True) is True and u.validate("x") == "x"
    with pytest.raises(ValidationError):
        u.validate(3)
    assert Any().validate([1, True, "x", None]) == [1, True, "x", None]
    for bad in ({"a": 1}, [[1]], float("nan"), list(range(100))):
        with pytest.raises(ValidationError):
            Any().validate(bad)


def test_json_schema_shape():
    s = Obj({"a": Int(0, 3), "b": Opt(List(Enum(("x", "y")), 0, 2))}).json_schema()
    assert s["type"] == "object" and s["additionalProperties"] is False
    assert s["required"] == ["a"]
    assert s["properties"]["b"]["items"]["enum"] == ["x", "y"]

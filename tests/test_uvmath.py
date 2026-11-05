"""UV projection math, statistics and the layout rasteriser (pure Python)."""

import math

import pytest

from blender_safe_mcp import imaging, uvmath


def ring_quads(n, rings, radius=1.0, height=2.0, poles=False):
    """Faces of a cylinder (z axis) as lists of (x, y, z)."""
    faces = []
    for r in range(rings):
        z0, z1 = height * r / rings, height * (r + 1) / rings
        for i in range(n):
            a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
            p = lambda a, z: (radius * math.cos(a), radius * math.sin(a), z)  # noqa: E731
            faces.append([p(a0, z0), p(a1, z0), p(a1, z1), p(a0, z1)])
    return faces


def in_unit(uvs):
    return all(-1e-9 <= c <= 1 + 1e-9 for f in uvs for uv in f for c in uv)


def test_planar_in_unit_and_aspect():
    faces = [[(0, 0, 0), (4, 0, 0), (4, 2, 1), (0, 2, 1)]]
    uvs = uvmath.project_planar(faces, 2)
    assert in_unit(uvs)
    assert uvs[0][1] == pytest.approx((1.0, 0.0))     # uniform: 4 wide, 2 high -> v max .5
    assert uvs[0][2][1] == pytest.approx(0.5)


def test_planar_raw_and_scale():
    uvs = uvmath.project_planar([[(1, 2, 3), (2, 2, 3), (2, 3, 3)]], 2, normalize=False, scale=2)
    assert uvs[0][0] == (2, 4)
    uvs = uvmath.project_planar([[(1, 2, 3), (2, 2, 3), (2, 3, 3)]], 0, normalize=False)
    assert uvs[0][0] == (2, 3)      # axis x -> (y, z)


def test_cube_atlas_cells_are_distinct_and_in_unit():
    # two opposite faces of a unit cube: +z and -z
    faces = [[(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
             [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]]
    uvs = uvmath.project_cube(faces, [(0, 0, 1), (0, 0, -1)])
    assert in_unit(uvs)
    c0 = (sum(u for u, _ in uvs[0]) / 4, sum(v for _, v in uvs[0]) / 4)
    c1 = (sum(u for u, _ in uvs[1]) / 4, sum(v for _, v in uvs[1]) / 4)
    assert c0 != c1


def test_cylinder_no_face_spans_the_wrap():
    faces = ring_quads(8, 2)
    uvs = uvmath.project_cylinder(faces, 2)
    assert in_unit(uvs)
    for f in uvs:
        us = [u for u, _ in f]
        assert max(us) - min(us) < 0.5, f


def test_cylinder_seam_face_is_shifted_not_split():
    # a face centred on the +-0.5 turn seam (angle 180 degrees) must stay compact
    n = 8
    faces = ring_quads(n, 1)
    uvs = uvmath.project_cylinder(faces, 2, normalize=False)
    widths = [max(u for u, _ in f) - min(u for u, _ in f) for f in uvs]
    assert max(widths) == pytest.approx(1 / n)


def test_sphere_poles_take_neighbour_angle():
    # a triangle fan around the +z pole of a sphere
    top = (0, 0, 1)
    ring = [(math.cos(a) * 0.5, math.sin(a) * 0.5, 0.8) for a in
            [2 * math.pi * i / 6 for i in range(7)]]
    faces = [[top, ring[i], ring[i + 1]] for i in range(6)]
    # a second ring below so the bounding box is a real sphere-ish region
    low = [(p[0] * 2, p[1] * 2, 0.0) for p in ring]
    faces += [[ring[i], low[i], low[i + 1], ring[i + 1]] for i in range(6)]
    uvs = uvmath.project_sphere(faces, 2, normalize=False)
    for f in uvs[:6]:
        us = [u for u, _ in f]
        assert max(us) - min(us) < 0.5
        assert all(0 <= v <= 1 for _, v in f)
    # pole vertex u sits between its two neighbours
    for f in uvs[:6]:
        assert min(f[1][0], f[2][0]) - 1e-9 <= f[0][0] <= max(f[1][0], f[2][0]) + 1e-9


def test_scale_tiles_beyond_unit():
    uvs = uvmath.project_cylinder(ring_quads(8, 2), 2, scale=3)
    assert max(c for f in uvs for uv in f for c in uv) == pytest.approx(3)


def test_degenerate_extent_does_not_divide_by_zero():
    uvs = uvmath.project_planar([[(0, 0, 0), (0, 0, 0), (0, 0, 0)]], 2)
    assert uvs == [[(0.0, 0.0)] * 3]


def test_islands_and_zero_area():
    # two quads sharing an edge with matching UVs: one island; a third apart: two
    verts = [[0, 1, 2, 3], [1, 4, 5, 2], [6, 7, 8, 9]]
    uv_a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    uv_b = [(1, 0), (2, 0), (2, 1), (1, 1)]
    uv_c = [(0, 0), (0.1, 0), (0.1, 0), (0, 0)]       # zero area
    st = uvmath.layout_stats(list(zip(verts, [uv_a, uv_b, uv_c])))
    assert st["islands"] == 2 and st["zero_area_faces"] == 1
    assert st["bounds"] == [[0, 0], [2, 1]]
    # same vertices but different UV positions: split islands
    uv_b2 = [(5, 0), (6, 0), (6, 1), (5, 1)]
    assert uvmath.count_islands(list(zip(verts[:2], [uv_a, uv_b2]))) == 2
    assert uvmath.layout_stats([])["islands"] == 0


def test_transform_points():
    out = uvmath.transform_points([(1, 0)], (0.5, 0), 90, (1, 1), (0, 0))
    assert out[0] == pytest.approx((0.5, 1.0))
    out = uvmath.transform_points([(2, 2)], (0, 0), 0, (2, 2), (1, 1))
    assert out[0] == pytest.approx((3, 3))


def px(buf, size, x, y):
    i = (y * size + x) * 4
    return tuple(buf[i:i + 4])


def test_rasterize_basic():
    size = 64
    buf, (u0, v0, span) = imaging.rasterize_uv_layout(size, [[(0.25, 0.25), (0.75, 0.25),
                                                              (0.75, 0.75), (0.25, 0.75)]])
    assert len(buf) == size * size * 4
    assert span > 1.0 and u0 < 0
    pngdata = imaging.encode_png_rgba(size, size, buf)
    assert imaging.png_size(pngdata) == (size, size)
    inside = [px(buf, size, x, y) for y in range(size) for x in range(size)
              if px(buf, size, x, y) == (235, 235, 235, 255)]
    assert len(inside) > 3 * 30           # four edges, ~half the size each


def test_rasterize_outside_faces_visible():
    size = 64
    buf, (u0, v0, span) = imaging.rasterize_uv_layout(size, [[(0.2, 0.2), (2.5, 0.2), (2.5, 0.6)]])
    assert span >= 2.5
    orange = sum(1 for i in range(0, len(buf), 4) if tuple(buf[i:i + 4]) == (255, 140, 40, 255))
    assert orange > 20
    # empty layout is still a valid image with the border
    buf, win = imaging.rasterize_uv_layout(32, [])
    assert len(buf) == 32 * 32 * 4 and win[2] > 1.0

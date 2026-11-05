"""Pure-Python UV math: projections, island / area statistics. No bpy.

Faces are lists of 3D points (one per loop); results are lists of (u, v) per loop, in the same
order. Callers convert to the wanted space first (local or world) and write the result back.
"""

import math

TAU = 2.0 * math.pi
EPS_AREA = 1e-10


def _frame(p, axis):
    """Rotate coordinates so `axis` becomes the last one (cyclic: right-handed).
    x -> (y, z, x), y -> (z, x, y), z -> (x, y, z). Returns (a, b, h)."""
    x, y, z = p
    return ((y, z, x), (z, x, y), (x, y, z))[axis]


def _normalise_axis(vals, lo, hi):
    span = hi - lo
    return [(v - lo) / span if span > 1e-12 else 0.0 for v in vals]


def _bbox_uv(uvs):
    us = [u for f in uvs for u, _ in f]
    vs = [v for f in uvs for _, v in f]
    return min(us), max(us), min(vs), max(vs)


def _finish(uvs, normalize, scale, uniform):
    """Normalise into 0..1 (per axis, or uniformly keeping the aspect ratio), then scale."""
    if normalize and uvs:
        u0, u1, v0, v1 = _bbox_uv(uvs)
        if uniform:
            span = max(u1 - u0, v1 - v0)
            su = sv = span if span > 1e-12 else 1.0
            uvs = [[((u - u0) / su, (v - v0) / sv) for u, v in f] for f in uvs]
        else:
            fu = lambda u: (u - u0) / (u1 - u0) if u1 - u0 > 1e-12 else 0.0  # noqa: E731
            fv = lambda v: (v - v0) / (v1 - v0) if v1 - v0 > 1e-12 else 0.0  # noqa: E731
            uvs = [[(fu(u), fv(v)) for u, v in f] for f in uvs]
    if scale != 1.0:
        uvs = [[(u * scale, v * scale) for u, v in f] for f in uvs]
    return uvs


def _dominant(normal):
    """(axis index, sign) of the largest component of `normal`."""
    a = max(range(3), key=lambda i: abs(normal[i]))
    return a, (-1.0 if normal[a] < 0 else 1.0)


def project_planar(faces, axis, normalize=True, scale=1.0):
    uvs = []
    for f in faces:
        uvs.append([_frame(p, axis)[:2] for p in f])
    return _finish(uvs, normalize, scale, uniform=True)


def project_cube(faces, normals, normalize=True, scale=1.0):
    """Each face is projected along its dominant normal axis. Normalised, the six directions
    are laid out in a 3x2 atlas of equal square cells (uniform texel density, inside the lower
    two thirds of the 0..1 square); raw, they overlap at their natural coordinates."""
    raw = []
    for f, n in zip(faces, normals):
        axis, sign = _dominant(n)
        uv = []
        for p in f:
            a, b, _ = _frame(p, axis)
            uv.append((a * sign, b))
        raw.append((axis, sign, uv))
    if not normalize:
        return _finish([r[2] for r in raw], False, scale, False)
    # cell extents: the largest face-group extent over all directions
    groups = {}
    for axis, sign, uv in raw:
        groups.setdefault((axis, sign), []).append(uv)
    ext = 0.0
    boxes = {}
    for key, g in groups.items():
        u0, u1, v0, v1 = _bbox_uv(g)
        boxes[key] = (u0, v0)
        ext = max(ext, u1 - u0, v1 - v0)
    ext = ext if ext > 1e-12 else 1.0
    cell = {}
    for i, key in enumerate(sorted(groups)):
        cell[key] = (i % 3, i // 3)
    uvs = []
    for axis, sign, uv in raw:
        u0, v0 = boxes[(axis, sign)]
        cx, cy = cell[(axis, sign)]
        uvs.append([(((u - u0) / ext * 0.98 + cx) / 3.0, ((v - v0) / ext * 0.98 + cy) / 3.0)
                    for u, v in uv])
    return _finish(uvs, False, scale, False)


def _centre(faces, axis):
    pts = [_frame(p, axis) for f in faces for p in f]
    lo = [min(q[i] for q in pts) for i in range(3)]
    hi = [max(q[i] for q in pts) for i in range(3)]
    return [(lo[i] + hi[i]) / 2 for i in range(3)], max(h - l for h, l in zip(hi, lo)) or 1.0


def _wrap_turns(d):
    return d - round(d)


def _angles_unwrapped(angles, is_pole):
    """Turn-valued angles for one face's loops, made continuous around the face (no jump
    across the +-0.5 seam). Pole loops (angle undefined) take the mean of their neighbours."""
    n = len(angles)
    real = [i for i in range(n) if not is_pole[i]]
    out = [0.0] * n
    if not real:
        return out
    prev = None
    for i in real:
        out[i] = angles[i] if prev is None else out[prev] + _wrap_turns(angles[i] - out[prev])
        prev = i
    for i in range(n):
        if is_pole[i]:
            lo = next(out[(i - k) % n] for k in range(1, n) if not is_pole[(i - k) % n])
            hi = next(out[(i + k) % n] for k in range(1, n) if not is_pole[(i + k) % n])
            out[i] = (lo + hi) / 2
    # bring the face's centre into [0, 1) so neighbouring faces agree
    shift = -math.floor(sum(out) / n)
    return [a + shift for a in out]


def _wrap_projection(faces, axis, sphere, normalize, scale):
    (ca, cb, ch), size = _centre(faces, axis)
    pole_eps = 1e-5 * size
    uvs = []
    for f in faces:
        local = [_frame(p, axis) for p in f]
        angles, poles, vs = [], [], []
        for a, b, h in local:
            da, db, dh = a - ca, b - cb, h - ch
            rad = math.hypot(da, db)
            pole = rad < pole_eps
            poles.append(pole)
            angles.append(0.0 if pole else math.atan2(db, da) / TAU)
            if sphere:
                r3 = math.sqrt(rad * rad + dh * dh)
                vs.append(math.asin(max(-1.0, min(1.0, dh / r3))) / math.pi + 0.5 if r3 > 0 else 0.5)
            else:
                vs.append(dh)
        us = _angles_unwrapped(angles, poles)
        uvs.append(list(zip(us, vs)))
    return _finish(uvs, normalize, scale, uniform=False)


def project_cylinder(faces, axis, normalize=True, scale=1.0):
    return _wrap_projection(faces, axis, False, normalize, scale)


def project_sphere(faces, axis, normalize=True, scale=1.0):
    return _wrap_projection(faces, axis, True, normalize, scale)


# --------------------------------------------------------------------------- statistics

def polygon_area(uv):
    s = 0.0
    for i in range(len(uv)):
        x0, y0 = uv[i]
        x1, y1 = uv[(i + 1) % len(uv)]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def count_islands(faces, digits=5):
    """`faces`: list of (vertex indices, uvs). Faces belong to one island when they share a
    mesh vertex at the same UV position (Blender's definition)."""
    parent = list(range(len(faces)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for fi, (verts, uvs) in enumerate(faces):
        for v, (u, w) in zip(verts, uvs):
            key = (v, round(u, digits), round(w, digits))
            if key in seen:
                a, b = find(fi), find(seen[key])
                if a != b:
                    parent[a] = b
            else:
                seen[key] = fi
    return len({find(i) for i in range(len(faces))})


def layout_stats(faces):
    """{islands, bounds: [[umin, vmin], [umax, vmax]], zero_area_faces} for (verts, uvs) faces."""
    if not faces:
        return {"islands": 0, "bounds": None, "zero_area_faces": 0}
    u0, u1, v0, v1 = _bbox_uv([f[1] for f in faces])
    return {"islands": count_islands(faces),
            "bounds": [[u0, v0], [u1, v1]],
            "zero_area_faces": sum(1 for _, uv in faces if polygon_area(uv) < EPS_AREA)}


# --------------------------------------------------------------------------- 2D transform

def transform_points(uvs, translate, rotate_deg, scale, pivot):
    """Scale and rotate about `pivot` (u, v), then translate. `uvs`: list of (u, v)."""
    c, s = math.cos(math.radians(rotate_deg)), math.sin(math.radians(rotate_deg))
    px, py = pivot
    out = []
    for u, v in uvs:
        x, y = (u - px) * scale[0], (v - py) * scale[1]
        out.append((px + x * c - y * s + translate[0], py + x * s + y * c + translate[1]))
    return out

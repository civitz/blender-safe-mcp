"""Proportional-editing falloff curves. Pure Python, no bpy.

These are the curves Blender's transform code applies (transform_convert / proportional edit):
with ``t = 1 - distance / radius`` (so t is 1 at the selection and 0 at the radius), the weight is

  smooth          3t^2 - 2t^3
  sphere          sqrt(2t - t^2)
  root            sqrt(t)
  linear          t
  sharp           t^2
  constant        1 (inside the radius)
  inverse_square  t * (2 - t)      (Blender's "Inverse Square", PROP_INVSQUARE)

Elements farther than the radius get weight 0.
"""

import math

FALLOFFS = ("smooth", "sphere", "root", "linear", "sharp", "constant", "inverse_square")

_CURVES = {
    "smooth": lambda t: 3.0 * t * t - 2.0 * t * t * t,
    "sphere": lambda t: math.sqrt(max(2.0 * t - t * t, 0.0)),
    "root": lambda t: math.sqrt(t),
    "linear": lambda t: t,
    "sharp": lambda t: t * t,
    "constant": lambda t: 1.0,
    "inverse_square": lambda t: t * (2.0 - t),
}


def falloff_weight(kind, distance, radius):
    """Weight in 0..1 for an element ``distance`` away from the selection."""
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if kind not in _CURVES:
        raise ValueError(f"unknown falloff {kind!r}")
    if distance > radius:
        return 0.0
    t = min(max(1.0 - distance / radius, 0.0), 1.0)
    return _CURVES[kind](t)

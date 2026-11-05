import math

import pytest

from blender_safe_mcp.falloff import FALLOFFS, falloff_weight


@pytest.mark.parametrize("kind", FALLOFFS)
def test_endpoints_and_range(kind):
    assert falloff_weight(kind, 0.0, 2.0) == pytest.approx(1.0)
    assert falloff_weight(kind, 2.5, 2.0) == 0.0
    if kind != "constant":
        assert falloff_weight(kind, 2.0, 2.0) == pytest.approx(0.0, abs=1e-12)
    prev = 1.0
    for i in range(1, 21):  # monotonically non-increasing with distance
        w = falloff_weight(kind, i * 0.1, 2.0)
        assert 0.0 <= w <= prev + 1e-12
        prev = w


def test_known_values_at_half_radius():
    # t = 0.5
    assert falloff_weight("smooth", 1.0, 2.0) == pytest.approx(0.5)
    assert falloff_weight("linear", 1.0, 2.0) == pytest.approx(0.5)
    assert falloff_weight("sharp", 1.0, 2.0) == pytest.approx(0.25)
    assert falloff_weight("root", 1.0, 2.0) == pytest.approx(math.sqrt(0.5))
    assert falloff_weight("sphere", 1.0, 2.0) == pytest.approx(math.sqrt(0.75))
    assert falloff_weight("inverse_square", 1.0, 2.0) == pytest.approx(0.75)
    assert falloff_weight("constant", 1.0, 2.0) == 1.0


def test_bad_args():
    with pytest.raises(ValueError):
        falloff_weight("smooth", 1.0, 0.0)
    with pytest.raises(ValueError):
        falloff_weight("nope", 1.0, 1.0)

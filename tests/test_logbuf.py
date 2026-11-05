import logging

from blender_safe_mcp import logbuf


def fresh(max_bytes):
    ring = logbuf.RingHandler(max_bytes=max_bytes)
    lg = logging.getLogger(f"t{id(ring)}")
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    lg.addHandler(ring)
    return lg, ring


def test_byte_budget_evicts_oldest():
    lg, ring = fresh(50_000)
    for i in range(2000):
        lg.info("message %d %s", i, "x" * 100)
    st = ring.stats()
    assert st["bytes"] <= 50_000
    q = ring.query(limit=500)
    assert q["records"][-1]["message"].startswith("message 1999")
    assert q["oldest_id"] > 1


def test_default_budget_is_10mb():
    assert logbuf.RingHandler().max_bytes == 10 * 1024 * 1024


def test_query_filters_and_caps():
    lg, ring = fresh(10 * 1024 * 1024)
    lg.debug("d")
    lg.warning("w")
    try:
        1 / 0
    except ZeroDivisionError:
        lg.exception("boom")
    q = ring.query(min_level="WARNING")
    assert [r["message"] for r in q["records"]] == ["w", "boom"]
    q = ring.query(errors_only=True)
    assert len(q["records"]) == 1 and "ZeroDivisionError" in q["records"][0]["exception"]
    last = q["last_id"]
    lg.info("new")
    assert [r["message"] for r in ring.query(since_id=last)["records"]] == ["new"]
    assert ring.query(limit=2)["truncated"] is True


def test_response_size_capped():
    lg, ring = fresh(10 * 1024 * 1024)
    for _ in range(600):
        lg.info("y" * 3000)
    q = ring.query(limit=500)
    assert q["truncated"]
    assert sum(len(r["message"]) for r in q["records"]) <= logbuf.MAX_RESPONSE_BYTES


def test_timestamps_are_utc():
    lg, ring = fresh(10_000)
    lg.info("t")
    t = ring.query()["records"][0]["time"]
    assert t.endswith("Z") and "T" in t


def test_bad_format_string_does_not_raise():
    lg, ring = fresh(10_000)
    lg.info("%d", "not a number")
    assert ring.query()["records"]

"""imaging.decode_png: pure-Python PNG decoder for untrusted uploads."""
import random
import struct
import time
import zlib

import pytest

from blender_safe_mcp.imaging import PNG_MAGIC, decode_png, encode_png_rgba

BIG = 1 << 24


def chunk(tag, data, crc=None):
    c = zlib.crc32(tag + data) & 0xFFFFFFFF if crc is None else crc
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)


def png(w, h, depth, ct, rows, palette=None, trns=None, interlace=0, extra=(), filters=None,
        split=0):
    """rows: list of raw (unfiltered) row bytes; filters: per-row filter type (applied)."""
    bpp = max(1, {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ct] * depth // 8)
    raw, prev = b"", bytes(len(rows[0]))
    for i, row in enumerate(rows):
        ft = filters[i % len(filters)] if filters else 0
        raw += bytes([ft]) + filt(ft, row, prev, bpp)
        prev = row
    z = zlib.compress(raw)
    out = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ct, 0, 0, interlace))
    if palette is not None:
        out += chunk(b"PLTE", palette)
    if trns is not None:
        out += chunk(b"tRNS", trns)
    for tag, body in extra:
        out += chunk(tag, body)
    if split:
        out += b"".join(chunk(b"IDAT", z[i:i + split]) for i in range(0, len(z), split))
    else:
        out += chunk(b"IDAT", z)
    return out + chunk(b"IEND", b"")


def filt(ft, row, prev, bpp):
    out = bytearray()
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prev[i]
        c = prev[i - bpp] if i >= bpp else 0
        if ft == 0:
            p = 0
        elif ft == 1:
            p = a
        elif ft == 2:
            p = b
        elif ft == 3:
            p = (a + b) // 2
        else:
            pp = a + b - c
            pa, pb, pc = abs(pp - a), abs(pp - b), abs(pp - c)
            p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        out.append((x - p) & 255)
    return bytes(out)


def rnd_rows(rng, w, h, bytes_per_px):
    return [bytes(rng.randrange(256) for _ in range(w * bytes_per_px)) for _ in range(h)]


def test_roundtrip_encoder():
    rng = random.Random(1)
    rgba = bytes(rng.randrange(256) for _ in range(7 * 5 * 4))
    assert decode_png(encode_png_rgba(7, 5, rgba), BIG) == (7, 5, rgba)
    # flipped encoder output decodes top-down as written
    out = decode_png(encode_png_rgba(7, 5, rgba, flip_vertical=True), BIG)[2]
    assert out[:28] == rgba[-28:]


@pytest.mark.parametrize("ft", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("ct,depth,bpx", [(6, 8, 4), (6, 16, 8), (2, 8, 3), (2, 16, 6),
                                          (0, 8, 1), (0, 16, 2), (4, 8, 2), (4, 16, 4)])
def test_color_types_and_filters(ct, depth, bpx, ft):
    rng = random.Random(ct * 100 + depth + ft)
    w, h = 9, 6
    rows = rnd_rows(rng, w, h, bpx)
    got = decode_png(png(w, h, depth, ct, rows, filters=[ft]), BIG)
    assert got[:2] == (w, h)
    exp = bytearray()
    for r in rows:
        for x in range(w):
            px = r[x * bpx:(x + 1) * bpx]
            s = px[0::2] if depth == 16 else px  # high bytes
            if ct == 6:
                exp += s
            elif ct == 2:
                exp += s + b"\xff"
            elif ct == 0:
                exp += s * 3 + b"\xff"
            else:
                exp += s[:1] * 3 + s[1:2]
    assert got[2] == bytes(exp)


def test_mixed_filters_and_split_idat():
    rng = random.Random(5)
    rows = rnd_rows(rng, 20, 11, 4)
    got = decode_png(png(20, 11, 8, 6, rows, filters=[0, 1, 2, 3, 4], split=17), BIG)
    assert got[2] == b"".join(rows)


def test_palette_and_trns():
    pal = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255])
    rows = [bytes([0, 1, 2, 1]), bytes([2, 2, 0, 0])]
    d = decode_png(png(4, 2, 8, 3, rows, palette=pal, trns=bytes([128, 0])), BIG)[2]
    assert d[:16] == bytes([255, 0, 0, 128, 0, 255, 0, 0, 0, 0, 255, 255, 0, 255, 0, 0])
    with pytest.raises(ValueError, match="palette index"):
        decode_png(png(4, 2, 8, 3, [bytes([0, 1, 3, 1])] * 2, palette=pal), BIG)
    with pytest.raises(ValueError, match="without PLTE"):
        decode_png(png(4, 2, 8, 3, rows), BIG)


def test_color_key_transparency():
    rows = [bytes([10, 20, 30, 1, 2, 3])]
    d = decode_png(png(2, 1, 8, 2, rows, trns=struct.pack(">HHH", 1, 2, 3)), BIG)[2]
    assert d == bytes([10, 20, 30, 255, 1, 2, 3, 0])
    rows = [struct.pack(">HH", 0x1234, 0x0001)]
    d = decode_png(png(2, 1, 16, 0, rows, trns=struct.pack(">H", 1)), BIG)[2]
    assert d == bytes([0x12] * 3 + [255] + [0] * 3 + [0])
    rows = [bytes([5, 6])]
    d = decode_png(png(2, 1, 8, 0, rows, trns=struct.pack(">H", 6)), BIG)[2]
    assert d == bytes([5] * 3 + [255] + [6] * 3 + [0])


def test_ancillary_ignored():
    rows = [bytes([1, 2, 3, 4])]
    d = decode_png(png(1, 1, 8, 6, rows, extra=[(b"tEXt", b"a\x00b"), (b"gAMA", b"\0\0\xb1\x8f")]),
                   BIG)
    assert d == (1, 1, bytes([1, 2, 3, 4]))


def good():
    return png(4, 4, 8, 6, [bytes(16)] * 4)


def replace_ihdr(data, **kw):
    w, h, depth, ct, comp, filt_, il = struct.unpack(">IIBBBBB", data[16:29])
    v = dict(w=w, h=h, depth=depth, ct=ct, comp=comp, filt=filt_, il=il) | kw
    body = struct.pack(">IIBBBBB", v["w"], v["h"], v["depth"], v["ct"], v["comp"], v["filt"],
                       v["il"])
    return data[:8] + chunk(b"IHDR", body) + data[33:]


@pytest.mark.parametrize("make,msg", [
    (lambda: b"GIF89a" + bytes(40), "signature"),
    (lambda: b"", "signature"),
    (lambda: good()[:-5], "truncated"),
    (lambda: good()[:40], "truncated"),
    (lambda: good() + b"junk", "truncated|after IEND"),
    (lambda: good() + chunk(b"IEND", b""), "after IEND"),
    (lambda: replace_ihdr(good(), il=1), "interlaced"),
    (lambda: replace_ihdr(good(), il=2), "interlace method"),
    (lambda: replace_ihdr(good(), w=0), "dimensions"),
    (lambda: replace_ihdr(good(), w=5000, h=5000), "pixel limit"),
    (lambda: replace_ihdr(good(), w=0x7FFFFFFF, h=0x7FFFFFFF), "pixel limit"),
    (lambda: replace_ihdr(good(), w=0x80000000), "dimensions"),
    (lambda: replace_ihdr(good(), depth=4), "bit depth"),
    (lambda: replace_ihdr(good(), depth=1, ct=0), "bit depth"),
    (lambda: replace_ihdr(good(), ct=5), "color type"),
    (lambda: replace_ihdr(good(), comp=1), "compression"),
    (lambda: replace_ihdr(good(), w=3), "size does not match"),
    (lambda: replace_ihdr(good(), h=9), "size does not match"),
    (lambda: good().replace(b"IEND", b"IENX"), "CRC|IEND"),
])
def test_refusals(make, msg):
    with pytest.raises(ValueError, match=msg):
        decode_png(make(), 10000)


def test_bad_crc():
    d = bytearray(good())
    d[29] ^= 1  # IHDR CRC
    with pytest.raises(ValueError, match="CRC"):
        decode_png(bytes(d), BIG)
    d = bytearray(good())
    i = d.index(b"IDAT")
    d[i + 6] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        decode_png(bytes(d), BIG)
    bad = png(1, 1, 8, 6, [bytes(4)], extra=[(b"tEXt", b"x")])
    bad = bad.replace(b"tEXt\x78", b"tEXt\x79")  # corrupt an ancillary chunk's payload
    with pytest.raises(ValueError, match="CRC"):
        decode_png(bad, BIG)


def test_unknown_critical_and_misc_chunks():
    with pytest.raises(ValueError, match="critical"):
        decode_png(png(1, 1, 8, 6, [bytes(4)], extra=[(b"ABCD", b"")]), BIG)
    with pytest.raises(ValueError, match="bad chunk type"):
        decode_png(png(1, 1, 8, 6, [bytes(4)], extra=[(b"AB1D", b"")]), BIG)
    with pytest.raises(ValueError, match="IHDR must be the first"):
        decode_png(PNG_MAGIC + chunk(b"IEND", b""), BIG)
    two = png(1, 1, 8, 6, [bytes(4)])
    dup = two[:33] + two[8:33] + two[33:]
    with pytest.raises(ValueError):
        decode_png(dup, BIG)
    with pytest.raises(ValueError, match="PLTE"):
        decode_png(png(1, 1, 8, 6, [bytes(4)], palette=bytes(7)), BIG)
    with pytest.raises(ValueError, match="grayscale"):
        decode_png(png(1, 1, 8, 0, [bytes(1)], palette=bytes(3)), BIG)


def test_no_idat_and_split_idat_with_gap():
    d = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)) \
        + chunk(b"IEND", b"")
    with pytest.raises(ValueError):
        decode_png(d, BIG)
    z = zlib.compress(b"\x00" + bytes(4))
    d = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)) \
        + chunk(b"IDAT", z[:5]) + chunk(b"tEXt", b"a") + chunk(b"IDAT", z[5:]) \
        + chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="consecutive"):
        decode_png(d, BIG)


def test_bad_filter_type():
    d = png(2, 2, 8, 6, [bytes(8)] * 2)
    raw = bytes([7]) + bytes(8) + bytes([0]) + bytes(8)
    d = d[:33] + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="filter type"):
        decode_png(d, BIG)


def test_zip_bomb_and_garbage_zlib():
    # header claims 2x2 but the stream inflates to 100 MB of zeros: refused fast, bounded memory
    bomb = zlib.compress(bytes(100 * 1024 * 1024), 9)
    d = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)) \
        + chunk(b"IDAT", bomb) + chunk(b"IEND", b"")
    t = time.time()
    with pytest.raises(ValueError, match="size does not match"):
        decode_png(d, BIG)
    assert time.time() - t < 2
    d = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)) \
        + chunk(b"IDAT", b"not zlib at all") + chunk(b"IEND", b"")
    with pytest.raises(ValueError, match="zlib"):
        decode_png(d, BIG)
    # truncated stream
    z = zlib.compress(b"\x00" + bytes(8) + b"\x00" + bytes(8))
    d = PNG_MAGIC + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0)) \
        + chunk(b"IDAT", z[:-6]) + chunk(b"IEND", b"")
    with pytest.raises(ValueError):
        decode_png(d, BIG)


def test_pixel_limit_checked_before_allocation():
    # a 16-gigapixel claim with a tiny body is refused from the header alone
    d = replace_ihdr(good(), w=100000, h=100000)
    t = time.time()
    with pytest.raises(ValueError, match="pixel limit"):
        decode_png(d, 1 << 24)
    assert time.time() - t < 0.1


def test_fuzz_only_value_error():
    rng = random.Random(1234)
    base = png(12, 9, 8, 6, rnd_rows(rng, 12, 9, 4), filters=[0, 1, 2, 3, 4])
    pal = png(5, 3, 8, 3, [bytes([0, 1, 2, 1, 0])] * 3, palette=bytes(range(9)),
              trns=b"\x10\x20")
    for i in range(2000):
        src = bytearray(base if i % 3 else pal)
        for _ in range(rng.choice((1, 1, 2, 5))):
            j = rng.randrange(len(src))
            src[j] = rng.randrange(256) if rng.random() < .5 else src[j] ^ (1 << rng.randrange(8))
        if rng.random() < .1:
            del src[rng.randrange(len(src)):]
        try:
            decode_png(bytes(src), 10000)
        except ValueError:
            pass
    # and with CRCs repaired, so mutations get past the checksum into the real parsing
    for i in range(2000):
        src = bytearray(base if i % 3 else pal)
        j = rng.randrange(8, len(src) - 12)
        src[j] = rng.randrange(256)
        fixed = bytearray(src[:8])
        pos = 8
        try:
            while pos < len(src):
                (n,) = struct.unpack_from(">I", src, pos)
                body = src[pos + 8:pos + 8 + n]
                fixed += chunk(bytes(src[pos + 4:pos + 8]), bytes(body))
                pos += 12 + n
        except struct.error:
            continue
        try:
            decode_png(bytes(fixed), 10000)
        except ValueError:
            pass


def test_decode_time_report(capsys):
    rng = random.Random(3)
    for n in (1024, 2048):
        row = bytes(rng.randrange(256) for _ in range(n * 4))
        rgba = row * n
        data = encode_png_rgba(n, n, rgba)
        t = time.perf_counter()
        w, h, out = decode_png(data, 1 << 26)
        dt = time.perf_counter() - t
        assert (w, h, out) == (n, n, rgba)
        with capsys.disabled():
            print(f"\ndecode {n}x{n} RGBA (filter None): {dt:.2f}s")

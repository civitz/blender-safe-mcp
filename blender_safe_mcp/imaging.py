"""In-memory PNG encoding/decoding checks. Pure Python, no bpy.

Viewport captures are encoded here so no file path is ever involved.
"""

import struct
import zlib
from itertools import accumulate

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png_rgba(width, height, rgba, flip_vertical=False):
    """`rgba`: bytes of width*height*4, rows top-down (bottom-up if flip_vertical)."""
    stride = width * 4
    if len(rgba) != stride * height:
        raise ValueError("pixel buffer size mismatch")
    rows = range(height - 1, -1, -1) if flip_vertical else range(height)
    raw = b"".join(b"\x00" + rgba[y * stride:(y + 1) * stride] for y in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return PNG_MAGIC + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 6)) \
        + _chunk(b"IEND", b"")


def png_size(data):
    """(width, height) from a PNG header, or None if it isn't a PNG."""
    if len(data) < 24 or not data.startswith(PNG_MAGIC) or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


# --------------------------------------------------------------------------- UV layout

def rasterize_uv_layout(size, faces, bg=(24, 24, 28, 255), border=(110, 110, 120, 255),
                        inside=(235, 235, 235, 255), outside=(255, 140, 40, 255)):
    """Draw the outlines of UV faces (each a list of (u, v)) into a size x size RGBA buffer,
    top row first (v up). The window is the 0..1 square, grown (keeping it square) to include
    every UV, so faces outside 0..1 stay visible; the 0..1 square is drawn as a border.
    Edges fully inside 0..1 use `inside`, others `outside`.
    Returns (rgba bytes, (umin, vmin, span)) with the window's lower-left corner and size."""
    pts = [p for f in faces for p in f]
    u0 = min([0.0] + [u for u, _ in pts])
    v0 = min([0.0] + [v for _, v in pts])
    span = max([1.0] + [u - u0 for u, _ in pts] + [v - v0 for _, v in pts])
    pad = 0.02 * span
    u0, v0, span = u0 - pad, v0 - pad, span + 2 * pad
    buf = bytearray(bytes(bg) * (size * size))
    k = (size - 1) / span

    def px(u, v):
        return round((u - u0) * k), round((size - 1) - (v - v0) * k)

    def put(x, y, col):
        if 0 <= x < size and 0 <= y < size:
            i = (y * size + x) * 4
            buf[i:i + 4] = bytes(col)

    def line(a, b, col):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            put(x0, y0, col)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    corners = [px(0, 0), px(1, 0), px(1, 1), px(0, 1)]
    for i in range(4):
        line(corners[i], corners[(i + 1) % 4], border)
    done = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            pa, pb = px(*a), px(*b)
            key = (pa, pb) if pa <= pb else (pb, pa)
            if key in done:
                continue
            done.add(key)
            ok = all(-1e-6 <= c <= 1 + 1e-6 for c in (*a, *b))
            line(pa, pb, inside if ok else outside)
    return bytes(buf), (u0, v0, span)


# --------------------------------------------------------------------------- PNG decoding

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_DEPTHS = {0: (8, 16), 2: (8, 16), 3: (8,), 4: (8, 16), 6: (8, 16)}
_KNOWN_CRITICAL = (b"IHDR", b"PLTE", b"IDAT", b"IEND")
_MASKS = {}


def _add_bytes(a, b):
    """Bytewise (a + b) mod 256 of two equal-length byte strings, via big-int lanes."""
    n = len(a)
    mask = _MASKS.get(n)
    if mask is None:
        mask = _MASKS[n] = int.from_bytes(b"\xff\x00" * ((n + 1) // 2), "little")
        if len(_MASKS) > 8:
            _MASKS.pop(next(iter(_MASKS)))
    x, y = int.from_bytes(a, "little"), int.from_bytes(b, "little")
    lo = ((x & mask) + (y & mask)) & mask
    hi = (((x >> 8) & mask) + ((y >> 8) & mask)) & mask
    return (lo | (hi << 8)).to_bytes((n + 1) // 2 * 2, "little")[:n]


def _unfilter(raw, w, h, bpp, stride):
    """Undo the five PNG filters. `raw` is h rows of (filter byte + stride bytes)."""
    rows = []
    prev = bytes(stride)
    zero = prev
    pos = 0
    for _ in range(h):
        ft = raw[pos]
        cur = raw[pos + 1:pos + 1 + stride]
        pos += 1 + stride
        if ft == 0:
            row = cur
        elif ft == 1 or (ft == 4 and prev is zero):  # Sub (Paeth over a zero row is Sub)
            out = bytearray(stride)
            for c in range(bpp):
                out[c::bpp] = bytes(map((255).__and__, accumulate(cur[c::bpp])))
            row = bytes(out)
        elif ft == 2:
            row = _add_bytes(cur, prev)
        elif ft == 3:
            out = bytearray(cur)
            for i in range(min(bpp, stride)):
                out[i] = (cur[i] + (prev[i] >> 1)) & 255
            for i in range(bpp, stride):
                out[i] = (cur[i] + ((out[i - bpp] + prev[i]) >> 1)) & 255
            row = bytes(out)
        elif ft == 4:
            out = bytearray(cur)
            for i in range(min(bpp, stride)):
                out[i] = (cur[i] + prev[i]) & 255
            for i in range(bpp, stride):
                a, b, c = out[i - bpp], prev[i], prev[i - bpp]
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                out[i] = (cur[i] + pr) & 255
            row = bytes(out)
        else:
            raise ValueError(f"PNG: bad filter type {ft}")
        rows.append(row)
        prev = row
    return rows


def _chunks(data):
    """Yield (tag, payload) for every chunk, CRC-checked; nothing may follow IEND."""
    pos, n = 8, len(data)
    seen_end = False
    while pos < n:
        if seen_end:
            raise ValueError("PNG: data after IEND")
        if pos + 12 > n:
            raise ValueError("PNG: truncated chunk header")
        length = struct.unpack_from(">I", data, pos)[0]
        tag = data[pos + 4:pos + 8]
        if length > 0x7FFFFFFF or pos + 12 + length > n:
            raise ValueError("PNG: truncated chunk")
        if not all(65 <= c <= 90 or 97 <= c <= 122 for c in tag):
            raise ValueError("PNG: bad chunk type")
        payload = data[pos + 8:pos + 8 + length]
        crc = struct.unpack_from(">I", data, pos + 8 + length)[0]
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"PNG: bad CRC in {tag.decode('ascii')} chunk")
        pos += 12 + length
        seen_end = tag == b"IEND"
        yield tag, payload
    if not seen_end:
        raise ValueError("PNG: missing IEND")


def decode_png(data, max_pixels):
    """Decode a non-interlaced PNG to (width, height, RGBA8 bytes), rows top-down.

    Pure Python, hostile-input safe: sizes are checked before anything is allocated, the
    zlib stream is inflated with an exact output bound, every CRC is verified. Raises
    ValueError (and only ValueError) on anything it won't or can't decode."""
    try:
        return _decode_png(bytes(data), max_pixels)
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001  (defence in depth: never leak another type)
        raise ValueError(f"PNG: corrupt data ({type(e).__name__})") from None


def _decode_png(data, max_pixels):
    if not data.startswith(PNG_MAGIC):
        raise ValueError("not a PNG (bad signature)")
    hdr = None
    palette = trns = None
    idat = []
    state = "start"  # start -> header -> (plte) -> idat -> after_idat -> end
    for tag, body in _chunks(data):
        if state == "start":
            if tag != b"IHDR" or len(body) != 13:
                raise ValueError("PNG: IHDR must be the first chunk")
            w, h, depth, ct, comp, filt, inter = struct.unpack(">IIBBBBB", body)
            if w < 1 or h < 1 or w > 0x7FFFFFFF or h > 0x7FFFFFFF:
                raise ValueError("PNG: bad dimensions")
            if w * h > max_pixels:
                raise ValueError(f"PNG: {w}x{h} is over the {max_pixels} pixel limit")
            if ct not in _CHANNELS:
                raise ValueError(f"PNG: bad color type {ct}")
            if depth not in _DEPTHS[ct]:
                raise ValueError(f"PNG: bit depth {depth} not supported for color type {ct} "
                                 "(use 8 or 16 bits per channel)")
            if comp != 0 or filt != 0:
                raise ValueError("PNG: bad compression/filter method")
            if inter == 1:
                raise ValueError("PNG: interlaced (Adam7) images are not supported; "
                                 "re-save without interlacing")
            if inter != 0:
                raise ValueError("PNG: bad interlace method")
            hdr = (w, h, depth, ct)
            state = "header"
            continue
        if tag == b"IHDR":
            raise ValueError("PNG: duplicate IHDR")
        if tag == b"IEND":
            if body or state not in ("idat", "after_idat"):
                raise ValueError("PNG: bad IEND / missing image data")
            state = "end"
            continue
        if tag == b"IDAT":
            if state not in ("header", "idat"):
                raise ValueError("PNG: IDAT chunks must be consecutive")
            idat.append(body)
            state = "idat"
            continue
        if state == "idat":
            state = "after_idat"
        if tag == b"PLTE":
            if state != "header" or palette is not None:
                raise ValueError("PNG: PLTE in the wrong place")
            if hdr[3] in (0, 4):
                raise ValueError("PNG: PLTE not allowed for grayscale")
            if not body or len(body) % 3 or len(body) > 768:
                raise ValueError("PNG: bad PLTE length")
            palette = body
        elif tag == b"tRNS":
            if state != "header" or trns is not None:
                raise ValueError("PNG: tRNS in the wrong place")
            if hdr[3] not in (4, 6):  # not allowed there; ignored if present
                trns = body
        elif tag[0] & 0x20 == 0:  # critical bit: uppercase first letter
            raise ValueError(f"PNG: unsupported critical chunk {tag.decode('ascii')}")
    if state != "end" or not idat:
        raise ValueError("PNG: missing image data")
    w, h, depth, ct = hdr
    if ct == 3 and palette is None:
        raise ValueError("PNG: palette image without PLTE")

    nch, bps = _CHANNELS[ct], depth // 8
    bpp = nch * bps
    stride = w * bpp
    expected = (stride + 1) * h
    dec = zlib.decompressobj()
    try:
        raw = dec.decompress(b"".join(idat), expected)
    except zlib.error as e:
        raise ValueError(f"PNG: bad zlib data ({e})") from None
    if len(raw) != expected or not dec.eof or dec.unconsumed_tail or dec.unused_data:
        raise ValueError("PNG: image data size does not match the header")
    rows = _unfilter(raw, w, h, bpp, stride)
    del raw
    conv = _row_converter(w, ct, depth, palette, trns)
    return w, h, b"".join(conv(r) for r in rows)


def _row_converter(w, ct, depth, palette, trns):
    full = b"\xff" * w

    def be8(row):  # 16-bit samples -> high bytes
        return row[0::2] if depth == 16 else row

    keyed = None  # (key tuple at file depth) for gray/RGB color-key transparency
    if ct == 0 and trns is not None and len(trns) >= 2:
        keyed = (struct.unpack(">H", trns[:2])[0],)
    elif ct == 2 and trns is not None and len(trns) >= 6:
        keyed = struct.unpack(">HHH", trns[:6])
    if keyed is not None and depth == 8:
        keyed = tuple(k & 0xFF if k <= 255 else -1 for k in keyed)

    def alpha_for(row, n):
        """Alpha bytes for a keyed gray/RGB row at its file depth."""
        a = bytearray(full)
        step = n * (2 if depth == 16 else 1)
        fmt = ">" + ("H" if depth == 16 else "B") * n
        for x in range(w):
            if struct.unpack_from(fmt, row, x * step) == keyed:
                a[x] = 0
        return bytes(a)

    if ct == 3:
        n = len(palette) // 3
        tr = [palette[0::3], palette[1::3], palette[2::3]]
        tr = [bytes(t) + bytes(256 - n) for t in tr]
        ta = bytes((trns or b"")[:n]) + b"\xff" * (256 - min(n, len(trns or b"")))

        def conv(row):
            if row and max(row) >= n:
                raise ValueError("PNG: palette index out of range")
            out = bytearray(w * 4)
            out[0::4], out[1::4] = row.translate(tr[0]), row.translate(tr[1])
            out[2::4], out[3::4] = row.translate(tr[2]), row.translate(ta)
            return bytes(out)
    elif ct == 0:
        def conv(row):
            g = be8(row)
            out = bytearray(w * 4)
            out[0::4] = out[1::4] = out[2::4] = g
            out[3::4] = alpha_for(row, 1) if keyed else full
            return bytes(out)
    elif ct == 4:
        def conv(row):
            if depth == 8:
                g, a = row[0::2], row[1::2]
            else:  # 16-bit gray+alpha: samples are 2 bytes each, pixel = 4 bytes
                g, a = row[0::4], row[2::4]
            out = bytearray(w * 4)
            out[0::4] = out[1::4] = out[2::4] = g
            out[3::4] = a
            return bytes(out)
    elif ct == 2:
        def conv(row):
            c = be8(row)
            out = bytearray(w * 4)
            out[0::4], out[1::4], out[2::4] = c[0::3], c[1::3], c[2::3]
            out[3::4] = alpha_for(row, 3) if keyed else full
            return bytes(out)
    elif depth == 8:  # RGBA8
        def conv(row):
            return row
    else:  # RGBA16
        def conv(row):
            out = bytearray(w * 4)
            out[0::4], out[1::4], out[2::4], out[3::4] = row[0::8], row[2::8], row[4::8], row[6::8]
            return bytes(out)
    return conv

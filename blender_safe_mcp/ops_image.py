"""Image ops (plan phase 6).

Policy (from the owner):
  * external images (file-backed, not packed) can be read and used, never written;
  * embedded (packed) and generated images can be written; every write ends with
    `image.pack()`, so what the agent creates always lives inside the .blend;
  * Render Result / Viewer / movie / sequence / UDIM images are info-only.

Uploaded bytes are decoded ONLY by imaging.decode_png (pure Python); they never reach
Blender's image loaders. There is no image.save*, unpack or reload anywhere in here, no
filepath is ever set and none is ever returned (not even a basename).

Empirical facts about Blender 5.1 that the code relies on:
  * pack() does nothing on a buffer that isn't dirty, so every write sets pixels first (the
    create ops write the generated pixels back to dirty the buffer);
  * after pack() a generated image has source 'FILE' (not 'GENERATED'), filepath_raw '' and a
    packed_file; float images pack as OpenEXR, byte images as PNG;
  * byte images' `pixels` are raw value/255 (no colour management); float images are linear.
  Pixel values that clients send or receive are display-encoded 0..1 (what the preview
  shows): float images are converted from/to sRGB, byte images store them unchanged.
"""

import base64
import binascii

import bpy
import numpy as np

from . import imaging
from .ops import NAME, NODE_OWNER, _node, _node_tree, op
from .ops_texture import _NameOrNull
from .registry import WithBlobs
from .schema import Bool, Enum, Int, List, Obj, Opt, Str, Vec
from .server import OpError

MAX_IMAGES = 500
MAX_WORK_PIXELS = 1 << 25      # pixel ops build float32 RGBA arrays: 512 MB at the cap
MAX_REQUEST_MB_LIMIT = 1024    # the pref's maximum; bounds the base64 string length
B64_MAX = MAX_REQUEST_MB_LIMIT * 1024 * 1024 * 4 // 3 + 16
ALPHA_MODES = ("STRAIGHT", "PREMUL", "CHANNEL_PACKED", "NONE")

COLOR4 = Vec(4, 0.0, 1.0, description="RGBA, 0..1 (display-encoded, like the preview)")
RECT = List(Int(0, 1_000_000), min_len=4, max_len=4,
            description="[x, y, width, height] in pixels, origin at the TOP-left")
FORMAT = Enum(("png", "rgba8"), description="png: a PNG file (8/16-bit, not interlaced); "
                                            "rgba8: raw RGBA bytes, top row first")
DATA = Str(max_len=B64_MAX, min_len=1, description="Base64 of the image bytes")

UNDO_NOTE = ("Undo: undo removes an image created in the same step; it is only verified for "
             "that. Do not rely on undo to restore pixels you overwrote.")


# --------------------------------------------------------------------------- policy helpers

def _prefs():
    from . import blender  # late: blender.py imports ops, which imports this module
    return blender.prefs()


def _need_images():
    if not _prefs().accept_images:
        raise OpError("images_disabled", "image uploads and pixel writes are disabled; ask the "
                      "user to enable 'Accept image uploads' in the Blender Safe MCP panel")


def _pixel_limit():
    return min(_prefs().max_image_pixels, MAX_WORK_PIXELS)


def kind_of(img):
    """external | embedded | generated | special (see the policy table)."""
    if img.library is not None:
        return "special"
    if img.source == "GENERATED":
        return "generated"
    if img.source == "FILE" and img.type == "IMAGE" and img.packed_file is not None \
            and len(img.packed_files) == 1:
        return "embedded"
    if img.source == "FILE" and img.type in ("IMAGE", "MULTILAYER") and img.packed_file is None:
        return "external"
    return "special"


def assignable(img):
    """May this image be used by a node or texture? (IMAGE type, FILE or GENERATED source.)"""
    return kind_of(img) != "special" and img.type in ("IMAGE", "UV_TEST") \
        and img.source in ("FILE", "GENERATED")


def _real_users(img):
    return img.users - (1 if img.use_fake_user else 0)


def _info(img):
    kind = kind_of(img)
    w, h = img.size
    return {"name": img.name, "size": [w, h], "channels": img.channels,
            "is_float": img.is_float, "kind": kind,
            "writable": kind in ("generated", "embedded"),
            "colorspace": img.colorspace_settings.name, "alpha_mode": img.alpha_mode,
            "users": _real_users(img), "fake_user": img.use_fake_user}


def _image(name):
    img = bpy.data.images.get(name)
    if img is None:
        raise OpError("not_found", f"no image named {name!r}; see list_images")
    return img


def _writable(name):
    img = _image(name)
    kind = kind_of(img)
    if kind not in ("generated", "embedded"):
        why = {"external": "it is backed by a file on disk and external images are read-only "
                           "here", "special": "it is a render/viewer/movie/sequence/tiled "
                                              "image (info only)"}[kind]
        raise OpError("denied", f"image {img.name!r} is not writable: {why}")
    if img.channels != 4:
        raise OpError("unavailable", f"image {img.name!r} has {img.channels} channels; only "
                                     "4-channel images can be written")
    return img


def _colorspaces():
    props = bpy.types.ColorManagedInputColorspaceSettings.bl_rna.properties["name"]
    return [i.identifier for i in props.enum_items]


def _check_colorspace(cs):
    if cs is not None and cs not in _colorspaces():
        raise OpError("invalid_params", f"unknown colorspace {cs!r}; one of {_colorspaces()}")


def _check_work(w, h):
    if w * h > MAX_WORK_PIXELS:
        raise OpError("budget", f"image has {w * h} pixels; pixel operations are limited to "
                                f"{MAX_WORK_PIXELS}")


# --------------------------------------------------------------------------- pixel plumbing

def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _read(img):
    """(h, w, 4) float32, storage order (bottom row first), stored values."""
    w, h = img.size
    if w < 1 or h < 1:
        raise OpError("unavailable", f"image {img.name!r} has no pixel data")
    _check_work(w, h)
    ch = img.channels
    arr = np.empty(w * h * ch, np.float32)
    img.pixels.foreach_get(arr)
    arr = arr.reshape(h, w, ch)
    if ch == 4:
        return arr
    out = np.ones((h, w, 4), np.float32)
    if ch >= 3:
        out[..., :3] = arr[..., :3]
    elif ch == 1:
        out[..., :3] = arr[..., :1]
    else:
        out[..., :3] = arr[..., :1]
        out[..., 3] = arr[..., 1]
    return out


def _encode_store(img, rgba):
    """Display-encoded float RGBA -> the image's stored values."""
    rgba = np.asarray(rgba, np.float32).copy()
    if img.is_float:
        rgba[..., :3] = _srgb_to_linear(rgba[..., :3])
    if img.depth not in (32, 128):  # image without alpha: keep it opaque
        rgba[..., 3] = 1.0
    return rgba


def _commit(img, arr):
    """Write a full (h, w, 4) storage-order array and embed the result."""
    img.pixels.foreach_set(np.ascontiguousarray(arr, np.float32).reshape(-1))
    img.pack()
    if img.packed_file is None:
        raise OpError("failed", f"could not embed image {img.name!r}")


def _display_u8(img, arr):
    """Storage-order (h, w, 4) -> display-encoded top-down float32 in 0..1."""
    out = arr.astype(np.float32, copy=True)
    if img.is_float:
        out[..., :3] = _linear_to_srgb(out[..., :3])
    return np.clip(out, 0.0, 1.0)[::-1]


def _rgba_bytes_to_float(buf, w, h):
    """RGBA8 bytes, top row first -> (h, w, 4) float32 0..1 display values, top-down."""
    return np.frombuffer(buf, np.uint8).reshape(h, w, 4).astype(np.float32) / 255.0


def _box(a, tw, th):
    h, w = a.shape[:2]
    if tw < w:
        e = np.arange(tw + 1) * w // tw
        a = np.add.reduceat(a, e[:-1], axis=1) / np.diff(e)[None, :, None]
    if th < h:
        e = np.arange(th + 1) * h // th
        a = np.add.reduceat(a, e[:-1], axis=0) / np.diff(e)[:, None, None]
    return a


# --------------------------------------------------------------------------- payload decoding

def _payload(fmt, data, width, height, max_pixels):
    """Validate and decode an upload to (w, h, display float32 (h, w, 4) top-down).
    Raises OpError; touches no Blender data."""
    if fmt == "png":
        if width is not None or height is not None:
            raise OpError("invalid_params", "width/height are only for format rgba8 "
                                            "(a PNG carries its own size)")
    else:
        if width is None or height is None:
            raise OpError("invalid_params", "format rgba8 needs width and height")
        if width * height > max_pixels:
            raise OpError("budget", f"{width}x{height} is over the {max_pixels} pixel limit "
                                    "(see 'Max image pixels' in the Blender Safe MCP panel)")
        want = width * height * 4
        if len(data) != (want + 2) // 3 * 4:
            raise OpError("invalid_params", f"data is not the base64 of exactly {want} bytes "
                                            f"(width*height*4)")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise OpError("invalid_params", "data is not valid base64") from None
    if fmt == "png":
        try:
            width, height, raw = imaging.decode_png(raw, max_pixels)
        except ValueError as e:
            raise OpError("invalid_params", f"cannot decode PNG: {e}") from None
    elif len(raw) != width * height * 4:
        raise OpError("invalid_params", f"rgba8 data must be {width * height * 4} bytes")
    _check_work(width, height)
    return width, height, _rgba_bytes_to_float(raw, width, height)


# --------------------------------------------------------------------------- read-only ops

@op("list_images", Obj({}), description=(
    "List image datablocks: name, size [w, h], channels, is_float, kind, writable, colorspace, "
    "alpha_mode, users (real users, not the fake user), fake_user. kind is 'external' (backed "
    "by a file; read/use only), 'embedded' (packed in the .blend; writable), 'generated' "
    "(made in Blender, no file; writable, becomes embedded on its first write) or 'special' "
    "(Render Result, viewer, movie, sequence, tiled; info only). File paths are never shown."))
def list_images():
    imgs = list(bpy.data.images)
    return {"count": len(imgs), "images": [_info(i) for i in imgs[:500]]}


@op("get_image", Obj({
    "name": NAME,
    "preview_max": Opt(Int(64, 2048), 512),
    "region": Opt(RECT),
}), description=(
    "Info plus a PNG preview of an image's pixels (as image content). The preview is the whole "
    "image, or the region [x, y, w, h] (origin top-left, like a PNG), box-filtered down so its "
    "longer side is at most preview_max (float images are shown sRGB-encoded). Works for "
    "external images too; 'special' images return info only. Use it to verify what you "
    "created or uploaded."))
def get_image(name, preview_max, region):
    img = _image(name)
    info = _info(img)
    if info["kind"] == "special":
        return {"image": info, "note": "this kind of image is info only; no preview"}
    arr = _read(img)
    h, w = arr.shape[:2]
    x, y, rw, rh = region if region else (0, 0, w, h)
    if rw < 1 or rh < 1 or x + rw > w or y + rh > h:
        raise OpError("out_of_range", f"region {[x, y, rw, rh]} is outside the {w}x{h} image")
    sub = arr[h - y - rh:h - y, x:x + rw]
    disp = _display_u8(img, sub)
    s = min(1.0, preview_max / max(rw, rh))
    tw, th = max(1, round(rw * s)), max(1, round(rh * s))
    disp = _box(disp, tw, th)
    rgba = np.rint(np.clip(disp, 0.0, 1.0) * 255.0).astype(np.uint8)
    png = imaging.encode_png_rgba(tw, th, rgba.tobytes())
    return WithBlobs({"image": info, "region": [x, y, rw, rh], "preview": {"$blob": 0},
                      "preview_size": [tw, th]}, [png])


# --------------------------------------------------------------------------- creating / writing

def _budget_images():
    if len(bpy.data.images) >= MAX_IMAGES:
        raise OpError("budget", f"at most {MAX_IMAGES} images")


@op("create_image", Obj({
    "name": NAME,
    "width": Int(1, 16384),
    "height": Int(1, 16384),
    "kind": Opt(Enum(("blank", "uv_grid", "color_grid")), "blank"),
    "color": Opt(COLOR4, None),
    "alpha": Opt(Bool(), True),
    "float": Opt(Bool(description="32-bit float buffer (stored as OpenEXR when embedded)"),
                 False),
    "colorspace": Opt(Str(max_len=64), None),
}), mutates=True, description=(
    "Create a new image, embedded in the .blend (never saved to a file). kind: blank (filled "
    "with 'color', default opaque black), uv_grid or color_grid (Blender's test patterns). "
    "Subject to the user's 'Accept image uploads' setting and 'Max image pixels'. Returns the "
    "(possibly suffixed) name and info. The image has a fake user so it survives saving. "
    + UNDO_NOTE))
def create_image(name, width, height, kind, color, alpha, float, colorspace):
    _need_images()
    limit = _pixel_limit()
    if width * height > limit:
        raise OpError("budget", f"{width}x{height} is over the {limit} pixel limit (see "
                                "'Max image pixels' in the Blender Safe MCP panel)")
    if color is not None and kind != "blank":
        raise OpError("invalid_params", "color only applies to kind 'blank'")
    _check_colorspace(colorspace)
    _budget_images()
    img = bpy.data.images.new(name, width, height, alpha=alpha, float_buffer=float)
    try:
        if colorspace is not None:
            img.colorspace_settings.name = colorspace
        if kind == "blank":
            fill = np.array(color if color is not None else (0, 0, 0, 1), np.float32)
            arr = np.broadcast_to(_encode_store(img, fill), (height, width, 4))
        else:
            img.generated_type = "UV_GRID" if kind == "uv_grid" else "COLOR_GRID"
            arr = _read(img)
        _commit(img, arr)
        img.use_fake_user = True
    except BaseException:
        bpy.data.images.remove(img)
        raise
    return _info(img)


@op("upload_image", Obj({
    "name": NAME,
    "format": FORMAT,
    "data": DATA,
    "width": Opt(Int(1, 16384), None),
    "height": Opt(Int(1, 16384), None),
    "replace": Opt(Bool(), False),
    "colorspace": Opt(Str(max_len=64), None),
    "alpha_mode": Opt(Enum(ALPHA_MODES), None),
}), mutates=True, description=(
    "Create an image from uploaded pixels (base64 PNG, or raw RGBA8 with width/height), "
    "embedded in the .blend. With replace=true the existing writable image of that name is "
    "overwritten (and resized if the size differs); otherwise a new image is made. PNGs are "
    "decoded by the addon itself (8/16-bit, no interlacing). Subject to 'Accept image "
    "uploads', 'Max image pixels' and the request size cap in the Blender Safe MCP panel. colorspace "
    "e.g. 'Non-Color' for normal/roughness maps. " + UNDO_NOTE))
def upload_image(name, format, data, width, height, replace, colorspace, alpha_mode):
    _need_images()
    _check_colorspace(colorspace)
    limit = _pixel_limit()
    target = _writable(name) if replace else None
    w, h, disp = _payload(format, data, width, height, limit)
    if target is None:
        _budget_images()
        img = bpy.data.images.new(name, w, h, alpha=True)
        fresh = True
    else:
        img, fresh = target, False
    try:
        if colorspace is not None:
            img.colorspace_settings.name = colorspace
        if alpha_mode is not None:
            img.alpha_mode = alpha_mode
        if not fresh and tuple(img.size) != (w, h):
            img.scale(w, h)
        _commit(img, _encode_store(img, disp[::-1]))
        if fresh:
            img.use_fake_user = True
    except BaseException:
        if fresh:
            bpy.data.images.remove(img)
        raise
    return _info(img)


@op("write_image_region", Obj({
    "name": NAME,
    "x": Int(0, 1_000_000),
    "y": Int(0, 1_000_000),
    "format": FORMAT,
    "data": DATA,
    "width": Opt(Int(1, 16384), None),
    "height": Opt(Int(1, 16384), None),
}), mutates=True, description=(
    "Paste uploaded pixels (PNG, or raw RGBA8 with width/height) into a writable image with "
    "its top-left corner at (x, y) (origin top-left, like a PNG). Must fit inside the image. "
    "The image is embedded afterwards. " + UNDO_NOTE))
def write_image_region(name, x, y, format, data, width, height):
    _need_images()
    img = _writable(name)
    iw, ih = img.size
    _check_work(iw, ih)
    w, h, disp = _payload(format, data, width, height, _pixel_limit())
    if x + w > iw or y + h > ih:
        raise OpError("out_of_range", f"a {w}x{h} patch at ({x}, {y}) does not fit the "
                                      f"{iw}x{ih} image")
    arr = _read(img)
    arr[ih - y - h:ih - y, x:x + w] = _encode_store(img, disp[::-1])
    _commit(img, arr)
    return _info(img)


@op("fill_image_rect", Obj({"name": NAME, "rect": RECT, "color": COLOR4}),
    mutates=True, description=(
        "Fill the rectangle [x, y, w, h] (origin top-left) of a writable image with a solid "
        "RGBA color (0..1, display-encoded; float images store it as linear). The image is "
        "embedded afterwards. " + UNDO_NOTE))
def fill_image_rect(name, rect, color):
    _need_images()
    img = _writable(name)
    iw, ih = img.size
    _check_work(iw, ih)
    x, y, w, h = rect
    if w < 1 or h < 1 or x + w > iw or y + h > ih:
        raise OpError("out_of_range", f"rect {rect} is outside the {iw}x{ih} image")
    arr = _read(img)
    arr[ih - y - h:ih - y, x:x + w] = _encode_store(img, np.array(color, np.float32))
    _commit(img, arr)
    return _info(img)


@op("set_image_settings", Obj({
    "name": NAME,
    "colorspace": Opt(Str(max_len=64), None),
    "alpha_mode": Opt(Enum(ALPHA_MODES), None),
}), mutates=True, description=(
    "Change the colorspace (e.g. 'Non-Color' for data maps, 'sRGB' for colors) and/or alpha "
    "mode of a writable (embedded or generated) image. External file-backed images are "
    "refused: they are never modified."))
def set_image_settings(name, colorspace, alpha_mode):
    if colorspace is None and alpha_mode is None:
        raise OpError("invalid_params", "give colorspace and/or alpha_mode")
    _check_colorspace(colorspace)
    img = _writable(name)
    generated = kind_of(img) == "generated"
    if generated:
        _need_images()  # embedding a generated image is a pixel write
        arr = _read(img)
    if colorspace is not None:
        img.colorspace_settings.name = colorspace
    if alpha_mode is not None:
        img.alpha_mode = alpha_mode
    if generated:
        _commit(img, arr)
    return _info(img)


@op("delete_image", Obj({"name": NAME, "force": Opt(Bool(), False)}), mutates=True,
    destructive=True, description=(
        "Delete a writable (embedded or generated) image datablock. Refused while something "
        "uses it unless force=true (users of it then lose the image). External images are "
        "never deleted here."))
def delete_image(name, force):
    img = _writable(name)
    users = _real_users(img)
    if users > 0 and not force:
        raise OpError("invalid_params", f"image {img.name!r} has {users} user(s); "
                                        "pass force=true to delete it anyway")
    bpy.data.images.remove(img)
    return {"deleted": name}


# --------------------------------------------------------------------------- using images

def _usable_image(name):
    img = _image(name)
    if not assignable(img):
        raise OpError("invalid_params", f"image {img.name!r} ({kind_of(img)}, {img.type}) can't "
                                        "be used here: only file or generated 2D images")
    return img


IMAGE_NODES = ("ShaderNodeTexImage", "ShaderNodeTexEnvironment")


@op("set_node_image", Obj({"owner": NODE_OWNER, "node": NAME,
                           "image": _NameOrNull(description="Image name, or null to clear")}),
    mutates=True, description=(
        "Assign an image to an Image Texture or Environment Texture shader node (create the "
        "node with add_shader_node first). Works for any file or generated image, including "
        "external ones; movie/sequence/tiled/render images are refused."))
def set_node_image(owner, node, image):
    img = None if image is None else _usable_image(image)
    tree = _node_tree(owner)
    n = _node(tree, node)
    if n.bl_idname not in IMAGE_NODES:
        raise OpError("invalid_params", f"{n.name} is a {n.bl_idname}; need one of {IMAGE_NODES}")
    n.image = img
    return {"node": n.name, "image": img.name if img else None}


@op("set_texture_image", Obj({"texture": NAME,
                              "image": _NameOrNull(description="Image name, or null to clear")}),
    mutates=True, description=(
        "Assign an image to an IMAGE-type texture datablock (create_texture type IMAGE), for "
        "use with set_modifier_texture. Same image rules as set_node_image."))
def set_texture_image(texture, image):
    tex = bpy.data.textures.get(texture)
    if tex is None:
        raise OpError("not_found", f"no texture named {texture!r}")
    if tex.type != "IMAGE":
        raise OpError("invalid_params", f"{tex.name} is a {tex.type} texture, not IMAGE")
    img = None if image is None else _usable_image(image)
    tex.image = img
    return {"texture": tex.name, "image": img.name if img else None}

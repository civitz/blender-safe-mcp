# Blender Safe MCP

> **Warning: highly experimental.** This is a toy project at this stage. Expect radical changes
> (tool names, parameters, behaviour, even the architecture) without notice, and bugs. The safety
> measures described below are a design goal, not a guarantee: don't rely on them with anything
> you can't afford to lose, and keep your work saved and under version control.

A Blender addon that is itself an MCP server (Streamable HTTP). It lets an AI agent drive a live
Blender through **just over a hundred fixed, validated operations** ([TOOLS.md](TOOLS.md)) and it
**cannot execute Python** and **never takes a file path or URL from the model**. The worst a
confused or hostile agent can do is make a mess of the open scene, and every change is one undo
step. The full rationale and threat model are in
[`CLAUDE.md`](CLAUDE.md#design).

## Design

```
┌────────────┐   MCP / Streamable HTTP    ┌──────────────────────────────────────────────┐
│ LLM client │ ──────────────────────────▶│ Blender + Blender Safe MCP addon             │
│            │  POST /mcp, Bearer <key>   │                                              │
│            │ ◀────────────────────────── │ HTTP threads ─▶ work queue ─▶ main thread    │
└────────────┘   JSON text + PNG images   │ (auth, limits)                (guard ─▶ op)  │
                                          └──────────────────────────────────────────────┘
```

There is no separate server or helper process: the MCP client connects straight to Blender, and the addon
validates everything itself. Because `bpy` is single-threaded, HTTP threads only parse and
authenticate; the work is handed to Blender's main thread through a queue drained by a
`bpy.app.timers` timer.

**What stops the agent from doing harm**

| Layer | What it does |
|---|---|
| Closed op vocabulary | Each tool is a hand-written op with a strict schema (unknown keys, wrong types, out-of-range numbers and oversized arrays are rejected). No tool takes code, an RNA path string, a file path or a URL. Datablocks are looked up by name in `bpy.data`, never resolved as paths. |
| Structural property access | `get_property` / `set_property` take the path as a JSON list and walk real RNA properties only (no Python attributes or methods), through allow-listed structs. Path-like strings, pointers, text/library/image/driver/preferences structs and addon-defined properties are unreachable; numbers are range-capped. |
| Guard | Everything is refused while Blender's *Auto Run Python Scripts* is on. Scene-changing tools are refused unless the user ticked **Allow scene changes** (off by default). Each mutating call is one undo step; refused calls leave no step. |
| Tripwires | If any call leaves a new text block, library, or an image that is file-backed or unpacked, the server disables itself until restarted from the panel. |
| Budgets | Estimated geometry is checked before an op runs (5 M faces), plus caps on modifiers, nodes, splines, UV maps, subdivision, render samples/resolution and image pixels. |
| Transport | Bearer key on every request (constant-time compare, per-address back-off on wrong keys), `Host`/`Origin` checks against DNS rebinding, request size, depth, connection and idle-time limits, loopback bind by default. |
| Images | Uploads are decoded by the addon's own PNG decoder; everything created is packed into the `.blend`; nothing is ever saved or exported and no path is ever set or returned. |

An OS sandbox (no network, read-only home) is recommended as a backstop but is not part
of this project.

**Code layout**

```
blender_safe_mcp/   Blender addon (the trust boundary, and the whole MCP server)
  mcp_http.py   MCP over Streamable HTTP: auth, Host/Origin checks, JSON-RPC (pure Python, tested)
  server.py     main-thread work queue, login back-off, OpError (pure Python, tested)
  schema.py     strict param validation → JSON Schema    (pure Python, tested)
  registry.py   op table + dispatch                      (pure Python, tested)
  props.py      allow-listed RNA walker for get/set_property (tested against fake RNA)
  logbuf.py     10 MB in-memory log ring for get_debug_logs   (pure Python, tested)
  keystore.py   the API key file                         (pure Python, tested)
  imaging.py    in-memory PNG encoder + hardened PNG decoder and UV-layout rasteriser (pure Python, tested)
  uvmath.py     UV projections, island/area stats, 2D transforms (pure Python, tested)
  falloff.py    proportional-editing falloff curves      (pure Python, tested)
  ops.py        core ops: scene, objects, modifiers, materials/nodes, property access (bpy)
  ops_transform.py  cursor, set_origin, apply_transform, align/distribute/snap/drop/look_at (bpy)
  ops_mesh.py   mesh selector, get_mesh_elements, extrude/inset/transform/delete/dissolve/... (bpy)
  ops_mesh2.py  bevel, loop_cut, bisect, subdivide, bridge, spin, smooth, symmetrize, separate, snap, tri, poke (bpy)
  ops_curve.py  add_curve, curve primitives, spline edit ops, bevel/taper objects, convert_object (bpy)
  ops_texture.py  procedural textures, set_modifier_texture, UV ops (bpy)
  ops_image.py  list/get/create/upload/write/fill/settings/delete images, set_node_image, set_texture_image (bpy)
  blender.py    prefs, sidebar panel, guard, tripwires, timers (bpy)
tests/                    pytest suite + in-Blender smoke tests + TOOLS.md generator
TOOLS.md                  every tool, one entry each (generated from the op registry)
```

The pure-Python modules never import `bpy`, so the pytest suite covers them (and the HTTP
endpoint, driven by the official MCP SDK client) without Blender; the `ops*.py` files and
`blender.py` are covered by the in-Blender smoke tests.

## Tools

There are 11 groups of tools; [TOOLS.md](TOOLS.md) documents each one (description,
parameters with types, ranges and defaults, whether it is read-only, changes the scene or is
destructive). The same data is what `tools/list` returns. In short:

- **Look**: `get_scene_info`, `list_objects`, `get_object_info`, `get_property`,
  `capture_viewport` (the 3D viewport or the scene camera as a PNG), `render_image`, `get_shader_graph`.
- **Build**: primitives, raw meshes, curves, text, lights, cameras, modifiers, materials and
  shader nodes, textures, UV maps, images, keyframes, the world.
- **Edit meshes**: a selector-based set of bmesh editing tools (extrude, inset, bevel, loop cut,
  bisect, bridge, spin, dissolve, symmetrize, separate, repair, ...).
- **Place**: align, distribute, snap, drop to surface, look-at, cursor and origin.
- **Session**: `undo` / `redo`, `get_status`, `list_ops`, `get_debug_logs`.

The tools file is regenerated with
`blender --background --factory-startup --python tests/gen_tools_md.py`; the
smoke test fails when it is out of date.

## Install

1. **Addon**: install `blender_safe_mcp.zip` (repo root) with *Edit → Preferences → Get
   Extensions → Install from Disk* (Blender 4.2+).
2. In the 3D view sidebar (`N`) → **Blender Safe MCP**: press **Start server**. The panel shows
   `Serving http://127.0.0.1:9877/mcp`. Turn on **Allow scene changes** when you want the agent
   to edit; it's read-only by default (and reinstalling the addon turns it off again). If the
   panel warns about *Auto Run Python Scripts*, disable it: the addon refuses to work while it's
   on.
3. **Client**: the panel's **Copy 'claude mcp add'** button gives a ready command, or put this in
   `.mcp.json`:

```json
{
  "mcpServers": {
    "blender-safe-mcp": {
      "type": "http",
      "url": "http://127.0.0.1:9877/mcp",
      "headers": {"Authorization": "Bearer ${BLENDER_SAFE_MCP_KEY}"}
    }
  }
}
```

`${BLENDER_SAFE_MCP_KEY}` is expanded from Claude Code's environment, which keeps the key out of
the file; a literal key works too. The key lives in `$XDG_CONFIG_HOME/blender-safe-mcp/api_key`
(mode 0600), so it survives
addon reinstalls and Blender upgrades. **Regenerate key** replaces it, effective immediately.

Bind address and port are in the addon preferences.

## Protocol

Protocol versions, advertised by the server:
- **2026-07-28** (stateless): `server/discover` returns `supportedVersions`, capabilities and
  instructions; each POST carries its version in `params._meta` and `MCP-Protocol-Version`
  (and `Mcp-Method`/`Mcp-Name`, which must match the body).
- **2025-11-25, 2025-06-18, 2025-03-26** (handshake): `initialize` negotiates one of these and
  opens a session (`Mcp-Session-Id`).

An unsupported version gets error `-32022` listing all four. Responses are single JSON bodies
(no SSE stream, so GET is 405). Tools come from the addon's op table on every `tools/list`
(`ttlMs: 0`), so a reinstalled addon's new ops show up on the next listing; a Blender restart
drops handshake sessions, and clients re-initialize on the resulting 404.

Security:
- the key is a bearer token sent on every request, compared in constant time; wrong keys
  trigger a per-address exponential back-off (429), a missing key doesn't (so a port probe
  can't lock you out). There is no TLS, so keep the bind address on loopback;
- requests are refused unless `Host` is an IP literal or `localhost` and any `Origin` is a
  loopback `http://` origin, so web pages can't reach the port through DNS rebinding;
- `Content-Length` is required and capped at 1 MB (64 MB by default while image uploads are
  accepted, see Images; the cap is checked only after the key was verified), JSON depth at 32, no batches; sockets time
  out after 15 s idle, and at most 16 connections are served at once.

If Blender isn't running when the client starts, the server shows as failed: start Blender's
server, then reconnect from `/mcp`.

## Errors

A refused or failed tool call returns an MCP tool result with `isError: true` and the text
`<code>: <message>`. Codes:

| Code | Meaning |
|---|---|
| `read_only` | The tool changes the scene and the user hasn't enabled **Allow scene changes** in the panel. |
| `unsafe_blender` | Blender's *Auto Run Python Scripts* is on; the user must turn it off. |
| `disabled` | A safety tripwire fired; the user must restart the server from the panel. |
| `images_disabled` | Image uploads/writes are turned off in the panel. |
| `invalid_params` | The arguments don't match the tool's schema, or the request is inconsistent (e.g. a selector that matches nothing). |
| `invalid_path` / `invalid_value` | `get_property` / `set_property`: a bad path segment, or a value of the wrong type or range. |
| `denied` | The property, struct or action is not reachable by design. |
| `not_found` | No datablock, node, element or modifier with that name. |
| `stale_indices` | `expect_counts` no longer matches the mesh: indices changed, re-query. |
| `budget` | The result would exceed a geometry, node, spline or pixel limit. |
| `out_of_range` | A value outside what the target supports (a region outside the image, ...). |
| `unavailable` | Not possible for this input right now (the object is in Edit mode, an image has no pixel data, ...). |
| `failed` | Blender's operation ran but failed. |
| `unknown_op` / `bad_request` | Not a tool of this server / malformed request. |
| `nothing_to_*` | The operation would change nothing (`nothing_to_undo`, `nothing_to_redo`). |
| `internal` | A bug in the addon: see `get_debug_logs(errors_only=true)`. |

## Self-debugging

The addon logs with Python's `logging` into a **10 MB in-memory ring buffer** (oldest records
evicted first) and to stderr. The `get_debug_logs` tool returns it:

- `min_level`: `DEBUG` | `INFO` | `WARNING` | `ERROR`
- `errors_only`: only records with a traceback
- `since_id`: a previous call's `last_id`, to tail new records

Each response is capped at about 768 KB, and `truncated: true` says older matches were left out.
What gets logged: sessions, auth failures (never the key), every op with its outcome and
duration, refusals, and full tracebacks of crashes. `get_debug_logs`, `get_status` and
`list_ops` work even when the guard would refuse everything else (tripwire fired, auto-run on).

## Tests

```sh
uv run --python 3.15 --with pytest --with 'mcp>=2.3,<3' --no-project python -I -m pytest tests -q
blender --background --factory-startup --python-exit-code 1 --python tests/blender_smoke.py
# regenerate TOOLS.md after adding or changing a tool (the smoke test fails when it is stale)
blender --background --factory-startup --python tests/gen_tools_md.py
# end to end: real Blender serves, the official MCP SDK client calls real ops
BLENDER_SAFE_MCP_SMOKE_CLIENT="uv run --python 3.15 --with 'mcp>=2.3,<3' --no-project python -I tests/http_smoke_client.py" \
  blender --background --factory-startup --python-exit-code 1 --python tests/blender_http_smoke.py
```

The pytest suite covers everything that doesn't need `bpy`, driving the HTTP endpoint with the
official MCP SDK client in both protocol eras. The smoke test exercises every scene op and the
escape attempts against real RNA. `capture_viewport` isn't covered, because background mode has
no viewport.

## Transform tools

`set_cursor` (world location, or onto an object's origin / bounds center / vertex median),
`set_origin` (bounds, median, center of mass, cursor, explicit point, geometry-to-origin) and
`apply_transform` (Ctrl+A for location/rotation/scale). They run fixed `bpy.ops` calls on
MESH/CURVE/FONT/SURFACE objects, restore the user's selection, active object and cursor, and refuse
shared mesh data unless `make_single_user=true`. `get_scene_info` also reports the cursor.

## Placement tools

`align_objects` (min/center/max of the evaluated world bounding boxes against the first object, an
object, the cursor, the world origin or the selection bounds), `distribute_objects` (equal center
spacing or equal gaps), `snap_objects` (grid, cursor, or an object's origin / bounds center /
nearest vertex / nearest surface), `drop_to_surface` (ray-cast onto target meshes and/or a
`floor_z` plane, optionally aligning to the surface normal) and `look_at` (aim any axis at a point
or object). They are plain `matrix_world` math (no `bpy.ops`), work on any object type (cameras,
lights and empties count as a point at their origin), handle parented objects in world space, and
plan every move before applying any, so a refusal changes nothing.

## Mesh selector and core mesh editing

Mesh edit ops take a **selector** instead of raw indices, because indices change after every
topology edit. A selector combines criteria (`verts`/`edges`/`faces` index lists, `all`, `normal`,
`box`, `sphere`, `material`, `boundary`, `edge_loop`, `edge_ring`) with `combine` (and/or), then
applies `grow` and `invert`. Domains convert like Blender's select modes (a face is selected when
all its vertices are, an edge when both are, a face or edge selects its vertices). `expect_counts`
(the `[vertices, edges, faces]` from an earlier result) makes a call fail with `stale_indices`
rather than hit the wrong elements; a selector that matches nothing is `invalid_params`.

`get_mesh_elements` lists vertices/edges/faces for a selector (paged, world or local space; works
in Edit mode). The edit ops (`extrude`, `inset`, `transform_elements`, `delete_elements`,
`dissolve`, `set_edge_flags`, `normals`) work in Object mode through bmesh, return the indices of
the elements they created (matching what `get_mesh_elements` reports afterwards) and the
resulting counts, and refuse in Edit mode. Index lists in results are compact: consecutive runs
come as `[first, last]` ranges (`[3, [16, 767]]`, with `*_count` giving the totals), and selector
index lists accept the same form, so a result can be passed straight back. `transform_elements` supports proportional editing
with Blender's falloff curves (checked against Blender's own operator in the smoke test).
`normals` covers selection-scoped recalculation, inside normals and flipping; `repair_mesh` still
does the whole-mesh outward recalculation as part of its cleanup.

More mesh editing (`ops_mesh2.py`), same selector, return shape and Object-mode rule: `bevel`
(edges or vertices; offset/width/depth/percent), `loop_cut`, `bisect` (clear inner/outer, optional
fill), `subdivide`, `bridge_edge_loops`, `spin`, `smooth_vertices`, `symmetrize`, `separate`
(selection / loose parts / material; the new objects copy the source's transform, collections,
modifiers and material slots; at most 256 new objects), `snap_vertices` (grid / cursor / another
mesh's surface or vertices), `triangulate_faces` and `poke_faces`. Each estimates its result and
is refused over the geometry budget before it runs. Notes: shape keys do not survive `separate`;
`spin` of bare edges re-orients the swept faces consistently; `bridge_edge_loops` between loops
with opposite facing needs `normals` afterwards.

## Curves and conversion

`add_curve` builds a curve object from explicit splines (bezier with handles, poly, NURBS with
weights; at most 256 splines, 10 000 points per spline, 50 000 per curve) using `bpy.data` only;
`add_curve_primitive` makes a bezier/NURBS circle, bezier curve, NURBS path or spiral. `get_curve_info`
reads splines, points, handles and the bevel/taper object names back; `set_spline` (replace, keeps the
spline's index), `add_spline` and `remove_spline` edit them. `set_curve_shape` sets `bevel_mode`
and the `bevel_object` / `taper_object` (CURVE objects only, no self-reference, no loops; `""`
clears). These four refuse curve data shared with other objects unless `make_single_user=true`
(which gives the object its own copy first). Bevel depth/extrude/resolution go through `set_property` (`resolution_u` is capped at 64,
`bevel_resolution` at 32, spline `order_u` at 2..6). `add_modifier` also takes `CURVE` (Curve
Deform; `target_object` is the CURVE object) and `curve_object` for Array fit-to-curve (the copy
count is estimated from the curve length and refused over the geometry budget).

`convert_object` turns MESH/CURVE/FONT/SURFACE/META into a **MESH** (the evaluated result: modifiers,
bevel, extrude and text shapes are baked in, the new object has no modifiers; hidden modifiers are
reported in `modifiers_not_baked`) or MESH/FONT into a **CURVE** (the one fixed `bpy.ops` call,
`object.convert`). It keeps matrix, parent, collections and materials; without `keep_original` the
source is deleted and its children are re-parented keeping their world transforms. The geometry
budget is checked on the evaluated result before anything is created. Animation and shape keys of
the source are not carried over (reported as warnings).

## Procedural textures and UVs

`create_texture` makes a procedural texture datablock (CLOUDS, DISTORTED_NOISE, BLEND, MAGIC, MARBLE,
MUSGRAVE, NOISE, STUCCI, VORONOI, WOOD, and IMAGE, which starts empty: assign one with
`set_texture_image`; the type is checked against Blender's own list, NONE is refused). Its `settings` go through the same RNA walker as
`set_property`; afterwards use `get_property` / `set_property` with `{"kind": "texture", "name": ...}`.
`Texture.type` is read-only there (a texture's type is fixed when it is created). `set_modifier_texture` points a Displace or Wave modifier at a texture (or clears
it with null) and sets `texture_coords` / `texture_coords_object` / `uv_layer`.

UV ops: `list_uv_maps` (read-only), `uv_maps` (add / remove / rename / set_active /
set_active_render, at most 8 maps), `uv_project` (cube / cylinder /
sphere / planar in bmesh math, no operator; cylinder and sphere shift each face so none spans the
wrap, and caps that surround the axis come out as flat strips), `uv_unwrap` (Blender's
`smart_project`, `unwrap` or `lightmap_pack`, run in Edit mode; select flags, active UV map,
select mode, UV-sync setting, active object, selection and mode are restored even on failure, and
a UV map created only for a failed call is removed), `transform_uvs` and `get_uv_layout` (a PNG of
the UV edges plus island count, bounds and zero-area faces). Selectors are the same as for the mesh
ops. The shader nodes `ShaderNodeUVMap`, `ShaderNodeNormalMap` and `ShaderNodeAttribute` can be
added; their `uv_map` / `attribute_name` strings are writable (plain names, no file paths).
`uv_unwrap` runs headless too (all three operators were checked in background mode).

## Images

Policy: **external** images (backed by a file, not packed) can be listed, previewed and used in
nodes/textures but never written, resized, re-colourspaced or deleted. **Embedded** (packed) and
**generated** (made in Blender, no file) images can be written; every write ends in `image.pack()`,
so a generated image becomes embedded on its first write. Render Result, viewer, movie, sequence
and tiled images are info only. Nothing here saves, unpacks or reloads an image, sets a file path,
or returns one (not even a basename).

Ops: `list_images`, `get_image` (info + PNG preview, optional region `[x, y, w, h]`, origin
top-left; box-filtered; float images shown sRGB-encoded), `create_image` (blank / uv_grid /
color_grid, optional float buffer, colorspace), `upload_image` (base64 `png` or raw `rgba8`,
optional `replace`, colorspace, alpha_mode), `write_image_region`, `fill_image_rect`,
`set_image_settings` (colorspace / alpha mode; writable images only), `delete_image` (destructive;
needs `force` while used), `set_node_image` (Image Texture / Environment Texture nodes) and
`set_texture_image` (IMAGE textures). `ShaderNodeTexImage` / `ShaderNodeTexEnvironment` can now be
added with `add_shader_node`; the `image` and `image_user` pointers stay out of `set_property`.
Created/uploaded images get a fake user (so they survive saving); `users` in the listings excludes it.

Uploads are decoded by the addon's own pure-Python PNG decoder (`imaging.decode_png`: sizes
checked first, zlib bounded by the exact expected size, every CRC checked, 8/16-bit grey / grey+alpha
/ RGB / RGBA / palette with tRNS, all filters, no Adam7). The bytes never reach Blender's image
loaders. Decoding runs on Blender's main thread, so a large filtered PNG pauses the UI: about 3 s
(1024², Paeth filter) to 12 s (2048², Paeth) on Python 3.11 (Blender's); unfiltered/Up-filtered PNGs are far
faster. Prefer `rgba8` for big images you generate yourself. Pixel values you send and receive are
display-encoded 0..1; float images convert from/to sRGB, byte images store them unchanged
(Non-Color maps therefore round-trip exactly). Pixel operations are limited to 33 554 432 pixels
(512 MB of float working memory) whatever the preference says.

Panel/preferences ("Images" box): **Accept image uploads** (default on; off makes `create_image`,
`upload_image`, `write_image_region` and `fill_image_rect` fail with `images_disabled` and drops the
HTTP body cap back to 1 MB; reading and assigning still work), **Max image pixels** (default
16 777 216, max 268 435 456; create/upload/resize) and **Max request size (MB)** (default 64,
1-1024; the HTTP body cap while uploads are accepted). The cap is checked after authentication.

Tripwire: around every mutating op the guard compares all images before/after. A pre-existing
image whose file path changed, that was re-sourced, or that went from packed to unpacked trips the
server (as texts/libraries do), and a new image that is neither packed nor an unpathed
generated/viewer image is deleted and trips it. Empirically, in Blender 5.1 `pack()` on a generated
image leaves `filepath_raw` empty but flips `source` from GENERATED to FILE (and sets
`packed_file`); the rule allows exactly that transition. `pack()` does nothing on an unmodified
buffer, so the create ops write the pixels back first.

**Undo and images** (measured in Blender 5.1 over HTTP): undoing the step that *created* an image
removes it. Undoing a `fill_image_rect` (or upload/region write) does **not** restore the old pixels:
the image keeps the new ones, though the undo step still counts. Treat pixel writes as not undoable;
re-upload or re-fill to change them back. Images are not tracked by the `undo` result report.

## Limitations

- Objects are addressed by name (looked up in `bpy.data`, never resolved as paths), not by
  opaque handles.
- No TLS yet. Keep the bind address on loopback.
- Changing the colorspace/alpha mode of an external image is not allowed (policy: external
  images are never written), so a user-loaded normal map cannot be marked Non-Color through MCP.
- No asset import, export or image-to-3D ops: by design nothing leaves or enters the `.blend` except
  through the tools above.
- No relay for running Blender inside a network-less sandbox (see "Ideas not built" in CLAUDE.md).
- The geometry budget is an estimate computed before evaluation, not a measurement afterwards.
- `set_property` can't switch `render.engine`: Blender's static enum list for it only has
  `BLENDER_EEVEE`, and `props.py` validates against that list.
- **`undo` is Blender's undo history, shared with the user.** Each MCP change is one step,
  but so is every edit made in Blender's UI (even a selection), so `undo` may revert the
  user's step instead of the agent's. `undo`/`redo` report the steps actually taken and which
  objects were added, removed or changed, with a note when nothing visible changed. (Fixed
  2026-11-04: refused ops used to record an empty step, so the next `undo` landed on it and
  changed nothing while reporting `ok`; `tests/http_smoke_client.py` covers it.)
- Mesh edit ops copy the mesh into bmesh and back, so very large meshes (above 2 million faces)
  are refused, and proportional editing refuses to touch more than 500 000 vertices.
- On Blender 5.1.2 the smoke test's `stitching` check fails: a mesh built with a duplicated face
  comes out manifold, so the duplicate is apparently dropped on creation.

## License

GPL-3.0-or-later (see [LICENSE.md](LICENSE.md)), the license Blender add-ons that use `bpy` are expected to
have. It matches the `license` field in `blender_safe_mcp/blender_manifest.toml`.

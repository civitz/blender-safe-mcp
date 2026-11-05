# TODO

## Bake tool (on hold: design not decided)

Do not implement yet; the owner wants to think it through first. Open questions:

- **Purpose:** high→low-poly normal maps, AO, flattening procedural shaders into an image,
  lightmaps? With export blocked, a bake result can only live packed in the .blend or come back
  to the agent as a PNG.
- **Target images:** bakes write into an Image. Should bakes only target images created through
  MCP (embedded, see the image rules in the README), or also embedded images the
  user created?
- **Security:** `BakeSettings` is in `DENIED_STRUCTS` because it has an output `filepath` and
  `save_mode = EXTERNAL`. A bake op must set its fields explicitly (type, margin, samples,
  cage/ray distance, selected-to-active) and never write files.
- **Cost:** a Cycles bake blocks Blender's main thread like `render_image`; it needs caps on
  samples and resolution (and maybe a time budget).

## Open decisions (owner)

Raised while implementing the v2 tools; current behaviour in parentheses.

- **Image settings on external images:** allow `set_image_settings` (colorspace, alpha mode
  only) on file-backed images, e.g. to mark a user-loaded normal map Non-Color? It doesn't touch
  the file. (Refused: external images are never written.) Recommended: allow these two.
- **PNG decode thread:** `upload_image` decodes on Blender's main thread; a 2048² Paeth PNG takes
  ~7–12 s and freezes the UI. Move decoding to the HTTP thread? (Left as is: the GIL would still
  stall Blender's Python UI; `rgba8` uploads are near-instant.) Recommended: leave.
- **`set_property` on shared data:** it still edits mesh/curve data shared by several objects
  (e.g. `["data", "bevel_depth"]`), unlike the dedicated ops, which refuse unless
  `make_single_user`. Add the same rule to the props writer?

## Release prep (do last, after the open decisions)

- **README for the release: done.** It now has the design summary (architecture, layers, code
  layout), install, client config, the error codes and the limitations; the details are in
  `TOOLS.md` and `CLAUDE.md`.
- **Separate tools reference: done.** `TOOLS.md` has one entry per tool and is
  generated from the op registry by `tests/gen_tools_md.py`; `blender_smoke.py` fails when it is
  stale. (A plain pytest can't check it: the op modules import `bpy`.)

## Defaults chosen by the implementing agents (change if unwanted)

- `separate` (loose_parts / material) keeps the first part in the source object; `selection`
  mode refuses to empty the source.
- `bridge_edge_loops` doesn't fix winding between loops facing the same way (run `normals`).
- `convert_object` without `keep_original`: other objects' references to the deleted source
  (bevel object, modifier targets) are cleared, as Blender does, not repointed.
- `uv_project` cylinder/sphere: faces surrounding the axis (caps) become near-zero-area strips;
  no disc projection.
- `get_uv_layout` draws edges only (no filled faces / checker).
- `drop_to_surface` is all-or-nothing: one object without a hit fails the call.

## Known issues / follow-ups

- **Pixel writes aren't undoable** (confirmed live on 5.1.1): undo removes a created image but
  leaves the pixels written by `fill_image_rect` / uploads / region writes. Documented in the tool
  descriptions and the README; making it undoable would need our own pixel snapshots.
- **Placement ops and constraints:** they set `matrix_world` directly, so objects with
  constraints may not end up exactly where requested.
- **Conservative estimates:** bevel/subdivide growth estimates are upper bounds, so some huge but
  valid edits are refused with `budget`.
- **`remove_duplicate_faces` rarely has work on 5.1.2:** `add_mesh` (validate) and
  `merge_vertices` (weld) already drop duplicate faces, so no MCP path leaves one behind. Keep it
  for meshes the user made; the smoke test accepts both behaviours.
- **Flaky pytest:** one unexplained failure during phase 1 review, not reproduced in 15+ later
  runs.

## Done (recent)

- Mesh-edit results use compact index lists (`[3, [16, 767]]` ranges) and selectors accept them
  (verified live: a 10-cut subdivide returns three ranges instead of ~2 900 indices, and
  the range selects the same 720 faces back).
- Verified live after reinstalling: `drop_to_surface` on a bevelled spiral curve (lands on
  the table, a second drop moves 0), `capture_viewport` right after `set_viewport`, and
  `add_modifier` exposes `CURVE` / `curve_object` (the earlier gap was a stale client schema).
- Smoke harness: `check()` now reports `expect()` failures under their section; the `stitching`
  duplicate-face check builds the duplicate via join + merge and accepts 5.1.2's behaviour.

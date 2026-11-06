---
name: blender
description: Guide for driving the user's live Blender through the Blender Safe MCP server (the `blender` MCP tools). Use whenever the user asks to look at, build, model, edit, light, texture, animate or render anything in Blender, or a blender tool call is refused or fails.
---

# Working in Blender through Blender Safe MCP

The `blender` MCP server is an addon running inside the user's open Blender. It exposes about a
hundred fixed, validated operations. It **cannot run Python**, and it never takes a file path or
URL. Every change is one undo step in the user's own `.blend`, so you're working on the user's
real file. Be deliberate.

## Before the first change

1. Call `get_status` (Blender version, file, safety state) and `get_scene_info` (objects,
   camera, world, render engine).
2. If a call is refused, don't try to work around it. Tell the user which switch to change in
   Blender's 3D view sidebar (`N`) under **Blender Safe MCP**:
   - `read_only`: turn on **Allow scene changes**. It is off by default, and reinstalling the
     addon turns it off again.
   - `unsafe_blender`: turn off *Auto Run Python Scripts* in Blender's preferences.
   - `disabled`: a safety tripwire fired. The user must restart the server from the panel.
   - `images_disabled`: image uploads or writes are turned off in the panel.
3. If the tools are missing or the connection fails, the server probably isn't running. Ask the
   user to press **Start server** in the panel, then reconnect from `/mcp`.

## Work loop

Look, make one change, then check it:

- Inspect with `list_objects`, `get_object_info`, `get_shader_graph` and `get_property`.
- Change one thing per call, then check visually with `capture_viewport` (`camera: true` frames
  the shot through the scene camera). Use `render_image` only for the final look, because it is
  slow and blocks Blender. Headless Blender may have no GPU, so render with Cycles there.
  `capture_viewport` doesn't work in background mode.
- Positions are world space unless the op has `space: "local"`. Angles are degrees (`*_deg`).
  Objects are addressed by name.
- To find a setting, call `get_property` with an empty `path`: it lists what the datablock
  exposes. Then change the value with `set_property`. Paths are JSON lists, e.g.
  `["modifiers", "Bevel", "width"]`. Links between datablocks (materials, parents, textures,
  images) are set only by dedicated ops such as `assign_material`, `set_parent` or
  `set_node_image`, never through `set_property`.
- If you are unsure an op exists or what it takes, call `list_ops`. Tool descriptions are
  authoritative.

## Mesh editing

- Mesh edit ops work in Object mode. If the object is in Edit mode, switch with
  `set_object_mode` first.
- Pick elements with a **selector** (`all`, `normal`, `box`, `sphere`, `material`, `boundary`,
  `edge_loop`, `edge_ring`, or index lists, plus `combine`/`grow`/`invert`), not with remembered
  indices. Indices change after every topology edit.
- Pass `expect_counts` (the `[vertices, edges, faces]` from the previous result). If the mesh
  changed since, the call fails with `stale_indices` instead of editing the wrong elements. When
  that happens, re-query with `get_mesh_elements`.
- Check topology with `analyze_mesh` and fix it with `repair_mesh`, `merge_vertices`,
  `fill_hole` or `normals`.

## Limits

- Results are capped by budgets: 5 M estimated faces, 16 modifiers per object, 500 nodes per
  tree, 8 UV maps, 256 splines per curve, and 33 554 432 pixels per image operation. A `budget`
  refusal means you should take a smaller step (fewer cuts, lower subdivision levels, a smaller
  image). Don't retry the same call.
- `undo` is Blender's undo history, which the user shares. A step the user made in Blender
  (even a selection) can be undone instead of yours, so read what `undo` reports. Pixel writes
  (`fill_image_rect`, `write_image_region`, `upload_image`) are **not** undoable.
- There is no Python, no file access, no import or export, and no network by design. Don't ask
  for them and don't try to emulate them. If a task needs one (e.g. "import this FBX", "save to
  disk"), tell the user to do that part in Blender.

## Errors

A failed call returns `<code>: <message>`. Refusals change nothing.

| Code | What to do |
|---|---|
| `invalid_params` | Fix the arguments against the tool schema. A selector that matches nothing is this code too. |
| `invalid_path` / `invalid_value` | Re-discover with `get_property` (bad path segment, or a wrong type or range). |
| `denied` | Not reachable by design. Find another way or tell the user. |
| `not_found` | Re-list names (`list_objects`, `get_shader_graph`, `list_images`). |
| `stale_indices` | Re-query the mesh, then rebuild the selector. |
| `budget` | Make the step smaller. |
| `out_of_range` / `unavailable` | Check the target's state (Edit mode, image size, ...). |
| `failed` | Blender's operation failed. Check `get_debug_logs`. |
| `nothing_to_*` | Nothing to undo or redo. Not an error in your work. |
| `internal` | An addon bug. Call `get_debug_logs` with `errors_only: true` and report the traceback to the user. |

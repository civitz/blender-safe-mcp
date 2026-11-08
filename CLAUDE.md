# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
It is also the project's design record (threat model, rules for new ops, ideas); user-facing docs
are `README.md` (design summary, install, errors) and `TOOLS.md` (one entry per tool).

## What this is

A security-focused MCP server for Blender, built into a Blender addon: the agent can drive a
live Blender **without any Python execution, file paths or URLs**. About 100 ops (see `TOOLS.md`).

- `blender_safe_mcp/`: the Blender addon (also the repo's Python package; zipped as
  `blender_safe_mcp.zip`). **This is the trust boundary** and the whole MCP server: it serves MCP
  over Streamable HTTP at `http://127.0.0.1:9877/mcp`. There is no separate server or helper process.
- `tests/`: pytest suite (no Blender needed), `blender_smoke.py` and `blender_http_smoke.py`
  (run inside Blender), and `gen_tools_md.py` (generates `TOOLS.md`).
- `TOOLS.md`: **generated** reference with one entry per tool. Never edit it by hand.
- `TODO.md`: open decisions, defaults chosen by agents, known issues.
- `plugin/`: the Claude Code plugin (MCP client config with a `headersHelper` that reads the key
  file, and the `blender` guide skill); `.claude-plugin/marketplace.json` makes the repo its
  marketplace. `tests/test_plugin.py` fails if the skill names a tool, parameter or error code
  that doesn't exist.

## Commands

From the repo root:

```sh
# full pytest suite
uv run --python 3.15 --with pytest --with 'mcp>=2.3,<3' --no-project python -I -m pytest tests -q -p no:cacheprovider
# single test
uv run --python 3.15 --with pytest --with 'mcp>=2.3,<3' --no-project python -I -m pytest tests/test_props.py::test_denied_paths -q -p no:cacheprovider
# regenerate TOOLS.md after adding/changing an op (needs Blender). The smoke test fails if it is stale.
blender --background --factory-startup --python-exit-code 1 --python tests/gen_tools_md.py
# lint (Blender's `prop: bpy.props.X(...)` annotations in blender.py are pyflakes false positives)
uvx --python 3.15 pyflakes blender_safe_mcp tests
# Blender bundles Python 3.11: the addon must still compile there
uv run --python 3.11 --no-project python -I -m py_compile blender_safe_mcp/*.py
# in-Blender smoke test
blender --background --factory-startup --python-exit-code 1 --python tests/blender_smoke.py
# end-to-end over HTTP: real Blender + official MCP SDK client (see the file's docstring)
BLENDER_SAFE_MCP_SMOKE_CLIENT="uv run --python 3.15 --with 'mcp>=2.3,<3' --no-project python -I tests/http_smoke_client.py" \
  blender --background --factory-startup --python-exit-code 1 --python tests/blender_http_smoke.py
```

Headless runs may have no GPU: EEVEE then crashes, so render with Cycles;
`capture_viewport` can't run in background mode.

The addon is shipped as `blender_safe_mcp.zip` at the repo root, with the files at the **top level
of the zip** next to `blender_manifest.toml` (i.e. the contents of `blender_safe_mcp/`, not the
folder). Rebuild it after every addon change (delete `__pycache__` first), then the user
reinstalls it via Preferences → Get Extensions → Install from Disk. Blender's official packer also
works:
`blender --command extension build --source-dir blender_safe_mcp --output-dir .`

## Design

**Goal:** a malicious or confused LLM can, at worst, make a mess of the current scene. It must
not run code, read or write files outside Blender's own data, reach the network, or persist
anything outside the open `.blend`. Non-goal: matching `execute_blender_code`. We give up "anything
bpy can do" for "a closed set of things we've checked"; features are new, individually reviewed
ops, never a generic escape hatch.

**Threat model.** The client holds the API key, so the key decides *who may connect*, not *what
they may do*: the op vocabulary is the real boundary. How each threat is closed:

| Threat | Defence |
|---|---|
| Code execution (`exec`/`eval`, `bpy.ops` by string, driver expressions, text blocks, script nodes) | No op takes code; properties are reached structurally; `Text`, `Driver`, `FCurve`, `AnimData`, script nodes denied; tripwire on new texts/libraries |
| Writing or reading files, SSRF, imports | No op takes a path or URL; path-like RNA subtypes denied; all created images packed; no import/export ops; no outbound connections |
| Changing Blender's own settings | `Preferences`, `WindowManager`, `Screen`… denied; server refuses while *Auto Run Python Scripts* is on |
| Resource exhaustion | Strict schemas with maxima, geometry budget checked *before* an op runs, `NUMERIC_CAPS`, request-size/connection limits |
| Another local process or web page driving the port | Bearer key (constant time), per-address back-off, `Host`/`Origin` checks, loopback bind |
| Damage to the user's work | Read-only until "Allow scene changes"; one undo step per mutation; refusals change nothing |

**Layers** (each should stop code execution on its own): (1) the closed op vocabulary;
(2) guard and tripwires inside Blender; (3) transport hardening; (4) an OS sandbox (no
network, read-only home), which is recommended but not part of this project.

**Absent on purpose:** any `filepath`, `bpy.ops` passthrough, `script.*`, `text.*`,
`wm.open_mainfile`, `preferences.*`, drivers with expressions, script nodes, asset/model
import, export.

**Transport.** Single JSON responses over HTTP; the key lives in a 0600 file outside addon
preferences and the scene (those can end up in shared `.blend` files or be reset by a
reinstall), is sent as a bearer token and never logged; no TLS, so keep the bind on loopback
or a trusted host link; `Content-Length` required and capped (1 MB, or the user-set cap while image
uploads are on, checked after authentication), JSON depth 32, no batches, 15 s socket timeout, 16
connections, 32 sessions. Wrong keys trigger back-off (429); a missing key does not, so a port
probe can't lock the user out.

**Budgets.** `MAX_POLYS` (5 M faces) is estimated before modifiers, joins, conversions and mesh
edits run; at most 16 modifiers per object, 500 nodes per tree, 8 UV maps, 256 splines per curve;
pixel operations capped at 33 554 432 pixels. Estimates are upper bounds, so some huge but valid
edits are refused with `budget`.

## Writing a new op

Read first: the sections below, `ops.py` (`@op(...)`, `_obj`, `_MeshEdit`, `_summary`,
`_analyze_bm`, `_window_override`, the geometry budget helper, the existing `OpError` codes:
reuse them, don't invent parallel ones), `props.py`, `schema.py`, `blender.py` (`_guard`), and
`tests/blender_smoke.py` (`call`, `expect`, `check`).

Hard rules (security: these are the point of the project):
- **No Python execution paths:** no `exec`/`eval`, no `bpy.ops` called with a name derived from
  input, no `path_resolve`, no driver expressions, no `getattr` with an input-derived name
  except through `props.py`.
- **No file system:** no file paths in or out, no `image.save*`, `unpack`, `reload` or `filepath`
  writes, no `bpy.ops` that open, save, import or export.
- `bpy.ops` only as **fixed calls** (a literal operator with literal or enum-validated
  arguments), wrapped in `bpy.context.temp_override(...)`. Afterwards the user's selection, active
  object and mode must be exactly as before, **even if the call fails** (`try/finally`). Check the
  return value is `{'FINISHED'}`, otherwise raise `OpError`.
- Every input is validated by a strict `schema.py` spec. Every count or size that drives work
  (cuts, segments, steps, points, pixels, list lengths) has a hard cap, checked **before** doing the
  work; estimate the resulting geometry and reject it over the budget.
- Pointers between datablocks are set **only** by dedicated op parameters that look the target
  up by name in the right `bpy.data` collection and check its type, never via `set_property`.
- Results contain only plain data: names, numbers, indices. **Never file paths.**
- Mutating ops: `mutates=True`; ops that delete or irreversibly rewrite data also get
  `destructive=True`. Validate and raise `OpError` before changing anything, so a refusal leaves
  the scene untouched (the guard relies on this: it records no undo step for refusals).
- Keep the pure-Python modules free of `bpy` (see Gotchas).

Conventions: object params are `object` (one) or `objects` (list, 1–500); positions and vectors
are world space unless a `space: "world"|"local"` param says otherwise; angles are in degrees
(`*_deg`); ops that change geometry return the mesh summary (`"after": _summary(...)`); mesh edits
take a selector, not raw indices (indices change after every topology edit; `expect_counts` turns a
stale selection into `stale_indices`); every `description` tells an LLM when to use the op and what
the results mean. New `bpy` ops go in the matching `ops_*.py` module (they register on the same
`registry`). Target **Blender 5.1** (bundled Python 3.11, addon minimum 4.2): check `bmesh.ops` /
`bpy.ops` signatures in Blender (`help(...)`) instead of trusting memory.

Checklist for an op: schema + op function; a section in `tests/blender_smoke.py` (each new op
called through `registry.dispatch`, plus refusal cases with `expect`); a place in `GROUPS` in
`tests/gen_tools_md.py` and a regenerated `TOOLS.md`; pytest, pyflakes and py_compile pass;
rebuilt `blender_safe_mcp.zip`; README limitations and this file updated if behaviour or
architecture changed; `plugin/skills/blender/SKILL.md` updated if the op changes how an agent
should work (and after renaming an op it names).

## Ideas not built

- **Declarative Geometry Nodes op** (not scheduled). Procedural work (scatter, arrays along
  curves, terrain) needs many ops today; other Blender MCP servers solve it with
  `execute_blender_code`, which we refuse. Idea: one op taking a Geometry Nodes graph as JSON
  *data* (`nodes`, `links`, exposed inputs), built into a fresh node tree, with companions
  `get_geometry_nodes`, `set_geometry_nodes_input`, `remove_geometry_nodes`. Validation: strict
  schema (≤~200 nodes, ≤~400 links, scalar/vector/colour inputs only); a node **allow-list**
  (primitives, transform, join, set position, math, noise, distribute/instance points, realize
  instances, attribute capture/store…); explicitly **denied**: import nodes, anything referencing
  images or external datablocks, **bake nodes and baked simulation zones** (they write to a bake
  directory), node-tool nodes, foreign node groups, newer unreviewed constructs; sockets resolved by
  identifier against the created node, type-compatible links, no cycles; node properties set through
  `props.py`; static budgets (resolution, density × area, repeat iterations, instance counts)
  plus an **evaluation check** (force a depsgraph update, count evaluated verts/faces, remove the
  tree and refuse if over budget); atomic build. Main risk is denial of service; node catalogues
  drift per Blender version (the smoke test should assert allowed types exist and no denied type
  slipped in). Node names in the idea are illustrative and need verifying on 5.1.
- **Bake tool** is on hold (see `TODO.md`).
- **OS sandbox with a relay:** run Blender with `--unshare-net` and a small relay outside the jail
  forwarding raw TCP to a unix socket inside it; authentication stays end to end. **TLS** for
  non-loopback binds; **opaque handles** instead of names; **asset import/export** only as separate
  reviewed ops (fixed provider allow-list, glTF only, addon-chosen filenames).

## Architecture

**Request path:** MCP client → `POST /mcp` → `mcp_http.py` (stdlib `ThreadingHTTPServer`
threads: Host/Origin checks, `Authorization: Bearer <key>`, per-address back-off, JSON-RPC) →
`server.WorkQueue` → `pump()` on Blender's main thread (a `bpy.app.timers` timer in
`blender.py`) → `registry.dispatch` → `guard` → op function in `ops.py` / `ops_*.py`.

**MCP (`mcp_http.py`):** speaks the 2026-07-28 stateless era (`server/discover`, `_meta`
envelope + `MCP-Protocol-Version`/`Mcp-Method`/`Mcp-Name` headers, which must match the body)
and the 2025-03-26..2025-11-25 handshake era (`initialize`, `Mcp-Session-Id`). Single JSON
responses only (GET is 405). Ops' results become MCP content: the JSON as text, and
`{"$blob": i}` PNGs as image content (never file paths). Keep it free of `bpy` and of the MCP
SDK (Blender can't easily bundle pydantic); `tests/test_mcp_http.py` checks it with the official
SDK client, in both eras.

**Ops (`ops.py`, `ops_transform.py`, `ops_mesh.py`, `ops_mesh2.py`, `ops_curve.py`, `ops_texture.py`, `ops_image.py`):** the entire network-reachable surface. Each op is
`@op(name, Obj({...schema}), mutates=..., destructive=..., max_blobs=...)` (`destructive`
becomes the MCP `destructiveHint`). Rules:
- schemas come from `schema.py` and are strict (unknown keys rejected);
- objects are looked up by name in `bpy.data`, never through RNA path strings;
- generic property access goes through `props.py`;
- creation uses `bpy.data`/`bmesh`, with `bpy.ops` only through fixed calls (join, mode_set,
  undo/redo, convert, origin_set, transform_apply, UV unwrap, render).

`tools/list` is built from `list_ops` on every call (`ttlMs: 0`), so a new op needs no
registration elsewhere. It does need: a place in `GROUPS` in `tests/gen_tools_md.py` (the
generator fails on an ungrouped op), a regenerated `TOOLS.md`, a section in `tests/blender_smoke.py`,
and a rebuilt `blender_safe_mcp.zip`.

**`props.py`** walks RNA structurally: the path is a JSON list, and each segment must be a real
`bl_rna` property (not a Python attribute). Every struct reached must be in `ALLOWED_STRUCTS` and
not in `DENIED_STRUCTS`. Path-like subtypes are denied, only `WRITABLE_STRINGS` may be written,
pointers are never set, runtime (addon-defined) properties are denied, and `NUMERIC_CAPS` add
geometry/render limits. Tests use a fake RNA (`tests/test_props.py`).

**Guard (`blender.py:_guard`):**
- refuses everything while Blender's "Auto Run Python Scripts" preference is on;
- refuses mutating ops unless the user enabled "Allow scene changes" (off by default);
- pushes one undo step per mutating op, except for `undo`/`redo`, where a push would wipe the
  redo history;
- a tripwire disables the server if `bpy.data.texts` or `libraries` gain entries, or the image
  tripwire fires (see Images);
- `ALWAYS_ALLOWED` ops (`get_debug_logs`, `get_status`, `list_ops`) bypass these checks.

**Mesh editing ops** (`analyze_mesh`, `repair_mesh`, `merge_vertices`, `fill_hole`, …) use the
`_MeshEdit` context manager (bmesh load/edit/write back) and return a fresh `_analyze_bm` report.
Vertex indices change after any topology edit, so re-query before the next edit.

**Logging:** stdlib `logging` into a 10 MB in-memory ring (`logbuf.py`, UTC
timestamps), served by the `get_debug_logs` tool. Check it first when something misbehaves. Never
log the API key.

**API key:** stored in `$XDG_CONFIG_HOME/blender-safe-mcp/api_key` (mode 0600) by `keystore.py`, not in addon preferences, so it
survives reinstalls. Clients send it as `Authorization: Bearer <key>` (in `.mcp.json`
`headers`, or `${BLENDER_SAFE_MCP_KEY}` expanded from the environment).

**Images (`ops_image.py`):** uploads are decoded only by the bpy-free `imaging.decode_png`;
external (unpacked file) images are never written; everything created or written is `pack()`ed;
no path is ever set or returned. `blender.py:_image_violations` is the image tripwire.
`accept_images` / `max_image_pixels` / `max_request_mb` are addon prefs; `mcp_http.McpHttpServer.max_body_bytes`
is set from them (1 MB when `accept_images` is off) and checked after authentication.
`Image`/`ImageUser` stay out of `props.py`.

## Gotchas

- Reinstalling the extension re-executes only `__init__.py` (in the same package module object);
  it purges the `blender_safe_mcp.*` submodules from `sys.modules` **and** drops the package's submodule
  attributes, so the new code actually loads (a Blender restart clears it too). A traceback whose line numbers don't match the source is the symptom. Python logging handlers attached by an older
  version survive until Blender restarts.
- Reinstalling resets addon preferences, which turns "Allow scene changes" back off; a
  `read_only` refusal after a reinstall usually means that.
- `pyflakes` and pytest can't import `bpy`. Keep `mcp_http`, `schema`, `registry`, `server`,
  `props`, `logbuf`, `keystore`, `imaging`, `uvmath` and `falloff` free of `bpy` (`__init__.py` only imports
  `blender.py` when `bpy` is available). Code that needs `bpy` goes in `ops*.py`/`blender.py`
  and gets coverage in `blender_smoke.py`.
- The MCP SDK (2.x, `mcp>=2.3,<3`) is only a test dependency: its client drives the endpoint.
  `mcp.Client(..., mode="auto")` probes `server/discover`; `mode="legacy"` forces `initialize`;
  a version pin like `"2026-07-28"` skips discover (so no server info/instructions).

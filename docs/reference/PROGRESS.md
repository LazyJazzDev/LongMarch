# Reference implementation progress

Current task: replace the single architecture page with independent documents,
review every tracked file in `code/`, and provide module design/interface
standards plus object/API analysis. Do not call extracted signatures reviewed
or publish the draft as a completed reference.

- Inventory: 629 tracked files, 75 directories. Count source with `git ls-files`.
- Extraction: pinned tree-sitter 0.25.2 + tree-sitter-cpp 0.23.4. Version 0.26.0
  crashed on this machine, so keep the working pin. Parser diagnostics must be
  reviewed; C++ extensions and shaders require manual coverage checks.
- Reviewed and authored: 262 / 629 files. Contradium, Practium, Grassland util,
  BVH, math and physics are complete at the file/API-note level. Graphics public
  interfaces, window/input/HDR/profiling and root build/umbrella files are also
  covered. All Snowberg modules (GUI/surface, draw, visualizer and CUDA solver)
  are covered. Backend, Sparkium rendering and Python remain in progress.
- `files.json` records each file's SHA-256 and authored responsibilities.
- `api.json` records semantic descriptions by qualified name. Overloads share a
  description only when it explicitly accounts for their different behavior.
- `modules.json` records architecture, flow and interface contracts.
- `coverage.json` is generated; strict builds reject missing/stale file notes,
  missing objects or missing module designs.
- Static reference scaffold is implemented. Pages are generated under ignored
  `website/reference/`; this output is copied into the Pages publication branch
  when complete. The main workflow builds strictly and copies the reference;
  gh-pages publication copies the generated snapshot.
- Home navigation now links to independent reference documents; legacy fragment
  links redirect. Do not publish until strict checks and browser QA pass.

Authoring helpers used during this session are in /private/tmp:
`add_reference_notes.py`, `document-simulation.py`, `document-util.py`.
They write the tracked JSON sources; their presence is not required for builds.
`longmarch-extracted.json` contains a convenient full source/AST snapshot for
review. Refresh it after changing extraction. The virtualenv is
`/private/tmp/longmarch-doc-tools`.

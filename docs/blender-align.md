# Blender scenes with inline ray queries

Blender scene loading builds on the JSON loader and the shared path tracer. It
supports binary meshes, triangulated hair, material graphs, thin lenses,
background/view transforms, subsurface scattering, opacity, and extended light
settings. On Apple Silicon, Metal ray queries traverse native triangle BLAS/TLAS
through `InlineIntersect`; hit reconstruction and shading remain shared.

The asset submodule uses `LongMarchAssetsLFS` snapshot `07f7bd6`, including the
classroom lamp filename fixes. Fixture and emitter are separately stored as
`blackboard_lamp_fixture_m0.spmesh` and `blackboard_lamp_emitter_m0.spmesh` with
matching JSON references.

## Material compilation

Each shader graph supplies `GraphImpl()` separately from its full sampler. The
compute shader dispatches graph evaluation by material ID to produce a
`GraphSurface`, then calls one shared `SampleGraphSurface` implementation in
`material/shader_graph/surface_sampler.hlsli`. Shadow opacity also uses the graph
parameters. Native RT hit shaders retain a wrapper around the same sampler.
Scenes without graphs retain their original compact material dispatch.

This avoids duplicating the entire Principled BSDF and subsurface sampler per
material. With DXC `1.10(5180-e3554182)(1.9.0.5180)`, the previous combined shader
crashed in SPIR-V `AggressiveDCEPass` for Monster/Junkshop and exceeded SPIR-V IDs
for Classroom. All three complete material sets now compile and render on Metal
ray queries, without replacing or simplifying the scene materials.

Auto prefers a full native RT pipeline, then native ray queries, then software
fallback. Older Blender JSON files explicitly requesting `ray_tracing` also
select ray queries when the device lacks a full RT pipeline but supports queries.
An explicit `rt_fallback` request still selects software traversal.

## Local validation

Reconfigure CMake when adding the new shader file, then build the CLI, GUI, and
`sparkium_fallback_test`. The complete-scene check requires Pillow and the Blender
LFS assets:

```sh
python3 scripts/check_blender_ray_query.py \
  --cli build-metal/demo/sparkium_cli/demo_sparkium_cli \
  --output out/blender-ray-query
```

By default this renders Monster, Classroom, and Junkshop at 256×256, 32 samples
per frame, two frames, and 32 bounces, with Metal API and shader validation. It
checks that every frame reports `native_ray_query=1` and produces a nonconstant
image. Logs, profiles, images, and JSON results are retained in the output
folder. `--pipeline scene` tests the original JSON pipeline request without an
override; `--pipeline auto` tests automatic selection. Compilation, render,
timeout, missing-image, and silent-fallback failures return a nonzero exit code.

The GUI can load the original scene directly:

```sh
build-metal/demo/sparkium_gui/demo_sparkium_gui \
  assets/scenes/blender_monster/scene.json --backend metal
```

On Apple M5, all three complete scenes passed the CLI check above and two-frame
GUI runs at their original resolutions (1024×1024, 1920×1080, 2000×1000).
The GPU suite passed 18 tests and skipped one full RT pipeline comparison.
Nine earlier ray-query snapshots (the six basic demos, graph smoke, hair, and
subsurface/lens) passed a normalized PNG RMSE threshold of 0.03.

The Apple M5 validation uses macOS 26.6.2, Release builds, and Metal fast math.
The GPU suite also covers native query intersections against a double-precision
oracle, transparent shadows, background accumulation, Auto selection, and legacy
JSON requests. The full RT pipeline image comparison skips on this device.

Validation targets working Metal ray queries with the complete scenes. Software
fallback image parity/performance is not a release condition for this change.
Rendered PNGs and shader validation do not establish numerical equivalence with
Blender Cycles; no Blender reference-image comparison is claimed.

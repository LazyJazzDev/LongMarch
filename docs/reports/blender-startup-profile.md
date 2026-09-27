# Blender scene startup diagnosis

Measured on 2026-09-27 at `1b9fa41`, assets `3a7d62b`, Windows x64,
RTX 3090 Ti / driver 596.49, Ninja Release, Slang 2026.18.2.
The dominant delay is shader compilation during lazy first-frame initialization.
This report diagnoses opportunities; it does not implement startup optimizations.

## Measurements

Each scene ran in a fresh D3D12 CLI process for two frames, using native Ray Query,
the scene's default resolution and rendering settings, and CPU profiling.
Filesystem caches were not flushed. These are single-run measurements, not cold-disk
benchmarks or GUI responsiveness measurements. Raw timings and counts are in
[blender-startup-profile.csv](blender-startup-profile.csv).

| Seconds | Classroom | Junkshop | Monster |
|---|---:|---:|---:|
| JSON load and resource preparation | 1.16 | 12.94 | 2.48 |
| Materials and textures (within load) | 0.50 | 5.30 | 1.85 |
| Geometry preparation (within load) | 0.64 | 7.63 | 0.63 |
| First frame, including lazy initialization | 136.63 | 49.14 | 30.28 |
| Graph material RT hit-shader compilation | 58.91 | 40.16 | 22.22 |
| Geometry-light shader compilation | 70.54 | 3.73 | 3.78 |
| Required combined Ray Query renderer compilation/finalization | 3.47 | 2.96 | 2.27 |
| Initial BLAS build CPU scope | 0.21 | 0.23 | 0.17 |
| Entire process, including both frames, PNG and shutdown | 140.47 | 64.28 | 34.77 |

Timings are inclusive: do not add nested rows to their parent totals. Resource
preparation includes CPU processing and GPU uploads, not just disk reads. The BLAS
scope is CPU wall time, not a standalone GPU timing. Device initialization is
outside the startup scopes but included in the entire-process measurement.

Earlier uninstrumented Vulkan measurements recorded first frames of 141.36,
53.38 and 32.79 seconds respectively. The detailed new breakdown above is D3D12
only; Vulkan shares the initialization and compilation call sites, but its exact
per-stage savings have not been measured.

## Causes and priorities

1. **Defer native RT shader creation until that pipeline actually needs it.**
   `MaterialShaderGraph` compiles three hit-shader entries whenever the device
   supports RT, even when the selected renderer is Ray Query. There were
   54 / 35 / 19 graph materials and 162 / 105 / 57 such compilations respectively.
   Their 58.91 / 40.16 / 22.22 seconds do not serve the current compute renderer.
   Native RT initialization must remain available when switching pipelines.

2. **Share geometry-light shaders and compute programs by source variant.**
   `LightGeometryMaterial` compiles a power-gather kernel and an RT callable for
   each entity. Classroom created 928 components with just three unique
   geometry-sampler/material-evaluator source combinations, causing 1,856 shader
   compilations. The gather kernels took 32.03 seconds and callables 38.51 seconds.
   Junkshop had 48 components / three combinations; Monster had 49 / two.
   A per-Core cache could share the gather shader and program, retaining separate
   runtime buffers and transforms. The known mesh/material sampler IDs use inline
   sampling; Ray Query does not need these separately compiled RT callables.
   Unknown callable fallbacks must still work when native RT is selected.

   These two categories account for 94.7% of Classroom's first frame, 89.3% of
   Junkshop's and 85.9% of Monster's. This identifies removable or redundant work,
   **not a measured speedup**: required variants, program creation and other
   initialization will remain.

3. **Cache the shaders that are actually required.**
   `CompileShader` retains a thread-local Slang global session, but creates a new
   compilation session and loads, links and generates code on each call. There
   is no shader-bytecode cache here. After the first two fixes, the combined
   renderer's 2.3-3.5 seconds becomes more significant. Cache keys must include
   generated source, include dependencies, target, options and compiler version;
   a filename-only cache is incorrect. Backend pipeline caches are a separate
   layer. Parallel compilation should follow deduplication, with explicit thread
   safety for shared graphics resources.

4. **Optimize resource preparation, particularly Junkshop.**
   Its geometry preparation takes 7.63 seconds and material/texture loading 5.30
   seconds. Two hair geometries are expanded to triangle meshes on the CPU;
   preprocessing or caching those meshes is a candidate, but hair-only time was
   not isolated. Graph image-texture nodes load files individually. Classroom
   references 41 unique paths 57 times and Monster 20 paths 41 times, so a cache
   keyed by normalized path and decoding/format options could avoid duplicates.
   Junkshop's 145 graph texture references are all unique, limiting that benefit.

## Source trail and procedure

- `code/sparkium/pipelines/raytracing/core/scene.cpp`: `UpdatePipeline` lazily
  creates entity components during first-frame scene registration.
- `code/sparkium/pipelines/raytracing/material/material_shader_graph.cpp`:
  eager native hit shaders gated on device capability.
- `code/sparkium/pipelines/raytracing/light/light_geometry_material.cpp`:
  per-instance shader/program construction and fixed sampler IDs.
- `code/sparkium/shaders/direct_lighting.slang`: inline sampler dispatch and
  native-only callable fallback.
- `code/grassland/graphics/shader.cpp`: Slang compilation lifecycle.
- `code/sparkium/scene_io/json_scene.cpp` and
  `code/sparkium/geometry/geometry_hair.cpp`: resource loading and hair expansion.

Temporary CPU scopes covered JSON loading, material/texture and geometry phases,
component constructors, initial BLAS builds, and each `CompileShader` source/entry
pair. Counters recorded compilation calls and exact light source combinations.
The diagnostic executable was isolated under `out/blender-startup/bin`; original
source bytes were restored and the normal CLI rebuilt successfully afterward.
The diagnostic patch and runner remain locally under `out/blender-startup`.
No production rendering code was changed for this diagnosis.

# Experimental Metal ray query

Branch `metal-ray-query` starts at `rt-fallback` commit `58e9023`.
It adds a working native Metal acceleration-structure traversal path to the
existing compute path tracer. The six basic JSON demos share the same material,
light sampling, path loop, transparency, and film code with `rt_fallback`.
Blender material graphs and hair from `blender-align` are outside this branch.

## Selection and implementation

```sh
cmake-build-metal/demo/sparkium_cli/demo_sparkium_cli \
  assets/scenes/cornell_box/scene.json \
  --backend metal --pipeline ray_query -o out/cornell-ray-query.png
```

Scene JSON accepts `renderer.pipeline: "ray_query"`; the GUI exposes
**Path Tracing - Ray Query** when the backend supports it. `auto` selects pipeline RT first,
then native ray query, then software fallback. On supported Metal devices,
`auto` therefore uses native queries. Explicit `ray_tracing` and `rt_fallback`
retain their previous behavior. Vulkan and D3D12 query support is
not enabled by this experiment and requesting it is rejected.

The GUI displays the resolved pipeline in the Auto option and status, for
example `Auto (Path Tracing - Ray Query)`. Auto is the first option; pipeline RT and
native ray query options are shown only when their respective capabilities are
supported. Its separate Backend label shows the actual
graphics API, such as Metal. Both rendering and display use
`Core::ResolveRenderPipeline()` so that the reported choice matches execution.

`DeviceRayQuerySupport()` is separate from `DeviceRayTracingSupport()`, which
continues to describe pipeline RT/SBT support. Metal reports query support using
`supportsRaytracing`; this API capability is not a test for dedicated RT hardware
(M1/M2 also support the API). Do not pass `--require-hardware-rt` for this path:
that existing option checks pipeline RT and rejects Metal.

- The shared render HLSL uses `cs_6_5` and `SPARKIUM_RAY_QUERY`. DXC emits SPIR-V,
  and the existing SPIRV-Cross backend translates `RayQuery` into MSL 3.0
  `intersection_query`. The current compiler chain works without a new compiler
  or postprocessing generated MSL.
- `metal_acceleration_structure.h/.cpp` builds native triangle/AABB BLAS and instance
  TLAS with Metal command encoders. Geometry BLAS are cached; identical TLAS
  descriptors and child handles skip rebuilding. Changed instances rebuild the
  TLAS synchronously. Native AS memory is opaque and independent of the software
  BVH node buffers.
- AS resources use the existing compute argument buffers. Both TLAS and child
  BLAS are declared with `useResource`; each encoded dispatch retains its exact
  AS generation and dependencies through command completion.
- Instance user IDs map to the shared instance metadata index; primitive index,
  distance, and barycentrics feed the existing mesh hit reconstruction. Native
  instance flags, masks, transforms, and user IDs are mapped in the backend.
- Queries force triangle surfaces opaque for intersection, then shared shading
  evaluates material transmission. Transparent shadows keep repeated ordered
  closest-hit queries; only opaque occlusion uses accept-first-hit.
- The existing `SoftwarePipeline` class hosts both compute traversal modes.
  Query mode skips software BVH allocation/build shaders. Switching between the
  two modes recreates the compute pipeline and resets film accumulation.

## Validation

Tested on a 10-core Apple M5, macOS 26.6.2, Release build, Metal fast math enabled.
Build instructions and dependencies are in [metal-backend.md](metal-backend.md).
CMake must be reconfigured to discover the new source files before building:

```sh
cmake -S . -B cmake-build-metal
cmake --build cmake-build-metal \
  --target demo_sparkium_cli demo_sparkium_gui sparkium_fallback_test -j8
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cmake-build-metal/test/sparkium/sparkium_fallback_test
```

The final GPU test run passed 18 tests and skipped one pipeline RT test.
GPU tests cover CPU-oracle intersection comparisons for 1/5/257 triangles,
transforms, empty TLAS, accumulation/reset, mode switching, layered transparent
shadows, masks, IDs, back-face culling, unchanged-TLAS caching, and AS binding
snapshots across rebuild/destruction before submission. Existing Metal buffer,
texture, raster, and resource-array tests remain included. Pipeline RT image
parity is skipped because Metal does not expose that pipeline.
The basic JSON regression suite passed all 31 cases. JSON-selected `ray_query`
also rendered successfully without a CLI override, and an explicit Vulkan query
request was rejected with exit code 1. Logs are retained locally under
`out/metal-ray-query`.

An exact boundary difference was observed for rays starting on a triangle with
`TMin=0`: native query excluded a zero-distance hit accepted by the software
oracle. The surface-origin test now uses positive `TMin`, matching the renderer.
Neither exact boundary equivalence nor bitwise image equivalence is claimed.

Six-scene comparison command (requires Pillow):

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
python3 scripts/check_rt_fallback.py \
  --cli cmake-build-metal/demo/sparkium_cli/demo_sparkium_cli \
  --backend metal --pipeline ray_query --compare-pipeline rt_fallback \
  --spp 256 --bounces 32 --debug --max-rmse 0.03 \
  --output out/metal-ray-query/parity
```

All six 96×96 comparisons passed normalized PNG RGB RMSE ≤ 0.03.
Cornell, principled, and specular were identical; area light measured 0.008667,
point light 0.010687, and texture 0.017092. The contact sheet was also inspected.
This is a display-image regression check, not a raw HDR error bound or a proof
that all scenes behave identically. Images, logs, and `results.json` are in
`out/metal-ray-query/parity` (local, ignored build artifacts).

## Performance

Measured sequentially with the same Release binary and Metal fast math, without
API/shader validation. Each scene renders four frames, discards the first, and
reports the median of the remaining three. Each frame adds 32 spp with a
32-bounce limit; accumulation is not reset between measured frames. This is one
run per scene and traversal mode, so these are preliminary local measurements.

The metric is `cpu_ms/frame_wall`: wall time for scene update, rendering, GPU
completion, and film development. Asset loading and PNG writing are excluded.
It is not an isolated GPU intersection timing, and nested profile scopes must
not be added together. First-frame shader compilation and AS creation are
excluded from the steady-frame median.

| Scene | Resolution | Software (ms/frame) | Ray query (ms/frame) | Speedup |
| --- | --- | ---: | ---: | ---: |
| cornell_box | 1024×1024 | 1162.508 | 285.660 | 4.07× |
| area_light | 1024×1024 | 2421.199 | 104.553 | 23.16× |
| point_light | 1024×1024 | 2745.687 | 78.056 | 35.18× |
| principled | 1024×1024 | 1386.190 | 361.330 | 3.84× |
| specular | 1024×1024 | 1159.621 | 236.448 | 4.90× |
| texture | 2048×1024 | 13362.102 | 824.092 | 16.21× |

Native BLAS/TLAS builds occurred only in the first frame of all six static
scenes. Their combined first-frame CPU scope measured 3.70–8.66 ms; no native
AS rebuild was recorded in the three warmed frames.

Reproduce with the same command separately for `ray_query` and `rt_fallback`:

```sh
python3 scripts/profile_rt_fallback.py \
  --cli cmake-build-metal/demo/sparkium_cli/demo_sparkium_cli \
  --backend metal --pipeline ray_query \
  --size 0 --spp 32 --bounces 32 --frames 4 --warmup 1 --repeat 1 \
  --cpu-only --output out/metal-ray-query/profile-query
# Repeat with --pipeline rt_fallback and --output out/metal-ray-query/profile-software.
```

`--size 0` retains each scene's original dimensions. `--cpu-only` disables the
profiler's Vulkan timestamp queries; rendering still executes on the Metal GPU.
Per-frame CSVs, logs, and summary JSON are saved under those output directories.

## Limits and next experiment

AS builds currently wait for GPU completion; there is no asynchronous build,
refit, or compaction. Mesh geometry is assumed immutable, matching the current
mesh API. This experiment covers triangle meshes, not custom AABB intersection
functions, procedural geometry, pipeline RT/SBT, or Blender hair. Only M5 was
tested; other Apple GPU generations need their own compatibility/performance
checks. The linked DXC and SPIRV-Cross must support RayQuery translation.

Apple's [M3/A17 Pro GPU discussion](https://developer.apple.com/videos/play/tech-talks/111375/)
explains that intersection queries add scratch traffic and disable ray reorder
on that architecture; Apple recommends `intersector` for the best performance.
The measured query speedup is relative to this software BVH implementation,
not a comparison with native `intersector`, Vulkan RT, or DXR.

The next useful experiment is a native MSL `intersector` traversal path using
the same AS and shading semantics. Keep this query implementation as the image
and performance reference. A compiler-supported native helper or a separate
intersection kernel with GPU ray/hit queues is needed; unstructured replacement
of generated MSL would make the shader compiler integration fragile. A queue
design also adds memory traffic and dispatches, so measure it before committing
to a broader wavefront renderer rewrite.

## Procedural geometry demo

The graphics hello launcher also supports inline procedural queries:

```sh
cmake --build cmake-build-metal-only --target demo_graphics_hello
cmake-build-metal-only/demo/graphics_hello/demo_graphics_hello --module ray_query --backend metal
```

The demo pairs a triangle BLAS with an AABB BLAS. Bounding-box candidates are
intersected with an analytic sphere in HLSL and committed through
`CommitProceduralPrimitiveHit`. The sphere instance uses nonuniform scaling to
exercise object-space rays and inverse-transpose normals. This is inline query
support; Metal still does not implement the RT pipeline/intersection-shader/SBT
API used by `rt_multi_shader_group` and `external_shader`.

# Experimental realtime GI

`realtime` is a new, opt-in hybrid pipeline. It rasterizes primary visibility,
then shades a bounded subset of a lower-resolution lighting grid using the
existing **software BVH**, material graphs, BSDFs, light sampling and shadow
semantics. It does not require or select hardware ray queries. Both direct and
indirect lighting are reconstructed; this prototype is not a diffuse-only GI
pass added to the old raster renderer.

```sh
cmake --build cmake-build-metal-only --target demo_sparkium_gui demo_sparkium_cli
cmake-build-metal-only/demo/sparkium_gui/demo_sparkium_gui \
  assets/scenes/blender_classroom/scene.json --backend metal --pipeline realtime
```

The GUI also exposes the pipeline in its selector. `Camera` permits translation
and field-of-view changes without resetting realtime history. Existing HDR
preview, SDR transforms and dual PNG/HDR CLI export remain available.

## Budget and controls

The initial target is the Apple M5 (10 GPU cores, 32 GB), 1920x1080 output at
30 FPS. These settings bound work; they are not an adaptive frame-time controller
or a guarantee for every scene. JSON `renderer` accepts:

```json
{
  "pipeline": "realtime",
  "realtime_scale": 4,
  "realtime_bounces": 3,
  "realtime_updates": 16,
  "realtime_history": 64
}
```

- `realtime_scale`: lighting-grid divisor, 1–8. Visibility remains full resolution.
- `realtime_bounces`: maximum surface interactions, 1–8. The first is the raster
  hit; subsequent paths and shadow queries traverse the software BVH, including
  off-screen geometry. This is finite-bounce transport, not infinite-bounce GI.
- `realtime_updates`: update one of N interleaved lighting subsets per frame,
  1–16. At the defaults the grid is 480x270 and at most 8100 grid locations receive
  a new lighting sample per frame. Camera motion does not increase that budget.
- `realtime_history`: maximum accumulated lighting samples per grid location,
  1–64. Moving views cap blending history to 4 samples. At period 16, the history
  count is not the number of display frames or full-resolution spp.

`Samples / frame` and path-tracer `max_bounces` do not determine realtime work.
The GUI reports lighting frames separately from path-tracing spp. The renderer
retains per-sample `clamping` to suppress fireflies but does not apply the path
tracer's `max_exposure` accumulation clamp. Exposure and SDR tone mapping remain
output-only. Depth of field is currently disabled in realtime primary visibility.

## Data sharing and passes

1. Shared path-tracing scene registration provides mesh/material/light buffers
   and software BVH nodes. Static scene registration is reused; unchanged software
   TLAS instances no longer rebuild. Transform/active/light/basic-material changes
   trigger an update and invalidate the view's lighting history.
2. A single visibility draw uses shared geometry buffers and an instance-range
   table. It writes instance/primitive IDs and perspective-correct barycentrics,
   allowing the same hit-record and shader-graph functions as path tracing.
3. Full lighting-grid reprojection validates instance, world position and
   geometric normal. Invalid history is rejected, and the scheduled subset gets
   new software-traced lighting. The schedule advances independently of resets.
4. A geometry-guided low-resolution filter and guided full-resolution resolve
   fill the display image. History is kept unfiltered and in linear HDR.

View resources belong to the film, not a global singleton. Reset or resize
invalidates history. After modifying graph source, texture contents, or other
resources outside the tracked scalar material fields, call `Film::Reset()`;
resource revision tracking is not yet a general scene API.

This implementation is based on the current main-branch scene API; it does not
merge pending PR #46 or duplicate its proposed SceneDefinition/Renderer API.
The reusable boundary is shared scene registration, material evaluation and
software traversal; the view-specific passes are in `realtime_view.*` and
`shaders/realtime/`.

## Validation and limitations

The regression test covers HDR emitter radiance at a non-divisible resolution,
background disocclusion on a camera cut, explicit history reset after a material
edit, transform invalidation, and absence of native ray queries. Existing
path-tracing tests also cover the shared software BVH optimization.

Use the CLI profiler for render-and-develop wall time (Metal currently supports
CPU wall-time profiling here, not per-pass GPU timestamps):

```sh
demo_sparkium_cli assets/scenes/blender_classroom/scene.json \
  --backend metal --pipeline realtime --frames 320 -o classroom.png \
  --profile timings.csv --profile-cpu-only
```

Exclude scene loading, initial shader compilation/BVH construction and warmup
when assessing steady state, but report startup separately. CLI timing excludes
GUI composition and presentation, so it is not a substitute for GUI validation.

Initial measurements on the Apple M5, Metal, Classroom at 1920x1080:

| Run | Median frame time | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Static, 320 frames, history 16 | 22.38 ms | 26.31 ms | 56.19 ms |
| Moving camera, 320 frames, history 64 | 22.12 ms | 26.07 ms | 33.23 ms |

Both CLI runs excluded the first 20 frames and used scale 4, three interactions,
and update period 16. The moving-camera harness translated the camera's world X
coordinate by `0.15 * sin(frame * 2*pi / 240)` without resetting the film. These
are CPU render-and-develop wall times, not GPU timestamps. The static outlier
means this is not a strict per-frame 30 FPS guarantee. A GUI run at the default
settings showed 42.0 FPS / 23.79 ms at lighting frame 363, including presentation;
that screenshot is a point observation, not a GUI percentile benchmark.
Metal API/shader validation passed the realtime regression, and the shared
fallback suite passed 25 tests with three environment-dependent skips.

Known prototype limits:

- Lighting, material color and specular response are all evaluated on the coarse
  grid. Fine textures, thin objects and glossy reflections can blur or alias.
- Staggered updates can take up to one update period to cover newly visible
  surfaces. Large lighting/geometry changes reset history and temporarily expose
  noise. There is no world-space irradiance cache, distance-field accelerator,
  variance-guided denoiser, temporal antialiasing or adaptive ray scheduler yet.
- Raster visibility stores one surface layer. Alpha/transmission is evaluated by
  the shared path material and consumes the finite interaction budget; this is
  not a full layered transparency or hair solution. Large coordinate ranges and
  near-plane clipping also follow the raster projection rather than path tracing.
- Camera reprojection assumes static surfaces. Transform edits conservatively
  invalidate history rather than using per-object motion vectors. Graph/texture
  edits require the explicit reset noted above.
- The old `rasterization` remains available for comparison; `auto` continues to
  choose a path tracer. No scene file is silently migrated.

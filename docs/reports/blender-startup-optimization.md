# Blender scene startup optimization

This implements the primary fixes from the [startup diagnosis](blender-startup-profile.md).
Ray Query no longer pays for native RT hit shaders, and geometry lights share
their power-gather shaders and compute programs instead of compiling per entity.

## Implementation

- Graph hit shaders and entity hit groups initialize on first native RT use.
  Common native hit shaders, ray generation/miss shaders and the camera callable
  are also deferred. Switching from compute to native RT still initializes them.
- Geometry-light power programs are cached per ray-tracing Core using both exact
  geometry-sampler and material-evaluator source strings. The enclosing Core
  fixes the shader VFS, backend, compilation options, entry point and layout.
  Shader lifetime covers program lifetime. Per-entity buffers, transforms and
  dispatches remain separate, so changing one light does not change another.
- Built-in mesh lights and point lights use inline sampler IDs and no longer
  compile unused callables. Custom geometry/material callable fallback remains
  lazy for native RT.
- JSON loading shares images with the same canonical path within one loaded
  scene. Both principled materials and graph texture nodes use the same loader
  settings; per-node sampling/color-space behavior remains in shader code.
- Hair expansion reserves triangle indices and precomputes the radial circle
  once, preserving vertex/index order and formulas.

## Measurements

Windows x64, RTX 3090 Ti / driver 596.49, Ninja Release, Slang 2026.18.2,
assets `3a7d62b`, code based on `ceb4211`. Fresh processes, warm filesystem caches,
two frames per scene, native Ray Query, default scene settings. These are individual
runs, not a distribution or cold-disk benchmark. No persistent shader cache is used.

| First frame including initialization (seconds) | Before | After | Ratio |
|---|---:|---:|---:|
| Classroom / D3D12 | 136.63 | 5.54 | 24.7x |
| Junkshop / D3D12 | 49.14 | 3.48 | 14.1x |
| Monster / D3D12 | 30.28 | 3.31 | 9.2x |
| Classroom / Vulkan | 141.36 | 7.80 | 18.1x |
| Junkshop / Vulkan | 53.38 | 4.11 | 13.0x |
| Monster / Vulkan | 32.79 | 4.88 | 6.7x |

| Entire D3D12 CLI process (seconds) | Before | After |
|---|---:|---:|
| Classroom | 140.47 | 8.97 |
| Junkshop | 64.28 | 18.32 |
| Monster | 34.77 | 6.45 |

Entire-process times include device/resource initialization, two frames, PNG
export and shutdown; they are not GUI time-to-first-pixel measurements. Optimized
Vulkan entire-process times were 10.84 / 18.62 / 8.00 seconds respectively.
Before D3D12 data comes from the diagnostic runs at `1b9fa41`; the documentation-only
`ceb4211` does not change rendering. Before Vulkan first-frame data comes from the
earlier 60-frame benchmark's frame zero, with the same renderer/toolchain and
byte-identical scene contents. No Vulkan entire-process ratio is claimed.

Required combined-renderer compilation/finalization now takes 4.30 / 2.98 / 3.00
seconds on D3D12 and accounts for much of the remaining first-frame time.
Junkshop still has substantial resource preparation time. The hair micro-optimization
is not claimed to eliminate that bottleneck: its individual savings were not isolated.
General persistent bytecode caching, asynchronous loading and baked hair assets
remain separate opportunities, beyond the redundant initialization fixed here.

Raw optimized frame timings and prior Vulkan frame-zero data are in
[blender-startup-optimization.csv](blender-startup-optimization.csv). Prior D3D12
startup and frame data are in [blender-startup-profile.csv](blender-startup-profile.csv).

The command for each optimized run was:

```text
demo_sparkium_cli assets/scenes/blender_<scene>/scene.json --backend <backend> --require-hardware-rt --frames 2 --profile-cpu-only --profile <csv> -o <png>
```

## Validation

CLI, GUI and fallback tests build with Ninja Release. The three D3D12 PNGs match
the pre-change two-frame images pixel-for-pixel, at 1920x1080 / 2000x1000 /
1024x1024 respectively, with scene-default eight samples per dispatch.
Both backends complete all three scenes. This validates startup and two-frame
rendering, not a long-run throughput benchmark.

Regression tests exercise graph-material Ray Query/native RT switching in both
directions and image equality, along with source-sensitive geometry-light program
sharing. Existing tests cover light parameter changes, transforms, visibility,
fallback traversal and native RT parity.

Full regression runs with D3D12 debug and Vulkan synchronization validation each
passed 36 tests; two unsupported-HDR cases were skipped because the test display
supports HDR. Vulkan reported no validation errors; an unused raster vertex-output
warning remains. Recovery tests intentionally log their injected presentation failures.
Repository pre-commit checks passed.

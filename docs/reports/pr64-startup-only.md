# Lazy pipeline initialization (PR64)

Ray Query and software tracing no longer initialize the native RT raygen, miss,
camera-callable or material hit shaders they do not use. Native RT initializes
these shaders and hit groups on demand, including when switching pipelines after
the first frame. Unused point-light callable initialization is removed. Geometry
light power programs with identical geometry/material source are shared within
one Core, with different source pairs kept separate.

This PR is limited to pipeline and shader initialization. Upload APIs, resource
ownership, material upload caching, texture loading and AS update scheduling use
the main-branch implementation. Those investigations are preserved in the separate
Draft resource-update PR. No steady-state upload improvement is claimed here.

## Validation of the split implementation

Ninja Release builds GUI, CLI and fallback tests. D3D12 debug and Vulkan
synchronization-validation full suites cover Ray Query/native RT switching and
source-keyed program sharing. Classroom renders use RTX 3090 Ti, original
1920 x 1080 scene settings, native Ray Query, 8 samples per dispatch, two frames.
The following single-run timings are diagnostic observations, not controlled
before/after speedup measurements or cold-cache benchmarks.

| Backend | Passed tests | Skipped tests | First frame (s) | CLI process (s) |
| --- | ---: | ---: | ---: | ---: |
| d3d12 | 32 | 4 | 4.563 | 8.427 |
| vulkan | 32 | 4 | 5.482 | 8.804 |

Four existing HDR-environment cases skip in each suite. Existing unused shader
output warnings and deliberately injected presentation failures remain. Metal is
not built or run in this Windows validation. Pre-commit passes.

- d3d12: output is 1920 x 1080; pixel equality with the earlier managed-AS two-frame output: **True**.
- vulkan: output is 1920 x 1080; pixel equality with the earlier managed-AS two-frame output: **True**.

![D3D12 Classroom, two frames at 8 samples per dispatch](../../assets/reports/pr64-startup-only/classroom-d3d12.png)

![Vulkan Classroom, two frames at 8 samples per dispatch](../../assets/reports/pr64-startup-only/classroom-vulkan.png)

These are actual renders from the isolated initialization changes, not the previous
upload-throughput chart. Shader-graph regression also verifies that hit shaders
remain absent in Ray Query, appear on switching to native RT, and rendering stays
equivalent across repeated switches.

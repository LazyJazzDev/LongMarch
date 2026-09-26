# Classroom Nsight profiling

Follow-up: [implemented scene-update optimization and measurements](nsight-scene-update-optimization.md).

Profiled on 2026-09-26 at LongMarch `72f475c`, Windows x64, Ninja Release,
RTX 3090 Ti / driver 596.49, Nsight Systems 2025.5.1 and Nsight Graphics 2025.5.0.
Classroom uses native hardware ray query, 1920 x 1080, 8 samples/dispatch,
16 maximum bounces, with the original scene/camera/settings.

## Finding

The largest demonstrated cause of low end-to-end primary Rays/s is the repeated
scene-maintenance and synchronization path. The static Classroom scene has 928
rendered instances, compared with 48 in Junkshop and 49 in Monster. Every frame
re-registers entities/materials/lights, submits thousands of small synchronous
uploads, reconstructs compute-renderer metadata and updates the native TLAS.
This leaves large gaps between GPU command-list executions.

This is not evidence that the main ray-query shader is cheap: it still takes
about 108 ms on the profiled D3D12 workload. Hardware-counter access is blocked,
so register occupancy, shader stalls, cache efficiency and traversal-versus-BSDF
cost have not been measured. Do not interpret queue activity as SM occupancy.

## Nsight evidence

Nsight Systems successfully captured the Classroom D3D12 API and GPU timeline
for eight seconds, after a 180-second delay to exclude initialization. The
program's per-frame CSV confirms that rendering was already past warmup.
There are 22 complete intervals between successive long GPU rendering tasks.
The long task matches the program's approximately 108 ms render-wait interval.

| Quantity | Mean per complete interval |
| --- | ---: |
| Main-render start to next main-render start | 342.271 ms |
| Recorded GPU command-list execution, interval union | 115.686 ms |
| Gaps without recorded command-list execution | 226.585 ms |
| Main rendering GPU task | 107.682 ms |
| Other recorded GPU work | 8.004 ms |
| Recorded queue-active fraction | 33.80% |
| GPU command-list submissions | 2,349 |
| CopyBufferRegion calls | 2,345 |
| Dispatch calls | 2,273 |

Counts are identical across the 22 analyzed intervals. Thousands of dispatches
include geometry-light power evaluation and prefix scans, not thousands of
full-image path traces. GPU work is dominated by the one long render task;
the small transfers mostly cost submission/synchronization time between tasks.

These are instrumented timings. The trace diagnostic contains "Not all DX12
events might have been collected"; quantities above describe the recorded
command stream and consistent complete intervals, not a completeness guarantee
or a hardware-wide utilization measurement. Other processes/queues are not
included. Trace-state resource-creation metadata outside the measured intervals
must not be counted as per-frame allocations.

## Corroboration from uninstrumented benchmark profiles

The previous CPU-only frame profiles at `d13974a` (same rendering code) give:

| Scene | Backend | Frame ms | Scene update ms | Registration ms | Compute preparation ms | Instances |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classroom | D3D12 | 314.446 | 197.505 | 153.786 | 39.386 | 928 |
| Classroom | Vulkan | 368.367 | 209.327 | 156.540 | 39.927 | 928 |
| Junkshop | D3D12 | 67.291 | 12.819 | 10.435 | 1.778 | 48 |
| Junkshop | Vulkan | 86.432 | 15.069 | 11.588 | 1.884 | 48 |
| Monster | D3D12 | 65.076 | 11.814 | 8.942 | 2.290 | 49 |
| Monster | Vulkan | 89.551 | 14.105 | 10.223 | 2.494 | 49 |

Registration and compute preparation are nested inside scene update; do not add
these columns together. CPU scope times include synchronous GPU/driver waits,
not just CPU execution. Scene update consumes about 63% of the uninstrumented
Classroom D3D12 frame and 57% of its Vulkan frame. In Vulkan, static uploads alone
account for approximately 150.085 ms/frame: 2,345 calls transferring 250,720 bytes
(244.84 KiB, about 107 bytes/call). This points to repeated calls and synchronization,
not evidence of bulk upload bandwidth saturation.

Rays/s is primary camera samples divided by total frame time, including scene
updates. It is not total hardware traversal throughput or a pure shader benchmark.
The previous [throughput comparison](slang-blender-performance.md) therefore
includes this substantial fixed per-frame cost. No new speedup is claimed here.

## Source paths responsible

- `Scene::UpdatePipeline` in `code/sparkium/pipelines/raytracing/core/scene.cpp`
  clears/reconstructs resource and instance lists and updates every active entity
  on every frame, even when the loaded scene is unchanged.
- `EntityGeometryMaterial::Update` calls material update and registers its geometry
  light for every instance. `MaterialShaderGraph::Update` uploads texture indices;
  shared materials can be updated repeatedly through their instances.
- `LightGeometryMaterial::SamplerData` uploads its transform on every access;
  `Scene::RegisterLight` records power/preprocessing dispatches for each light.
- Both backend static-buffer `UploadData` implementations wait for GPU completion
  and perform a separate single-time transfer. D3D12 `SingleTimeCommand` executes,
  signals and waits on a fence; Vulkan's path submits and calls `vkQueueWaitIdle`.
- `SoftwarePipeline::Update` reconstructs material source descriptions and instance
  metadata, uploads the instance data and calls `native_tlas_->UpdateInstances`
  each frame. Even the unchanged-instance path refits/updates the TLAS.

## Recommended optimization order

1. Retain scene/resource registration and stable resource indices across frames;
   invalidate them on entity, material, texture, transform or visibility changes.
   Update each shared material once, only when its data or resource mapping changes.
2. Cache unchanged light transforms/power distributions and instance metadata;
   skip TLAS updates when instance transforms/topology are unchanged. Preserve
   correct invalidation for dynamic scenes and emissive/material changes.
3. Batch small buffer copies into one upload/command submission with a frame-level
   synchronization point, or use appropriate persistently mapped/ring-buffer
   storage, instead of synchronously waiting per tiny static-buffer update.
4. Cache generated material descriptions and avoid reconstructing/comparing shader
   source per instance per frame. Then reprofile the remaining main GPU task with
   Nsight Graphics counters to decide between traversal, shading and memory work.

This task changes no rendering behavior; these are profiling findings and proposed
follow-up work, not an implemented optimization or a promised speedup.

## Capture availability and limitations

- Nsight Graphics injected successfully but stopped with: `GPU Performance Counters
  unavailable. Please enable access to GPU performance counters.` Counter access
  needs enabling in NVIDIA Control Panel before shader/SM counter analysis.
- Nsight Systems Vulkan injected but reported `No Vulkan events collected` and
  exported no Vulkan API table. That report is not used as performance evidence.
- A short D3D12 cube probe verified API/GPU collection before the Classroom capture.
  The Classroom D3D12 report is the usable Nsight evidence above.
- CPU sampling/context-switch tracing were disabled because the session is not
  elevated. CPU function attribution comes from existing application scopes and
  source inspection, not sampled stacks.

Local artifacts (not tracked):

- `out/nsight-classroom/systems-classroom-d3d12.nsys-rep`: open in Nsight Systems.
- `out/nsight-classroom/classroom-d3d12.sqlite`: exported timeline.
- `out/nsight-classroom/trace-analysis.json`: per-interval measurements.
- `out/nsight-classroom/nsys-d3d12.csv`: application frame scopes.
- `out/nsight-classroom/analyze.py`: interval analysis script.
- `out/nsight-classroom/*launch.log`: capture diagnostics, including failed paths.

Reproduction, using the installed Nsight Systems executable:

```powershell
& 'C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.1/target-windows-x64/nsys.exe' `
  profile --trace=dx12 --sample=none --cpuctxsw=none --delay=180 --duration=8 `
  --kill=true --output=out/nsight-classroom/systems-classroom-d3d12 `
  cmake-build-ninja/demo/sparkium_cli/demo_sparkium_cli.exe `
  assets/scenes/blender_classroom/scene.json --backend d3d12 --require-hardware-rt `
  --frames 600 --profile-cpu-only --profile out/nsight-classroom/nsys-d3d12.csv `
  -o out/nsight-classroom/nsys-d3d12.png
```

The profiler deliberately terminates its launched process after collection;
the final PNG is therefore not expected. Verify warmup completion for the chosen
delay on each machine before treating the capture as steady state.

# Batched Sparkium data updates

`sparkium::Core` owns one `DataUpdateTracker`. Material-local offset caches and
geometry-light transform caches have been removed. The tracker covers CPU uploads
for both raster and ray-tracing pipelines: geometry, materials, entities, cameras,
scene metadata, compute build parameters, film development and loaded textures.

## Resource and update lifecycle

`Core::CreateBuffer` and `Core::CreateImage` register each resource when created.
Pipeline-specific Core objects forward these factories to the owning Sparkium Core.
The JSON image decoder registers textures and queues their pixels before freeing
the decoded data. Graphics clients outside Sparkium retain immediate image loading
unless they explicitly supply the new upload callback.

`GetDataUpdateTracker().Update(resource, data, ...)` copies changed data into an
owned CPU snapshot. Each resource has a single snapshot, initialized-byte mask
and dirty interval list. Overlapping updates have last-write-wins semantics;
adjacent/overlapping dirty intervals merge before submission. No comparison reads
GPU memory. CPU ranges are copied exactly, preserving GPU-written CDFs and other
disjoint resource regions. Images support full updates and rectangular patches.

The caller can immediately release or modify its input data. A weak lifetime
identity distinguishes an expired resource from a new allocation at the same
address. Unsubmitted updates to destroyed resources are discarded on flushing;
the registry does not keep GPU resources alive.

## Submission and ordering

Raster drawing and ray-tracing preprocessing flush after collecting their CPU
updates, before the GPU reads them. A flush records all pending buffer/image
transfers into one CommandContext and submits it once. Empty batches submit
nothing. Film development uses the same mechanism before its compute dispatch.

Native BLAS construction is an earlier GPU consumer. It drains the pending batch
only when its geometry input is still pending. Subsequent BLAS builds reuse the
already uploaded geometry; ordinary frames do not eagerly flush texture edits at
the start of Render. First-time geometry preparation can therefore require an
earlier batch in addition to the normal frame batch.

`CmdUploadBuffer` and `CmdUploadImage` copy source bytes into owned staging
allocations, record backend copy commands and retain staging storage until GPU
completion through post-execution callbacks. D3D12 resource states, Vulkan image
layouts and transfer dependencies, and Metal blit encoding are handled by the
backend. The tracker itself issues no `WaitGPU`; existing queue/frame throttling
and render completion waits still apply. One upload batch does not mean that the
entire frame, acceleration-structure construction and presentation use one submit.

## API contract

- Calls belong on the render thread, like the graphics Core.
- Sparkium's tracked buffers use static GPU storage, including the formerly
  dynamic raster parameter buffers. This avoids frame-slot-dependent storage
  when CPU updates are deduplicated. Explicit registration requires static buffers.
- External resources must be registered before calling `Update`. Direct graphics
  `UploadData` remains available outside this managed upload path.
- Before direct GPU writes or resizing invalidate a tracked CPU range, use
  `Invalidate` and supply the new CPU data when needed. A changed buffer size is
  also detected on the next update. GPU writes to disjoint, GPU-owned regions need
  no invalidation. This tracker does not rebuild acceleration structures when
  geometry topology changes.
- Immediate CPU readback outside rendering requires `Flush` first; graphics
  readback performs the necessary completion wait. Resources must remain alive
  until submitted commands finish, as for other graphics commands.
- Snapshots consume host memory proportional to the initialized extent of CPU
  data, plus an initialized-byte mask. GPU-only resources allocate no snapshot.
  This is the cost of exact comparisons without relying on hash equality.

## Validation

Regression coverage includes coalesced buffer/image updates in one batch,
overlapping writes and source-data ownership, unchanged-data suppression, HDR
pixels and partial image regions, preserving a GPU-written buffer tail, destroyed
resources, replacement/resizing, material edits, texture-index changes, light
transforms, restoring image pixels after Film reset, and switching between Ray
Query and native RT. Film reset and render output invalidate the raw-film snapshot;
development invalidates the destination image snapshot before GPU writes.

The implementation includes D3D12, Vulkan and Metal upload commands. Local runtime
validation is on Windows D3D12/Vulkan; Metal requires verification on macOS.

On the RTX 3090 Ti / driver 596.49 machine used in the
[startup report](blender-startup-optimization.md), Ninja Release CLI, GUI and
fallback-test builds pass. Debug D3D12 and Vulkan synchronization-validation runs
each pass 39 tests, with two unsupported-HDR cases skipped because the display
supports HDR. The final three tracker-focused cases also pass on both backends.
There are no validation errors; the existing unused raster vertex-output warning
and intentionally injected presentation recovery errors remain.

All six Blender scene/backend pairs run for two frames at the original scene
resolution and eight samples per dispatch. Each PNG is pixel-identical to the
corresponding two-frame image from before this tracker change. For every pair,
the second frame records **one update batch, one buffer copy, 32 bytes** (the
film sampling information). Unchanged material, camera, light, instance and scene
metadata require no copies. First-frame initialization is excluded from this
statement. These runs validate the upload schedule and image equivalence, not
steady-state Rays/s gains or a startup speedup.

[Recorded frame timings and counters](data-update-tracker-frames.csv) preserve both
frames from each run. Reproduction uses the CLI scene command in the startup report;
inspect `data_update_batches`, `data_update_copies` and `data_update_bytes` in the
CPU-profile CSV. The PNG comparison artifacts and full validation logs remain
locally under `out/blender-startup/tracker*`.

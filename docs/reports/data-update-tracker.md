# Batched Sparkium data updates

`sparkium::Core` owns one `DataUpdateTracker`. Material-local offset caches and
geometry-light transform caches have been removed. The tracker covers CPU uploads
for both raster and ray-tracing pipelines: geometry, materials, entities, cameras,
scene metadata, compute build parameters, film development and loaded textures.
It also owns typed registries for Sparkium BLAS/TLAS wrappers and schedules their
deferred builds after uploads.

## Resource and update lifecycle

`Core::CreateBuffer` and `Core::CreateImage` return owning `sparkium::Buffer`
and `sparkium::Image` wrappers. Pipeline-specific Core objects forward these
factories to the owning Sparkium Core. The wrappers own the native graphics
resource and its CPU update state: snapshot, initialized-byte mask, dirty intervals
and tracker registration. They use composition; `Get()` supplies the unchanged
native backend object to graphics commands. `graphics::Buffer` and
`graphics::Image` no longer contain `Lifetime()` or a shared lifetime token.

Wrappers register at construction and unregister before destroying their native
resource. Destroying an owner immediately removes its pending updates and releases
its snapshot, rather than waiting for the next flush. Moving the owning unique_ptr
does not move the wrapper or change registration. Tracker destruction detaches
remaining owners; subsequent updates through them throw, and their destruction
is safe as long as the graphics Core still lives. Resources already submitted to
the GPU must still outlive execution.

`buffer->Update(data, size, offset)` and `image->Update(data, ...)` compare bytes
locally and copy changes into the wrapper's owned snapshot. Overlapping updates
have last-write-wins semantics; adjacent/overlapping dirty intervals merge before
submission. No comparison reads GPU memory. CPU ranges are copied exactly,
preserving GPU-written CDFs and other disjoint resource regions. Images support
full updates and rectangular patches. Callers can immediately reuse their data.

`Core::LoadImageFromFile` adopts the decoded native image into an owning wrapper
and records pixels before the decoder releases its CPU data. JSON scenes retain
these wrappers. Graphics clients outside Sparkium keep immediate image loading.
The tracker retains raw-address lookup for APIs exposing native resources, but
registration/unregistration are private to the owning wrappers. The tracker keeps
separate typed Buffer and Image registries and records their uploads in separate
loops into the same command context. DataResource contains only common state:
tracker association, byte capacity, snapshot, initialized-byte mask and dirty
intervals. It has no native Buffer/Image pointers or resource-type checks; concrete
wrappers handle registration, destruction and resource-specific invalidation.
A newly allocated
resource cannot inherit the previous owner's registration or CPU cache.

## Submission and ordering

Raster drawing and ray-tracing preprocessing flush after collecting their CPU
updates, before the GPU reads them. A flush records all pending buffer/image
transfers into one CommandContext and submits it once. Empty batches submit
nothing. Film development uses the same mechanism before its compute dispatch.

BLAS/TLAS use independent owning wrappers and typed registries alongside Buffer
and Image. Creating a wrapper registers a pending build without creating the
native AS; `Get()` returns null until the first successful flush. Scene registration
stores instances referencing the BLAS wrappers rather than native GPU objects.
Both native RT and Ray Query use this path.

After submitting Buffer/Image uploads, Flush builds changed BLAS resources, then
creates or updates dependent TLAS resources. Buffer revisions detect changes to
BLAS inputs, and successful BLAS rebuild generations invalidate dependent TLAS.
TLAS instance comparisons include BLAS identity, transform, ID, mask, hit-group
offset and flags. Unchanged instances and unchanged BLAS generations skip the
native TLAS update entirely, including its submission and synchronization.

The managed BLAS factory covers Sparkium's indexed triangle geometry. Descriptors
retain buffer ranges, counts, stride and flags; changing those descriptors requires
replacing the wrapper, while changed vertex/index data rebuilds the existing
wrapper. Input ranges are checked before invoking the backend. Cross-tracker
BLAS dependencies are rejected. Destruction unregisters pending builds; destroying
an input invalidates dependent wrappers, cancels their builds, and makes Get reject
stale resources without blocking unrelated uploads. Tracker destruction safely
detaches surviving owners.

Native AS creation/update still uses the existing synchronous graphics interfaces.
This change centralizes ordering and suppresses redundant builds; it does not
encode multiple AS builds into the upload CommandContext or eliminate the backend's
initial build waits. Counters `data_update_blas_builds`, `data_update_tlas_builds`
and `data_update_tlas_updates` record actual managed backend calls.

`CmdUploadBuffer` and `CmdUploadImage` copy source bytes into owned staging
allocations, record backend copy commands and retain staging storage until GPU
completion through post-execution callbacks. D3D12 resource states, Vulkan image
layouts and transfer dependencies, and Metal blit encoding are handled by the
backend. The tracker itself issues no `WaitGPU`; synchronous AS build calls, queue/frame throttling
and render completion waits still apply. One upload batch does not mean that the
entire frame, acceleration-structure construction and presentation use one submit.

## API contract

- Calls belong on the render thread, like the graphics Core.
- Sparkium's tracked buffers use static GPU storage, including the formerly
  dynamic raster parameter buffers. This avoids frame-slot-dependent storage
  when CPU updates are deduplicated. The owning wrapper requires static buffers.
- External resources enter this path by transferring a native unique_ptr into a
  Sparkium Buffer/Image wrapper. Bare graphics resources cannot register directly.
  Direct graphics `UploadData` remains available outside this managed upload path.
  Film development invalidates managed output images when present, without
  registering externally owned presentation images.
- Wrapper `Buffer::Resize` automatically invalidates its snapshot. After direct
  writes or resizing through the native resource, use `Invalidate` and supply new
  CPU data when needed. A changed buffer size is also detected on the next update.
  GPU writes to disjoint, GPU-owned regions need
  no invalidation. Managed BLAS input changes trigger rebuilds and invalidate
  dependent TLAS; changing geometry ranges/counts requires a new BLAS wrapper.
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
resources, replacement/resizing, tracker-before-owner destruction, ownership
transfer and rejection of unowned resources, material edits, texture-index changes,
light transforms, restoring image pixels after Film reset, and switching between Ray
Query and native RT. Film reset and render output invalidate the raw-film snapshot;
development invalidates the destination image snapshot before GPU writes.

The implementation includes D3D12, Vulkan and Metal upload commands. Local runtime
validation is on Windows D3D12/Vulkan; Metal requires verification on macOS.

On the RTX 3090 Ti / driver 596.49 machine used in the
[startup report](blender-startup-optimization.md), Ninja Release CLI, GUI and
fallback-test builds pass. Debug D3D12 and Vulkan synchronization-validation runs
each pass 41 tests, with four HDR-dependent cases skipped. D3D12 exposes HDR,
so its SDR-only case skips; Vulkan exposes no HDR surface in this run, so its
HDR recovery, presentation/ImGui and reference-white cases skip. Its SDR fallback
case passes. The earlier owning-wrapper run passed 41 tests with two skips.
All seven tracker-focused cases cover upload ownership, both destruction orders,
BLAS/TLAS deferred construction, unchanged-instance suppression, geometry changes,
instance changes, dependency destruction and cross-tracker rejection.
There are no validation errors; the existing unused raster vertex-output warning
and intentionally injected presentation recovery errors remain.

The Windows builds, full regression suites and all six Blender scene/backend
pairs were repeated after adding managed BLAS/TLAS. Each scene runs for two frames
at the original scene resolution and eight samples per dispatch. Each PNG is pixel-identical to the
corresponding two-frame image from before this tracker change. For every pair,
the second frame records **one update batch, one buffer copy, 32 bytes** (the
film sampling information). Unchanged material, camera, light, instance and scene
metadata require no copies. The second frame also performs no managed BLAS builds,
TLAS builds or TLAS updates. First-frame initialization is excluded from this
statement. These runs validate the upload schedule and image equivalence, not
steady-state Rays/s gains or a startup speedup.

[Recorded frame timings and counters](data-update-tracker-frames.csv) preserve both
frames from each run. Reproduction uses the CLI scene command in the startup report;
inspect `data_update_batches`, `data_update_copies` and `data_update_bytes` in the
CPU-profile CSV. The PNG comparison artifacts and full validation logs remain
locally under `out/blender-startup/managed-as/` and `out/managed-as-*`.

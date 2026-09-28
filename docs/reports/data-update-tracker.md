# Batched Sparkium upload tasks

`DataUpdateTracker` schedules explicit Buffer/Image uploads in one batch, followed
by dependent BLAS/TLAS builds. It does not maintain resource-wide CPU snapshots,
compare new data with previous frames, or decide whether application data changed.

## Resource ownership and queued data

Buffer and Image are independent owning wrappers in `buffer.h/.cpp` and
`image.h/.cpp`. `Get()` exposes the original graphics resource. Constructors
register the wrappers; destructors unregister them and cancel pending tasks.
Moving an owning unique_ptr preserves registration. Tracker destruction detaches
surviving wrappers and releases their pending data; subsequent updates throw.
Registration rejects foreign, duplicate or detached owners.

Every nonempty `Update` enqueues an explicit upload request and copies only the
supplied data, so callers can immediately reuse or destroy the source:

- Buffer tasks own an offset and exactly the supplied bytes. A four-byte write
  near the end of a large buffer retains four bytes, not a copy of its prefix.
- Image tasks own a 2D offset, extent and tightly packed pixels for that rectangle.
  A one-pixel update retains one pixel, not a full-image allocation or row fragments.
- No snapshots, initialized-byte masks, known-region sets or cross-frame content
  comparisons are retained. A repeated explicit update is uploaded again even if
  its contents match a previous request. Change detection belongs to callers.
- Requests for each resource execute in call order, preserving last-write-wins
  behavior for overlaps. Each image rectangle produces one upload command. Tasks
  are not merged into bounding boxes, so holes and other GPU-written areas remain
  untouched. Adjacent requests remain separate commands within the same batch.

`PendingUploadBytes()` reports the payload currently awaiting submission, excluding
backend staging. After successful submission, acknowledgement destroys both task
payloads and the task vector allocation. Cancellation, resource destruction and
tracker destruction also release pending data. There is no persistent comparison
copy for static textures or buffers. An initial whole-image upload temporarily
owns that image's supplied pixels until submission, as any deferred upload must.

## Submission and cancellation

`Flush()` records pending tasks into one graphics CommandContext and submits it
once. Empty batches submit nothing. `CmdUploadBuffer` and `CmdUploadImage` copy the
source into backend-owned staging while recording; task data is released after
successful submission, while staging remains alive until GPU execution completes.
Failed submissions keep queued tasks available for retry. Resources must outlive
GPU execution, and immediate readback requires Flush first.

`Invalidate()` cancels pending uploads and advances the resource revision; it does
not discard a comparison snapshot because none exists. Buffer resizing cancels
obsolete requests. Explicit invalidation after external geometry writes also
notifies dependent AS builds. Film reset/output cancels obsolete writes before
replacing a target's contents. A normal GPU write no longer requires invalidation
merely to make a later identical `Update` upload again.

## Acceleration structures and encapsulation

After uploads, Flush builds changed BLAS resources and then dependent TLAS
resources. Every explicit Buffer update advances its revision, even when its bytes
match a previous upload. BLAS rebuilds when an input revision changes; successful
build generations drive TLAS updates. Repeated Flush with no new tasks or instance
changes skips AS work. TLAS retains its required instance descriptors and compares
those descriptors; it does not mirror geometry-buffer contents. Backend AS
creation remains synchronous rather than being encoded into the upload batch.

Destroyed inputs invalidate dependent wrappers. Cross-tracker dependencies are
rejected. All ten original class-wide friendships are removed; ten narrowly
scoped friendships remain for the relevant Flush/Unregister members across Buffer,
Image, BLAS and TLAS. No mutable task lists or registries are exposed.

## Validation and historical measurements

Regression tests cover owned source data, ordered overlapping writes, repeated
explicit requests, tall 2D rectangles, GPU-owned holes, bounds, cancellation,
resizing, both destruction orders, registration restrictions, deferred AS builds,
material/texture edits, Film reset and native RT/Ray Query switching.

The memory-lifetime regression queues a four-byte write near the end of a 1 MiB
buffer and one RGBA8 pixel in a 1024 x 1024 image. Each wrapper reports exactly four
pending bytes, then zero after submission, cancellation or detachment. The tall
image case submits three supplied rectangles in three copies (2,584 bytes), rather
than expanding them into per-row tasks or allocating a full-image CPU snapshot.

The earlier [frame counters](data-update-tracker-frames.csv) and six Blender image
comparisons were measured on the superseded snapshot/deduplication implementation
at `8f4c52d`. Its second-frame claim of one copy / 32 bytes does **not** describe the
current task-only implementation. Startup and steady-state performance reports
remain historical measurements of their named commits; this revision does not
claim the same upload counts or throughput. Metal requires macOS verification.

Ninja Release GUI, CLI and fallback-test targets build successfully. Final Windows
D3D12 debug and Vulkan synchronization-validation runs each pass 45 tests with
four existing HDR-environment skips. Pre-commit passes. No Vulkan validation
errors were reported; existing shader-output warnings and intentionally injected
presentation failures remain. Logs are local under `out/pr64-upload-tasks-*`.

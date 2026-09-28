# Deferred resource update research

This Draft follows the startup-only PR64. The implementation and tests under
`code/` and `test/` are preserved byte-for-byte from `e72d7e8`, before the PR split.
The startup-only report/assets remain available through the parent PR. This work
is preserved for research, not proposed as a finished performance improvement.

## Preserved scope

- Typed Buffer/Image owning wrappers and explicit upload task queues, without
  persistent CPU snapshots or automatic byte-content deduplication.
- D3D12/Vulkan/Metal staged upload commands, batch submission, lifetime management
  and deferred BLAS/TLAS dependency scheduling.
- Earlier scene-data work, including texture-file reuse, hair radial computation
  reuse and per-scene shared-material source normalization.
- Existing tests and the historical startup/upload benchmark reports, including
  the remote macOS measurements. Historical measurements describe their named
  commits, not guaranteed performance of the present upload-task implementation.

## Unresolved performance issue

On `e72d7e8`, Classroom CLI profiling ran six frames per backend, original scene
settings with native Ray Query and 8 samples per dispatch. Frames 1-5 each recorded
2 upload batches, 2,346 copies and 250,876 uploaded bytes (about 107 bytes per copy).
D3D12 mean preprocess_submit was 331.250 ms; Vulkan was 20.879 ms. That scope includes
upload recording and submission, so it does not isolate allocation cost. The GUI
was left open; these timings are diagnostic, not controlled benchmark results.

Each upload still allocates/maps/unmaps its own staging buffer and records its own
copy. Vulkan buffer uploads also retain per-copy barriers. Reducing submission
count alone does not remove these costs. Follow-up research should examine shared
per-batch staging allocation, safe copy/barrier grouping and caller-owned dirty
flags or versions, without restoring resource-wide CPU mirrors.

## Validation and dependency

The preserved source was already built with Ninja Release (GUI, CLI and fallback
tests); D3D12 debug and Vulkan synchronization validation each passed 45 tests with
four HDR-environment skips. The split verifies code/test tree equality with that
validated commit, rather than rerunning unchanged source. Metal remains unverified.

The PR targets the PR64 branch to keep its diff limited to deferred work. After
PR64 is squash-merged, update this branch's baseline to main before resuming or
merging it, so startup changes are not presented a second time.

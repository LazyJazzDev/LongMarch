# Offscreen queue throughput benchmark

This compares the same work on main before PR113 (`6ddcd48e`), the current Main
queue, and current Assist preparation plus Main consumption. It does not create
windows, swapchains, or artificial scheduling delays. It measures end-to-end CPU
wall-clock throughput, including recording, uploads, submissions, slot reuse waits,
and final completion. It does not measure isolated GPU execution time.

Build `demo_rhi_queue_bench` in Release. For the old revision, copy this exact demo
source into a temporary baseline checkout and configure with `RHI_BENCH_LEGACY=ON`.
That switch changes only context creation, queue naming and submission API calls.
Old revision Assist is deliberately excluded: it was not a distinct native queue.

Arguments:

```
demo_rhi_queue_bench <vulkan|d3d12> <main|main-batched|assist> <upload|as-build|as-update> <elements-or-grid-side> <compute-loops-or-rays> <inflight> <frames> <validation:0|1>
```

- Upload: upload a uint buffer and compute an integer recurrence for every element.
- AS build/update: upload a changing triangular grid, rebuild or update BLAS and
  TLAS, then trace a 512x512 grid of rays with the requested rays per pixel.
  Updates use uncompressed AS allocated with update support.
- Every in-flight slot has separate input/output and AS resources. The CPU waits
  before reusing a slot. Assist mode adds the producer Event to Main submission;
  all-Main mode relies on native queue order. Standard Main and Assist modes submit
  twice per frame. `main-batched` records preparation and consumption in one context
  and submits once, providing a practical single-queue batching comparison.
- Slot geometry/data alternate between two real inputs on successive reuse cycles.
  The first max(32, 2*inflight) frames warm up before timing. Resource allocation,
  shader compilation, input generation, output downloads and CPU verification are
  outside timing. Every final slot's output is checked in full against CPU results.
- Throughput includes final retirement. This intentionally measures any CPU
  scheduling/collector overhead introduced by the refactor as well as GPU work.
- Validation must be checked separately, then disabled for timing in every mode.

`scripts/rhi_queue_bench.py` runs five repetitions of four workload configurations,
randomizing case and baseline/Main/Assist order with a fixed seed. It writes CSV
measurements and JSONL commands, full output and before/after NVIDIA telemetry.
Each process uses one API and one device, avoiding repeated Vulkan instance loading.
Do not run other GPU benchmarks concurrently on the same machine. Report medians
and variation, not just the fastest sample. Distinct native queues alone do not
prove hardware overlap or a speedup; throughput is the observed outcome.

Use `--profile ray-heavy` for the supplementary AS cases: the same 256x256 grid,
64 rays per output pixel, 2,048 measured frames, and the same in-flight depth.
This increases consumer work without changing upload/AS preparation size, to check
whether a longer Main workload provides more opportunity for useful overlap.

Use `--profile batched-check` to compare old/current `main-batched` against current
Assist for the AS-update case with eight rays per pixel. This checks whether the
positive Assist result against two Main submissions survives comparison with a
single Main submission. GPU task inputs are unchanged; submission granularity and
inter-context resource transitions can differ, as they would in an application.

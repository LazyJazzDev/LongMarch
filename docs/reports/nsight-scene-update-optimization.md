# Scene update optimization guided by Nsight

Measured on 2026-09-27, Windows x64, RTX 3090 Ti / driver 596.49,
MSVC 19.44, Ninja Release, Slang 2026.18.2, Windows SDK 10.0.26100.0
DXC 1.8.2502.11. This follows the [Classroom Nsight diagnosis](classroom-nsight-profile.md).

![Measured D3D12 and Vulkan throughput before and after optimization](https://media.githubusercontent.com/media/LazyJazzDev/LongMarchAssetsLFS/3a7d62bd12ef307be500b0b4a5d2242c86326af3/reports/nsight-scene-updates/throughput.png)

The chart is generated from the recorded frame data using
`scripts/plot_scene_update_benchmark.py`; its content-only [asset PR #34](https://github.com/LazyJazzDev/LongMarchAssetsLFS/pull/34)
is merged. Each panel uses its own vertical scale; bars within a panel share a scale.

## Change and scope

Baseline: `ab8c5c1`; optimized implementation: `082d216` on
`perf/nsight-scene-updates`, originally branched from PR60. After PR60 merged,
the optimization was replayed onto `e9e6ff1` as `84dbed6`, with identical code
and tests. The merged scene assets are byte-identical to the measured revision;
the current asset pointer additionally publishes the chart above.
Both measured executables use the same
vcpkg baseline and compiler DLLs. Measured scene assets:
`f5d2bcde1b3712a3fd3bdd37c4575666f0661ee1`.

The previous Nsight trace identified thousands of tiny synchronous uploads
between rendering tasks. This change removes redundant work at its source:

- Ray-tracing materials compare their CPU-owned parameter/index ranges with the
  last bytes uploaded. Unchanged data no longer triggers a synchronous transfer.
  Public parameter edits still upload immediately on the next buffer access.
- Texture images are registered against the current scene each update; the
  resulting indices are compared, so changing scene registration order remains
  correct even when material objects are shared across scenes.
- Geometry lights upload their 48-byte transform only when its bytes change.
  Their GPU-written power distribution occupies a separate buffer range and is
  still recomputed each frame, including after material or geometry changes.
- The compute renderer normalizes shader source once per distinct material
  object per update, retaining source-based deduplication and instance order.
  This cache is local to the update, so source edits and object destruction do
  not require persistent source-cache invalidation.

Scene registration, light preprocessing and TLAS updates still run each frame.
This does not introduce a scene-wide dirty flag, cache GPU-written buffer data,
or change shader code, sampling settings or the tracing algorithm.

## Throughput

Native hardware ray query, as confirmed in all twelve logs; original scene
resolution/camera, eight samples per dispatch, 16 bounces for Classroom/Junkshop
and 32 for Monster. Classroom is 1920 x 1080, Junkshop 2000 x 1000, Monster 1024 x 1024.
Each process renders 60 frames. Frames 0-9 are discarded; frames 10-59 are measured.
The CLI CPU-only frame timer includes render completion and SDR development,
excluding initial scene loading, final PNG output, GUI and presentation.
No debug layers, GPU timestamps or Nsight capture are enabled for these runs.

Rays/s means primary camera samples per second, calculated as
`width * height * 8 / mean_frame_seconds`. It does not count secondary or shadow
rays. Runs are sequential, one process per condition, with all baseline runs
followed by all optimized runs in the table's order. Small percentage differences
can vary with clocks and run order; these are local measurements, not a multi-run
confidence estimate or a prediction for other devices.

| Scene | Backend | Before, M Rays/s | After, M Rays/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| Classroom | d3d12 | 52.80 | 124.39 | 2.36x |
| Classroom | vulkan | 43.62 | 88.49 | 2.03x |
| Junkshop | d3d12 | 232.25 | 251.14 | 1.08x |
| Junkshop | vulkan | 181.68 | 201.24 | 1.11x |
| Monster | d3d12 | 126.27 | 140.47 | 1.11x |
| Monster | vulkan | 91.03 | 100.38 | 1.10x |

[Measured per-frame data](nsight-scene-update-frames.csv) contains all 600 retained
frames. Empty upload-counter cells indicate D3D12, which does not emit those
application counters; the Nsight API trace below provides its copy counts.

## Classroom CPU scopes

| Scope, ms | D3D12 before | D3D12 after | Vulkan before | Vulkan after |
| --- | ---: | ---: | ---: | ---: |
| Frame | 314.155 | 133.360 | 380.297 | 187.471 |
| Scene update | 194.650 | 14.921 | 213.394 | 23.848 |
| Registration | 149.839 | 5.666 | 159.790 | 7.478 |
| Compute preparation | 40.446 | 4.484 | 40.590 | 3.362 |
| Render submission | 0.653 | 0.645 | 164.524 | 161.185 |
| Render wait | 111.149 | 110.089 | 0.029 | 0.029 |

Registration and compute preparation are nested inside scene update; these
inclusive CPU scopes must not be added together. They include driver/GPU waits.
Vulkan waits inside submission, so its separate render-wait scope is near zero.
The stable D3D12 render-wait interval and reduced scene-update interval locate
the main gain in scene maintenance, rather than a faster ray-query shader.
Vulkan's static-buffer upload count falls from 2,345 to 12 per frame.

## Nsight confirmation

A separate Nsight Systems 2025.5.1 D3D12 capture at `082d216` collects eight
seconds after a 180-second initialization delay. The application's flushed CSV
already reached frame 139 before capture began and reaches frame 357 by the end,
confirming collection after warmup. This capture is separate from throughput runs.

The 59 complete intervals between successive long rendering tasks give:

| Recorded quantity | Historical trace | Optimized trace |
| --- | ---: | ---: |
| Complete intervals | 22 | 59 |
| Main-render start to next start | 342.271 ms | 133.149 ms |
| Union of GPU command-list execution | 115.686 ms | 116.916 ms |
| Gaps without recorded execution | 226.585 ms | 16.233 ms |
| Main rendering task | 107.682 ms | 110.552 ms |
| Queue-active fraction of those intervals | 33.80% | 87.81% |
| Command-list submissions / interval | 2,349 | 16 |
| CopyBufferRegion / interval | 2,345 | 12 |
| Dispatch / interval | 2,273 | 2,273 |

The historical trace is the earlier `72f475c` capture described in the
[diagnosis](classroom-nsight-profile.md), with its older compiler/dependency setup.
It provides context for the submission pattern; use the **same-toolchain
ab8c5c1 versus 082d216 measurements above** for the controlled speedup comparison.
The new trace has identical copy, submission and dispatch counts in all 59
intervals. The unchanged dispatch count is expected: this iteration retains
light-power preprocessing and reduces redundant transfers and CPU source work.

[Per-interval trace data](nsight-scene-update-intervals.csv) records the new
measurements. Intervals are bounded by consecutive `DX12_WORKLOAD` tasks longer
than 50 ms (the main ray-query dispatch). GPU intervals are clipped and unioned
inside each period; gaps are its duration minus that union. API calls are counted
by their start timestamp in that period.

The diagnostic still says "Not all DX12 events might have been collected".
These values describe the recorded stream and are corroborated by Vulkan's
application counters; they are not a completeness guarantee for every queue.
The queue-active fraction is **not SM occupancy**. No shader hardware counters
were collected, so this report does not attribute stalls within the main shader.

Reproduce the optimized capture with:

```powershell
& 'C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.5.1/target-windows-x64/nsys.exe' `
  profile --trace=dx12 --sample=none --cpuctxsw=none --delay=180 --duration=8 `
  --kill=true --output=out/nsight-optimization/optimized-classroom `
  cmake-build-ninja/demo/sparkium_cli/demo_sparkium_cli.exe `
  assets/scenes/blender_classroom/scene.json --backend d3d12 --require-hardware-rt `
  --frames 6000 --profile-cpu-only --profile out/nsight-optimization/nsys-after.csv `
  -o out/nsight-optimization/nsys-after.png
```

Nsight deliberately terminates the process at capture end, so this run has no
final PNG. The local `.nsys-rep`, exported `optimized-classroom.sqlite`,
`trace-analysis.json` and `analyze.py` are in `out/nsight-optimization/`.
Check warmup completion again before reusing this delay on another machine.

## Correctness and build validation

- Ninja Release builds pass for Sparkium GUI, CLI and `sparkium_fallback_test`.
- D3D12 and Vulkan debug/validation runs each pass 34 tests; two unsupported-HDR
  cases are skipped because this system supports HDR. HDR-window tests are enabled.
  No validation-layer errors were found in either run.
- New GPU readback tests cover material parameter edits/reverts, emitter flags,
  alternating scene texture indices, texture removal and light transform changes.
  Vulkan counters also verify that repeated unchanged accesses upload nothing.
- All six optimized final 60-frame PNGs match their corresponding baseline
  images exactly, with zero differing pixel values. This is a same-backend,
  before/after comparison, not a cross-backend image comparison.
- Pre-commit checks pass. Metal is not tested on this Windows machine.

## macOS follow-up

[Apple M5 / Metal verification](nsight-scene-update-macos.md) measures three runs
per condition against the merged baseline. Classroom improves by 1.09%; Junkshop
and Monster show no consistent gain across runs. All nine before/after PNG pairs
are byte-identical. This follow-up supplements the Windows results above.

## Reproduction and artifacts

For each baseline/optimized executable and each scene/backend, run from the
repository root (changing the output stem to keep both measurements):

```powershell
& $cli assets/scenes/blender_classroom/scene.json --backend d3d12 `
  --require-hardware-rt --frames 60 --profile-cpu-only `
  --profile out/nsight-optimization/after-classroom-d3d12.csv `
  -o out/nsight-optimization/after-classroom-d3d12.png
```

Local logs, PNGs, executable snapshot, comparison JSON and benchmark runner are
under `out/nsight-optimization/`. The CSV linked above is tracked; the local
binary artifacts are not. Build the parent commit separately or retain its
executable and DLLs before building the optimization to reproduce the baseline.

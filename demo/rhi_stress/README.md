# RHI stress benchmark

`demo_rhi_stress` renders independently bound, animated textured quads to three
rotating offscreen images. It exercises per-object uniform bindings, 16 vertex
buffers, eight textured materials, four pipelines, per-batch uploads, and
multiple submissions. This deliberately does not batch objects into instanced
or indirect draws: its purpose is to measure the cost of many RHI operations.

Example CPU/command-heavy workload:

```
demo_rhi_stress --backend vulkan --grid 64 --width 1280 --height 720 --batch 256 --iterations 1 --frames 1000 --warmup 60
```

Example shader-heavy workload:

```
demo_rhi_stress --backend d3d12 --grid 16 --width 3840 --height 2160 --batch 64 --iterations 2048 --frames 1000 --warmup 60
```

The first case performs 4096 draws and 16 submissions and uploads 1 MiB of object
constants each frame. The second performs 256 draws and four submissions; its
fragment shader executes a dependent recurrence whose result affects output.
The shader-heavy case is a synthetic arithmetic workload, not a production
material performance claim. Check scaling with iteration count before treating
it as GPU-limited.

`RESULT` is a JSON record. `seconds` measures complete-frame wall time including
CPU updates, recording/submission, frame-slot waits, and the final GPU drain.
`record_seconds` sums wall time from after each slot wait through the final
submission; it includes allocation and any stalls inside submission, and is
not CPU cycle time or a GPU timestamp. `wait_seconds` covers frame-slot waits
and releasing the old slot token; the final drain appears only in total time.
Setup/compilation and warmup are excluded. At most three frames are in flight;
reuse waits for the last submission on the same queue, which also orders earlier
batches. Each slot owns separate constant storage and binding groups.

`--capture DIR` downloads the final image after the timing region and writes PNG,
raw pixels and metadata. Compare raw outputs from identical frame/warmup counts
before timing binaries built from different revisions. Image capture is not used
for performance samples. Use Release without validation, the same build options,
and identical benchmark sources for both revisions.

For paired measurements, preserve each executable together with its runtime DLLs.
Build the same `demo/rhi_stress` source on both revisions using the same Release
configuration. From the repository root, run (replace executable paths):

```powershell
python scripts/compare_rhi_stress.py prepare --baseline BASELINE.exe --candidate CANDIDATE.exe --output out/stress-comparison --performance-cores
python scripts/compare_rhi_stress.py measure --baseline BASELINE.exe --candidate CANDIDATE.exe --output out/stress-comparison --performance-cores
python scripts/summarize_rhi_stress.py out/stress-comparison
```

Preparation compares exact pixels, checks shader-work scaling, and calibrates a
common frame count targeting five seconds averaged across the two binaries.
Measurement uses 120 warmup frames, full-length priming runs, two randomized
blocks of eight A/B pairs and two interleaved A/A controls per case. Each block
balances A/B execution order. All valid samples are retained. The summary uses
paired log-ratio bootstrap intervals and also reports intervals adjusted for
four wall-time comparisons. The predeclared practical equivalence band is +/-3%.

`--performance-cores` restricts the Windows runner and its children to the fastest
CPU efficiency class. Omit it in both commands to test normal scheduling instead;
use a fresh output directory. Scheduling can materially change the magnitude of
the observed overhead, so report the chosen configuration explicitly.

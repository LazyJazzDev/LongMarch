# Blender throughput: Slang versus the previous HLSL build

Measured on 2026-09-26, Windows x64, NVIDIA RTX 3090 Ti, driver 596.49,
MSVC 19.44, Ninja Release.

- Previous HLSL/DXC measurements: `ec43a025d2ae22fc62b09823518fdc0d6871d723`,
  assets `8ef3107d99e9f29b840fa89346378c9238823a08`.
- Current Slang 2026.18.3 measurements: `d13974ade57aa882035529615ad9e3d759685ef2`,
  assets `f5d2bcde1b3712a3fd3bdd37c4575666f0661ee1`.
- The three scene directories and CLI timing code are unchanged between those
  versions. The current version also contains subsequent main-branch fixes, so
  this is a version comparison, not an isolated compiler-only experiment.

## Method

Native hardware ray query (compute with native acceleration structures), as
selected by Auto and confirmed in every run's log. Eight samples per dispatch;
120 frames per process, discarding frames 0-19 and measuring frames 20-119.
The CPU `frame_wall` timer includes render completion and SDR image development,
but excludes initial scene loading, final PNG writing, GUI and presentation.
Shader compilation and scene registration occur in the discarded warmup frames.
No debug layers or GPU timestamp instrumentation were enabled.

Rays/s here means **primary camera samples/s**, calculated as
`width * height * samples_per_dispatch / mean_frame_seconds`;
secondary and shadow rays are not counted.

| Scene | Resolution | Max bounces |
| --- | --- | ---: |
| Classroom | 1920 x 1080 | 16 |
| Junkshop | 2000 x 1000 | 16 |
| Monster | 1024 x 1024 | 32 |

The current version ran twice per combination, sequentially, with backend order
reversed in round two. The reported throughput uses the pooled mean frame time
of the 200 measured frames. Historical HLSL data has only one 100-frame run per
combination and was not rerun in this session. Desktop background load and GPU
clocks were not fixed; small percentage changes should not be overinterpreted.
The round range below is descriptive, not a confidence interval.

## Steady-state throughput

All throughput values are million primary Rays/s (higher is better).

| Scene | Backend | Previous HLSL | Current Slang | Change | Slang round 1 / 2 | Mean frame ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classroom | D3D12 | 53.31 | 52.73 | -1.10% | 52.76 / 52.70 | 314.62 |
| Classroom | Vulkan | 44.36 | 44.73 | +0.84% | 45.03 / 44.43 | 370.87 |
| Junkshop | D3D12 | 231.16 | 236.25 | +2.20% | 237.77 / 234.74 | 67.73 |
| Junkshop | Vulkan | 196.75 | 187.21 | -4.85% | 185.12 / 189.35 | 85.47 |
| Monster | D3D12 | 124.30 | 127.68 | +2.72% | 128.90 / 126.48 | 65.70 |
| Monster | Vulkan | 83.75 | 94.63 | +12.99% | 93.67 / 95.61 | 88.64 |

Classroom is essentially unchanged. The largest observed improvement is Monster
on Vulkan; Junkshop on Vulkan is slower in both current runs than the historical
run. D3D12 remains faster than Vulkan in all three scenes. This is not a uniform
speedup or slowdown from the migration. The historical single-run baseline and
uncontrolled desktop load limit causal and statistical conclusions.

All 12 processes exited successfully and reported native ray query. No Slang
warnings or shader compilation errors occurred. Output images were produced at
full scene resolution, 960 accumulated samples; image inspection is a smoke
check, not proof of pixel-identical shading or a full quality comparison.

## First-frame cost (excluded from throughput)

These wall times include scene registration, shader/pipeline compilation,
acceleration structure preparation and the first render. They are not isolated
compiler timings; driver caches and initialization conditions can affect them.
The values are the historical first frame and current round-one first frame.

| Scene | Backend | HLSL first frame, s | Slang first frame, s |
| --- | --- | ---: | ---: |
| Classroom | D3D12 | 62.79 | 136.89 |
| Classroom | Vulkan | 249.50 | 142.26 |
| Junkshop | D3D12 | 32.58 | 49.64 |
| Junkshop | Vulkan | 154.05 | 55.34 |
| Monster | D3D12 | 19.29 | 30.50 |
| Monster | Vulkan | 91.52 | 34.52 |

## Reproduction and raw data

Build `demo_sparkium_cli` with Ninja Release, then run each combination sequentially:

```powershell
.\cmake-build-ninja\demo\sparkium_cli\demo_sparkium_cli.exe `
  assets/scenes/blender_classroom/scene.json --backend vulkan `
  --require-hardware-rt --frames 120 --profile-cpu-only `
  --profile out/benchmark/classroom-vulkan.csv -o out/benchmark/classroom-vulkan.png
```

Repeat with `d3d12`, `blender_junkshop` and `blender_monster`; exclude frames below
20 and use only `cpu_ms,frame_wall` rows. Swap backend order on the second round.
Local raw data is retained in `out/blender-rays-benchmark/` (historical) and
`out/blender-slang-benchmark/` (current), including per-frame CSVs, logs, outputs,
commands, scene settings/hashes and per-round/block summaries. These temporary
artifacts are not tracked in Git.

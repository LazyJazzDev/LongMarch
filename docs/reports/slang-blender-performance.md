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

## Apple M5 / Metal measurements

Measured on 2026-09-26 at PR #60 commit `72f475c1bcee2a8d21212fb22c462df9529daa50`,
assets `f5d2bcde1b3712a3fd3bdd37c4575666f0661ee1`. Apple M5 (10 GPU cores),
macOS 26.6.2 (25G83), AC power, Ninja Release, vcpkg Slang 2026.18.3.
Slang generates SPIR-V, then SPIRV-Cross generates MSL for Metal.

Each scene ran twice in separate sequential processes, with scene order reversed
in round two. Each process rendered 120 frames at 8 samples/frame (960 spp total).
Frames 0-19 were discarded; the table pools 200 measured frames per scene.
All runs explicitly selected native `ray_query`; logs and every frame's native
query counter confirmed this. GPU debug layers and GPU timestamp profiling were
not enabled. The old GUI was closed before timing; other desktop background load,
GPU clocks and thermal conditions were not controlled. No thermal warning was
reported when queried during the first run.

The CLI's `--require-hardware-rt` currently checks standalone ray-tracing pipeline
support and rejects Metal even when native inline queries are supported. These
runs instead use `--pipeline ray_query`, which fails if ray queries are unavailable;
there is no software fallback in these measurements.

| Scene | Resolution | Bounces | Mean frame ms | M primary Rays/s | Round 1 / 2 M Rays/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Classroom | 1920 × 1080 | 16 | 436.45 | 38.01 | 38.29 / 37.73 |
| Junkshop | 2000 × 1000 | 16 | 315.31 | 50.74 | 50.96 / 50.53 |
| Monster | 1024 × 1024 | 32 | 392.24 | 21.39 | 21.60 / 21.18 |

The timing includes rendering, GPU completion and SDR Film development, but not
initial scene loading, final PNG output or GUI/presentation. Throughput counts
primary camera samples only, not secondary/shadow rays or hardware intersections.
Junkshop has the highest primary-sample throughput, followed by Classroom and
Monster; these scenes have different material/path complexity and bounce limits,
so this ranking is not an isolated measure of GPU traversal cost.

The largest round-to-round throughput spread is 2.00% (max/min − 1).
All six runs exited successfully with no shader warnings or compilation errors.
One 960-spp full-resolution output per scene was visually inspected: all showed
the expected scene rather than black output. Visible Monte Carlo noise remains;
this is a rendering smoke check, not proof of Blender parity or a quality regression.
No matching historical DXC run was performed on this Mac, so these results do not
establish a Slang speedup/slowdown. Windows RTX 3090 Ti numbers above are a different
machine and backend, not a compiler-only baseline for the M5.

### First-frame wall time

Includes registration, shader/pipeline compilation, acceleration-structure work
and rendering. These are not isolated compiler measurements or cold-cache results;
Metal/driver caches were not cleared between runs.

| Scene | Round 1 seconds | Round 2 seconds |
| --- | ---: | ---: |
| Classroom | 24.64 | 23.03 |
| Junkshop | 5.85 | 4.15 |
| Monster | 4.20 | 3.94 |

### Reproduce on macOS

```sh
cmake -S . -B out/pr60-metal/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVCPKG_PATH=/path/to/vcpkg -DLONGMARCH_DISABLE_PYTHON=ON
cmake --build out/pr60-metal/build --target demo_sparkium_cli -j6
out/pr60-metal/build/demo/sparkium_cli/demo_sparkium_cli \
  assets/scenes/blender_classroom/scene.json --backend metal \
  --pipeline ray_query --frames 120 --profile-cpu-only \
  --profile out/benchmark/classroom-metal.csv -o out/benchmark/classroom-metal.png
```

Repeat for Junkshop and Monster, then repeat in reverse order. Compute the pooled
mean of `cpu_ms,frame_wall` for frames 20-119 and divide width × height × 8 by the
mean seconds. [All 720 frame wall-time measurements](slang-blender-metal-frames.csv)
are committed alongside this report, including discarded warmup frames. Full logs,
images, scene hashes, commands and metadata remain in `out/pr60-metal/benchmark/`.

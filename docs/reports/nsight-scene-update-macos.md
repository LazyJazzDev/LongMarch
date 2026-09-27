# Scene update optimization: macOS Metal verification

Measured on 2026-09-27 for [PR #64](https://github.com/LazyJazzDev/LongMarch/pull/64).
On this Apple M5, Classroom throughput improves by **1.09%**; Junkshop and
Monster show no consistent gain across three runs. The Windows RTX 3090 Ti
speedups in the [original report](nsight-scene-update-optimization.md) do not
reproduce on this Metal configuration.

## Environment and method

- Baseline: `e9e6ff1`; PR: `1b9fa4137801a7be84b0757362cd710ea1ed1c95`.
- Apple M5, 10-core GPU, 32 GB; macOS 26.6.2 (25G83).
- Ninja Release, Apple Clang 21.0.0 (`clang-2100.3.34.2`), Slang 2026.18.2.
  Both builds share the same installed vcpkg dependencies. Manifest installation
  was disabled for both after the local vcpkg checkout could not resolve the
  manifest baseline; this is a controlled local comparison, not a fresh
  dependency-installation validation.
- Both use assets `3a7d62bd12ef307be500b0b4a5d2242c86326af3`, including the same
  Sobol direction-number resource. The baseline worktree's asset directory
  points to this shared asset checkout.
- Battery power, low power mode off. System queries reported no thermal or
  performance warnings. GPU clocks and desktop background load were not
  controlled. An existing iOS simulator Sparkium process remained open and was
  observed at 0.0% CPU; it was not terminated.
- Explicit Metal `ray_query`, confirmed in all 18 successful run logs.
  `--require-hardware-rt` is not used: it checks standalone RT pipeline support
  and rejects Metal even when native inline queries are supported. Explicit
  `ray_query` fails if native queries are unavailable, without software fallback.
- Three independent processes per version and scene, 60 frames each, discarding
  frames 0–9. There are 150 retained frames per condition, 900 total.
  Scene order is Classroom, Junkshop, Monster. Version order is before/after in
  rounds 1 and 3, after/before in round 2. Builds and debug tests finish before
  timed runs; rendering processes run sequentially.
- Eight samples per dispatch, original cameras and scene settings: Classroom
  1920 × 1080 / 16 bounces, Junkshop 2000 × 1000 / 16, Monster 1024 × 1024 / 32.
- CPU-only `frame_wall` includes scene updates, rendering completion and SDR
  development. It excludes scene loading, final PNG output and presentation.
  GPU timestamps and debug layers are disabled for timed runs.

## Results

M samples/s counts primary camera samples, not secondary or shadow rays.
Throughput change is `(before_mean_ms / after_mean_ms - 1) * 100`.

| Scene | Before ms | After ms | Before M samples/s | After M samples/s | Throughput change | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Classroom | 442.616 | 437.837 | 37.479 | 37.888 | +1.09% | 1.0109× |
| Junkshop | 318.659 | 320.916 | 50.210 | 49.857 | −0.70% | 0.9930× |
| Monster | 398.307 | 397.227 | 21.061 | 21.118 | +0.27% | 1.0027× |

Each value pools the 150 retained frames; equal run lengths also make this the
mean of the three per-process means. These are local measurements, not confidence
intervals or predictions for other devices.

| Scene | Before ms, rounds 1 / 2 / 3 | After ms, rounds 1 / 2 / 3 | Throughput change, rounds 1 / 2 / 3 |
| --- | --- | --- | --- |
| Classroom | 441.907 / 443.233 / 442.709 | 437.170 / 438.226 / 438.115 | +1.08% / +1.14% / +1.05% |
| Junkshop | 319.125 / 317.013 / 319.840 | 319.221 / 324.451 / 319.076 | −0.03% / −2.29% / +0.24% |
| Monster | 398.174 / 399.902 / 396.847 | 397.632 / 396.824 / 397.226 | +0.14% / +0.78% / −0.10% |

Classroom consistently gains about 1%. Junkshop and Monster change in both
directions across rounds; their aggregate differences do not establish a
consistent improvement or regression.

### Classroom CPU scopes

| Scope | Before ms | After ms |
| --- | ---: | ---: |
| Scene update | 19.988 | 14.741 |
| Scene registration | 13.994 | 13.906 |
| Compute preparation (`software_prepare`) | 5.873 | 0.714 |
| Render wait | 417.818 | 418.312 |

These are inclusive, nested CPU scopes and must not be added together. Scene
update saves about 5.25 ms (26.25%), largely in compute preparation, while scene
registration stays nearly unchanged. Render wait still dominates the frame.
This explains why the measured total gain is much smaller than on Windows;
it does not establish GPU occupancy or hardware-counter behavior.

## Correctness and build validation

Both Release CLI builds pass. The PR's `sparkium_fallback_test` also builds and,
with `SPARKIUM_TEST_BACKEND=metal SPARKIUM_TEST_DEBUG=1`, reports **29 passed,
12 skipped, zero failures**. Skips include disabled window tests and unavailable
standalone RT pipeline coverage. The new material edit/restore, texture mapping,
texture removal and geometry-light transform readback cases pass.

All nine corresponding before/after final PNG pairs are byte-identical
(three scenes × three rounds). This is same-backend image equivalence for the
measured static scenes; it is not cross-backend parity or dynamic-scene coverage.

## Data and reproduction

[Retained per-frame measurements](nsight-scene-update-macos-frames.csv) contain
all 900 frames and the five CPU scopes used above, preserving CLI precision.
The [benchmark script](../../scripts/benchmark_scene_updates_macos.py) summarizes
this tracked CSV without requiring a GPU:

```sh
python3 scripts/benchmark_scene_updates_macos.py
```

For a new run, build the baseline and PR CLI separately with Ninja Release and
the same dependencies, Slang SDK and assets. The local baseline build used
`-DVCPKG_INSTALLED_DIR` pointing at the PR build's installed dependencies and
`-DVCPKG_MANIFEST_INSTALL=OFF`; both builds used the same vcpkg toolchain file.
Ensure both builds can find the full shared assets directory, including Sobol
resources, not only the scene JSON. Then run from the PR checkout:

```sh
caffeinate -i python3 scripts/benchmark_scene_updates_macos.py \
  --before /path/to/baseline/demo/sparkium_cli/demo_sparkium_cli \
  --after /path/to/pr/demo/sparkium_cli/demo_sparkium_cli \
  --output out/scene-updates-macos-repeat
```

The output directory must be new. The script runs the same 18 conditions,
compares all nine PNG pairs and writes `frames.csv` next to the raw artifacts,
leaving the published CSV unchanged. Each individual invocation uses:

```sh
/path/to/demo_sparkium_cli assets/scenes/blender_classroom/scene.json \
  --backend metal --pipeline ray_query --frames 60 --profile-cpu-only \
  --profile out/benchmark/classroom.csv -o out/benchmark/classroom.png
```

Original full logs, warmup frames, PNGs, build logs, test output and metadata
remain locally under `out/pr64-local/`; these temporary binary artifacts are not
part of the commit. The tracked CSV and script reproduce the reported aggregates.

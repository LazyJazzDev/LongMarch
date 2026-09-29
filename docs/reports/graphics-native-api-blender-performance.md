# Blender performance after removing the D3D12/Vulkan wrapper layers

Compared `main` at `35215476f55592508ab726ad2382ba18b43afaa3` with the native
graphics backend branch at `22d306cb4015e44e3d20f19330cbbb81d14dc89e`.
The CLI and Sparkium renderer sources are identical between these revisions.
Both binaries rendered the same scene files; the scene directories are unchanged
between the checked-out asset revision `90ce963` and the `main` asset revision
`9f19985`.

## Method

- Windows 11 Pro, Intel Core i9-13900K, NVIDIA GeForce RTX 3090 Ti, driver
  596.49. The GPU was idle before the runs.
- Both binaries were built with Ninja, `CMAKE_BUILD_TYPE=Release`, MSVC
  `/O2 /Ob2 /DNDEBUG`, and `LONGMARCH_DISABLE_PYTHON=ON`.
- The Sparkium CLI rendered 120 frames per process with `--backend vulkan` or
  `--backend d3d12`, `--pipeline ray_query`, `--require-hardware-rt`, and
  `--profile-cpu-only`. Every run reported `native ray query (compute, native AS)`.
- Classroom: 1920 x 1080, 8 samples/dispatch, 16 maximum bounces. Junkshop:
  2000 x 1000, 8 samples/dispatch, 16 maximum bounces. Monster: 1024 x 1024,
  8 samples/dispatch, 32 maximum bounces.
- Each scene/backend pair ran twice. Round 1 ran `main` before the branch;
  round 2 reversed version and backend order. First frame is frame 0 of each
  process. Steady time is the pooled mean of frames 20-119 from both runs
  (200 measurements per scene/backend/version).
- The CPU `frame_wall` timer includes render completion and SDR image
  development, as in the GUI. It excludes scene loading, PNG writing, and
  window presentation. GPU timestamps were disabled. Shader and driver caches
  were **not** cleared between runs, so these are not cold-cache startup times.

## Results

All times are milliseconds; the first-frame columns show rounds 1 / 2.
Positive steady change means the branch took longer.

| Scene | Backend | First `main` | First branch | Steady `main` | Steady branch | Change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Classroom | Vulkan | 5561 / 5790 | 5832 / 5793 | 370.527 | 369.284 | -0.34% |
| Classroom | D3D12 | 4674 / 4968 | 4965 / 5008 | 315.188 | 316.711 | +0.48% |
| Junkshop | Vulkan | 6030 / 4049 | 4157 / 4074 | 72.201 | 72.936 | +1.02% |
| Junkshop | D3D12 | 5045 / 3661 | 3756 / 3675 | 63.225 | 64.273 | +1.66% |
| Monster | Vulkan | 5019 / 3213 | 3480 / 3499 | 97.809 | 97.618 | -0.20% |
| Monster | D3D12 | 3473 / 2640 | 2966 / 2978 | 70.352 | 69.516 | -1.19% |

The largest observed steady slowdown was Junkshop/D3D12 at 1.66%. Across all
six pairs, steady changes stayed within 1.7%. First-frame measurements changed
substantially between rounds for Junkshop and Monster, including on unchanged
`main`; this test does not isolate a small startup difference from cache and
run-order effects. Classroom first frames stayed near 5-6 seconds on Vulkan and
4.7-5.0 seconds on D3D12, with no large startup regression.

The 12 matching `main`/branch output PNG pairs (three scenes, two backends, two
rounds) were compared pixel by pixel and were identical. The complete 2,880
`frame_wall` measurements, including warmup frames, are in
[graphics-native-api-blender-frames.csv](graphics-native-api-blender-frames.csv).

For reproduction, configure each revision in a separate build directory with
the Release flags above, then run, for example:

```powershell
demo_sparkium_cli.exe assets/scenes/blender_classroom/scene.json `
  --backend vulkan --pipeline ray_query --require-hardware-rt `
  --frames 120 --profile-cpu-only --profile classroom-vulkan.csv `
  -o classroom-vulkan.png
```

Repeat for the other scenes and backend, reverse the order in round 2, and
calculate the mean of `cpu_ms,frame_wall` for frames 20-119.

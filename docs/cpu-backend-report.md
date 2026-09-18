# Sparkium CPU backend — results

An illustrated summary of the CPU path tracing backend: what it produces, how
closely it matches the GPU, and what it costs. The reference documentation is
[Sparkium CPU backend](cpu-backend.md).

Everything below is reproducible with the scripts named in each section.

## What it does

Sparkium can trace paths on the host with no graphics device involved:

```sh
demo_sparkium_cli assets/scenes/cornell_box/scene.json --backend host --pipeline cpu -o out.png
```

The pipeline does not reimplement the renderer. The shaders in
`code/sparkium/shaders/` are compiled as C++ and executed natively — the same
traversal, BSDFs, materials and lighting the compute fallback runs on the GPU.
A build-time rewrite handles the handful of constructs with no C++ spelling, and
a compatibility layer supplies HLSL's types and intrinsics. The checked-in
shaders are never modified.

## Does it look the same?

![Reference and CPU renders side by side](https://github.com/LazyJazzDev/LongMarchAssetsLFS/blob/blender-align/docs/cpu-backend-comparison.png?raw=true)

*For each scene, the compute fallback on the left and the CPU backend on the
right. Nine scenes at 1024×1024.*

The figure lives in the asset library rather than here, because binaries belong
with the other LFS assets and a markdown image inside a submodule does not
render on GitHub. `scripts/check_cpu_backend.py` regenerates it from the two
renders it writes next to it; it is copied to `assets/docs/` and committed
there, on the branch the `assets` submodule tracks.

`scripts/check_cpu_backend.py` renders every basic scene both ways and compares
the tone mapped images:

| Scene | Mean absolute pixel difference | Max difference | Over 2/255 |
| --- | --- | --- | --- |
| cornell_box | 0.001 / 255 | 77 | 0.01% |
| principled | 0.001 / 255 | 77 | 0.01% |
| specular | 0.001 / 255 | 73 | 0.01% |
| texture | 0.508 / 255 | 228 | 4.73% |
| blender_junkshop | 1.085 / 255 | 255 | 3.68% |
| blender_monster | 1.971 / 255 | 255 | 5.06% |
| point_light | 2.146 / 255 | 12 | 22.84% |
| blender_classroom | 2.828 / 255 | 255 | 6.19% |
| area_light | 5.350 / 255 | 27 | 97.14% |

Three scenes match to within 8-bit quantization. The rest fall into two groups,
and they behave differently as samples accumulate:

| | 1 frame | 16 frames | Behaviour |
| --- | --- | --- | --- |
| area_light (mean) | 5.35 | 0.16 (32 frames) | falls as 1/sqrt(N) |
| point_light (mean) | 2.146 | — | broad and shallow, noise-like |
| texture (mean) | 0.797 | 0.580 | plateaus, does not reach zero |
| blender_junkshop (mean) | 1.486 | 1.640 | plateaus, does not reach zero |

- **area_light and point_light** are lit almost entirely by light sampling, and
  the two backends' sampling CDFs round differently, so their sample sequences
  diverge from the first ray. The estimates stay unbiased and the difference
  falls with sample count, which is what Monte Carlo divergence looks like:
  area_light reads 1.486 at one frame and 0.16 at thirty-two.
- **texture and the graph scenes plateau** at roughly 0.5 to 1.7 out of 255 and
  do not fall further. A residual that survives averaging is systematic, so
  something in those paths differs beyond sampling. It has not been identified.
  Two candidates were ruled out by changing them and re-measuring: the graph
  engines' `object_origin` mapping (a real bug, fixed, but unused by these
  scenes) and the shim's `lerp` rounding (changed to match HLSL's definition,
  no measurable effect at 8-bit output).

The `max` column explains the rest. The graph scenes reach 255 because a
firefly lands on a different pixel in the two renders, and cornell_box's 77 is a
silhouette pixel where a ray grazes a shared edge and the two trees report
different triangles. Both are 0.01% to a few percent of the image.

## What does it cost?

`scripts/profile_backends.py` runs every scene on both backends at the same
resolution, sample count and bounce limit, drops the warmup frames and reports
the median:

| Scene | Resolution | spp | Bounces | GPU ms | GPU Mray/s | CPU ms | CPU Mray/s | CPU / GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cornell_box | 512² | 8 | 12 | 72.89 | 28.77 | 166.05 | 12.63 | 2.28 |
| area_light | 512² | 8 | 12 | 137.01 | 15.31 | 457.05 | 4.59 | 3.34 |
| point_light | 512² | 8 | 12 | 148.10 | 14.16 | 478.18 | 4.39 | 3.23 |
| principled | 512² | 8 | 12 | 90.73 | 23.11 | 167.00 | 12.56 | 1.84 |
| specular | 512² | 8 | 12 | 70.87 | 29.59 | 163.21 | 12.85 | 2.30 |
| texture | 512² | 8 | 12 | 458.33 | 4.58 | 1023.04 | 2.05 | 2.23 |
| blender_classroom | 512² | 8 | 12 | 1631.31 | 1.29 | 31704.29 | 0.07 | **19.43** |
| blender_junkshop | 512² | 8 | 12 | 1874.73 | 1.12 | 16744.77 | 0.13 | **8.93** |
| blender_monster | 512² | 8 | 12 | 1372.08 | 1.53 | 20365.24 | 0.10 | **14.84** |

Measured on an Apple M5 (4 performance and 6 efficiency cores). "Mray/s" counts
camera rays, the usual convention: width × height × spp per frame.

The table splits into two groups, and the split is exactly the shader graphs:

- **Scenes without node graphs are 1.8–3.3× slower than the GPU.** That is the
  expected range: no hardware traversal and much less SIMD. Per camera ray,
  cornell_box costs 79 ns against 35 ns on the GPU.
- **Scenes with node graphs are 9–19× slower.** Per camera ray,
  blender_classroom costs 15 µs — about 190× the cornell_box figure. The
  rendering is not the problem; the graph *interpreter* is, because it walks an
  expression tree per shading event over a dynamically typed value rather than
  running compiled code.

That last point is the reason the JIT matters more than it first appeared: it
decides whether graph scenes are usable on the CPU at all.

Per frame, the fixed costs are negligible: on cornell_box at 1024² the path
tracing is 2576.9 ms of a 2606.3 ms frame, scene registration and the BVH build
are 6.5 ms, and the tone mapping is 23.1 ms. The first frame is only 1.9% slower
than the steady state, because mesh trees are cached and only the instance set
is rebuilt.

## Shader graph engines

Two engines evaluate the generated graph source; the same source the GPU hands
to DXC.

- **Interpreter** — parses and walks the generated code. This is what runs, and
  what every graph scene in the tables above used.
- **JIT** — assembles the whole material set into one module and compiles it
  with Clang, linking LLVM and Clang into the backend.

The JIT is implemented but **not enabled**: it compiles the generated module
correctly, every symbol it needs resolves, and the entry point is found, but
calling into the MCJIT'd code faults. Until that is understood the backend
withholds JIT programs and uses the interpreter, so `--graph-engine jit` is
accepted and degrades cleanly rather than crashing.

## How to reproduce

```sh
cmake --build <build> --target demo_sparkium_cli sparkium_fallback_test sparkium_cpu_bvh_test

# Does it look the same? Writes both renders and the figure under out/cpu-backend.
python3 scripts/check_cpu_backend.py --cli <build>/demo/sparkium_cli/demo_sparkium_cli

# What does it cost? Writes the table to out/backend-profile/table.txt.
python3 scripts/profile_backends.py --cli <build>/demo/sparkium_cli/demo_sparkium_cli

# Unit tests, including a CPU render with no device and a comparison against the
# compute fallback.
./<build>/test/sparkium/sparkium_fallback_test --gtest_filter='SparkiumCpuBackend.*'

# The BVH builder and the shared traversal against a brute force oracle.
./<build>/test/sparkium/sparkium_cpu_bvh_test
```

## What is not verified

- Hardware image parity is checked against the compute fallback only. No device
  with hardware ray tracing was available, so the CPU backend has not been
  compared against pipeline RT.
- D3D12 and Vulkan have not been exercised on the CPU path; the results above
  are from Metal.
- The JIT does not run, as described above.
- Textured and shader-graph scenes keep a systematic residual of roughly 0.5 to
  1.7 out of 255 that averaging does not remove. It is small enough to be
  invisible in the figure above, but it is not noise and its cause is not known.
  Until it is, the CPU backend should be treated as matching the GPU closely
  rather than exactly.

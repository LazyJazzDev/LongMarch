# TLAS pressure test

Build the `demo_tlas_pressure_test` CMake target.

`demo_tlas_pressure_test --backend vulkan` (or `d3d12` / `metal`) renders
100,000 randomly positioned and colored cubes/spheres on a large ground plane.
All bodies, including hidden ones, are simulated on the CPU with gravity along Y;
ground contact assigns a new random upward speed. A sinusoid with an eight-second
period selects 20,000 through 100,000 visible bodies, randomly moving IDs between the
visible and hidden sets. Stable instance IDs retain each body's color.

Three immutable BLASes (cube, sphere AABB, ground) are shared by the instances.
The TLAS is allocated once for 100,001 instances. Every frame uses BUILD with exactly
`visible bodies + 1 ground` instances: hidden objects are absent, not masked out.
Native UPDATE/refit cannot change the previous instance count. The allocation
capacity in `TopLevelAccelerationStructureDesc::instance_count` is now an upper
bound for BUILD; UPDATE still requires the preceding build's actual count.
This scene tests instance transforms and membership, not deforming BLAS vertices.

Compute ray queries shade one primary ray per pixel and a shadow ray for lit hits
under a fixed directional light. A fullscreen pass displays the result. P toggles
VSync; the title reports FPS, visible body count and total TLAS instance count.
This is a standalone executable, separate from the rhi_hello module menu.
It supports `--backend auto|d3d12|vulkan|metal`, `--headless`, `--frames N` and `--help`.
Without arguments it opens a window. Headless defaults to 3000 measured frames.

```sh
demo_tlas_pressure_test --headless --backend vulkan --frames 3000
```

Headless runs at 1280x720 with three frames in flight and 60 warmup frames, using
seed 87 and a fixed 1/120-second simulation step for reproducible scene evolution.
Window mode advances with elapsed time, using physics substeps capped at 1/120 s.
The report includes completed-work FPS, actual visible-count range and CPU time
spent on simulation/instance packing, TLAS command recording, rendering command
recording, submission and frame waiting. CPU recording times are not GPU durations;
use a GPU profiler to separate TLAS build from ray traversal/shading costs.

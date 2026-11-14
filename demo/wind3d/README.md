# Wind3D — asynchronous CUDA wind tunnel

A three-dimensional incompressible flow demonstration around a procedural
airliner model. CUDA advances the fluid on a worker thread while an RHI frontend
renders the latest completed state. NVIDIA CUDA is required; the Windows
frontend supports Vulkan and D3D12.

```powershell
cmake --build cmake-build-release --target demo_wind3d
cmake-build-release/demo/wind3d/demo_wind3d --backend vulkan
cmake-build-release/demo/wind3d/demo_wind3d --backend d3d12 --grid 256
```

The panel controls wind speed, incidence, collective flaps, differential
ailerons, elevator and rudder. Drag outside the panel to orbit, or use the
camera sliders; the mouse wheel zooms. Pausing the solver leaves the camera and
window responsive. Reset starts a new flow history with the requested controls.
Changes take effect at a solver-step boundary. The displayed model, flow and
loads always belong to the same completed snapshot, so a slow backend can lag
behind the requested panel values.

The model is a generic scaled narrow-body airliner, 6 m long with a 6.04 m
span. It has a constant-section cabin, rounded forebody, upswept tailcone,
continuous belly/wing-root fairing, cockpit glazing, passenger windows and doors.
The low-mounted main wing has sweep, dihedral, camber and raked tips. Two
stationary nacelles and pylons are part of both the visible geometry and solid
boundary. They are unpowered solid nacelles with a dark fan-face representation;
this is not an engine mass-flow or thrust model.

Inboard Fowler-type flaps extend rearward as they deflect. Outboard differential
ailerons, the horizontal stabilizer's elevators and the vertical stabilizer's
rudder rotate about their actual swept three-dimensional hinge lines. Control
surfaces have closed hinge, root and tip faces. Muted tan, blue and green surfaces
identify flaps, ailerons and elevators; the fin has a blue livery. The explicit
mesh and occupancy test share the complete geometry specification and hinge
transforms in `aircraft.h`. Rendering does not reconstruct an implicit surface.
The fluid still voxelizes the solid at its own grid resolution.

Use `--model-only` to inspect the aircraft without flow/load vectors. Camera
arguments `--yaw`, `--pitch` (radians) and `--distance` (metres) support repeatable
inspection views; normal mouse orbit and zoom remain available.

## Coordinates and visualization

- World +x is downstream, +y is up and +z is starboard. Positive incidence raises
  the nose. Flaps and elevator deflect trailing edges down for positive input;
  positive aileron input deflects the starboard trailing edge down and port up.
  Positive rudder input moves its trailing edge toward -z.
- Three-dimensional streamlines integrate the latest cell-centred velocity
  using midpoint tracing. They are instantaneous streamlines, not particle
  histories. Blue is slower, yellow is faster; camera rotation reveals their
  three-dimensional shape.
- Cyan is the integrated fluid force; magenta is the integrated moment about
  the fixed body origin. The vectors share a raised display origin connected
  to the body origin by a grey line. Their lengths are compressed as
  `3 * magnitude / (100 + magnitude)`; directions are unchanged. The panel
  reports all three components in N and N m. Moments follow the right-hand rule.
- The corner triad is red +x, green +y and blue +z.

## Numerical method and limits

The domain is 16 × 8 × 12 m, spanning x = [-6, 10], y = [-4, 4], z = [-6, 6].
The default MAC grid is 192 × 96 × 144. `--grid` sets the x resolution (64–640,
multiple of 16); the other dimensions retain this aspect ratio. Velocity lives
on staggered faces and pressure at cell centres. Each step performs:

1. Rebuild the solid mask when the aircraft geometry changes. Also sample each
   pressure-cell connection at h/8 spacing to block velocity faces crossed by
   surfaces whose neighbouring cell centres are both fluid. Connections still
   marked open are checked against the complete deflected foil solids using
   segment clipping and convex-section minimization, so a thin trailing edge
   between sample points is not omitted. This uses the current convex NACA
   profiles and a 1 micrometre grazing-contact tolerance; it does not remove
   the stair-step error of representing a curved wall on Cartesian faces.
2. Midpoint characteristic tracing, bounded MacCormack transport correction,
   and explicit viscous diffusion. Tracing, forward/reverse transport and limiter
   bounds reconstruct tangential samples using stationary-wall ghost values.
   Each adjacent pressure-face flag contributes half of the MAC dual-face
   reflection, retaining the open part of a partially blocked face. This is a
   local Cartesian reconstruction, not a curved-wall cut-cell scheme. Wall
   coverage is interpolated in the tangent plane to avoid half-cell jumps
   when a traced position moves along a wall edge.
   Tangential diffusion uses stationary no-slip
   ghost velocities across blocked faces, including thin walls with fluid cells
   on both sides. `--first-order` disables the transport correction.
3. Enforce stationary no-penetration face velocities at solid boundaries.
4. Solve the fine-grid pressure equation with multigrid-preconditioned conjugate
   gradients, warm-started from the previous pressure. The true `b - A p`
   residual is recomputed every iteration; default relative tolerance is `1e-4`.
5. Subtract the pressure gradient and measure the remaining divergence.
6. Integrate force, torque and diagnostics on the full solver grid using GPU
   block reductions. Publish a display field capped at 192 × 96 × 144;
   downsampling the display field does not downsample the load integration.

The air density is 1.225 kg/m³; the effective kinematic viscosity is 0.005 m²/s.
The time step is bounded by advective CFL 0.5 per coordinate and an explicit
diffusion bound. The CFL maximum includes every MAC velocity component,
including padded faces used by interpolation, rather than cell-centred averages
that can cancel opposite face velocities.
The inlet and lateral far-field faces impose the wind; the outlet has zero
pressure. Pressure force is integrated over the same blocked velocity faces used by the
pressure operator. For a thin wall with fluid on both sides, their pressure
contributions form the pressure jump; no solid cell is required for that load. A first-order
tangential wall-shear estimate uses the same pre-step velocities and viscosity
as the explicit diffusion update, and each
surface force contributes `r × F` to the moment. GPU fields use single precision;
GPU reductions use single precision and their block totals accumulate on the
CPU in double precision, with no double atomicAdd.

This is a qualitative laminar demonstration. It has no turbulence model,
compressibility, ground effect or moving-wall actuator dynamics. Controls are
quasi-static boundary changes, not a simulation of the transient swept volume
of a moving flap. Transport limiting remains dissipative, the pressure solve
has finite residual, and voxel boundaries underresolve thin trailing edges and
small deflections, especially at coarse grids. Higher grids improve geometric
resolution at substantially higher step cost. Grid 640 uses 640 × 320 × 480
cells at 2.5 cm spacing and approximately 8.1 GiB of explicit solver/display
allocations, before CUDA/runtime and graphics overhead. Mesh smoothness does not imply
equally fine fluid resolution. Do not interpret the loads as calibrated
aircraft performance data.

The absence of a turbulence model does not itself forbid unsteady vortices.
However, the effective viscosity here is over 300 times that of room-temperature
air (about 1.5e-5 m²/s), the default cell width is about 8.3 cm, and the
limited transport scheme still damps unresolved structures. This setup does
not resolve a realistic turbulent aircraft wake. Instantaneous streamlines can
show resolved recirculation, but do not convey particle histories or temporal
fluctuations as directly as pathlines or vorticity visualization would.

## Asynchronous ownership

A single worker owns the CUDA stream and all fluid allocations. It reduces full-resolution
loads/diagnostics on the GPU, copies a completed display field to a host snapshot, and publishes
an immutable `shared_ptr` under a short mutex. The frontend acquires that pointer
without waiting for a CUDA step. Only the latest snapshot is retained by the
publisher; intermediate snapshots can be dropped. Aircraft meshes are reused
until controls change. The renderer uploads the mesh and derived streamlines,
not the full volume, and waits only for its previous graphics submission.

This version deliberately uses a host snapshot rather than CUDA/graphics shared
buffers: the frontend has no GPU timeline dependency on unfinished CUDA work.
CPU copies consume bandwidth and both workloads still share a GPU, so independent
submission does not guarantee a fixed display frame rate. Shutdown joins the
worker and completes graphics use before releasing resources. Reset generations
prevent an obsolete in-flight step from publishing over a newer reset request.

## Bounded runs and validation

```powershell
demo_wind3d --validate --grid 128 --steps 80
demo_wind3d --validate --grid 256 --steps 80 --flap 30
demo_wind3d --headless --frames 300 --capture out/wind3d/flow.ppm
demo_wind3d --backend d3d12 --frames 600 --debug
```

`--validate` cross-checks explicit mesh vertices against the solid-boundary
function under combined extreme controls. It also exercises the actual CUDA
diffusion and force kernels on a thin stationary plate for all six normal/tangent
axis combinations, checking that fluid momentum loss matches analytical wall
friction and the integrated force. It checks finite fluid state, requires
true pressure residual below the requested tolerance, and checks the divergence
after projection (twice the requested relative tolerance, with a 1e-5 s⁻¹
roundoff floor). `--pressure-tolerance` and `--pressure-cycles` set the residual
target and maximum PCG iterations. Exhausting the iteration limit is visible in
the panel and fails validation; it is not silently treated as convergence.

`--validate-time SECONDS` runs to a specified physical time instead of a step
count, for comparable grid studies. For example:

```powershell
demo_wind3d --validate-time 1 --grid 384 --pressure-tolerance 0.0001 --flap 20
demo_wind3d --validate-time 1 --grid 512 --pressure-tolerance 0.0001 --flap 20
```

Validation also checks pause acknowledgement, reset while paused, applying all
controls, resume, and retained snapshot immutability. A failed check returns
exit code 2. Use `--help` for parameter bounds.
Startup has an impulsive loading transient; compare evolved states at equivalent
physical times, not just equal step counts. `--headless` renders offscreen and
adds an 8 ms pacing delay, so its reported FPS is not a window benchmark.
Use Nsight for GPU and full-frame profiling; this demo adds no timestamp query
interface.

## Single-node CUDA batch task

`wind3d_task` runs the same solver without the renderer or graphics dependencies.
It supports 1–8 consecutive visible CUDA devices (starting at device 0). The
multi-GPU path partitions the global grid into contiguous index ranges, places
managed-memory pages on their owning GPUs, and accesses neighbouring ranges
through peer mappings. Every physical boundary remains a global-domain boundary;
partition boundaries introduce no artificial walls. CUDA events join each
producer stage before its consumers run on any GPU. This path targets a
peer-connected NVLink node with concurrent managed-memory access.

Transport, masks, boundary enforcement, projection and full-grid diagnostics run
across the selected GPUs. The globally coupled pressure PCG/multigrid solve runs
on GPU 0, with pressure stored there to avoid repeated remote accesses during
iterations. This preserves the original pressure operator, tolerance, warm start
and true-residual checks. It also limits strong scaling: eight GPUs do not imply
an eightfold speedup. The coarsest multigrid sweeps use one synchronized CUDA
block instead of 64 separate kernels, preserving the symmetric preconditioner.
On even-width levels, red/black smoothing launches only the cells of the active
colour, so half the CUDA lanes are no longer discarded by a parity branch.

Build **on a compute node**, with CUDA, CMake and Ninja available:

```sh
cmake -S . -B cmake-build-release -G Ninja \
  -DLONGMARCH_WIND3D_STANDALONE=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build cmake-build-release --target wind3d_task -j 8
cmake-build-release/demo/wind3d/wind3d_task \
  --gpus 8 --grid 384 --steps 80 --warmup 10 --no-export --validate
```

On an allocated eight-GPU node, `ctest --test-dir cmake-build-release -R wind3d
--output-on-failure` compares complete 80-grid snapshots from 1/2/4/8 GPUs bit
for bit in both per-step and final-only export modes, and checks geometry,
projection, wall friction and lifecycle behavior.

The standalone configuration does not require vcpkg, Vulkan, a display server or
assets. It is separate from the graphics configuration; reconfigure the build
directory when switching between them. Architecture 90 targets H100/H200; use the
architecture of the allocated node for other GPUs.

On the envelope cluster, submit the provided single-node job:

```sh
meshy-envelope submit demo/wind3d/task.star
```

`task.star` requests one entire eight-GPU node and runs `task.sh` there. The shell
script builds the executable and passes through its arguments. It also searches
`$HOME/.local/bin` for shared-home CMake/Ninja installations. Set
`WIND3D_CUDA_ARCHITECTURES` or `WIND3D_BUILD_JOBS` through the job environment to
override native architecture detection or build parallelism. Job logs contain
per-step diagnostics and a final machine-readable `RESULT` line. The process
returns nonzero on CUDA errors, nonfinite state, pressure nonconvergence,
excessive post-projection divergence, or an incomplete timing sample.

`--steps` bounds the run exactly; `--warmup` excludes the first steps from the
reported mean. Timing includes transport, pressure, load/diagnostic reductions,
CPU synchronization and any requested display snapshot export, but excludes initialization,
compilation and command-line printing. Stage times are CUDA-event measurements;
the reported step time is CPU wall time. `--validate` additionally exercises
geometry, analytical wall friction, pause/reset/resume, combined control changes
and snapshot immutability. `--dump PATH` writes the final display field as three
native int32 dimensions followed by `Cell` records (four float32 values: u, v, w,
pressure); the field is the full cell-centred grid up to grid 192, and is
spatially downsampled above that size.

The supplied cluster job uses `--no-export`: all fluid updates and full-grid
force/divergence/CFL diagnostics still run, but the visualization volume is not
copied to the CPU every step. This avoids unused display work in a batch task.
The GUI and task runs without this flag keep exporting every snapshot.
`--dump` with `--no-export` exports the final field only, for numerical comparison;
that final step includes the export cost, so omit `--dump` in timing runs.

For a fair speedup measurement, run `--gpus 1` and `--gpus 8` on the **same
reserved node**, with identical grid, step count, warmup, transport scheme and
pressure tolerance and export mode. Repeat in alternating order. Compute speedup as single-GPU
mean step time divided by eight-GPU mean step time. Do not compare different
resolutions or weaken the pressure tolerance to improve this ratio. The original
graphics demo remains single-GPU by default.

## Live web console on the login node

The CUDA-only `wind3d_web` worker supports the same wind speed, incidence,
flap, differential aileron, elevator and rudder controls as the native demo,
plus pause/resume and reset. The browser displays the actual aircraft mesh,
instantaneous streamlines, full-grid force and moment, and solver diagnostics.
Geometry and streamline construction are shared with the native renderer in
`visualization.h`; the web console does not substitute a different aircraft.

On the login node, start the relay (Python 3 standard library only):

```sh
python3 demo/wind3d/web/server.py \
  --directory "$PWD/out/wind3d-web/live" --port 10510
```

Then submit `meshy-envelope submit demo/wind3d/web.star`. It builds and starts
the worker on one eight-GPU node at grid 384, with an eight-hour job limit.
The login service and GPU worker must use the same shared directory. The relay
binds to `0.0.0.0` by default; use `--bind 127.0.0.1` when accessing it solely
through an SSH tunnel. This is an interactive cluster service for users who can
reach the chosen port, not a multi-tenant authenticated web application. POST
controls require same-origin JSON and are range/finite-value checked on both
sides. Assets, including Three.js r160 (MIT), are served locally without a CDN.

Only display data crosses the node boundary:

- The solver volume remains on the compute node. Its bounded display snapshot
  is consumed there to trace the same 84 streamlines as the original renderer.
  No velocity/pressure volume is written into the shared mailbox or served by HTTP.
- At most twice per second, the worker atomically replaces `frame.bin`: a padded
  JSON metadata header and float32 line positions/colors. Typical raw display
  frames are about 650 KB. Old line frames are discarded, not queued.
- The original aircraft mesh is published only when its geometry changes.
  Immutable revision IDs pair each frame with the correct mesh; the latest 32
  meshes are retained for clients finishing an in-flight fetch. HTTP gzip and a
  bounded cache reduce browser transfer sizes; ETags skip unchanged frames.
- Force, moment, time, step, pressure residual and the **applied** controls are
  from that same completed simulation snapshot. Requested controls are separate;
  acknowledgements require matching controls, reset generation and pause state.
  Edits while paused take effect on resume or reset, as in the native demo.

The browser preserves its last image on interruption and explicitly marks it
stale after five seconds. A worker failure/stoppage is shown rather than being
presented as live simulation. Starting a replacement worker in the same mailbox
uses unique mesh revisions and applies the persisted requested controls. After
the envelope time limit, resubmit `web.star` to begin a new flow history.

Validation: `python3 demo/wind3d/web/test_server.py` checks API persistence,
invalid controls, same-origin enforcement, ETags and the restricted file surface.
The worker can also be run as `wind3d_web DIRECTORY GRID GPUS` on an allocated
GPU node for an independent mailbox integration test. Runtime mailboxes, process
logs and browser screenshots belong in ignored `out/`, not in the source tree.

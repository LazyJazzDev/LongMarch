# Adaptive-grid implementation and acceptance

## Objective and constraints

Port numerical improvements from `~/LongMarch`, then add GPU-parallel adaptive
meshes that materially reduce wall time without changing steady force/moment
conclusions across the supported control envelope. The user accepted a 2%
relative error target for principal nonzero load components. Near-zero components
need nondimensional absolute tolerances and comparison against baseline sampling
uncertainty; changes in control-response trends must also be preserved.

Experimental jobs submitted for this work may occupy at most **8 GPUs in total**.
Other jobs owned by the same user are outside this budget and must not be altered.
Track this experiment's submissions explicitly, including queued requests, so
simultaneous starts cannot exceed the budget. The former live eight-GPU job
2000941 and port-10510 relay were stopped at the user's request. Do not restart
them as part of an experiment without a new reason/authorization.

## Current evidence

- Monotone cubic transport from upstream b20f9123 is available via `--cubic` in
  task and sweep drivers. The 1/2/4/8-device uniform-grid test verifies bitwise
  identical fields and per-step loads, including full/final export paths.
  This is correctness evidence, not a steady-state accuracy or speed result.
- `amr_topology` constructs a rectangular-domain octree of equal-cell-count
  blocks. Regridding supports refinement/coarsening, face-adjacent 2:1 balance,
  deterministic ordering, and rollback on capacity exhaustion. The coverage
  test checks every finest-level voxel for exactly one owner. It is not yet
  connected to fluid advancement or the live display.
- `amr_mesh` constructs unique coarse/fine face patches and flat signed CSR
  cell incidence suitable for GPU gathers without flux atomics. CPU tests verify
  exact per-cell face closure, div(x,y,z)=3, global random-flux cancellation,
  and symmetry/positive energy of the two-point graph operator. These do not
  prove pressure consistency: coarse/fine cell centres are not aligned in
  tangential directions, so a two-point graph alone is not an acceptable
  second-order composite pressure discretization. Transverse reconstruction
  and manufactured pressure tests remain mandatory before solver integration.
- `amr_gradient` reconstructs normal pressure derivatives at coarse/fine face
  centres from a scaled quadratic least-squares stencil over two cell rings.
  Tests verify all ten polynomial basis fields through degree two, and cubic
  max-error reduction of 0.137310 -> 0.034328 when cell width halves.
- `amr_projection` is a CPU all-fluid reference with fixed boundary fluxes and
  zero pressure correction at the outlet. Right-preconditioned BiCGStab handles
  the nonsymmetric reconstruction; convergence checks use the actual residual.
  The manufactured-velocity test converges in 59 iterations to 7.69e-10 relative
  residual and checks independent divergence reduction, fixed boundary fluxes
  and no mutation on nonconvergence. This is not yet the GPU projection, a
  cut-cell aircraft solve, or an energy/long-time stability validation.
- `amr_gpu_projection` assembles the same composite operator and runs sparse
  products, BiCGStab vector updates and double-precision reductions on one CUDA
  device, with persistent device matrices/workspace. Job 2000952 on H200 passed
  all five AMR tests and CUDA memcheck (zero errors). Corrected CPU/GPU velocity
  max difference was 2.34e-9; GPU true relative residual was 9.83e-10 in 62
  iterations. Tests include workspace reuse, zero RHS, failed-build rollback
  and iteration-limit rollback. This all-fluid prototype still copies input/
  output at each call, uses Jacobi rather than a scalable multilevel
  preconditioner, and is not yet connected to aircraft advancement or multi-GPU
  halo exchange. No acceleration claim follows from these correctness tests.
- Embedded boundaries now use 6x6 open-area sampling, quantized 0..255
  apertures, the 1/8 sliver cutoff, and thin-wall centre-connection probes.
  A cell is active if any face is open, matching the uniform cut-cell rule.
  CPU and GPU operators weight fluxes by aperture and pin inactive rows;
  closed faces are zeroed only after successful projection. Coarse/fine faces
  intersecting the body are rejected to enforce a fine body-surface region.
  CPU box tests cover partial apertures, solid cells, projection, and rollback;
  the actual aircraft with flap/elevator deflection passes constant-pressure
  force and moment cancellation. Surface friction and advancing an aircraft
  flow are still not implemented in the production adaptive path. Job 2000954
  passed all six tests and CUDA memcheck, including embedded CPU/GPU velocity
  agreement to 3.13e-9 and true relative residual 6.98e-10.
- `amr_transport` supplies a CPU sparse-MAC sampler and midpoint tracing.
  Coincident native faces retain their values; cross-level virtual faces use
  bounded linear reconstruction, with aperture-weighted trilinear sampling.
  Monotone cubic transported values are optional; tracing stays trilinear.
  Tests preserve constant flow and exactly translate analytic shear across
  levels; smooth-wave squared transport error was 0.0182403 (linear) versus
  0.000097015 (cubic) in the tested uniform region. These isolated tests do not
  establish equivalence of near-wall transport to the uniform solver.
- `AdvanceReference` now couples CPU transport, boundary conditions and
  projection, recovers physical pressure using rho/dt, and integrates pressure
  forces/moments. Three small steps around the actual aircraft on a two-level
  mesh pass a finite-state/projection smoke test. This CPU reference includes
  slip extension, bounded MacCormack correction, molecular diffusion and an
  outlet sponge. Production solver integration, dynamic regridding and
  long-time load equivalence remain required.
- Two-layer slip extension and load-only Spalding friction are now connected
  to the CPU reference advance. The wall law is factored into `wall_model.h`
  for both backends. Wall loads use unweighted pre-step MAC centre velocities
  after extension, the cut-cell wetted-area vector, half-cell wall distance,
  and centre-based shear moment arms, matching the uniform formulation.
  The planar-wall test agrees with an independent double-precision bisection
  solution (force 0.0256771 N, Mz -0.00320964 N m), checks zero normal-flow
  friction, and passes ASan/UBSan. Job 2000957 passed all nine AMR tests;
  the uniform CUDA field dump is bitwise unchanged and all eight sampled force
  vectors are identical after sharing the wall law. This is a refactor
  regression check, not a steady-load equivalence claim.
- Corrected transport reduces the smooth-wave squared error from 0.0182403
  to 0.000276643 with linear sampling; discontinuous transport stays bounded.
  Diffusion matches the discrete sine decay, rejects unstable explicit steps
  without changing state, and preserves constant tangential slip flow. The
  outlet sponge is checked independently. The combined aircraft smoke test
  has projection residuals below 1e-8 for three steps. These tests do not
  establish conservative viscous fluxes across refinement interfaces, native
  far-field boundary equivalence, or the 2% steady-load acceptance criterion.
- `GpuProjection::ProjectDevice` accepts caller-owned disjoint device buffers,
  keeping full velocity and potential arrays on the GPU. The host-vector API
  uses the same solver. Job 2000960 passed all 11 tests and CUDA memcheck
  (zero errors), including CPU agreement, numerical-failure rollback, invalid
  sizes and zero-divergence solid-face clearing. Scalar Krylov reductions
  still synchronize with the host.
- `GpuTransport` implements sparse-MAC midpoint transport, optional monotone
  cubic sampling, two-layer slip extension and bounded MacCormack correction
  in persistent device buffers. Leaf lookup uses a level-wise binary search;
  cell reconstruction and face transport run as separate parallel kernels.
  Job 2000964 passed all 12 tests and both projection/transport CUDA memchecks
  (zero errors). Constant, smooth and discontinuous fields on a two-level
  hierarchy, with and without an embedded box, agree with CPU transport to
  1.11e-16 across linear/cubic and corrected/uncorrected modes. Tests also
  cover in-place buffers, invalid inputs and failed-build recovery.
- `GpuTransport::AdvanceDevice` now combines transport, molecular diffusion,
  outlet sponge, boundary prescription and projection on resident fields.
  GPU pressure/friction load reductions return 12 scalars; full velocity and
  physical pressure stay on the device. Numerical failure leaves caller state
  unchanged. Job 2000967 passed all 13 tests and projection, transport and
  full-timestep CUDA memchecks (zero errors). Tests cover freestream,
  viscous shear with a shorter sponge, three aircraft steps and invalid steps.
  Same-input CPU/GPU velocity differences are at most 2.20e-9 in these cases.
  Independent aircraft trajectories differ by up to 8.07e-6 in velocity and
  1.29e-4 in pressure after small input perturbations; same-input step checks
  isolate this from operator parity (aircraft error <=1.84e-10). Independent
  pressure/friction load component differences are <=2.43e-6. Tests bound both
  local operator error and accumulated state differences separately. This is
  startup validation only: nonlinear sensitivity still requires long-time
  load statistics, and neither the 2% uniform-reference gate nor an end-to-end
  speedup has been established.
- `TransferVelocity` supplies a host-side conservative regrid path. It sums
  open-face flux on coarsening and uses bounded, flux-recentered reconstruction
  on refinement; newly introduced interior faces use the old sparse sampler.
  Changes to fluid-adjacent cut-face locations, areas or apertures are rejected
  transactionally, preserving wall spacing even when aggregate fluxes match. Direct strict C++
  and ASan/UBSan tests pass: constant preservation, moved refinement, unchanged
  embedded surfaces, changed-body rejection, no new extrema and failure rollback.
  Refine/coarsen face roundtrip errors are 1.11e-16; post-transfer projection
  gives max divergence <=1.76e-9 in the tested fields. The transfer itself is
  not divergence-free: its caller must project, and that correction potential
  must not be interpreted as physical pressure. Load equivalence through
  regrids remains pending.
- `GpuFlow` owns the resident flow and replaces it transactionally on regrid:
  temporary topology/geometry and GPU operators, conservative transfer,
  projection, then commit. It retains physical time and step count, increments
  grid generation, and invalidates/zeros pressure until the next physical step.
  Failed initialization or regrid preserves the old flow. Explicit snapshots
  are copies; no mutable internal geometry/device buffers are exposed.
  Lifecycle testing found that equal aggregate apertures could still permit a
  coarser wall distance; transfer now rejects that case, with a regression test.
  Job 2000973 passed transfer and runtime GPU lifecycle tests and CUDA memcheck
  (zero errors); the preceding full run passed the other 13 tests. Tested
  refinement/coarsening preserves freestream and gives maximum post-regrid
  divergence 2.10e-11 around an embedded box. Production UI integration,
  multi-GPU ownership and steady-load/performance validation remain outstanding.
- `ProtectedTopology` and `PlanRefinement` add body protection and flow-driven
  hierarchy selection. Protected physical boxes stay at the finest spacing;
  existing cut-cell blocks retain their level. A dimensionless cell-scale
  velocity variation indicator requests one-level refinement, retains levels
  between two thresholds, and permits one-level coarsening below the lower
  threshold. Configurable neighboring-block padding and 2:1 balancing provide
  spatial buffers. Capacity failures leave the plan and report unchanged.
  `GpuFlow::Adapt` runs this policy at an explicit adaptation interval and uses
  transactional regridding. It currently downloads velocity for the host policy;
  thresholds are heuristic defaults, not a certified 2% load-error bound.
  Job 2000974 passed policy, transfer and GPU lifecycle tests and CUDA memcheck
  (zero errors); direct strict C++ and ASan/UBSan policy tests also passed.
  The aircraft batch runner schedules this policy; production web integration,
  performance tuning and the uniform-reference acceptance matrix remain pending.
- `wind3d_task --adaptive` now runs the actual aircraft with conservative
  world-space bounds obtained from the geometry's rotated body-space broad
  phase, plus two finest cells of padding. `--amr-uniform` runs the same new
  operators on a uniformly fine hierarchy. Both retain native geometry and
  optimized viscosity/wall parameters; a GPU cell-speed reduction drives CFL
  time steps. All backends accept initial speed/angle/control deflections,
  `--max-dt`, and step-complete six-component `--csv` load output. Adaptive
  batch mode currently requires one GPU and `--no-export`; web/display and
  adaptive dumps are not integrated. Native pressure CLI controls apply to
  the native multigrid solver; adaptive projection currently uses 1e-8 and
  at most 2000 BiCGStab iterations.
  Job 2000975 passed transport/advance/lifecycle regressions and CUDA memcheck;
  native default fields remained bitwise identical and all eight force vectors
  unchanged. Three backends completed six steps at h=0.25 and dt=0.0002.
  At step six, AMR-uniform differs from native by about 0.13% in Fx and 0.06%
  in Fy. Adaptive uses 39,168 initial cells versus 98,304 uniform cells and
  changes to 43,648 after refinement. These are startup samples, not load
  acceptance. Current timings are unacceptable for the speedup goal: native
  averaged 2.09 ms/step, AMR-uniform 46.6 ms, and ordinary adaptive steps about
  37-42 ms; one regrid cost 0.451 s. Different projection tolerances and this
  short startup sequence prevent a formal equal-accuracy performance claim.
  Jacobi-preconditioned adaptive projection takes roughly 170-230 iterations;
  profiling and a scalable preconditioner are the next priorities. Job 2000976
  also verified changed initial control parameters and complete CSV files.
- Adaptive pressure projection now defaults to a fixed linear aggregation
  V-cycle preconditioner. Coarse matrices are Galerkin sums of the current
  operator; inactive cells are excluded from aggregation, two damped Jacobi
  sweeps run before/after correction, and the small coarsest system uses a
  pivoted dense inverse. Two CUDA Graphs replay the fixed preconditioning
  operations inside BiCGStab. `GpuProjection::Build(..., false)` retains the
  Jacobi baseline. True residual checks and numerical-failure rollback remain.
  Job 2000978 passed pressure, aircraft advance and regrid lifecycle tests;
  CUDA memcheck reported zero errors. Small composite cases dropped from
  62/57 iterations to 9, with CPU/GPU field differences below 7.80e-9. Aircraft
  cases dropped to roughly 15-19 iterations. Across the six-step diagnostic,
  the largest load-component difference from the previous preconditioner was
  6.39e-4; this does not replace steady-load acceptance.
  Uniform-AMR ordinary steps improved from 46.6 to 18.7 ms; adaptive ordinary
  steps improved to about 16 ms, while regrid overhead still dominates that
  short run's overall mean. Stage timing (job 2000979) measured adaptive
  transport 8.81 ms, pressure 7.20 ms and loads 0.049 ms. Native remains much
  faster in this small-grid case. Transport neighborhood caching, further
  pressure optimization and less expensive regridding are still required.
- GPU transport now precomputes each block's MAC sampling sites and refreshes
  two value halos after slip extension and slope reconstruction. Interpolation,
  aperture weighting, bounded coarse/fine reconstruction and correction limits
  keep their existing rules. `GpuTransport::Build(..., false)` retains direct
  sampling for comparison. The cache adds `72 * (block_cells + 5)^3` bytes per
  leaf (site indices and two double value fields), about 34.1 MiB for 682 leaves
  with four cells per block; geometry is rebuilt when the hierarchy changes.
  Job 2000982 passed transport, full advance and regrid lifecycle tests, with
  zero CUDA memcheck errors. Constant, smooth and discontinuous inputs include
  large and negative tracing steps, cubic/linear sampling and corrected/plain
  advection. Cached/direct GPU outputs agreed exactly in these cases; CPU
  differences were at most 2.23e-16. Across the six-step aircraft diagnostic,
  maximum load-component changes were below 1.41e-9 for uniform-AMR and 9.80e-11
  for adaptive. Adaptive ordinary transport fell from 8.81 to 0.983 ms and
  ordinary total steps from 16.1 to 8.15 ms; uniform-AMR total steps fell from
  18.7 to 7.64 ms. Pressure now dominates (adaptive 7.09 ms). These small-grid
  startup results exclude regrid steps from the ordinary-step averages and
  still do not establish steady-load acceptance or speedup over native.
- The longer startup scaling diagnostic (job 2001076, 100 steps, first 20
  excluded, dt=0.0002, no regrids) confirms the performance target is not met.
  Mean step times for native / AMR-uniform / adaptive were 1.26 / 7.77 / 7.47 ms
  at grid 64 and 2.81 / 30.98 / 18.25 ms at grid 128. At grid 128 the adaptive
  hierarchy used 212,992 cells versus 786,432 uniform cells. The simulation
  reached only 0.02 physical seconds, and native/AMR pressure tolerances differ;
  these are bottleneck diagnostics, not equal-accuracy or steady-load acceptance.
  In particular, fewer cells and faster execution than AMR-uniform do not yet
  constitute speedup over the original solver.
- Non-binary spacing tests exposed a CPU/GPU discrepancy in virtual MAC sample
  coordinates: separate multiply/add versus GPU FMA could select opposite
  reconstruction cells at a coarse/fine boundary. Explicit FMA coordinates in
  CPU and GPU sampling and correction bounds remove this mismatch. A CPU-only
  reproduction changed from the failing value to the GPU value by changing
  only sampling-coordinate evaluation. Job 2001211 passed five transport,
  full-step and regrid tests and CUDA memcheck. Widths 1, 1.3 and 2/3, with and
  without embedded boundaries, now differ by at most 4.45e-16 between CPU/GPU;
  cached/direct GPU results agree exactly. Acceptance tolerances were unchanged.
- CUDA Graph node profiling (job 2001214, grid 128, 12 startup steps)
  identified V-cycle smoothing as 46.3% of GPU kernel time, followed by sparse
  multiplication (11.7%) and residual evaluation (10.0%). Host dot-product
  transfers also wait for queued work; their API duration must not be interpreted
  as pure transfer overhead. The first Jacobi sweep of each V-cycle level now
  directly initializes `0.65 * rhs / diagonal`, eliminating the zero-vector
  matrix traversal and preceding memset while retaining the same sweep count.
  Job 2001215 passed projection, full-step and regrid tests plus CUDA memcheck;
  all six aircraft load components agreed within 1e-8 over 12 steps. Unprofiled
  ordinary steps measured 17.26 ms, including 13.21 ms pressure. The older trace
  includes profiler overhead, so these figures alone do not establish an
  optimization speedup ratio or the full performance goal.
- A sliced-ELL sparse layout experiment (job 2001216) preserved exact pressure
  fields and iteration counts against CSR in free and embedded test cases, and
  passed full-step/regrid tests and CUDA memcheck. It did not improve the
  grid-128 startup diagnostic: pressure measured 14.47 ms versus 13.21 ms in
  the preceding CSR run, with matching iteration counts and loads. This is a
  short sequential-run comparison, not a controlled performance acceptance
  measurement. The experimental layout was removed; CSR remains the default.
  The built task from this experiment must be rebuilt from current sources
  before further baseline measurements.
- Pressure CSR products, smoothing and residual kernels now use four threads
  per row with subgroup shuffle reduction. Matrix storage and convergence
  tolerances remain unchanged; `GpuProjection::Build(..., true, false)` retains
  scalar rows for comparison. Job 2001219 passed projection, full-step, regrid
  and CUDA memcheck tests. Free/embedded field differences from scalar rows
  were below 1.25e-14. In two alternating, same-device 30-step aircraft pairs
  (grid 128, first 10 steps excluded), pressure fell from 13.02/13.04 ms to
  10.06/10.08 ms, while total steps fell from 17.07/17.08 ms to 14.11/14.33 ms.
  Both variants averaged 19.4 pressure iterations. Maximum component changes
  were 2.07e-5 in force and 1.0e-5 in moment over the 30 steps. These controlled
  startup comparisons support this kernel optimization, not steady-load
  acceptance or speedup over the original uniform backend.
- Conservative remapping uses hash lookup for integer face keys and ancestor
  flux sums. Accumulation still follows source-face order; parent correction
  order, wall-resolution checks and failure rollback are unchanged. Strict
  CPU transfer tests and ASan/UBSan pass. In a local CPU diagnostic with 212,992
  source cells and 56 additional leaf blocks, transfer time fell from 0.649 s
  to 0.493 s and constant-flow output matched the ordered-map baseline bitwise.
  This is a remapping microbenchmark, not a GPU or steady-load speedup claim.
- Long native batch runs exposed dropped samples when the task polled only
  `Latest()` and its consumer thread was descheduled. Batch tasks now enable
  a two-snapshot queue and consume every step through `TakeNext()`. Backpressure
  waits before the next solver step, outside its timing interval. Interactive
  consumers retain the default latest-snapshot behavior. Reset clears queued
  old-generation snapshots; destruction wakes a producer blocked by a full
  queue. Job 2001223 passed slow-consumer, reset, shutdown and load-history tests,
  native lifecycle validation, and bitwise before/after field-dump comparison.
  The restarted 30,000-step baseline completed with contiguous CSV samples;
  its original 12,362-step prefix retained identical loads and clocks. This
  sampling regression does not by itself establish steady-load acceptance.
- Completed six-second grid-128 native and AMR-uniform runs (jobs 2001223 and
  2001221) do not establish the 2% gate. In the common 2-6 second window,
  native mean Fx/Fy/Mz were 17.8371 / 13.3289 / -15.0727; AMR-uniform means
  were 17.7253 / 12.6388 / -13.7308. Differences relative to native magnitudes
  are -0.63% / -5.18% / +8.90%. Native Fy and Mz half-window drift was
  0.9555 / 1.1596; AMR-uniform drift was -0.5783 / 0.5634. Sampling uncertainty
  and nonstationarity prevent interpreting those differences as settled bias,
  but they also prevent claiming equivalence. Uniform AMR already differs
  before any grid adaptation: boundary/numerical parity and longer-time
  statistics must be resolved alongside adaptive transfer. Native consumed
  14.01 wall seconds per physical second versus 158.58 for AMR-uniform in this
  window; differing pressure tolerances remain a comparison limitation.
  Physical windows use compensated sums of dt to avoid float display-clock
  drift. Dynamic and fixed adaptive-hierarchy controls are still running.
- Existing uniform-grid pressure projection is on one GPU; merely distributing
  transport is insufficient for the requested speedup.

## Remaining implementation

1. Add multi-GPU block ownership and explicit halo exchange to the existing
   coarse/fine face connectivity. Preserve shared conservative fluxes and the
   gradient/divergence operator across peer-device partitions.
2. Optimize the host-side refinement/transfer path and validate adaptation
   frequency, wake buffers and hysteresis over the control envelope. Preserve
   finest wall spacing, flux conservation and post-transfer divergence checks.
3. Optimize and distribute the composite pressure preconditioner. The
   reconstructed operator is nonsymmetric: validate manufactured solutions,
   interface flux cancellation, true residuals and partition equivalence rather
   than assuming an SPD operator.
4. Integrate transport, boundary geometry, full wetted-surface loads, display
   sampling and diagnostics with the adaptive hierarchy. Keep the uniform
   reference implementation selectable for every comparison.
5. Benchmark uniform versus adaptive at equal finest surface resolution,
   numerical options, physical settling times and averaging windows. Include
   baseline, incidence/speed changes, flap/aileron/elevator/rudder ranges,
   sign reversals and combined extreme deflections, plus pause/reset/regrid.
   Extend runs until drift and statistical uncertainty support the 2% gate;
   do not use early-transient means as steady reference values.
6. Report all six force/moment components, nondimensional coefficients, response
   increments, sampling uncertainty, active cells, memory, solver convergence,
   wall time per physical second and regrid/communication overhead. Benchmark
   without competing experiment kernels on the same GPU. Demonstrate speedup
   at matched accepted accuracy; fewer cells alone is not success.

The objective remains incomplete until the adaptive solver is integrated and
passes the full load-preservation and performance comparison. Wing-camber
changes are geometry changes, not part of the numerical-only import baseline.

## Aircraft batch comparisons

The original uniform backend remains the default. For a matched small-grid
startup diagnostic (not a steady-load or speedup acceptance test):

```sh
cmake --build cmake-build-release --target wind3d_task -j 4
cmake-build-release/demo/wind3d/wind3d_task --grid 64 --steps 100 --warmup 20 --cubic --no-export --max-dt 0.0002 --csv out/native.csv
cmake-build-release/demo/wind3d/wind3d_task --amr-uniform --grid 64 --steps 100 --warmup 20 --cubic --no-export --max-dt 0.0002 --csv out/amr-uniform.csv
cmake-build-release/demo/wind3d/wind3d_task --adaptive --grid 64 --steps 100 --warmup 20 --cubic --no-export --max-dt 0.0002 --adapt-every 25 --csv out/adaptive.csv
```

`--levels`, `--block-cells` and `--max-blocks` control the adaptive hierarchy.
All three domain dimensions must be divisible by `block-cells * 2^levels`;
no grid resolution is silently rounded. `--speed`, `--angle`, `--flap`,
`--aileron`, `--elevator` and `--rudder` set the initial case before solving.
CSV rows retain the physical step's grid generation even when a subsequent
regrid at the same time invalidates pressure; remapping corrections are never
reported as load samples. Adaptive timing includes speed diagnostics and any
scheduled regrid, while initialization is reported separately.

## Native pressure reference limitation

Tightening the native float PCG tolerance is not currently a reliable way to
produce a higher-accuracy reference. Aircraft probes at grid 128 reached a
relative residual of 6133.8 after 128 iterations at tolerance 1e-6; tolerance
1e-5 passed the first step but diverged on the second. Batch failures now print
iterations, residual and divergence before/after projection.

A separate full-fluid manufactured pressure diagnostic on 128 x 64 x 96 cells
also reproduced the problem, without cut cells. The field was
`cos(pi*(x+0.5)/(2*nx)) * (1+0.1*cos(2*pi*(y+0.5)/ny)*cos(2*pi*(z+0.5)/nz))`,
with the existing outlet anchor, RHS generated by the discrete operator and a
zero initial guess. Unmodified float PCG reached a relative residual of
6.58e6 after 128 iterations. Restarting the direction every 8 iterations
limited the final residual to 0.00181 but did not meet even 1e-4. Promoting the
same diagnostic's pressure/operator arithmetic and storage to double, without
restarts, reached 7.07e-5, 7.69e-6 and 5.05e-7 in 16, 19 and 22 iterations.
These are diagnostic results from single-GPU jobs 2001228/2001229, not changes
to the production solver or aircraft accuracy validation. Each precision
variant generated its RHS in its own arithmetic; the comparison isolates
precision sensitivity rather than bitwise-identical right-hand sides.

The V-cycle clears its fine correction and each restricted coarse correction;
stale preconditioner state does not explain this failure. Float residual
replacement plus the ordinary CG recurrence is numerically fragile for this
case. A periodic restart alone is insufficient. Establish a reliable pressure
reference, including actual projected divergence and aircraft loads, before
using tighter native tolerances to assess adaptive error.

## Pressure relaxation performance

The GPU aggregation preconditioner's two pre/post Jacobi sweeps now use weight
0.9. Grid-128 aircraft startup controls compared 1/2/4/6 sweeps and weights
0.65/0.8/0.9; more sweeps reduced iterations but increased total time. Two
alternating baseline/candidate runs on one H200 (job 2001236, 30 steps, first
10 excluded) reduced adaptive pressure time from 10.14--10.27 ms to
8.87--8.88 ms and total step time from 14.33--14.34 ms to 13.12--13.14 ms.
Mean iterations fell from 19.4 to 17.2. The maximum startup force-component
change was 4.4e-5 and moment-component change 3.8e-6 at unchanged true-residual
criteria. GPU projection, complete advancement and transactional regrid tests
passed, as did projection memcheck (zero errors). This improves the current
adaptive implementation; it does not establish speedup over the original
uniform backend, steady-load preservation, or the full control envelope.

The BiCGStab omega stage queues its two cuBLAS dot products with device scalar
outputs and transfers both scalars together. Job 2001237 passed the same three
GPU regressions and memcheck. Alternating 30-step aircraft runs retained
identical printed force/moment components and 17.2 mean iterations; total time
fell from 12.96--12.98 ms to 12.83--12.86 ms (about 1%). This removes one host
synchronization per full iteration; it does not fuse the reduction kernels or
change the pressure operator/tolerance.

## Stationary embedded-face reuse

Regridding can reuse sampled apertures when face centre, area, axis, distance
and both neighboring cell geometries exactly match the previous embedded mesh.
The caller must retain the same stationary body/distance function. Changed
faces are sampled normally; rebuilding a mesh invalidates its embedding flag.
Active cells are recomputed from the completed apertures. GpuFlow uses this
only for regrids with its unchanged body callback, preserving transactional
failure behavior.

CPU refine/coarsen tests at root widths 1 and 1.3 compare every aperture and
active cell with fresh sampling and reject reuse of an unembedded mesh. Strict
compiler warnings and ASan/UBSan passed. A grid-128 aircraft CPU diagnostic
reduced embedding time from about 0.53 to 0.20 seconds and distance calls from
about 6 million to 321024 after a small off-body refinement. GPU job 2001240
passed mesh, boundary, conservative-transfer and runtime-flow regressions. Its
100-step performance comparison did not change the hierarchy, so it is not
rebuild-performance evidence.

Job 2001242 extended both variants to 1200 steps with actual regrids at steps
600, 700 and 1100. Clock, hierarchy, residual, speed and printed six-component
loads matched at every step. Total adaptation time fell from 7.00 to 5.76
seconds; mean step time after step 100 fell from 18.99 to 17.90 ms. This is one
matched startup comparison, not steady-load or full-solver speedup acceptance.

## Transport consistency at refinement interfaces

Transported-value interpolation previously selected its virtual MAC spacing
from the cell containing the departure point. A fine face at a coarse/fine
interface could therefore be resampled on the coarse grid even at zero time.
For a smooth transverse velocity diagnostic, the maximum error was 0.0029488
at dt=0 and remained 0.0029488 at dt=-1e-9, while the positive-time error was
about 6e-11. The reverse trace used by MacCormack therefore introduced a
spacing-dependent error that did not vanish with time step.

Forward/reverse transported values and the correction limiter now retain the
target face spacing. Velocity tracing remains midpoint tracing; interpolation
still uses aperture weighting and bounded virtual faces. A cached halo is used
only when its spacing matches the requested grid, with direct sampling
otherwise. The diagnostic now has zero error at dt=0 and about 6e-11 for either
sign of dt=1e-9. GPU regression asserts identity and small-time continuity for
constant, smooth and discontinuous fields, linear/cubic and corrected/plain
transport, with/without embedded walls and root widths 1, 1.3 and 2/3.

Job 2001243 passed five CPU/GPU transport, full-step and lifecycle regressions,
plus CUDA memcheck (zero errors). CPU/GPU differences stayed below 4.45e-16;
cached/direct differences were zero. This fixes an identified consistency
error, but does not by itself establish long-time aircraft stability or the
2% load gate. The fixed-hierarchy aircraft run is being repeated separately.

The transported face's adjacent fine cell now supplies the cached block halo
when a short trace crosses the refinement boundary. Requested spacing must
still match the cache, and traces outside its halo fall back to direct
sampling. This preserves the repaired target-face grid rather than reverting
to departure-cell spacing for speed. Job 2001245 passed transport/full-step/
regrid regressions and memcheck; cached/direct errors were zero and CPU/GPU
errors stayed below 4.45e-16. Two alternating grid-128 startup runs reduced
transport from 5.25--5.33 ms to 4.08--4.09 ms and total step time from
14.04--14.17 ms to 12.93--12.99 ms, with unchanged printed loads and iteration
counts. Long-time stability and load acceptance remain separate requirements.

## Remaining interface stability evidence

After the transport-spacing repair, the fixed-hierarchy grid-128 aircraft run
(job 2001244) completed 30000 steps / six physical seconds. Peak speed fell
from 204.94 m/s in job 2001239 to 48.31 m/s, for a 12 m/s freestream. This is
an improvement, not a stability or load-equivalence pass. In the 2--5.9 s
window, repaired fixed-grid Fx/Fy/Mz means were 18.8023/13.5218/-15.9193 versus
17.7264/12.6396/-13.7245 for the AMR framework's uniform fine mesh; statistical
drift remains material. A two-point pressure-interface control with the old
transport also failed (peak 821.87 m/s), so pressure-only replacement does not
repair the transport inconsistency.

An independent linear-algebra diagnostic exported the actual pressure-gradient
stencils on an all-fluid 2 x 1 x 1 root domain with four cells per block. It
compared uniform refinement (1024 cells) and refinement of the left root only
(576 cells), holding prescribed normal boundary perturbations at zero and
retaining the zero-pressure outlet. For D equal to the implemented signed
flux divergence, G the implemented gradient and A=D G, it formed the exact
projection P=I-G A^-1 D using sparse direct solves. With positive face weights
M=diag(area*normal_distance), the squared largest singular value of
sqrt(M) P invsqrt(M) was:

| Grid / interface gradient | Maximum weighted energy amplification |
| --- | ---: |
| Uniform / existing gradient | 1.0000000000 |
| Mixed / quadratic reconstruction | 1.8058245070 |
| Mixed / two-point diagnostic | 1.0000000000 |

Relative divergence and idempotence errors were below 5e-16 in all cases.
Thus an accurate, idempotent projection can still amplify this discrete energy
norm. This diagnostic does not prove instability of the full nonlinear time
step, nor does it validate the inconsistent two-point interface as a final
replacement. Interface consistency, energy behavior and aircraft load accuracy
must be assessed together; a smaller linear-solver residual alone is not the
missing acceptance evidence.

## Compatible mass-matrix prototype

A small all-fluid CPU/SciPy diagnostic now tests a mimetic alternative to
prescribing G directly. For each cell, let N contain outward face normals,
R contain face area times (face centre minus cell centre), V be cell volume,
and Q=I-N(N^T N)^-1 N^T. The local velocity mass is
`H = R R^T / V + Q^T diag(face_area * cell_width / 2) Q`.
Exact face closure gives `N^T R = V I`, hence `H N = R`; the two complementary
terms make H positive definite. Signed local-to-global assembly gives M.
Prescribed normal pressure-correction velocities are removed from its free
submatrix. The pressure operator is then `D M^-1 D^T`, and velocity correction
is `M^-1 D^T p`, yielding an M-orthogonal projection rather than the oblique
quadratic-gradient projection.

On the same uniform/mixed geometry used in the energy diagnostic, the uniform
M reduces to the existing diagonal face-volume weights. The mixed free-face
matrix has 2272 nonzeros for 1568 unknowns, with diagonal-scaled eigenvalues
about 0.748--1.263. Manufactured linear pressure with matching Dirichlet data
on all outer faces yields gradient errors below 3.34e-16. Separately, the
inlet/wall fixed-normal and outlet-pressure projection has pressure-matrix
asymmetry below 1.12e-16 and energy decomposition error below 2.55e-16 for ten
random perturbations. These are prototype results, not production code.
Quadratic gradients are not reproduced exactly by this construction; spatial
convergence, cut-cell treatment, GPU inverse-mass cost and aircraft loads remain
to be assessed before choosing this formulation. Positive energy alone does
not replace the accepted accuracy requirements.

## Compatible operator implementation and convergence

`amr_mass` now builds the compatible face mass as deterministic symmetric CSR.
It is an independent operator building block, not yet the pressure backend.
Uniform cut cells keep the native aperture-weighted diagonal face volumes;
closed-face rows are pinned for inversion. Split cells must be fully open:
unsupported cut/split configurations fail transactionally. Tests cover split
planes in all three directions, intersecting interfaces, binary/nonbinary
spacing, embedded bodies inside the uniform fine region, linear consistency,
sampled positive energies, exact matrix symmetry and failure rollback.
CMake/CTest passed (initial build job 2001249 and expanded local tests); strict
compiler warnings and ASan/UBSan also passed on the initial cases.

An independent manufactured Poisson diagnostic on `[0,2] x [0,1] x [0,1]`
used `p=(sin(pi*x/2)+0.3*sin(pi*x))*sin(pi*y)*sin(pi*z)`, zero Dirichlet data,
midpoint-integrated forcing and a relative iterative residual below 1e-11.
The left root was refined one level relative to the right. Errors relative to
analytic cell-centre pressure and face-centre gradients were:

| Coarse spacing | Pressure volume-weighted relative L2 | Velocity face-volume-weighted relative L2 | Maximum gradient error |
| --- | ---: | ---: | ---: |
| 1/4 | 0.0246740 | 0.0231857 | 0.169156 |
| 1/8 | 0.00617194 | 0.00758450 | 0.0924482 |
| 1/16 | 0.00154615 | 0.00251316 | 0.0469043 |

Pressure converged at about order 2, integrated velocity at about order 1.6,
and maximum velocity error approached order 1. These results must not be
reported as second-order velocity accuracy. They support further evaluation
of a compatible projection; aircraft loads, embedded geometry and GPU
inverse-mass performance still require validation.

The repaired-transport dynamic aircraft run (job 2001246, old quadratic
pressure interface) completed six seconds with a peak speed of 16.85 m/s and
248 hierarchy changes. Its 2--6 s Fx/Fy/Mz means were
18.0451/13.6146/-14.1117, versus 17.7253/12.6388/-13.7308 for the uniform fine
AMR-framework reference. Recorded wall time was 215.17 seconds per physical
second. Thus the transport repair and existing performance changes do not yet
establish either the 2% load gate or the requested overall speedup.


## GPU compatible mass inverse

`GpuMassInverse` applies the free-face principal inverse of `FaceMass`.
Closed faces and, by default, inlet/wall normal corrections are fixed to zero;
the pressure outlet remains free. Uniform uncoupled faces use direct diagonal
inversion. Coupled interface faces use diagonal face-volume scaling and a fixed
32-step Chebyshev polynomial captured in a CUDA graph. Fixed iteration counts
keep the approximate inverse linear and symmetric for a future compatible
pressure operator; no residual-dependent inner iteration is used.

An independent enumeration of all 64 canonical coarse-cell split-face patterns
found local generalized spectra in `[0.5,2]` relative to their diagonal face
volumes. Summing local Rayleigh bounds and restricting to free faces preserves
these bounds; the implementation uses the slightly enlarged `[0.49,2.01]`.
The geometry restriction remains the fully open standard 2:1 split cells
accepted by `FaceMass`, with diagonal uniform cut cells.

Job 2001251 passed CTest and Compute Sanitizer memcheck (zero errors).
Its 140 cases cover all split-face subsets, both boundary modes, nonbinary
spacing and embedded bodies in fine regions. Independent CPU mass products
check true inverse residuals below 5e-12 (observed around machine precision),
linearity, bilinear symmetry, positive energy, fixed-face zeros and rollback.
This validates the inverse building block, not a complete pressure projection.
Pressure integration, aircraft stability/load acceptance and overall speedup
remain outstanding.


## Compatible GPU pressure projection

`GpuProjection::Build` accepts an optional compatible mode. Its matrix-free
pressure operator is `D M^-1 D^T`; the velocity correction uses the same mass
inverse and transpose, including fixed inlet/wall and closed-face constraints.
Inactive pressure rows are pinned. A sparse two-point operator supplies the
existing multilevel preconditioner only; it is not the converged pressure
operator. BiCGStab and the true operator residual check are retained. The
original quadratic-gradient mode remains the default for controlled comparisons.

New regression cases cover uniform/mixed grids, binary/nonbinary root widths
and embedded bodies. They independently measure post-projection flux divergence,
the compatible-mass Pythagorean energy identity, energy nonincrease for homogeneous
fixed boundaries, idempotence, uniform-grid reference agreement, and unchanged
outputs on numerical nonconvergence. These tests do not establish physical
kinetic-energy monotonicity for arbitrary boundary forcing or full nonlinear
flow stability. Flow/task selection and aircraft load/speed acceptance remain
outstanding.

Job 2001253 passed all four GPU projection/transport/advance/flow CTests and
projection Compute Sanitizer memcheck with zero errors. Mixed-grid cases took
11 iterations, with relative flux divergence below 6.1e-12 and relative mass
energy-balance defects below 9e-13. These are operator tests, not the aircraft
2% steady-load acceptance test.


## Aircraft task selection for compatible pressure

`wind3d_task --amr-compatible-pressure` selects the compatible projection when
combined with `--adaptive` or `--amr-uniform`; it is rejected for the native
backend. `GpuFlow` preserves this initialization choice through transactional
refinement/coarsening and remap projection. Setup logs identify the pressure
mode. The existing quadratic path remains available as the default comparison.

Job 2001254 passed both pressure modes through the runtime lifecycle CTest
(refinement, coarsening, adaptation, body protection, clock/pressure validity
and rollback) and Compute Sanitizer with zero errors. Compatible body regrids
had maximum volume-normalized divergence about 1.6e-11. Static mixed, dynamic
adaptive and uniform aircraft smoke runs each completed 100 steps. The 100-step
dynamic smoke did not change the hierarchy; runtime remap coverage comes from
the lifecycle test, not from that aircraft smoke.

Three single-GPU 30000-step aircraft comparisons were submitted as jobs
2001255 (static), 2001256 (dynamic) and 2001257 (uniform), using the frozen
`out/wind3d-amr/compatible-task` executable, finest spacing 0.125, speed 12,
angle 5 degrees, cubic transport and maximum dt 0.0002. Their load/stability
and performance conclusions remain pending.

The completed repaired-transport/two-point-pressure diagnostic (job 2001247)
had peak speed 16.514 over six seconds. Its 2--6 second Fx/Fy/Mz means were
18.8709/12.5340/-13.9435, with 56.38 wall seconds per physical second. The Fx
mean differs by about 6.5% from the earlier uniform-fine reference; reduced
velocity peaks alone therefore do not establish load accuracy. This diagnostic
is not the compatible mass projection and does not replace the acceptance gate.

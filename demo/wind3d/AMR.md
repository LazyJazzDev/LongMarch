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


## Compatible pressure conjugate-gradient acceleration

Compatible pressure now defaults to preconditioned conjugate gradients; the
nonsymmetric quadratic mode continues using BiCGStab. The two-point aggregation
preconditioner uses adjoint restriction/prolongation and matched pre/post Jacobi
smoothing. The optional final `Build` argument retains compatible BiCGStab for
comparisons. CG verifies candidate convergence with the actual pressure operator;
periodic residual replacement (every 50 iterations) restarts the search direction.
Reported final residuals are true residuals, including on numerical failure.

Job 2001259 passed projection and flow lifecycle CTests and projection memcheck
(zero errors). Additional comparisons cover CG against compatible BiCGStab and
Jacobi-preconditioned CG on all eight uniform/mixed, body/no-body, width cases.
The existing divergence, energy, idempotence and failure rollback checks remain.
Alternating 100-step static-aircraft runs (10 warmup steps) measured 17.61--17.69 ms
for BiCGStab versus 14.31--14.43 ms for CG, about 18--19% less total step time.
At step 100 pressure time fell from 13.46--13.47 to 9.98--10.01 ms. The maximum
absolute force-component difference over the 100 steps was 1.91e-4 and the maximum
moment-component difference was 3.89e-5, including the impulsive first step.
These short runs validate a solver acceleration, not long-term load equivalence
or the requested speedup against the original native solver. Jobs 2001255--2001257
continue using the frozen compatible BiCGStab executable for their original
comparison; they are not silently switched to CG.


## Avoiding unused reconstruction during compatible regrids

Compatible projection no longer reads or validates the quadratic `FaceGradient`;
an empty instance is accepted in that mode. `GpuFlow` skips constructing this
unused reconstruction during initialization and regrids. The original quadratic
projection continues requiring it. This removes setup work without changing the
compatible operator or its iteration sequence.

Job 2001262 passed projection and flow lifecycle CTests, including compatible
projection constructed with an empty gradient and comparisons against the
existing populated-gradient paths. A paired 1200-step adaptive aircraft run
performed three actual hierarchy changes (steps 600, 700 and 1100). CSV physical
time, dt, all six loads, cells and generations were exactly identical as printed;
this is not a claim of full-field bitwise comparison. Initialization fell from
1.970 to 1.844 seconds. The three remap/adaptation events fell from
1.771/1.674/1.798 to 1.601/1.512/1.683 seconds (about 6--10%). Overall mean step
time including adaptation fell from 19.472 to 19.131 ms in this paired sample.
Timing is diagnostic on shared hardware, not an overall production speedup gate.


## First completed compatible aircraft run

Static compatible BiCGStab job 2001255 completed 30000 steps (six physical
seconds). Peak speed was 21.078 m/s. Its time-weighted 2--6 second means were:

| Component | Mean | RMS fluctuation | Mean(4--6 s) minus mean(2--4 s) |
| --- | ---: | ---: | ---: |
| Fx | 18.8571 | 0.6015 | 0.3289 |
| Fy | 12.9675 | 2.9831 | 2.2788 |
| Fz | -0.3902 | 2.6235 | -1.3430 |
| Mx | 1.0749 | 2.4794 | 1.2881 |
| My | -0.6910 | 3.9343 | 0.8145 |
| Mz | -15.4851 | 3.0119 | 0.1669 |

This run used 212992 cells and 91.51 wall seconds per physical second over the
window. The Fy half-window drift is about 18% of its mean, so these figures
must not be certified as steady-load accuracy. Lower peaks than the previous
quadratic projection do not establish the 2% gate or overall speedup.

Matched CG uniform-128 (2001263), static-192 (2001264), and uniform-192
(2001265) comparisons are running to distinguish solver trajectory sensitivity
from spatial-resolution and interface errors. A separate static CG diagnostic
(2001266) increases protected fine-grid padding from 2h to 8h at grid 128;
its initial mesh has 367104 cells versus 212992 at 2h. This tests interface
placement without changing finest body spacing. Its modified source, isolated
object/link commands and frozen executable are under `out/wind3d-amr`; the
production refinement policy is unchanged. These experiments remain pending.


## Pressure tolerance sensitivity

Static compatible CG job 2001260 completed six seconds. Its 2--6 second
Fx/Fy/Fz/Mx/My/Mz means were 19.0546/12.5696/-0.3190/-0.3047/-1.1778/-14.4640;
peak speed was 17.5965 m/s and cost was 72.24 wall seconds per physical second.
This is about 21% less cost than compatible BiCGStab job 2001255, but the Mz
means differ by about 7%. They must not be described as long-term load-equivalent.
For 0.5-second batches, approximate Mz mean standard errors were 0.309 and 0.234;
these short correlated, possibly nonstationary series do not certify equivalence.
Force-vector differences grew from 4.5e-6 at 0.02 seconds to 0.735 at 0.1 seconds.
This motivates tolerance sensitivity checks rather than attributing all observed
differences to mesh interfaces.

`FlowParameters::pressure_tolerance` now controls physical-step projection through
`GpuTransport::AdvanceDevice`. `--pressure-tolerance` is forwarded by the adaptive
task, logged at setup, and validated as finite and positive; the adaptive default
remains 1e-8. Previously this CLI option only affected the native backend and the
adaptive physical-step tolerance was hard-coded. Remap-only projection still uses
its existing 1e-9 tolerance. The original native default is unchanged.

Job 2001269 passed transport, advance and flow CTests, flow memcheck (zero
errors), CLI rejection of zero tolerance, and 100-step aircraft checks at 1e-8
and 1e-11. The default run's printed physical CSV data matched the prior CG run
exactly. Maximum true residual in the tighter run was 9.735e-12. A subsequent
30000-step static CG run at 1e-11 is continuing in the same job; its long-term
load conclusions remain pending.


## Underresolved thin-body connectivity diagnostic

Matched early-window (0.5--0.9 s, not steady-state) CG comparisons show a strong
base-resolution effect. Mean Fx/Fy/Mz were 19.096/16.396/-12.578 (static 128),
18.688/15.781/-15.886 (uniform 128), 22.558/100.014/-29.730 (static 192), and
22.247/102.829/-31.581 (uniform 192). This cannot be attributed solely to the
coarse/fine interface: the uniform grids show a similar lift change.

A CPU-only manufactured slab diagnostic confirms a geometry connectivity defect
shared by the current AMR and native cut-face logic. On a unit cube with h=0.25,
an infinite horizontal slab of total thickness 0.05 centred at y=0.5 closes 16
normal faces. Translating the same slab to y=0.625 (a cell centre) leaves **all
normal faces fully open**. When a neighbouring cell centre is solid, the current
code resamples the face-plane aperture; both faces outside a thin solid can be
reopened, reconnecting the two fluid sides through one pressure cell. The script
and executable are `out/wind3d-amr/thin-slab-diagnostic.*`.

An isolated diagnostic variant keeps faces adjacent to solid centres closed.
It restores the slab barrier (32 closed normal faces at the cell-centred phase),
but changes the cut-cell geometry into a more stair-stepped representation.
The existing embedded-box test then fails its partial-face-count assertion;
this variant is therefore **not** a validated replacement for the cut-cell
method and has not been applied to production sources. Job 2001270 runs its
static grid-128 aircraft diagnostic using an isolated boundary object and
frozen executable. Geometry accuracy, cut-cell connectivity and spatial
convergence must be resolved before accepting a production correction.


## Completed matched compatible BiCGStab comparison

Jobs 2001255/2001256/2001257 all completed 30000 steps. Over the same 2--6 s
window, uniform-grid Fx/Fy/Mz means were 17.7368/12.0686/-13.4793; dynamic
adaptive means were 17.8756/11.9426/-13.9738. Relative mean differences were
+0.78%/-1.04%/-3.67%, respectively (signed difference divided by absolute
reference mean). Static mixed-grid differences were +6.32%/+7.45%/-14.88%.
These short-window averages do not certify steady-state equivalence; dynamic
Mz exceeds the 2% gate even before resolving stationarity and near-zero loads.

Uniform and dynamic costs were 192.62 and 222.41 wall seconds per physical
second, including adaptation. Dynamic AMR was about 15% slower in this run.
The requested load accuracy and overall speedup therefore remain unachieved.
The full six-component means, RMS fluctuations, half-window drifts and batch
mean error estimates are in `out/wind3d-amr/compatible-statistics.json`.


## Targeted thin-cell connectivity repair

AMR embedding and native face masks now identify solid-centred cells whose two
opposite face-centre probes are both fluid along at least one axis. Such an
underresolved feature cannot safely be reopened by face-area fractions: its
incident connections remain closed. Ordinary cut cells retain their sampled
partial apertures. Native occupancy temporarily carries a thin-feature bit;
`CellMask` subsequently rebuilds active-cell occupancy as before.

The new flood-fill regression checks whether an infinite slab's two sides are
connected through positive-aperture faces. It covers three coordinate planes,
seven angles, 32 phases and two root widths (1344 cases), with slab thickness
0.2h. All cases pass; the original algorithm leaked in 447/672 width-1 cases.
The ordinary embedded-box partial-aperture and constant-pressure cancellation
tests still pass, unlike the earlier all-solid-centres diagnostic fallback.

Job 2001271 passed all 18 AMR CTests, GPU flow memcheck with zero errors, native
geometry/wall-diffusion/closed-surface/lifecycle validation, and a 100-step
compatible aircraft smoke. Native validation checked all six incident faces
of 131 flagged cells and found them closed. This is a conservative fallback
for underresolved geometry, not a multi-region cut-cell method: it can shift
the effective boundary by O(h), and does not guarantee detection of arbitrarily
thin unsampled features. Aircraft load and resolution comparisons must be
repeated with matching corrected geometry before accuracy can be accepted.


## Pressure hierarchy construction overhead

Cold-stage profiling (job 2001277) attributes typical changed-hierarchy cost to
mesh construction (~0.09 s), embedding (~0.20 s), pressure construction
(~0.7--0.9 s), velocity transfer (~0.45--0.50 s), and remap projection (~0.012 s).
Within pressure construction, sparse row assembly, mass/transpose construction
and aggregation all contribute materially; GPU remap solve time is not the
main bottleneck.

Aggregation now borrows the immutable fine matrix instead of deep-copying every
sparse row. Spatial group lookup uses a hash table, with IDs still assigned by
first encounter in unchanged cell order; hash iteration never determines an
operator entry or reduction order. Coarse matrices retain the existing ordered
assembly and arithmetic.

Job 2001278 passed projection/flow CTests and flow memcheck (zero errors).
Paired 1200-step corrected-geometry aircraft runs had exactly identical printed
physical CSV data (loads, clocks, dt, cell counts and generations), including
three actual regrids. Initialization measured 1.763 versus 1.636 s; individual
regrids measured 1.566/1.513/1.535 versus 1.513/1.511/1.474 s. Overall mean step
time was effectively unchanged (19.169 versus 19.179 ms), so this sample does
not establish a meaningful whole-run speedup. Avoided fine-matrix allocations
and lookup overhead remain a limited construction improvement.


## Corrected-geometry stability isolation

Static corrected-geometry job 2001272 completed six seconds with peak speed
57.629 m/s. Its 2--6 s Fx/Fy/Mz means were 31.715/42.375/-28.790, but Mz changed
by -7.650 between the two half-windows. The thin-cell connectivity repair does
not resolve mixed-grid time-integration stability or establish load accuracy.

Jobs 2001279--2001281 isolate the corrected transport, removal of MacCormack
correction, and diagonal face-volume mass projection, respectively. All retain
the same corrected geometry, fine spacing, controls and time step. Diagnostic
readbacks record peak face positions and before/advection/diffusion/projection
maximum speeds every 100 steps. The instrumented unchanged case matched the
original static run's printed physical CSV rows exactly through the first 6329
available steps. The simplified variants are diagnostic controls, not accepted
production replacements; their load accuracy remains unproven.

A body-free small-perturbation diagnostic (job 2001282) completed 3000 steps at
dt=0.002, speed 1 and initial random interior perturbations of magnitude 1e-4,
projected before stepping. Uniform and quadrant-refined meshes used the same
physical domain and compatible projection. At six seconds, mass-weighted
perturbation energy ratios were 0.4337 and 1.2179; mixed-grid maximum component
error grew to 4.12e-4. Mixed energy initially decreased and subsequently grew.
This is a smaller reproduction of growth, not a closed-domain eigenvalue or
energy-stability proof: the existing sampler clamps upstream tangential values
to the first interior layer rather than prescribing freestream ghost values,
so inlet treatment remains a possible contributor and needs isolation.


## Inlet isolation and completed corrected-geometry comparisons

Job 2001284 prescribed first-layer inlet velocities after each projection in
an otherwise identical body-free diagnostic. At six seconds the uniform and
mixed perturbation energy ratios fell to 0.001589 and 0.024103. This isolates
inlet involvement but is not a valid production repair: overwriting projected
velocities can invalidate the divergence constraint.

Job 2001287 instead reflected upstream interpolation samples about the inlet
plane with ghost value `2 * prescribed_velocity - interior_velocity`. It made
no post-projection overwrite. Mixed perturbation energy fell to 0.023543 at
six seconds, with maximum component error 3.625e-5. The temporary GPU prototype
uses direct sampling near the inlet, leaves diffusion unchanged, and stores
boundary state in transport fields; standalone-advection state isolation,
CPU/GPU parity, cached sampling and aircraft-load validation are still needed.
Jobs 2001289--2001291 test static, uniform and dynamic aircraft meshes; job
2001292 extends the body-free test to thirty seconds with a different seed.
These experiments do not establish general stability or acceptance.

Corrected thin-geometry jobs 2001272--2001274 all completed 30000 steps.
Time-weighted 2--6 s means and measured wall cost were:

| Mesh | Fx | Fy | Mz | Wall seconds / physical second |
| --- | ---: | ---: | ---: | ---: |
| Static mixed | 31.715 | 42.375 | -28.790 | 72.658 |
| Dynamic mixed | 30.912 | 41.932 | -27.993 | 223.212 |
| Uniform fine | 31.212 | 39.127 | -28.582 | 157.613 |

Dynamic Fx/Fy/Mz deviations from uniform are -0.961%, +7.167%, and +2.061%
(using absolute reference denominators). The main-component 2% gate fails;
dynamic execution also costs about 42% more than uniform. These windows are
not certified stationary, and near-zero components require separate absolute
and statistical checks. Static speed alone is not acceptance.

Removing MacCormack correction (2001280) gives Fx/Fy/Mz means
62.963/88.517/-45.378. Diagonal face-volume mass (2001281) gives
32.639/39.426/-35.459. Neither is an acceptable simplification: both change
loads substantially relative to the uniform compatible reference. Temporary
sources, full statistics and raw outputs remain under `out/wind3d-amr`.


## Inlet cache and transfer follow-up

The inlet ghost diagnostic with seed 321 completed thirty seconds (2001292).
Mixed-grid mass-weighted perturbation energy reached 9.110e-7 of its initial
value and maximum component error was 3.159e-7. This longer body-free test did
not exhibit late regrowth, but is not aircraft or control-envelope validation.

A temporary cached inlet sampler (2001295) preserves ghost values in block
halos and rejects cubic nearest-neighbor stencils containing upstream ghost
sites, matching the direct prototype. All 62 printed energy/max-error records
for uniform and mixed six-second tests matched the direct version. Aircraft
job 2001297 then compared both versions on the same GPU for 1200 steps with
three actual regrids. All physical CSV fields matched exactly; elapsed time
fell from 28.560 to 23.130 seconds (19.0%). This comparison is against the
slower direct inlet prototype, not against the original uniform solver.
Boundary-state isolation and CPU reference parity are still required before
promoting the prototype into the production implementation.

Commit 64d19a4c separately avoids computing unused velocity components during
CPU mesh transfer. In a 212992-cell aircraft setup with 56 additional blocks,
six paired transfers produced exactly equal vectors. Excluding the first pair,
mean transfer time fell from 0.4161 to 0.3816 seconds for a constant field and
from 0.4789 to 0.4354 seconds for a perturbed field. These local measurements
are noisy and do not establish whole-simulation speedup. Raw paired data and
prototype sources remain under `out/wind3d-amr`.


## Scoped inlet implementation and remaining validation limits

`FaceField` now accepts an optional freestream inlet speed. Upstream stencil
samples reflect about the inlet plane with `2 * prescribed - interior` values,
including the MacCormack reverse pass and limiter. CPU `AdvanceReference` and
GPU `AdvanceDevice` select this boundary for physical stepping. Standalone
sampling/transfer remains clamped unless explicitly requested. GPU
`AdvectDevice` accepts the same optional speed and sets it only on local field
copies, so success or failure cannot leave an inlet setting on later calls.
The persistent halo mapping stays boundary-independent; each transport pass
fills reflected halo values as required. No velocity is overwritten after
projection. Diffusion's existing boundary treatment is unchanged.

Job 2001298 passed all eighteen AMR tests. Expanded transport comparisons cover
three spacings, embedded/unembedded mixed meshes, omitted/positive/negative
inlet speed, linear/cubic interpolation, corrected/uncorrected stepping and
zero, tiny, positive and negative steps. CPU/GPU differences stayed below
1e-15 and cached/direct differences were zero. Invalid speed rollback and
subsequent standalone calls are checked. An analytical CPU test verifies zero
tangential inlet velocity for a constant interior field. A follow-up run with
explicitly matched cubic options in the standalone-call check also passed.
Compute Sanitizer reported zero memory errors. The production 1200-step
aircraft smoke run completed with three regrids and physical CSV fields
identical to the cached prototype. Job 2001298 succeeded; this validation
does not establish long-window aircraft load acceptance.

The direct-inlet static aircraft run (2001289) completed six seconds. In the
2--6 s window its Fx/Fy/Mz means were 31.674/36.475/-29.187; peak speed was
44.954 m/s. My changed by 10.650 between half-windows, so inlet isolation has
not resolved all aircraft drift. Cadence-500 and cadence-1000 runs completed
with Fy means 45.244 and 40.758 N, and approximate 0.5 s batch standard errors
2.898 and 1.620 N. These uncertainties alone are too large to certify a 2%
load comparison; short-window proximity is not acceptance. Their wall costs
were 141.089 and 128.662 seconds per physical second, using the direct inlet
prototype. Await the matching uniform/dynamic full windows before comparison.

The older-geometry grid-192 uniform run (2001265) also completed. Its Fx/Fy/Mz
means were 21.754/93.230/-29.797, versus 17.845/13.003/-14.264 at grid 128.
These are historical pre-repair results and cannot serve as the corrected
geometry baseline. Jobs 2001300/2001301 repeat static/uniform grid 192 with
corrected thin-cell geometry and cached inlet sampling. Resolution convergence,
long-window stationarity, the control envelope and overall speedup remain open.


## Pressure partition ownership and exchange plan

`PressurePartition` constructs block-balanced Morton ownership without
truncating topology coordinates to a 64-bit interleaved key. Each cell and
unique flux face has one owner; a coarse/fine face belongs to its finer cell's
rank. Directed, sorted import lists contain pressure cells needed for D^T and
face values needed for D and face-mass rows. The same face imports can be
refreshed during each iteration of a distributed mass inverse. Failed builds
preserve the preceding plan.

The CPU partition test covers 1/2/4/8 ranks, uniform/mixed topology, open/cut
geometry and two physical scales. It fills only owned entries and explicit
imports, leaving unavailable entries NaN, then requires exact equality with
unpartitioned D, D^T and M applications. Ownership uniqueness, block balance,
ordered duplicate-free payloads and invalid-build rollback are also checked.
This is pressure communication metadata, not a distributed GPU solver:
peer-device copies, local index conversion, Krylov reductions, the multilevel
preconditioner and transport halos still require integration and validation.


## Peer-device pressure halo exchange

`GpuPressureExchange` converts a pressure partition to compact local layouts:
owned cells/faces precede imported entries. Persistent gather indices, send
buffers and per-device streams pack only owned values and copy imports with
`cudaMemcpyPeerAsync`. The API requires peer-accessible devices, waits for all
copies before returning, leaves owned values untouched, and restores the
caller's current CUDA device. It does not change process-wide peer-access
enable state. Callers must finish producing owned values before exchange;
CUDA failures can leave imports incomplete, and all enqueued streams are
drained before returning. Invalid builds preserve the preceding exchange plan.

Two-GPU job 2001310 succeeded. Direct execution and CTest verified one-rank
no-op and two-rank exchanges in both device orders on uniform and mixed meshes.
Eight changing payloads per case check every owned and imported value for both
pressure and face fields; invalid build/input recovery and device restoration
are covered. Compute Sanitizer reported zero errors. The test skips explicitly
on hosts with fewer than two GPUs; the reported job used two real GPUs and did
not skip. This establishes exchange correctness, not distributed pressure
iteration, transport integration or multi-GPU speedup. Compact local operators,
mass-inverse iteration exchanges, Krylov reductions and the preconditioner
remain to be implemented.


## Distributed pressure operator building blocks

`GpuPressureOperators` assembles compact local CSR rows for negative integrated
flux divergence D, its transpose D^T, and the compatible face mass M. Each rank
stores only owned rows and locally indexed columns; the peer exchange refreshes
required imports before rank kernels run. Calls complete all device streams,
preserve output ghost slots and restore the caller's current device. Input and
output allocations must be disjoint. Failed construction preserves the previous
operators. These are raw operator applications: D^T still needs M^-1 to become
the compatible velocity gradient, and pressure pinning/iteration is not included.

Two-GPU job 2001311 passed direct execution, CTest and Compute Sanitizer with
zero errors. Twenty-four combinations cover uniform/mixed topology,
open/embedded geometry, two spacings, one rank and both two-device orders.
NaN-filled input ghosts test actual imports; output sentinels verify that only
owned rows are written. All D/D^T/M results agree with full-mesh CPU application
within 1.12e-16. Invalid build/input recovery and current-device restoration
are also checked. Distributed mass inverse, global Krylov reductions, a
partitioned preconditioner and coupling to flow advancement remain outstanding;
no multi-GPU whole-solver speedup is claimed.


## Distributed compatible mass inverse

`GpuPressureOperators::InverseMass` now applies the same fixed 32-term
Chebyshev inverse as the single-device implementation, using the normalized
spectral interval [0.49,2.01]. Closed faces return zero; the default pressure
boundary mask also fixes inlet and wall normals while leaving outlet normals
free. Uncoupled free rows use exact diagonal division. Coupled rows exchange
normalized iterates between devices on every polynomial step. Only owned
output slots are written, and repeated calls reset their internal recurrence.

Job 2001312 passed direct tests, CTest and Compute Sanitizer (zero errors) on
two GPUs. Forty-eight geometry/spacing/partition/boundary-mode combinations
compare against the existing single-device inverse. Maximum absolute output
difference was 1.71e-13; independently applying the CPU mass matrix to the
assembled free-face result left true principal-system residual below 1e-12.
Repeated scaled right-hand sides, fixed-face zeros, untouched output ghosts,
alias rejection and device restoration were checked. The implementation still
synchronizes rank streams and peer transfers at every iteration and stores
full owned-face work vectors. Performance optimization and distributed pressure
iteration remain necessary; correctness here does not demonstrate speedup.


## Composed distributed pressure action

`GpuPressureOperators::Gradient` now composes M^-1 D^T using private local face
buffers. `Pressure` applies D M^-1 D^T and pins inactive cell rows to the input
potential, matching the single-device compatible operator. Fixed boundary
corrections use the existing free-face mask. Caller input/output allocations
remain disjoint, only owned output entries change, and all device work finishes
before return. This is a complete matrix action, not yet a pressure iteration.

Two-GPU job 2001313 passed the expanded 48-case direct tests and CTest, with
zero Compute Sanitizer errors. Composed gradients and pressure actions match
CPU D/D^T combined with the existing single-device mass inverse within the
relative 1e-12 test tolerance, including cut geometry, inactive rows, both
boundary modes, reversed devices and unchanged output ghost slots. The overall
suite maximum absolute difference remains 1.71e-13. Global Krylov reductions,
pressure convergence/rollback, distributed preconditioning and runtime flow
integration remain open.


## Distributed pressure projection baseline

`GpuDistributedProjection` now solves the compatible pressure system using
rank-local vectors, device dot products reduced as host scalars, and Jacobi
preconditioning from the two-point pressure diagonal. CG verifies candidate
solutions with the actual D M^-1 D^T operator, replaces the residual and restarts
the search direction every fifty steps, and reports the true residual on
nonconvergence. The projected velocity subtracts M^-1 D^T p, preserves fixed
normal fluxes and closes embedded faces. Owned caller fields commit only after
convergence; numerical failure preserves all caller entries, including ghosts.
A zero right-hand side resets the previous potential to zero.

Two-GPU job 2001316 passed direct execution, CTest and Compute Sanitizer with
zero errors. Twenty-four cases cover uniform/mixed meshes, open/cut geometry,
two scales and one/two ranks in both device orders. Against the existing
single-device compatible Jacobi-CG projection, maximum velocity and potential
differences were 3.89e-15 and 8.89e-16. Independently reconstructed post-projection
flux divergence met the requested 1e-10 relative tolerance. Tests also exercise
one-iteration nonconvergence, invalid tolerance/aliasing, failed-build rollback,
unchanged output ghosts, zero RHS and restoration of the caller's CUDA device.

This completes a distributed projection correctness baseline, not production
multi-GPU flow. It still uses host scalar reductions, synchronous peer phases
and Jacobi rather than a distributed multilevel preconditioner. Performance
measurements, scalable preconditioning, distributed transport, adaptation and
runtime integration remain required before claiming speedup or task completion.

### Repeated projection and measured scaling

The first larger repeated benchmark (jobs 2001318 and 2001321) was incomplete.
The latter exited with diagnostic code 13 (field mismatch), not a CUDA failure;
the platform subsequently reported PENDING/unknown. Its pageable host-to-device
uploads had not been explicitly completed before the solver's nonblocking
streams consumed the input, violating the documented producer-completion
contract. Job 2001324 added device synchronization after each input upload,
outside the timed region, and completed all four repeats of all configurations.
Distributed velocities matched the single-device Jacobi reference within
2.23e-16 on every repeat. Production projection code was unchanged.

For the body-free mixed mesh with 55,296 cells and 170,624 faces, tolerance 1e-9,
mean times excluding the first repeat and setup were:

| Projection | Iterations | Seconds |
| --- | ---: | ---: |
| Existing single GPU, Jacobi | 2033 | 0.331825 |
| Existing single GPU, multilevel | 27 | 0.006481 |
| Distributed implementation, one GPU, Jacobi | 2033 | 1.030326 |
| Distributed implementation, two GPUs, Jacobi | 2033 | 3.277667 |

This is a substantial slowdown, not a speedup. Distributed multilevel
preconditioning and fewer host-synchronized exchange phases are necessary.
These timings concern this projection benchmark, not aircraft flow throughput.
The regression now explicitly completes input producers and restores the
original velocity for three further solves on each existing test configuration,
checking owned results and unchanged ghosts on each repeat.

Two-GPU job 2001327 passed the expanded direct regression, CTest (4.40 s),
and Compute Sanitizer memcheck with zero errors.

### Distributed fine-grid preconditioning stencil

PressurePartition now imports open-face neighbor pressures for every owned cell,
including faces owned by another rank. The earlier D^T-only pressure imports
were insufficient for the two-point fine-grid residual on the opposite side of
a partition interface. A NaN-initialized CPU stencil regression fails with the
previous planner (code 17) and passes with the extended plan for 1/2/4/8 ranks,
uniform/mixed meshes, two scales and open/cut geometry.

GpuPressureOperators::TwoPointPressure applies the rank-local CSR approximation
D diag(area * distance * aperture)^-1 D^T with the configured free-face mask and
inactive identity rows. It refreshes cell ghosts and writes only owned outputs.
It is a building block for multilevel preconditioning, not a replacement for
the compatible pressure operator or a completed V-cycle. Two-GPU job 2001328
passed CPU partition, expanded operator and repeated projection CTests, plus
operator Compute Sanitizer memcheck with zero errors. Operator comparisons
include both device orders and both boundary masks, using an independent CPU
face-wise two-point reference and NaN ghosts.

### Corrected-inlet 192-resolution comparison

Jobs 2001300/2001301 both completed 30,000 steps (6 physical seconds) with the
thin-cell barrier fix and cached inlet ghosts. The matched 2--6 second means
in inlet-corrected-192-final.json are:

| Mesh | Fx | Fy | Mz | Wall seconds / physical second |
| --- | ---: | ---: | ---: | ---: |
| Static adaptive 192 | 33.810144 | 73.257571 | -22.345676 | 179.754583 |
| Uniform 192 | 33.573625 | 81.765550 | -25.319858 | 536.613313 |

Static adaptive deviations are +0.704% Fx, -10.405% Fy and +11.746% Mz,
using the magnitude of the reference component as denominator. Its roughly
2.99x throughput advantage does not satisfy the 2% load criterion. The adaptive
Fy mean also shifts by -10.0133 between the two halves of the window, so these
are finite-window comparisons, not established steady-state estimates. Mesh
coverage and statistical convergence remain required alongside solver speed.

### Configurable body protection extent

Adaptive batch tasks accept `--protect-padding-cells N`, the isotropic distance
outside the rotated conservative aircraft bounding box in finest-cell widths.
The default and minimum are 2, retaining the existing geometry/stencil guard.
The value must be finite; explicit use requires `--adaptive` and is rejected for
native or uniform mode. Setup reports the corresponding physical distance.
This changes the protected region used by both initial and subsequent dynamic
refinement, without changing refinement thresholds or solver parameters.

GPU job 2001329 checked invalid values/modes and five 80-step trajectories.
Default and explicit 2-cell padding matched the frozen production-inlet binary
exactly in every physical CSV column (timings excluded). At grid 128, cell
counts increased from 212,992 to 367,104 and 614,400 for padding 2, 8 and 16.
These short runs validate configuration, not load accuracy.

Jobs 2001330/2001331 run static adaptive grid 192 with padding 8/16 respectively,
30,000 steps, dt capped at .0002, speed 12 and angle 5. They use the same physical
model as the completed 192-resolution reference, and will be compared over the
same 2--6 second window. Expanded protection must still demonstrate both load
accuracy and useful cost reduction; no acceptance is inferred from cell counts.

### Distributed pressure hierarchy assembly

PressureHierarchy builds immutable host CSR levels for the two-point
preconditioner with the standard pressure boundary mask. Active cells aggregate
geometrically at doubling widths; inactive cells remain fine identity rows and
have no coarse parent. Restriction sums integrated residuals and prolongation
is constant, giving R=P^T and A_coarse=P^T A_fine P. The hierarchy stops at at
most 64 unknowns or an identity operator; failed builds retain the old levels.
Fine ownership comes from PressurePartition. Coarse ownership is balanced in
aggregate-ID order and may leave ranks empty on small coarse levels.

Per-level directed import plans include owned matrix-row columns, fine children
needed by coarse owners for restriction, and coarse parents needed by fine
owners for prolongation. The latter two requirements are distinct from matrix
halos and explicitly tested with NaN-initialized unavailable entries.

Forty CPU configurations cover uniform/mixed meshes, open/cut geometry, uniform
all-solid identity, two scales and 1/2/4/8 ranks. Tests verify the independent
fine face stencil, Galerkin actions, adjoint transfers, positive sampled energy,
ownership, complete A/R/P imports and failure rollback. Direct Release,
ASan+UBSan and the standard CMake/CTest target passed (CTest 0.13 s, built using
the existing allocation 2001306 without requesting additional GPUs). This is
host assembly only: GPU V-cycle kernels, coarse solve, integration and measured
iteration/performance improvement are still pending.

### GPU distributed V-cycle and pressure integration

GpuMultilevel now uploads rank-local CSR matrices, adjoint transfer operators,
diagonals and persistent work arrays from PressureHierarchy. Every level uses
two pre/post weighted-Jacobi sweeps (weight .9), compact gathered peer imports,
residual restriction and additive constant prolongation. Only the terminal
matrix (at most 64 cells) is inverted on the host during setup; its inverse rows
and RHS are distributed for device evaluation. All-solid identity levels bypass
the inverse. Empty coarse ranks retain imports required by prolongation without
launching empty kernels. Fine caller fields use owned-cell order, and caller
ghosts are untouched. Build is transactional and Apply restores the CUDA device.

GpuDistributedProjection uses this V-cycle by default; Build(..., false) retains
Jacobi for comparisons. The compatible D M^-1 D^T operator, true-residual checks,
periodic CG restarts and convergence-only field commit remain unchanged. This
is a synchronous peer implementation, not yet an optimized captured V-cycle.

Job 2001337 passed 15 direct V-cycle configurations with independent CPU
Cholesky coarse solves, CPU V-cycle parity, linearity, symmetry, positive sampled
energy, repeated inputs, alias rejection and output ghost preservation, plus
CTest and memcheck. Job 2001339 repeated validation with a one-cell coarse limit,
exercising empty coarse ranks, and passed memcheck. Job 2001338 passed 48 full
projection configurations (both preconditioners) and repeated-solve regression,
CTest and memcheck. Main mixed-grid cases dropped from 50 to 11 iterations;
actual reconstructed divergence remained within the 1e-10 test tolerance.

The complete 55,296-cell / 170,624-face benchmark in job 2001339, tolerance 1e-9,
excluded setup, input copies and the first of four repeats:

| Implementation | GPUs | Preconditioner | Iterations | Mean seconds |
| --- | ---: | --- | ---: | ---: |
| Existing single-device | 1 | Jacobi | 2033 | 0.333093 |
| Existing single-device | 1 | Multilevel | 27 | 0.006039 |
| Distributed | 1 | Jacobi | 2033 | 1.052081 |
| Distributed | 2 | Jacobi | 2033 | 3.117378 |
| Distributed | 1 | Multilevel | 27 | 0.023825 |
| Distributed | 2 | Multilevel | 27 | 0.078805 |

The two-GPU change is about 39.6x faster than its Jacobi baseline but still about
13x slower than the existing optimized single-device multilevel solver on this
mesh. All multilevel repeats agreed with the Jacobi velocity reference within
2.28e-10 and verified a true relative residual about 4.96e-10. Reducing peer/host
synchronization remains essential. This does not establish aircraft flow
speedup, distributed transport, or the 2% steady-load acceptance criterion.

### Event-ordered V-cycle phases

V-cycle phase boundaries now record one CUDA event per rank before enqueuing
cross-device stream waits. All producers are recorded before any wait, and each
wait captures that recording before events are reused at the next boundary.
This replaces per-phase host stream synchronization while preserving the same
matrix operations and ordering. Apply still waits for all device work before
returning; failure paths drain streams and destruction releases events after
stream completion. Coarse ranks with no owned rows participate in dependencies.

Job 2001340 passed direct projection regression, V-cycle/projection CTests and
projection memcheck with zero errors. Job 2001341 completed the same larger
benchmark with unchanged iterations, residuals and velocity errors. Warm means:

| Implementation | Previous seconds | Event-ordered seconds |
| --- | ---: | ---: |
| Distributed multilevel, one GPU | 0.023825 | 0.017391 |
| Distributed multilevel, two GPUs | 0.078805 | 0.074412 |

These are separate job measurements on shared hardware, not an isolated paired
performance experiment; the small two-GPU difference should not be treated as
a general scaling guarantee. The existing optimized single-device solver took
0.006086 seconds in the new job. Mass-inverse exchange phases still synchronize
the host, and multi-GPU transport/runtime integration remains incomplete.

### Stream-ordered mass-inverse exchanges

GpuPressureExchange::EnqueueFaces queues packed peer transfers on the caller's
per-rank streams, with event barriers before producers are consumed and before
subsequent consumers run. Repeated enqueues on the same streams are ordered;
callers must drain those streams before destroying/rebuilding the exchange or
using synchronous exchange methods, including after enqueue failures. The
single-rank path relies on stream order without recording cross-device events.

The distributed Chebyshev mass inverse now uses this path for its 31 iterative
exchange phases instead of synchronizing the host after each kernel and copy.
It preserves the same fixed polynomial, final completion guarantee and error
handling; an exchange failure now also drains the operator streams.

Job 2001343 passed direct full-projection regression, exchange/operator/V-cycle/
projection CTests and projection Compute Sanitizer memcheck with zero errors.
Job 2001344 completed the 55,296-cell benchmark with unchanged 27 multilevel
iterations, approximately 4.96e-10 true residual, and <=2.28e-10 velocity error.
Warm distributed multilevel means were 0.010856 s for one GPU and 0.072459 s for
two GPUs, versus 0.017391/0.074412 s before this change. The existing optimized
single-device solver remained 0.006045 s. These separate shared-node job
measurements show limited two-GPU improvement and do not establish scalable
flow speedup. Launch/communication overhead and larger-mesh scaling still need
measurement, alongside distributed transport and load-accuracy acceptance.

### Larger mesh scaling and explicit peer access

Job 2001346 completed the same four-repeat multilevel projection comparison on
442,368 cells / 1,346,048 faces (16x8x12 root blocks, four cells per block,
refining the x<8 half). All configurations took 35 iterations and verified a
true relative residual about 8.48e-10. Distributed velocities matched the
single-device multilevel reference within 2.23e-16. Excluding the first repeat,
the optimized single-device solver took 0.014431 s; the distributed implementation
took 0.033596/0.134819/0.326778 s on 1/2/4 GPUs. Larger size alone did not produce
positive scaling.

Job 2001347 isolated peer-access initialization in separate benchmark processes
on one four-GPU allocation. The topology reported NV18 links between every GPU
pair. Each process ran the same mesh/solver, with the middle process explicitly
enabling every directed CUDA peer-access pair before allocation. Baseline
processes ran before and after it; production code was unchanged.

| Implementation | Baseline before | Peer enabled | Baseline after |
| --- | ---: | ---: | ---: |
| Optimized single-device | 0.014594 | 0.014405 | 0.014583 |
| Distributed, one GPU | 0.033889 | 0.033930 | 0.034162 |
| Distributed, two GPUs | 0.139737 | 0.073811 | 0.137218 |
| Distributed, four GPUs | 0.344826 | 0.206142 | 0.344122 |

Values are warm mean seconds. Every repeat passed the same residual and field
checks, and all three processes exited successfully. This supports explicitly
initializing peer access at the future multi-GPU runtime boundary; merely
checking cudaDeviceCanAccessPeer does not establish this performance behavior.
It does not establish the internal transfer route, bandwidth, or eliminate the
remaining scaling deficit. Library teardown must not disable another solver's
process-wide peer-access state when this is integrated.

### Automatic pressure peer-access initialization

GpuDistributedProjection::Build now validates mutual peer capability and enables
every directed selected-device pair. Already-enabled pairs are accepted. This
is a documented monotonic CUDA-context initialization: neither destruction nor
failed construction disables peer access, preserving other solver instances.
The caller's current device is restored, and failed builds preserve the previous
solver. No transport or flow-runtime integration is implied by this change.

The expanded regression creates and destroys a second solver sharing the same
GPUs, rejects its duplicate-device rebuild, checks that access remains enabled,
and then exercises the first solver's full projections and repeated inputs.
Job 2001348 passed numerical regression/CTest, but default Compute Sanitizer
reported 190 expected API statuses when cudaGetLastError cleared error 704
(cudaErrorPeerAccessAlreadyEnabled). A minimal probe in job 2001349 generated
the sanitizer XML schema. The committed amr_peer_status_suppressions.xml matches
only kind Api, api cudaGetLastError, error cudaErrorPeerAccessAlreadyEnabled and
result 704; it does not disable API or memory checking generally.

Job 2001350 passed direct regression, CTest and memcheck with zero unsuppressed
errors using `--suppressions demo/wind3d/amr_peer_status_suppressions.xml`.
It also completed the 442,368-cell benchmark without explicit peer setup in the
driver: warm means were 0.033890/0.075493/0.218619 seconds on 1/2/4 GPUs, versus
0.014472 seconds for the optimized single-device solver. All distributed cases
used 35 iterations, about 8.48e-10 true residual and <=2.23e-16 velocity deviation
from that reference. This reproduces the peer-enabled improvement but does not
resolve the remaining negative scaling or establish aircraft-load acceptance.

### Captured cross-device V-cycle

Component profiling (job 2001351, 20 warm applications per component) isolated
both the mass inverse and V-cycle as scaling bottlenecks. On the 442,368-cell
mesh, four-GPU Pressure, InverseMass and V-cycle calls took approximately
2.695, 2.485 and 3.194 milliseconds respectively; divergence alone was 0.060 ms.

A native-sm_90 probe (2001353) verified cross-device CUDA Graph fork/join capture
and replay with memcheck. Capturing the full V-cycle exposed an unsupported
cudaMemcpyPeerAsync call (diagnostic 2001355). GpuMultilevel now gathers directly
into peer-accessible destination buffers with device kernels, enabling required
peer access monotonically, and captures its fixed V-cycle with every device
stream forked from and rejoined into the root stream. Apply completes input
copies before launching the graph, waits for the joined root, then commits owned
output entries. The same matrices, smoothing weights/counts, coarse inverse and
transfer operators are used; packed intermediate copy buffers are removed.

Job 2001356 passed direct full pressure regression, standalone V-cycle and
projection CTests, and projection memcheck with the previously documented exact
704-status suppression. Repeated inputs, both device orders, empty coarse ranks,
CPU V-cycle parity and convergence checks remain covered. Job 2001358 completed
the four-GPU larger benchmark and component measurements:

| GPUs | V-cycle before (ms) | Captured V-cycle (ms) | Full projection (s) |
| --- | ---: | ---: | ---: |
| 1 | 0.244307 | 0.224725 | 0.033398 |
| 2 | 1.015094 | 0.488614 | 0.067438 |
| 4 | 3.194386 | 1.236546 | 0.143557 |

The prior automatic-peer four-GPU projection mean was 0.218619 s. All larger
runs still took 35 iterations with about 8.48e-10 true residual and <=2.23e-16
velocity difference from the optimized single-device multilevel reference.
Component checksums matched their pre-capture values. These are shared-node
measurements; the optimized single-device reference remains faster (0.014585 s
in the new job), and the mass inverse is still not captured. No load-accuracy
acceptance or production distributed flow is claimed.

### Captured distributed mass inverse

The fixed 32-step Chebyshev mass inverse is now captured once per operator
build across all device streams. Persistent owned-face input/result arrays keep
graph addresses stable; Apply uploads its owned RHS, launches the joined graph,
and copies only owned results back before returning. Initialization, polynomial
coefficients, fixed-face masking and uncoupled diagonal treatment are unchanged.
GpuPressureExchange::EnqueueFaces uses capturable peer-store gather kernels;
Build enables required directed peer access monotonically. Synchronous exchange
methods retain their packed peer-copy path.

Job 2001363 passed direct repeated full-pressure regression, exchange/operator/
V-cycle/projection CTests and projection memcheck with the exact expected-704
suppression. This covers the operator's independent free-principal mass
residual, changed RHS, uniform/mixed/cut cases, both device orders and output
sentinels. Job 2001364 completed the 442,368-cell full and component benchmarks.
Warm projection means were 0.032688/0.048459/0.089598 s on 1/2/4 GPUs, compared
with 0.033398/0.067438/0.143557 s before mass capture. Four-GPU InverseMass
component time decreased from 2.366 ms to 1.117 ms. All full runs retained
35 iterations, about 8.48e-10 true residual and <=2.23e-16 velocity difference
from the optimized single-device reference, which took 0.014551 s in this job.
Separate shared-node timings do not establish positive scaling; the single-GPU
baseline still wins, and full-flow integration/accuracy acceptance remain open.

### Compact coupled-face iteration

The normalized mass matrix now stores only owned coupled rows, with a separate
local-face index per row. Chebyshev kernels launch only those rows; uncoupled
faces retain exact diagonal inversion in the final kernel. Both iteration
buffers are initialized because the shared face exchange still includes
uncoupled entries. The polynomial, row arithmetic and global face layout remain
unchanged. The diagnostic mixed meshes contain 2,648 coupled faces out of
170,624 (55,296 cells), and 10,672 out of 1,346,048 (442,368 cells), so traversing
all rows in every iteration previously did mostly unnecessary work.

Job 2001365 passed direct projection regression, all four relevant CTests and
memcheck. Job 2001366 passed operator initcheck and full/component 1/2/4 GPU
benchmarks. Residuals, iterations and velocity comparisons were unchanged.
The first four-GPU timing appeared slower than the previous separate job, so
job 2001367 alternated frozen full-row/compact-row/full-row executables within
one allocation (binary hashes recorded). Warm projection means in seconds:

| GPUs | Full rows before | Compact rows | Full rows after |
| --- | ---: | ---: | ---: |
| 1 | 0.032987 | 0.020288 | 0.032980 |
| 2 | 0.043026 | 0.039499 | 0.042175 |
| 4 | 0.112107 | 0.101350 | 0.101739 |

Single- and two-GPU improvements persist in this comparison. Four-GPU benefit
is not established beyond the baseline variation, and the earlier apparent
regression was not reproduced against both adjacent baselines. Communication
still exchanges the full pressure face plan; compact iteration alone does not
resolve scaling. The optimized single-device reference remains faster, and
steady-load acceptance and distributed flow integration remain incomplete.

### Selected coupled-face communication

GpuPressureExchange::Build accepts an optional global face mask for queued
imports only. Empty masks retain the full queued exchange; a full-sized zero
mask queues no payload. Separate source and destination index lists scatter
selected faces into their original ghost slots, preserving the layouts and
full synchronous cell/face exchanges used by D, D^T and M. Invalid mask sizes
leave the previous plan intact. The captured mass inverse supplies its free
coupled-face mask; its polynomial, arithmetic and boundary conditions remain
unchanged.

A CPU partition diagnostic on the 442,368-cell mixed mesh counts 22,504 imported
faces on four ranks, of which 5,560 are coupled. Across the 31 iterative
exchanges this reduces double payload from 5,580,992 to 1,378,880 bytes (75.3%).
This counts payload, not event overhead or total solver traffic. The two-rank
counts are 11,840 and 5,360 (54.7% less payload). Eight-rank counts are 35,828
and 9,268; the global coupled mask is conservative and may include a coupled
face imported by a destination that only needs it for divergence.

Job 2001374 passed the four exchange/operator/V-cycle/projection CTests and
projection plus exchange memcheck with zero unsuppressed errors (the existing
exact expected-704 suppression applies). Exchange tests cover unchanged full
layouts and synchronous imports under all/none/sparse queued masks, invalid
mask rollback, repeated enqueues, preserved unselected ghost sentinels, mixed
and uniform meshes, and both device orders. Operator/projection tests retain
the independent free-principal residual and repeated full-pressure checks.

Job 2001375 passed operator initcheck and the full/component benchmarks. Within
one allocation, frozen full-payload / selected-payload / full-payload warm
projection means were:

| GPUs | Full payload before | Selected payload | Full payload after |
| --- | ---: | ---: | ---: |
| 1 | 0.020156 | 0.020071 | 0.021638 |
| 2 | 0.039198 | 0.039512 | 0.039447 |
| 4 | 0.081271 | 0.084615 | 0.088241 |

All runs retained 35 iterations, approximately 8.48e-10 true residual, and
2.22e-16 velocity difference from the single-device reference. There is no
established wall-time improvement beyond shared-node variation: less payload
does not remove the per-iteration event barriers and graph scheduling cost.
The change retains lower traffic without changing numerics; positive scaling,
full-flow integration and the 2% steady-load acceptance remain unproven.

### Directed dependencies for queued face exchange

Queued face imports now synchronize only directed links with selected payload.
All participating streams first record their current producers/consumers;
each source waits for its destination to finish reading old ghosts before
issuing peer stores. After all source completion events are recorded, each
destination waits for its sources before running new consumers. Recording all
events before adding waits avoids phase mixing when events are reused, and
preserves progress through cyclic communication graphs. A rank with no queued
links needs no exchange events; enclosing graph capture still forks and joins
all ranks.

The 442,368-cell diagnostic has six active directed links on four ranks, so
each iteration requires 12 cross-stream waits instead of 24. Eight ranks have
18 links (36 waits instead of 112); two ranks retain both links and four waits.
These are dependency counts, not predicted wall-time speedups.

Job 2001376 passed all four CTests and projection/exchange memcheck with zero
unsuppressed errors. Exchange coverage now includes four ranks in forward and
reverse device order and one-way masks (only rank zero's owned faces are
selected). It snapshots old ghosts, changes owned producers asynchronously,
and exchanges again without a host barrier; both the old snapshot and new
imports must be correct, with unselected slots unchanged. Existing captured
mass-inverse and repeated pressure projection regressions also pass.

Job 2001377 passed operator initcheck and full/component benchmarks. Paired
all-rank / directed-link / all-rank warm projection means in seconds:

| GPUs | All-rank before | Directed links | All-rank after |
| --- | ---: | ---: | ---: |
| 1 | 0.020008 | 0.021660 | 0.020117 |
| 2 | 0.039378 | 0.039919 | 0.039222 |
| 4 | 0.083887 | 0.081746 | 0.081822 |

The four-GPU component mass inverse measured 0.663 ms, versus 1.035 ms in the
previous separate job; this component comparison is not paired. Full projection
improvement remains indistinguishable from baseline variation. The one-GPU
warm mean contains a timing outlier despite having no exchange dependencies.
All configurations retained 35 iterations, about 8.48e-10 true residual and
2.22e-16 velocity difference from the single-device reference. Fewer graph
edges are retained, but neither positive GPU scaling nor full-flow acceleration
has been demonstrated. The next integration work must address transport and
runtime ownership as well as the still-open steady-load accuracy gate.

### Shared transport kernels and distributed dependencies

The sampling field and the slip/reconstruction/advection CUDA kernels now live
in the internal `amr_gpu_transport_kernels.cuh`. The single-device transport
includes this implementation; the extraction preserves numerical kernel tokens
apart from whitespace. This prepares a shared implementation for distributed
transport rather than maintaining a second cubic interpolation or limiter.
It does not yet add a distributed transport runtime.

The required distributed stages follow the existing arithmetic:

1. Clear closed-face values; extend first-layer slip ghosts from open faces,
   then second-layer ghosts from the completed first layer.
2. Reconstruct cell centres from incident face values; reconstruct slopes and
   bounds from completed neighbouring cell centres across open faces.
3. Sample original velocity for midpoint trajectories and forward transport.
   Native samples read faces, virtual samples read centre/slope/bound tuples.
4. Apply the same slip and reconstruction stages to the forward field.
5. Correct using the forward field at the reverse trajectory, but trace both
   directions with original velocity. Apply the original eight-donor limiter
   using target face spacing, reverting to the forward result when required.

Pressure imports alone do not cover these dependencies. The current sampler
supports trajectories crossing block boundaries and retains a direct lookup
fallback beyond cached halos; distributed ownership must preserve that access
or enforce and validate an explicit trajectory bound. A fixed pressure stencil
must not silently truncate transport donors. Runtime integration also needs
consistent diffusion, sponge, load reductions and transactional regridding;
none of these requirements is satisfied by pressure scaling benchmarks alone.

Job 2001380 rebuilt and passed the GPU transport, complete-step and adaptive
flow CTests (153.21/46.39/0.78 seconds). Transport compares CPU/GPU and cached/
direct results across inlet modes, embedded geometry, linear/cubic and
corrected/uncorrected paths, signed timesteps and long traces. This validates
the extraction's existing single-device behavior, not distributed transport.

### Owned-field distributed advection

`GpuDistributedTransport` builds the same block/face ownership as
`PressurePartition`. Each rank stores only its owned dynamic velocity,
cell-centre, slope and limiter arrays for original/forward fields, plus its
owned output buffer. Global face/cell IDs resolve through owner/local-index
maps and device pointer tables. Geometry, incidence, lookup topology and slip
classification are currently replicated; dynamic fields are neither replicated
nor gathered onto the host. Required CUDA peer access remains enabled for the
context lifetime, as in distributed pressure.

The shared transport kernels map work indices to owned global IDs, while the
single-device path retains contiguous indexing. Each preparation stage launches
on every rank and completes before peer reads in the next stage. Forward and
corrected advection reuse the shared formulas and retain original-velocity
tracing in both directions. Direct sampling supports cross-partition long
trajectories without truncating donors or introducing a new timestep limit.
The distributed implementation currently omits the optional block sampling
cache; this is a functional ownership/communication path, not yet a performance
claim.

Public inputs and outputs use each rank's owned face prefix, matching pressure
layouts; trailing pressure imports are untouched. Input/output aliasing is
supported because all inputs first copy into private storage. Finite-value
validation completes on every rank before output commits; numerical failures
leave every caller output unchanged. Invalid builds preserve the previous
instance. CUDA failures may leave partially copied outputs, and all internal
streams are drained before returning from failed operations.

Adaptive runtime integration is still required before full-flow or 2%
steady-load acceptance.

Job 2001385 passed distributed transport plus the existing single-device
transport/advance/flow CTests and distributed transport memcheck with zero
unsuppressed errors (the exact expected-704 peer-status suppression applies).
All 32 geometry/scale/device configurations, each with 16 transport cases,
reported zero maximum difference from single-device direct sampling. Coverage
includes one/two/four GPUs, reversed/permuted device order, uniform/mixed grids,
embedded bodies, inlet modes, signed/zero/long timesteps, linear/cubic and
corrected/uncorrected transport. It also checks ownership against the pressure
partition, aliasing, untouched trailing ghosts, build rollback, current-device
restoration and all-rank output rollback on a non-finite input. This establishes
advection consistency only; it does not establish a full-flow speedup or the
steady-load acceptance gate.

### Distributed diffusion and outlet treatment

`GpuDistributedTransport::AdvectDiffuse` composes corrected advection, molecular
diffusion, outlet sponge and prescribed normal boundary values. Diffusion uses
the original slip-extended velocity and reconstructed field, adding its
increment to the advected result; it does not diffuse the newly advected field.
The shared `DiffuseBoundary` CUDA kernel now maps work to owned faces, so the
single- and multi-device paths use the same formula and ordering. Peer reads
use completed original-field data; each rank writes only its owned result.

The call is self-contained, including original/forward preparation, and commits
only after all output values pass validation. It preserves aliasing and trailing
pressure ghosts. Positive finite timesteps, nonnegative finite viscosity and
sponge length, finite speed, and `dt * viscosity / min(face area) <= 1/6` are
required. Invalid or unstable parameters reject before any output mutation.
This is the transport portion of a timestep; pressure projection and distributed
load evaluation still need to be composed transactionally into the runtime.

Job 2001386 passed the four distributed-transport/single-transport/advance/flow
CTests and distributed transport memcheck with zero unsuppressed errors using
the exact expected-704 suppression. The 128 added CPU-reference comparisons
span all 32 geometry/device configurations, linear/cubic sampling, zero or
strong diffusion (`dt*viscosity/min_area=0.1`), disabled/enabled sponge and
positive/negative prescribed speed. Maximum error was 6.67e-16; existing
advection-only comparisons retained zero error. Checks also cover aliasing,
trailing ghost preservation, non-finite input rollback and unchanged outputs
when diffusion stability, viscosity, sponge length, timestep or speed validation
fails. Full timestep coupling, steady loads and acceleration remain unverified.

### Distributed pressure and friction loads

`GpuDistributedTransport::EvaluateLoads` accepts owned pre-step velocity and
owned physical-pressure prefixes and returns pressure/friction/total force XYZ
followed by moment XYZ about the world/body origin. `OwnedCells` exposes the
same cell ordering as the pressure partition, complementing `OwnedFaces`.
The operation is self-contained: it copies inputs into private owned buffers,
validates every owned entry and applies the two slip-extension layers before
evaluating load-only wall friction. Trailing pressure/velocity imports are
neither read nor modified; caller inputs and output loads survive every failure.

Single- and multi-device paths share `LoadPartials` and `ReduceLoads`. The
shared cell-work mapping selects each owned cell exactly once, with its local
pressure entry and globally addressed incident face velocities. Pressure
moments retain face positions; friction moments retain cell centres, half-cell
wall sampling distance and the existing float Spalding viscosity evaluation.
GPU partials use deterministic reductions; only 12 scalars per rank return to
the host, where compensated sums combine ranks. Finite checks also cover the
final summed components and totals before committing the result.

The method does not rescale pressure: the timestep driver must convert the
projection potential to physical pressure using density/dt before evaluation.
It must also retain pre-step velocity until friction loads are complete.
Transactional composition with projection and the adaptive runtime remains
outstanding; operator load agreement alone is not steady-load acceptance.

Job 2001394 passed distributed loads, distributed transport, single-device
complete-step and adaptive-flow CTests, plus distributed-load memcheck with
zero unsuppressed errors under the exact expected-704 suppression. Forty
geometry/scale/device configurations each compare six pressure/velocity cases
against independent CPU `Mesh::PressureLoads` and `WallLoads` references.
Maximum component differences were 2.33e-16 for pressure loads and 1.25e-11
for friction loads (the latter retains the existing float wall model).
Coverage includes one/two/four GPUs, permuted devices, uniform/mixed grids,
embedded and fully solid bodies, constant/linear/nonlinear signed pressure,
velocity reversal and changed density/viscosity. Repeated results are identical;
inputs including trailing NaN ghosts are preserved, invalid parameters leave
outputs unchanged, and non-finite owned pressure/velocity rejects on any rank.

### Transactional distributed timestep

`GpuDistributedAdvance` owns distributed transport and compatible pressure
projection built from the same geometry. Build verifies identical owned face
and cell ordering before allocating layout-sized private velocity/potential
buffers. A failed build preserves the previous stepper. The public layouts
match the pressure solver and include its ghosts.

Advance composes corrected advection, original-field diffusion, sponge and
boundary treatment, compatible projection, density/dt pressure conversion, and
pressure plus pre-step wall-friction loads. Original caller velocity remains
available until friction evaluation completes. Only after every stage succeeds
are owned velocity and physical-pressure prefixes copied to callers; trailing
ghosts remain untouched. Projection nonconvergence and non-finite load results
leave the entire caller state unchanged. A load failure may report a converged
projection while `advanced` remains false; callers must use `advanced` as the
commit signal. CUDA failures during final copies can still leave partial output.
Stage timings include synchronization and the final load/commit work.

This supplies a complete fixed-topology multi-GPU step, not yet the adaptive
runtime: device selection, initialization, distributed timestep selection,
regridding/remapping, snapshots and task integration remain required. Positive
scaling and the 2% steady-load gate still require full-flow measurements.

Job 2001401 passed direct/CTest full-step regression and memcheck with zero
unsuppressed errors under the exact expected-704 peer-status suppression.
Thirty-two geometry/scale/device configurations each advance three consecutive
steps against single-device compatible projection and transport, changing
speed, timestep and linear/cubic sampling. Coverage includes one/two/four GPUs,
permuted devices, uniform/mixed grids and embedded bodies. Maximum accumulated
velocity/physical-pressure/load component differences were 6.89e-15, 8.53e-14
and 1.07e-14. Forced one-iteration projection nonconvergence and non-finite wall
loads after a converged projection preserve every caller velocity/pressure
entry, including ghosts; subsequent valid steps succeed. Build rollback,
current-device restoration, untouched success-path ghosts and finite stage
timings are also checked. These short comparisons do not establish steady-state
load agreement across adaptive and uniformly fine meshes.

### Distributed adaptive runtime

`GpuDistributedFlow` provides initialization, advancement, speed maximum,
readback, explicit regridding and policy-driven adaptation using the shared
`FlowParameters`, `FlowInfo` and `FlowSnapshot` contracts. It retains owned GPU
fields between timesteps. Ordinary advancement and speed selection do not
gather complete fields on the host; snapshots and explicit adaptation remain
cold paths that assemble global IDs from uniquely owned prefixes.

Regridding builds a candidate mesh with the retained immutable body callback
and aperture reuse, conservatively transfers velocity on the host, scatters it
to the new ownership, and projects it before swapping state. The stepper's
`ProjectVelocity` uses private projection buffers and discards its remap
potential. Candidate physical pressure therefore remains zero and invalid;
time and step count persist, while generation increments once on a changed
successful grid. No-op grids preserve the complete state. Rejected domain,
body-resolution, capacity or numerical changes preserve the previous runtime
and refinement report.

The speed maximum shares the single-device area-weighted cell-centre formula,
reads raw owned velocity without slip extension, and returns one scalar per
rank for a host maximum. Non-finite owned inputs or computed speeds preserve
the caller's output. Shared kernels select owned cells, including inactive-cell
handling, and all public runtime operations restore the current CUDA device.

The command-line adaptive task has not yet selected this runtime; task wiring,
large-case memory/performance measurements and the 2% steady-load acceptance
remain open. Host adaptation/remapping and replicated immutable geometry also
remain performance limitations to measure, rather than assumed speedups.

Job 2001405 passed distributed-flow, distributed-step, distributed-load/speed
and existing single-flow CTests, plus distributed-flow memcheck with zero
unsuppressed errors under the exact expected-704 suppression. Lifecycle tests
cover refinement/coarsening, geometry preservation, pressure clearing,
clock/generation continuity, no-op grids, invalid initialization/domain/body
resolution, capacity/report rollback and tight pressure tolerances on one/two/
four GPUs with permuted mappings. Eight explicit single-device comparisons
advance through fine/mixed regrids and changed speed, checking fields, loads,
speed maxima and current-device restoration. Maximum velocity/pressure
component differences were 6.67e-16 and 7.44e-15; post-remap divergence stayed
below 1.69e-11. The speed kernel also agrees with an independent CPU cell-average
maximum in the 240 load scenarios, including all-solid grids, and preserves its
output on non-finite input. These small cases are lifecycle verification, not
large-case acceleration or steady-load acceptance.

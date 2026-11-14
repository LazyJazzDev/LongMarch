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
- Existing uniform-grid pressure projection is on one GPU; merely distributing
  transport is insufficient for the requested speedup.

## Remaining implementation

1. Build coarse/fine face connectivity, block ownership and explicit halo
   exchange suitable for CUDA kernels and single-node peer communication.
   Keep conservative face flux restriction and consistent pressure gradient /
   divergence pairs at refinement interfaces. A face-centre neighbor query alone
   is not a complete coarse-to-fine face stencil.
2. Add field transfer and regridding, preserving constant fields and fluxes;
   project transferred velocities and validate divergence. Refine aircraft and
   control surfaces to the reference spacing; use flow-error indicators with
   a buffer and hysteresis for wakes and moving refinement boundaries.
3. Implement composite-grid pressure projection and a parallel preconditioner;
   validate symmetry, nonnegative energy, manufactured pressure solutions,
   interface flux cancellation and residual criteria on the composite operator.
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

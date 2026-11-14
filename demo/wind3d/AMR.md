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

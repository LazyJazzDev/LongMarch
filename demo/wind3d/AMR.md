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

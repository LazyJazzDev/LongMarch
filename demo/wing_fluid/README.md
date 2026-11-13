# Wing Fluid

Interactive two-dimensional incompressible Navier–Stokes flow around a NACA 0012
wing section. CUDA computes the flow; RHI displays a shared CUDA buffer through
Vulkan or D3D12. An NVIDIA GPU with CUDA/RHI external-memory support is required.
The demo is built only when `LONGMARCH_CUDA_ENABLED` is enabled.

```sh
cmake --build cmake-build-release --target demo_wing_fluid
cmake-build-release/demo/wing_fluid/demo_wing_fluid --backend vulkan
```

On Windows append `.exe`; `--backend d3d12` selects D3D12. With no backend option,
RHI's platform default is used. The first CUDA-capable RHI adapter is selected.

## Controls

- **Flow speed:** 0.1–30 m/s (logarithmic slider) inlet velocity.
- **Angle of attack:** −25° to +25°; positive incidence raises the leading edge.
- **Display:** advected dye streaks, speed, signed vorticity, or pressure.
- **Pause / Reset flow:** freeze time or restart from projected uniform inflow and straight dye streaks.
- **Simulation speed:** requested simulated seconds per wall-clock second.
- **Solver settings:** smooth wing boundary (enabled by default) and kinematic viscosity, 0.0005–0.01 m²/s. Switching boundary modes resets the flow.
  `--boundary smooth|voxel` selects the mode at startup.
- **Load trends window:** a separate window with a cyan resultant-force arrow,
  an orange moment arc, and recent Fx/Fy/Mz curves. Closing it leaves the simulation
  running; this checkbox opens it again. Pause freezes the curves; Reset clears them.

Loads combine pressure and approximate tangential viscous shear integrated over
the curved wing contour (or fluid/solid faces in voxel mode), per unit span.
The smooth-mode contour has 512 cosine-spaced segments; pressure and tangential
velocity are sampled one grid spacing outside it. Wall shear remains approximate. Force is normalized by `0.5 rho U² c` and
moment by `0.5 rho U² c²`, using the current simulated inlet speed and chord.
Positive x points downstream, positive y upward, and positive moment is
counterclockwise about mid-chord. The body remains fixed; no mass or inertia is assumed.
The display uses 0.2 simulated seconds of exponential smoothing. Arrow length and
arc extent use fixed compressive scales (not per-frame auto-normalization); the
three history plots auto-scale symmetrically around zero. These are qualitative
load trends, not calibrated aerodynamic coefficients. Speed normalization removes
the direct dynamic-pressure factor, so doubling inlet speed need not double an arrow.
Reset's initialization projection is excluded until an actual timestep completes.

The default grid is **512×256**, with **eight multigrid levels**, ending at 4×2.
The domain is a 6×3 m channel. The 1.25 m chord is centered at (2, 1.5) m;
approximately 107 cells resolve the chord. `--grid 128|256|512|1024` changes the
resolution at startup. The hierarchy continues to 4×2 at every resolution.

Interactive simulation uses wall-clock time and automatically subdivides each
frame subject to the advection and explicit-viscosity timestep bounds. At most
32 substeps and 50 ms of elapsed wall time are consumed per frame, so slow devices,
large grids, high speeds or a stalled UI can make simulation time lag real time.
The displayed simulation clock makes this visible. Resizing preserves aspect ratio.

## Numerical method

With unit density, the equations are

```text
du/dt + (u · ∇)u = −∇p + ν∇²u
∇ · u = 0
```

1. A staggered MAC grid stores horizontal/vertical face velocities and cell-centered
   pressure. The NACA 0012 closed-trailing-edge curve is rotated about mid-chord.
   Smooth mode computes each grid face's open fraction using eight subsegments
   and interpolated intersections with the analytic implicit curve. Fractions below
   0.01 are closed and above 0.99 are fully open to avoid vanishing apertures.
   Cells with any open face participate in the pressure solve. Voxel mode retains
   the original binary cell mask and fully open/closed faces.
2. Midpoint-backtraced semi-Lagrangian velocity advection is followed by a separate
   explicit viscous-diffusion pass. The diffusion bound is `ν dt / dx² <= 0.2`.
   Partially blocked faces receive an additional implicit wall-drag approximation
   proportional to their blocked fraction; this is not a resolved no-slip boundary layer.
3. Pressure projection uses open-face-weighted flux divergence and the matching
   variable-coefficient Poisson operator, with the ordinary pressure-gradient velocity
   correction. The reported divergence is net open-face flux divided by full cell area,
   not by the possibly tiny fluid portion of a cut cell.
   Inlet velocity is prescribed, the downstream pressure is zero, top/bottom walls
   are free-slip, and fully blocked faces have zero velocity. Stationary solid surfaces
   contribute zero normal flux. Voxel mode closes all faces adjacent to solid cells.
4. The pressure Poisson equation uses five geometric multigrid V-cycles per substep:
   three red/black Gauss–Seidel sweeps before and after coarse correction, summed
   residual restriction, averaged coarse face conductances, and bilinear correction
   prolongation. Boundary extensions preserve Neumann/Dirichlet conditions. The
   final eight-variable system is solved directly on the GPU. Repeated V-cycles
   are recorded in a CUDA Graph; the graph sees updated coefficient arrays when
   the wing angle changes. Reset uses 32 V-cycles to project its initial velocity
   without advancing the simulation clock, so a paused reset is also incompressible.
5. Passive dye uses bounded MacCormack correction and inlet stripes. It does not
   exert forces or introduce artificial vorticity into the velocity field.

The shared output is an RHI `ICUDABuffer` containing cell velocity, pressure and
dye. A CUDA timeline waits for the previous graphics event before export, then
signals an event consumed by the next graphics submission. Queue submission
handles Vulkan external ownership barriers. One graphics frame is kept in flight;
shutdown drains both CUDA and graphics before releasing imported resources.

This is an exploratory laminar 2D visualization, not an aerodynamic design tool:
approximate cut-face geometry and wall drag, semi-Lagrangian numerical diffusion, finite channel
walls and a coarse near-wall mesh affect separation and wake behavior. There is
no turbulence model, 3D sweep/span, compressibility or validated lift/drag output.
Changing incidence rebuilds a stationary obstacle and introduces a transient;
it does not model the velocity of a physically pitching wing.

## Reproducible validation

```sh
demo_wing_fluid --validate --backend vulkan --debug --frames 120
demo_wing_fluid --validate --backend d3d12 --debug --frames 120
demo_wing_fluid --headless --frames 2000 --steps 16 --speed 30 --angle 25 --viscosity 0.0005
demo_wing_fluid --headless --frames 1200 --view vorticity --capture wake.ppm
```

`--validate` runs a manufactured **discrete** pressure solve (`b = A p_exact`),
checks relative residual and solution error, then tests projection across sudden
speed/incidence changes without resets between cases. It rejects nonfinite fields,
nonzero solid velocity or insufficient divergence reduction. It subsequently runs
the actual RHI offscreen rendering path. This verifies the implemented discrete
solver and interop; it does not establish continuum/grid convergence.
It also checks the surface-load integration against constant and linear pressure
fields on the selected closed surface, including force and moment signs.
The speed/incidence transition checks include 30 m/s at 25 degrees.

`--headless` creates no window and defaults to 120 frames. `--steps` specifies
deterministic substeps per headless frame (default four); it does not override
interactive automatic substepping. `--capture` writes a PPM from the final RHI
render target, including the UI when used with bounded window rendering.
Diagnostics sample divergence and maximum speed every 20 frames and at the final
bounded frame, with a small host readback. No flow fields are uploaded from the CPU.

References: [Bridson's fluid simulation course notes](https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf)
for MAC projection and boundary conditions, and
[NVIDIA GPU Gems, Fast Fluid Dynamics](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
for semi-Lagrangian splitting. This implementation uses multigrid rather than the
GPU Gems chapter's Jacobi-only pressure iteration.

Smooth pressure boundaries follow the open-face-weighted approach described by
[Batty, Bertails and Bridson (2007)](https://www.cs.ubc.ca/labs/imager/tr/2007/Batty_VariationalFluids/).
This demo uses approximate face intersections and split viscosity, not their full
rigid-body coupling method. The displayed smooth contour is independently antialiased;
the solver itself also uses fractional apertures.

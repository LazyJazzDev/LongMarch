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

- **Flow speed:** 0.1–3 m/s inlet velocity.
- **Angle of attack:** −25° to +25°; positive incidence raises the leading edge.
- **Display:** advected dye streaks, speed, signed vorticity, or pressure.
- **Pause / Reset flow:** freeze time or restart from projected uniform inflow and straight dye streaks.
- **Simulation speed:** requested simulated seconds per wall-clock second.
- **Solver settings:** kinematic viscosity, 0.0005–0.01 m²/s.
- **Load trends window:** a separate window with a cyan resultant-force arrow,
  an orange moment arc, and recent Fx/Fy/Mz curves. Closing it leaves the simulation
  running; this checkbox opens it again. Pause freezes the curves; Reset clears them.

Loads combine pressure and approximate tangential viscous shear integrated over
fluid/solid voxel faces, per unit span. Force is normalized by `0.5 rho U² c` and
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
   pressure. The NACA 0012 closed-trailing-edge thickness formula defines a binary
   cell mask, rotated about mid-chord.
2. Midpoint-backtraced semi-Lagrangian velocity advection is followed by a separate
   explicit viscous-diffusion pass. The diffusion bound is `ν dt / dx² <= 0.2`.
3. Pressure projection uses matching masked divergence and gradient operators.
   Inlet velocity is prescribed, the downstream pressure is zero, top/bottom walls
   are free-slip, and all faces touching the stationary voxelized wing have zero
   velocity. Pressure has homogeneous Neumann conditions at solid walls and inlet.
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
binary/stair-step geometry, semi-Lagrangian numerical diffusion, finite channel
walls and a coarse near-wall mesh affect separation and wake behavior. There is
no turbulence model, 3D sweep/span, compressibility or validated lift/drag output.
Changing incidence revoxelizes a stationary obstacle and introduces a transient;
it does not model the velocity of a physically pitching wing.

## Reproducible validation

```sh
demo_wing_fluid --validate --backend vulkan --debug --frames 120
demo_wing_fluid --validate --backend d3d12 --debug --frames 120
demo_wing_fluid --headless --frames 2000 --steps 8 --speed 3 --angle 25 --viscosity 0.0005
demo_wing_fluid --headless --frames 1200 --view vorticity --capture wake.ppm
```

`--validate` runs a manufactured **discrete** pressure solve (`b = A p_exact`),
checks relative residual and solution error, then tests projection across sudden
speed/incidence changes without resets between cases. It rejects nonfinite fields,
nonzero solid velocity or insufficient divergence reduction. It subsequently runs
the actual RHI offscreen rendering path. This verifies the implemented discrete
solver and interop; it does not establish continuum/grid convergence.
It also checks the surface-load integration against constant and linear pressure
fields on the closed voxel body, including force and moment signs.

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

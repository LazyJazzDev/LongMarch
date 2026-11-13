# Wing Fluid

Interactive two-dimensional incompressible Navier–Stokes flow around selectable
wing sections (NACA 0012 by default). The flow runs on either CUDA or portable RHI
Compute Shaders. Compute mode uses ordinary GPU buffers and supports non-CUDA
adapters; the demo also builds without CUDA Toolkit. RHI provides rendering and
compute through Vulkan, D3D12 or Metal (subject to the platform/backend).

```sh
cmake --build cmake-build-release --target demo_wing_fluid
cmake-build-release/demo/wing_fluid/demo_wing_fluid --backend vulkan
```

On Windows append `.exe`; `--backend d3d12` selects D3D12. With no backend option,
RHI's platform default is used; `--backend metal` selects Metal on macOS.
The first creatable RHI adapter is selected unless `--device INDEX` is specified.
Use `--list-devices` to see adapter indices for the selected graphics backend.

## Compute backend

- `--solver auto` (default): CUDA when compiled in and supported by the selected
  device; otherwise Compute Shader. CUDA initialization failure also falls back
  to compute in auto mode.
- `--solver compute`: force the RHI compute implementation, including on NVIDIA.
- `--solver cuda`: require CUDA; report failure rather than silently changing solver.

```sh
demo_wing_fluid --backend d3d12 --list-devices
demo_wing_fluid --backend d3d12 --device 1 --solver compute
demo_wing_fluid --solver compute --validate --grid 256 --frames 20
```

The two solvers implement the same MAC discretization, multigrid V-cycles, boundary
weights, dye advection, load integration and validation. CUDA uses a cached pressure
CUDA Graph. Compute uses ordered RHI dispatches and a GPU buffer arena; diagnostics
and loads use FP32 tree reductions followed by small CPU readbacks, without FP64
or floating-point atomic requirements. Results need not be bit-identical. Compute
submissions are batched and use the same queue as rendering. Integrated GPUs may
benefit from `--grid 128` or `--grid 256`.

The compute path batches immutable dispatch parameters into aligned uniform-buffer
slices, avoiding an upload between every pair of kernels. Red/black smoothing uses
one lane per active cell in a dedicated pipeline. The final 16x8, 8x4 and 4x2 grids
run the same V-cycle inside one thread group, with device-memory barriers between
stages; larger grids retain separate dispatches for global synchronization. These
optimizations do not reduce resolution, pressure cycles or smoothing sweeps.

On a CUDA-capable device in a CUDA-enabled build, the main window offers
**Switch to: Compute Shader / CUDA**. Switching preserves the staggered velocity
fields, pressure, dye, simulation time, current parameters, pause state and load
history. It does not reset the flow. The previous frame is drained, state is copied
through host memory, and the output binding is replaced only after the destination
solver is ready. The alternate solver is created on first use and then retained;
the first switch may briefly pause for allocation and shader compilation. Subsequent
switches still incur a state-copy pause. Failed initialization/transfer leaves the
original solver selected and displays an error. The button is absent on non-CUDA
devices and in CUDA-free builds; the startup `--solver` option remains available.

For bounded switching regression, use `--switch-every N` (requires CUDA support).
With `--validate`, each transfer also verifies exact equality of the copied fields:

```sh
demo_wing_fluid --solver cuda --validate --grid 128 --frames 24 --switch-every 3
```

In interactive mode, the switching frame does not advance simulation time. Headless
switch tests keep deterministic stepping, so their final state can be compared with
an otherwise identical run without switching.

## Wing sections

Select **Wing section** in the main window, or use `--profile NAME`:

| CLI name | Section | Shape |
| --- | --- | --- |
| `naca0012` | NACA 0012 (default) | Symmetric, 12% thickness |
| `naca0006` | NACA 0006 | Thin symmetric, 6% thickness |
| `naca0024` | NACA 0024 | Thick symmetric, 24% thickness |
| `cambered2` | Cambered 2% | 12% thickness; 2% maximum camber at 40% chord |
| `cambered4` | Cambered 4% | 12% thickness; 4% maximum camber at 40% chord |
| `reflex` | Reflex | 12% thickness; S-shaped mean line |
| `ellipse` | Ellipse | Symmetric ellipse, 12% thickness |
| `double-wedge` | Double wedge | Diamond section, 12% thickness |

All use the same chord, pivot and fluid parameters. Switching section resets the
flow and load history, including while paused. Both boundary modes support all
sections. The solver, surface-load integration, load-window outline and display
shader share the same profile functions in `wing_profile.h`.

These are **2D sections**, not 3D swept/delta wing planforms. The symmetric NACA
profiles use a closed trailing edge. The two cambered sections apply the four-digit
mean line with *vertical* thickness offsets, so they are not exact normal-offset
NACA 2412/4412 profiles. The reflex mean line is `0.16 x (1-x) (1-2x)`.
Coarse grids resolve the thin section poorly; 512×256 or higher is preferable.

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
   pressure. The selected section curve is rotated about mid-chord.
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

The output buffer contains cell velocity, pressure and dye. CUDA uses an RHI
`ICUDABuffer` and an external timeline to order CUDA writes and graphics reads;
queue submission handles Vulkan external ownership barriers. Compute mode uses
an ordinary `IBuffer` and RHI-managed dispatch dependencies. The application waits
for the preceding graphics frame before advancing the simulation and waits for the
solver's export event before drawing. Shutdown drains outstanding work before
releasing GPU resources.

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
demo_wing_fluid --validate --profile cambered4 --speed 30 --angle 25 --frames 120
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

The symmetric thickness and cambered mean-line formulas follow
[PDAS thickness](https://www.pdas.com/naca456thick4.html) and
[PDAS mean lines](https://www.pdas.com/naca456mean2.html), with the closed trailing
edge and vertical-offset approximations described above.

A CUDA-free build can be verified by configuring the existing build directory
with `-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON`; this omits the CUDA solver
and external-memory path while retaining the compute demo.

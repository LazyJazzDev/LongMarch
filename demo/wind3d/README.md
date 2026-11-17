# Wind3D — asynchronous CUDA wind tunnel

A three-dimensional incompressible flow demonstration around a procedural
airliner model. CUDA advances the fluid on a worker thread while an RHI frontend
renders the latest completed state. NVIDIA CUDA is required; the Windows
frontend supports Vulkan and D3D12.

```powershell
cmake --build cmake-build-release --target demo_wind3d
cmake-build-release/demo/wind3d/demo_wind3d --aero --backend vulkan
cmake-build-release/demo/wind3d/demo_wind3d --aero --backend d3d12 --grid 256
```

The procedural airframe is built around a 6 m-long narrow-body
transport: nominal span about 5.80 m (5.72 m with raked tips), cabin diameter 0.63 m, and nacelle outer
diameter approximately 0.40 m. These proportions use the A320 family as a
reference, not an exact certified aircraft surface:
[Airbus aircraft characteristics](https://www.aircraft.airbus.com/sites/g/files/jlcbta126/files/2024-06/AC_A320_0624.pdf).
The default blended wingtip has a circular 0.18 m root blend and a 70-degree
canted outer section. Section normals rotate with the centerline, preserving
normal thickness; the same analytic forward/inverse mapping defines the mesh
and solid boundary. Select **Blended winglets** in the desktop control panel,
`--winglets`, or `--raked-tips` to compare tip configurations. This adds a
geometric option, not a guaranteed drag improvement. Diagnostics record the
choice as `winglets`; projected span accounts for the bend rather than using
the tip's developed length.

Horizontal and vertical tail roots now flare in both chord and thickness,
with a smooth compact transition back to their regular sections. The nacelle
pylon has separate lower and upper shoulders meeting the nacelle crown and
wing underside. The same invertible section deformation generates their
rendered skins and solid boundaries. These are generic junction fairings;
no claim of globally tangent production surfaces or measured drag reduction
is implied. The wing-body belly and root shoulders remain integrated in the
fuselage skin.

Aerodynamic coefficient normalization now uses the main-wing trapezoid,
including its buried carry-through and excluding wingtip devices: nominal
area 3.127 m2, reference span 5.30 m, and MAC 0.667345 m. The area-weighted
quarter-chord body x coordinate is -0.293852 m. Displayed tip-to-tip span still
includes the selected tip device. `aircraft_reference.h` supplies this shared
convention; legacy sweep coefficients that included tip developed length use
a different reference and must not be compared without renormalization.

Desktop and web load readouts additionally show moment about the assumed trim
CG at `(quarter_chord_x, 0, 0)` in body coordinates. The offset is rotated into
world coordinates before applying `M_CG = M_origin - r_CG cross F`. This is an
explicit reference assumption, not inferred aircraft mass properties. Existing
history curves and moment arrows remain about the body origin and are labeled
accordingly; a nonzero origin moment can coexist with zero CG pitch moment.

The horizontal stabilizer has an independent incidence setting via
`--tail-incidence -10..10` and **Wind and controls / Stabilizer trim** in the
desktop UI. Positive incidence raises its leading edge. The whole stabilizer
rotates about its root quarter-chord; the elevator remains independently
hinged. Diagnostics record `tail_incidence`, and geometry changes invalidate
the existing solid mask through the normal airframe configuration path.
This is a geometric trim control, not an automatic trim solver.

The main wing now varies its sections continuously: thickness/chord falls
from 15% to 12%, camber/chord from 2.5% to 1.5%, and incidence decreases by
3 degrees from root to tip. A shared reversible section deformation includes
the control hinge fairings; the tip extension inherits the exact terminal
section, with continuity checked in the geometry test. These are initial
transport-like design choices, not a measured A320 airfoil reconstruction.
The wing, empennage and pylon dimensions share the same geometric definitions
between the renderer and fluid boundary. Older checkpoints and the historical
9.3099 L/D results below describe the previous, larger-span geometry; do not
restore those fields as a settled state for the revised aircraft. New trim
results must be generated from the revised geometry.

`--aero` reproduces the aerodynamic study settings: LFM, air kinematic viscosity
and molecular wall viscosity of 1.5e-5 m²/s, slip cut cells with wall-shear loads,
CFL 0.25, a 2 m outlet sponge and up to 128 pressure iterations per projection.
It preserves grid, camera, control, export and diagnostic options. Put it before
individual solver overrides. Without this preset, the original high-viscosity
desktop demonstration defaults remain in effect.

Desktop streamlines use seeds throughout the 16 x 8 x 12 m flow domain and
trace both directions, including regions disconnected from inlet streamlines.
Occupied seed bins limit duplicate seeds without truncating downstream paths.
In **View**, **Transparent curvature (1/m)** sets the fully transparent lower
threshold (default 0.01/m), and **Opaque curvature (1/m)** sets the fully opaque,
warm-colored upper threshold (default 0.15/m). Opacity scales linearly between
these thresholds using the **largest curvature encountered upstream**, so it
never decreases along the flow direction, even after a bend straightens out.
Entirely straight lines remain invisible. Each new streamline starts its own
history; this is spatial accumulation within a snapshot, not persistence across
simulation frames. Camera motion does not change the metric, and both sliders
update shading without rebuilding trajectories. Initial thresholds can also be
set with `--streamline-transparent-curvature 0.01 --streamline-curvature 0.15`.
The lower threshold must be smaller than the upper threshold.

The desktop ImGui window includes side-by-side force (Fx/Fy/Fz) and moment
(Mx/My/Mz) histories, using the same RGB axes and physical-time windows as the
web frontend. Each completed solver step is recorded, retaining up to two
seconds (65,536 samples maximum). Choose the time window and startup-step
filter, subtract each component mean, or hide the first 20 steps after a
control change. Hover for exact values; tables report time-weighted means,
peak-to-peak ranges and standard deviations. Moments use world axes about the
world origin. Pause freezes history; Reset clears it. Reopen a closed chart
using **Load history plots** in the right panel.

To inspect an existing solver checkpoint without repeating its startup, use
`--aero --grid 256 --restore path/to/flow.checkpoint`. Supply the original grid,
controls and solver settings: the checkpoint stores fields and load integrals,
not those settings. The viewer resumes from the saved cycle; its Reset control
still starts a fresh flow. Restoration errors are reported instead of silently
starting over.

For a reusable settled state, add `--save-checkpoint FILE` to a `--validate`
run. The requested aircraft state is paused and saved before the lifecycle
checks change controls or reset the flow. This checkpoint can be restored by
the desktop with the same settings, avoiding another startup transient.
The save option requires validation mode and reports write failures.

The desktop also accepts bounded airframe parameters: `--wing-span` (half-span
excluding the raked tip, 2–4.2 m), `--wing-camber` (0–0.06 chord), `--wing-sweep`
(leading-edge slope, 0–0.7), `--tail-span` (horizontal half-span, 0.6–1.4 m), and
`--tail-leading` (horizontal-tail leading edge at its root, body x 1–2 m).
They share one specification between the explicit mesh and CUDA occupancy,
including control-surface stations and the wing broad-phase bound. Diagnostic
CSV rows record all five parameters. Keep them identical when restoring a
checkpoint. These options vary the current conventional aircraft family;
they do not import arbitrary meshes or establish aerodynamic accuracy for a
new configuration.

## Leapfrog Flow Maps desktop solver

The desktop executable now defaults to **LFM on a 256 × 128 × 192 grid**.
`--grid 128` provides a faster preview; `--maccormack --grid 192` selects the previous
desktop default. The library, batch and AMR defaults are unchanged. LFM currently
requires one GPU; requesting it through `SolverOptions` with multiple devices
fails explicitly.

```powershell
cmake-build-release/demo/wind3d/demo_wind3d --lfm --grid 256
cmake-build-release/demo/wind3d/demo_wind3d --lfm --validate --grid 96 --steps 1000 --diagnostics out/lfm.csv
cmake --build cmake-build-release --target wind3d_lfm_check
cmake-build-release/demo/wind3d/wind3d_lfm_check --lfm
```

The implementation follows the hybrid velocity/impulse algorithm in
[Sun et al., Leapfrog Flow Maps, SIGGRAPH 2025, sections 4.1–4.3](https://yuchen-sun-cg.github.io/projects/lfm/static/pdfs/SIG_2025_Leapfrog_Flow_Maps.pdf):

1. Project the initial velocity after initialization, geometry or inlet changes.
2. Construct five projected midpoint velocities with leapfrog transport. The
   first transport interval is dt/2, the second dt, and subsequent intervals 2dt
   from the preceding alternating midpoint field.
3. March backward and forward maps and their Jacobian columns with RK4 through
   the stored midpoint fields. Pull back the initial velocity as an impulse,
   perform a forward/backward BFECC error correction, and limit that correction.
   The limiter clamps to the six adjacent **uncorrected pulled-back impulses**,
   matching the reference implementation's `BfeccClampKernel`. Initial velocity
   bounds are not interchangeable with impulse bounds after a Jacobian pullback.
4. Apply viscosity over the complete cycle, enforce boundaries, then project
   the reconstructed impulse into a divergence-free velocity.
5. Recover **physical pressure separately** from a short Eulerian momentum
   predictor with centered spatial derivatives. Its Poisson RHS uses the divergence of the velocity increment,
   removing the residual divergence of the final flow. The impulse projection
   potential is never used as surface pressure.

This is a native adaptation to Wind3D's MAC arrays, geometry and pressure solver,
not a binary-equivalent port of the authors' tiled CUDA solver. Map interpolation,
its analytic derivatives and impulse pullback use the reference quadratic
B-spline (N2) kernel on extended MAC velocities. Leapfrog midpoint prediction
also uses N2 for RK2 velocity tracing and transported values, as in the reference.
Apertures remain in divergence and projection; they do not normalize the map interpolant.
`--cubic` affects the MacCormack path; LFM consistently uses quadratic interpolation.
Values outside each component's MAC extent use a constant extension. Viscosity uses a
first-order operator split, not the paper's accumulated-force path integral.
The existing wall-shear and pressure surface integration supplies all six load
components. Maps are reinitialized every five substeps; published snapshots,
control changes and checkpoints occur only at complete cycle boundaries.
`--cfl` applies per substep; `--max-dt` caps the entire published cycle.

The panel shows force and moment components and magnitudes, lift = Fy, drag = Fx,
instantaneous L/D, and time-weighted mean lift / mean drag since reset (including
startup). Near-zero drag produces an undefined ratio, not infinity. Moments are
about the fixed body origin. `--diagnostics file.csv` exports every completed
cycle, including loads, magnitudes, instantaneous/mean L/D and the solver flag;
undefined ratios are empty CSV fields. The PCG count sums the cycle's projections
and physical-pressure solve; the residual is their maximum. The existing timing
bins include midpoint projections in transport, final impulse projection in
pressure, and physical-pressure recovery in diagnostics; `step_seconds` is the
complete worker-cycle wall time. Frontend FPS, solver Hz and physical time / wall
time are displayed separately. Interactive rendering is capped at 60 Hz to leave
GPU time for solving.

### Aerodynamic investigation controls

The desktop CLI exposes `--viscosity`, `--wall-viscosity`, `--wall-model`,
`--slip-walls`, `--cut-cells`, `--cubic`, `--sponge` and `--no-export` for controlled
solver comparisons. Viscosities are kinematic, in m²/s. `--cut-cells` enables the
existing slip-flow/wall-shear load split; it is not a resolved boundary layer.
LFM extrapolates masked face velocities for every midpoint and for the physical
pressure predictor, consistently with this slip-wall interpolation. The physical
velocity still satisfies the blocked-face boundary condition after projection.

Diagnostic CSV rows now include pressure and friction force components, each
aircraft component's drag/lift, all control settings, grid and physical solver
parameters. Component ordering is fuselage, wing, horizontal tail, vertical tail,
and nacelle (including pylons). Use a fixed physical-time averaging window and
check pressure convergence before comparing runs; a higher L/D by itself does
not prove accuracy. The desktop defaults are unchanged by these investigation
controls.

```powershell
demo_wind3d --lfm --grid 128 --validate-time 3 --no-export --viscosity 0.000015 --wall-viscosity 0.000015 --cut-cells --cubic --sponge 2 --cfl 0.25 --diagnostics out/aero.csv
```

### Vorticity volume rendering

For the web-style streamline view, use `--streamlines --no-volume` or the
corresponding **View** controls. Desktop streamlines share the web tracer and
speed colors, with the same four seed densities (84/287/610/1053, default 287)
and opacity rule (0.72 divided by the density level). The RHI draws instanced
triangle ribbons with analytic pixel coverage, connected limited-miter joins
and round endpoints instead of hardware line primitives. Width stays constant
in screen pixels and is adjustable from 0.75 to 4 with **Streamline width** or
`--streamline-width`; `--streamline-density` selects the density. They depth-test
against the aircraft without writing depth, and composite after the vorticity
volume so enabling both views does not make transparent lines terminate volume
rays. Grid and load-vector overlays retain their existing rendering.

The default view ray-marches the full solver-grid magnitude of curl(u), computed
in CUDA from the final projected MAC velocity. The volume has 256 × 128 × 192
samples at the default resolution; it does not use the 192-wide display-field
cap used by streamlines. Solid cells and the outermost domain cells are zeroed.
The renderer uses trilinear sampling, exponential absorption and three light-ray
samples for volumetric shadows, in the style of the standalone LFM demo. This is
vorticity visualization, not a separate smoke-density or combustion simulation.

Open **View** for volume visibility, grey/color modes, density, threshold in
s^-1, ray-step size and optional streamlines. The aircraft and load vectors remain
visible through depth-aware composition: an R32 color attachment stores the
rasterized surface depth, so the RHI does not need sampled depth-format support.
The ray stops at the nearest surface and composites with its shaded color.
Immutable snapshots keep the volume, geometry and load panel on the same completed
simulation cycle. `--no-flow` hides both volume and streamlines; `--model-only`
also hides load vectors. Give the higher-resolution flow time to develop its wake.

### Validation and measured limits

The pressure-recovery test samples analytic potential flow around a 0.75 m sphere
at 12 m/s, then exercises the actual predictor, face apertures and pressure solve.
At grids 64/128/256, the area-weighted pressure-coefficient RMS errors are
0.136/0.078/0.047; net pressure drag stays below 0.006 N (analytic drag is zero).
The previous semi-Lagrangian pressure predictor produced 126/77/44 N of numerical
drag on the same fields. Centered momentum derivatives remove this interpolation
contribution to the recovered physical pressure. This validates a curved slip
boundary, not turbulent aircraft aerodynamics. `wind3d_lfm_check --lfm` includes
these sphere checks, an analytic rotating-flow predictor check, and checkpoints.

With quadratic maps and centered pressure recovery, the current aircraft gives
the following time-weighted loads in the 2–3 second window (12 m/s, 5 degrees,
cut cells, molecular viscosity 1.5e-5 m²/s, CFL 0.25, 2 m sponge; `--aero`):

| Grid | Drag (N) | Lift (N) | Mean lift / mean drag |
| --- | ---: | ---: | ---: |
| 128 × 64 × 96 | 22.60 | 143.38 | 6.3456 |
| 256 × 128 × 192 | 28.82 | 268.30 | 9.3099 |
| 320 × 160 × 240 | 27.28 | 289.95 | 10.6298 |

The 256 run completed 731 cycles to 3.00071 seconds, with no failed projections
or nonfinite loads, and passed checkpoint writing and lifecycle validation.
The 320 run completed 819 cycles to 3.00041 seconds and passed the same checks.
For both runs, the 2–2.5 and 2.5–3 second L/D values differ by less than 1e-6.
This demonstrates temporal settling, not spatial convergence: L/D increases
14.2% from 256 to 320, with most of the lift increase on the main wing and lower
tail/nacelle drag. The 384 study was stopped before completion when the evaluation
priority changed; its partial trajectory is not a convergence result.

A current-build continuation of the 256 checkpoint through 3.123 seconds
reproduced L/D 9.309895 (mean lift 268.297 N, drag 28.819 N), with all
projections and pause/reset/control/resume lifecycle checks passing. This
checks preservation of the settled state; it does not replace the fresh-start
trajectory above or establish spatial convergence.

The evaluation priority is computational efficiency and credible lift, drag,
and control forces/moments across aircraft configurations. Strict spatial
convergence is not an acceptance requirement. L/D near 10 at this one condition
is a reference result, not a calibration target for other aircraft. Compare
force and moment responses to incidence and flap/aileron/elevator/rudder changes,
using consistent geometry, reference axes and physical settings, and report
accuracy together with computation cost. Temporal settling and failed-projection
checks remain necessary. Reference aerodynamic data are needed before claiming
quantitative accuracy across configurations; the current fixed-aircraft runs
alone do not establish it.

A 128-grid control-response screen at 12 m/s and 5 degrees used the time-weighted
1–1.5 second window. Each fresh-start case required about 12.6–12.8 seconds of
cumulative solver time on the RTX 5080 (excluding setup, rendering and lifecycle
checks). All seven cases passed projection and lifecycle checks. Body-axis
moments follow the right-hand rule about the fixed body origin, not a specified
center of mass; roll/pitch/yaw are body x/z/y respectively.

| Control | Body roll (N m) | Body pitch (N m) | Body yaw (N m) |
| --- | ---: | ---: | ---: |
| Neutral | 0.10 | -10.14 | 0.02 |
| Aileron +10° | -33.92 | -3.13 | 1.03 |
| Aileron -10° | 34.14 | -3.13 | -0.99 |
| Elevator +10° | 0.11 | 32.53 | 0.02 |
| Elevator -10° | 0.09 | -41.17 | 0.02 |
| Rudder +10° | 1.13 | -10.14 | -3.50 |
| Rudder -10° | -0.93 | -10.14 | 3.54 |

This checks control-response direction and lateral symmetry, not agreement with
measured aircraft derivatives. The desktop load panel displays both world-axis
components and these body-axis moments using the snapshot's applied incidence.

Reusing transverse products in the quadratic sampler reduces the median cost
from 8.275 to 8.166 wall seconds per physical second in three alternating
128-grid fresh-start runs on the RTX 5080, with display exports disabled.
The reference-window lift and drag changes are below 0.0001 N. This is a small
arithmetic optimization, not a different integration or boundary model.
Faster transport choices need separate aerodynamic validation: in the same
128-grid 1–1.5 second window, cubic MacCormack gives 75.58 N lift versus LFM's
143.40 N, and increasing LFM CFL from 0.25 to 0.5 gives 120.05 N. These settings
are not equivalent accelerations of the reference result.

The pressure hierarchy caches each assembled double-precision diagonal and its
reciprocal when geometry changes. Three alternating 128-grid runs reduce the
median solver cost from 8.231 to 6.080 wall seconds per physical second (26.1%),
with the same CFL, pressure tolerance and physics. Reference-window lift and
drag differ by less than 0.000001 N. Krylov vectors, matrix application and true
residual checks remain double precision. The cache costs about 110 MiB at grid
256 across all hierarchy levels, included in the allocation estimate.
The main half-wing span increased from 2.7 to 3.6 m with the same root/tip
chords. Including raked tips, total span is 7.84 m and planform aspect
ratio is about 11.56. Flap/aileron stations and the occupancy broad phase follow
the new span. Maximum main-wing camber increased from 2.5% to 4% chord,
keeping its 40%-chord peak, 13% thickness and incidence unchanged. The 6.04 m
span with 2.5% camber gave L/D 3.5752; increasing span alone gave 5.1137 under
the same algorithm and 128-grid conditions. At grid 256, increasing camber from
2.5% to 4% on the longer wing changes the 2–3 second L/D from 8.6627 to 9.3099.
Keeping the centered pressure predictor but using the earlier trilinear maps
gave L/D 0.51 on the same grid. The interpolation regression checks a sampled
quadratic and its derivative across a stencil transition, and constant-velocity
transport of a quadratic field, in addition to affine flow maps.
Later performance/load measurements below predate these corrections
and should not be used as current aerodynamic predictions.

Pressure PCG and its multigrid work vectors use double precision, including dot
products and true residual evaluation. Velocity and published pressure remain
single precision. This prevents loss of convergence during large inlet-speed
changes; it increases solver memory and time, especially on consumer GPUs.
LFM stops if an initial or midpoint projection fails to converge, before using
that field to advance flow maps. CTest `wind3d_lfm_lifecycle` and
`wind3d_lfm_cut_lifecycle` cover reset, combined control-surface changes and a
live inlet change at grid 128, for voxel and cut-cell configurations respectively.

Windows Release, RTX 5080, 12 m/s, 5 degree incidence, default voxel/no-slip model:

| Grid | Run | Median cycle time after cycle 9 | Simulated time / solver wall time |
| --- | --- | --- | --- |
| 96 × 48 × 72 | 1,000 cycles, about 30.4 simulated seconds | 22.3 ms | 1.35× |
| 128 × 64 × 96 | 100 cycles, about 2.2 simulated seconds | 38.2 ms | 0.58× |
| 256 × 128 × 192 | 100 cycles, about 0.97 simulated seconds, full vorticity export | 328 ms | 0.030× |

These are solver measurements, not frontend frame rates or a claim that LFM is
faster than MacCormack at the same resolution. The 96/128-grid results are preview measurements; they do not describe the
new 256-grid default. Higher grids do not retain the same real-time rate. Vulkan and D3D12 window captures verify actual aircraft, flow,
load vectors and the analysis panel. D3D12 validation reports the existing
missing-optimized-clear-value performance warnings (820/821).

GPU affine-field tests check both map directions, Jacobian columns and impulse
pullback against constant translation and analytic shear (maximum error
2.9e-6). A manufactured isolated impulse overshoot checks that the limiter uses
the impulse neighbourhood and preserves a constant field. Runtime validation checks pressure convergence and divergence reduction,
plus pause/reset/control changes and immutable snapshots. The checkpoint test
uses cut-cell geometry and verifies exact resumed loads, integrals and time,
including a fresh solver instance and a bounded snapshot queue.
CUDA Compute Sanitizer memcheck at grid 64 (three cycles plus lifecycle changes)
reports zero errors.

The 96-grid run's time-weighted L/D after cycle 9 is about 0.35; the short 128-grid
run gives about 0.41. The different time windows and discretizations do **not**
establish grid convergence. The default is an interactive qualitative wind-tunnel
preview, not a validated aircraft performance prediction. Refinement, time-step
convergence, turbulence modelling and comparison with reference aerodynamic data
remain necessary for quantitative design use.

The panel controls wind speed, incidence, collective flaps, differential
ailerons, elevator and rudder. Drag outside the panel to orbit, or use the
camera sliders; the mouse wheel zooms. Pausing the solver leaves the camera and
window responsive. Reset starts a new flow history with the requested controls.
Changes take effect at a solver-step boundary. The displayed model, flow and
loads always belong to the same completed snapshot, so a slow backend can lag
behind the requested panel values.

The model is a generic scaled narrow-body airliner, 6 m long with a 7.84 m
span. It has a constant-section cabin, rounded forebody, upswept tailcone,
continuous belly/wing-root fairing, cockpit glazing, passenger windows and doors.
The low-mounted main wing has sweep, dihedral, camber and raked tips. Two
stationary nacelles and pylons are part of both the visible geometry and solid
boundary. They are unpowered flow-through nacelles with an annular wall;
this is not an engine mass-flow or thrust model.

The trailing edge is split into inner and outer flap panels plus differential
ailerons. **Link inner/outer flaps** preserves a single flap command by default;
unlink it to set each panel independently. `--outer-flap 0..40` unlinks the
outer panel from `--flap`. **Cruise camber** (`--cruise-camber -5..5`) adds a
symmetric angular bias to both flap panels and ailerons without extending the
Fowler tracks. **Aileron droop** (`--aileron-droop 0..15`) adds a separate
symmetric bias to the roll controls. These are explicit geometric deflections,
not force multipliers. Load histories track these changes, and diagnostics
append all four configuration fields.

Five leading-edge slat elements per wing can be deployed with **Leading-edge
slats** / `--slat 0..30`. **Link slats to flaps** / `--link-slats` commands 75%
of the larger flap deployment. Each element is a small closed airfoil, buried
inside the main wing when retracted. The simplified telescoping track moves it
forward by 16% and down by 9% of the local chord, reaching full travel at 15
degrees, while rotating its section nose down. Deployed aft gaps and stowed
containment are checked against the same analytic solids used by the solver.
The main wing's existing nose remains as the exposed main-element nose; this
is a generic slotted leading edge, not an exact production slat/cove mechanism.
It adds no empirical lift increment or stall correction.

Four finite-thickness spoiler panels per wing follow the upper wing skin when
stowed and rotate upward about their swept leading hinges. **Speed brakes**
(`--spoiler 0..60`) raises all panels symmetrically. **Spoiler roll**
(`--spoiler-roll -30..30`) adds differential deployment to the three outer
panels per side; **Link spoilers to ailerons** (`--link-spoiler-roll`) uses the
aileron command instead. Each panel clamps to 0..60 degrees. The 6 mm panel
solid and its closed rendering mesh use the same transform. The shallow pocket
below a stowed spoiler is not modeled; the fixed wing skin remains underneath.
Thin-panel load accuracy remains limited by the chosen fluid grid.
A 128-grid, 30 m/s, 6-degree-incidence smoke check (mean over 0.5..0.6 s)
measured `(drag, lift)` in N as `(115.84, 368.85)` stowed,
`(113.14, 325.45)` at 30 degrees and `(156.17, 300.74)` at 60 degrees.
Thus full deployment increases drag and dumps lift in this configuration;
30-degree deployment reduces lift but does not monotonically increase total
drag. These are coarse-grid behavior checks, not calibrated aircraft data.
The 30-degree roll-only command produced a weak world-x moment of -0.55 N m
at grid 128 and +0.36 N m at grid 256 in the same averaging window. The sign
change and small response mean spoiler roll authority is not validated at
these resolutions; the geometry/control path is implemented, but these values
must not be treated as calibrated control derivatives. No solver changes or
artificial force gains were introduced to hide this limitation.



Inboard Fowler-type flaps extend rearward and down as they deflect. Their
rounded leading edges translate with the flap, leaving a real open slot behind
the fixed wing; no connecting capsule fills this passage. Travel reaches
16% of root chord rearward and 6% downward at 20 degrees. The surface and solid
boundary share this motion. CPU geometry checks verify fluid passages through
the deployed slots as well as sealed ordinary hinges. Resolving the slot flow
still depends on the local fluid grid; this is not a validation of high-lift
performance. Outboard differential
ailerons, the horizontal stabilizer's elevators and the vertical stabilizer's
rudder rotate about their actual swept three-dimensional hinge lines. Control
surfaces have closed hinge, root and tip faces. Muted tan, blue and green surfaces
identify flaps, ailerons and elevators; the fin has a blue livery. The explicit
mesh and occupancy test share the complete geometry specification and hinge
transforms in `aircraft.h`. Rendering does not reconstruct an implicit surface.
The fluid still voxelizes the solid at its own grid resolution.

Cockpit glazing and two cabin-window rows are shallow surface markings used to
make the model's passenger-aircraft scale legible. They do not alter the
fluid solid or loads. Structural proportions and all movable aerodynamic
surfaces remain defined by the shared geometry functions.

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

## MacCormack numerical method and shared limits

The domain is 16 × 8 × 12 m, spanning x = [-6, 10], y = [-4, 4], z = [-6, 6].
The previous desktop MAC grid was 192 × 96 × 144. `--grid` sets the x resolution (64–640,
multiple of 16); the other dimensions retain this aspect ratio. Velocity lives
on staggered faces and pressure at cell centres. A MacCormack step performs:

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
surface force contributes `r × F` to the moment. Velocity and published fields
use single precision; pressure iterations and pressure dot products use double
precision. Load reductions use single-precision blocks with double-precision
CPU accumulation, with no double atomicAdd.

This is a qualitative laminar demonstration. It has no turbulence model,
compressibility, ground effect or moving-wall actuator dynamics. Controls are
quasi-static boundary changes, not a simulation of the transient swept volume
of a moving flap. Transport limiting remains dissipative, the pressure solve
has finite residual, and voxel boundaries underresolve thin trailing edges and
small deflections, especially at coarse grids. Higher grids improve geometric
resolution at substantially higher step cost. Grid 640 uses 640 × 320 × 480
cells at 2.5 cm spacing. Memory depends on transport mode: LFM adds maps,
Jacobians and midpoint velocities to the double-precision pressure workspace.
Startup estimates these allocations and rejects grids that exceed available
device memory. Mesh smoothness does not imply
equally fine fluid resolution. Do not interpret the loads as calibrated
aircraft performance data.

The absence of a turbulence model does not itself forbid unsteady vortices.
However, the default desktop viscosity is over 300 times that of room-temperature
air (about 1.5e-5 m²/s), the cell width at grid 256 is 6.25 cm, and the
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
plus stabilizer trim, wingtip selection, independent outer flaps, cruise camber,
aileron droop, slats, spoilers and their linkages. The mailbox preserves the
legacy six-control/density prefix and appends a `wing-v1` extension; old saved
relay states acquire neutral/default values for the new fields. Control-change
history events include all these parameters. The worker also supports
pause/resume and reset. The browser displays the actual aircraft mesh,
instantaneous streamlines, full-grid force and moment, and solver diagnostics.
Geometry and streamline construction are shared with the native renderer in
`visualization.h`; the web console does not substitute a different aircraft.

For a native Windows preview with the desktop aerodynamic settings, build
`wind3d_web` in the existing release directory, then run these in separate
terminals from the repository root:

```powershell
cmake-build-release/demo/wind3d/wind3d_web.exe out/wind3d-web/local-modern 128 1 aero 0
python demo/wind3d/web/server.py --directory out/wind3d-web/local-modern --bind 127.0.0.1 --port 10510
```

Open `http://127.0.0.1:10510`. The `aero` profile uses the existing LFM desktop
settings and starts at 30 m/s. The trailing zero means no automatic stop time.
The full procedural surface is exported independently of fluid grid resolution.
After geometry source changes, rebuild and restart the worker; the page fetches
the new mesh ID automatically. No fluid algorithm is changed by this profile.

On the login node, start the relay (Python 3 standard library only):

```sh
python3 demo/wind3d/web/server.py \
  --directory "$PWD/out/wind3d-web/live-optimized" --port 10510
```

Then submit `meshy-envelope submit demo/wind3d/web.star`. It builds and starts
the worker on one eight-GPU node at grid 512, with an eight-hour job limit.
The login service and GPU worker must use the same shared directory. The relay
binds to `0.0.0.0` by default; use `--bind 127.0.0.1` when accessing it solely
through an SSH tunnel. This is an interactive cluster service for users who can
reach the chosen port, not a multi-tenant authenticated web application. POST
controls require same-origin JSON and are range/finite-value checked on both
sides. Assets, including Three.js r160 (MIT), are served locally without a CDN.

Only display data crosses the node boundary:

- The solver volume remains on the compute node. Its bounded display snapshot
  is consumed there to trace the same tracing method as the original renderer.
  No velocity/pressure volume is written into the shared mailbox or served by HTTP.
- At most twice per second, the worker atomically replaces `frame.bin`: a padded
  JSON metadata header and float32 line positions/colors. Raw display frames grow with the selected streamline density. Old line frames are discarded, not queued.
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

## Optimized aerodynamic model

The task and web worker use the geometry and numerical model imported from
`~/LongMarch` commit `13ea18de`: higher-aspect-ratio wings, flow-through nacelles,
cut-face apertures, aperture-weighted interpolation, two-layer slip extrapolation,
Spalding wall-friction loads and the low-CFL MacCormack fallback. The eight-GPU
partitioning and fused coarse pressure smoother are retained. This is a
viscous/inviscid load model, not a turbulence-resolved engineering prediction.

`OptimizedSolverOptions()` selects resolved viscosity 1.5e-5 m²/s, wall viscosity
9.2e-7 m²/s (the source flight-Reynolds-number sweep), friction-only wall model,
cut cells, a 2 m outlet sponge and CFL 0.125. The native GUI retains its existing
solver defaults. Web default controls are 30 m/s and 4° incidence. Displayed
vectors are instantaneous; L/D uses differences of per-step integrated loads
across at least 0.15 s, after at least 0.55 s at unchanged controls. Reset or a
control change clears that average. Pressure convergence alone does not imply
that the aerodynamic transient has settled.

Build `wind3d_sweep` for time-averaged coefficient comparisons. It accepts the
source sweep arguments plus `--gpus N`; for the source 512-grid alpha4 reference:

```sh
wind3d_sweep --grid 512 --gpus 8 --cases cases.txt --wall-viscosity 9.2e-7 \
  --settle-time 0.4 --case-time 0.3 --average-time 0.15 \
  --friction-only --cut-cells --cfl 0.125 --output results.csv
```

Here `cases.txt` contains `alpha4 4 0 0 0 0`. Compare the averaged 0.55–0.70 s
window, not the first-frame force ratio. The original reference recorded
CL 0.384566, CD 0.048739 and L/D 7.8903. Validation of this port additionally
checks closed-surface pressure cancellation and bitwise 1/2/4/8-GPU fields.

The port's eight-H200 replay reproduced that reference at CSV precision:
Fx 107.986 N, Fy 852.039 N, CL 0.384566, CD 0.048739, L/D 7.8903
over 0.55008–0.70007 s (7,910 steps). The 1/2/4/8-GPU regression also
compares per-step loads as well as full/final-export fields.

The current `high-bypass-v2` geometry uses a neutral gray skin, with colors only
on flap, aileron, elevator and rudder surfaces. Decorative glazing, door outlines,
belly/tail paint and inlet/duct tinting are removed. The pylons are 0.18 m high
(previously 0.32 m); the unpowered flow-through nacelles use a larger rounded
inlet, a short 0.83 m fan cowl and a contracted aft body. Rendering and the solver
share these profiles. The reference coefficients above describe the earlier
imported geometry, not this revised engine installation.

The web console provides four streamline seed densities: 84, 287, 610 and
1,053 (default 287). Density changes retrace the current display field on the
compute node, including while paused; they do not change resolution, reset the
flow or transfer the volume. Streams can terminate at the aircraft or low speed.
The optional domain outline uses the solver's bounds: x [-6,10], y [-4,4],
z [-6,6] metres. Enabling it frames the entire box; this shows the computational
extent, not every individual grid cell.

For SSH forwarding, the browser requests `/api/frame?compact=1`. The relay
simplifies display polylines with a 2 mm positional and 0.02 RGB tolerance,
preserving streamline seeds and leaving simulation fields and loads unchanged.
It caches the compressed representation separately from the original packet.
Requests are serial and wait two seconds after each transfer to avoid filling
a slow tunnel with display updates. Meshes are still transferred only on change.

The viewport shows model length and span measured from the received mesh in
body axes (undoing incidence), plus a metre scale bar that follows zoom and
viewport size. The bar is calibrated at the orbit target depth; perspective
means it is not a universal ruler for objects at other depths. The current
model is 6.00 m long with 7.84 m span; the domain remains 16 × 8 × 12 m.

### Per-step load history

The web worker enables `SolverOptions::record_loads`. Every completed solver step
records physical time, actual adaptive dt, force (N), moment about the fixed body
origin (N·m), and applied controls, under the same publication lock as its
snapshot. The scalar-only ring retains the most recent 2 physical seconds
(with one boundary sample), capped at 65,536 samples; capacity truncation is
reported explicitly. Pause creates no duplicate samples, reset clears the ring,
and run/generation identifiers distinguish restarts. Native viewers do not
record this history unless requested.

The GPU worker atomically publishes `loads.bin` approximately once per wall-clock
second, independently of display-frame sampling. Its version-1 binary format is
a little-endian uint32 JSON header length, the JSON header, then packed 44-byte
`<Qdf6f` rows (step, time, dt, Fx/Fy/Fz, Mx/My/Mz). No solver-step samples are
thinned. The relay's `/api/loads?seconds=0.25` selects a complete window from
0.05, 0.1, 0.25, 0.5, 1 or 2 physical seconds and calculates dt-weighted means,
standard deviations, extrema and peak-to-peak amplitudes on the backend.
ETags, gzip and a bounded response cache limit transfer/processing costs.

The page's load-history section replaces both curves with each complete server
window. It never assembles history from arriving visualization frames. Its
2-second polling interval affects freshness, not physical sample spacing.
Two charts show force and moment, each overlaying its X/Y/Z scalar components
in red/green/blue. Each vertical axis fits the combined current-window extrema
of all three components with 8% padding on every update. Constant values receive
a small nonzero range. Subtract the backend mean to inspect fluctuations, and hover for
individual step/time/dt/load values. Dashed
lines mark control changes; the status reports actual sample count and dt range.
Early runs and capacity limits can provide less history than the requested window.

Validation: build `wind3d_load_history` and run CTest `wind3d_load_history` on a
CUDA node; run `python3 demo/wind3d/web/test_loads.py` and `test_server.py` for
variable-dt statistics, narrow spikes, window transport, gzip and ETags.

The chart defaults to excluding solver steps 1–20 after each reset from the
selected window and all its statistics/axis bounds. The startup-step selector
supports 0/5/20/50/100; zero restores the complete raw history. The relay applies
`skip_steps` before statistics and returns an explicit warm-up state if no
eligible samples exist yet. This is a configurable startup exclusion, not an
outlier detector: later spikes are retained. Raw worker history is unchanged.

Input-control transitions also mask the first 20 steps using the backend's
applied-control events (the first event is only the retained baseline). Masked
samples retain their timestamps and raw values but carry `invalid_indices`;
all statistics exclude them. The browser draws a gap without bridging it,
reports null on hover, and handles a completely masked window with empty axes.
Repeated changes extend the gap; display-density changes do not trigger it.

### Numerical import and adaptive-grid work

The task and sweep drivers accept `--cubic`, ported from upstream b20f9123.
It uses monotone Hermite transported values in open fluid and retains weighted
trilinear interpolation around cut faces; departure tracing stays linear.
The existing linear mode remains available as a reference. GPU parity is
checked by CTest `wind3d_multigpu_cubic`. Adaptive-grid scope, the accepted
accuracy gate, resource limit and remaining work are recorded in [AMR.md](AMR.md).

## Uniform solver deployment (Claude source 46184928)

The web agent uses the uniform MAC solver, monotone cubic transport, cut-cell
boundaries, wall-modelled loads and CFL 0.125. It retains this repository's
1/2/4/8-GPU execution and web controls; AMR is not part of the web deployment.
The original 2.5% wing camber and sealed control hinges were ported from the source;
the aerodynamic geometry update increases main-wing camber to 4%.
Neutral decorative colours, control-surface colours, short pylons and high-bypass
nacelles remain as requested for this frontend.

The source sweep checkpoint/restore implementation and its flushed CSV header
are also ported. Build `wind3d_sweep` in `cmake-build-release`; use
`--checkpoint DIRECTORY` to preserve sweep progress across preemption.
Source revision `46184928` is provenance, not a claim of aerodynamic validation
at every resolution or control setting.

## 通用气动模型描述（基础接口）

目标用途是游戏中的不可压缩气动近似：按物体离线采集不同来流与联动控制状态下的合力、合力矩，再拟合供实时引擎使用的模型。优先考虑采样效率、响应连续性与稳定性，不以严格网格收敛或指定升阻比作为验收要求。

`wind3d_aero_model` 提供与飞机参数无关的 JSON 描述、部件层级和控制混合。`wind3d_aero_mesh` 进一步加载 OBJ 部件，构建可在 CPU/CUDA 查询的局部三角形 BVH。相同 mesh 路径复用几何；部件姿态变化时不重建 BVH。原生 CUDA 求解器和 wind3d_model_sample 已接入这些网格与控制状态。代理模型训练和实时推理尚未实现。

```json
{
  "version": 1,
  "units": "metres",
  "reference_point": [0, 0, 0],
  "controls": [{"name": "pitch", "min": -1, "max": 1, "default": 0}],
  "parts": [
    {"name": "body", "mesh": "body.obj"},
    {"name": "tail", "mesh": "tail.obj", "parent": "body",
     "translation": [2, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
     "hinge": {"pivot": [0, 0, 0], "axis": [0, 0, 1],
               "offset_deg": 0, "min_deg": -30, "max_deg": 30,
               "weights": {"pitch": 25}}}
  ]
}
```

mesh 路径相对于 JSON 文件；坐标使用右手系与米，旋转使用单位四元数。部件变换按 `parent * bind * hinge` 组合，铰链轴和支点位于部件局部坐标中。角度为 `offset_deg + sum(weight * control)`，按机械行程限幅并返回 `limited`。多个部件可引用同一控制量，同一部件可混合多个控制量；控制输入自身越界时返回错误。

`BodyLoads` 接收世界坐标中的合力及关于世界原点的合力矩，返回物体坐标中的合力及关于指定物体局部参考点的合力矩。调用者必须传入刚体变换；力使用 N，力矩使用 N·m。

在现有原生构建目录构建 `wind3d_model_inspect` 和 `wind3d_aero_model_check`。运行 `wind3d_model_inspect model.json [按声明顺序排列的控制值]` 查看部件姿态；省略控制值时使用默认状态。运行 `wind3d_aero_model_check out/wind3d-aero/model-contract-test.json` 验证控制混合、限位、父子变换、铰链支点以及力矩参考点转换（参数为可覆盖并删除的临时测试文件）。

`wind3d_model_inspect --geometry model.json [控制值]` 同时加载网格并报告三角形和 BVH 节点数量。OBJ 应为无自交、封闭、法线朝外的实体表面；导入器焊接完全相同位置的顶点，检查退化三角形、边的封闭性与一致绕序以及正有向体积。它不进行完整自交检测，也不会自动填洞或修复错误的面朝向。材料、纹理不参与气动几何。

距离查询使用最近三角形特征的角度加权伪法线判断内外，参考 [Bærentzen / Aanæs 2005](https://orbit.dtu.dk/en/publications/signed-distance-computation-using-the-angle-weighted-pseudo-norma/)。本实现的数值表面容差为局部查询坐标尺度乘 64 倍双精度机器 epsilon，避免平面上的点因切向舍入或刚体变换误差而被误判为流体。`MeshView` 为直接上传 CUDA 的只读数组视图，复用同一查询实现；`ModelGeometry::Distance` 将查询点变换至各部件局部坐标，取有符号距离最小值构造实体并集（交叠内部不保证精确欧氏距离）。

构建并运行 `wind3d_aero_mesh_check out/wind3d-aero/mesh-contract-test.obj` 验证立方体解析距离、凹形内外判定、BVH 与穷举距离、几何共享、旋转平移和无效输入。启用 CUDA 的原生构建还会在 GPU 上检查相同查询。测试参数是可覆盖并删除的临时 OBJ 文件，并会使用相邻的 `.obj.json` 临时描述文件。

### 通用模型批量载荷采样（原生 CUDA）

构建目标 `wind3d_model_sample`，运行：

```text
cmake-build-release/demo/wind3d/wind3d_model_sample.exe model.json cases.json loads.csv --grid 128 --settle 0.5 --average 0.5
```

`cases.json` 使用以下格式，控制值按模型 `controls` 的声明顺序排列。省略控制值、速度或姿态时分别采用模型默认控制、12 m/s、单位旋转与零平移。case 名称必须唯一。

```json
{"version": 1, "cases": [
  {"name": "neutral", "speed": 12, "controls": [0]},
  {"name": "pitch_positive", "speed": 12, "controls": [0.5]},
  {"name": "rotated", "speed": 12, "controls": [0],
   "rotation_xyzw": [0, 0, 0.7071067811865476, 0.7071067811865476],
   "translation": [0, 0, 0]}
]}
```

来流沿世界 +X；四元数把物体坐标转至世界坐标。CSV 保存物体坐标中的来流速度、控制量、平均合力（N）和关于模型 `reference_point` 的平均合力矩（N·m），以及窗口内载荷波动的 RMS、最大压力残差和该状态采样阶段的墙钟耗时（不含一次性初始化）。RMS 是各分量时间方差之和的平方根，不是绝对载荷的均方根，也不是对实测数据的误差。

整个批次复用一套求解器和每个网格的 GPU BVH。切换控制/姿态时更新实例变换、固体掩码与压力算子，保留流场作为初值；因此每个状态先等待 `--settle` 秒物理时间，再在 `--average` 秒物理时间内逐步时间加权平均。CSV 的窗口时间相对于该状态首个纳入处理的步长，从零计时，并用双精度累计物理步长。窗口长度由使用者根据输出波动选择，目前不自动判定稳态，也不模拟舵面运动时的移动壁面速度。失败时返回非零并保留 `.partial` 文件；全部状态完成后才发布目标 CSV，不覆盖已有输出。

默认采用 LFM、空气运动黏度 1.5e-5 m²/s、密度 1.225 kg/m³、滑移切割单元与壁面摩擦载荷、CFL 0.25、2 m 出口缓冲区。可显式选择 `--maccormack`，但其耗散和载荷可能不同，CSV 会记录算法。初版使用一张 CUDA GPU，计算域仍为 X[-6,10]、Y[-4,4]、Z[-6,6] 米，所有姿态的包围盒须离边界至少两个单元；不支持地面边界与移动地面。

程序接口为 `SolverOptions::model / initial_model_state / initial_body_pose` 和 `Simulation::ConfigureModel`。`Snapshot::force/moment` 保持世界坐标、关于世界原点的原语义；通用模型的 `Snapshot::body_loads` 才是物体坐标下关于模型参考点的载荷。传统飞机部件分项不适用于通用模型，保持为零。通用模型渲染顶点会随姿态更新；当前桌面窗口的交互入口仍是原有飞机控件，通用模型的用户入口是上述批采样工具。

`wind3d_aero_solver_check out/wind3d-aero/solver-integration-test.obj` 覆盖真实 OBJ 流体边界、控制切换后的流场复用、显示网格姿态、载荷坐标/参考点转换，以及越界控制、非刚体变换和域外几何的拒绝。测试路径及相邻 `.obj.json` 为可覆盖并删除的临时文件。验证这些数据通路不代表已验证任意外形的气动精度。

For settling checks, repeat a state without changing controls and compare
successive physical-time windows for **both force and moment**. Also test
returning to that state after a different deflection. A 128-grid box test with
2 s settling and 0.5 s averaging changed its mean moment by 8.38% when the
same state continued for another 2.5 s, even though pressure converged. After
an A-to-B-to-A sequence and the same additional continuation, mean force and
moment agreed with the earlier continued A state within 0.010% and 0.048%,
respectively. This is evidence of repeatable relaxation for that test, not a
universal settling time or validation of arbitrary-shape aerodynamic accuracy.
The default 0.5 s wait is a starting setting, not a declaration of steady loads;
small within-window RMS alone cannot rule out drift over longer time scales.

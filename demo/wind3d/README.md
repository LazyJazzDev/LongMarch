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

The model has a streamlined fuselage, swept tapered main wings with dihedral,
inboard flaps, outboard ailerons, a horizontal tail/elevator and a vertical
tail/rudder. It has no engines. Its length is 6 m and its span is 9 m; this is a
scaled demonstration geometry, not a particular full-size aircraft. Orange
marks flaps, blue marks ailerons, green marks elevator surfaces and purple marks
the rudder. Rendering extracts a smooth surface from the same implicit geometry
used by the fluid boundary.

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
The default MAC grid is 128 × 64 × 96. `--grid` sets the x resolution (64–256,
multiple of 16); the other dimensions retain this aspect ratio. Velocity lives
on staggered faces and pressure at cell centres. Each step performs:

1. Rebuild the solid mask from the current aircraft geometry.
2. Semi-Lagrangian velocity advection and explicit viscous diffusion.
3. Enforce stationary no-penetration face velocities at solid boundaries.
4. Solve the pressure Poisson equation using 400 red-black SOR iterations.
5. Subtract the pressure gradient and measure the remaining divergence.
6. Export the completed velocity/pressure field and integrate loads.

The air density is 1.225 kg/m³; the effective kinematic viscosity is 0.005 m²/s.
The time step is bounded by advective CFL 0.5 and an explicit diffusion bound.
The inlet and lateral far-field faces impose the wind; the outlet has zero
pressure. Pressure force is integrated over exposed voxel faces. A first-order
tangential wall-shear estimate is added using the same viscosity, and each
surface force contributes `r × F` to the moment. GPU fields use single precision;
CPU diagnostic norms accumulate in double precision, with no double atomicAdd.

This is a qualitative laminar demonstration. It has no turbulence model,
compressibility, ground effect or moving-wall actuator dynamics. Controls are
quasi-static boundary changes, not a simulation of the transient swept volume
of a moving flap. Semi-Lagrangian transport is dissipative, the pressure solve
has finite residual, and voxel boundaries underresolve thin trailing edges and
small deflections, especially at coarse grids. The 256 grid improves geometric
resolution at substantially higher step cost. Mesh smoothness does not imply
equally fine fluid resolution. Do not interpret the loads as calibrated
aircraft performance data.

## Asynchronous ownership

A single worker owns the CUDA stream and all fluid allocations. It copies a
completed field to a host snapshot, calculates diagnostics/loads, and publishes
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

`--validate` checks finite fluid state and requires pressure projection to reduce
RMS divergence below 10% of its input (or 0.002 s⁻¹ for nearly divergence-free
inputs). It prints force, moment, physical time and solver wall time per observed
snapshot. It then checks pause acknowledgement, reset while paused, applying all controls, resume, and retained snapshot immutability; a failed check returns exit code 2. Use `--help` for parameter bounds.
Startup has an impulsive loading transient; compare evolved states at equivalent
physical times, not just equal step counts. `--headless` renders offscreen and
adds an 8 ms pacing delay, so its reported FPS is not a window benchmark.
Use Nsight for GPU and full-frame profiling; this demo adds no timestamp query
interface.

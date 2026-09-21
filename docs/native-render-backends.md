# Native CPU and CUDA path tracing

## Library boundary and backend selection

Sparkium's public rendering interface consumes an independent in-memory
`SceneDefinition`. The loader reads scene files and assets once; GUI, CLI and
library clients then pass the same snapshot to `Renderer::SetScene`. Switching
backends reconstructs execution state without reloading scene files.
See [the scene API](sparkium-scene-api.md) for the complete ownership contract.

```cpp
auto scene = sparkium::LoadScene("scene.json");
auto renderer = sparkium::CreateRenderer({sparkium::RenderBackend::CPU});
renderer->SetScene(scene);
renderer->Render();
auto image = renderer->ReadImage();
```

`RendererSettings` selects Graphics, CPU or CUDA and optionally the Graphics API.
Neither the scene model nor the renderer interface exposes Buffer, Image, Shader,
CommandContext or graphics device handles. The GUI owns a separate display device.
CPU textures share immutable scene pixels; CPU BVH construction reads the host
mesh directly. Shader geometry packing remains backend-derived data.

The existing execution machinery remains under `code/sparkium/backend/`:

- `graphics/` adapts Vulkan, D3D12 and Metal devices from graphics.
- `cpu/` owns LLVM function JIT execution and the persistent thread pool.
- `cuda/` owns CUDA context, NVRTC compilation, kernel launches and optional OptiX.
- `common/` contains shared bindings, resources, Slang frontend and compatibility.

These internal resource adapters still use Graphics abstract resource contracts
to run the existing pipelines. They are not the scene-facing backend API.
`detail::SceneInstance` creates private execution objects from the scene model;
legacy `Core` callers remain supported. Graphics itself contains no CPU/CUDA
compute backend. OptiX remains a CUDA traversal pipeline.

For NVIDIA hardware traversal on CUDA, see optional
[OptiX ray tracing](optix-ray-tracing.md). Use `--pipeline rt_fallback` to
explicitly select software BVH traversal on CUDA.

Sparkium's headless path tracer can execute without a graphics API:

```sh
build/demo/sparkium_cli/demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend cpu --frames 2 -o cpu.png --linear-output cpu.pfm
build/demo/sparkium_cli/demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend cuda --pipeline rt_fallback --frames 2 -o cuda.png --linear-output cuda.pfm
```

On CPU, `auto`, `ray_tracing` (including legacy JSON requests), and
`rt_fallback` resolve to the shared compute path tracer. On CUDA, `ray_tracing`
uses OptiX, `rt_fallback` uses software BVH, and `auto` selects hardware tracing
when available. An explicit CUDA `ray_tracing` request fails if OptiX is
unavailable; use `auto` to allow fallback. `--require-hardware-rt` accepts CUDA
hardware tracing and rejects software traversal.

Vulkan and D3D12 `auto` prefer Ray Query when supported, then pipeline RT,
then software fallback. Explicit pipeline choices remain available. CPU/CUDA are
headless compute devices, **not new rasterizers or window-system backends**.
Native `rasterization`, inline `ray_query`, presentation and CUDA/graphics
interop buffers are unsupported. CUDA supports triangle acceleration structures
when OptiX is available. The GUI uses a
separate graphics device to display their output.

## GUI preview

```sh
cmake --build cmake-build-ninja --target demo_sparkium_gui
cmake-build-ninja/demo/sparkium_gui/demo_sparkium_gui assets/scenes/cornell_box/scene.json --backend cpu
cmake-build-ninja/demo/sparkium_gui/demo_sparkium_gui assets/scenes/cornell_box/scene.json --backend cuda --display-backend vulkan
```

On Windows, append `.exe` to the executable name. `--backend` selects scene
rendering; `--display-backend auto|d3d12|vulkan|metal` independently selects window
presentation. The default display backend follows the platform default. CPU and
CUDA cannot be selected as display backends.

The ImGui **Render backend** dropdown selects Graphics, CPU or CUDA without
reopening the window. Graphics exposes a separate **Graphics API** dropdown for
Vulkan, D3D12 and Metal. `--backend` sets the initial selection.
Switching waits for the current dispatch on the worker thread, recreates the
render device and scene there, resets accumulation, and selects the Auto
pipeline for the new device. The scene and samples-per-dispatch selection are
retained; the display device and window remain active. Shader compilation can
delay the first preview, but the dropdown remains usable while loading or after
an initialization error, so another backend can be selected immediately.

The render worker owns its device, scene, film, shader compilation and readback
on one dedicated thread, including creation and destruction of the CUDA context.
The main thread owns GLFW, ImGui and a separate display device. They exchange
immutable RGBA8 frames through a bounded latest-frame mailbox; no GPU resources
or graphics command queues are shared. Preview readback is limited to 30 Hz,
while the GUI refreshes at up to 60 Hz and rendering continues at its own rate.
Sampling does not wait for a preview to be consumed. CPU/GPU transfer and shared
hardware contention still have a cost; this is thread isolation, not a separate
OS process or a guarantee of zero rendering overhead.

Scene changes, reloads, pipeline edits, sample edits and film resets are applied
between dispatches. Pending edits are coalesced, and stale frames from a previous
request are discarded. Errors appear in the GUI and a new scene/reload can retry.
Closing the window requests shutdown; resource cleanup waits for the current
load/dispatch to finish. Render timing and camera-ray throughput use backend
dispatch time, separately from GUI FPS. `--frames N` exits after the worker has
completed N render dispatches and the final preview has been presented; errors
produce a nonzero exit status in this mode.

CPU offers Auto and the fallback path tracer in the pipeline selector. CUDA
also offers Path Tracing when OptiX is available; Auto selects OptiX. A
scene whose default is rasterization or ray query uses the fallback path tracer
when viewed on CPU/CUDA.

## Build

The optional native backend requires the Slang compiler **library and headers**.
CPU rendering additionally requires a matching **slang-llvm shared library** and
Slang's direct LLVM emitter. The vcpkg manifest pins `shader-slang` 2026.7.1#1
through its registry baseline. This port packages official Slang binaries,
including the LLVM plugin. No external C++ compiler or linker is used at runtime. CUDA additionally
requires the CUDA driver and NVRTC. NVRTC
selects the compute architecture of the actual selected device at runtime.
OptiX is optional and only needed for CUDA hardware ray tracing.

```sh
cmake -S . -B build -G Ninja -DVCPKG_PATH=/opt/vcpkg \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target demo_sparkium_cli sparkium_native_test \
  sparkium_fallback_test -j6
```

`LONGMARCH_ENABLE_NATIVE_RENDER` defaults to `ON`. CMake discovers the vcpkg
package with `find_package(slang 2026.7.1 CONFIG)` and uses its `slang::slang`
and `slang::slang-llvm` targets. Windows builds deploy both matching DLLs beside
executables, including the dynamically loaded LLVM plugin. No temporary download
directory or Vulkan SDK Slang fallback is needed. For a custom package, set
`slang_DIR` to the directory containing its `slangConfig.cmake`.
If either target is absent, configuration reports native rendering disabled;
the existing graphics backends remain buildable. Set
`-DLONGMARCH_ENABLE_NATIVE_RENDER=OFF` to disable it explicitly.

On Linux/macOS, make the matching `libslang-llvm.so`/`libslang-llvm.dylib`
available to the Slang shared-library loader. Mixing compiler and LLVM plugin
versions is unsupported. The bundled Windows LLVM DLL is approximately 105 MiB.

For a CUDA-free build, including no CUDA runtime or NVRTC linkage:

```sh
cmake -S . -B build-cpu -G Ninja -DVCPKG_PATH=/opt/vcpkg \
  -DCMAKE_BUILD_TYPE=Release -DLONGMARCH_DISABLE_CUDA=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON
cmake --build build-cpu --target demo_sparkium_cli sparkium_native_test -j6
```

Both flags matter: the pre-existing root CMake configuration allows the CUDA
runtime/graphics interop even when CUDA-language compilation is disabled.
The second flag also disables that toolkit discovery. This configuration can
still link Vulkan/DXC for the other available backends; selecting CPU does not
initialize or use them. Slang and its LLVM shared library must remain available
during rendering; GCC/MSVC/Clang executables are not required.

## Embedded CPU compilation

CPU first reflects compute declarations with code generation disabled, then
selects explicit CPU entry macros and exports ordinary functions. Machine-code
compilation selects `SLANG_HOST_HOST_CALLABLE`, whole-program generation and
`-emit-cpu-via-llvm`, without registering a compute entry point.
`spGetTargetHostCallable` returns the module containing the exported range function.
The pipeline is:

```text
Shared HLSL + explicit native contract -> Slang IR -> LLVM IR -> in-process JIT -> CPU function
```

There is no generated C++ compilation step and no compiler/linker subprocess.
The compiler still uses private temporary source files for includes;
executable code is generated and retained by the JIT module. A missing LLVM
plugin or unsupported direct-LLVM option fails with an error, without falling
back to an external toolchain. CUDA already uses the in-process NVRTC API.

The direct LLVM emitter is described as experimental by
[Slang's LLVM target documentation](https://github.com/shader-slang/slang/blob/v2026.8/docs/llvm-target.md).
It lacks native texture/sampler types and group barriers; this backend supplies
its own texture representation and serial CPU scan variants. Graphics-only
`NonUniformResourceIndex` annotations are removed for ordinary native arrays.
Existing buffer/image ABI checks continue to validate reflected layouts.

On Windows, `sparkium_native_no_subprocess_test` enables the OS
`NoChildProcessCreation` mitigation before running the CPU suite. It covers
descriptor arrays, images, sampling, accumulation/reset, and a mesh/material/BVH
test with known radiance, with compiler/linker child processes prohibited.

## Shared implementation

`sparkium/backend/{graphics,cpu,cuda,common}` implements Sparkium's rendering devices and
resources. JSON loading, mesh/hair conversion, shader-graph generation, material
registration, film and camera semantics remain shared. `SoftwarePipeline` selects
a dedicated CPU acceleration-structure builder independently of the GPU path.

The **same HLSL sources** supply:

- watertight triangle intersection arithmetic (tree construction/traversal is backend-specific);
- `RenderPixel`, camera/lens sampling, Sobol/Wang random generation and roulette;
- hit records, all material samplers, Principled BSDF evaluation/sampling;
- direct-light sampling, light power, MIS, alpha shadows and subsurface walks;
- accumulation, reset, clamping, persistence, film resolve and tone mapping.

CPU JIT-compiles the shared shading functions and its dedicated traversal into
one Slang LLVM module; there is no callback across the JIT/C++ boundary per ray. A
persistent, process-wide `std::thread` pool dynamically distributes ranges of eight
workgroups; the generated function computes invocation IDs and calls the shared
shader body. The submitting thread also participates. Up to 16 workgroups run
inline. Concurrent submissions share the pool and are serialized. There is no
OpenMP dependency, and workers are reused across frames and backend switches.

By default the pool uses `std::thread::hardware_concurrency()` threads (including
the caller). Set `SPARKIUM_CPU_THREADS` to limit this count; `OMP_NUM_THREADS` is
accepted as a legacy fallback. The value must be a positive integer, is capped
at the hardware count, and is read when the pool is first used.
`SPARKIUM_CPU_GRAIN` optionally sets the positive number of workgroups claimed
per task (default 8), also read once. These are diagnostic tuning controls.

CUDA compiles Slang's CUDA source through NVRTC to PTX,
loads it with the CUDA driver, and calls `cuLaunchKernel`. OptiX instead links
the tracing module and calls `optixLaunch`. Software BVH construction,
light preprocessing, path tracing, film resolve and tone mapping all execute
on the selected device. CUDA descriptors and pixels use device memory, not
managed CPU rendering followed by GPU copies. Submission is synchronous.

Only three parallel prefix-scan shaders need `SPARKIUM_NATIVE_CPU` variants:
CPU performs the per-workgroup scan serially because Slang host execution does
not implement GPU wave/barrier scheduling. Their inputs, power evaluators,
hierarchical scan and CDF consumers remain shared. Summation order can differ.

### Compiler and resource boundary

Sparkium shaders use `native_contract.hlsli`: explicit macros describe buffer
specialization, mutating methods, resource bindings and entry attributes. Sources
are passed unchanged to Slang's preprocessor; the Sparkium path performs no
regex rewriting, template erasure, cbuffer splitting or resource-symbol renaming.
`SP_CONTEXT`/`SP_CONTEXT_ARG` propagate an explicit CPU context through resource
consumers. Pure computation and the shading algorithms remain shared with GPU.
Graphics backends expand these macros to ordinary HLSL declarations.

Each CPU call receives a `NativeContext*` followed by the workgroup range and grid
sizes. The context stores descriptors inline in 64 fixed 32-byte slots (2 KiB total),
avoiding repeated pointer chasing in resource access. It lives until the
synchronous pool submission finishes. Resources are not JIT module
globals, and explicit modules need no resource-binding mutex. Native buffers use
pointer/size spans, arrays use `NativeArray<T>`, and typed constant buffers point
to host data. Binding validates reflected descriptor sizes, required constant
field ranges, resource ownership and slot presence. An exported ABI probe checks
the plain context/sampler structures. Slang's `sizeof(resource)` reports its HLSL
logical size, so resource descriptors are checked using reflection and integration
tests, not that operator. CUDA continues to bind `SLANG_globalParams`.

The older raw-HLSL `graphics::Core::CreateShader` interface retains its constrained
adapter in `native_shader_compat.cpp`, including module-global bindings and a
per-module mutex. This is compatibility debt; it is isolated and bypassed for the
entire Sparkium renderer, not silently removed from existing callers.

CPU compilation is serialized around the shared Slang session. Up to 16 explicit
CPU modules are retained in an LRU cache keyed by exact VFS contents, source,
entry and compiler defines. A shader edit invalidates the key. Resource bindings
are excluded because they are per call. Repeated materials and GUI reloads can
reuse executable modules; cache eviction does not invalidate live shader objects.
This is process-local reuse, not a disk machine-code cache.

### CPU acceleration structure

`cpu_bvh.cpp` builds a deterministic binary BVH using twelve SAH bins and normally
up to four primitives per BLAS leaf and one instance per TLAS leaf. Adjacent child nodes and a packed primitive index
array replace the GPU's complete heap/Morton layout. Depth is capped at 48, with
64-entry traversal stacks. BLAS data is retained; TLAS construction/upload is
skipped when instances are unchanged. Transform or instance changes update TLAS
without recompiling shading. As in the original pipeline, geometry identity and
primitive count drive BLAS invalidation; in-place geometry mutation needs an
explicit revision mechanism before supporting it as a new feature.

`software/cpu_traversal.hlsli` traverses near children first and reuses per-ray
reciprocals. Pending stack entries retain their near distance, avoiding a second
box test after popping; helper functions return a small child pair rather than
copying traversal stacks. GPU software BVH construction and traversal remain independent.
Hit semantics are distance in original ray units, instance/primitive identity
and barycentric coordinates; nonuniform transforms do not normalize object rays.
Transparent-shadow filtering remains in shared material code above intersection.
`SPARKIUM_CPU_BVH=heap` selects the original shared heap BVH only for differential
and performance-ablation checks; `sah` is the default. This is not a GUI option.

Byte buffer-backed image types implement the path tracer's explicit level-zero
loads and nearest/bilinear sampling, repeat, mirrored repeat, edge and zero-border
addressing. SDR stays packed RGBA8 (decoded before filtering, rounded to UNORM on
shader writes); HDR values are retained as floats. Keeping SDR packed matters
for the full Junkshop scene, which otherwise exceeds a 12 GiB CPU memory limit.
The
loader and shader-side color-space conversion remain shared. This is not a
general graphics image implementation: there are no mip chains, derivatives,
anisotropic texture filtering or arbitrary typed format reinterpretation.
Graphics hardware's texture interpolation/UNORM rounding may
differ from the native float implementation.

## Feature coverage and inherited limits

Slang 2026.7.1's LLVM CPU target incorrectly evaluates
`all(isfinite(float3(...)))` for finite inputs. This previously bypassed the
material-graph and subsurface direct-light BSDF/PDF weighting and left raw light
power in the shadow contribution, severely overexposing CPU renders. Shared
`AllFinite(float3)` now classifies each scalar component explicitly. This keeps
CPU/CUDA/GPU shading semantics aligned without source rewriting or an external
compiler. `FiniteLightClassification` covers finite values, infinities and NaN
in each channel; `ShaderGraphDirectLightingMatchesPrincipled` compares actual
one-bounce radiance against the equivalent built-in material.

The repaired CPU/CUDA PNG RMSE is 0.000553 on `graph_smoke` (64x64, 32 spp),
and 0.002702 / 0.003144 / 0.003603 on Blender Monster / Classroom / Junkshop
(long edge 256, preserved aspect ratio, 64 spp, complete geometry/materials).
All pass the unchanged 0.01 native-backend threshold. These checks establish
backend agreement for these fixtures, not full-resolution or Cycles equivalence.
See `out/cpu-graph-fix/` for before/after images, commands and regression results;
`out/blender-native-correctness/` preserves the failing baseline.

The native path tracer retains the baseline's:

| Feature family | Native path |
| --- | --- |
| Scene geometry | OBJ, inline mesh, `SPKMESH1`, generated triangle spheres, `SPKHAIR1` hair tubes, normals/UV/tangents/colors |
| Instances | Matrix/TRS/look-at, nonuniform/mirrored transforms, activation and transform updates |
| Materials | Lambertian, specular, emissive, Principled and generated shader graphs |
| Principled inputs | Base/SSS color and radius, metallic, specular/tint, roughness, anisotropy/rotation, sheen/tint, coat/roughness, IOR, transmission/roughness, emission and existing texture slots |
| Graph surface | Opacity, normal, emission, Principled parameters, both thin-glass branches, radius/scale/method random-walk subsurface |
| Lighting | Emissive geometry, two-sided/visibility/blocking/falloff flags, delta and finite-radius point lights, soft falloff, sampling weights, MIS and transparent shadows |
| Camera | Look-at pinhole, FOV/aspect, thin lens, focus, polygonal/rotated/anamorphic aperture |
| Film | Multi-frame accumulation/reset, persistence, radiance/exposure clamps, normalized/standard/approximate Filmic, exposure/gamma/contrast |
| Sampling | Same pixel/sample initialization and Sobol table, Wang fallback, bounce limit and roulette |

All graph node families accepted by the existing loader use the same generated
code: value/RGB, texture coordinate, image/mapping, noise/Voronoi/gradient/wave/
sky/brick textures, vertex attribute, object/geometry/light-path information,
invert/mix/math, ramps/curves, hue-saturation/gamma/brightness-contrast, layer
weight, normal map/bump, combine/separate, passthrough and invert-Y. “Supported”
means **the baseline's implementation**, not Blender/Cycles equivalence; e.g.
bump is a passthrough, several graph nodes are approximations, hair is
triangulated, Filmic is a curve approximation, and SSS is the existing walk.

Software-BVH limits are inherited: immutable geometry cache identities,
2,097,152 leaves per tree, 32-bit byte addresses, bounded traversal stacks and
rejection of singular transforms. Custom procedural geometry requires a
traversal adapter. The native backend does not add volumes, denoising, motion
blur or other features absent from the baseline. No promise of pixel-identical
results across compilers or traversal implementations is made.

## Regression and diagnostics

```sh
build/test/sparkium/sparkium_native_test
python3 scripts/check_native_backends.py \
  --cli build/demo/sparkium_cli/demo_sparkium_cli \
  --output out/native-validation --repeat --no-gpu-cpu
```

The script needs NumPy and Pillow. It renders six pinned basic scenes plus graph,
lens/film/SSS, hair/vertex-color and 70-light fixtures. Default settings are
64×64, 16 samples per frame, two frames, 12 bounces, with the same JSON for each
backend. There is no new seed parameter: a fresh process starts the baseline's
deterministic Sobol sample index at zero. `--repeat` checks both PNG and PFM bytes
for the first scene; `--no-gpu-cpu` hides Vulkan/CUDA devices for CPU runs.

Inputs, complete commands, logs, return codes, wall times, PNGs, pre-display linear
RGB PFMs (after the baseline path/film clamps), ×8 difference maps and numerical metrics are saved. PFM export rejects
non-finite radiance. The default PNG RMSE limits are 0.03 against Vulkan and 0.01
between CPU and CUDA; failures and timeouts are not treated as skips. These
thresholds are regression alarms, not an assertion that every low-sample
production scene will pass. Sparse fireflies, stochastic alpha/SSS, texture
interpolation and different intersection rounding can decorrelate paths. Compare
linear means and higher-sample renders too; do not loosen a threshold to hide a
failed comparison.

`--scenes blender_monster blender_classroom blender_junkshop` exercises the full
pinned Blender scenes. These are expensive to compile, especially Classroom:
there is currently no persistent shader cache or asynchronous CUDA submission.
`--backends cpu cuda vulkan_fallback vulkan_query` helps separate shader/compiler
differences from hardware traversal differences.

The C++ native tests cover descriptor-array ABI, buffer subranges, retained
bindings across program switches, independent module entries, partial/odd-size
images, HDR/float3 and sampler addressing, three-dimensional invocation IDs,
constant-buffer rebinding, accumulation/reset and explicit
unsupported-pipeline errors. CUDA cases skip when not built; a compiled CUDA
backend that cannot initialize its device fails the test. Existing software-BVH
oracle/alpha tests can also run using `SPARKIUM_TEST_BACKEND=cpu|cuda`, excluding
tests that explicitly require raster/graphics shader compilation. Do not modify
or weaken those original tests to run them on headless devices.

`sparkium_cpu_thread_pool_test` checks exact work coverage, persistent worker
reuse, concurrent submissions, exception recovery and single-thread execution.

Developer diagnostics:

- `SPARKIUM_NATIVE_TRACE=1`: reflected bindings and native dispatches.
- `SPARKIUM_NATIVE_DUMP=/path`: Slang intermediates, complete adapted CPU sources
  and generated CUDA source.
- The previous `SPARKIUM_NATIVE_ASAN` option is rejected: downstream C++ ASan
  instrumentation is not available in the direct LLVM JIT pipeline.
- CUDA `compute-sanitizer --tool memcheck` and Nsight Systems work on the native
  kernels. Host-only profile CSVs are available with `--profile-cpu-only`.

The direct LLVM path is validated on Windows with vcpkg Slang 2026.7.1#1 and
the official standalone Slang 2026.8 package. Earlier Linux
validation used the former C++ path; it does not establish direct-LLVM parity on
Linux or macOS.

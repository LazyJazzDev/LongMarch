# Native CPU and CUDA path tracing

Sparkium's headless path tracer can execute without a graphics API:

```sh
build/demo/sparkium_cli/demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend cpu --frames 2 -o cpu.png --linear-output cpu.pfm
build/demo/sparkium_cli/demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend cuda --frames 2 -o cuda.png --linear-output cuda.pfm
```

`auto`, `ray_tracing` (including legacy JSON requests), and `rt_fallback`
resolve to the shared compute path tracer on these devices. Here `rt_fallback`
is a **pipeline identifier**, not a Vulkan device: CPU dispatch calls compiled
host functions, and CUDA dispatch launches CUDA kernels. Neither creates a
Vulkan instance or submits graphics commands. An explicit unavailable backend
fails rather than silently selecting Vulkan. `--require-hardware-rt` is not
appropriate for these backends; it specifically requires graphics pipeline RT.

Vulkan, D3D12, Metal, their default selection, and their existing raster,
pipeline RT and inline-query implementations are unchanged. CPU/CUDA are
headless compute devices, **not new rasterizers or window-system backends**.
Native `rasterization`, `ray_query`, acceleration-structure commands, presentation
and CUDA/graphics interop buffers are explicitly unsupported. Use a graphics
backend for the GUI.

## Build

The optional native backend requires the Slang compiler **library and headers**,
plus its downstream C++ toolchain at runtime for CPU shaders. The Linux
implementation has been exercised with Slang 2026.8 and GCC 13. CUDA additionally
requires the CUDA driver and NVRTC (tested with Toolkit 13.2.1 on sm_86). NVRTC
selects the compute architecture of the actual selected device at runtime.
There is no OptiX dependency.

```sh
cmake -S . -B build -G Ninja -DVCPKG_PATH=/opt/vcpkg \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target demo_sparkium_cli sparkium_native_test \
  sparkium_fallback_test -j6
```

`LONGMARCH_ENABLE_NATIVE_RENDER` defaults to `ON` when Slang is discoverable.
The search considers the Vulkan SDK's `include/slang` and `lib` directories,
including `/opt/vulkan`. For another installation, set
`LONGMARCH_SLANG_INCLUDE_DIR` and `LONGMARCH_SLANG_LIBRARY` explicitly. If the
library or headers are absent, configuration reports native rendering disabled;
the existing graphics backends remain buildable. Set
`-DLONGMARCH_ENABLE_NATIVE_RENDER=OFF` to disable it explicitly.

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
initialize or use them. Slang/GCC must remain available during rendering.

## Shared implementation

`grassland/graphics/backend/native` implements the compute/resource subset of
`graphics::Core`. The existing Sparkium host scene code runs without backend
branches: JSON loading, mesh/hair conversion, shader-graph generation, material
registration, instance updates, descriptor construction, film and camera data
are unchanged.

The **same HLSL sources** supply:

- Morton keys, sorting, BLAS/TLAS construction and watertight software traversal;
- `RenderPixel`, camera/lens sampling, Sobol/Wang random generation and roulette;
- hit records, all material samplers, Principled BSDF evaluation/sampling;
- direct-light sampling, light power, MIS, alpha shadows and subsurface walks;
- accumulation, reset, clamping, persistence, film resolve and tone mapping.

CPU compiles these sources to a Slang host-callable C++ module. Workgroups are
dispatched on the host, optionally with OpenMP (at most six workers, or fewer
with `OMP_NUM_THREADS`). CUDA compiles Slang's CUDA source through NVRTC to PTX,
loads it with the CUDA driver, and calls `cuLaunchKernel`. BVH construction,
light preprocessing, path tracing, film resolve and tone mapping all execute
on the selected device. CUDA descriptors and pixels use device memory, not
managed CPU rendering followed by GPU copies. Submission is synchronous.

Only three parallel prefix-scan shaders need `SPARKIUM_NATIVE_CPU` variants:
CPU performs the per-workgroup scan serially because Slang host execution does
not implement GPU wave/barrier scheduling. Their inputs, power evaluators,
hierarchical scan and CDF consumers remain shared. Summation order can differ.

### Compiler and resource boundary

`native_shader.cpp` is a constrained adapter for the repository's DXC shader
dialect, **not a general HLSL translator**. It copies the shader VFS to a private,
RAII-managed temporary directory and mechanically lowers the known single-type
buffer templates, value-method mutability and resource declarations. It does
not translate or duplicate BSDF mathematics. Unknown templates/options fail
explicitly. Changes to the original buffer-helper template layout also fail
with an actionable diagnostic rather than silently rewriting different code.

Register spaces become reflected `NativeBinding` attributes. Native byte buffers
are pointer/size spans, descriptor arrays are explicit `NativeArray<T>` spans,
and constant buffers are pointers to the original uploaded structs. The global
parameter block is sized from reflected member offsets (its enclosing native
constant-buffer reflection size is only a pointer). CUDA binds
`SLANG_globalParams` in module constant memory. CPU module entry names are
unique to avoid ELF symbol interposition between separately compiled shaders.
ABI sizes, resource ownership types, missing bindings and host transfer ranges
are checked.

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
images, HDR/float3 and sampler addressing, accumulation/reset and explicit
unsupported-pipeline errors. CUDA cases skip when not built; a compiled CUDA
backend that cannot initialize its device fails the test. Existing software-BVH
oracle/alpha tests can also run using `SPARKIUM_TEST_BACKEND=cpu|cuda`, excluding
tests that explicitly require raster/graphics shader compilation. Do not modify
or weaken those original tests to run them on headless devices.

Developer diagnostics:

- `SPARKIUM_NATIVE_TRACE=1`: reflected bindings and native dispatches.
- `SPARKIUM_NATIVE_DUMP=/path`: Slang intermediates and generated CUDA source.
- `SPARKIUM_NATIVE_ASAN=1`: instrument downstream CPU shader compilation; preload
  the matching GCC `libasan.so` when the host executable is not ASan-instrumented.
- CUDA `compute-sanitizer --tool memcheck` and Nsight Systems work on the native
  kernels. Host-only profile CSVs are available with `--profile-cpu-only`.

Linux CPU/CUDA/Vulkan builds and execution are covered by the implementation
validation. D3D12/Metal runtime compatibility still requires their native OS;
this Linux validation is not evidence of a Windows/macOS runtime pass.

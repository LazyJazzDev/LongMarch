# Native Metal backend

On Apple Silicon, Sparkium can use a native Grassland Metal backend built with
Apple's **metal-cpp**. It supports compute pipelines, indexed rasterization,
multiple render targets, depth testing, blending, resource transfers, GLFW
presentation through CAMetalLayer, and ImGui. Metal is the default API when this
backend is enabled; `--backend vulkan` selects the existing Vulkan backend.

HLSL shaders go through DXC → SPIR-V → SPIRV-Cross → MSL → Metal's runtime
compiler. Metal shader compilation enables fast math by default, allowing floating-point
reassociation and approximate math; results need not match strict floating-point compilation.
GPU commands and resources use Metal directly, without MoltenVK.
Tier 2 argument buffers hold resource arrays, including scenes with more than
31 storage buffers. Shared-memory buffer uploads wait for outstanding GPU work
before writing, then copy directly into unified memory without a staging command
submission per upload. Texture transfers use private textures and staging blits.

`auto`, `ray_tracing`, and `rt_fallback` use Sparkium's compute BVH construction
and traversal on Metal. Native Metal acceleration structures / intersection
functions are not implemented in this backend; `--require-hardware-rt` rejects
Metal even on chips with hardware ray tracing. Geometry shaders and HDR window
presentation are not supported. Offscreen floating-point film buffers are supported.

## Source layout

`code/grassland/graphics/backend/metal` follows the other graphics backends:
`metal_backend.h` is the entry header; `metal_core`, `metal_buffer`, `metal_image`,
`metal_sampler`, `metal_shader`, `metal_program`, `metal_command_context`,
`metal_window`, and `metal_util` have separate headers and implementation files.
Window integration uses `.mm`; the other implementations use `.cpp`.
SPIRV-Cross translation lives in `metal_shader.cpp`, and the metal-cpp private
implementation macros are defined once in `metal_util.cpp`.

## Build and run

Requirements: Apple Silicon, macOS 13 or later (MSL 3.0), Xcode Command Line
Tools, CMake/Ninja, existing vcpkg dependencies, and Vulkan SDK's DXC and
SPIRV-Cross development libraries. The SDK supplies shader compilation tools;
the Vulkan runtime backend itself can be disabled.

CMake fetches the hash-pinned official metal-cpp macOS 15/iOS 18 archive.
For an offline build, set `LONGMARCH_METAL_CPP_DIR` to a local metal-cpp header
root containing `Metal/Metal.hpp` and `Foundation/Foundation.hpp`.

```sh
cmake -S . -B build-metal -G Ninja \
  -DVCPKG_PATH=/path/to/vcpkg -DCMAKE_BUILD_TYPE=Release \
  -DLONGMARCH_DISABLE_PYTHON=ON -DLONGMARCH_ENABLE_METAL=ON
cmake --build build-metal --target demo_sparkium_cli demo_sparkium_gui sparkium_fallback_test

build-metal/demo/sparkium_cli/demo_sparkium_cli assets/scenes/cornell_box/scene.json \
  --backend metal --pipeline rt_fallback -o cornell.png
build-metal/demo/sparkium_gui/demo_sparkium_gui assets/scenes --backend metal
```

Add `-DLONGMARCH_DISABLE_VULKAN=ON` for a Metal-only build, or
`-DLONGMARCH_ENABLE_METAL=OFF` to retain the previous macOS Vulkan default.
The CLI and GUI accept `--backend auto|metal|vulkan|d3d12`; unavailable backends
are rejected explicitly. GUI `--frames N` exits after N frames for smoke tests.

## Validation

```sh
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  build-metal/test/sparkium/sparkium_fallback_test

# Requires Pillow and a build with both APIs.
MTL_DEBUG_LAYER=1 python3 scripts/check_rt_fallback.py \
  --cli build-metal/demo/sparkium_cli/demo_sparkium_cli \
  --backend metal --compare-backend vulkan --pipeline rasterization \
  --debug --max-rmse 0.001 --output out/metal-raster

MTL_DEBUG_LAYER=1 python3 scripts/check_rt_fallback.py \
  --cli build-metal/demo/sparkium_cli/demo_sparkium_cli \
  --backend metal --compare-backend vulkan --spp 256 \
  --debug --max-rmse 0.03 --output out/metal-compute
```

The checks cover GPU BVH construction/traversal against a CPU oracle,
transparent shadows, accumulation/reset, odd image sizes, partial texture
transfers, buffer copies/resizing, completion callbacks, 80-resource argument
arrays, binding snapshots, and light-selection workgroup tail bounds. An
attachmentless raster pass checks inherited viewport/scissor state. Hardware RT
parity explicitly skips on this device.

The commands above compare the six basic scenes at 96×96. Raster output must
pass a normalized RGB RMSE threshold of 0.001; compute output at 256 spp and
12 bounces must pass 0.03. Floating-point differences can change Monte Carlo
paths; these image comparisons do not establish bitwise parity.

After separating the branch from Blender extensions and rebasing onto the
merged JSON loader, the CLI/GUI/nbody/test targets build on Apple M5. With Metal
API and shader validation, 12 tests pass and hardware RT parity skips. All six
raster comparisons are identical; the maximum compute RMSE is 0.01120 (point
light), below 0.03. Results are in `out/stack-rt/{raster,compute}` when reproduced.

The port exposed an existing light-selection shader bounds bug: inactive lanes
in the final 64-thread workgroup read metadata and wrote beyond the selector
buffer. These reads/writes are now guarded. A sentinel regression test covers
1, 5, 63, and 65 lights.

## Diagnostics and current limits

For the nbody compute demo, see the [Metal/Vulkan performance comparison](nbody-metal-vulkan-performance.md)
and `scripts/profile_nbody.py`. The demo supports synchronized frame benchmarks,
per-frame CPU/GPU timing CSVs, and deterministic particle-state readback.

Set `LONGMARCH_METAL_SHADER_DUMP=/path/to/directory` to save generated MSL and
an HLSL virtual-filesystem snapshot for debugging compiler failures. Native GPU
command errors are checked when waiting for completion.

Frame profiling supports `--profile timings.csv --profile-cpu-only` on Metal.
Per-stage GPU timestamps still require Vulkan; Metal timings are not inferred
from Vulkan measurements. Use Xcode GPU tools for native GPU captures.

Blender scene import, material graphs, and hair are added by the dependent
`blender-align` branch. Their validation and scene-specific limits belong there.

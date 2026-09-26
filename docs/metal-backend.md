# Native Metal backend

On Apple Silicon, Sparkium can use a native Grassland Metal backend built with
Apple's **metal-cpp**. It supports compute pipelines, indexed rasterization,
multiple render targets, depth testing, blending, resource transfers, GLFW
presentation through CAMetalLayer, and ImGui. Metal is the default API when this
backend is enabled; `--backend vulkan` selects the existing Vulkan backend.

Slang shaders go through Slang → SPIR-V → SPIRV-Cross → MSL → Metal's runtime
compiler. Metal shader compilation enables fast math by default, allowing floating-point
reassociation and approximate math; results need not match strict floating-point compilation.
GPU commands and resources use Metal directly, without MoltenVK.
Tier 2 argument buffers hold resource arrays, including scenes with more than
31 storage buffers. Shared-memory buffer uploads wait for outstanding GPU work
before writing, then copy directly into unified memory without a staging command
submission per upload. Texture transfers use private textures and staging blits.

`auto` selects native ray query on supported Metal devices, otherwise software
fallback. Explicit `ray_tracing` and `rt_fallback` use Sparkium's compute BVH
construction and traversal on Metal. The `ray_query` pipeline uses native Metal
triangle acceleration structures and inline queries; see [Metal ray query](metal-ray-query.md)
for implementation, validation, and performance results. Pipeline RT/SBT and custom
intersection functions remain unsupported. `--require-hardware-rt` checks pipeline RT
and still rejects Metal, including devices with hardware ray tracing; omit it for
`ray_query`. Geometry shaders are not supported. Offscreen floating-point film
buffers and HDR window presentation through macOS EDR are supported.

## Source layout

`code/grassland/graphics/backend/metal` follows the other graphics backends:
`metal_backend.h` is the entry header; `metal_core`, `metal_buffer`, `metal_image`,
`metal_sampler`, `metal_shader`, `metal_program`, `metal_acceleration_structure`, `metal_command_context`,
`metal_window`, and `metal_util` have separate headers and implementation files.
Window integration uses `.mm`; the other implementations use `.cpp`.
SPIRV-Cross translation lives in `metal_shader.cpp`, and the metal-cpp private
implementation macros are defined once in `metal_util.cpp`.

## Build and run

Requirements: Apple Silicon and MSL 3.0 (Metal backend: macOS 13+; the
prebuilt Slang SDK may require a newer macOS), Xcode Command Line Tools,
CMake/Ninja, vcpkg dependencies, Slang 2026.18.1+ and SPIRV-Cross development
libraries. **Vulkan SDK is not required for Metal.** See [Slang setup](slang-shaders.md).

The default vcpkg `metal` feature supplies Apple metal-cpp macOS 15/iOS 18
headers and the upstream `spirv-cross` package. CMake resolves the exported
`spirv_cross_core`, `spirv_cross_glsl` and `spirv_cross_msl` packages and links
their imported targets; it does not derive include/library paths from Vulkan.
CMake does not download dependencies. To use external dependencies, set
`LONGMARCH_METAL_CPP_DIR` to the header root containing `Metal/Metal.hpp`, and
`CMAKE_PREFIX_PATH` to an installed SPIRV-Cross prefix exporting those packages.
Then disable the manifest `metal` feature if desired (see [Slang setup](slang-shaders.md)).

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
a Slang virtual-filesystem snapshot for debugging compiler failures. Native GPU
command errors are checked when waiting for completion.

Frame profiling supports `--profile timings.csv --profile-cpu-only` on Metal.
Per-stage GPU timestamps still require Vulkan; Metal timings are not inferred
from Vulkan measurements. Use Xcode GPU tools for native GPU captures.

Blender scene import, material graphs, and hair are added by the dependent
`blender-align` branch. Their validation and scene-specific limits belong there.

## HDR / EDR presentation

`Window::SetHDR(true)` switches the Metal layer and presentation pipeline to
`RGBA16Float`, tags the layer as extended linear sRGB, and enables
`wantsExtendedDynamicRangeContent`. HDR source images must contain linear-light
sRGB values: 1.0 is SDR reference white, and values above 1.0 request extended
brightness. No application-side clamp or tone mapping is applied. Switching HDR
off restores the existing 8-bit SDR path (`BGRA8Unorm`, sRGB). Pending GPU work
is completed before the layer format changes; ImGui uses the current attachment
format in both modes.

```sh
cmake --build cmake-build-metal-only --target demo_graphics_hello sparkium_fallback_test
cmake-build-metal-only/demo/graphics_hello/demo_graphics_hello --module hdr --backend metal
LONGMARCH_TEST_METAL_WINDOWS=1 MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  cmake-build-metal-only/test/sparkium/sparkium_fallback_test --gtest_filter='MetalWindowTest.*:MetalBackendTest.WindowCloseAfterPresent'
```

The demo has an HDR gradient and an SDR-white reference bar. Press **H** to
compare HDR and SDR output. Logs report the current screen's EDR headroom and
potential headroom at the time HDR is enabled. Current headroom can initially
be 1.0 and increase after EDR content starts presenting. Physical brightness
also depends on the screen, its brightness setting, and macOS power/thermal
management. An SDR-only screen cannot show highlights above its white level;
ordinary SDR screenshots cannot establish physical HDR brightness.

## PR #61 reference-white validation on macOS (2026-09-26)

Validated `fix/sparkium-hdr-auto` based on `84d8fed` on Apple M5 with the built-in
3024 × 1964 Liquid Retina XDR display. Ninja / Release builds of Sparkium GUI,
Graphics Hello and `sparkium_fallback_test` passed. An existing external metal-cpp
installation was selected explicitly; CMake did not download that SDK.

With Metal API and shader validation enabled, four selected tests passed:
`MetalWindowTest.HDRPresentationAndImGuiSwitching`,
`MetalBackendTest.WindowCloseAfterPresent`,
`SoftwareBVHTest.HDRFilmDevelopmentPreservesHighlightsAndAccumulation`, and
`HDRSurfaceFormatTest.PreferredFormatWinsRegardlessOfEnumerationOrder`.
`HDRBrightnessTest.RefreshNotifiesAndUnknownReferenceFallsBack` passed separately.

The Metal window regression now also checks the PR's display-brightness API:
reference white is known, native EDR reference-white scale remains 1 with automatic
alignment both on and off, absolute SDR nits remain unknown, and reported headroom
and HDR capability match the window's actual `NSScreen`. Repeated HDR/SDR switching,
with and without ImGui, verifies RGBA16Float / BGRA8Unorm, extended linear sRGB /
sRGB, and the layer's extended-dynamic-range flag.

The live HDR gradient demo and Sparkium GUI both ran with Metal validation. The
GUI used a temporary Cornell scene: 768 × 768, Auto pipeline, 1 sample/frame,
8 bounces, max exposure 100. The committed scene was not modified. GUI logs show
repeated HDR/SDR transitions and an active screen headroom of **12.34×**, with
potential headroom **16×**; initial headroom was 1× before EDR activation settled.
No Metal validation errors were found in these test/application logs.

This establishes the Metal HDR configuration, display query and rendering path;
it is not a physical peak-luminance measurement or proof from an SDR screenshot.
Automated visual interaction through Computer Use remained unavailable in this
session (`CUA_REPL_ENABLED_SURFACES is required`), including after a retry.

Local build and logs: `out/pr61-metal/build`, `/tmp/pr61-metal-tests.log`,
`/tmp/pr61-brightness-test.log`, `/tmp/pr61-gradient.log`, `/tmp/pr61-gui.log`.

### HDR artistic grading

HDR film development now retains exposure, artistic gamma and contrast. The
legacy Filmic approximation matches the SDR curve up to scene-linear input 1,
then continues along its tangent (matching value and first derivative) rather
than clipping the highlights. Both paths share the Filmic grading operation;
HDR omits its upper clamp and decodes the graded look back to linear sRGB for
presentation. This is an extension of Sparkium's existing approximation, not
Blender's full OCIO Filmic transform or a display-headroom-adaptive tone mapper.
Standard and Normalized HDR previews also allow gamma/contrast; Normalized does
not perform the SDR brightness normalization. SDR behavior remains unchanged.

The GUI exposes these controls under View settings in HDR mode. GPU readback
coverage checks Monster's gamma 1.15 / contrast 1.2, each control independently,
SDR/HDR Filmic midtone agreement, continuity at the extension point, extended
highlights, finite output under extreme grading, alpha and unchanged accumulation.
Physical display appearance still requires visual review; these checks do not
measure display luminance or establish an exact match to Blender.

### Metal dependencies without Vulkan SDK discovery

Metal now resolves SPIRV-Cross through its exported CMake packages, supplied by
vcpkg's macOS `metal` feature or an external `CMAKE_PREFIX_PATH`. Neither
`Vulkan_LIBRARY`, `Vulkan_INCLUDE_DIRS` nor `VULKAN_SDK` is used for Metal
dependency discovery. Disabling the Vulkan backend also skips Vulkan discovery.
Shared-shader DXIL tests run only when the D3D12 backend is built; a Metal-only
build does not require a downstream DXC installation merely to run its tests.

Validated on Apple M5/macOS 26.6.2 with upstream Slang 2026.18.2 and vcpkg
SPIRV-Cross 1.4.341.0. A fresh build with Vulkan explicitly disabled succeeded.
Configuration/build also succeeded with `VULKAN_SDK` removed from the environment,
`LONGMARCH_DISABLE_VULKAN=OFF` and `CMAKE_DISABLE_FIND_PACKAGE_Vulkan=ON`, exercising
normal optional-backend selection when Vulkan discovery is unavailable.
The installed Vulkan SDK was not physically removed from this machine.

Sparkium CLI/GUI and regression targets built. Metal regression passed **25**
tests with **6 conditional skips** (desktop opt-ins / standalone RT parity).
A native-query Monster smoke render completed at 96×96, 2 spp, 4 bounces.
SPIRV-Cross package paths resolve inside the vcpkg install tree; the generated
build contains no Vulkan SDK path and `otool -L` shows no Vulkan library in the
GUI executable. This verifies dependency separation, not GUI visual quality.

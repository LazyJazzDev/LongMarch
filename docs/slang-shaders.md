# Slang shader compilation

LongMarch compiles its shaders with the Slang compiler API. Project-owned shader
sources and includes use `.slang`, including the external ray tracing examples in
the assets submodule. Existing shader-model strings such as `vs_6_0`, `cs_6_5` and
`lib_6_5` remain the public API's stage/profile selection convention.

## Dependency

As checked on 2026-09-26:

| Distribution | Version |
| --- | --- |
| Local `/usr/local/bin/slangc` | `2026.1-52-gc8ddf20bb` |
| Local vcpkg checkout (`shader-slang`) | `2026.2` |
| Upstream vcpkg `shader-slang` | `2026.18.2` |
| Official Slang release | `2026.18.3` |

The default `slang` manifest feature installs **2026.18.3** through vcpkg.
`vcpkg-configuration.json` selects the project overlay port, so an older vcpkg
checkout does not silently substitute its registry version. The port packages
the official SDK with pinned SHA-512 hashes for Windows, Linux and macOS,
each on ARM64 and x86-64. This is the same compiler release used in the
[NonUniform comparison](reports/nonuniform-compiler-comparison.md).
The overlay can be retired once the registry provides the required release.

`cmake/Slang.cmake` only finds an installed package (2026.18.3 or newer).
It never downloads Slang and fails with setup instructions if none is found.
vcpkg manages downloads, installation and binary caching; for offline builds,
populate the vcpkg caches/install tree beforehand. The system `slangc` and the
Vulkan SDK's compiler are not replaced.

To supply an external SDK, disable the default features and explicitly keep
any vcpkg features you still need. For example, on macOS with vcpkg metal-cpp:

```sh
cmake -S . -B out/external-slang -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVCPKG_PATH=/path/to/vcpkg -DVCPKG_MANIFEST_NO_DEFAULT_FEATURES=ON \
  -DVCPKG_MANIFEST_FEATURES=metal -Dslang_DIR=/sdk/lib/cmake/slang
```

On Windows the official package directory is `/sdk/cmake`. `CMAKE_PREFIX_PATH`
can also select an SDK. Use a fresh build directory when changing providers,
or clear the cached `slang_DIR` and `SLANGC_EXECUTABLE` entries. The chosen package
path is logged at configure time. To provide metal-cpp externally as well,
omit the `metal` feature and set `LONGMARCH_METAL_CPP_DIR` to its header root.
CMake does not download metal-cpp either.

The official 2026.18.3 macOS ARM64 compiler dylib declares macOS **26.0** as
its minimum OS (verified from LC_BUILD_VERSION). Older macOS hosts need a Slang
SDK built for their deployment target; the Metal backend's own macOS 13 minimum
does not override the compiler library requirement. These prebuilt SDKs cover
desktop hosts, not native iOS/HarmonyOS compiler libraries; mobile offline shader
packaging needs separate integration on its development branch.

## Targets and compatibility

- Vulkan: Slang directly emits SPIR-V 1.5; the runtime no longer applies the
  DXC-specific `RestoreStorageBufferNonUniform` repair to shader modules.
  The obsolete repair header and its DXC regression test have been removed.
- Metal: the same Slang SPIR-V passes through the existing SPIRV-Cross argument
  buffer mapping and Apple's MSL compiler. Native Metal RT paths remain intact.
- D3D12: Slang emits DXIL. Slang may invoke downstream DXC for this target;
  removing direct DXC API integration does not remove that downstream tool.
  The official Windows Slang archive does not bundle DXC (verified from the
  2026.18.3 x86-64 archive). The Windows-only vcpkg `directx-dxc` dependency
  supplies dxcompiler.dll and dxil.dll; deployment copies these next to Slang.
  LongMarch does not link the DXC API.

Column-major matrix layout, DirectX-compatible buffer layout, entry point names,
and register/space bindings are explicitly selected. Per-thread Slang global
sessions cache the standard library; each compile has an independent session,
virtual file system and generated material source. Include guards are handled by
Slang rather than suppressing every second include at the file-system layer.

HLSL template helpers were ported to constrained Slang generics and interfaces.
Methods that change stream offsets, geometry transforms or BSDF closure state
are explicitly marked `[mutating]`. Truncating vector conversions are explicit.
NonUniformResourceIndex must still be placed at the actual descriptor access;
Slang 2026.18.3 does not fix every caller-to-callee integer-index propagation case.
See the [full compiler comparison](reports/nonuniform-compiler-comparison.md).

The mobile development branch has its own packaging/cache integration. This
main-based migration does not import that branch, its dispatch tiling, or its
signing configuration. Mobile shader caches must be regenerated when integrating
this compiler change there.

## Validation

Ninja / Release on Apple Silicon:

- Built the graphics compatibility tests, compiler corpus test, Sparkium tests,
  Sparkium CLI/GUI, 2048, GoL, graphics_hello and nbody_cs.
- Vulkan/MoltenVK Sparkium run: 19 passed, 6 skipped because native hardware RT
  was unavailable. Metal run: 24 passed, 1 hardware RT pipeline comparison skipped.
  These runs exclude interactive window tests.
- The 31 JSON scene/raster checks (including invalid-input rejection) passed.
- Shared renderer entries compile to SPIR-V and DXIL, including generated
  material hit groups, ray-generation/miss/callable entries, and BVH kernels.
- Standalone demo, Snowberg, Python-example and external asset shader entries
  compile through the API; malformed input returns no bytecode.
- Seven scene checks (six basic scenes plus generated Blender-material graph
  smoke): Vulkan and Metal software tracing at 48x48, 2 spp, 4 bounces all
  succeeded. The paired PNG RGB outputs were identical at these settings.
  This is a small deterministic regression, not a full-resolution quality or
  performance benchmark.
- The vcpkg overlay installs the official SDK; package discovery and runtime
  loading are validated on macOS ARM64. Its compiler dylib is byte-for-byte
  identical to the previously tested official SDK. External SDK discovery and
  the missing-package failure path were also checked. The system Slang is unchanged.

Windows x64 / MSVC 19.44 / Ninja Release was also validated on an RTX 3090 Ti
(driver 596.49), using the vcpkg Slang 2026.18.3 package:

- The nine targets listed above build with Python enabled, including the graphics
  tests that link Grassland directly. Python linkage propagates with its headers.
- D3D12 and Vulkan each pass all 18 Sparkium regression tests, including hardware
  image parity. Both runs enable backend debugging; Vulkan also enables
  synchronization validation with `VK_LAYER_VALIDATE_SYNC=1`.
- The default D3D12 CLI passes all 31 JSON scene/raster and invalid-input checks
  at the script's 96x96 resolution.
- All three Slang tests and both Vulkan compatibility tests pass. The native
  backend compiler test requires a GPU and checks valid and invalid shader input
  through the D3D12/Vulkan API on each compiled backend.
- D3D12 no longer forwards DXC-only warning/debug options to Slang and rejects
  empty compilation results. Vulkan queries and enables supported
  `shaderDrawParameters`, required by Slang's vertex/instance ID lowering.

### Windows validation after rebasing onto main

Rebased onto `12c859d` (including the HDR presentation fixes) and repeated
validation on the Windows / RTX 3090 Ti configuration above:

- Built all 20 current demo targets and `test/all`, plus the explicitly excluded
  Slang compiler, Vulkan compatibility and window-input tests, with Ninja Release.
- All 47 automated demo runs exited successfully: all 11 graphics_hello modules,
  2048, GoL, Sparkium CLI and Sparkium GUI SDR/HDR on both D3D12 and Vulkan;
  NBody CS compute/offscreen/window on both backends; eight console/CUDA demos;
  and CUDA NBody headless mode.
- Opened ImGui, DrawNGUI, joystick_test, Practium, Franka and CUDA NBody windows,
  inspected their rendered content and closed them normally (exit code zero).
  No joystick was connected, so physical controller input was not checked.
- Slang compiler tests (3), Vulkan compatibility tests (2), window-input tests (2)
  and GoL tests (31) passed.
- Sparkium regression runs with each of `SPARKIUM_TEST_BACKEND=d3d12` and
  `vulkan` passed 32 tests and skipped two unsupported-HDR cases because this
  desktop supports HDR. Backend debugging, Vulkan synchronization validation
  and HDR window tests were enabled. No Vulkan validation errors were reported.
- External ray tracing shaders use assets commit `f5d2bcd`, rebased onto the
  assets main branch; both native backends ran the external_shader demo.

These are short smoke runs (generally six frames; CLI two frames), not exhaustive
interaction, long-running stability or calibrated HDR luminance measurements.
Slang still emits existing implicit-conversion and possible-uninitialized-variable
warnings. Linux deployment and Metal were not rerun during this Windows rebase
validation; the Apple Silicon results above describe the earlier revision.

```sh
cmake -S . -B out/slang-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DVCPKG_PATH=/path/to/vcpkg -DLONGMARCH_DISABLE_PYTHON=ON
cmake --build out/slang-build --target test_slang_compiler \
  test_vulkan_compatibility sparkium_fallback_test demo_sparkium_cli -j 6
out/slang-build/test/graphics/test_slang_compiler
VK_LAYER_VALIDATE_SYNC=1 out/slang-build/test/graphics/test_vulkan_compatibility
SPARKIUM_TEST_BACKEND=metal out/slang-build/test/sparkium/sparkium_fallback_test \
  --gtest_filter='-MetalBackendTest.WindowCloseAfterPresent:MetalWindow*'
```

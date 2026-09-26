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

`cmake/Slang.cmake` first looks for a compatible Slang CMake package, version
2026.18.3 or newer. This includes an appropriately updated vcpkg package. When
none exists, CMake downloads the official **2026.18.3** SDK for the host platform,
with a pinned SHA-256 for each supported macOS/Linux/Windows ARM64/x86-64 archive.
It does not replace `/usr/local/bin/slangc` or select the Vulkan SDK's compiler.

The official 2026.18.3 macOS ARM64 compiler dylib declares macOS **26.0** as
its minimum OS (verified from LC_BUILD_VERSION). Older macOS hosts need a Slang
SDK built for their deployment target; the Metal backend's own macOS 13 minimum
does not override the compiler library requirement.

An existing SDK can be selected with `-Dslang_DIR=/sdk/lib/cmake/slang`.
The SDK choice is logged at configure time. vcpkg's port also packages official
prebuilt releases; it is not a separate compiler implementation. Its current
release lag is why an unconditional dependency on the registry package is not
used in the manifest yet.

## Targets and compatibility

- Vulkan: Slang directly emits SPIR-V 1.5; the runtime no longer applies the
  DXC-specific `RestoreStorageBufferNonUniform` repair to shader modules.
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
- Official SDK download, hash verification, CMake import and runtime loading
  were exercised; the system Slang installation was not changed.

D3D12 execution and Windows/Linux SDK deployment have not been exercised on this
macOS machine. DXIL compilation alone is not a D3D12 rendering test.

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

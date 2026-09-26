# Slang mobile integration — 2026-09-26

Integrated the desktop `feat/slang-shaders` compiler migration into the mobile
code shared by HarmonyOS and iOS. The work is based on `harmonyos-app` and retains
its HDR presentation, game rendering fixes, and explicit-origin compute tiling.

## Changes

- Host preparation uses the same vcpkg-packaged Slang **2026.18.3** as desktop.
  Device and replay builds have no Slang/DXC/SPIRV-Cross runtime dependency.
- Mobile source embedding, demo shader paths and bundle staging now use `.slang`.
  SPIR-V cache keys use a versioned `slang-` namespace; old DXC bundles are rejected.
  Removed the obsolete mobile DXC NonUniform repair regression test.
- Enabled SPIRV-Cross's iOS BaseVertex/BaseInstance support, required for Slang's
  vertex/instance ID semantics. iOS 18 supports these attributes. The MSL cache
  namespace was bumped to invalidate previously generated stages.
- Both platforms use a shared vcpkg host manifest. CMake does not download Slang,
  metal-cpp, FreeType or MikkTSpace. Portable sources are compiled by each target
  SDK, never linked from host binaries. See [setup](README.md).

## Verified

Apple Silicon, Ninja / Release:

- iOS ARM64 device and simulator apps built. Final simulator app installed and
  launched. No physical iPhone validation was performed in this update.
- Nine scenes prepared and replayed without a compiler: all replay PNGs exactly
  matched preparation at 128-pixel maximum edge, 2 spp. Cornell also passed at
  256 pixels / 32 spp. Missing and corrupt cache rejection passed.
- All 11 mobile demos passed preparation and compiler-free replay. GoL and 2048
  shared desktop renderer/input/animation/focus/orientation checks passed.
- All 46 generated MSL stages compiled for `air64-apple-ios18.0`, Metal 3.0.
  Final packaged MSL sources were checked against those compiled files.
- Nine forced software-tracing scenes passed on the Metal preparation host at
  64-pixel maximum edge / 1 spp, including all three Blender scenes.
- HarmonyOS ARM64 native library and signed HAP built with the installed DevEco
  SDK. HAP contents verified against all 748 manifest entries, including 60 Slang
  SPIR-V caches; no MSL or legacy HLSL caches are shipped.
- Final signed HAP installed and launched on the connected PLU-AL10. The running
  2048 page displayed rendered tiles and advancing frame statistics. Independent
  HDC scene-check execution was denied by the device even with executable file
  permissions, so the nine-scene host result is not a phone rendering claim.
- Native dependency inspection: no Slang, DXC or SPIRV-Cross library in either app.
- Five HarmonyOS resource/hash tests passed, including rejection of a legacy DXC
  bundle. Repository pre-commit checks passed.

These are compiler migration and small-scene regression checks, not full-resolution
image-quality or performance measurements. Existing mobile ray-tracing capability
and driver limitations are unchanged.

## Local outputs

- iOS: `out/mobile/slang/ios-device/LongMarch.app` (unsigned device bundle).
- Simulator: `out/mobile/slang/ios-simulator/LongMarch.app`.
- HarmonyOS: `platforms/harmonyos/entry/build/default/outputs/default/entry-default-signed.hap`.
- Canonical iOS resources: `out/ios/Resources`; previous contents retained under
  `out/mobile/slang/Resources-default-before-slang`.
- HarmonyOS rawfile resources were refreshed; the previous bundle is retained
  under `out/mobile/slang/HarmonyResources-before-slang`.

Generated outputs and local signing settings are not committed.

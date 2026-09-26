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
  launched. Physical iPhone validation is recorded in the follow-up below.
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

## Physical iPhone follow-up — 2026-09-26

Tested commit `eb61816` on iPhone 16 Plus, Apple A18 GPU, iOS 26.6.2.
The Release device app was signed with the existing local development profile,
verified with `codesign --verify --strict`, installed, and tested through its
existing smoke-run entry points. No renderer changes were needed.

All **20 cases passed**: 11 demos and 9 Sparkium scenes. Before each launch the
previous result was replaced by a unique pending marker, so results cannot be
mistaken for a previous run. Each demo reported completed GPU frames; each scene
reported 2 spp, an empty error string, and a saved image of the expected size.

Demos: 2048, GoL, Triangle, Texture, Blend, Cube, Resize, SDR Sample, HDR,
Ray Query, and NBody. 2048/GoL reached their settled idle state. HDR reported
`edr_enabled=true` with RGBA16Float drawable format; this verifies the EDR path,
not a measurement of physical display brightness.

| Sparkium scene | Device render resolution | Samples | Result |
| --- | --- | --- | --- |
| cornell_box | 1024 × 1024 | 2 spp | Passed |
| texture | 2048 × 1024 | 2 spp | Passed |
| area_light | 1024 × 1024 | 2 spp | Passed |
| point_light | 1024 × 1024 | 2 spp | Passed |
| principled | 1024 × 1024 | 2 spp | Passed |
| specular | 1024 × 1024 | 2 spp | Passed |
| blender_classroom | 1920 × 1080 | 2 spp | Passed |
| blender_junkshop | 2000 × 1000 | 2 spp | Passed |
| blender_monster | 1024 × 1024 | 2 spp | Passed |

All nine saved images were checked for dimensions and nonuniform RGB output.
Cornell Box, Texture, and the three Blender images were visually inspected:
geometry/material content is visible, with the expected substantial noise at
2 spp. This is a functional smoke check, not a converged image comparison,
interactive gesture suite, prolonged memory test, or sustained performance test.
No missing-cache errors, renderer-reported failures, or app termination occurred
during these cases.

The signed device app is `out/mobile/slang/iphone/LongMarch.app`. Raw results,
rendered images, device screenshots, and the local runner are under
`out/mobile/slang/iphone/` (including `report.json`). These generated diagnostics
remain outside Git. The app is relaunched without smoke environment variables
after testing, returning to its normal demo browser.

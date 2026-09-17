# Initial iOS port validation

Validated on an Apple M5 Mac with Xcode 27.0 (27A266a), iOS 27 SDK and an
`arm64-apple-ios18.0` deployment target.

- Release iPhone and arm64 simulator applications build successfully.
- The application installs and launches in the iPhone 18 Pro / iOS 27 simulator.
  Native SwiftUI scene and render-setting controls were visually inspected.
- This simulator fails the Metal capability check before rendering. The app reports
  an unsupported-simulator error. No physical iPhone/iPad was attached; actual iOS
  GPU execution, device memory limits and performance remain unverified.
- All 24 cached MSL stages compile with the Apple Metal compiler for
  `air64-apple-ios18.0`, Metal 3.0 and fast math.
- The macOS runtime built with `LONGMARCH_OFFLINE_SHADERS` renders all nine bundled
  scenes at a longest edge of 128 pixels and 2 spp. Every output PNG is byte-identical
  to the corresponding preparation output, including Blender Classroom, Junkshop
  and Monster. The replay executable links neither DXC nor SPIRV-Cross.
- Cornell Box also renders at 256 pixels / 32 spp using the same cache.
- Missing and intentionally corrupted shader caches fail with explicit errors.
- Existing scene files and referenced resources remain unchanged; the prepared
  bundle includes LFS data, the Sobol table and hashes of its asset files. The
  resource bundle is approximately 667 MiB.

Local artifacts are generated under `out/ios/`: preparation previews,
`validation/report.json`, `metal-validation/`, and simulator screenshots. Reproduce
these checks using the commands in [README.md](README.md).

## Full-screen viewport and floating controls (previous behavior)

The subsequent UI update was built for the simulator and signed for the connected
iPhone 16 Plus (Apple A18 GPU), using the existing local signing team.

- Landscape-only app fills the screen, including the viewport behind safe areas.
  The control panel overlays the image; its corner button remains when collapsed.
- Resolution comes from the viewport size multiplied by display scale. The physical
  device produced a successful Texture-scene result at 2796 × 1290 pixels, 2 spp,
  exactly matching the captured screen dimensions. The full-screen image and
  collapsed control button were visually inspected on the device screenshot.
- A successful Cornell Box device render and an expanded panel over the rendered
  image were also inspected during layout development.
- Mac replay at viewport aspect 2.17 produced 256 × 118 pixels. Without an aspect
  override, the original 128 × 128 Cornell Box PNG remains byte-identical to the
  preparation reference.
- Local artifacts: `out/ios/native-phone-result.json`, `native-phone-screen.png`,
  `landscape-phone-screen.png`, and `landscape-sidebar.png`.

This verifies the viewport and native-resolution path on this device, not the
memory requirements or performance of every scene at native resolution.

## Scene presets, automatic accumulation and image navigation

This update supersedes the viewport-derived resolution above. The app now renders
with the original JSON film dimensions and camera aspect ratio, then fits the image
to the screen. Asset JSON files and the asset submodule reference are unchanged.

- Debug device and Release simulator builds pass with the existing signing setup.
- The installed app on iPhone 16 Plus automatically renders Cornell Box at its
  original 1024 × 1024 resolution. The 2-spp smoke report records 0.215 s, Apple A18
  GPU and no error. A subsequent normal launch automatically reaches the default
  32-spp cap in about 2.4 s, with 14.76 M camera Ray/s and 14.1 FPS for its last frame.
  These timings cover rendering and readback, excluding scene loading and pauses.
- The normal-launch screenshot confirms the square image fits the landscape screen
  without changing aspect, with the floating sidebar showing the scene picker,
  sample cap, Ray/s, FPS, accumulated spp, time and device. Further statistics remain
  accessible by scrolling. There is no Start/Render button.
- The actual Objective-C++ render controller passes its macOS GPU integration check:
  original Cornell Box and Texture dimensions, increasing/decreasing the cap without
  reloading or losing accumulation, pause/resume, two rapid film resets reproducing
  the first sample exactly, and superseded scene requests suppressing old callbacks.
- Native UIScrollView implements pinch zoom, pan and double-tap to fit. Progressive
  pixel updates retain the same viewer and transform. Physical gesture behavior has
  not been exercised automatically; this check does not establish successful loading
  of every large Blender scene on the phone.
- Python syntax checks and `git diff --check` pass.

Reproduce the controller check using [README.md](README.md). Local evidence:
`out/ios/preset-scene-phone-result.json` and `out/ios/preset-scene-phone-fit.png`.

## LongMarch Demos browser

Integrated `main` at `ee01050`, including stable entity registration and the
nonblocking-emitter shadow fix. Refreshed all nine Sparkium scene shader variants
through the preparation tool; all nine rendered successfully at 128 pixels / 2 spp.

- The installed display name is LongMarch Demos, using the existing Cornell Box
  icon, bundle identifier and signing configuration. The entry list groups
  Sparkium, NBody CS and graphics demos. Sparkium's nine-scene picker is inside
  the Sparkium demo, with automatic rendering and its existing spp controls.
- Debug iPhone and Release simulator builds pass. The simulator entry list and
  physical-device list, triangle, Resize controls and NBody view were inspected.
- All six graphics/compute adapters run with the preparation build and the
  offline replay build. Their PNG outputs match byte for byte. Checks cover
  finite pixels, NBody motion, finite particle coordinates, pause, deterministic
  reset, and Resize at 900 × 600. Sparkium's render-controller regression check
  also passes after introducing the shared serial render queue.
- Each of the six new demos was independently launched on iPhone 16 Plus
  (Apple A18 GPU) and successfully presented at least three frames. Triangle,
  Texture, Blend and SDR use 1280 × 720; Resize and NBody use 2796 × 1290.
  The NBody smoke frame with 4096 particles took 2.41 ms GPU time and 5.87 ms
  render/presentation time. This is a smoke sample, not a sustained benchmark.
- Sparkium also passes a fresh physical-device smoke run after the main update:
  Cornell Box at its original 1024 × 1024 resolution, 2 spp, 0.202 s render time,
  with no error. Its scene picker and Demos return button were visually inspected.
- Graphics/compute frames are presented from GPU textures through MTKView without
  CPU image readback. The sidebar reports actual completed-frame FPS separately
  from render time and GPU time; the display is capped at 60 Hz.
- Full RT Pipeline demos are explicitly unavailable on the current Metal backend;
  the list explains the missing pipeline/procedural-intersection support.
  Sparkium continues to use hardware Ray Query.

Local device reports and screenshots are under `out/ios/device-demos/`; Mac
outputs are in `out/ios/demos-prepare/` and `out/ios/demos-replay/`. These tests
do not cover every finger gesture or eliminate the known large-Blender-scene
memory limitation.

## LongMarch project and application identity

The Xcode project, application target, executable and app bundle are now named
`LongMarch`. Automatic signing uses `dev.lazyjazz.longmarch` with the existing
development team, while the display name remains LongMarch Demos.

- Debug iPhone and Release simulator builds pass after project regeneration.
- `codesign --verify --strict` passes; the signed application identifier contains
  the new `dev.lazyjazz.longmarch` bundle ID under the existing team.
- The new automatically provisioned app installs and launches successfully on
  the connected iPhone 16 Plus.
- This bundle ID installs independently of the previous Sparkium app, preserving
  the old application's data.

## Mobile texture packaging (2026-09-17)

The packager now limits PNG/JPEG texture dimensions to a 1024-pixel longest edge
by default. Original source assets, JSON paths, geometry and film dimensions are
preserved. Five texture tests pass, covering PNG alpha (including palette and
color-key transparency), JPEG format/aspect ratio, source preservation and
unchanged copies of small textures and nontexture assets.

- The generated bundle contains 225 textures within the configured size limit.
  All 677 packaged asset hashes match; all nontexture assets match their sources.
- All nine scene replay images match the preparation images. Resolution/sample
  changes and missing/corrupt shader-cache checks also pass. All six graphics
  and compute demos pass during resource preparation.
- Junkshop's 145 texture allocations shrink from 2944 MiB to 580 MiB of estimated
  base RGBA8 storage. The signed Release app installs and renders Junkshop on
  iPhone 16 Plus / Apple A18 at its original 2000 × 1000 resolution, completing
  2 spp with no error (3.55 seconds of render time, including first-render setup).
- A 20-second Instruments Activity Monitor recording completes without app
  termination. Sampled peak physical footprint is 2.03 GiB, settling to 1.72 GiB.
  Before downsampling, the last recorded sample was 3.24 GiB before termination;
  that value is not an exact system memory limit.

The tested bundle is `out/ios/Resources-mobile`. Packaging reports live inside
that bundle; device images, results and memory traces use the
`out/ios/mobile-textures-*` prefix. This verifies startup and 2-spp rendering,
not an extended stress test of every Blender scene.

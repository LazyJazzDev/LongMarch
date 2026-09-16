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

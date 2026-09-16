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

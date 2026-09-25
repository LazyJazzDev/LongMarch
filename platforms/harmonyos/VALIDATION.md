# HarmonyOS port validation

Status as of 2026-09-25: ARM64 C++ and ArkTS compilation and unsigned HAP packaging
succeeded. A development-signed HAP is installed and running on Mate 70 RS.

## Completed on Apple M5 / macOS

- Portable SHA-256 compared against Python hashlib, including empty input,
  padding/block boundaries, binary input, and one million bytes.
- Resource staging verifies copied-file manifests, rejects corrupt SPIR-V and
  unresolved LFS pointers, omits Metal shaders, and refuses output overwrite.
- Headless Vulkan native engine and shared mobile adapters compiled with Ninja.
- Vulkan 2048 and GoL shared-game checks passed: input, animation, focus/idle,
  simulation controls and orientation handling.
- Vulkan Triangle, Texture, Blend, Cube, Resize, SDR Sample, HDR, and NBody checks
  passed. NBody checks motion, pause, consecutive resets, and finite positions;
  image checks verify finite pixels. HDR gradient exceeds SDR reference white.
- Vulkan GPU timestamp queries returned GPU timing for NBody.
- Separate `LONGMARCH_PREPARE=OFF` replay build passed Cornell Box fallback,
  NBody, and both shared-game checks directly from the packaged read-only cache.
  Its executable links Vulkan and system libraries, without DXC or GLFW.
- Vulkan compute fallback rendered Cornell Box and Area Light at a 64-pixel edge.
- All nine scenes rendered one compute-fallback sample on the Metal preparation
  backend at a 64-pixel edge. These generate portable SPIR-V, not a claim of
  successful Vulkan rendering for every scene.
- iOS shared library rebuilt; Metal game checks and a two-sample Cornell Box
  replay passed with the existing shader bundle, including HDR display checks.

## HarmonyOS SDK build

- Installed official macOS ARM DevEco Studio 26.0.0.851; Huawei developer signature
  and Apple notarization verified.
- SDK 26.0.0.105: standalone ARM64 CMake/Ninja build and Hvigor native build pass.
- ArkTS compiler and HAP packaging pass with target SDK 26.0.0.
- Unsigned HAP includes `liblongmarch.so`, `libc++_shared.so`, ArkTS bytecode, and
  768 resources whose SHA-256 digests were checked inside the archive.
- Native dependencies are HarmonyOS N-API/XComponent/window/rawfile, Vulkan,
  libc++ and libc. No host DXC, GLFW, Metal, or Swift library is linked.
- Official emulator image installation returned “Currently, this capability is
  available only in the Chinese mainland.” No emulator image was installed.
- The user configured automatic development signing; the signed HAP installed
  successfully through HDC on Mate 70 RS. Signing material remains local.

## Mate 70 RS device checks

- PLU-AL10, system 7.0.0.107 / API 26, Vulkan Maleoon 920.
- First launch extracted the resource bundle and displayed the demo browser.
- Triangle, Texture, Blend, Cube, Resize, SDR Sample, HDR and NBody rendered
  frames and reported GPU timestamp measurements.
- Ray Query reports unsupported on this driver. Sparkium Cornell Box uses
  compute fallback, rendering at 1024 x 1024 through the 32 SPP limit.
- Linear extended-sRGB HDR swapchain creation succeeds. This establishes the
  surface format, not measured HDR display brightness or color accuracy.
- Device screenshots exposed a rotated landscape presentation and unreadable
  light-theme text. Identity surface transform and application dark mode fixed
  both; Cornell Box now displays upright with a square aspect ratio.
- Moved game content below the navigation bar to expose its native controls.
- Added an ArkUI touch layer over the native-library XComponent. Device swipes
  now move and merge 2048 tiles (score increased to 4); GoL's native width dialog
  opens and increments the board width from 64 to 65.
- 2048 returned from Home to the active game without an initialization error.
- Frame statistics now use presentation intervals, including frame pacing,
  instead of reporting the reciprocal of render work time as display FPS.
- Large Blender scenes, exhaustive scene controls, file pickers and physical
  multi-finger gestures still need broader runtime coverage.

## Observed host limitation

MoltenVK rejected Blender Classroom's compute path because its 1370 storage
buffer bindings exceed the host limit of 31. Native HarmonyOS device limits must
be measured; host success on Metal does not establish large-scene Vulkan support.

## Pending

- Extended XComponent lifecycle stress, repeated navigation and surface
  destruction during a large scene sample.
- Device game touch input, native size dialogs, document import/export, orientation
  lock/icon rotation and bottom system gesture clearance.
- Hardware ray-query coverage, all full-resolution scenes, memory use, GPU timings,
  HDR surface availability and actual display output on supported hardware.

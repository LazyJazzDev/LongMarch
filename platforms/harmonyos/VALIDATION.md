# HarmonyOS port validation

Status as of 2026-09-25: ARM64 C++ and ArkTS compilation and unsigned HAP packaging
succeeded. A signed application has not yet been run on HarmonyOS.

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
- `hdc list targets` returned no connected device. Signing and runtime checks
  require a usable emulator or device and development signing configuration.

## Observed host limitation

MoltenVK rejected Blender Classroom's compute path because its 1370 storage
buffer bindings exceed the host limit of 31. Native HarmonyOS device limits must
be measured; host success on Metal does not establish large-scene Vulkan support.

## Pending

- Emulator/device launch and XComponent surface lifecycle, including resizing,
  background/resume, repeated navigation, and surface destruction during a sample.
- Device game touch input, native size dialogs, document import/export, orientation
  lock/icon rotation and bottom system gesture clearance.
- Hardware ray-query coverage, all full-resolution scenes, memory use, GPU timings,
  HDR surface availability and actual display output on supported hardware.
- HAP signing and installation using the user's development identity.

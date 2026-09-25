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
- Immersive fullscreen hides the status bar and navigation indicator; device
  screenshots confirm the app fills the screen.
- Double-tap recognition runs in parallel with pointer events. GoL's document
  picker saved `Life.cells` into Documents and loaded it back without errors.
- Cornell Box retained 32 SPP after changing exposure to +2.3 EV and after
  Home/background/resume; film reset restarted accumulation (observed 3 SPP).
- NBody pause held the displayed frame counter at 1363 across repeated reads;
  resume advanced it to 1457. Particle reset completed without a visible error.
- Frame statistics now use presentation intervals, including frame pacing,
  instead of reporting the reciprocal of render work time as display FPS.
- Large Blender scenes, exhaustive scene controls, physical
  multi-finger gestures still need broader runtime coverage.

## Observed host limitation

MoltenVK rejected Blender Classroom's compute path because its 1370 storage
buffer bindings exceed the host limit of 31. Native HarmonyOS device limits must
be measured; host success on Metal does not establish large-scene Vulkan support.

## Rendering correctness follow-up

- SDR-encoded 2048/GoL and ordinary graphics demos now use an SDR swapchain,
  matching the iOS presentation formats. Sending these colors to a linear HDR
  surface caused the washed-out palette. HDR remains available for Sparkium,
  the HDR demo and NBody.
- Vulkan upload copies now use the graphics queue family, matching exclusive
  buffer ownership, with an explicit memory dependency before shader reads.
  Noncoherent mapped allocations are invalidated/flushed. This conservative
  upload path waits for the shared queue; desktop multi-frame throughput has
  not been benchmarked (the HarmonyOS host uses one frame in flight).
- Image resource barriers cover compute stages. Attachment load operations
  synchronize reads as well as writes, and depth barriers include both early
  and late fragment tests. The game regression previously triggered attachment
  read-after-write and depth write-after-write validation errors; the corrected
  2048/GoL run passes with no validation errors using
  `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LAYER_VALIDATE_SYNC=1`.
  Triangle, Texture, Blend, Cube, Resize, SDR Sample, HDR and NBody also pass
  their host checks with synchronization validation enabled and no errors.
- A DXC 1.10 minimal reproduction exposed lost `NonUniform` decorations on
  storage-buffer access-chain results. The Vulkan shader loader restores these
  decorations and the storage-buffer nonuniform-indexing capability, and the
  device enables the corresponding feature. This also repairs read-only bundled
  shaders without changing the cache files. The regression compiles direct and
  local-variable buffer accesses, checks idempotence and unchanged uniform
  accesses, and validates the repaired SPIR-V with `spirv-val`.
- Cornell Box compute fallback at 256 x 256 / 32 SPP produces byte-identical SDR
  output on the host Vulkan and Metal backends, including after the repair.
  The Vulkan run also passes synchronization validation. This is a host
  comparison, not a pixel-exact comparison against the phone's display.
- On Mate 70 RS at 1024 x 1024 / 32 SPP / 0 EV, the nonuniform-indexing repair
  removes Cornell Box's clustered black/white artifacts and block-shaped geometry edges; normal sampling noise
  remains. Six screenshots during 2048 autoplay include moving/spawning/merging
  tiles and a score increase from 12 to 76 without visible broken tiles. These
  samples do not establish exhaustive frame-by-frame animation coverage.
- Signed ARM64 HAP builds and installation succeed with these changes. Device
  screenshots confirm the restored 2048/GoL palette.

## Pending

- Extended XComponent lifecycle stress, repeated navigation and surface
  destruction during a large scene sample.
- Broader device touch and multi-finger coverage, orientation lock/icon rotation,
  document cancellation/invalid-file handling, and system-edge gesture clearance.
- Hardware ray-query coverage, all full-resolution scenes, memory use, GPU timings,
  HDR surface availability and actual display output on supported hardware.

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
- The original linear extended-sRGB HDR swapchain creation succeeded on this
  device, but did not establish actual HDR display output. That presentation
  path was subsequently replaced by HDR10; see the MLN-AL00 checks below.
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

## HDR10 on MLN-AL00

- Tested system 6.1.0.135(SP17C00E135R3P4), Vulkan Maleoon 935F, with a refreshed
  development signature for this device. Signed HAP build/install pass.
- The driver advertises `A2B10G10R10_UNORM_PACK32` with sRGB/P3 WSI color spaces,
  but no `RGBA16Float + extended-linear-sRGB` pair. The previous selection thus
  fell back to an 8-bit SDR buffer despite the HDR-capable display.
- The new presentation pass encodes linear Rec.709 into BT.2020/PQ on the GPU.
  Native-window color-space, HDR type and static-metadata setters all return
  success. RenderService buffer dumps show 10-bit format 34, color metadata
  2360324 (BT.2020/PQ), and metadata type 2 (HDR10). Off/on tests restore 8-bit
  format 12, sRGB metadata 2294273 and type 0, then re-enable HDR10 correctly.
- RenderService reported approximately 62.83 for the SDR layer's `displayNit`
  and 309.52 for HDR after adaptation. These are system diagnostics, not measured
  panel luminance. The user confirmed that the upper HDR gradient highlights
  are visibly brighter than the lower reference-white bar on the actual screen.
- Sparkium Cornell Box and NBody render using the same 10-bit/PQ/HDR10 buffer
  metadata. Switching to 2048 restores the SDR buffer and its original palette.
  Cornell Box's original scene settings cap `max_exposure` at 1.0; at default
  display exposure it is not a strong super-white highlight test. These scene
  settings remain aligned with iOS; the app's exposure control still applies.
- The host `harmony_hdr_check` passes with Vulkan synchronization validation:
  decoding the GPU output reproduces 203-nit reference white, 609-nit highlights,
  the 1000-nit signal ceiling and transformed red primaries. SDR clipping and
  NBody's gamma decoding also pass. The display ceiling is a content encoding
  choice, not a claim about the panel's maximum brightness.
- This replaces the original linear floating-point surface path. The public
  [OpenHarmony 6.0 HDR classification implementation](https://github.com/openharmony/graphic_graphic_2d/blob/OpenHarmony-6.0-Release/rosen/modules/render_service/core/feature/hdr/rs_hdr_util.cpp)
  checks supported 10-bit buffer formats and PQ/HLG transfer functions; selecting
  a floating-point format alone does not establish HDR output.

## Pending

- Extended XComponent lifecycle stress, repeated navigation and surface
  destruction during a large scene sample.
- Broader device touch and multi-finger coverage, orientation lock/icon rotation,
  document cancellation/invalid-file handling, and system-edge gesture clearance.
- Hardware ray-query coverage, all full-resolution scenes, memory use, GPU timings,
  HDR colorimetry/peak-brightness measurements and additional supported devices.

## Complex compute scenes on MLN-AL00

- The Maleoon 935F driver reports Vulkan 1.3, but does not advertise
  `VK_KHR_ray_tracing_pipeline`, `VK_KHR_ray_query`, or
  `VK_KHR_acceleration_structure`. Its ray-tracing pipeline feature is false.
  This is an observation about the installed 6.1.0.135 driver/API exposure,
  not a statement that the silicon lacks ray-tracing hardware. The app logs
  these capabilities under `LongMarchGPU`.
- The full-resolution Texture scene previously returned a black image while
  the driver repeatedly reported GPU resets/device loss. Its monolithic compute
  dispatch is now split into 128 x 128 pixel jobs using Vulkan dispatch bases.
  Native resolution, global pixel coordinates, materials and bounce counts are
  preserved. GPU submission/wait failures now propagate instead of silently
  advancing sample counters.
- The phone renders Texture at 2048 x 1024 after tiling, with nonzero finite
  HDR pixels. The host scene check now rejects an entirely black result.
  Vulkan synchronization validation passes for Texture at 259 x 130 / 2 SPP,
  including partial edge tiles. The tiling-only SDR output was byte-identical
  to the untiled Metal result; with the final function-boundary changes,
  6 of 101010 channels differ (mean absolute error 0.00172/255). Cornell Box at 256 x 256 / 32 SPP is also byte-identical
  to the existing Metal reference. HDR conversion, shared game animation/input,
  and the standalone Texture raster demo pass host regression checks.
- Complex shader graphs now retain function boundaries in the software path
  tracer, material evaluators and high-level Principled BSDF functions. The
  previous fully inlined Blender kernel could spend several minutes inside
  `vkCreateComputePipelines`. Buffer indices are passed between graph functions
  instead of opaque buffer pointers, avoiding a VariablePointers requirement.
  Ray Query retains its original inlining behavior.
- All nine updated fallback scenes pass a 128-pixel-edge, 1-SPP Metal check;
  all nine Ray Query scenes pass at 64-pixel edge / 1 SPP. Classroom and Monster
  fallback images are byte-identical to their previous host references;
  Junkshop differs in 11 of 24576 channels (mean absolute error 0.0139/255).
  All 162 packaged SPIR-V modules pass `spirv-val` for Vulkan 1.2 with scalar
  block layout. Resource, digest and NonUniform regression tests pass (5 tests).
- The updated phone build displays Blender Monster at native 1024 x 1024 / 1 SPP.
  The final build also displayed Classroom at native 1920 x 1080 / 1 SPP;
  its first frame took 117.277 seconds including pipeline preparation. Native
  telemetry confirmed successful presentation and the UI reached 1 / 1 SPP.
  These are low-sample black-screen checks, not noise convergence or
  interactive-performance acceptance. Junkshop did not reach its first frame
  during an approximately five-minute attempt; the worker remained CPU-bound
  and no first-frame completion was logged. That scene remains unresolved on
  the phone, despite passing the host check. The attempt was stopped to verify
  Classroom; it is not counted as a device pass.
- The signed app also successfully queried the public XEngine extension API.
  It returned four non-RT extensions and no RTGI, reflection or shadow/AO
  capability. See [RAY_TRACING.md](RAY_TRACING.md) for the exact results, SDK
  contracts and implementation routes. No hardware RT execution is claimed.

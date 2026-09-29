# LongMarch on HarmonyOS NEXT

Native ArkTS/ArkUI application with an N-API C++ renderer and Vulkan XComponent
presentation. The mobile sessions in `platforms/ios` share the original desktop
2048/GoL UI, Slang, NBody simulation, and Sparkium scene loader. Swift and Metal
are not linked into the HarmonyOS application.

The ARM64 native library, ArkTS application, and unsigned HAP build successfully
with DevEco Studio 26.0.0.851. A signed build runs on Mate 70 RS; broader
device validation remains in progress. See `VALIDATION.md` for the
actual checks completed and remaining device limitations.

## Functionality

- Immersive fullscreen throughout the app, with system status/navigation bars hidden.

- Demo browser with 2048, Game of Life, Sparkium, Graphics Hello, and NBody CS.
- Original game geometry, controls, animations, palette, AI, and life simulation.
  Swipe to move in 2048; five score taps start AI and one stops it. GoL supports
  cell editing, two-finger pan/pinch, native grid-size controls, and document pickers.
- Sparkium retains all nine scene presets and full scene resolution. It starts
  automatically, preserves accumulation when the SPP limit changes, supports
  film reset, scene reload, exposure, pause, and a persistent image pan/zoom.
- NBody controls particle/galaxy counts, time step, pause, reset, render scale,
  and camera rotation. Vulkan timestamp queries report GPU time where supported.
- Backgrounding stops scheduling frames and pauses game AI. Resuming resets game
  clocks. Returning to the browser releases the active renderer. Idle games sleep
  until input or the next animation/simulation deadline.
- GoL locks its entry orientation; gravity updates rotate the original icons.
  Sparkium requests landscape; leaving a demo restores automatic rotation.

Full RT pipeline Graphics Hello entries retain the iOS browser's explanatory
unavailable state. Ray Query requires device support. Sparkium uses shared
compute ray tracing on devices without ray queries. Scene resource requirements
still apply: a device may reject a large Blender scene if descriptor or memory
limits are exceeded. The error is shown in the app; another scene can be selected.

HDR uses a 10-bit UNORM surface and a GPU presentation pass that converts linear
Rec.709 to BT.2020/ST.2084 PQ. Native-window color space and HDR10 static metadata
identify the content to HarmonyOS RenderService, including drivers that advertise
10-bit WSI formats only with sRGB/P3 color spaces. The signal uses 203-nit reference
white and a 1000-nit content ceiling; actual display brightness remains controlled
by the system. Devices without the required 10-bit format report SDR output.
NBody's display-encoded particles are decoded before HDR conversion; games retain
their original SDR palette. Turning HDR off resets buffer metadata to SDR.

## Prerequisites

- DevEco Studio with HarmonyOS SDK and native C++ toolchain; ARM64 phone/tablet.
- Vulkan 1.2 with the engine's dynamic rendering, extended dynamic state, and
  descriptor-indexing features. Unsupported devices report initialization errors.
- Python 3, CMake 3.25+, Ninja, portable dependency headers, and host
  `glslangValidator` (Vulkan SDK/glslang). CMake uses it to compile the presentation
  shader into a build-generated header; no shader compiler is shipped in the HAP.
- The assets submodule initialized with its real LFS objects.

The first build fetches pinned MikkTSpace and FreeType sources. CMake
`FETCHCONTENT_SOURCE_DIR_MIKKTSPACE` and `FETCHCONTENT_SOURCE_DIR_FREETYPE` can
point to existing sources for offline builds. All target libraries are compiled
with the HarmonyOS SDK; host libraries are never linked into the HAP.

## Resources and dependency headers

Install the shared host manifest as described in `platforms/ios/README.md`:

```sh
/path/to/vcpkg/vcpkg install --x-manifest-root=platforms/mobile \
  --x-install-root=out/mobile/deps --triplet=arm64-osx
```

The project automatically finds `out/mobile/deps`. This manifest supplies portable
headers and target-independent FreeType/MikkTSpace sources, not host Vulkan headers.
Vulkan comes from the HarmonyOS SDK. GLFW supplies constants only; its host library
is not linked. CMake does not download dependencies. `LONGMARCH_MOBILE_DEPS` can
select another installation; `LONGMARCH_HEADERS` remains an optional header override.

Prepare the mobile assets following `platforms/ios/README.md`. Its Slang cache
contains portable Vulkan 1.2 SPIR-V before Metal conversion. Build the additional
fallback preparation tool in the same macOS preparation build:

```sh
cmake --build build-ios-prepare --target harmony_fallback_prepare
python3 platforms/harmonyos/prepare_resources.py \
  --source out/ios/Resources \
  --fallback-renderer build-ios-prepare/harmony_fallback_prepare \
  --output platforms/harmonyos/entry/src/demos/resources/rawfile/Resources
```

Use the current, freshly prepared iOS bundle. Staging refuses to overwrite an
existing output, verifies shader digests and SPIR-V magic, rejects LFS pointers,
and generates a file manifest with SHA-256 digests. The fallback preparation
renders one 64-pixel-edge sample of each scene on the host to generate its
compute shaders; it does not change scene JSON or mobile render resolution.
Generated MSL is omitted from the final HAP.

The first launch extracts resources to a versioned directory in the app's own
files directory, verifies every file's digest, and marks extraction complete only
after success. It never writes to the signed rawfile bundle. Interrupted extraction
is retried. Old resource versions are retained until app data is cleared.

## Build and run

Open `platforms/harmonyos` in DevEco Studio, let it synchronize the project, and
select the installed HarmonyOS SDK. Local SDK paths and signing configuration
belong in local settings and must not be committed. Use the `entry` module and
`default` product. Build the HAP and select a device or emulator to run it.

For the standard macOS installation, after staging headers and resources:

```sh
sh platforms/harmonyos/build_hap.sh
```

Set `DEVECO_STUDIO_DIR` to an alternate app's `Contents` directory if needed.
The unsigned output is
`entry/build/default/outputs/default/entry-default-unsigned.hap` (about 366 MiB
with the current nine-scene bundle). The project targets SDK 26.0.0, with API 12
as the declared compatibility floor; older systems still need runtime validation.

The native CMake target is `longmarch`, built with Ninja by Hvigor. The
`LONGMARCH_HEADERS` CMake cache variable or environment variable can override the
staged headers. `LONGMARCH_PREPARE` must remain off for device builds: runtime Slang
and SPIRV-Cross are not shipped.

Device installation requires the user's HarmonyOS development signing setup.
The bundle identifier is `dev.lazyjazz.longmarch`. No network, storage-wide, or
sensor permissions are requested; import/export uses user-selected document URIs.

### Game of Life app

The `gameoflife` product builds **Game of Life** (生命游戏,
`dev.lazyjazz.gameoflife`), a standalone app with only the GoL page: no demo list,
status header or back navigation. It has its own name and layered icon in
`AppScope/gol/resources` (generated by `demo/gol/tools/app_icon.py`), and its entry
target bundles `entry/src/gol/resources` instead of the scene bundle in
`entry/src/demos/resources`. Stage the game's resources from the iOS extraction
(`platforms/ios/prepare_game_resources.py`), which has no scenes:

```sh
python3 platforms/harmonyos/prepare_resources.py --source out/ios/GameOfLifeResources \
  --output platforms/harmonyos/entry/src/gol/resources/rawfile/Resources
PRODUCT=gameoflife sh platforms/harmonyos/build_hap.sh
```

The HAP is about 5 MiB. Signing is per bundle name: add a signing configuration for
`dev.lazyjazz.gameoflife` locally and reference it from the product. For AppGallery,
use release signing material from AppGallery Connect and build the `.app` package:

```sh
PRODUCT=gameoflife BUILD_MODE=release TASK=assembleApp sh platforms/harmonyos/build_hap.sh
```

## Host checks

A headless Vulkan build verifies the same mobile adapters without ArkUI:

```sh
cmake -S platforms/harmonyos -B out/harmonyos/host-prepare -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLONGMARCH_PREPARE=ON
cmake --build out/harmonyos/host-prepare
out/harmonyos/host-prepare/mobile_games_check /path/to/Resources
out/harmonyos/host-prepare/mobile_demo_check /path/to/Resources nbody_cs out/harmonyos/replay
out/harmonyos/host-prepare/harmony_hdr_check
python3 -m unittest discover -s platforms/harmonyos/tests -v
```

Host Vulkan needs a Vulkan loader/SDK, and preparation needs Slang. To verify the
read-only shader path, configure a separate build with `LONGMARCH_PREPARE=OFF` and
run it against the staged bundle. Ray Query checks require a ray-query capable
host backend; MoltenVK on macOS does not substitute for that device coverage.

## Device scene checks and ray-tracing capabilities

A signed development build can select a scene and sample limit at launch:

```sh
hdc shell aa force-stop dev.lazyjazz.longmarch
hdc shell aa start -a EntryAbility -b dev.lazyjazz.longmarch \
  --ps demo sparkium --ps scene blender_monster --pi samples 2
hdc shell hilog -x -T LongMarchGPU
```

The scene must belong to the bundled catalog; the sample limit is clamped to
1–4096. Omit these arguments for normal interactive startup. First/final sample
logs complement screen inspection when a complex scene takes time to prepare.

The native build optionally detects the installed HMS XEngine headers. Its
public library is loaded only when present on the device. Standard Vulkan and
XEngine capabilities are logged separately. See [the ray-tracing investigation](RAY_TRACING.md)
for verified device results, SDK contracts and the proposed hardware RT routes.
